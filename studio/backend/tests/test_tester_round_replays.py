# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Replays of the external tester's own rounds, on a CPU runner, in a few seconds.

The tester re-runs Studio's diffusion matrix after each batch of merges on an RTX 6000 Ada (sm89, 300 W cap, PCIe 4.0
x16), an RTX 3090 (sm86, 350 W, PCIe 3.0 x4) and the Ada with simulated 8 / 12 / 16 / 24 GB budgets.
``fixtures/tester_round_decisions.json`` is distilled from that tester's server logs and result files of rounds r6 to
r10 (``scripts/tester_smoke/distill_tester_logs.py``): per cell, the card facts and the load-time decisions the server
logged (INT8 attention, exact start, CUTLASS Linear count, H3 placement, H3-Image plan, deferred-race timing), next to
the measured s/step, SM clock and power.

Three groups of tests:

1. Log replays: the current resolvers are driven with each logged card's facts and must reach the decision of the
   rounds that were fast (r6, r8, r10 for H3 video; r9 and r10 for the deferred race; every round for the H3-Image
   plan), not the one of the round that regressed.
2. Clock droop: a deterministic power-capped card (``DroopCard``) whose clock lags the power draw, so a kernel that
   draws more wins a burst and loses when held. The card replaces the CUDA event timer and ``time.perf_counter``, not
   the race code, so every race consumer times with its own loops: the int8 GEMM CUTLASS race (H3 video and
   H3-Image shapes), the INT8 attention long race and the cross-request batching race must keep the incumbent.
3. A structural lint: every module-level table of CUDA capabilities under ``core/inference`` is listed in
   ``CAPABILITY_TABLES`` with the capabilities it holds and the measurement behind it, so a new per-arch exclusion
   (#19098's ``H3_INT8_AUTO_OFF_ARCHS = frozenset({(8, 9)})``) or a card dropped from a table fails until reviewed.

No product code is imported at module level, so the file collects on any Studio tree (the root-cause trees were checked
out and run against it).
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Optional

import pytest

_TESTS = Path(__file__).resolve().parent
_BACKEND = _TESTS.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

FIXTURE = _TESTS / "fixtures" / "tester_round_decisions.json"
LOG = json.loads(FIXTURE.read_text(encoding = "utf-8"))
CARDS = LOG["cards"]
# rounds the tester measured fast per class of decision (HANDOFF series: r7 lost H3 video INT8 attention, r9 lost
# the Ada H3 video clock to 313 CUTLASS Linears, r8 never ran the deferred image race)
H3_GOOD = ("r6", "r8", "r10")
DEFER_GOOD = ("r9", "r10")


def cells(
    prefix: str,
    rounds: Optional[tuple] = None,
    tags: Optional[tuple] = None,
) -> list:
    out = [c for c in LOG["cells"] if c["cell"].startswith(prefix)]
    if rounds is not None:
        out = [c for c in out if c["r"] in rounds]
    if tags is not None:
        out = [c for c in out if c["tag"] in tags]
    return out


def cell_id(c: dict) -> str:
    return f"{c['r']}-{c['cell']}-{c['tag']}"


def card_of(c: dict) -> dict:
    return CARDS[c["cell"].split(".")[1]]


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch, tmp_path):
    """Studio defaults only: no inherited UNSLOTH_* lever, an empty Studio home (no stored race verdicts)."""
    for k in list(os.environ):
        if k.startswith("UNSLOTH_"):
            monkeypatch.delenv(k, raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


@contextlib.contextmanager
def on_tester_card(
    monkeypatch,
    card: dict,
    free_mib: Optional[int] = None,
):
    """torch reports the logged card (capability, name, total, free VRAM); anything that really touches a device
    raises, as on a CPU build. Self-contained (older trees have no ``gpu_profiles``)."""
    import torch

    try:
        import torch._dynamo  # noqa: F401 -- imported before torch functions are replaced
    except Exception:  # noqa: BLE001
        pass
    cap = tuple(card["cap"])
    total = int(card["total_mib"]) * 2**20
    free = int(free_mib if free_mib is not None else card["total_mib"] - 800) * 2**20
    props = types.SimpleNamespace(
        name = card["name"],
        major = cap[0],
        minor = cap[1],
        total_memory = total,
        multi_processor_count = 142 if cap == (8, 9) else 82,
        is_integrated = False,
        gcnArchName = "",
        uuid = "virtual",
        pci_bus_id = 1,
        pci_device_id = 0,
        L2_cache_size = 96 * 2**20,
        max_threads_per_multi_processor = 1536,
        shared_memory_per_block_optin = 101376,
        shared_memory_per_multiprocessor = 102400,
        regs_per_multiprocessor = 65536,
        warp_size = 32,
    )

    def no_device(*a, **k):
        raise RuntimeError("on_tester_card: no real device")

    m = monkeypatch
    m.setattr(torch.cuda, "is_available", lambda: True)
    m.setattr(torch.cuda, "device_count", lambda: 1)
    m.setattr(torch.cuda, "current_device", lambda: 0)
    m.setattr(torch.cuda, "is_initialized", lambda: True)
    m.setattr(torch.cuda, "_lazy_init", no_device)
    m.setattr(torch.cuda, "get_device_capability", lambda device = None: cap)
    m.setattr(torch.cuda, "get_device_name", lambda device = None: card["name"])
    m.setattr(torch.cuda, "get_device_properties", lambda device = None: props)
    m.setattr(torch.cuda, "mem_get_info", lambda device = None: (free, total))
    m.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation = True: True)
    if hasattr(torch.cuda, "is_current_stream_capturing"):
        m.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    m.setattr(torch.version, "hip", None, raising = False)
    m.setattr(torch.version, "cuda", "13.0", raising = False)
    yield card


# ------------------------------------------------------------------------------------------------ 1. log replays


def _h3_video_default_gate():
    """H3 video's load-time INT8 attention gate for Studio's default request (``VideoLoadRequest`` defaults, the
    hosted INT8 ConvRot denoiser Auto loads on CUDA), on the card ``on_tester_card`` reports."""
    import inspect

    from core.inference import diffusion_int8_attention as i8
    from core.inference import video_minimax_h3_attn as h3
    from core.inference.diffusion_attention import _auto_request
    from models.inference import VideoLoadRequest

    fields = VideoLoadRequest.model_fields
    backend = fields["attention_backend"].default
    policy = i8.resolve_policy(fields["int8_attention"].default)
    kwargs = dict(
        mode = policy.mode,
        attention_backend = backend,
        auto_attention = _auto_request(backend),
        speed_on = True,
        hosted_int8 = True,
        comfy_checkpoint = False,
        adapters = bool(fields["loras"].default),
        capability = h3.cuda_capability("cuda:0"),
    )
    accepted = inspect.signature(h3.h3_int8_load_gate).parameters
    return h3.h3_int8_load_gate(**{k: v for k, v in kwargs.items() if k in accepted}), kwargs[
        "capability"
    ]


@pytest.mark.parametrize("c", cells("h3v.", H3_GOOD), ids = cell_id)
def test_h3_video_auto_int8_attention_engages_on_every_logged_card(c, monkeypatch):
    """r6 / r8 / r10 logged ``INT8 attention engaged (policy auto from default, raced)`` on the Ada (every budget) and
    the 3090; r7 (#19098) logged ``not engaged (... not eligible)`` on the Ada and lost 25%. Today's gate for the
    default request on that card must race it, with an exact start no longer than the good rounds ran."""
    logged = c["dec"]["int8_attn"]
    assert logged == {
        "engaged": True,
        "how": "raced",
        "policy": "auto",
        "source": "default",
    }, logged
    with on_tester_card(monkeypatch, card_of(c), c["free_mib"]):
        gate, cap = _h3_video_default_gate()
    assert cap == tuple(card_of(c)["cap"])
    assert gate.auto and not getattr(gate, "forced", False) and gate.reason is None, (
        cell_id(c),
        gate,
    )
    longest = max(x["dec"].get("exact_start", 0) for x in cells("h3v.", H3_GOOD))
    assert getattr(gate, "exact_steps", 0) <= longest == 4, gate


def test_r7_round_is_the_one_that_logged_int8_attention_off():
    """Fixture sanity: the regressed round is the only one whose Ada cells logged INT8 attention off."""
    off = {(x["r"], x["cell"]) for x in cells("h3v.") if not x["dec"]["int8_attn"]["engaged"]}
    assert off and {r for r, _ in off} == {"r7"} and all(".gpu0." in cid for _, cid in off), off


def _h3_video_placement(free_bytes: int) -> tuple:
    """(tier, pinned-set GB) of H3 video's default int8 load at ``free_bytes`` free, sized as the load sizes it."""
    import torch

    from core.inference import video as v
    from core.inference import video_families as vf
    from core.inference.video_minimax_h3 import H3_TRANSFORMER_PREQUANT_GB

    fam = next(f for f in vf._FAMILIES if getattr(f, "name", None) == "minimax-h3")
    den = int(H3_TRANSFORMER_PREQUANT_GB["int8"] * 1e9)
    comps = fam.bf16_components_gb
    whole, pinned = (
        (den, v._h3_others_bytes(fam, comps, te_scheme = "int8", dtype = torch.bfloat16, rotating = r))
        for r in (False, True)
    )
    tier = v._h3_placement_tier(
        quantised = True,
        whole_set_sizes = whole,
        pinned_sizes = pinned,
        free_bytes = free_bytes,
        speed_off = False,
    )
    return tier, sum(pinned) / 1e9


@pytest.mark.parametrize("c", cells("h3v."), ids = cell_id)
def test_h3_video_placement_matches_the_logged_line(c):
    """Every round logged ``video.h3_placement: <free> GB free is under the 56.4 GB a pinned int8 denoiser plus the
    larger rotating component need, so it streams block by block``: the same free memory resolves to the same tier
    and the same 56.4 GB today (the streamed tier is the one whose block pinning #15118 / #15796 made fast)."""
    logged = c["dec"]["h3_place"]
    tier, need_gb = _h3_video_placement(int(logged["free_gb"] * 1e9))
    assert tier == logged["tier"] == "stream", (cell_id(c), tier)
    assert abs(need_gb - logged["need_gb"]) < 0.05, (need_gb, logged)


# H3-Image plan inputs the load logs: DiT / encoder GB at int8 (``dit=int8 ... te=int8``), two streamed members in flight
_H3I = dict(dit_gb = 21.9, te_gb = 27.2, inflight = 2)
# host budget: the tester runs Studio inside a MemoryMax=70G scope
_HOST_GB = 70 * 2**30 / 1e9


@pytest.mark.parametrize("c", [c for c in cells("h3i.") if c["dec"].get("h3i_budget")], ids = cell_id)
def test_h3_image_plan_keeps_the_logged_layout_and_residency(c):
    """Every round's H3-Image cells on the 8-24 GB Ada budgets and the 3090 (PCIe 3.0 x4) logged a streamed plan
    with its budget, link speed and resident block count; R5 OOMed at 8 GB and streamed 19 of 50 blocks on the 3090
    (#18183 / #18284). Replayed with the logged budget and link, today's plan must fit (no refusal), stream, and keep
    at least the logged resident blocks and the logged parked layout."""
    from core.inference import diffusion_minimax_h3_image as h3i

    d = c["dec"]
    link = d.get("h3i_link")
    plan = h3i.plan_placement(
        vram_gb = d["h3i_budget"],
        host_gb = _HOST_GB,
        h2d_gib_s = link[0] if link else None,
        block_tops = link[1] if link else None,
        **_H3I,
    )
    assert plan.placement == d["h3i_plan"]["placement"] == "stream", (cell_id(c), plan.reasons)
    assert plan.dit_resident_blocks >= d["h3i_blocks"][0], (
        cell_id(c),
        plan.dit_resident_blocks,
        plan.reasons,
    )
    # the logged "<layout> layout priced faster"; the 8 GB cell logs no pricing, but R5's OOM was today's layout there
    want = d.get("h3i_layout")
    assert (
        (plan.layout == want)
        if want
        else (plan.layout in (h3i.LAYOUT_PARKED, h3i.LAYOUT_PARKED_VAE))
    ), (
        cell_id(c),
        plan.layout,
        want,
        plan.reasons,
    )


def _deferred_race_render(renders: int = 4) -> tuple:
    """Back-to-back image renders after a fresh load whose CUTLASS race was deferred (R8 logged ``CUTLASS race of N
    weight shape(s) ... runs after the first image``): the 1-based render whose GEMM calls first see the race's
    winner (0 = none of ``renders``), and where the race ran."""
    import torch

    from core.inference import diffusion_int8_cutlass as cut
    from core.inference import diffusion_int8_cutlass_defer as defer
    from core.inference import diffusion_int8_gemm as g8

    n, k = 3072, 3072
    w = types.SimpleNamespace(
        qdata = torch.empty(n, k, dtype = torch.int8), scale = torch.empty(n, 1, dtype = torch.bfloat16)
    )
    recs = [(types.SimpleNamespace(bias = None), ("v2", None, False, w, True))]
    key = g8._cut_key(n, k, torch.float32, torch.bfloat16, False)
    defer._reset_for_tests()
    saved = {
        name: dict(getattr(g8, name))
        for name in ("_CUT_PLAN", "_CUT_DONE", "_CUT_DEFER_KEYS", "_CUT_TRITON_ON")
    }
    pins = [
        (
            g8,
            "_cutlass_plan",
            lambda index, r, triton_on, logger = None: g8._CUT_PLAN.setdefault(index, {}).update(
                {key: {512: 0, 4096: 0}}
            )
            or g8._CUT_PLAN[index],
        ),
        (g8, "_cut_misses", lambda index, r: ([], [(m, rec, key) for m, rec in r])),
        (g8, "_cut_cache_key", lambda index: None),
        (cut, "probed_tiles", lambda index: (0,)),
        (
            cut,
            "library",
            lambda: object(),
        ),  # the prebuilt wheel is installed, as on the tester's cards
        (torch.cuda, "device", lambda *a, **k: contextlib.nullcontext()),
        (torch.cuda, "synchronize", lambda *a, **k: None),
        (torch.cuda, "empty_cache", lambda: None),
    ]
    old = [(o, a, getattr(o, a)) for o, a, _ in pins if hasattr(o, a)]
    try:
        for o, a, f in pins:
            setattr(o, a, f)
        g8._CUT_TRITON_ON[0] = True
        with defer.allow():
            g8._plan_or_defer(0, recs, True, False)
        where = "deferred" if g8._CUT_DEFER_KEYS.get(0) else "at_load"
        served = 0
        for i in range(1, renders + 1):
            defer.render_begin()
            entry = (g8._CUT_PLAN.get(0) or {}).get(key)
            if not served and entry and any(t is not None for t in entry.values()):
                served = i
            defer.render_end()
        return served, where
    finally:
        defer._reset_for_tests()
        for o, a, f in old:
            setattr(o, a, f)
        for name, value in saved.items():
            getattr(g8, name).clear()
            getattr(g8, name).update(value)


@pytest.mark.parametrize(
    "c", [c for c in cells("k1.") + cells("h3i.") if c["dec"].get("defer_queued")], ids = cell_id
)
def test_deferred_cutlass_race_lands_before_render_3_back_to_back(c, monkeypatch):
    """The tester's client sends the next render 0.00-0.06 s after the last returns. r8 (#19308) logged the race
    queued and never run in every image cell (the timed renders all ran Triton); r9 / r10 logged
    ``deferred CUTLASS race done ... `` after render 1 (#19547). Replayed on the logged card with no idle spell
    ever (a 600 s idle threshold), the winner must serve from render 2."""
    d = c["dec"]
    if c["r"] in DEFER_GOOD:
        assert d["defer_done"]["renders_before"] == 1, d
    elif c["r"] == "r8":
        assert "defer_done" not in d, d
    from core.inference import diffusion_int8_cutlass_defer as defer

    monkeypatch.setenv(defer.IDLE_ENV, "600")
    with on_tester_card(monkeypatch, card_of(c), c["free_mib"]):
        served, where = _deferred_race_render()
    assert where == "deferred" and served == 2, (cell_id(c), where, served)


# ------------------------------------------------------------------------------------------------ 2. clock droop


@dataclasses.dataclass(frozen = True)
class Droop:
    """A power-capped card. ``boost``: SM MHz after idle. ``alone``: MHz a kernel class settles at when it runs by
    itself at the cap; ``mix``: MHz when the incumbent and a challenger share the card (R9: each alone near the cap,
    together lower). ``cycles``: work per call relative to the incumbent at equal clock (< 1 = faster). ``tau``: clock
    response time constant; a burst much shorter than ``tau`` runs at the clock the card had before it."""

    boost: float
    alone: dict
    mix: float
    cycles: dict
    tau: float
    source: str


# ``tau`` = 0.1 s: the slowest clock response the CUTLASS race's sustained window (0.25 s settle + 0.25 s timed per
# candidate) resolves on these profiles (0.2 s already lets a 4128-row shape through, see the xfail below). The real
# response is unmeasured at kernel granularity; the L4 settles "in about 1 s" at the 12.6 s H3 step level.
TAU_S = 0.1
DROOP = {
    # R9 class: CUTLASS wins alone, sustained, and loses beside today's kernel (tester R9: 1240 vs 1035 MHz at 291 W)
    "ada_paired": Droop(
        2505.0,
        {"incumbent": 1240.0, "challenger": 1180.0},
        1035.0,
        {"incumbent": 1.0, "challenger": 0.90},
        TAU_S,
        "tester R9 Ada H3 video: 5.40 s/step @ ~1240 MHz (CUTLASS=0) vs 6.13 s/step @ ~1035 MHz (313 CUTLASS Linears)",
    ),
    # R5 class: wins a short burst ~23%, loses held (#18105: L4 72 W, 855 vs 1290 MHz)
    "l4_burst": Droop(
        2040.0,
        {"incumbent": 1290.0, "challenger": 855.0},
        1171.0,
        {"incumbent": 1.0, "challenger": 0.77},
        TAU_S,
        "#18105 / #18149 docstrings: L4 H3 shapes at 16384 rows, CUTLASS 23% faster short, 25% slower sustained",
    ),
}


class DroopCard:
    """Virtual device time. ``run(cls, units)`` executes a kernel of ``units`` incumbent Mcycles of class ``cls``;
    the clock relaxes with ``tau`` toward the target the recent mix of classes sets (an EWMA of the challenger's busy
    share over ``tau / 3``: mostly incumbent, mixed, mostly challenger)."""

    def __init__(self, d: Droop):
        self.d = d
        self.t = 0.0
        self.f = d.boost
        self.share = 0.0
        self.calls = {"incumbent": 0, "challenger": 0}

    def _target(self) -> float:
        if 0.2 < self.share < 0.8:
            return self.d.mix
        return self.d.alone["challenger" if self.share >= 0.8 else "incumbent"]

    def run(self, cls: str, units: float) -> None:
        self.calls[cls] += 1
        left = units * self.d.cycles[cls]
        hit = 1.0 if cls == "challenger" else 0.0
        while left > 1e-12:
            dt = min(self.d.tau / 20.0, left / self.f)
            self.share += (hit - self.share) * (1.0 - math.exp(-dt / (self.d.tau / 3.0)))
            self.f += (self._target() - self.f) * (1.0 - math.exp(-dt / self.d.tau))
            left -= self.f * dt
            self.t += dt

    def idle(self, seconds: float) -> None:
        self.share *= math.exp(-seconds / (self.d.tau / 3.0))
        self.f += (self.d.boost - self.f) * (1.0 - math.exp(-seconds / self.d.tau))
        self.t += seconds

    def now(self) -> float:
        return self.t


def held_s_per_call(
    d: Droop,
    plan: list,
    units: float,
    seconds: float = 2.0,
) -> float:
    """Seconds per round of ``plan`` once the clock has settled (a ``seconds`` warm-up, then ``seconds`` timed)."""
    dc = DroopCard(d)
    while dc.t < seconds:
        for cls in plan:
            dc.run(cls, units)
    start, rounds = dc.t, 0
    while dc.t - start < seconds:
        for cls in plan:
            dc.run(cls, units)
        rounds += 1
    return (dc.t - start) / rounds


@contextlib.contextmanager
def droop_timers(monkeypatch, dc: DroopCard):
    """CUDA events and ``time.perf_counter`` read ``dc``'s virtual time; synchronize is a no-op."""
    import time as _time

    import torch

    class Event:
        def __init__(self, *a, **k):
            self.t = None

        def record(self, *a, **k):
            self.t = dc.t

        def synchronize(self):
            pass

        def query(self):
            return True

        def elapsed_time(self, other):
            return (other.t - self.t) * 1000.0

    with monkeypatch.context() as m:
        m.setattr(torch.cuda, "Event", Event)
        m.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
        m.setattr(torch.cuda, "empty_cache", lambda: None)
        m.setattr(torch.cuda, "device", lambda *a, **k: contextlib.nullcontext())
        m.setattr(_time, "perf_counter", dc.now)
        yield dc


@pytest.mark.parametrize("key", sorted(DROOP))
def test_droop_card_wins_a_burst_and_loses_held(key):
    """Model sanity: from idle a 10 ms burst of the challenger beats the incumbent's; settled it loses (alone on the
    L4 profile; beside the incumbent on the Ada profile, where it wins alone)."""
    d = DROOP[key]
    burst = {}
    for cls in ("incumbent", "challenger"):
        dc = DroopCard(d)
        while dc.t < 0.01:
            dc.run(cls, 0.5)
        burst[cls] = dc.t / dc.calls[cls]
    assert burst["challenger"] < burst["incumbent"] * 0.95, burst
    today = held_s_per_call(d, ["incumbent"], 0.5)
    alone = held_s_per_call(d, ["challenger"], 0.5)
    beside = held_s_per_call(d, ["challenger", "incumbent"], 0.5) - today
    if key == "l4_burst":
        assert alone > today * 1.10, (alone, today)
    else:
        assert alone < today < beside / 1.05, (alone, today, beside)


# H3 weight shapes (N, K, Linears): QKV, out, FFN in / out. H3 video at the tester's 960x544x124 runs 19318 rows per
# call; H3-Image at 2048^2 ~4100 (its large-row races pair-check, ``PAIRED_ROW_FAMILIES``, #19879).
H3_SHAPES = ((7168, 5376, 150), (5376, 7168, 50), (28672, 5376, 50), (5376, 14336, 50))
VIDEO_ROWS, IMAGE_ROWS = 19318, 4128
H3_IMAGE_DIT = "MiniMaxH3ImageTransformer2DModel"


@contextlib.contextmanager
def cutlass_on_droop(monkeypatch, card: dict, droop: Optional[Droop]):
    """The REAL ``diffusion_int8_gemm._cutlass_plan`` / ``diffusion_int8_cutlass.race`` on the logged card, timing on
    ``DroopCard``: the CUTLASS launch bills the challenger class, today's Triton GEMM / cuBLAS the incumbent; work per
    call is M x N x K (~4 ms per H3 GEMM at 1240 MHz). ``droop=None``: a clock-stable card (the 3090 held 1550-1590 MHz
    on H3 video in every round)."""
    import torch

    from core.inference import diffusion_int8_cutlass as cut
    from core.inference import diffusion_int8_gemm as g8

    flat = Droop(
        1590.0,
        {"incumbent": 1590.0, "challenger": 1590.0},
        1590.0,
        {"incumbent": 1.0, "challenger": 0.90},
        TAU_S,
        "flat",
    )
    dc = DroopCard(droop or flat)

    def units(a, w):
        return a.shape[0] * w.shape[0] * a.shape[1] / 1.5e11

    def launch(a, w, *args, **kw):
        dc.run("challenger", units(a, w))
        return a

    def today(index, a, w, *args, **kw):
        dc.run("incumbent", units(a, w))
        return a

    def int_mm(a, b):
        dc.run("incumbent", a.shape[0] * b.shape[1] * a.shape[1] / 1.5e11)
        return a

    to = torch.Tensor.to

    def to_device(self, *a, **k):
        dest = a[0] if a else k.get("device")
        if dest is not None and str(dest).startswith("cuda"):
            return self
        return to(self, *a, **k)

    def randint(
        low,
        high,
        size,
        generator = None,
        dtype = None,
        device = None,
        **kw,
    ):
        return torch.empty(size, dtype = dtype or torch.int64)

    names = (
        "_CUT_PLAN",
        "_CUT_FAMILY",
        "_CUT_RACES",
        "_CUT_DONE",
        "_CUT_CKEY",
        "_CUT_ROWS",
        "_CUT_ROWS_PENDING",
        "_CUT_LIVE",
    )
    state = [getattr(g8, n) for n in names if hasattr(g8, n)]  # older trees lack the later ones
    state += [getattr(cut, n) for n in ("_PROBED", "_PAIRED_ROWS") if hasattr(cut, n)]
    saved = [dict(x) for x in state]
    for x in state:
        x.clear()
    tiles = (0, 1, 2)
    with (
        on_tester_card(monkeypatch, card),
        droop_timers(monkeypatch, dc),
        monkeypatch.context() as m,
    ):
        m.setattr(cut, "launch", launch)
        m.setattr(
            cut, "library", lambda: object()
        )  # the prebuilt wheel is installed, as on the tester's cards
        m.setattr(cut, "eligible", lambda *a, **k: True)
        m.setattr(cut, "has_kernel", lambda *a, **k: True)
        m.setattr(cut, "probed_tiles", lambda index: tiles)
        m.setattr(
            cut,
            "tiles",
            lambda fam: tuple((128 if t < 2 else 256, 128, 64, 0, 0, 0, 0, 0) for t in tiles),
        )
        m.setattr(g8, "_launch_or_ref", today, raising = False)
        m.setattr(g8, "_cut_cache_key", lambda index: None, raising = False)
        m.setattr(g8, "_xs_dtype", lambda *a, **k: torch.float32, raising = False)
        m.setattr(g8, "_huge_race_fits", lambda *a, **k: True, raising = False)
        m.setattr(torch, "_int_mm", int_mm)
        m.setattr(torch, "randint", randint)
        m.setattr(torch.Tensor, "to", to_device)
        try:
            yield dc
        finally:
            for x, old in zip(state, saved):
                x.clear()
                x.update(old)


def cutlass_rows_served(rows: int, dit: Optional[str] = None) -> int:
    """H3 Linears whose ``rows``-row calls the race routes to CUTLASS (inside ``cutlass_on_droop``)."""
    import torch

    from core.inference import diffusion_int8_cutlass as cut
    from core.inference import diffusion_int8_gemm as g8

    if dit and hasattr(cut, "set_paired_rows"):
        cut.set_paired_rows(0, type(dit, (), {})())
    recs = []
    for n, k, _count in H3_SHAPES:
        w = types.SimpleNamespace(
            qdata = torch.empty(n, k, dtype = torch.int8), scale = torch.empty(n, 1, dtype = torch.bfloat16)
        )
        recs.append((types.SimpleNamespace(bias = None), ("v2", None, False, w, True)))
    plan = g8._cutlass_plan(0, recs, True)
    served = 0
    for n, k, count in H3_SHAPES:
        entry = plan.get(g8._cut_key(n, k, torch.float32, torch.bfloat16, False)) or {}
        if entry and _slot(rows, entry) is not None:
            served += count
    return served


def _slot(rows: int, entry: dict) -> Optional[int]:
    """The tile a ``rows``-row call runs: ``cut.slot``; trees before #18105 routed >= 1024 rows to the 4096 race."""
    from core.inference import diffusion_int8_cutlass as cut
    from core.inference import diffusion_int8_gemm as g8

    if hasattr(cut, "slot"):
        return cut.slot(rows, entry)
    return entry.get(max(cut._RACE_M) if rows >= g8._CUT_LARGE_M else min(cut._RACE_M))


def test_h3_video_power_capped_ada_keeps_triton_at_video_rows(monkeypatch):
    """r8 / r10 logged 52 CUTLASS Linears on the Ada (every budget) at 5.37-5.40 s/step, ~1225 MHz; r9 logged 313 at
    6.13 s/step, ~1045 MHz, with the GEMM files unchanged (a race outcome; #19807's pair check fixed it). On the
    R9-class card the race must leave every 19318-row H3 call on today's kernel."""
    good = {x["dec"]["cutlass_linears"] for x in cells("h3v.gpu0.", ("r8", "r10"))}
    bad = {x["dec"]["cutlass_linears"] for x in cells("h3v.gpu0.", ("r9",), ("r9m1",))}
    assert good == {52} and bad == {313}, (good, bad)
    with cutlass_on_droop(monkeypatch, CARDS["gpu0"], DROOP["ada_paired"]) as dc:
        served = cutlass_rows_served(VIDEO_ROWS)
        assert dc.calls["challenger"] > 0
    assert (
        served == 0
    ), f"{served} H3 video Linears on CUTLASS at {VIDEO_ROWS} rows on the power-capped Ada (r9: 300)"


def test_h3_video_clock_stable_3090_keeps_cutlass(monkeypatch):
    """Control: the 3090 logged 313 CUTLASS Linears in every round with no clock loss; the Ada fix must not take them
    from a card whose clock holds."""
    assert {x["dec"]["cutlass_linears"] for x in cells("h3v.gpu1.")} == {313}
    with cutlass_on_droop(monkeypatch, CARDS["gpu1"], None):
        served = cutlass_rows_served(VIDEO_ROWS)
    assert served == sum(c for _n, _k, c in H3_SHAPES) == 300


@pytest.mark.parametrize("droop", sorted(DROOP))
@pytest.mark.parametrize("rows", [VIDEO_ROWS, IMAGE_ROWS], ids = ["h3_video_rows", "h3_image_rows"])
def test_cutlass_race_keeps_the_incumbent_on_a_drooping_card(droop, rows, monkeypatch):
    """Both int8 GEMM CUTLASS race consumers the tester times (H3 video at 19318 rows, H3-Image at 2048^2) keep
    today's kernel when the tile wins a burst but loses held (L4 profile) or beside today's kernel (Ada profile)."""
    with cutlass_on_droop(monkeypatch, CARDS["gpu0"], DROOP[droop]) as dc:
        served = cutlass_rows_served(rows, H3_IMAGE_DIT if rows == IMAGE_ROWS else None)
        assert dc.calls["challenger"] > 0, "the race never timed the tile"
    assert served == 0, f"{droop}: {served} Linears on CUTLASS at {rows} rows"


@pytest.mark.xfail(
    strict = True,
    reason = "bound of the CUTLASS race: its sustained window (0.25 s settle + 0.25 s per candidate, back to back) "
    "cannot see a clock that responds with tau 0.33 s (an L4-like ~1 s settle); the first tile after today's kernel "
    "inherits its clock and wins. Flips to XPASS when the race resolves slower clocks.",
)
@pytest.mark.parametrize("rows", [VIDEO_ROWS, IMAGE_ROWS], ids = ["h3_video_rows", "h3_image_rows"])
def test_cutlass_race_with_a_one_second_clock_settle(rows, monkeypatch):
    slow = dataclasses.replace(DROOP["l4_burst"], tau = 0.33)
    with cutlass_on_droop(monkeypatch, CARDS["gpu0"], slow):
        served = cutlass_rows_served(rows, H3_IMAGE_DIT if rows == IMAGE_ROWS else None)
    assert served == 0, served


@contextlib.contextmanager
def int8_attention_on_droop(monkeypatch, dc: DroopCard, heads: int, head_dim: int):
    """The REAL ``diffusion_int8_attention_dispatch.long_verdict`` (H3 video's Auto race) timing on ``dc``: the INT8
    kernel bills the challenger, the incumbent attention the incumbent; work per call 4 S^2 H D (~20 ms at 16384
    tokens on the Ada). Yields the incumbent callable and the per-call work for ``tokens``."""
    import torch

    from core.inference import diffusion_int8_attention as i8
    from core.inference import diffusion_int8_attention_dispatch as dispatch

    size = {"n": 0}

    def work(n: Optional[int] = None) -> float:
        n = size["n"] if n is None else n
        return 4.0 * n * n * heads * head_dim / 3e11

    def probe_inputs(
        device,
        dtype,
        n,
        h,
        hd,
        seed = 0,
    ):
        size["n"] = n
        return tuple(torch.zeros(1, 8, 1, hd) for _ in range(3))

    def int8_attention(q, k, v, *a, **kw):
        dc.run("challenger", work())
        return v

    def incumbent(q, k, v):
        dc.run("incumbent", work())
        return v

    store = types.SimpleNamespace(get = lambda *a, **k: None, put = lambda *a, **k: None)
    with (
        on_tester_card(monkeypatch, CARDS["gpu0"]),
        droop_timers(monkeypatch, dc),
        monkeypatch.context() as m,
    ):
        m.setattr(i8, "probe_inputs", probe_inputs)
        m.setattr(i8, "int8_attention", int8_attention)
        m.setattr(i8, "implementation", lambda *a, **k: "cuda")
        m.setattr(i8, "card_name", lambda *a, **k: "virtual")
        m.setattr(i8, "persisted", lambda: store)
        m.setattr(dispatch, "_LONG_VERDICTS", {})
        yield incumbent, work


@pytest.mark.xfail(
    strict = True,
    reason = "latent gap, not a measured regression: the INT8 attention long race (dispatch._race_wins -> "
    "_median_ms) times the two kernels interleaved, one call each, so both run at the shared clock and a kernel that "
    "only loses once held at its own clock still wins. On the tester's cards INT8 attention wins held (R7: the Ada "
    "lost 25% with it off), so nothing regressed; the race has no held stage like the CUTLASS one (#18105).",
)
def test_int8_attention_long_race_keeps_the_incumbent_on_a_drooping_card(monkeypatch):
    """H3 video's Auto INT8 attention race on the L4 profile: INT8 wins a burst ~23% and runs ~16% slower held."""
    import torch

    from core.inference import diffusion_int8_attention as i8
    from core.inference import diffusion_int8_attention_dispatch as dispatch

    d = DROOP["l4_burst"]
    dc = DroopCard(d)
    heads, hd = 56, 128
    tokens = tuple(i8.FAMILY_RACE_TOKENS["MiniMaxH3Transformer3DModel"])
    with int8_attention_on_droop(monkeypatch, dc, heads, hd) as (incumbent, work):
        floor = dispatch.long_verdict(
            "cuda:0", torch.bfloat16, heads, hd, incumbent, "_native_flash", tokens = tokens
        )
        held = {
            cls: held_s_per_call(d, [cls], work(tokens[0])) for cls in ("incumbent", "challenger")
        }
    assert dc.calls["challenger"] > 0 and held["challenger"] > held["incumbent"] * 1.10, held
    assert (
        floor is None
    ), f"INT8 attention engaged from {floor} tokens; held it runs {held['challenger'] / held['incumbent']:.2f}x"


def test_int8_attention_long_race_engages_where_it_wins_held(monkeypatch):
    """Control for the xfail above: on the Ada profile INT8 attention wins held, and the race engages it at the
    floor the tester's r10 Ada log printed (``long race min_tokens=6144``)."""
    import torch

    from core.inference import diffusion_int8_attention as i8
    from core.inference import diffusion_int8_attention_dispatch as dispatch

    dc = DroopCard(DROOP["ada_paired"])
    tokens = tuple(i8.FAMILY_RACE_TOKENS["MiniMaxH3Transformer3DModel"])
    with int8_attention_on_droop(monkeypatch, dc, 56, 128) as (incumbent, _work):
        floor = dispatch.long_verdict(
            "cuda:0", torch.bfloat16, 56, 128, incumbent, "_native_flash", tokens = tokens
        )
    assert floor == 6144, floor


@pytest.mark.parametrize("tau", [TAU_S, 0.33])
def test_batching_race_keeps_serial_on_a_drooping_card(tau, monkeypatch):
    """Cross-request batching race (``diffusion_batch_race.run_race``, injectable clock): the batched call of k
    requests needs 0.77x the cycles of k solo renders but settles at the L4 challenger clock, so held it loses ~16%;
    the race times whole renders and must not promote it (also with the slow 0.33 s clock response)."""
    from core.inference import diffusion_batch_race as br

    d = dataclasses.replace(DROOP["l4_burst"], tau = tau)
    dc = DroopCard(d)

    def generate(
        prompt = None,
        seed = None,
        prompts = None,
        seeds = None,
        batch_size = None,
        **kw,
    ):
        rows = len(prompts) if prompts else 1
        if rows == 1:
            dc.run("incumbent", 1.2e3)  # ~0.9 s at 1290 MHz: a short image render
        else:
            for _ in range(10):
                dc.run("challenger", 1.2e3 * rows / 10)
        dc.idle(0.002)  # back to back, as the tester's client
        return {"images": [object()] * rows}

    state = types.SimpleNamespace(cuda_graphs = (), speed_optims = (), speed_deferred = False)
    report = br.run_race(
        generate,
        state,
        idle = lambda: True,
        width = 1024,
        height = 1024,
        steps = 4,
        guidance = 1.0,
        sizes = (2,),
        repeats = 5,
        accuracy = False,
        persist = False,
        clock = dc.now,
    )
    verdict = report["sizes"]["2"]
    assert dc.calls["challenger"] > 0 and verdict.get("promoted") is False, verdict


# ------------------------------------------------------------------------------------------------ 3. structural lint


# Every module-level table of CUDA capabilities under core/inference (a set / frozenset / tuple / list of (major, minor)
# tuples, a dict keyed by them, or such tables summed): what it holds and the measurement it came from. An Auto lever's
# per-arch on / off list lives in one of these; a new one, or a card added to / dropped from one, fails until it is
# listed here with its measurement.
CAPABILITY_TABLES = {
    "diffusion_attention.py:FLASH_FASTER_CAPABILITIES": (
        [(8, 0), (8, 9)],
        "native flash faster than the cuDNN pin at every DiT/U-Net shape, bf16+fp16, torch 2.14.1+cu130 (A100 FLUX 0.92x, Lumina-2 0.74x, SD3.5 0.97x per cuDNN call; L4)",
    ),
    "diffusion_convrot_quant.py:_ROTQ_CONFIG": (
        [(8, 0), (8, 9), (12, 0)],
        "fused ConvRot act-quant tiles measured end to end per arch; absent = stock (R1: no sm86 entry cost the 3090 22% until #14988 / #15654)",
    ),
    "diffusion_convrot_quant.py:_ROTQ_K_TILES": (
        [(8, 0), (8, 9), (12, 0)],
        "the same per-arch ConvRot sweep: narrow K tiles",
    ),
    "diffusion_convrot_quant.py:_H16_TF32X3_ARCHS": (
        [(8, 0), (9, 0), (10, 0), (10, 3)],
        "tf32x3 Hadamard where fp32 FMA rate is a fraction of tf32 (A100 1.2x faster); sm86 / sm89 / sm120 keep IEEE FMAs",
    ),
    "diffusion_convrot_quant.py:_H16X_WARPS": (
        [(8, 0), (8, 9)],
        "Colab sweep (A100: 557 vs 601 us two-pass at 4608 x 12288)",
    ),
    "diffusion_convrot_quant.py:_H16X_PAD_CAT": (
        [(8, 0)],
        "same Colab sweep: padded one-pass cap on A100",
    ),
    "diffusion_convrot_quant.py:_SITU_ARCHS": (
        [(8, 9), (12, 0)],
        "MOSS-Image SiTU-GLU in the act quant: RTX PRO 6000 4.914 -> 4.859 s/image, L4 30.76 -> 29.45; A100 slower 8.791 -> 8.985",
    ),
    "diffusion_convrot_quant.py:_GELU_ARCHS": (
        [(8, 9), (12, 0)],
        "FLUX.1-schnell ConvRot INT8 fused GELU producer measured end to end on L4 and RTX PRO 6000",
    ),
    "diffusion_convrot_quant.py:_GELU_MIN_ROWS": (
        [(8, 9), (12, 0)],
        "row floors from the same sweep (L4 from 1280 rows 1.22-1.50x; RTX PRO 6000 4352 rows 1.28x)",
    ),
    "diffusion_cuda_graph.py:FP8_ROWWISE_STREAMK_CAPS": (
        [(8, 9)],
        "correctness: torch < 2.12 sm89 row-wise fp8 stream-K memsets outside capture, replay hangs (pytorch/pytorch#177573)",
    ),
    "diffusion_families.py:_REDUCTION_RACE_ARCHS": (
        [(8, 0), (8, 9), (12, 0)],
        "determinism: fresh processes rendered one seed into 2-4 variants (norm-reduction configs near-tied)",
    ),
    "diffusion_families.py:_FLUX1_REDUCTION_RACE_ARCHS": (
        [(8, 0), (8, 9), (10, 0), (12, 0)],
        "determinism: FLUX.1 B200 15 fresh processes 3 latents; 12/12 bitwise with the filter",
    ),
    "diffusion_families.py:_QWEN_REDUCTION_RACE_ARCHS": (
        [(8, 0), (8, 9), (10, 0), (12, 0)],
        "determinism: Qwen-Image B200 26 dB apart; 6/6 bitwise with the filter",
    ),
    "diffusion_fp8_unit_scale.py:_ADA_CAPS": (
        [(8, 9)],
        "fp8 unit-scale path is the Ada fp8 GEMM route",
    ),
    "diffusion_grouped_int8_gemm.py:_ARCH_TILES": (
        [(8, 9), (12, 0)],
        "grouped int8 GEMM tiles: sm120 wins <= 64 rows 6 bits, sm89 <= 128 rows (0 spills on L4); sm80 spills under Triton 3.8",
    ),
    "diffusion_int8_attention.py:KERNEL_CONFIGS": (
        [(7, 5), (8, 0), (8, 6), (8, 9), (12, 0)],
        "INT8 attention CUDA kernel configs per measured card and head dim (T4, A100, 3090, L4 / Ada, RTX PRO 6000)",
    ),
    "diffusion_int8_attention.py:TRITON_KERNEL_CONFIGS": (
        [(8, 0), (8, 6), (8, 9), (12, 0)],
        "INT8 attention Triton stand-in configs per measured card",
    ),
    "diffusion_int8_gemm.py:_ARCH_CONFIG": (
        [(8, 0), (8, 9), (12, 0)],
        "fused int8 GEMM tile per arch tuned on DiT shapes; unlisted archs race at load (#15654)",
    ),
    "diffusion_int8_gemm.py:_UNLISTED_SKIP": (
        [(10, 0), (10, 3)],
        "unlisted-arch race skipped: measured slower than cuBLAS on B200 / B300",
    ),
    "diffusion_int8_gemm.py:_SHAPE_TILES": (
        [(12, 0)],
        "per-shape tiles on RTX PRO 6000; no L4 rule on purpose (power-capped clocks lose end to end)",
    ),
    "diffusion_int8_gemm.py:_TMA_CONFIG": (
        [(12, 0)],
        "host-TMA tile for M >= _SHAPE_MIN_M on RTX PRO 6000",
    ),
    "diffusion_nvfp4_install.py:FLASHINFER_JIT_CACHE_ARCH": (
        [(10, 0), (10, 3), (12, 0)],
        "packaging: which flashinfer jit-cache provider holds each arch's NVFP4 kernels",
    ),
    "diffusion_nvfp4_ops.py:NVFP4_FLASHINFER_CAPS": (
        [(10, 0), (10, 3), (12, 0)],
        "hardware: NVFP4 tensor cores (necessary condition, preflight decides)",
    ),
    "diffusion_transformer_quant.py:_AUTO_LADDER": (
        [(8, 0), (8, 9), (10, 0)],
        "Auto quant ladder floors: int8 full rate on consumer cards, fp8 from sm89, mxfp8 from sm100",
    ),
    "diffusion_transformer_quant.py:_HOSTED_AUTO_FLOORS": (
        [(7, 5), (8, 9)],
        "hosted checkpoint floors: int8 from Turing (_int_mm), fp8 from sm89",
    ),
    "diffusion_transformer_quant.py:_FP16_CARD_INT8_CAPS": (
        [(7, 5), (8, 0)],
        "hardware: sm70 has no int8 tensor cores; Ampere+ bf16 through torchao",
    ),
    "diffusion_vae_strips.py:ALWAYS_CAPS": (
        [(7, 5), (8, 9)],
        "strip decode within A/A noise of the fused one or faster on T4 and L4 (A100 3% slower)",
    ),
    "diffusion_vae_strips.py:CARRY_CAPS": (
        [(7, 5), (8, 0), (8, 9), (12, 0)],
        "carried-bias strips 3.5-7.4% faster than fused on T4, A100, L4, RTX PRO 6000 (FLUX.1 / SDXL / FLUX.2 at 1024 and 2048 px)",
    ),
    "diffusion_vae_strips.py:_FOLD_CAPS": (
        [(7, 5), (8, 0), (8, 9), (12, 0)],
        "the same strip sweep: folded upsample",
    ),
    "video_minimax_h3_attn.py:H3_FLASH_FASTER_ARCHS": (
        [(8, 0), (8, 9)],
        "H3 shape, torch 2.11: flash beats cuDNN on sm80 / sm89, ties or loses on sm120; sm86 unmeasured keeps cuDNN (tester R4 row)",
    ),
    "video_minimax_h3_vae.py:_AUDIO_VAE_NO_SEARCH_CAPABILITIES": (
        [(8, 0), (10, 0), (12, 0)],
        "skipping the cuDNN search <= 1.4% slower there; T4 / L4 gain 17-19% / 2-8% from it",
    ),
}
# Tables that may legitimately be empty on a tree (deny-lists): listed with their measurement, must stay a known name.
EMPTY_OK = {
    "video_minimax_h3_attn.py:H3_INT8_AUTO_OFF_ARCHS": "H3 video Auto INT8 attention deny-list: empty since #19344 (#19098's (8, 9) entry cost the Ada 25% in R7)",
}


def _is_cap(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Tuple)
        and len(node.elts) == 2
        and all(isinstance(e, ast.Constant) and type(e.value) is int for e in node.elts)
        and 5 <= node.elts[0].value <= 15
        and 0 <= node.elts[1].value <= 9
    )


def _members(v: ast.AST, depth: int = 0) -> list:
    """(major, minor) tuples that are members of a collection literal: elements, dict keys, pair elements, through
    frozenset() / set() / tuple() calls, comprehensions over them and ``+`` / ``|`` of tables."""
    out: list = []
    if (
        isinstance(v, ast.Call)
        and isinstance(v.func, ast.Name)
        and v.func.id in ("frozenset", "set", "tuple", "list")
    ):
        return _members(v.args[0], depth) if v.args else []
    if isinstance(v, (ast.Set, ast.List, ast.Tuple)) and not _is_cap(v):
        for e in v.elts:
            if _is_cap(e):
                out.append(e)
            elif depth < 1:
                out += _members(e, depth + 1)
    elif isinstance(v, ast.Dict):
        out += [k for k in v.keys if k is not None and _is_cap(k)]
    elif isinstance(v, (ast.DictComp, ast.SetComp, ast.ListComp)):
        for g in v.generators:
            out += _members(g.iter, depth)
    elif isinstance(v, ast.BinOp):
        out += _members(v.left, depth) + _members(v.right, depth)
    return out


def capability_tables(root: Path) -> dict:
    """{"<module path>:<NAME>": sorted capabilities} for every module-level capability table under ``root``. A table
    built from other tables (``A + ((10, 0),)``) also holds what its named parts hold."""
    found: dict = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        local: dict = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names, value = [t.id for t in node.targets if isinstance(t, ast.Name)], node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                names, value = [node.target.id], node.value
            else:
                continue
            caps = {(e.elts[0].value, e.elts[1].value) for e in _members(value)}
            for sub in ast.walk(value) if isinstance(value, ast.BinOp) else ():
                if isinstance(sub, ast.Name) and sub.id in local:
                    caps |= local[sub.id]
            empty_deny = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in ("frozenset", "set")
                and not value.args
                and any(n.endswith(("_ARCHS", "_CAPS", "_CAPABILITIES")) for n in names)
            )
            for name in names:
                if caps or empty_deny:
                    local[name] = caps
                    found[f"{rel}:{name}"] = sorted(caps)
    return found


def test_every_cuda_capability_table_is_listed_with_its_measurement():
    """A per-arch table that turns an Auto lever on or off must be reviewed: #19098 added
    ``H3_INT8_AUTO_OFF_ARCHS = frozenset({(8, 9)})`` and the tester's Ada H3 video cell lost 25% (R7); R1's ConvRot
    tables had no sm86 entry and the 3090 lost 22%. New tables, and cards added to or dropped from a listed one, fail
    here until ``CAPABILITY_TABLES`` names the measurement."""
    got = capability_tables(_BACKEND / "core" / "inference")
    want = {k: sorted(tuple(c) for c in v[0]) for k, v in CAPABILITY_TABLES.items()}
    problems = []
    for key, caps in sorted(got.items()):
        if key in EMPTY_OK:
            if caps:
                problems.append(
                    f"{key} = {caps}: a deny-list that was empty now excludes cards; measure and list it"
                )
            continue
        if key not in want:
            problems.append(
                f"NEW capability table {key} = {caps}: add it to CAPABILITY_TABLES with its measurement"
            )
        elif caps != want[key]:
            problems.append(
                f"{key}: was {want[key]}, now {caps}: update CAPABILITY_TABLES with the measurement"
            )
    for key in sorted(set(want) - set(got)):
        problems.append(f"{key} is gone: drop it from CAPABILITY_TABLES")
    assert not problems, "\n".join(problems)
    assert all(
        len(v[1]) > 20 for v in CAPABILITY_TABLES.values()
    ), "every table names its measurement"


def test_the_lint_sees_a_19098_style_exclusion(tmp_path):
    """The lint bites: a new module-level deny-list of one capability, a card added to a listed table and a table
    built from another are all found."""
    (tmp_path / "x.py").write_text(
        "A_ARCHS = frozenset({(8, 9)})\nB = {(8, 0): 1, (8, 6): 2}\nC = B_PART + ((10, 0),)\nB_PART = ((8, 9),)\n"
        "NOT_CUDA = (3, 2)\nVALUES = {'L': (1, 0)}\nEMPTY_ARCHS = frozenset()\n",
        encoding = "utf-8",
    )
    got = capability_tables(tmp_path)
    assert got == {
        "x.py:A_ARCHS": [(8, 9)],
        "x.py:B": [(8, 0), (8, 6)],
        "x.py:C": [(10, 0)],
        "x.py:B_PART": [(8, 9)],
        "x.py:EMPTY_ARCHS": [],
    }, got
