# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The VRAM Budget setting on sd.cpp image loads: a budget lowered below the default passes ``--max-vram``, an
untouched one leaves the argv exactly as before. Hermetic: no binary, no GPU, no socket."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

from core.inference import sd_cpp_args as args_mod
from core.inference import sd_cpp_backend as bk
from core.inference.sd_cpp_backend import SdCppDiffusionBackend
from utils import vram_budget_settings

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_sd_cpp_backend import _FakeEngine, _loaded_backend, _run_server_load  # noqa: E402

GiB = 1 << 30
DEFAULT = vram_budget_settings.VRAM_FRACTION_DEFAULT
_OPTIONS = frozenset({"--max-vram", "--stream-layers"})


def _budget(flags = (), *, free = 20 * GiB, total = 24 * GiB, fraction = 0.80, options = _OPTIONS, env = None):
    return args_mod.lowered_vram_budget_flags(
        flags,
        free_bytes = free,
        total_bytes = total,
        fraction = fraction,
        default = DEFAULT,
        options = options,
        env = env,
    )


def test_a_lowered_budget_is_total_times_fraction():
    # 24 GiB * 0.80 = 19.2 GiB, capped by 20 GiB free less the 2 GiB reserve.
    assert _budget() == ["--max-vram", "18.00"]
    assert _budget(free = 23 * GiB, fraction = 0.50) == ["--max-vram", "12.00"]


@pytest.mark.parametrize("fraction", [DEFAULT, 0.98, 1.0, None])
def test_the_default_budget_adds_nothing(fraction):
    assert _budget(fraction = fraction) == []


def test_guards_keep_the_argv_unchanged():
    # A build without the flag, the kill switch, an unread card, a card too full, or a --max-vram already present.
    assert _budget(options = frozenset()) == []
    assert _budget(env = {args_mod.GRAPH_CUT_EXPLICIT_BUDGET_ENV: "0"}) == []
    assert _budget(free = None) == []
    assert _budget(total = None) == []
    assert _budget(free = int(2.5 * GiB)) == []
    assert _budget(fraction = 0.10, total = 8 * GiB) == []  # 0.8 GiB is under the 1 GiB floor
    assert _budget(["--max-vram", "-17.05"]) == []
    assert _budget(["--max-vram", "-1"]) == []


@pytest.fixture
def cuda_card(monkeypatch):
    monkeypatch.delenv(args_mod.GRAPH_CUT_EXPLICIT_BUDGET_ENV, raising = False)
    monkeypatch.setattr(bk, "sd_cpp_graph_cut_options", lambda binary: _OPTIONS)
    reads: list = []

    def _card(device, ordinal, pinned):
        reads.append((device, ordinal, pinned))
        return 20 * GiB, 24 * GiB

    monkeypatch.setattr(bk, "_image_card_memory_bytes", _card, raising = False)
    return reads


def _set_fraction(monkeypatch, fraction):
    monkeypatch.setattr(vram_budget_settings, "get_vram_budget_fraction", lambda: fraction)


def test_server_load_under_a_lowered_budget_passes_max_vram(monkeypatch, cuda_card):
    _set_fraction(monkeypatch, 0.80)
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers, device = "cuda")
    assert servers[0].started["offload"][-2:] == ["--max-vram", "18.00"]
    assert b._state.vram_budget_flags == ("--max-vram", "18.00")
    # A budget is not an offload: status still reports none.
    assert "--max-vram" not in b._state.offload_flags
    assert b.status()["cpu_offload"] is False


@pytest.mark.parametrize("fraction", [DEFAULT, 1.0])
def test_server_load_at_the_default_budget_is_unchanged(monkeypatch, cuda_card, fraction):
    _set_fraction(monkeypatch, fraction)
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers, device = "cuda")
    assert "--max-vram" not in servers[0].started["offload"]
    assert b._state.vram_budget_flags == ()
    assert cuda_card == []  # the card is not even read


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_off_cuda_loads_never_get_a_budget(monkeypatch, cuda_card, device):
    _set_fraction(monkeypatch, 0.50)
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers, device = device)
    assert "--max-vram" not in (servers[0].started["offload"] or [])
    assert cuda_card == []


def test_oneshot_generation_carries_the_budget(monkeypatch):
    fake = _FakeEngine()
    b = _loaded_backend(engine = fake)
    b._state = dataclasses.replace(
        b._state, offload_flags = ("--offload-to-cpu",), vram_budget_flags = ("--max-vram", "12.00")
    )
    b.generate(prompt = "budget", steps = 1, seed = 1)
    assert fake.calls[0][3]["offload"] == ["--offload-to-cpu", "--max-vram", "12.00"]

    fake = _FakeEngine()
    b = _loaded_backend(engine = fake)
    b.generate(prompt = "default", steps = 1, seed = 1)
    assert fake.calls[0][3]["offload"] is None


def test_unpinned_multi_gpu_reads_no_card(monkeypatch):
    from core.inference import video as video_mod

    monkeypatch.setattr(video_mod, "_h3_visible_card_count", lambda device: 2)
    monkeypatch.setattr(
        video_mod, "_h3_card_memory_bytes", lambda *a, **k: pytest.fail("read an unpinned card")
    )
    assert bk._image_card_memory_bytes("cuda", None, False) == (None, None)
