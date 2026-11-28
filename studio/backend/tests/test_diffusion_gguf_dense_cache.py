# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF dense cache: a GGUFLinear keeps the weight its first forward dequantised, later forwards skip the dequant and
match the per-step layer bit for bit (stock and llama.cpp-exact forwards, bf16 / fp16), auto engages on MPS only,
the memory reserve and byte cap stop the fill, a swapped weight is re-captured, uninstall restores. Runs on CPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")
gguf_utils = pytest.importorskip("diffusers.quantizers.gguf.utils")

from core.inference import diffusion_gguf_dense_cache as dc  # noqa: E402
from core.inference import diffusion_gguf_exact as ex  # noqa: E402

T = gguf.GGMLQuantizationType
# byte offsets of each block's fp16 scale fields (ggml-common.h)
F16_OFFSETS = {"Q8_0": (0,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,), "Q4_0": (0,)}
_HUGE = 1 << 50


def _raw(name: str, rows: int, cols: int, seed: int) -> "torch.Tensor":
    bs, sz = gguf.GGML_QUANT_SIZES[getattr(T, name)]
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (rows, cols // bs, sz), dtype = np.uint8)
    for off in F16_OFFSETS[name]:
        scale = np.float16(rng.uniform(0.002, 0.02, raw.shape[:2])).view(np.uint16)
        raw[..., off] = scale & 0xFF
        raw[..., off + 1] = scale >> 8
    return torch.from_numpy(raw.reshape(rows, -1).copy())


def _layer(
    name: str,
    dtype,
    *,
    rows = 48,
    cols = 512,
    seed = 0,
    bias = True,
):
    layer = gguf_utils.GGUFLinear(cols, rows, bias = bias, compute_dtype = dtype, device = "meta")
    layer.weight = gguf_utils.GGUFParameter(
        _raw(name, rows, cols, seed), quant_type = getattr(T, name)
    )
    if bias:
        g = torch.Generator().manual_seed(seed)
        layer.bias = torch.nn.Parameter(torch.randn(rows, generator = g) * 0.1)
    layer.device = None
    layer.requires_grad_(False)
    return layer


class _Net(torch.nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        return [layer(x) for layer in self.layers]


@pytest.fixture
def counted(monkeypatch):
    """Counts the class forward (the per-step dequant + matmul) each GGUFLinear call reaches."""
    calls = {"n": 0}
    inner = gguf_utils.GGUFLinear.forward_native

    def forward_native(self, inputs):
        calls["n"] += 1
        return inner(self, inputs)

    monkeypatch.setattr(gguf_utils.GGUFLinear, "forward_native", forward_native)
    return calls


@pytest.fixture(params = [False, True], ids = ["stock", "exact"])
def exact(request, monkeypatch):
    monkeypatch.delenv(ex.EXACT_ENV, raising = False)
    if request.param:
        ex.install_exact_dequant()
        yield True
        ex.uninstall_exact_dequant()
    else:
        ex.uninstall_exact_dequant()
        yield False


def _inputs(
    dtype,
    seeds,
    cols = 512,
):
    return [
        torch.randn(2, 7, cols, generator = torch.Generator().manual_seed(s)).to(dtype) for s in seeds
    ]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids = ["bf16", "fp16"])
@pytest.mark.parametrize("name", list(F16_OFFSETS))
def test_kept_weight_is_bit_identical_and_dequantises_once(
    monkeypatch, counted, exact, name, dtype
):
    monkeypatch.setenv(dc.ENV, "1")
    layer, ref = _layer(name, dtype), _layer(name, dtype)
    assert dc.install(layer, device = "cpu", free = lambda: _HUGE, reserve = 0) == 1
    xs = _inputs(dtype, (1, 2, 3, 4))
    want = [ref(x) for x in xs]
    counted["n"] = 0
    got = [layer(x) for x in xs]
    assert counted["n"] == 1, "only the first forward dequantises"
    for a, b in zip(got, want):
        assert a.dtype == b.dtype and torch.equal(a, b)
    assert dc.kept_bytes(layer) >= 48 * 512 * 2


def test_auto_is_mps_only(monkeypatch):
    monkeypatch.delenv(dc.ENV, raising = False)
    net = _Net(_layer("Q4_K", torch.bfloat16))
    assert dc.install(net, device = "cpu") == 0
    assert "forward_native" not in net.layers[0].__dict__
    assert dc.decline_reason("cuda") is not None
    assert dc.install(net, device = "mps", free = lambda: _HUGE, reserve = 0) == 1
    assert "forward_native" in net.layers[0].__dict__


def test_off_switch(monkeypatch):
    monkeypatch.setenv(dc.ENV, "0")
    net = _Net(_layer("Q4_K", torch.bfloat16))
    assert dc.install(net, device = "mps", free = lambda: _HUGE, reserve = 0) == 0


def test_reserve_stops_the_fill_and_the_rest_stay_per_step(monkeypatch, counted):
    monkeypatch.setenv(dc.ENV, "1")
    layers = [_layer("Q4_K", torch.bfloat16, seed = s) for s in range(3)]
    refs = [_layer("Q4_K", torch.bfloat16, seed = s) for s in range(3)]
    one = 48 * 512 * 2
    net = _Net(*layers)

    def free():  # unified memory: what the kept weights take is gone
        return 2 * one - dc.kept_bytes(net)

    assert dc.install(net, device = "cpu", free = free, reserve = one) == 3
    xs = _inputs(torch.bfloat16, (5, 6, 7))
    want = [[r(x) for r in refs] for x in xs]
    counted["n"] = 0
    for x, w in zip(xs, want):
        got = net(x)
        assert all(torch.equal(a, b) for a, b in zip(got, w))
    # layer 0 kept; layer 1 would leave less than the reserve, so the fill stops for every layer
    assert dc.kept_bytes(net) == dc.kept_bytes(layers[0]) > 0
    assert counted["n"] == 3 + 2 * 2


def test_byte_cap(monkeypatch):
    monkeypatch.setenv(dc.ENV, "1")
    monkeypatch.setenv(dc.CAP_ENV, "0")
    layer = _layer("Q8_0", torch.bfloat16)
    dc.install(layer, device = "cpu", free = lambda: _HUGE, reserve = 0)
    layer(_inputs(torch.bfloat16, (1,))[0])
    assert dc.kept_bytes(layer) == 0


def test_unreadable_memory_keeps_nothing(monkeypatch):
    monkeypatch.setenv(dc.ENV, "1")
    layer = _layer("Q8_0", torch.bfloat16)
    dc.install(layer, device = "cpu", free = lambda: None, reserve = 0)
    layer(_inputs(torch.bfloat16, (1,))[0])
    assert dc.kept_bytes(layer) == 0


def test_swapped_weight_is_recaptured(monkeypatch):
    monkeypatch.setenv(dc.ENV, "1")
    layer = _layer("Q4_K", torch.bfloat16, seed = 1)
    dc.install(layer, device = "cpu", free = lambda: _HUGE, reserve = 0)
    x = _inputs(torch.bfloat16, (9,))[0]
    layer(x)
    other = _layer("Q4_K", torch.bfloat16, seed = 2)
    layer.weight, layer.bias = other.weight, other.bias
    assert torch.equal(layer(x), other(x))
    assert torch.equal(layer(x), other(x))


def test_uninstall_restores_the_class_forward(monkeypatch):
    monkeypatch.setenv(dc.ENV, "1")
    net = _Net(_layer("Q4_K", torch.bfloat16))
    dc.install(net, device = "cpu", free = lambda: _HUGE, reserve = 0)
    net(_inputs(torch.bfloat16, (1,))[0])
    assert dc.kept_bytes(net) > 0
    assert dc.uninstall(net) == 1
    assert dc.kept_bytes(net) == 0 and "forward_native" not in net.layers[0].__dict__


def test_a_forward_that_is_not_one_linear_is_declined(monkeypatch):
    monkeypatch.setenv(dc.ENV, "1")
    layer, ref = _layer("Q4_K", torch.bfloat16), _layer("Q4_K", torch.bfloat16)

    def twice(self, inputs):
        w = gguf_utils.dequantize_gguf_tensor(self.weight).to(self.compute_dtype)
        h = torch.nn.functional.linear(inputs, w)
        return torch.nn.functional.linear(h[..., :0].new_zeros(inputs.shape), w) + h

    monkeypatch.setattr(gguf_utils.GGUFLinear, "forward_native", twice)
    dc.install(layer, device = "cpu", free = lambda: _HUGE, reserve = 0)
    x = _inputs(torch.bfloat16, (3,))[0]
    assert torch.equal(layer(x), twice(ref, x))
    assert dc.kept_bytes(layer) == 0 and layer.__dict__.get("_unsloth_gguf_dense_declined")


def test_load_installs_it_for_gguf_transformers():
    import inspect

    from core.inference import diffusion

    src = inspect.getsource(diffusion)
    at = src.index("gguf_dense_cache.install(")
    assert src.rfind("if gguf_transformer:", 0, at) > src.rfind("install_mps_fp16(", 0, at) > 0


def test_autograd_inputs_take_the_per_step_path(monkeypatch, counted):
    monkeypatch.setenv(dc.ENV, "1")
    layer = _layer("Q4_K", torch.bfloat16)
    dc.install(layer, device = "cpu", free = lambda: _HUGE, reserve = 0)
    x = _inputs(torch.bfloat16, (1,))[0].requires_grad_(True)
    counted["n"] = 0
    layer(x).sum().backward()
    layer(x)
    assert counted["n"] == 2 and dc.kept_bytes(layer) == 0 and x.grad is not None


def test_a_dequant_path_installed_later_is_recaptured(monkeypatch):
    # The kept tensor came from the stock forward; once the exact forward is installed, its bits must be served.
    monkeypatch.setenv(dc.ENV, "1")
    monkeypatch.delenv(ex.EXACT_ENV, raising = False)
    ex.uninstall_exact_dequant()
    layer, ref = _layer("Q6_K", torch.bfloat16, seed = 4), _layer("Q6_K", torch.bfloat16, seed = 4)
    dc.install(layer, device = "cpu", free = lambda: _HUGE, reserve = 0)
    x = _inputs(torch.bfloat16, (2,))[0]
    assert torch.equal(layer(x), ref(x))
    try:
        ex.install_exact_dequant()
        assert torch.equal(
            layer(x), ref(x)
        ), "served the stock-path weight after the exact path was installed"
    finally:
        ex.uninstall_exact_dequant()


# ---- footprint pricing: on MPS the fill pooled about one more kept-size of dequant scratch (task 0042)

GB = 10**9
GIB = 1024**3


def _no_footprint_env(monkeypatch):
    monkeypatch.delenv(dc.FOOTPRINT_ENV, raising = False)
    monkeypatch.delenv(dc.TRIM_ENV, raising = False)


def test_footprint_multiplier_follows_the_measured_torch_versions(monkeypatch):
    _no_footprint_env(monkeypatch)
    assert dc.footprint_multiplier("mps", "2.10.0") == 2.5
    assert dc.footprint_multiplier("mps", "2.13.1") == 2.5
    assert dc.footprint_multiplier("mps", "2.14.1") == 1.4
    assert dc.footprint_multiplier(torch.device("mps"), "2.15.0.dev20270101") == 1.4
    assert dc.footprint_multiplier("cpu", "2.10.0") == 1.0
    monkeypatch.setenv(dc.FOOTPRINT_ENV, "1.1")
    assert dc.footprint_multiplier("mps", "2.10.0") == 1.1


def test_auto_turns_off_when_the_priced_fill_does_not_fit(monkeypatch):
    # 1.5x the dense bytes free above the reserve: the old guard kept every layer, the priced one (2.5x) keeps none
    _no_footprint_env(monkeypatch)
    monkeypatch.delenv(dc.ENV, raising = False)
    monkeypatch.setattr(dc, "footprint_multiplier", lambda device, torch_version = None: 2.5)
    layers = [_layer("Q4_K", torch.bfloat16, seed = s) for s in range(3)]
    net = _Net(*layers)
    dense = dc.dense_bytes(layers)
    reserve = 10 * dense
    assert (
        dc.install(net, device = "mps", free = lambda: reserve + int(1.5 * dense), reserve = reserve) == 3
    )
    net(_inputs(torch.bfloat16, (1,))[0])
    assert dc.kept_bytes(net) == 0
    assert dc.STATUS["state"] == "off" and "priced footprint" in dc.STATUS["reason"]
    assert dc.STATUS["priced_bytes"] == int(dense * 2.5)


def test_auto_fills_when_the_priced_fill_fits(monkeypatch):
    _no_footprint_env(monkeypatch)
    monkeypatch.delenv(dc.ENV, raising = False)
    monkeypatch.setattr(dc, "footprint_multiplier", lambda device, torch_version = None: 2.5)
    monkeypatch.setattr(dc, "_release_pool", lambda device: None)
    layers = [_layer("Q4_K", torch.bfloat16, seed = s) for s in range(3)]
    net = _Net(*layers)
    dense = dc.dense_bytes(layers)
    assert dc.install(net, device = "mps", free = lambda: 3 * dense, reserve = 0) == 3
    net(_inputs(torch.bfloat16, (1,))[0])
    assert (
        dc.kept_bytes(net) == dense - 3 * 48 * 2 and dc.STATUS["state"] == "on"
    )  # weights; biases tiny


def test_each_layer_admit_prices_the_multiplier(monkeypatch):
    # forced on, constant free = reserve + 1.5 layers: the old guard (1x) admitted all three, at 2x none fit
    _no_footprint_env(monkeypatch)
    monkeypatch.setenv(dc.ENV, "1")
    monkeypatch.setenv(dc.FOOTPRINT_ENV, "2")
    layers = [_layer("Q4_K", torch.bfloat16, seed = s) for s in range(3)]
    net = _Net(*layers)
    one = 48 * 512 * 2
    dc.install(net, device = "cpu", free = lambda: one * 10 + int(1.5 * one), reserve = one * 10)
    net(_inputs(torch.bfloat16, (1,))[0])
    assert dc.kept_bytes(net) == 0 and dc.STATUS["state"] == "off"


def test_dense_bytes_counts_weights_and_biases_in_the_compute_dtype():
    layers = [_layer("Q8_0", torch.bfloat16), _layer("Q8_0", torch.float16, bias = False)]
    assert dc.dense_bytes(layers) == (48 * 512 + 48) * 2 + 48 * 512 * 2


def test_each_kept_layer_hands_the_pool_back(monkeypatch):
    _no_footprint_env(monkeypatch)
    monkeypatch.setenv(dc.ENV, "1")
    released = []
    monkeypatch.setattr(dc, "_release_pool", lambda device: released.append(device))
    net = _Net(*[_layer("Q4_K", torch.bfloat16, seed = s) for s in range(3)])
    dc.install(net, device = "cpu", free = lambda: _HUGE, reserve = 0)
    x = _inputs(torch.bfloat16, (1,))[0]
    net(x)
    net(x)
    assert len(released) == 3, "one release per kept layer, none on hits"


def test_release_pool_is_mps_only_and_has_a_switch(monkeypatch):
    _no_footprint_env(monkeypatch)
    calls = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append(1))
    dc._release_pool(torch.device("cpu"))
    assert calls == []
    dc._release_pool(torch.device("mps"))
    assert calls == [1]
    monkeypatch.setenv(dc.TRIM_ENV, "0")
    dc._release_pool(torch.device("mps"))
    assert calls == [1]


def test_mps_budget_honours_a_watermark_below_one():
    assert dc.mps_budget_bytes(100, "") == 100
    assert dc.mps_budget_bytes(100, "1.7") == 100
    assert dc.mps_budget_bytes(100, "0") == 100
    assert dc.mps_budget_bytes(100, "0.5") == 50


def test_plan_prices_available_memory_and_keeps_the_total_ram_reserve():
    # reserve from total RAM (small-host logic), headroom from available memory
    p = dc.plan_fill(10 * GIB, available = 40 * GIB, total = 64 * GIB, multiplier = 2.0)
    assert p["reserve_bytes"] == 16 * GIB and p["left_bytes"] == 20 * GIB and p["on"]
    p = dc.plan_fill(10 * GIB, available = 30 * GIB, total = 64 * GIB, multiplier = 2.0)
    assert not p["on"] and "under the 16.0 GiB reserve" in p["reason"]
    p = dc.plan_fill(10 * GIB, available = 30 * GIB, total = 16 * GIB, multiplier = 2.0)
    assert p["reserve_bytes"] == 8 * GIB and p["on"]
    assert not dc.plan_fill(1, available = None, total = 64 * GIB)["on"]
    assert dc.plan_fill(10 * GIB, available = 20 * GIB, total = 16 * GIB, multiplier = 2.0, cap = 4 * GIB)[
        "on"
    ]


# M5 Max task 0042: kept bytes and the base arm's peak footprint per model and torch version.
_MEASURED = {
    ("klein", "2.10.0"): (7.361 * GB, 21.4 * GB),
    ("klein", "2.14.1"): (7.361 * GB, 16.6 * GB),
    ("zimage", "2.10.0"): (12.304 * GB, 24.6 * GB),
    ("zimage", "2.14.1"): (12.304 * GB, 19.7 * GB),
}
# Emulated Mac: available at the fill = RAM - the base arm's peak - 4 GiB for macOS and other apps.
_EXPECTED_ON = {
    ("klein", "2.10.0"): {
        16: False,
        24: False,
        32: False,
        48: False,
        64: True,
        96: True,
        128: True,
    },
    ("klein", "2.14.1"): {16: False, 24: False, 32: False, 48: True, 64: True, 96: True, 128: True},
    ("zimage", "2.10.0"): {
        16: False,
        24: False,
        32: False,
        48: False,
        64: False,
        96: True,
        128: True,
    },
    ("zimage", "2.14.1"): {
        16: False,
        24: False,
        32: False,
        48: False,
        64: True,
        96: True,
        128: True,
    },
}


@pytest.mark.parametrize("key", list(_MEASURED), ids = lambda k: f"{k[0]}-{k[1]}")
def test_emulated_smaller_macs(monkeypatch, key):
    _no_footprint_env(monkeypatch)
    kept, base_peak = _MEASURED[key]
    mult = dc.footprint_multiplier("mps", key[1])
    for ram_gib, want in _EXPECTED_ON[key].items():
        total = ram_gib * GIB
        plan = dc.plan_fill(
            int(kept), available = int(total - base_peak - 4 * GIB), total = total, multiplier = mult
        )
        assert plan["on"] is want, (key, ram_gib, plan)
        if plan["on"]:  # the priced fill never eats into the reserve
            assert total - base_peak - kept * mult >= dc.reserve_bytes(total)


# ---- device gate, kill switch, offload, counters / status, trim, stale-entry accounting


def _snapshot(net):
    return [dict(m.__dict__) for m in net.modules()]


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", torch.device("cpu"), None])
def test_auto_leaves_non_mps_devices_untouched(monkeypatch, device):
    # CUDA / ROCm (reported as cuda) / CPU: no layer is patched and outputs are the stock ones
    _no_footprint_env(monkeypatch)
    monkeypatch.delenv(dc.ENV, raising = False)
    net = _Net(_layer("Q4_K", torch.bfloat16), _layer("Q8_0", torch.bfloat16, seed = 1))
    ref = _Net(_layer("Q4_K", torch.bfloat16), _layer("Q8_0", torch.bfloat16, seed = 1))
    before = _snapshot(net)
    counts = dict(dc.COUNTS)
    assert dc.install(net, device = device, free = lambda: _HUGE, reserve = 0) == 0
    assert _snapshot(net) == before and dc.COUNTS == counts
    assert dc.STATUS["state"] == "off" and "MPS only" in dc.STATUS["reason"]
    x = _inputs(torch.bfloat16, (1,))[0]
    assert all(torch.equal(a, b) for a, b in zip(net(x), ref(x)))
    assert dc.kept_bytes(net) == 0


@pytest.mark.parametrize("value", ["0", "off", "false", "no", " OFF "])
def test_kill_switch_wins_on_mps(monkeypatch, value):
    monkeypatch.setenv(dc.ENV, value)
    net = _Net(_layer("Q4_K", torch.bfloat16))
    before = _snapshot(net)
    assert dc.install(net, device = "mps", free = lambda: _HUGE, reserve = 0) == 0
    assert _snapshot(net) == before
    assert dc.STATUS == {"state": "off", "reason": "off"}


def test_offloaded_transformer_declines_in_auto(monkeypatch):
    monkeypatch.delenv(dc.ENV, raising = False)
    net = _Net(_layer("Q4_K", torch.bfloat16))
    assert dc.install(net, device = "mps", offloaded = True, free = lambda: _HUGE, reserve = 0) == 0
    assert "forward_native" not in net.layers[0].__dict__
    assert dc.STATUS["state"] == "off" and "offloaded" in dc.STATUS["reason"]
    monkeypatch.setenv(dc.ENV, "1")  # forcing on still installs
    assert dc.install(net, device = "mps", offloaded = True, free = lambda: _HUGE, reserve = 0) == 1


def test_load_passes_the_offload_plan():
    import inspect

    from core.inference import diffusion

    src = inspect.getsource(diffusion)
    at = src.index("gguf_dense_cache.install(")
    call = src[at : src.index(")\n", src.index("logger = logger", at))]
    assert "offloaded = plan.offload_policy != OFFLOAD_NONE" in call


def test_auto_counters_and_status_on_mps(monkeypatch):
    # N = 1 and N = 4: the first forward (the fill) is already bit-identical, later ones are hits
    _no_footprint_env(monkeypatch)
    monkeypatch.delenv(dc.ENV, raising = False)
    monkeypatch.setattr(dc, "footprint_multiplier", lambda device, torch_version = None: 2.5)
    layers = [_layer("Q4_K", torch.bfloat16, seed = s) for s in range(3)]
    refs = [_layer("Q4_K", torch.bfloat16, seed = s) for s in range(3)]
    net, ref = _Net(*layers), _Net(*refs)
    counts = dict(dc.COUNTS)
    assert dc.install(net, device = "mps", free = lambda: _HUGE, reserve = 0) == 3
    assert dc.STATUS["state"] == "pending"
    xs = _inputs(torch.bfloat16, (1, 2, 3, 4))
    first = net(xs[0])
    assert all(torch.equal(a, b) for a, b in zip(first, ref(xs[0])))
    for x in xs[1:]:
        assert all(torch.equal(a, b) for a, b in zip(net(x), ref(x)))
    delta = {k: dc.COUNTS[k] - counts.get(k, 0) for k in dc.COUNTS}
    assert delta["installs"] == 1 and delta["layers"] == 3 and delta["cached"] == 3
    assert (
        delta["hits"] == 3 * 3 and delta["declined"] == 0 and delta["bytes"] == dc.kept_bytes(net)
    )
    dense = dc.dense_bytes(layers)
    assert dc.STATUS["state"] == "on" and dc.STATUS["reason"] is None
    assert dc.STATUS["dense_bytes"] == dense and dc.STATUS["priced_bytes"] == int(dense * 2.5)


def test_status_reports_a_partial_fill(monkeypatch):
    _no_footprint_env(monkeypatch)
    monkeypatch.setenv(dc.ENV, "1")
    layers = [_layer("Q4_K", torch.bfloat16, seed = s) for s in range(3)]
    net = _Net(*layers)
    one = 48 * 512 * 2

    def free():
        return 2 * one - dc.kept_bytes(net)

    dc.install(net, device = "cpu", free = free, reserve = one)
    net(_inputs(torch.bfloat16, (1,))[0])
    assert dc.STATUS["state"] == "partial" and dc.STATUS["reason"].startswith(
        "fill stopped at the reserve after"
    )


def test_trim_counts_and_switch(monkeypatch):
    _no_footprint_env(monkeypatch)
    calls = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append(1))
    trims = dc.COUNTS["trims"]
    dc._release_pool(torch.device("mps"))
    assert calls == [1] and dc.COUNTS["trims"] == trims + 1
    for value in ("0", "off", "false", "no"):
        monkeypatch.setenv(dc.TRIM_ENV, value)
        assert not dc.trim_enabled()
        dc._release_pool(torch.device("mps"))
    assert calls == [1] and dc.COUNTS["trims"] == trims + 1


def test_a_dropped_stale_weight_frees_its_cap_bytes(monkeypatch):
    # cap = one layer: after a weight swap the stale weight is dropped, so the new one still fits under the cap
    _no_footprint_env(monkeypatch)
    monkeypatch.setenv(dc.ENV, "1")
    layer = _layer("Q4_K", torch.bfloat16, seed = 1)
    one = 48 * 512 * 2
    monkeypatch.setenv(dc.CAP_ENV, str(one / GIB))
    dc.install(layer, device = "cpu", free = lambda: _HUGE, reserve = 0)
    x = _inputs(torch.bfloat16, (9,))[0]
    layer(x)
    assert dc.kept_bytes(layer) == one
    other = _layer("Q4_K", torch.bfloat16, seed = 2)
    layer.weight, layer.bias = other.weight, other.bias
    assert torch.equal(layer(x), other(x))
    assert dc.kept_bytes(layer) == one, "the swapped-in weight is kept again"
