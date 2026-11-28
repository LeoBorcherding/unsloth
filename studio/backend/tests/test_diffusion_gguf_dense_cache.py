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
