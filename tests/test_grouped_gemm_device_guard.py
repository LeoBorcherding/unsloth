# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Triton launches on the current device, so each grouped GEMM launch must sit under the operands'
device, and NUM_SMS must come from that device. A MoE layer on cuda:1 under device_map="auto"
otherwise launches on cuda:0. The kernels are stubbed: what is checked is where they are launched."""

import contextlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

pytest.importorskip("triton", reason = "the grouped GEMM is a Triton kernel")

try:
    from unsloth.kernels.moe.grouped_gemm import interface
except Exception as exc:  # pragma: no cover - depends on the installed stack
    pytest.skip(f"grouped_gemm is unimportable here: {exc}", allow_module_level = True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "the impls allocate their output on a real device"
)

E, T, N, K = 2, 4, 32, 32


@pytest.fixture
def seen(monkeypatch):
    log = {"guard": [], "launched_under": [], "props": []}

    @contextlib.contextmanager
    def device(dev):
        log["guard"].append(dev)
        yield
        log["guard"].pop()

    class _Kernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                log["launched_under"].append(log["guard"][-1] if log["guard"] else None)
                return types.SimpleNamespace()

            return launch

    monkeypatch.setattr(interface.torch.cuda, "device", device)
    monkeypatch.setattr(
        interface.torch.cuda,
        "get_device_properties",
        lambda dev: log["props"].append(dev) or types.SimpleNamespace(multi_processor_count = 4),
    )
    for name in ("_grouped_gemm_forward_kernel", "_grouped_gemm_dX_kernel", "_grouped_gemm_dW_kernel"):
        monkeypatch.setattr(interface, name, _Kernel())
    monkeypatch.setattr(interface, "log_kernel_info", lambda *a, **k: None)
    return log


def _operands():
    dev = torch.device("cuda", torch.cuda.current_device())
    X = torch.zeros(E * T, K, device = dev, dtype = torch.float16)
    W = torch.zeros(E, N, K, device = dev, dtype = torch.float16)
    dY = torch.zeros(E * T, N, device = dev, dtype = torch.float16)
    m_sizes = torch.full((E,), T, device = dev, dtype = torch.int32)
    return dev, X, W, dY, m_sizes


def test_forward_launches_under_the_operands_device(seen):
    dev, X, W, _, m_sizes = _operands()
    interface._grouped_gemm_forward_impl(X, W, 1, m_sizes, autotune = False)
    assert seen["launched_under"] == [dev]
    assert seen["props"] == [dev]


def test_dX_launches_under_the_gradients_device(seen):
    dev, _, W, dY, m_sizes = _operands()
    interface._grouped_gemm_dX_impl(dY, W, None, m_sizes, 1, autotune = False)
    assert seen["launched_under"] == [dev]
    assert seen["props"] == [dev]


def test_dW_launches_under_the_operands_device(seen):
    dev, X, _, dY, m_sizes = _operands()
    interface._grouped_gemm_dW_impl(X, dY, m_sizes, None, 1, autotune = False)
    assert seen["launched_under"] == [dev]
    assert seen["props"] == [dev]
