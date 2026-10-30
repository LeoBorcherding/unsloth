# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_TP_SIZE process group backend: NCCL only where torch was built with it."""

import importlib.util
import os

import pytest
import torch
import torch.distributed as dist

_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models", "tp_env.py")


def _tp_env():
    spec = importlib.util.spec_from_file_location("_tp_env_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Stop(Exception):
    pass


def _mesh(monkeypatch, cuda, nccl, windows):
    seen = []

    def init_process_group(backend, *a, **k):
        seen.append(backend)
        raise _Stop

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: nccl)
    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    module = _tp_env()
    monkeypatch.setattr(module, "_WINDOWS", windows)
    with pytest.raises(_Stop):
        module.TPEnvLayout(tp = 2, dp = 1, world = 2, fsdp2 = False).mesh()
    return seen


@pytest.mark.parametrize(
    "cuda, nccl, windows, expected",
    [
        (True, True, False, "nccl"),
        (True, False, False, "gloo"),  # Linux torch without NCCL: gloo has GPU collectives there
        (False, False, False, "gloo"),
        (False, False, True, "gloo"),
    ],
)
def test_mesh_picks_a_backend_torch_has(monkeypatch, cuda, nccl, windows, expected):
    assert _mesh(monkeypatch, cuda, nccl, windows) == [expected]


def test_windows_gpu_without_nccl_refuses_before_init(monkeypatch):
    # Windows gloo has no GPU all_reduce and crashes natively on the first one (pytorch#186535).
    with pytest.raises(RuntimeError, match = "needs NCCL"):
        _mesh(monkeypatch, cuda = True, nccl = False, windows = True)
