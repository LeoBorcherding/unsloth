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


@pytest.mark.parametrize(
    "cuda, nccl, expected",
    [
        (True, True, "nccl"),
        (True, False, "gloo"),  # Windows CUDA / ROCm torch
        (False, False, "gloo"),
    ],
)
def test_mesh_picks_a_backend_torch_has(monkeypatch, cuda, nccl, expected):
    seen = []

    def init_process_group(backend, *a, **k):
        seen.append(backend)
        raise _Stop

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: nccl)
    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    with pytest.raises(_Stop):
        _tp_env().TPEnvLayout(tp = 2, dp = 1, world = 2, fsdp2 = False).mesh()
    assert seen == [expected]
