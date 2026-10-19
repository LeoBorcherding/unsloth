# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os

import pytest
import torch

autotune_cache = pytest.importorskip("unsloth.kernels.moe.autotune_cache")
from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    monkeypatch.setattr(
        autotune_cache, "_get_cache_file_path", lambda key: str(tmp_path / f"{key}.json")
    )
    monkeypatch.delenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", raising = False)
    autotune_cache.clear_cache()
    yield tmp_path
    autotune_cache.clear_cache()


def _configs():
    return (
        KernelConfigForward(BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 128, BLOCK_SIZE_K = 64),
        KernelConfigBackward_dX(BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 128),
        KernelConfigBackward_dW(BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 64),
    )


def test_saved_config_loads_back_and_keeps_the_file(cache_dir):
    autotune_cache.save_cached_config("k", *_configs())
    assert autotune_cache.load_cached_config("k") is not None
    assert os.path.exists(cache_dir / "k.json")


def test_a_different_device_still_invalidates(cache_dir, monkeypatch):
    autotune_cache.save_cached_config("k", *_configs())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    assert autotune_cache.load_cached_config("k") is None
    assert not os.path.exists(cache_dir / "k.json")


def test_second_process_reuses_the_disk_cache(cache_dir, monkeypatch):
    calls = []

    def fake_tune(*args):
        calls.append(args)
        return _configs()

    monkeypatch.setattr(autotune_cache, "_run_moe_autotuning", fake_tune)
    kw = dict(num_experts = 8, hidden_dim = 2048, intermediate_dim = 768, top_k = 2, dtype = torch.bfloat16)
    first = autotune_cache.get_or_autotune_moe_kernels(**kw)
    autotune_cache.clear_cache()  # what a fresh process starts with
    second = autotune_cache.get_or_autotune_moe_kernels(**kw)
    assert len(calls) == 1
    assert second == first
