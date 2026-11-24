# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A non-PEFT GGUF export of a Hub model the GPU could not hold converts its cached snapshot: the
offloaded parameters sit on the meta device, where save_pretrained cannot read them."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from unsloth import save as unsloth_save  # noqa: E402

REPO = "TheDrummer/Orion-26B-A4B-v1"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
SHARDS = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]


def _offloaded_model(meta = True):
    """MoE / multimodal names: layer 12 onward went to the meta device, as in the field report."""
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    layers = torch.nn.ModuleList()
    for index in range(13):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        device = "meta" if meta and index >= 12 else "cpu"
        layer.self_attn.q_proj = torch.nn.Linear(4, 4, bias = False, device = device)
        layers.append(layer)
    model.model.language_model.layers = layers
    model.config = types.SimpleNamespace(_name_or_path = REPO, _commit_hash = COMMIT)
    return model


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    import huggingface_hub.constants as constants

    cache = tmp_path / "hub"
    repo = cache / ("models--" + REPO.replace("/", "--"))
    snapshot = repo / "snapshots" / COMMIT
    snapshot.mkdir(parents = True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(COMMIT)
    (snapshot / "config.json").write_text("{}")
    weight_map = {
        f"model.language_model.layers.{i}.self_attn.q_proj.weight": SHARDS[i % 3] for i in range(13)
    }
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for shard in SHARDS:
        (snapshot / shard).write_bytes(b"\0" * 8)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return snapshot


def test_offloaded_hub_model_converts_its_cached_snapshot(hub_cache):
    model = _offloaded_model()
    assert unsloth_save._gguf_reuses_loaded_checkpoint(model)
    # The preflight must not charge the 16-bit save it no longer writes.
    assert not unsloth_save._gguf_writes_16bit_checkpoint(model)
    assert unsloth_save._gguf_reused_checkpoint_dir(model) == str(hub_cache)


def test_a_missing_shard_falls_back_to_saving(hub_cache):
    (hub_cache / SHARDS[1]).unlink()
    assert not unsloth_save._gguf_reuses_loaded_checkpoint(_offloaded_model())


def test_a_resident_hub_model_still_saves_as_before(hub_cache):
    assert not unsloth_save._gguf_reuses_loaded_checkpoint(_offloaded_model(meta = False))


def test_an_uncached_offloaded_model_still_saves_as_before(tmp_path, monkeypatch):
    import huggingface_hub.constants as constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert not unsloth_save._gguf_reuses_loaded_checkpoint(_offloaded_model())


def _bin_snapshot(hub_cache, sharded):
    for name in [*SHARDS, "model.safetensors.index.json"]:
        (hub_cache / name).unlink()
    if not sharded:
        (hub_cache / "pytorch_model.bin").write_bytes(b"\0" * 8)
        return []
    shards = [f"pytorch_model-0000{i}-of-00002.bin" for i in (1, 2)]
    weight_map = {
        f"model.language_model.layers.{i}.self_attn.q_proj.weight": shards[i % 2] for i in range(13)
    }
    (hub_cache / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for shard in shards:
        (hub_cache / shard).write_bytes(b"\0" * 8)
    return shards


def test_an_offloaded_hub_model_with_a_single_bin_converts_its_snapshot(hub_cache):
    _bin_snapshot(hub_cache, sharded = False)
    assert unsloth_save._gguf_reused_checkpoint_dir(_offloaded_model()) == str(hub_cache)


def test_an_offloaded_hub_model_with_sharded_bins_converts_its_snapshot(hub_cache):
    shards = _bin_snapshot(hub_cache, sharded = True)
    assert unsloth_save._gguf_reused_checkpoint_dir(_offloaded_model()) == str(hub_cache)
    (hub_cache / shards[0]).unlink()
    assert unsloth_save._gguf_reused_checkpoint_dir(_offloaded_model()) is None


def test_an_offloaded_model_loaded_with_a_custom_cache_dir_finds_its_snapshot(
    hub_cache, tmp_path, monkeypatch
):
    """from_pretrained(cache_dir = ...): the default cache is empty, the snapshot lives in the caller's."""
    import huggingface_hub.constants as constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "default-empty"))
    model = _offloaded_model()
    assert unsloth_save._gguf_reused_checkpoint_dir(model) is None
    model._unsloth_hub_cache_dir = str(hub_cache.parents[2])
    assert unsloth_save._gguf_reused_checkpoint_dir(model) == str(hub_cache)


def test_the_loader_records_only_an_explicit_cache_dir(tmp_path):
    from unsloth.models.loader import _record_hub_cache_dir

    model = types.SimpleNamespace()
    _record_hub_cache_dir(model, None)
    assert not hasattr(model, "_unsloth_hub_cache_dir")
    _record_hub_cache_dir(model, tmp_path)
    assert model._unsloth_hub_cache_dir == str(tmp_path)
