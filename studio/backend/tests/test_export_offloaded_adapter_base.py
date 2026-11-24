# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A 16-bit adapter base the GPU cannot hold spills to the meta device, where the merge cannot read it.
Sizes and memory are pinned: the field report is a 26B MoE (52 GB bf16) on a 24 GB card with 31 GB RAM."""

from __future__ import annotations

import contextlib
import json
import sys
import types
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_export_absolute_paths import (  # noqa: E402
    _install_export_backend_stubs,
    _load_module,
)

GB = 1024**3
BASE = "TheDrummer/Orion-26B-A4B-v1"


class _Peft:
    pass


class _Loader:
    calls: list[dict] = []
    device_map: dict = {}

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        model = _Peft()
        model.hf_device_map = cls.device_map
        return model, types.SimpleNamespace()


def _backend(
    monkeypatch,
    weight_bytes,
    budget,
    device_map = None,
):
    _install_export_backend_stubs(monkeypatch)
    mod = _load_module("test_core_export_offloaded_adapter", "core/export/export.py", monkeypatch)
    _Loader.calls = []
    _Loader.device_map = device_map or {"": 0}
    monkeypatch.setattr(mod, "_IS_MLX", False)
    monkeypatch.setattr(mod, "FastLanguageModel", _Loader)
    monkeypatch.setattr(mod, "PeftModel", _Peft, raising = False)
    monkeypatch.setattr(mod, "PeftModelForCausalLM", _Peft, raising = False)
    monkeypatch.setattr(mod, "get_base_model_from_lora", lambda *a, **k: BASE)
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: {})
    monkeypatch.setattr(
        mod, "_checkpoint_weight_bytes", lambda *a, **k: weight_bytes, raising = False
    )
    monkeypatch.setattr(mod, "_export_memory_budget", lambda: budget, raising = False)
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    return mod, backend


def _adapter(tmp_path):
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": BASE}))
    return str(checkpoint)


def test_a_16bit_base_too_big_for_vram_and_ram_is_refused_before_the_load(monkeypatch, tmp_path):
    mod, backend = _backend(monkeypatch, 52 * GB, (23 * GB, 28 * GB))
    ok, message = backend.load_checkpoint(_adapter(tmp_path), load_in_4bit = False)
    assert not ok
    assert _Loader.calls == []
    assert "52 GB" in message and "23 GB" in message and "28 GB" in message
    assert "device_map" not in message and "meta" not in message


def test_a_16bit_base_that_fits_with_host_ram_is_block_swapped(monkeypatch, tmp_path):
    mod, backend = _backend(monkeypatch, 40 * GB, (23 * GB, 28 * GB))
    ok, message = backend.load_checkpoint(_adapter(tmp_path), load_in_4bit = False)
    assert ok, message
    assert len(_Loader.calls) == 1
    assert _Loader.calls[0]["offload_layers"] == "auto"
    assert _Loader.calls[0]["device_map_planner_kwargs"]["reserve_bytes"] == 1 << 30


def test_a_16bit_base_that_fits_the_gpu_loads_as_before(monkeypatch, tmp_path):
    mod, backend = _backend(monkeypatch, 16 * GB, (23 * GB, 28 * GB))
    ok, message = backend.load_checkpoint(_adapter(tmp_path), load_in_4bit = False)
    assert ok, message
    assert "offload_layers" not in _Loader.calls[0]


def test_a_4bit_adapter_load_is_left_to_the_13720_retry(monkeypatch, tmp_path):
    """#13720's case: the 4-bit load is untouched, and its spill retry still owns the small-GPU path."""
    mod, backend = _backend(monkeypatch, 52 * GB, (6 * GB, 28 * GB))
    ok, message = backend.load_checkpoint(_adapter(tmp_path), load_in_4bit = True)
    assert ok, message
    assert "offload_layers" not in _Loader.calls[0]


def test_a_plain_checkpoint_is_not_sized_or_refused(monkeypatch, tmp_path):
    """No adapter: the GGUF export converts the checkpoint's own files, so the load is unchanged."""
    mod, backend = _backend(monkeypatch, 52 * GB, (23 * GB, 28 * GB))
    checkpoint = tmp_path / "Orion-26B-A4B-v1"
    checkpoint.mkdir()
    ok, message = backend.load_checkpoint(str(checkpoint), load_in_4bit = False)
    assert ok, message
    assert "offload_layers" not in _Loader.calls[0]


def test_unknown_sizes_load_as_before(monkeypatch, tmp_path):
    mod, backend = _backend(monkeypatch, None, None)
    ok, message = backend.load_checkpoint(_adapter(tmp_path), load_in_4bit = False)
    assert ok, message
    assert "offload_layers" not in _Loader.calls[0]


def test_a_single_gpu_spill_is_reported_after_the_load(monkeypatch, tmp_path):
    """Sizes unknown, the load spilled anyway: say so now, not after the merge fails on meta weights."""
    spilled = {"model.language_model.layers.0": 0, "model.language_model.layers.12": "cpu"}
    mod, backend = _backend(monkeypatch, None, None, device_map = spilled)
    ok, message = backend.load_checkpoint(_adapter(tmp_path), load_in_4bit = False)
    assert not ok
    assert "did not fit in GPU memory" in message
    assert getattr(backend, "current_model", None) is None


def test_weight_bytes_count_every_shard(monkeypatch, tmp_path):
    _install_export_backend_stubs(monkeypatch)
    mod = _load_module("test_core_export_offloaded_adapter", "core/export/export.py", monkeypatch)
    base = tmp_path / "base"
    base.mkdir()
    for i, size in enumerate((300, 200, 100), start = 1):
        (base / f"model-0000{i}-of-00003.safetensors").write_bytes(b"\0" * size)
    (base / "model.safetensors.index.json").write_text("{}")
    (base / "config.json").write_text("{}")
    assert mod._checkpoint_weight_bytes(str(base), None, False) == 600
    assert mod._checkpoint_weight_bytes("org/not-local", None, True) is None


def test_weight_bytes_fall_back_to_bin_shards(monkeypatch, tmp_path):
    _install_export_backend_stubs(monkeypatch)
    mod = _load_module("test_core_export_offloaded_adapter", "core/export/export.py", monkeypatch)
    base = tmp_path / "base"
    base.mkdir()
    for i, size in enumerate((300, 200), start = 1):
        (base / f"pytorch_model-0000{i}-of-00002.bin").write_bytes(b"\0" * size)
    (base / "pytorch_model.bin.index.json").write_text("{}")
    (base / "training_args.bin").write_bytes(b"\0" * 50)
    assert mod._checkpoint_weight_bytes(str(base), None, False) == 500
    # Safetensors win when a repo ships both.
    (base / "model.safetensors").write_bytes(b"\0" * 70)
    assert mod._checkpoint_weight_bytes(str(base), None, False) == 70
