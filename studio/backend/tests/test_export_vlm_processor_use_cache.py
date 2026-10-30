# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A text-only LoRA on a VLM exports with the base's processor and without training's use_cache=False."""

from __future__ import annotations

import contextlib
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


def _export_mod(monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    return _load_module("test_core_export_vlm_processor", "core/export/export.py", monkeypatch)


class _Tokenizer:
    chat_template = "{{ trained }}"


class _Processor:
    def __init__(self):
        self.tokenizer = object()
        self.chat_template = "{{ base }}"


def _stub_auto_processor(monkeypatch, load):
    calls = []

    class AutoProcessor:
        @staticmethod
        def from_pretrained(name, **kw):
            calls.append((name, kw))
            return load()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoProcessor = AutoProcessor))
    return calls


def test_bare_tokenizer_is_wrapped_in_the_base_processor(monkeypatch):
    mod = _export_mod(monkeypatch)
    calls = _stub_auto_processor(monkeypatch, _Processor)
    tok = _Tokenizer()
    out = mod._with_base_processor(tok, "org/vlm", trust_remote_code = True, token = None)
    assert isinstance(out, _Processor)
    assert out.tokenizer is tok
    assert out.chat_template == "{{ trained }}"
    assert calls == [("org/vlm", {"trust_remote_code": True, "token": None})]


def test_a_loaded_processor_is_kept(monkeypatch):
    mod = _export_mod(monkeypatch)
    calls = _stub_auto_processor(monkeypatch, _Processor)
    proc = _Processor()
    assert mod._with_base_processor(proc, "org/vlm") is proc
    assert calls == []


def test_base_without_processor_keeps_the_tokenizer(monkeypatch):
    mod = _export_mod(monkeypatch)
    _stub_auto_processor(monkeypatch, _Tokenizer)
    tok = _Tokenizer()
    assert mod._with_base_processor(tok, "org/text") is tok


def test_processor_load_failure_keeps_the_tokenizer(monkeypatch):
    mod = _export_mod(monkeypatch)

    def _boom():
        raise OSError("no processor_config.json")

    _stub_auto_processor(monkeypatch, _boom)
    tok = _Tokenizer()
    assert mod._with_base_processor(tok, "org/vlm") is tok


def test_export_load_restores_use_cache(monkeypatch):
    mod = _export_mod(monkeypatch)
    restored = []
    zoo = types.ModuleType("unsloth_zoo")
    training_utils = types.ModuleType("unsloth_zoo.training_utils")
    training_utils.restore_use_cache = restored.append
    zoo.training_utils = training_utils
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.training_utils", training_utils)
    model = object()
    mod._restore_use_cache(model)
    assert restored == [model]


def test_restore_use_cache_tolerates_an_old_zoo(monkeypatch):
    mod = _export_mod(monkeypatch)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.training_utils", types.ModuleType("x"))
    mod._restore_use_cache(object())


class _VisionLoader:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        return types.SimpleNamespace(), _Tokenizer()


def _load_audio_vlm(monkeypatch, tmp_path, is_vlm):
    mod = _export_mod(monkeypatch)
    _VisionLoader.calls = []
    text_calls = []
    monkeypatch.setattr(mod, "FastVisionModel", _VisionLoader)
    monkeypatch.setattr(mod, "FastLanguageModel", types.SimpleNamespace(
        from_pretrained = lambda **kw: text_calls.append(kw) or (types.SimpleNamespace(), _Tokenizer())))
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: "audio_vlm")
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: is_vlm)
    monkeypatch.setattr(mod, "get_base_model_from_lora", lambda _path: "org/omni")
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: {})
    monkeypatch.setattr(mod, "_IS_MLX", False)
    monkeypatch.setattr(mod, "_restore_use_cache", lambda model: None)
    monkeypatch.setattr(mod, "PeftModel", _VisionLoader, raising = False)
    monkeypatch.setattr(mod, "PeftModelForCausalLM", _VisionLoader, raising = False)
    _stub_auto_processor(monkeypatch, _Processor)
    checkpoint = tmp_path / "checkpoint-30"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text('{"base_model_name_or_path": "org/omni"}')
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    ok, message = backend.load_checkpoint(str(checkpoint))
    assert ok, message
    return backend, text_calls, message


def test_audio_vlm_adapter_on_a_vlm_exports_with_the_base_processor(monkeypatch, tmp_path):
    """LFM3-Omni detects as audio_vlm; the text load dropped its processor, so no AutoProcessor or mmproj."""
    backend, text_calls, message = _load_audio_vlm(monkeypatch, tmp_path, is_vlm = True)
    assert len(_VisionLoader.calls) == 1 and text_calls == []
    assert backend.is_vision
    assert isinstance(backend.current_tokenizer, _Processor)
    assert "Audio (audio_vlm)" in message


def test_audio_vlm_without_vision_keeps_the_text_load(monkeypatch, tmp_path):
    backend, text_calls, _ = _load_audio_vlm(monkeypatch, tmp_path, is_vlm = False)
    assert _VisionLoader.calls == [] and len(text_calls) == 1
    assert not backend.is_vision
