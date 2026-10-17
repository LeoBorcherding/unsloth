# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import importlib  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402


_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - stub unusable imports
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.training import trainer as tmod  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


@pytest.fixture
def trainer(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    captured = {}

    class _FakeSFTConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeSFTTrainer:
        def __init__(self, **kwargs):
            pass

        def add_callback(self, cb):
            pass

        def train(self, **kwargs):
            pass

    monkeypatch.setattr(tmod, "SFTConfig", _FakeSFTConfig, raising = False)
    monkeypatch.setattr(tmod, "SFTTrainer", _FakeSFTTrainer, raising = False)
    monkeypatch.setattr(tmod, "_drop_hf_stdout_callbacks", lambda trainer: None, raising = True)
    monkeypatch.setattr(
        tmod.UnslothTrainer, "_finalize_training", lambda self, *a, **k: None, raising = True
    )
    monkeypatch.setattr(
        tmod.UnslothTrainer, "_preflight_first_batch", lambda self: None, raising = True
    )
    t = tmod.UnslothTrainer()
    t._audio_type = "bicodec"
    t.model = object()
    t.tokenizer = object()
    t.model_name = "unsloth/spark-tts"
    t.captured = captured
    return t


def _run(trainer, output_dir, **extra):
    with contextlib.suppress(AttributeError, TypeError, ValueError, KeyError):
        trainer._train_worker(
            {"dataset": [{"text": "a"}, {"text": "b"}], "final_format": "audio_bicodec"},
            batch_size = 2,
            gradient_accumulation_steps = 1,
            max_steps = 1,
            warmup_steps = 0,
            output_dir = output_dir,
            **extra,
        )


def test_cli_output_dir_outside_the_studio_home_is_used(trainer, tmp_path):
    target = tmp_path / "elsewhere" / "my_run"

    _run(trainer, str(target), allow_external_output_dir = True)

    assert trainer.captured.get("output_dir") == str(target)
    assert target.is_dir()


def test_output_dir_outside_the_studio_home_still_escapes_without_the_opt_in(trainer, tmp_path):
    target = tmp_path / "elsewhere" / "my_run"

    _run(trainer, str(target))

    assert "output_dir" not in trainer.captured
    assert "escapes root" in (trainer.get_training_progress().error or "")
    assert not target.exists()
