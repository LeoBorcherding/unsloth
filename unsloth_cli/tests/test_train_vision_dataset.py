# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import typer
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unsloth_cli.commands.train as train_module  # noqa: E402

# conftest stubs the peek for every other train test; these ones run the real thing on local files.
_REAL_PEEK = getattr(train_module, "_detect_dataset_modalities", None)


class _VisionModelTrainer:
    """A vision model: is_vlm follows the dataset flag the way trainer.load_model computes it."""

    training_thread = None

    def __init__(self):
        self.load_kwargs = None
        self.prepare_kwargs = None
        self.is_vlm = False

    def load_model(self, **kwargs):
        self.load_kwargs = kwargs
        self.is_vlm = bool(kwargs.get("is_dataset_image", False))
        return True

    def prepare_model_for_training(self, **kwargs):
        self.prepare_kwargs = kwargs
        return True

    def load_and_format_dataset(self, **kwargs):
        return [], None

    def start_training(self, **kwargs):
        return True

    def get_training_progress(self):
        return SimpleNamespace(error = None, is_completed = True)


def _rows(tmp_path, rows):
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding = "utf-8")
    return str(path)


def _train(monkeypatch, tmp_path, dataset, *args):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    if _REAL_PEEK is not None:
        monkeypatch.setattr(train_module, "_detect_dataset_modalities", _REAL_PEEK)
    trainer = _VisionModelTrainer()
    monkeypatch.setattr(train_module, "_create_cli_trainer", lambda *a: trainer)
    app = typer.Typer()
    app.command()(train_module.train)
    result = CliRunner().invoke(
        app,
        [
            "--model", "unsloth/Qwen2.5-VL-3B-Instruct",
            "--local-dataset", dataset,
            "--output-dir", "run",
            *args,
        ],
    )
    return result, trainer


_IMAGE_ROWS = [{"image": "page1.png", "text": "x^2"}, {"image": "page2.png", "text": "y"}]


def test_image_dataset_takes_the_vision_path(tmp_path, monkeypatch):
    result, trainer = _train(
        monkeypatch, tmp_path, _rows(tmp_path, _IMAGE_ROWS), "--vision-all-linear"
    )

    assert result.exit_code == 0, result.output
    assert trainer.load_kwargs.get("is_dataset_image") is True
    assert trainer.load_kwargs.get("is_dataset_audio") is False
    assert trainer.prepare_kwargs["target_modules"] == "all-linear"


def test_text_dataset_stays_on_the_text_path(tmp_path, monkeypatch):
    rows = [{"instruction": "hi", "output": "hello"}]
    result, trainer = _train(monkeypatch, tmp_path, _rows(tmp_path, rows), "--vision-all-linear")

    assert result.exit_code == 0, result.output
    assert trainer.load_kwargs.get("is_dataset_image") is False
    assert trainer.prepare_kwargs["target_modules"] != "all-linear"


def test_audio_column_is_passed_on(tmp_path, monkeypatch):
    rows = [{"audio": "clip.wav", "text": "hello"}]
    result, trainer = _train(monkeypatch, tmp_path, _rows(tmp_path, rows))

    assert result.exit_code == 0, result.output
    assert trainer.load_kwargs.get("is_dataset_audio") is True


def test_flag_overrides_detection(tmp_path, monkeypatch):
    result, trainer = _train(
        monkeypatch, tmp_path, _rows(tmp_path, _IMAGE_ROWS), "--no-image-dataset"
    )

    assert result.exit_code == 0, result.output
    assert trainer.load_kwargs.get("is_dataset_image") is False


def test_unreadable_dataset_warns_and_trains_as_text(tmp_path, monkeypatch):
    result, trainer = _train(monkeypatch, tmp_path, str(tmp_path / "missing.jsonl"))

    assert result.exit_code == 0, result.output
    assert "--image-dataset" in result.output
    assert trainer.load_kwargs.get("is_dataset_image") is False


def test_image_flag_holds_when_detection_fails(tmp_path, monkeypatch):
    result, trainer = _train(
        monkeypatch, tmp_path, str(tmp_path / "missing.jsonl"), "--image-dataset"
    )

    assert result.exit_code == 0, result.output
    assert trainer.load_kwargs.get("is_dataset_image") is True
