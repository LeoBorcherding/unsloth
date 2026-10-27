# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Quality pass on a finished run, and the per-run seed that makes sampler columns repeatable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from core.data_recipe.jobs.worker import _apply_run_seed
from core.data_recipe.quality import compute_quality
from routes.data_recipe import jobs

ANALYSIS = {
    "column_statistics": [
        {"column_name": "question", "column_type": "llm-text"},
        {"column_name": "answer", "column_type": "llm-structured"},
        {"column_name": "answer_valid", "column_type": "validation"},
        {"column_name": "topic", "column_type": "sampler"},
    ],
    "side_effect_column_names": [],
}


def _row(question: str, answer, valid: bool = True, topic: str = "math") -> dict:
    return {
        "question": question,
        "answer": answer,
        "answer_valid": {"is_valid": valid, "error_message": ""},
        "topic": topic,
    }


def test_clean_run_passes():
    rows = [
        _row("What is two plus two?", {"final": "4"}),
        _row("Name the capital of France.", {"final": "Paris"}),
        _row("How many legs does a spider have?", {"final": "8"}),
    ]
    quality = compute_quality(rows, ANALYSIS)
    assert quality["verdict"] == "ok"
    assert quality["warnings"] == []
    assert quality["content_columns"] == ["question", "answer"]
    assert quality["exact_duplicates"] == 0
    assert quality["near_duplicates"] == 0
    assert quality["validators"] == [
        {"name": "answer_valid", "checked": 3, "passed": 3, "pass_rate": 1.0}
    ]


def test_exact_and_near_duplicates_are_counted_on_generated_columns():
    rows = [
        _row("What is two plus two?", {"final": "4"}, topic = "a"),
        # Same generated text, different sampler value: still an exact duplicate for training.
        _row("What is two plus two?", {"final": "4"}, topic = "b"),
        # Case and punctuation only.
        _row("what is TWO plus two", {"final": "4"}),
        _row(
            "Explain why the sky looks blue during a clear afternoon in summer please",
            {"final": "scattering"},
        ),
        _row(
            "Explain why the sky looks blue during a clear afternoon in summer please now",
            {"final": "scattering"},
        ),
        _row("Something entirely different about cooking rice.", {"final": "boil"}),
    ]
    quality = compute_quality(rows, ANALYSIS)
    assert quality["exact_duplicates"] == 1
    assert quality["near_duplicates"] == 2
    assert quality["near_duplicates_checked"] is True
    assert quality["verdict"] == "warn"
    assert quality["warnings"][0] == "3 of 6 rows are duplicates or near-duplicates (50%)."


def test_format_empty_and_validator_failures_warn():
    rows = [
        _row("Q one about the moon", "```json\n{not json"),
        _row("", {"final": "x"}, valid = False),
        _row("Q three<end_of_turn>", json.dumps({"final": "y"})),
        _row("Q four about rivers", {"final": "z"}, valid = False),
    ]
    quality = compute_quality(rows, ANALYSIS)
    by_name = {c["name"]: c for c in quality["columns"]}
    assert by_name["answer"]["format_failures"] == 1
    assert by_name["answer"]["format_check"] == "json"
    assert by_name["question"]["format_failures"] == 1
    assert by_name["question"]["empty"] == 1
    assert by_name["question"]["min_chars"] == len("Q one about the moon") - 1
    assert quality["validators"][0]["pass_rate"] == 0.5
    assert quality["verdict"] == "warn"
    joined = " ".join(quality["warnings"])
    assert "1 generated values are empty" in joined
    assert "2 values fail the format check" in joined
    assert "Validator answer_valid passes only 50% of rows." in joined


def test_types_are_inferred_without_analysis():
    rows = [
        {"text": "alpha beta gamma", "check": json.dumps({"is_valid": True})},
        {"text": "delta epsilon zeta", "check": json.dumps({"is_valid": False})},
    ]
    quality = compute_quality(rows, None, total_rows = 10)
    assert quality["content_columns"] == ["text"]
    assert quality["validators"][0]["passed"] == 1
    assert quality["truncated"] is True
    assert quality["rows_total"] == 10


def test_empty_run_warns():
    assert compute_quality([], ANALYSIS)["warnings"] == ["The run produced no rows."]


class _FakeManager:
    def __init__(self, status: str, rows: list[dict]):
        self._status = status
        self._rows = rows
        self.pages: list[tuple[int, int]] = []

    def get_status(self, job_id):
        return {"status": self._status} if job_id == "job" else None

    def get_dataset(self, job_id, *, limit, offset = 0):
        self.pages.append((offset, limit))
        return {"dataset": self._rows[offset : offset + limit], "total": len(self._rows)}

    def get_analysis(self, job_id):
        return ANALYSIS


def test_quality_route_reads_every_page(monkeypatch):
    rows = [_row(f"question number {i} about item {i}", {"final": str(i)}) for i in range(25)]
    manager = _FakeManager("completed", rows)
    monkeypatch.setattr(jobs, "get_job_manager", lambda: manager)
    monkeypatch.setattr(jobs, "_IN_MEMORY_DOWNLOAD_PAGE_SIZE", 10)
    quality = jobs.job_quality("job")
    assert quality["rows_checked"] == 25
    assert quality["verdict"] == "ok"
    assert manager.pages == [(0, 10), (10, 10), (20, 10)]


@pytest.mark.parametrize("status, code", [("active", 409), ("missing", 404)])
def test_quality_route_needs_a_finished_run(monkeypatch, status, code):
    manager = _FakeManager(status, [])
    monkeypatch.setattr(jobs, "get_job_manager", lambda: manager)
    with pytest.raises(HTTPException) as exc:
        jobs.job_quality("job" if status != "missing" else "other")
    assert exc.value.status_code == code


@pytest.mark.parametrize("seed", [None, 0, 7, 2**32 - 1])
def test_run_seed_accepts_unsigned_32_bit(seed):
    assert jobs._normalize_run_seed(seed) == seed


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, "7", True])
def test_run_seed_rejects_other_values(seed):
    with pytest.raises(HTTPException):
        jobs._normalize_run_seed(seed)


def _sampler_preview(tmp_path: Path, seed: int | None) -> list:
    pytest.importorskip("data_designer")
    from core.data_recipe.service import build_config_builder, create_data_designer

    recipe = {
        "model_providers": [],
        "model_configs": [],
        "columns": [
            {
                "column_type": "sampler",
                "name": "pick",
                "sampler_type": "category",
                "params": {"values": [f"v{i}" for i in range(50)]},
            },
            {
                "column_type": "sampler",
                "name": "n",
                "sampler_type": "uniform",
                "params": {"low": 0, "high": 1000},
            },
        ],
    }
    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    _apply_run_seed({"seed": seed})
    dataset = designer.preview(build_config_builder(recipe), num_records = 20).dataset
    return dataset[["pick", "n"]].values.tolist()


def test_same_seed_repeats_sampler_columns(tmp_path: Path):
    first = _sampler_preview(tmp_path / "a", 1234)
    second = _sampler_preview(tmp_path / "b", 1234)
    other = _sampler_preview(tmp_path / "c", 99)
    assert first == second
    assert first != other
