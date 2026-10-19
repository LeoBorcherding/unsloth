# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a user reads when a data recipe run, Check or seed preview fails."""

from __future__ import annotations

import queue
import threading

import pytest
from fastapi import HTTPException

from core.data_recipe.jobs import worker
from core.data_recipe.jobs.constants import EVENT_JOB_ERROR
from core.data_recipe.jobs.manager import JobManager
from core.data_recipe.jobs.parse import apply_update, parse_log_message
from core.data_recipe.jobs.types import Job
from utils.paths import studio_root


def run_failing_job(monkeypatch, tmp_path, exc):
    def boom(recipe):
        raise exc

    monkeypatch.setattr(worker, "build_config_builder", boom)
    monkeypatch.setattr(worker, "_ARTIFACT_ROOT", tmp_path)
    monkeypatch.setenv("PYTHONWARNINGS", "default")
    events = queue.Queue()
    worker.run_job_process(event_queue = events, recipe = {}, run = {"execution_type": "preview"})
    while True:
        event = events.get_nowait()
        if event["type"] == EVENT_JOB_ERROR:
            return event


@pytest.fixture
def no_log_config(monkeypatch):
    from loggers.config import LogConfig

    monkeypatch.setattr(LogConfig, "setup_logging", lambda **kwargs: None)


def test_an_exception_with_no_message_names_its_type(monkeypatch, tmp_path, no_log_config):
    assert run_failing_job(monkeypatch, tmp_path, AssertionError())["error"] == "AssertionError"


def test_an_engine_wrapper_around_an_empty_cause_names_the_cause(
    monkeypatch, tmp_path, no_log_config
):
    try:
        try:
            raise ConnectionError()
        except ConnectionError as cause:
            raise RuntimeError("Error generating dataset:") from cause
    except RuntimeError as wrapped:
        exc = wrapped

    error = run_failing_job(monkeypatch, tmp_path, exc)["error"]

    assert error == "Error generating dataset: ConnectionError"


def manager_with(job):
    m = JobManager.__new__(JobManager)
    m._lock = threading.Lock()
    m._job = job
    m.emitted = []
    m._emit = m.emitted.append
    m._retire_workflow_key = lambda job: None
    return m


def drop_warning(index, column, kind):
    return {
        "type": "log",
        "message": (
            f"Generation for record at index {index} failed in column '{column}' ({kind}). "
            "Will omit this record from the dataset. Detail: Request timed out."
        ),
    }


def test_a_run_where_every_row_failed_says_which_column_and_why(no_log_config):
    job = Job(job_id = "job-1", status = "active")
    m = manager_with(job)
    for index in range(3):
        m._handle_event(job, drop_warning(index, "question", "timeout"))
    missing = studio_root() / "assets" / "datasets" / "recipe_x" / "parquet-files"
    m._handle_event(
        job,
        {
            "type": EVENT_JOB_ERROR,
            "error": (
                "Failed to load generated dataset - all records may have been dropped due to "
                f"generation failures. Check the warnings above for details. Original error: "
                f"[Errno 2] No such file or directory: '{missing}'"
            ),
            "stack": f'Traceback (most recent call last):\n  File "{missing}"',
        },
    )

    assert job.error.startswith("Every row failed: 3 in column 'question' (3 timeout).")
    assert "warnings above" not in job.error
    assert str(studio_root()) not in job.error
    assert m.emitted[-1]["error"] == job.error


def test_the_traceback_stays_on_the_server(no_log_config):
    job = Job(job_id = "job-1", status = "active")
    m = manager_with(job)
    m._handle_event(job, {"type": EVENT_JOB_ERROR, "error": "boom", "stack": "Traceback ..."})

    assert "stack" not in m.emitted[-1]
    assert job.error == "boom"


def test_the_drop_warning_is_counted_per_column_and_kind():
    job = Job(job_id = "job-1")
    for message in (
        drop_warning(0, "q", "timeout")["message"],
        drop_warning(1, "q", "generation error")["message"],
        drop_warning(2, "judge", "timeout")["message"],
    ):
        apply_update(job, parse_log_message(message))

    assert job.record_failures == {"q": {"timeout": 1, "generation error": 1}, "judge": {"timeout": 1}}


def test_a_missing_seed_table_reads_as_a_missing_dataset():
    from core.data_recipe.public_errors import public_error_text

    exc = RuntimeError(
        "Catalog Error: Table with name hf://unsloth/alpaca-cleaned does not exist!\n"
        'Did you mean "pg_namespace"?\n\nLINE 1: DESCRIBE SELECT * FROM \'hf://unsloth/alpaca-cleaned\''
    )
    text = public_error_text(exc)

    assert text.startswith("Seed dataset hf://unsloth/alpaca-cleaned wasn't found.")
    assert "pg_namespace" not in text and "SELECT" not in text


def test_a_missing_local_seed_table_does_not_show_the_server_path():
    from core.data_recipe.public_errors import public_error_text

    path = studio_root() / "assets" / "datasets" / "uploads" / "seed"
    text = public_error_text(RuntimeError(f"Catalog Error: Table with name {path} does not exist!"))

    assert text.startswith("Seed dataset ")
    assert str(studio_root()) not in text


def test_a_config_validation_error_drops_the_pydantic_dump():
    from pydantic import BaseModel

    from core.data_recipe.public_errors import public_error_text

    class Column(BaseModel):
        output_format: dict

    try:
        Column.model_validate({"output_format": "{not json"})
    except Exception as exc:
        text = public_error_text(exc)

    assert text == "Invalid recipe settings. output_format: Input should be a valid dictionary"


def test_server_paths_are_redacted():
    from core.data_recipe.public_errors import public_error_text

    path = studio_root() / "unsloth_studio" / "Lib" / "site-packages" / "x.py"
    assert str(studio_root()) not in public_error_text(FileNotFoundError(f"missing {path}"))


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            ValueError("Bad split: nonexistent. Available splits: ['train', 'test']"),
            "Bad split: nonexistent. Available splits: ['train', 'test']",
        ),
        (OSError("C:\\private\\cache\\thing is locked"), "seed inspect failed"),
    ],
)
def test_a_seed_preview_failure_keeps_a_safe_reason(exc, expected):
    from routes.data_recipe.seed import _seed_failure_detail

    assert _seed_failure_detail(exc) == expected


def test_a_gated_or_missing_hub_dataset_says_so():
    from datasets.exceptions import DatasetNotFoundError

    from routes.data_recipe.seed import _seed_failure_detail

    message = (
        "Dataset 'bigcode/the-stack-smol' is a gated dataset on the Hub. Visit the dataset page "
        "at https://huggingface.co/datasets/bigcode/the-stack-smol to ask for access."
    )
    assert _seed_failure_detail(DatasetNotFoundError(message)) == message


def test_a_latin1_csv_preview_says_it_is_not_utf8(tmp_path, monkeypatch):
    from routes.data_recipe import seed

    monkeypatch.setattr(seed, "account_path", lambda path: path)
    path = tmp_path / "names.csv"
    path.write_bytes("name,city\nJos\xe9,M\xfcnchen\n".encode("latin-1"))

    with pytest.raises(HTTPException) as caught:
        seed._read_preview_rows_from_local_file(path, 5)

    assert caught.value.status_code == 422
    assert "UTF-8" in caught.value.detail


def test_check_shows_the_plain_reason_not_the_sql(monkeypatch):
    from models.data_recipe import RecipePayload
    from routes.data_recipe import validate as route

    class CatalogException(Exception):
        pass

    def missing_seed(recipe, *_, **__):
        raise CatalogException(
            "Catalog Error: Table with name hf://unsloth/nope does not exist!\n"
            "LINE 1: DESCRIBE SELECT * FROM 'hf://unsloth/nope'"
        )

    monkeypatch.setattr(route, "validate_recipe", missing_seed)
    monkeypatch.setattr(route, "_collect_validation_errors", lambda *a, **k: [])
    result = route.validate(RecipePayload(recipe = {"columns": [{"name": "a"}]}))

    assert not result.valid
    assert result.errors[0].message.startswith("Seed dataset hf://unsloth/nope wasn't found.")
    assert "SELECT" not in (result.raw_detail or "")
