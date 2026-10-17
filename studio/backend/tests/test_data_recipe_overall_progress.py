# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Overall progress for batched and multi-column data recipe runs, fed data-designer's log lines."""

from __future__ import annotations

import queue
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.data_recipe.jobs import manager as manager_module  # noqa: E402
from core.data_recipe.jobs import parse  # noqa: E402
from core.data_recipe.jobs.parse import apply_update, parse_log_message  # noqa: E402
from core.data_recipe.jobs.types import Job  # noqa: E402

_SAMPLING_LINES = {
    "samplers": "🎲 Preparing samplers to generate {n} records across 1 columns",
    "seed": "🌱 Sampling {n} records from seed dataset",
}


def _feed(job: Job, line: str) -> None:
    update = parse_log_message(line)
    assert update is not None, line
    apply_update(job, update)


def _column_start(column: str) -> str:
    return f"⚡️ Processing llm-text column '{column}' with 4 concurrent workers"


def _progress(column: str, done: int, total: int, ok: int, failed: int, eta: float = 1.0) -> str:
    pct = round(done / total * 100)
    return (
        f"  🕑 llm-text column '{column}' progress: {done}/{total} ({pct}%) complete, "
        f"{ok} ok, {failed} failed, 0.50 rec/s, eta {eta}s"
    )


def _job(rows: int, llm_columns: int = 1) -> Job:
    # What JobManager.start sets from the request.
    job = Job(job_id = "job-progress", rows = rows)
    job.progress_columns_total = llm_columns
    return job


@pytest.mark.parametrize("phrasing", sorted(_SAMPLING_LINES))
def test_batched_run_reports_progress_against_the_whole_run(phrasing):
    job = _job(30)
    seen = []
    for batch in (1, 2, 3):
        _feed(job, f"⏳ Processing batch {batch} of 3")
        _feed(job, _SAMPLING_LINES[phrasing].format(n = 10))
        _feed(job, _column_start("generated_instruction"))
        for done in (2, 5, 10):
            _feed(job, _progress("generated_instruction", done, 10, done, 0))
            seen.append((batch, job.progress.done, job.progress.total, job.progress.percent))

    assert job.rows == 30
    assert seen[2] == (1, 10, 30, pytest.approx(100 / 3))
    assert seen[3][:3] == (2, 12, 30), "batch 2 starts a third of the way in, not at 100%"
    assert all(percent < 100 for batch, _, _, percent in seen if batch < 3)
    assert seen[-1] == (3, 30, 30, 100.0)


def test_shorter_last_batch_still_ends_at_the_requested_total():
    job = _job(25)
    for batch, size in ((1, 10), (2, 10), (3, 5)):
        _feed(job, f"⏳ Processing batch {batch} of 3")
        _feed(job, _SAMPLING_LINES["seed"].format(n = size))
        _feed(job, _column_start("answer"))
        _feed(job, _progress("answer", size, size, size, 0))

    assert (job.progress.done, job.progress.total, job.progress.percent) == (25, 25, 100.0)


def test_rows_dropped_by_the_first_column_count_as_finished():
    job = _job(12, llm_columns = 2)
    _feed(job, _SAMPLING_LINES["samplers"].format(n = 12))
    _feed(job, _column_start("question"))
    _feed(job, _progress("question", 12, 12, 0, 12))

    assert (job.progress.done, job.progress.total) == (12, 12)
    assert (job.progress.ok, job.progress.failed) == (0, 12)


def test_partial_drops_carry_into_the_next_column():
    job = _job(12, llm_columns = 2)
    _feed(job, _SAMPLING_LINES["samplers"].format(n = 12))
    _feed(job, _column_start("question"))
    _feed(job, _progress("question", 12, 12, 8, 4))
    assert job.progress.done == 8, "column 2 has the 4 dropped rows done, the other 8 to go"

    _feed(job, _column_start("answer"))
    _feed(job, _progress("answer", 8, 8, 8, 0))
    assert (job.progress.done, job.progress.ok, job.progress.failed) == (12, 8, 4)


def test_eta_comes_from_the_whole_run(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(parse.time, "time", lambda: clock["now"])
    job = _job(30)
    _feed(job, "⏳ Processing batch 1 of 3")
    _feed(job, _SAMPLING_LINES["seed"].format(n = 10))
    _feed(job, _column_start("answer"))
    _feed(job, _progress("answer", 1, 10, 1, 0, eta = 9.0))
    clock["now"] = 110.0
    _feed(job, _progress("answer", 10, 10, 10, 0, eta = 0.0))

    # 10 of 30 rows in the 10 s since the first progress line: 20 rows left at 1 row/s.
    assert job.progress.rate == pytest.approx(1.0)
    assert job.progress.eta_sec == pytest.approx(20.0)


def test_start_records_the_requested_row_count(monkeypatch):
    class _Process:
        pid = None

        def start(self):
            pass

        def is_alive(self):
            return True

    class _Thread:
        def start(self):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(manager_module._CTX, "Process", lambda **kwargs: _Process())
    monkeypatch.setattr(manager_module._CTX, "Queue", queue.Queue)
    monkeypatch.setattr(manager_module, "account_thread", lambda **kwargs: _Thread())
    m = manager_module.JobManager()
    job_id = m.start(recipe = {}, run = {"rows": 30})

    assert m.get_status(job_id)["rows"] == 30
