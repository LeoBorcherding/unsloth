# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A finished data recipe run stays readable, downloadable and unchanged by a late Cancel after
the next run starts. Fake processes, queues and Hub only."""

from __future__ import annotations

import json
import queue
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from auth import policy  # noqa: E402
from core.data_recipe import huggingface as recipe_hf  # noqa: E402
from core.data_recipe.jobs import manager as module  # noqa: E402
from core.data_recipe.jobs.constants import (  # noqa: E402
    EVENT_JOB_CANCELLED,
    EVENT_JOB_COMPLETED,
    EVENT_JOB_ENQUEUED,
)
from routes.data_recipe import jobs as route  # noqa: E402
from utils.account_context import AccountContext, run_as  # noqa: E402


class _FakeProcess:
    pid = None
    exitcode = 0

    def __init__(self):
        self.running = False
        self.terminated = False

    def start(self):
        self.running = True

    def is_alive(self):
        return self.running

    def terminate(self):
        self.terminated = True
        self.running = False


class _FakeThread:
    def __init__(self, target):
        self.target = target

    def start(self):
        pass

    def is_alive(self):
        return False

    def run(self):
        self.target()


@pytest.fixture
def harness(monkeypatch):
    from auth import storage

    monkeypatch.setattr(storage, "revoke_internal_api_key", lambda key_id: None)
    procs: list[_FakeProcess] = []
    pumps: list[_FakeThread] = []

    def make_process(**kwargs):
        procs.append(_FakeProcess())
        return procs[-1]

    def make_thread(*, target, **_):
        pumps.append(_FakeThread(target))
        return pumps[-1]

    monkeypatch.setattr(module._CTX, "Process", make_process)
    monkeypatch.setattr(module._CTX, "Queue", queue.Queue)
    monkeypatch.setattr(module, "account_thread", make_thread)
    manager = module.JobManager()
    monkeypatch.setattr(route, "get_job_manager", lambda: manager)
    return SimpleNamespace(manager = manager, procs = procs, pumps = pumps)


def _finish(harness, job_id, **event):
    m = harness.manager
    m._handle_event(m._job, {"type": EVENT_JOB_COMPLETED, **event})
    harness.procs[-1].running = False


def test_cancel_right_after_completed_keeps_the_run_completed(harness):
    m = harness.manager
    job_id = m.start(recipe = {}, run = {})
    m._handle_event(m._job, {"type": EVENT_JOB_COMPLETED, "artifact_path": "/tmp/run-1"})
    assert harness.procs[0].is_alive(), "the worker outlives its completed event"

    assert m.cancel(job_id) is True
    assert not harness.procs[0].terminated
    harness.procs[0].running = False
    harness.pumps[0].run()  # the worker exits and the pump finalizes

    assert m.get_status(job_id)["status"] == "completed"
    assert route._resolve_download_artifact_path(job_id = job_id, artifact_path = None) == (
        "/tmp/run-1"
    )


def test_finished_preview_stays_readable_after_the_next_run_starts(harness):
    m = harness.manager
    first = m.start(recipe = {}, run = {"execution_type": "preview"})
    _finish(harness, first, dataset = [{"q": "a"}, {"q": "b"}], analysis = {"rows": 2})
    second = m.start(recipe = {}, run = {})

    assert m.get_current_job_id() == second
    assert m.get_status(first)["status"] == "completed"
    assert m.get_dataset(first, limit = 10) == {"dataset": [{"q": "a"}, {"q": "b"}], "total": 2}
    assert m.get_analysis(first) == {"rows": 2}
    assert m.cancel(first) is True
    assert m.get_status(first)["status"] == "completed"
    assert m.get_status(second)["status"] == "pending"


def test_events_still_queued_when_the_next_run_starts_land_on_the_kept_run(harness):
    m = harness.manager
    first = m.start(recipe = {}, run = {"execution_type": "preview"})
    first_job = m._job
    harness.procs[0].running = False  # exited; its completed event is still in the queue
    second = m.start(recipe = {}, run = {})
    seen_before = len(m._events)

    m._handle_event(first_job, {"type": EVENT_JOB_COMPLETED, "dataset": [{"q": "a"}]})

    assert m.get_status(first)["status"] == "completed"
    assert m.get_dataset(first, limit = 10) == {"dataset": [{"q": "a"}], "total": 1}
    assert m.get_status(second)["status"] == "pending"
    assert len(m._events) == seen_before, "the old run's event must not reach the new run's stream"


def test_an_event_handled_while_the_next_run_starts_stays_off_its_stream(harness, monkeypatch):
    m = harness.manager
    first = m.start(recipe = {}, run = {})
    first_job = m._job
    harness.procs[0].running = False
    started = []

    def next_run_starts_meanwhile(job):
        started.append(m.start(recipe = {}, run = {}))

    monkeypatch.setattr(m, "_retire_workflow_key", next_run_starts_meanwhile)
    m._handle_event(first_job, {"type": EVENT_JOB_COMPLETED})

    assert m.get_status(first)["status"] == "completed"
    assert [e["type"] for e in m._events] == [EVENT_JOB_ENQUEUED]
    assert m.get_current_job_id() == started[0]


def test_only_the_last_few_finished_runs_are_kept(harness):
    m = harness.manager
    ids = []
    for _ in range(module._FINISHED_JOBS_KEPT + 2):
        ids.append(m.start(recipe = {}, run = {}))
        _finish(harness, ids[-1])

    assert m.get_status(ids[0]) is None
    assert all(m.get_status(job_id) is not None for job_id in ids[2:])


def test_another_account_starting_drops_the_previous_accounts_finished_runs(harness, monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    alice = AccountContext("alice-recipe-account", "alice")
    bob = AccountContext("bob-recipe-account", "bob")
    m = harness.manager

    alice_job = run_as(alice, m.start, recipe = {}, run = {})
    _finish(harness, alice_job, dataset = [{"secret": "alice"}])
    run_as(bob, m.start, recipe = {}, run = {})

    assert run_as(bob, m.get_status, alice_job) is None
    assert run_as(bob, m.get_dataset, alice_job, limit = 10) is None


def test_cancelled_full_run_downloads_its_finished_batches(harness, monkeypatch, tmp_path):
    m = harness.manager
    job_id = m.start(recipe = {}, run = {})
    m._job.artifact_path = "/tmp/run-cancelled"
    m._handle_event(m._job, {"type": EVENT_JOB_CANCELLED})
    m._job.status = "cancelled"
    harness.procs[0].running = False

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        out = tmp_path / "out.jsonl"
        out.write_text('{"ok": true}\n', encoding = "utf-8")
        return out, "application/x-ndjson", "run.jsonl"

    monkeypatch.setattr(route, "build_dataset_download", fake_build)
    route.download_job_dataset(
        job_id,
        background_tasks = BackgroundTasks(),
        export_format = "jsonl",
        artifact_path = None,
        filename = None,
    )
    assert captured["artifact_path"] == "/tmp/run-cancelled"


@pytest.mark.parametrize("status", ["cancelled", "error"])
def test_preview_that_ended_without_rows_says_so(harness, status):
    m = harness.manager
    job_id = m.start(recipe = {}, run = {"execution_type": "preview"})
    m._job.status = status
    harness.procs[0].running = False

    with pytest.raises(HTTPException) as dataset_exc:
        route.job_dataset(job_id, limit = 20, offset = 0)
    assert "artifact path missing" not in dataset_exc.value.detail
    assert status in dataset_exc.value.detail

    with pytest.raises(HTTPException) as link_exc:
        route.create_job_dataset_download_url(
            job_id, export_format = "jsonl", artifact_path = None, filename = None
        )
    assert link_exc.value.status_code == 422

    with pytest.raises(HTTPException) as analysis_exc:
        route.job_analysis(job_id)
    assert analysis_exc.value.detail != "analysis not ready"
    assert status in analysis_exc.value.detail


def _stub_hub(monkeypatch, created):
    class _Client:
        def __init__(self, token):
            self._api = None

        def _validate_repo_id(self, *, repo_id):
            pass

        def _validate_dataset_path(self, *, base_dataset_path):
            pass

        def _create_or_get_repo(self, *, repo_id, private):
            created.append(repo_id)

        def __getattr__(self, name):
            return lambda **kwargs: None

    card = SimpleNamespace(
        from_metadata = lambda **kwargs: SimpleNamespace(text = "", push_to_hub = lambda *a, **k: None)
    )
    modules = {
        "data_designer.engine.storage.artifact_storage": {
            "FINAL_DATASET_FOLDER_NAME": "parquet-files",
            "METADATA_FILENAME": "metadata.json",
            "PROCESSORS_OUTPUTS_FOLDER_NAME": "processors-files",
            "SDG_CONFIG_FILENAME": "builder_config.json",
        },
        "data_designer.integrations.huggingface.client": {
            "HuggingFaceHubClient": _Client,
            "HuggingFaceHubClientUploadError": type("UploadError", (Exception,), {}),
        },
        "data_designer.integrations.huggingface.dataset_card": {"DataDesignerDatasetCard": card},
    }
    for name, attrs in modules.items():
        stub = ModuleType(name)
        for key, value in attrs.items():
            setattr(stub, key, value)
        monkeypatch.setitem(sys.modules, name, stub)


@pytest.mark.parametrize(
    "metadata, published",
    [
        ({"num_completed_batches": 1, "total_num_batches": 6}, False),
        ({"num_completed_batches": 6, "total_num_batches": 6}, True),
        ({}, False),
        ({"num_completed_batches": None, "total_num_batches": None}, False),
    ],
)
def test_publish_after_replacement_needs_every_batch(
    harness, monkeypatch, tmp_path, metadata, published
):
    created: list[str] = []
    _stub_hub(monkeypatch, created)
    monkeypatch.setattr(recipe_hf, "_resolve_recipe_artifact_path", lambda _: tmp_path)
    monkeypatch.setattr(route, "allow_ambient_hf_token", lambda: True)
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding = "utf-8",
    )
    payload = SimpleNamespace(
        repo_id = "org/ds",
        description = "d",
        hf_token = "hf_x",
        artifact_path = str(tmp_path),
        private = False,
    )
    request = SimpleNamespace(client = SimpleNamespace(host = "127.0.0.1"), headers = {})
    monkeypatch.setattr(route, "client_reachable_endpoint", lambda ip: "https://hf.test")
    monkeypatch.setattr(route, "client_ip", lambda request: "127.0.0.1")

    if published:
        route.publish_job_dataset(request, "replaced-job", payload, allow_ambient = True)
        assert created == ["org/ds"]
    else:
        with pytest.raises(HTTPException) as exc:
            route.publish_job_dataset(request, "replaced-job", payload, allow_ambient = True)
        assert exc.value.status_code == 400
        assert "batches" in exc.value.detail
        assert created == [], "nothing may reach the Hub for a partial run"


def _create_job(monkeypatch, harness, rows):
    from models.data_recipe import RecipePayload

    minted: list = []
    started: list = []
    monkeypatch.setattr(
        route, "_inject_local_providers", lambda recipe, request, kind: minted.append(1)
    )
    monkeypatch.setattr(
        route,
        "get_job_manager",
        lambda: SimpleNamespace(start = lambda **kwargs: started.append(kwargs["run"]) or "job"),
    )
    run = {"execution_type": "preview"}
    if rows is not ...:
        run["rows"] = rows
    payload = RecipePayload(recipe = {"columns": [{"name": "x"}]}, run = run)
    route.create_job(payload, SimpleNamespace(), (None, "generation"), False)
    return minted, started


@pytest.mark.parametrize(
    "rows",
    [0, -1, "abc", 1.5, True, 200_001, "²", pytest.param("9" * 5000, id = "5000-digits")],
)
def test_create_job_refuses_bad_rows_before_minting_a_key(harness, monkeypatch, rows):
    with pytest.raises(HTTPException) as exc:
        _create_job(monkeypatch, harness, rows)
    assert exc.value.status_code == 400
    assert "rows" in exc.value.detail


@pytest.mark.parametrize("rows, expected", [(5, 5), ("7", 7), (..., None)])
def test_create_job_accepts_good_rows(harness, monkeypatch, rows, expected):
    minted, started = _create_job(monkeypatch, harness, rows)
    assert minted == [1]
    assert started[0]["rows"] == expected
