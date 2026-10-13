# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Back-to-back recipe runs: the previous run's pump and worker teardown must not touch the
next run. Fake processes and queues only."""

from __future__ import annotations

import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.data_recipe.jobs import manager as module  # noqa: E402
from core.data_recipe.jobs.constants import EVENT_JOB_COMPLETED  # noqa: E402


class _FakeProcess:
    pid = None
    exitcode = 0

    def __init__(self, exits_on_join = True):
        self.running = False
        self.exits_on_join = exits_on_join
        self.joins = 0

    def start(self):
        self.running = True

    def is_alive(self):
        return self.running

    def terminate(self):
        self.running = False

    def join(self, timeout = None):
        self.joins += 1
        if self.exits_on_join:
            self.running = False


class _FakeThread:
    def __init__(self, target, args, kwargs):
        self.target, self.args, self.kwargs = target, args, kwargs

    def start(self):
        pass

    def is_alive(self):
        return False

    def run(self):
        self.target(*self.args, **self.kwargs)


@pytest.fixture
def harness(monkeypatch):
    from auth import storage

    revoked: list[int] = []
    monkeypatch.setattr(storage, "revoke_internal_api_key", lambda key_id: revoked.append(key_id))
    procs: list[_FakeProcess] = []
    queues: list = []
    pumps: list[_FakeThread] = []

    def make_process(**kwargs):
        procs.append(_FakeProcess())
        return procs[-1]

    def make_queue():
        return queues.pop(0) if queues else queue.Queue()

    def make_thread(*, target, args = (), kwargs = None, **_):
        pumps.append(_FakeThread(target, args, kwargs or {}))
        return pumps[-1]

    monkeypatch.setattr(module._CTX, "Process", make_process)
    monkeypatch.setattr(module._CTX, "Queue", make_queue)
    monkeypatch.setattr(module, "account_thread", make_thread)
    return SimpleNamespace(
        manager = module.JobManager(),
        revoked = revoked,
        procs = procs,
        queues = queues,
        pumps = pumps,
    )


def test_previous_pump_does_not_finalize_or_revoke_the_next_job(harness):
    m = harness.manager

    class _FirstRunQueue:
        """Completed, then the worker exits and Run is clicked during the next poll."""

        def __init__(self):
            self.reads = 0

        def get(self, timeout = None):
            self.reads += 1
            if self.reads == 1:
                return {"type": EVENT_JOB_COMPLETED, "artifact_path": "/tmp/run-1"}
            if self.reads == 2:
                harness.procs[0].running = False
                harness.second_job_id = m.start(recipe = {}, run = {}, internal_api_key_id = 202)
            raise queue.Empty

        def get_nowait(self):
            raise queue.Empty

    harness.queues.append(_FirstRunQueue())
    first_job_id = m.start(recipe = {}, run = {}, internal_api_key_id = 101)
    harness.pumps[0].run()

    assert m.get_current_job_id() == harness.second_job_id != first_job_id
    second = m.get_status(harness.second_job_id)
    assert second["status"] == "pending", second
    assert second["error"] is None
    assert harness.revoked == [101], "only the finished run's key may be revoked"
    assert m._job.internal_api_key_id == 202


def test_start_right_after_completed_waits_for_worker_teardown(harness):
    m = harness.manager
    first_job_id = m.start(recipe = {}, run = {}, internal_api_key_id = 101)
    m._handle_event(m._job, {"type": EVENT_JOB_COMPLETED})
    assert harness.procs[0].is_alive(), "worker reports completed before it exits"

    second_job_id = m.start(recipe = {}, run = {})

    assert second_job_id != first_job_id
    assert harness.procs[0].joins == 1


def test_start_still_refuses_a_generating_or_stuck_worker(harness):
    m = harness.manager
    m.start(recipe = {}, run = {})
    with pytest.raises(RuntimeError, match = "job already running"):
        m.start(recipe = {}, run = {})
    assert harness.procs[0].joins == 0, "a worker that is still generating is not waited on"

    harness.procs[0].exits_on_join = False
    m._handle_event(m._job, {"type": EVENT_JOB_COMPLETED})
    with pytest.raises(RuntimeError, match = "job already running"):
        m.start(recipe = {}, run = {})


def test_busy_create_job_does_not_mint_a_workflow_key(harness, monkeypatch):
    from auth import storage
    from models.data_recipe import RecipePayload
    from routes.data_recipe import jobs as route

    minted: list[dict] = []
    monkeypatch.setattr(
        storage,
        "create_api_key",
        lambda **kwargs: minted.append(kwargs) or ("sk-unsloth-test", {"id": 7}),
    )
    monkeypatch.setattr(route, "_resolve_local_v1_endpoint", lambda request: "http://x/v1")
    monkeypatch.setattr(route, "_ensure_selected_local_model_loaded", lambda recipe, names: None)
    monkeypatch.setattr(route, "_local_chat_serves_gguf", lambda: False)
    monkeypatch.setattr(route, "get_job_manager", lambda: harness.manager)
    harness.manager.start(recipe = {}, run = {})

    payload = RecipePayload(
        recipe = {
            "model_providers": [{"name": "local", "is_local": True}],
            "model_configs": [{"provider": "local", "alias": "model"}],
            "columns": [{"column_type": "llm-text", "name": "x", "model_alias": "model"}],
        },
        run = {"rows": 1},
    )
    with pytest.raises(HTTPException) as exc:
        route.create_job(payload, SimpleNamespace(), (None, "generation"), False)

    assert exc.value.status_code == 409
    assert minted == []
    assert harness.revoked == []
