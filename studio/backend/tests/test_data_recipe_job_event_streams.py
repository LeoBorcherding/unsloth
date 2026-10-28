# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data recipe job event streams: reconnects during a live job, job changes and slow clients.
Real JobManager and events route, fake worker processes."""

from __future__ import annotations

import asyncio
import json
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from auth import policy  # noqa: E402
from core.data_recipe.jobs import manager as module  # noqa: E402
from core.data_recipe.jobs.constants import EVENT_JOB_COMPLETED  # noqa: E402
from routes.data_recipe import jobs as route  # noqa: E402


class _FakeProcess:
    pid = None
    exitcode = 0

    def __init__(self):
        self.running = False

    def start(self):
        self.running = True

    def is_alive(self):
        return self.running

    def terminate(self):
        self.running = False


class _FakeThread:
    def start(self):
        pass

    def is_alive(self):
        return False


@pytest.fixture
def manager(monkeypatch):
    from auth import storage

    monkeypatch.setattr(storage, "revoke_internal_api_key", lambda key_id: None)
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: False)
    procs: list[_FakeProcess] = []

    def make_process(**kwargs):
        procs.append(_FakeProcess())
        return procs[-1]

    monkeypatch.setattr(module._CTX, "Process", make_process)
    monkeypatch.setattr(module._CTX, "Queue", queue.Queue)
    monkeypatch.setattr(module, "account_thread", lambda **kwargs: _FakeThread())
    m = module.JobManager()
    m.procs = procs
    monkeypatch.setattr(route, "get_job_manager", lambda: m)
    return m


class _Request:
    def __init__(self, after = None):
        self.headers = {}
        self.query_params = {} if after is None else {"after": str(after)}

    async def is_disconnected(self):
        return False


def _read_stream(job_id, *, after = None, before_read = None, timeout = 5.0):
    """Frames the events route sends, and whether the stream ended by itself."""

    async def run():
        response = await route.job_events(_Request(after), job_id)
        if before_read is not None:
            before_read()
        frames: list[dict] = []

        async def collect():
            async for chunk in response.body_iterator:
                text = chunk.decode() if isinstance(chunk, bytes) else chunk
                for line in text.splitlines():
                    if line.startswith("data: "):
                        frames.append(json.loads(line[len("data: ") :]))

        try:
            await asyncio.wait_for(collect(), timeout)
            return frames, True
        except asyncio.TimeoutError:
            return frames, False

    return asyncio.run(run())


def test_reconnect_during_a_live_job_never_hits_a_mutating_buffer(manager):
    job_id = manager.start(recipe = {}, run = {})
    job = manager._job
    for i in range(4000):
        manager._handle_event(job, {"type": "tick", "i": i})

    stop = threading.Event()
    errors: list[BaseException] = []

    def pump():
        i = 0
        while not stop.is_set():
            manager._handle_event(job, {"type": "tick", "i": i})
            i += 1

    emitter = threading.Thread(target = pump, daemon = True)
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)  # interleave the two threads often enough to see the race
    emitter.start()
    try:
        for _ in range(400):
            try:
                sub = manager.subscribe(job_id, after_seq = 10)
                manager.unsubscribe(sub)
            except RuntimeError as exc:
                errors.append(exc)
    finally:
        stop.set()
        emitter.join(timeout = 5)
        sys.setswitchinterval(switch_interval)

    assert errors == [], f"{len(errors)} of 400 reconnects raised: {errors[0]!r}"
    seqs = [event["seq"] for event in manager._events]
    assert len(seqs) == len(set(seqs))


def test_a_finished_jobs_subscriber_gets_nothing_from_the_next_job(manager):
    first = manager.start(recipe = {}, run = {})
    sub = manager.subscribe(first)
    manager._handle_event(manager._job, {"type": EVENT_JOB_COMPLETED})
    manager.procs[-1].running = False
    second = manager.start(recipe = {}, run = {})

    drained = []
    while True:
        try:
            drained.append(sub._q.get_nowait())
        except queue.Empty:
            break
    assert [event["type"] for event in drained] == [EVENT_JOB_COMPLETED]
    assert not any(event.get("job_id") == second for event in drained)
    assert sub.ended


def test_stream_closes_after_the_terminal_event(manager):
    job_id = manager.start(recipe = {}, run = {})
    manager._handle_event(manager._job, {"type": "job.started"})
    manager._handle_event(manager._job, {"type": EVENT_JOB_COMPLETED})

    frames, ended = _read_stream(job_id)

    assert [frame["type"] for frame in frames] == ["job.enqueued", "job.started", "job.completed"]
    assert ended, "a finished job's stream must close instead of idling on keepalives"


def test_stream_on_a_replaced_job_ends_without_the_next_jobs_events(manager):
    first = manager.start(recipe = {}, run = {})

    def next_job_starts():
        manager.procs[-1].running = False  # worker gone before the pump finalized it
        manager.start(recipe = {}, run = {})

    frames, ended = _read_stream(first, before_read = next_job_starts)

    assert [frame.get("job_id") for frame in frames] == [first]
    assert ended


def test_a_slow_client_catches_up_from_the_replay_buffer(manager):
    job_id = manager.start(recipe = {}, run = {})
    job = manager._job

    def flood():
        for i in range(2100):
            manager._handle_event(job, {"type": "tick", "i": i})
        manager._handle_event(job, {"type": "tick", "i": "LATE"})
        manager._handle_event(job, {"type": EVENT_JOB_COMPLETED})

    frames, ended = _read_stream(job_id, before_read = flood, timeout = 15.0)

    seqs = [frame["seq"] for frame in frames]
    assert seqs == list(range(1, len(seqs) + 1)), "no gaps and no repeats"
    assert frames[-2]["i"] == "LATE"
    assert frames[-1]["type"] == EVENT_JOB_COMPLETED
    assert ended
