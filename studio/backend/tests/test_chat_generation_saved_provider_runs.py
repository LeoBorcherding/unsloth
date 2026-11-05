# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A saved-connection turn runs durably, so its Unsloth Studio tools outlive the browser."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.inference.chat_generation_runs import ChatGenerationSupervisor
from routes import chat_generation_runs as run_routes
from routes import inference
from state import active_generations
from storage import chat_generation_runs_db as runs_db
from storage import studio_db

from .test_external_confirm_gate_and_saved_keys import _install


@pytest.fixture
def thread():
    studio_db.upsert_chat_thread(
        {"id": "thread-1", "title": "Chat", "modelType": "base", "modelId": "m", "createdAt": 1}
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "list the files"}],
            "createdAt": 2,
        }
    )
    active_generations.reset_for_tests()
    yield
    active_generations.reset_for_tests()


def _create(**overrides):
    request_payload = {
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "list the files"}],
        "provider_id": "saved-1",
        "provider_type": "openai",
        "external_model": "gpt-5.4",
        "provider_base_url": "http://127.0.0.1:9/v1",
        "enable_tools": True,
        "enabled_tools": ["terminal"],
        "permission_mode": "full",
        "stream": True,
    }
    request_payload.update(overrides)
    return run_routes.CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = request_payload,
    )


def _route_request(started):
    supervisor = SimpleNamespace(start = lambda run_id, **_identity: started.append(run_id))
    return SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(chat_generation_supervisor = supervisor)),
        headers = {},
        state = SimpleNamespace(),
    )


def test_saved_connection_keeps_routing_and_drops_url_headers_and_key():
    sanitized = run_routes._sanitize_request(_create(extra_headers = {"X-Org": "a"}))
    assert (sanitized["provider_id"], sanitized["provider_type"], sanitized["external_model"]) == (
        "saved-1",
        "openai",
        "gpt-5.4",
    )
    for field in ("encrypted_api_key", "provider_base_url", "extra_headers"):
        assert field not in sanitized


@pytest.mark.parametrize(
    "overrides",
    [
        {"encrypted_api_key": "ciphertext"},
        {"provider_id": None},
    ],
    ids = ["inline-key", "no-saved-connection"],
)
def test_a_turn_that_needs_a_browser_key_stays_on_the_live_stream(overrides):
    with pytest.raises(HTTPException, match = "only for local"):
        run_routes._sanitize_request(_create(**overrides))


@pytest.mark.asyncio
async def test_an_api_key_caller_cannot_spend_the_saved_key_from_a_durable_run(thread, monkeypatch):
    monkeypatch.setattr(inference, "_request_has_api_key", lambda _request: True)
    started = []
    with pytest.raises(HTTPException, match = "only for local"):
        await run_routes.create_chat_generation_run(_create(), _route_request(started), "alice")
    assert started == [] and runs_db.get_run("run-1", "alice") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("keyless_scope", ["off", "full"])
async def test_saved_connection_tool_run_finishes_with_no_client(
    thread, monkeypatch, keyless_scope
):
    """The phone locks mid tool call: nothing follows the run, and the tool and reply still land."""
    from auth import policy as auth_policy
    from utils import keyless_api_access

    # Full keyless access must not reclassify the producer's credential-less request as keyless.
    monkeypatch.setattr(keyless_api_access, "get_keyless_api_access_scope", lambda: keyless_scope)
    monkeypatch.setattr(keyless_api_access, "keyless_transport_allowed", lambda *_a: True)
    monkeypatch.setattr(auth_policy, "installation_has_managed_accounts", lambda: False)
    _install(monkeypatch, "openai")
    seen = {}

    def _loop(
        *_args,
        cancel_event = None,
        **_kwargs,
    ):
        async def gen():
            yield 'data: {"type":"tool_start","tool_name":"terminal","tool_call_id":"c1"}\n\n'
            for _ in range(20):
                if cancel_event.is_set():
                    seen["cancelled_mid_tool"] = True
                    return
                await asyncio.sleep(0.01)
            seen["tool_finished"] = True
            yield 'data: {"type":"tool_end","tool_name":"terminal","tool_call_id":"c1"}\n\n'
            yield 'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        return gen()

    monkeypatch.setattr(inference, "stream_with_studio_tools", _loop)
    started = []
    run = await run_routes.create_chat_generation_run(_create(), _route_request(started), "alice")
    assert started == ["run-1"] and run["status"] == "queued"

    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    await asyncio.wait_for(supervisor._produce("run-1"), timeout = 10)

    assert seen == {"tool_finished": True}
    run = runs_db.get_run("run-1", "alice")
    assert (run["status"], run["finishReason"]) == ("completed", "stop")
    types = [
        (event["payload"] or {}).get("type")
        for event in runs_db.list_events("run-1")
        if event["type"] == "chunk"
    ]
    assert "tool_start" in types and "tool_end" in types
    assert "ciphertext" not in json.dumps(run["requestPayload"])


@pytest.mark.asyncio
async def test_saved_connection_run_persists_no_key_and_restart_reads_interrupted(thread):
    started = []
    await run_routes.create_chat_generation_run(_create(), _route_request(started), "alice")
    stored = runs_db.get_run("run-1", "alice")["requestPayload"]
    assert stored["provider_id"] == "saved-1"
    assert not {"encrypted_api_key", "provider_base_url", "extra_headers"} & set(stored)

    assert runs_db.mark_running("run-1", runs_db.get_worker_token("run-1"))
    assert runs_db.reconcile_orphaned_runs() == 1
    run = runs_db.get_run("run-1", "alice")
    assert (run["status"], run["finishReason"]) == ("failed", "interrupted")
