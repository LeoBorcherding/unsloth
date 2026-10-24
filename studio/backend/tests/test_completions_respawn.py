# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Text completions and tokenization recover the child's port and private key."""

import asyncio
import inspect
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import HTTPException

from core.inference.llama_cpp import LlamaCppBackend
from routes import inference

DEAD = "http://127.0.0.1:57953"
FRESH = "http://127.0.0.1:62933"


class Backend:
    is_loaded = True
    context_length = 65536
    model_identifier = "test/model.gguf"
    _process = None

    def __init__(
        self,
        *,
        recover = True,
        mtp = False,
    ):
        self.base_url = DEAD
        self._auth_headers = {"Authorization": "Bearer old-child-key"}
        self.respawns = 0
        self.recover = recover
        self.mtp = mtp

    def _maybe_recover_from_mtp_crash(self, exc):
        return self.mtp

    def _respawn_if_dead(self):
        self.respawns += 1
        if self.recover:
            self.base_url = FRESH
            self._auth_headers = {"Authorization": "Bearer new-child-key"}
            self.is_loaded = True
            self._process = None
        return self.recover


class Request:
    method = "POST"
    url = SimpleNamespace(path = "/v1/completions")
    scope = {}
    headers = {}

    async def is_disconnected(self):
        return False


def install_route(monkeypatch, backend, body):
    async def request_body(*args, **kwargs):
        return dict(body)

    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference, "_auto_switch_from_request_body", request_body)
    monkeypatch.setattr(inference, "_should_validate_before_switch", lambda: False)
    monkeypatch.setattr(inference, "_monitor_active_model", lambda: "test/model.gguf")
    monkeypatch.setattr(inference, "api_monitor", Mock())


@pytest.mark.parametrize("prompt", ["Question: 2+2?", [1, 2, 3], ["first", "second"]])
def test_completions_route_recovers_port_and_key(monkeypatch, prompt):
    backend = Backend()
    seen = []

    def handler(request):
        seen.append(
            (str(request.url), request.headers["Authorization"], json.loads(request.content))
        )
        if str(request.url).startswith(DEAD):
            raise httpx.RemoteProtocolError("server died before returning its response")
        return httpx.Response(200, json = {"choices": [{"index": 0, "text": " answer\n\n"}]})

    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        return real_client(transport = httpx.MockTransport(handler))

    monkeypatch.setattr(inference, "_cancelable_nonstreaming_client", client)
    monkeypatch.setattr(inference.httpx, "AsyncClient", client)
    install_route(monkeypatch, backend, {"prompt": prompt, "max_tokens": 32768})

    response = asyncio.run(inspect.unwrap(inference.openai_completions)(Request(), "unsloth"))

    assert response.status_code == 200
    assert backend.respawns >= 1
    successful = [item for item in seen if item[0].startswith(FRESH)]
    assert successful
    assert all(item[1] == "Bearer new-child-key" for item in successful)
    assert all(item[0].endswith("/v1/completions") for item in successful)
    assert all(item[2]["max_tokens"] == 32768 for item in successful)
    assert json.loads(response.body)["choices"][0]["text"] == " answer\n\n"


@pytest.mark.parametrize(
    "error", [httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError]
)
def test_only_one_retry_is_made_if_replacement_also_fails(error):
    backend = Backend()
    calls = []

    def handler(request):
        calls.append(str(request.url))
        raise error("connection lost")

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as client:
            await inference._post_llama_nonstreaming_with_respawn(client, backend, {}, timeout = 30)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(run())
    assert raised.value.status_code == 502
    assert backend.respawns == 1
    assert calls == [DEAD + "/v1/completions", FRESH + "/v1/completions"]


@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout])
def test_slow_server_is_not_respawned_or_replayed(error):
    backend = Backend()
    calls = []

    def handler(request):
        calls.append(str(request.url))
        raise error("slow generation")

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as client:
            await inference._post_llama_nonstreaming_with_respawn(client, backend, {}, timeout = 30)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(run())
    assert raised.value.status_code == 504
    assert backend.respawns == 0
    assert len(calls) == 1


@pytest.mark.parametrize("backend", [Backend(recover = False), Backend(mtp = True)])
def test_failed_respawn_and_mtp_recovery_do_not_replay(backend):
    def handler(request):
        raise httpx.ConnectError("gone")

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as client:
            await inference._post_llama_nonstreaming_with_respawn(client, backend, {}, timeout = 30)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(run())
    assert raised.value.status_code == 502
    assert backend.respawns == (0 if backend.mtp else 1)


def test_cancel_during_respawn_prevents_replay():
    backend = Backend()
    cancel = threading.Event()
    calls = []

    def respawn():
        cancel.set()
        return True

    backend._respawn_if_dead = respawn

    def handler(request):
        calls.append(str(request.url))
        raise httpx.ConnectError("gone")

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as client:
            await inference._post_llama_nonstreaming_with_respawn(
                client, backend, {}, timeout = 30, cancel_event = cancel
            )

    with pytest.raises(inference._NonStreamingRequestCancelled):
        asyncio.run(run())
    assert len(calls) == 1


def test_dead_child_recovers_after_health_probe_marked_it_unloaded(monkeypatch):
    backend = Backend()
    backend.is_loaded = False
    backend._process = SimpleNamespace(poll = lambda: -11)
    seen = []

    class Client:
        async def post(self, url, **kwargs):
            seen.append(url)
            return httpx.Response(200, json = {"tokens": [1, 2]})

        async def aclose(self):
            pass

    install_route(monkeypatch, backend, {"content": "hello", "add_special": True})
    monkeypatch.setattr(inference, "_cancelable_nonstreaming_client", Client)
    response = asyncio.run(inspect.unwrap(inference.llama_tokenize)(Request(), "unsloth"))
    assert response.status_code == 200
    assert backend.respawns == 1
    assert seen == [FRESH + "/tokenize"]


def test_deliberate_unload_is_not_recovered():
    backend = LlamaCppBackend(manages_processes = False)
    backend._process = SimpleNamespace(poll = lambda: -15)
    backend._cancel_event.set()
    backend.load_model = Mock(side_effect = AssertionError("must not reload"))
    asyncio.run(inference._recover_dead_llama_passthrough(backend))
    backend.load_model.assert_not_called()
