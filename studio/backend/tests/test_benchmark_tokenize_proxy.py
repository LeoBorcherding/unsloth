# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""lm_eval tokenization must authenticate to Studio's private llama-server."""

import asyncio
import inspect
from types import SimpleNamespace

import httpx

from routes import inference


def test_tokenize_forwards_the_child_server_key(monkeypatch):
    backend = SimpleNamespace(
        is_loaded = True,
        base_url = "http://127.0.0.1:12345",
        _auth_headers = {"Authorization": "Bearer child-key"},
    )
    body = {"content": "Question: 2+2?", "add_special": False}
    seen = []

    class Client:
        closed = False

        async def post(self, url, **kwargs):
            seen.append((url, kwargs))
            return httpx.Response(200, json = {"tokens": [1, 2]})

        async def aclose(self):
            self.closed = True

    client = Client()

    async def request_body(*args, **kwargs):
        return body

    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference, "_auto_switch_from_request_body", request_body)
    monkeypatch.setattr(inference, "_cancelable_nonstreaming_client", lambda: client)
    response = asyncio.run(inspect.unwrap(inference.llama_tokenize)(SimpleNamespace(), "unsloth"))
    assert response.status_code == 200
    assert client.closed
    assert seen == [
        (
            backend.base_url + "/tokenize",
            {
                "json": body,
                "timeout": 30,
                "headers": {"Authorization": "Bearer child-key", "Connection": "close"},
            },
        )
    ]
