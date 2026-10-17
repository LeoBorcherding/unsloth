# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every provider endpoint keeps a gateway base URL's query string behind the path."""

import asyncio

import httpx
import pytest

from core.inference import external_provider as ep
from core.inference.external_provider import ExternalProviderClient, _append_provider_path

BASE = "https://gateway.example/openai/v1?api-version=preview"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta?api-version=preview"
MESSAGES = [{"role": "user", "content": "Hi"}]


def _record(monkeypatch, body = None):
    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(400, json = body or {"error": {"message": "stop here"}})

    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(ep, "_http_client", real_async_client(transport = transport))
    # delete_openai_container opens a fresh client of its own.
    monkeypatch.setattr(
        ep.httpx, "AsyncClient", lambda **kw: real_async_client(transport = transport, **kw)
    )
    return sent


async def _drain(agen):
    try:
        async for _ in agen:
            pass
    except Exception:
        pass


async def _call(coro):
    try:
        await coro
    except Exception:
        pass


CASES = {
    "chat_completions_stream": (
        "custom",
        "/openai/v1/chat/completions",
        lambda c: _drain(c.stream_chat_completion(MESSAGES, "m")),
    ),
    "chat_completions": (
        "custom",
        "/openai/v1/chat/completions",
        lambda c: _call(c.chat_completion(MESSAGES, "m")),
    ),
    "kimi_web_search": (
        "kimi",
        "/openai/v1/chat/completions",
        lambda c: _drain(c.stream_chat_completion(MESSAGES, "m", enabled_tools = ["web_search"])),
    ),
    "anthropic_messages": (
        "anthropic",
        "/openai/v1/messages",
        lambda c: _drain(c.stream_chat_completion(MESSAGES, "claude-test")),
    ),
    "gemini_stream": (
        "gemini",
        "/v1beta/models/gemini-test:streamGenerateContent",
        lambda c: _drain(c.stream_chat_completion(MESSAGES, "gemini-test")),
    ),
    "models": ("custom", "/openai/v1/models", lambda c: _call(c.list_models())),
    "models_lightweight": (
        "custom",
        "/openai/v1/models",
        lambda c: _call(c.verify_models_endpoint_lightweight()),
    ),
    "transcriptions": (
        "custom",
        "/openai/v1/audio/transcriptions",
        lambda c: _call(c.create_transcription(b"RIFF", "a.wav", "audio/wav", "whisper")),
    ),
    "containers_list": (
        "openai",
        "/openai/v1/containers",
        lambda c: _call(c.list_openai_containers()),
    ),
    "containers_create": (
        "openai",
        "/openai/v1/containers",
        lambda c: _call(c.create_openai_container("box", 20)),
    ),
    "containers_delete": (
        "openai",
        "/openai/v1/containers/cntr_1",
        lambda c: _call(c.delete_openai_container("cntr_1")),
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_endpoint_keeps_base_url_query(monkeypatch, case):
    provider_type, path, call = CASES[case]
    sent = _record(monkeypatch)
    # Only Google-hosted Gemini takes the native streamGenerateContent path.
    base = GEMINI_BASE if case == "gemini_stream" else BASE
    client = ExternalProviderClient(provider_type, base, "test-key")
    asyncio.run(call(client))
    assert sent, "no request was sent"
    url = sent[0].url
    assert url.path == path
    query = url.query.decode()
    if case == "gemini_stream":
        assert query == "api-version=preview&alt=sse"
    else:
        assert query == "api-version=preview"


@pytest.mark.parametrize(
    "base,endpoint,query,expected",
    [
        ("https://h/v1", "/models", "", "https://h/v1/models"),
        ("https://h/v1/", "models", "", "https://h/v1/models"),
        ("https://h/v1?k=1", "/models", "", "https://h/v1/models?k=1"),
        ("https://h/v1", "/m:stream", "alt=sse", "https://h/v1/m:stream?alt=sse"),
        ("https://h/v1?k=1", "/m:stream", "alt=sse", "https://h/v1/m:stream?k=1&alt=sse"),
    ],
)
def test_append_provider_path(base, endpoint, query, expected):
    assert _append_provider_path(base, endpoint, query) == expected
