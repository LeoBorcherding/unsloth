# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A safetensors / MLX non-streaming chat completion must stop generating once its client is gone.

Starlette does not cancel a non-streaming handler on disconnect, so without a watcher an abandoned
request decoded to max_tokens while the next one waited on the generation lock. A durable run
(cancel_on_disconnect=False) must still finish.
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from routes.inference import openai_chat_completions, produce_openai_chat_completions  # noqa: E402
from test_sf_client_tools_passthrough import _Request, _ScriptedBackend, _install, _request  # noqa: E402

_STEPS = 40


class _GoneRequest(_Request):
    async def is_disconnected(self):
        return True


class _SlowBackend(_ScriptedBackend):
    """Decodes ``_STEPS`` tokens, stopping early once ``cancel_event`` is set."""

    def __init__(self):
        super().__init__(lambda messages, tools: [])
        self.steps = 0

    def _decode(self, cancel_event, as_event):
        text = ""
        for _ in range(_STEPS):
            if cancel_event is not None and cancel_event.is_set():
                return
            time.sleep(0.05)
            text += "tok "
            self.steps += 1
            yield {"type": "content", "text": text} if as_event else text

    def generate_chat_response(self, *, messages, cancel_event = None, **kwargs):
        yield from self._decode(cancel_event, as_event = False)

    def generate_chat_completion_with_tools(self, *, messages, cancel_event = None, **kwargs):
        yield from self._decode(cancel_event, as_event = True)


@pytest.mark.parametrize("tool_loop", [False, True], ids = ["plain", "tool_loop"])
def test_a_gone_client_stops_the_non_streaming_generation(monkeypatch, tool_loop):
    backend = _SlowBackend()
    monitor = _install(monkeypatch, backend, supports_tools = tool_loop)
    payload = _request(stream = False, enable_tools = True) if tool_loop else _request(stream = False)

    async def _run():
        return await openai_chat_completions(payload, request = _GoneRequest(), current_subject = "u")

    asyncio.run(_run())
    assert backend.steps < _STEPS, f"ran all {backend.steps} steps for a client that was gone"
    assert [e.status for e in monitor._entries] == ["cancelled"]


def test_a_durable_run_keeps_generating_after_the_client_leaves(monkeypatch):
    backend = _SlowBackend()
    monitor = _install(monkeypatch, backend, supports_tools = False)

    async def _run():
        return await produce_openai_chat_completions(
            _request(stream = False),
            _GoneRequest(),
            "u",
            cancel_on_disconnect = False,
        )

    asyncio.run(_run())
    assert backend.steps == _STEPS
    assert [e.status for e in monitor._entries] == ["completed"]
