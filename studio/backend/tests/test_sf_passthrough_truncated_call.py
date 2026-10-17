# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A safetensors / MLX client-tool call cut off at max_tokens must keep finish_reason "length".

The healer promotes whatever was written, so a call truncated mid-argument used to go out as
"tool_calls" and an agent would run half a command. The GGUF passthrough only upgrades "stop".
"""

import json
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_sf_client_tools_passthrough import (  # noqa: E402
    _CALL_XML,
    LOOKUP_TOOL,
    _ScriptedBackend,
    _call,
    _collect_sse,
    _json_body,
    _request,
    _sse_objects,
)

WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}
# Qwen3-Coder XML call cut mid-argument at the cap.
_CUT = (
    "<tool_call>\n<function=write_file>\n<parameter=path>\napp.py\n</parameter>\n"
    "<parameter=content>\nimport os\nos.remove("
)
_AT_CAP = {"truncated": True, "usage": {"prompt_tokens": 10, "completion_tokens": 32}}
_DONE = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}


def _finish_and_calls(stream, monkeypatch, backend, **kwargs):
    payload = _request(stream = stream, max_tokens = 32, **kwargs)
    response = _call(payload, monkeypatch, backend)
    if not stream:
        choice = _json_body(response)["choices"][0]
        return choice["finish_reason"], choice["message"].get("tool_calls") or []
    objs = [o for o in _sse_objects(_collect_sse(response)) if o.get("choices")]
    finish = [o["choices"][0]["finish_reason"] for o in objs if o["choices"][0].get("finish_reason")]
    calls = [
        tc for o in objs for tc in (o["choices"][0].get("delta") or {}).get("tool_calls") or []
    ]
    return finish[-1], calls


@pytest.mark.parametrize("stream", [False, True], ids = ["json", "sse"])
def test_a_call_cut_off_at_the_cap_reports_length(monkeypatch, stream):
    backend = _ScriptedBackend(lambda m, t: [_CUT], stats = _AT_CAP)
    finish, calls = _finish_and_calls(stream, monkeypatch, backend, tools = [WRITE_TOOL])
    assert finish == "length"
    # The healed call stays attached, as on the GGUF path; only the reason changes.
    assert calls and calls[0]["function"]["name"] == "write_file"
    assert "os.remove(" in json.dumps(calls)


@pytest.mark.parametrize("stream", [False, True], ids = ["json", "sse"])
def test_a_complete_call_still_reports_tool_calls(monkeypatch, stream):
    backend = _ScriptedBackend(lambda m, t: [_CALL_XML], stats = _DONE)
    finish, calls = _finish_and_calls(stream, monkeypatch, backend, tools = [LOOKUP_TOOL])
    assert finish == "tool_calls"
    assert calls[0]["function"]["name"] == "lookup"


def test_a_nudge_retry_cut_off_at_the_cap_reports_length(monkeypatch):
    def responder(messages, tools):
        nudged = any(
            "native tool-call format" in (m.get("content") or "")
            for m in messages
            if m.get("role") == "user"
        )
        return [_CALL_XML] if nudged else ['<tool_call>{"name": "lookup"']

    backend = _ScriptedBackend(responder, stats = [_DONE, _AT_CAP])
    finish, calls = _finish_and_calls(
        False, monkeypatch, backend, tools = [LOOKUP_TOOL], nudge_tool_calls = True
    )
    assert len(backend.calls) == 2
    assert finish == "length"
    assert calls[0]["function"]["name"] == "lookup"
