# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A literal ``</tool_call>`` inside an open Qwen3-Coder XML parameter is argument
data. Auto-Heal used to cut the call body there and run the truncated argument."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.passthrough_healing import heal_openai_message
from core.inference.tool_call_parser import parse_tool_calls_from_text

CASES = [
    pytest.param(
        "<tool_call>\n<function=python>\n<parameter=code>\n"
        'marker = "</tool_call>"\nprint(len(marker))\n'
        "</parameter>\n</function>\n</tool_call>",
        "python",
        {"code": 'marker = "</tool_call>"\nprint(len(marker))'},
        id = "code",
    ),
    pytest.param(
        "<tool_call>\n<function=write>\n<parameter=q>\n"
        "Wrap calls as <tool_call>...</tool_call> in the prompt.\n"
        "</parameter>\n</function>\n</tool_call>",
        "write",
        {"q": "Wrap calls as <tool_call>...</tool_call> in the prompt."},
        id = "prose",
    ),
]


@pytest.mark.parametrize("allow_incomplete", [True, False], ids = ["autoheal", "strict"])
@pytest.mark.parametrize("text, name, args", CASES)
def test_parser_keeps_literal_end_tag_in_parameter(text, name, args, allow_incomplete):
    calls = parse_tool_calls_from_text(text, allow_incomplete = allow_incomplete)
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == [
        (name, args)
    ]


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call>\n<function=x>\n<parameter=a>\nval\n</tool_call>",
        "<tool_call>\n<function=x>\n<parameter=a>\nval\n</tool_call>\nThanks, done.",
        "<tool_call><function=x><parameter=a>val</tool_call>"
        "<tool_call><function=y><parameter=b>w</parameter></function></tool_call>",
    ],
)
def test_a_parameter_that_never_closes_still_ends_at_the_end_tag(text):
    calls = parse_tool_calls_from_text(text)
    assert (calls[0]["function"]["name"], json.loads(calls[0]["function"]["arguments"])) == (
        "x",
        {"a": "val"},
    )


@pytest.mark.parametrize("text, name, args", CASES)
def test_passthrough_heal_keeps_literal_end_tag_in_parameter(text, name, args):
    msg = {"role": "assistant", "content": text}
    assert heal_openai_message(msg, {"python", "write"})
    calls = msg["tool_calls"]
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == [
        (name, args)
    ]
    assert not (msg.get("content") or "").strip()


def test_many_literal_end_tags_in_one_parameter_stay_linear(monkeypatch):
    import core.tool_healing as tool_healing

    real = tool_healing._inside_open_parameter
    calls = []
    monkeypatch.setattr(
        tool_healing, "_inside_open_parameter", lambda *a, **k: calls.append(1) or real(*a, **k)
    )

    def parse(n):
        calls.clear()
        value = "x </tool_call> " * n
        text = (
            "<tool_call>\n<function=write>\n<parameter=q>\n"
            f"{value}\n</parameter>\n</function>\n</tool_call>"
        )
        parsed = parse_tool_calls_from_text(text)
        assert json.loads(parsed[0]["function"]["arguments"]) == {"q": value.strip()}
        return len(calls)

    assert parse(2000) - parse(2) < 10
