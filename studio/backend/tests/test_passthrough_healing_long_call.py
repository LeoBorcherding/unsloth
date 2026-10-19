# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A long text-form tool call streamed token by token (a file write through a client tool) is
promoted without re-parsing the whole held block on every token, and one over 64 KiB is still a
call rather than text."""

import json
import time

from core.inference.passthrough_healing import StreamToolCallHealer


def _stream(content: str, step: int = 4) -> list:
    call = '<tool_call>{"name": "write", "arguments": {"content": "' + content + '"}}</tool_call>'
    healer = StreamToolCallHealer({"write"})
    events = []
    for i in range(0, len(call), step):
        events.extend(healer.feed(call[i : i + step]))
    events.extend(healer.finalize())
    return events


def _calls(events: list) -> list:
    return [json.loads(e[1]["function"]["arguments"])["content"] for e in events if e[0] == "tool_call"]


def test_a_16_kb_call_streamed_in_small_tokens_is_fast():
    content = "lorem ipsum " * 1400
    start = time.monotonic()
    events = _stream(content)
    elapsed = time.monotonic() - start
    assert _calls(events) == [content]
    # Base re-parsed the held buffer per token: about 20 s here.
    assert elapsed < 5, elapsed


def test_a_call_over_64_kib_is_still_a_call():
    content = "x" * (80 * 1024)
    events = _stream(content, step = 4096)
    assert _calls(events) == [content]
    assert not any(e[0] == "text" and "x" * 100 in e[1] for e in events)


def test_a_call_whose_argument_is_full_of_closers_is_fast():
    # HTML or JSON source in a file write: every token carries a > } or ], none closes the call.
    content = '<div class="a">{"x": [1, 2]}</div>\n' * 900
    call = (
        '<tool_call>{"name": "write", "arguments": '
        + json.dumps({"content": content})
        + "}</tool_call>"
    )
    healer = StreamToolCallHealer({"write"})
    events = []
    start = time.monotonic()
    for i in range(0, len(call), 4):
        events.extend(healer.feed(call[i : i + 4]))
    events.extend(healer.finalize())
    elapsed = time.monotonic() - start
    assert _calls(events) == [content]
    assert elapsed < 5, elapsed
