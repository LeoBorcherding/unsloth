# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Only a leading <think> block is reasoning; a later tag is the answer's own text."""

import pytest

from routes.inference import _extract_responses_reasoning, _ResponsesReasoningExtractor

CODE = 'if line.startswith("<think>"):\n    in_trace = True\nreturn in_trace'
PROSE = "Qwen3 wraps its trace in <think> and </think>. After the close tag comes the answer."


def _stream(chunks, reasoning_content = None, **kwargs):
    ex = _ResponsesReasoningExtractor(parse_think_markers = True, **kwargs)
    reasoning, visible = ex.feed("", reasoning_content)
    for chunk in chunks:
        r, v = ex.feed(chunk)
        reasoning += r
        visible += v
    r, v = ex.finish()
    return reasoning + r, visible + v


def _chunkings(text):
    return [[text], list(text), [text[i : i + 5] for i in range(0, len(text), 5)]]


@pytest.mark.parametrize("chunked", range(3), ids = ["whole", "per-char", "five"])
def test_a_code_answer_naming_the_tag_stays_visible(chunked):
    assert _stream(_chunkings(CODE)[chunked]) == ("", CODE)


@pytest.mark.parametrize("chunked", range(3), ids = ["whole", "per-char", "five"])
def test_prose_after_structured_reasoning_keeps_both_tags(chunked):
    assert _stream(_chunkings(PROSE)[chunked], "Let me explain the tags.") == (
        "Let me explain the tags.",
        PROSE,
    )


@pytest.mark.parametrize("chunked", range(3), ids = ["whole", "per-char", "five"])
def test_a_leading_block_is_still_reasoning_and_a_later_tag_is_not(chunked):
    text = "\n<think>plan</think>Answer: " + CODE
    assert _stream(_chunkings(text)[chunked]) == ("plan", "\nAnswer: " + CODE)


def test_a_prefilled_block_ends_at_its_first_close():
    assert _stream(["plan</think>Use </think> to close, <think> to open."], reasoning_prefilled = True) == (
        "plan",
        "Use </think> to close, <think> to open.",
    )


def test_the_non_streaming_split_follows_the_same_rule():
    assert _extract_responses_reasoning(CODE, parse_think_markers = True) == ("", CODE)
    assert _extract_responses_reasoning(
        "<think>plan</think>" + PROSE, parse_think_markers = True
    ) == ("plan", PROSE)
