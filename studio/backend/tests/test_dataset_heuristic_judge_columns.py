# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data recipe judge and validator columns are named like answers ("code_judge_result") but
hold scores. The heuristic picked the judge as the assistant turn on Text to Python output,
so training learned the score dict instead of the code."""

import json
import sys
from pathlib import Path

import pytest
from datasets import Dataset

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from hub.utils import dataset_format  # noqa: E402
from utils.datasets import format_detection  # noqa: E402

_HEURISTICS = [
    pytest.param(format_detection.detect_custom_format_heuristic, id = "utils"),
    pytest.param(dataset_format.detect_custom_format_heuristic, id = "hub"),
]

# Two rows of a real Text to Python recipe run.
_TEXT_TO_PYTHON = [
    json.loads(line)
    for line in (Path(__file__).parent / "data" / "recipe_text_to_python.jsonl")
    .read_text(encoding = "utf-8")
    .splitlines()
]

_LONG = "x" * 600
_MID = "y" * 120


@pytest.mark.parametrize("heuristic", _HEURISTICS)
def test_text_to_python_trains_the_code_not_the_judge(heuristic):
    mapping = heuristic(Dataset.from_list(_TEXT_TO_PYTHON))

    assert mapping == {"instruction": "user", "code_implementation": "assistant"}


@pytest.mark.parametrize("heuristic", _HEURISTICS)
@pytest.mark.parametrize(
    "verdict",
    [
        {"Correctness": {"reasoning": _LONG, "score": "3"}},
        {
            "Helpfulness": {"score": 4, "reasoning": _LONG},
            "Safety": {"score": 5, "reasoning": _MID},
        },
        {"is_valid": False, "error_messages": _LONG},
    ],
)
def test_judge_and_validator_results_are_never_a_turn(heuristic, verdict):
    row = {"question": _MID, "answer_judge_result": verdict, "answer": _MID}

    assert heuristic(Dataset.from_list([row])) == {"question": "user", "answer": "assistant"}


@pytest.mark.parametrize("heuristic", _HEURISTICS)
@pytest.mark.parametrize(
    "reply",
    [
        {"next_action": "ask_logs", "response": _MID},
        {"overall": {"score": 4, "comment": _MID}},
    ],
)
def test_a_structured_answer_is_still_an_answer(heuristic, reply):
    row = {"question": _MID, "reply": reply}

    assert heuristic(Dataset.from_list([row])) == {"question": "user", "reply": "assistant"}


@pytest.mark.parametrize("heuristic", _HEURISTICS)
@pytest.mark.parametrize(
    "side",
    [
        {"answer__reasoning_content": _LONG},
        {
            "answer__trace": [
                {"role": "user", "content": _MID},
                {"role": "assistant", "content": _LONG},
            ]
        },
    ],
)
def test_reasoning_and_trace_columns_are_never_a_turn(heuristic, side):
    row = {"question": _MID, "answer": _MID, **side}

    assert heuristic(Dataset.from_list([row])) == {"question": "user", "answer": "assistant"}


@pytest.mark.parametrize("heuristic", _HEURISTICS)
def test_an_implementation_input_does_not_outrank_the_answer(heuristic):
    row = {"instruction": _MID, "implementation": _LONG, "expected_answer": _MID}

    assert heuristic(Dataset.from_list([row]))["expected_answer"] == "assistant"
