# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.data_recipe.benchmark import (
    parse_judge_verdict,
    run_benchmark,
    score_with_validator,
)
from routes.data_recipe import benchmark as route

ROWS = [
    {"question": "2+2?", "answer": "4"},
    {"question": "Capital of France?", "answer": "Paris"},
    {"question": "3*3?", "answer": "9"},
]


def test_reference_scorer_matches_whole_words_only():
    assert score_with_validator("reference", "The answer is 4.", "4")[0]
    assert not score_with_validator("reference", "It is 14", "4")[0]
    assert not score_with_validator("reference", "anything", "")[0]


@pytest.mark.parametrize(
    "scorer, output, ok",
    [
        ("python", "```python\ndef f():\n    return 1\n```", True),
        ("python", "def f(:\n", False),
        ("json", '```json\n{"a": 1}\n```', True),
        ("json", "{a: 1}", False),
        ("markdown", "```py\nunclosed", False),
    ],
)
def test_validator_scorers(scorer, output, ok):
    assert score_with_validator(scorer, output, None)[0] is ok


def test_judge_verdict_ignores_thinking():
    assert parse_judge_verdict("<think>maybe FAIL</think>PASS\nfine") is True
    assert parse_judge_verdict("FAIL: wrong") is False
    assert parse_judge_verdict("no idea") is None


def test_run_benchmark_scores_candidate_against_baseline():
    answers = {
        "candidate": {"2+2?": "4", "Capital of France?": "Paris", "3*3?": "6"},
        "baseline": {"2+2?": "5", "Capital of France?": "Paris", "3*3?": "6"},
    }

    def chat(target, messages):
        return answers[target][messages[-1]["content"]]

    result = run_benchmark(
        {
            "rows": ROWS,
            "prompt_column": "question",
            "reference_column": "answer",
            "scorer": "reference",
            "with_baseline": True,
        },
        chat = chat,
    )
    assert result["candidate"]["score"] == pytest.approx(2 / 3)
    assert result["baseline"]["score"] == pytest.approx(1 / 3)
    assert result["delta"] == pytest.approx(1 / 3)
    assert result["failed_total"] == 1
    failed = result["failed_examples"][0]
    assert failed["prompt"] == "3*3?" and failed["output"] == "6" and failed["reference"] == "9"


def test_run_benchmark_judge_and_row_errors():
    def chat(target, messages):
        if target == "judge":
            return "PASS" if "Paris" in messages[-1]["content"].split("Answer to grade:")[1] else "FAIL"
        if messages[-1]["content"] == "3*3?":
            raise RuntimeError("HTTP 500: boom")
        return "Paris"

    result = run_benchmark(
        {"rows": ROWS, "prompt_column": "question", "scorer": "judge", "rubric": "Says Paris."},
        chat = chat,
    )
    assert result["candidate"] == {"total": 3, "scored": 2, "passed": 2, "errors": 1, "score": 1.0}
    assert "baseline" not in result
    assert result["failed_examples"][0]["reason"] == "HTTP 500: boom"


def test_run_benchmark_stops_on_cancel():
    calls = []
    result = run_benchmark(
        {"rows": ROWS, "prompt_column": "question", "reference_column": "answer", "scorer": "reference"},
        chat = lambda target, messages: calls.append(1) or "4",
        is_cancelled = lambda: len(calls) >= 1,
    )
    assert result["cancelled"] is True
    assert result["candidate"]["total"] == 1


def _payload(**overrides):
    body = {
        "rows": ROWS,
        "prompt_column": "question",
        "reference_column": "answer",
        "scorer": "judge",
        "judge_alias": "local_model",
        "model_alias": "local_model",
        "baseline": {"mode": "adapter_off"},
        "model_providers": [{"name": "local", "is_local": True}],
        "model_configs": [{"alias": "local_model", "model": "unsloth/Qwen3-0.6B", "provider": "local"}],
    }
    body.update(overrides)
    return route.BenchmarkRequest.model_validate(body)


def _fake_inject(recipe, request, expect_gen = None):
    for provider in recipe["model_providers"]:
        if provider.pop("is_local", None):
            provider.update(endpoint = "http://127.0.0.1:8888/v1", api_key = "sk-unsloth-test")
    return 7


def test_adapter_off_baseline_toggles_lora_on_one_loaded_model(monkeypatch):
    seen = {}
    revoked = []
    monkeypatch.setattr(route, "_inject_local_providers", _fake_inject)
    monkeypatch.setattr(route, "_revoke_internal_api_key_safe", revoked.append)
    monkeypatch.setattr(route, "managed_account", lambda: False)

    def fake_run(spec, *, chat, on_progress, is_cancelled):
        seen["spec"] = spec
        return {"cancelled": False, "candidate": {}, "failed_examples": [], "failed_total": 0}

    monkeypatch.setattr(route, "run_benchmark", fake_run)
    route._jobs.clear()
    job_id = route.start_benchmark(_payload(), SimpleNamespace(), (None, False))["job_id"]
    for _ in range(100):
        if route._jobs[job_id]["status"] != "running":
            break
        time.sleep(0.01)
    assert route._jobs[job_id]["status"] == "completed"
    assert seen["spec"]["with_baseline"] is True
    assert revoked == [7]


def test_resolve_targets_sets_use_adapter_per_side(monkeypatch):
    captured = {}
    monkeypatch.setattr(route, "_inject_local_providers", _fake_inject)
    monkeypatch.setattr(route, "managed_account", lambda: False)

    class _Thread:
        def __init__(self, target, args, **kwargs):
            captured["targets"] = args[2]

        def start(self):
            pass

    monkeypatch.setattr(route.threading, "Thread", _Thread)
    route._jobs.clear()
    route.start_benchmark(_payload(), SimpleNamespace(), (None, False))
    targets = captured["targets"]
    assert targets["candidate"]["body"]["use_adapter"] is True
    assert targets["baseline"]["body"]["use_adapter"] is False
    assert targets["judge"]["body"]["use_adapter"] is False
    assert targets["candidate"]["url"] == "http://127.0.0.1:8888/v1/chat/completions"
    assert targets["candidate"]["headers"]["Authorization"] == "Bearer sk-unsloth-test"
    assert targets["candidate"]["body"]["temperature"] == 0
    route._jobs.clear()


def test_judge_scorer_needs_a_judge_model():
    with pytest.raises(HTTPException) as exc:
        route.start_benchmark(_payload(judge_alias = None), SimpleNamespace(), (None, False))
    assert exc.value.status_code == 400


def test_managed_account_cannot_reach_external_providers(monkeypatch):
    monkeypatch.setattr(route, "managed_account", lambda: True)
    payload = _payload(
        scorer = "reference",
        baseline = {"mode": "none"},
        model_alias = "remote",
        model_providers = [{"name": "openai", "endpoint": "https://example.com/v1"}],
        model_configs = [{"alias": "remote", "model": "gpt", "provider": "openai"}],
    )
    route._jobs.clear()
    with pytest.raises(HTTPException) as exc:
        route.start_benchmark(payload, SimpleNamespace(), (None, False))
    assert exc.value.status_code == 403
