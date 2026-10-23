# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Held-out recipe split scoring for the Benchmark card: answer each row with the model under
test (and optionally a baseline), score the answer with a validator or an LLM judge."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Callable, Optional

from core.data_recipe.jsonable import to_jsonable
from core.data_recipe.text_format_validators import _validate_text_format

VALIDATOR_SCORERS = ("reference", "json", "markdown", "python")
SCORERS = (*VALIDATOR_SCORERS, "judge")
MAX_BENCHMARK_ROWS = 200
MAX_FAILED_EXAMPLES = 50

DEFAULT_JUDGE_RUBRIC = (
    "The answer is correct, complete and follows the instruction. "
    "If a reference answer is given, the answer must agree with it."
)

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)
_VERDICT_RE = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

ChatFn = Callable[[str, list[dict[str, str]]], str]


class BenchmarkHoldoutError(ValueError):
    """Raised when a Train card's held-out split can't be read."""


def read_holdout_rows(holdout_path: str, max_rows: int) -> list[dict[str, Any]]:
    """First ``max_rows`` rows of a Train card's ``splits/holdout-<pct>-seed<seed>/holdout.parquet``.
    The split is already a seeded shuffle, so its head is a fair sample."""
    from core.data_recipe.huggingface import RecipeDatasetPublishError, _resolve_recipe_artifact_path

    path = Path(holdout_path).expanduser().resolve(strict = False)
    if path.name != "holdout.parquet" or path.parent.parent.name != "splits":
        raise BenchmarkHoldoutError("That is not a Train card's held-out split.")
    try:
        _resolve_recipe_artifact_path(str(path.parent.parent.parent))
    except RecipeDatasetPublishError as exc:
        raise BenchmarkHoldoutError(str(exc)) from exc
    if not path.is_file():
        raise BenchmarkHoldoutError("The held-out split is gone. Train again to recreate it.")

    import pyarrow.parquet as pyarrow_parquet

    table = pyarrow_parquet.read_table(path)
    return [to_jsonable(row) for row in table.slice(0, max(1, max_rows)).to_pylist()]


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _unfence(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def score_with_validator(scorer: str, output: str, reference: str | None) -> tuple[bool, str]:
    """(passed, reason) for one answer under a validator scorer."""
    if scorer == "reference":
        expected = _normalize(reference or "")
        if not expected:
            return False, "Row has no reference answer."
        got = _normalize(output)
        if got == expected or f" {expected} " in f" {got} ":
            return True, ""
        return False, "Answer does not contain the reference."
    if scorer == "python":
        code = _unfence(output)
        if not code.strip():
            return False, "Answer is empty."
        try:
            ast.parse(code)
        except (SyntaxError, ValueError) as exc:
            return False, f"Python syntax error: {exc}"
        return True, ""
    if scorer in ("json", "markdown"):
        value = _unfence(output) if scorer == "json" else output
        result = _validate_text_format(value = value, format_kind = scorer)
        return bool(result["is_valid"]), str(result.get("error_message") or "")
    raise ValueError(f"unknown scorer: {scorer}")


def build_judge_messages(
    rubric: str, prompt: str, output: str, reference: str | None
) -> list[dict[str, str]]:
    parts = [
        f"Criteria: {rubric.strip() or DEFAULT_JUDGE_RUBRIC}",
        f"Instruction:\n{prompt}",
    ]
    if reference:
        parts.append(f"Reference answer:\n{reference}")
    parts.append(f"Answer to grade:\n{output}")
    parts.append("Reply with PASS or FAIL on the first line, then one short sentence why.")
    return [
        {
            "role": "system",
            "content": "You grade answers strictly against the criteria. You only reply PASS or FAIL plus one sentence.",
        },
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_judge_verdict(text: str) -> Optional[bool]:
    match = _VERDICT_RE.search(strip_thinking(text))
    if match is None:
        return None
    return match.group(1).upper() == "PASS"


def _cell(row: dict[str, Any], column: str | None) -> str:
    if not column:
        return ""
    value = row.get(column)
    return "" if value is None else str(value)


def _score_answer(
    spec: dict[str, Any], chat: ChatFn, prompt: str, output: str, reference: str
) -> tuple[bool, str]:
    scorer = spec["scorer"]
    if scorer != "judge":
        return score_with_validator(scorer, output, reference)
    verdict_text = chat(
        "judge", build_judge_messages(spec.get("rubric") or "", prompt, output, reference or None)
    )
    verdict = parse_judge_verdict(verdict_text)
    if verdict is None:
        return False, "Judge reply had no PASS or FAIL."
    reason = strip_thinking(verdict_text)
    return verdict, "" if verdict else reason[:500]


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in results if r.get("error") is None]
    passed = sum(1 for r in scored if r["passed"])
    return {
        "total": len(results),
        "scored": len(scored),
        "passed": passed,
        "errors": len(results) - len(scored),
        "score": (passed / len(scored)) if scored else None,
    }


def run_benchmark(
    spec: dict[str, Any],
    *,
    chat: ChatFn,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Score ``spec["rows"]``. ``chat(target, messages)`` answers for target in
    {"candidate", "baseline", "judge"}; a raised exception marks only that row as errored."""
    rows: list[dict[str, Any]] = spec["rows"]
    prompt_column: str = spec["prompt_column"]
    reference_column: str | None = spec.get("reference_column") or None
    system_prompt: str = spec.get("system_prompt") or ""
    targets = ["candidate"] + (["baseline"] if spec.get("with_baseline") else [])
    per_target: dict[str, list[dict[str, Any]]] = {t: [] for t in targets}
    total = len(rows) * len(targets)
    done = 0
    on_progress(done, total)
    cancelled = False
    for index, row in enumerate(rows):
        prompt = _cell(row, prompt_column)
        reference = _cell(row, reference_column)
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": prompt}
        ]
        for target in targets:
            if is_cancelled():
                cancelled = True
                break
            result: dict[str, Any] = {"index": index, "passed": False, "error": None}
            try:
                output = strip_thinking(chat(target, messages))
                passed, reason = _score_answer(spec, chat, prompt, output, reference)
                result.update(output = output, passed = passed, reason = reason)
            except Exception as exc:
                result["error"] = str(exc)[:500] or type(exc).__name__
            per_target[target].append(result)
            done += 1
            on_progress(done, total)
        if cancelled:
            break

    candidate = per_target["candidate"]
    failed = []
    for result in candidate:
        if result["passed"]:
            continue
        row = rows[result["index"]]
        failed.append(
            {
                "prompt": _cell(row, prompt_column),
                "reference": _cell(row, reference_column) or None,
                "output": result.get("output"),
                "reason": result.get("error") or result.get("reason") or "",
                "row": row,
            }
        )
    summary = {"candidate": _summary(candidate)}
    if "baseline" in per_target:
        summary["baseline"] = _summary(per_target["baseline"])
        a, b = summary["candidate"]["score"], summary["baseline"]["score"]
        summary["delta"] = (a - b) if a is not None and b is not None else None
    return {
        **summary,
        "cancelled": cancelled,
        "failed_examples": failed[:MAX_FAILED_EXAMPLES],
        "failed_total": len(failed),
    }
