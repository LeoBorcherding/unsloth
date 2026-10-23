# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Quality pass over a finished run: duplicates, lengths, format and validator pass rates."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from typing import Any, Callable, Iterable

MAX_ROWS = 50_000
# Pairwise near-duplicate search is quadratic; past this, only normalized-equal text counts.
NEAR_DUP_MAX_ROWS = 1_000
NEAR_DUP_JACCARD = 0.8

DUPLICATE_WARN_RATE = 0.05
EMPTY_WARN_RATE = 0.02
FORMAT_WARN_RATE = 0.02
VALIDATOR_WARN_PASS_RATE = 0.9

# Chat-template control tokens a local GGUF can leak into its text when the template is wrong.
_LEAKED_TOKEN_RE = re.compile(
    r"<\|(?:im_start|im_end|eot_id|end_of_text|start_header_id|end_header_id|endoftext)\|>"
    r"|<(?:start|end)_of_turn>"
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, RecursionError):
            return None
    return value


def _validator_result(value: Any) -> bool | None:
    parsed = _parse_json(value)
    if isinstance(parsed, dict) and isinstance(parsed.get("is_valid"), bool):
        return parsed["is_valid"]
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys = True, ensure_ascii = False, default = str)
    except (TypeError, ValueError):
        return str(value)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list)):
        return len(value) == 0
    return False


def _structured_ok(value: Any) -> bool:
    return isinstance(_parse_json(value), (dict, list))


def _text_ok(value: Any) -> bool:
    return not (isinstance(value, str) and _LEAKED_TOKEN_RE.search(value))


def _column_types(analysis: dict[str, Any] | None) -> dict[str, str]:
    stats = analysis.get("column_statistics") if isinstance(analysis, dict) else None
    types: dict[str, str] = {}
    for item in stats if isinstance(stats, list) else []:
        if isinstance(item, dict) and isinstance(item.get("column_name"), str):
            types[item["column_name"]] = str(item.get("column_type") or "")
    return types


def _infer_type(values: Iterable[Any]) -> str:
    for value in values:
        if _is_empty(value):
            continue
        if _validator_result(value) is not None:
            return "validation"
        if isinstance(value, (dict, list)):
            return "structured"
        return "text"
    return "text"


def _shingles(normalized: str) -> frozenset[str]:
    if len(normalized) <= 5:
        return frozenset([normalized])
    return frozenset(normalized[i : i + 5] for i in range(len(normalized) - 4))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _count_duplicates(texts: list[str]) -> tuple[int, int, bool]:
    exact = 0
    near = 0
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    kept: list[frozenset[str]] = []
    pairwise = len(texts) <= NEAR_DUP_MAX_ROWS
    for text in texts:
        digest = hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()
        if digest in seen_exact:
            exact += 1
            continue
        seen_exact.add(digest)
        normalized = " ".join(_WORD_RE.findall(text.lower()))
        if normalized in seen_normalized:
            near += 1
            continue
        seen_normalized.add(normalized)
        if not pairwise:
            continue
        shingles = _shingles(normalized)
        size = len(shingles)
        is_near = any(
            min(size, len(other)) >= NEAR_DUP_JACCARD * max(size, len(other))
            and _jaccard(shingles, other) >= NEAR_DUP_JACCARD
            for other in kept
        )
        if is_near:
            near += 1
            continue
        kept.append(shingles)
    return exact, near, pairwise


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def compute_quality(
    rows: list[dict[str, Any]],
    analysis: dict[str, Any] | None = None,
    *,
    total_rows: int | None = None,
) -> dict[str, Any]:
    """Score a run's rows. ``total_rows`` is the run's size when ``rows`` is a capped prefix."""
    row_count = len(rows)
    total = max(total_rows or 0, row_count)
    side_effects = analysis.get("side_effect_column_names") if isinstance(analysis, dict) else None
    skip = set(side_effects) if isinstance(side_effects, list) else set()

    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names and key not in skip:
                names.append(key)

    declared = _column_types(analysis)
    types: dict[str, str] = {}
    for name in names:
        column_type = declared.get(name) or _infer_type(row.get(name) for row in rows)
        types[name] = column_type

    validators = [n for n in names if types[n] == "validation"]
    llm_columns = [n for n in names if types[n].startswith("llm")]
    content_columns = llm_columns or [n for n in names if n not in validators]

    columns: list[dict[str, Any]] = []
    empty_cells = 0
    format_failures = 0
    for name in content_columns:
        values = [row.get(name) for row in rows]
        column_type = types[name]
        structured = column_type in {"llm-structured", "llm-judge", "structured"}
        checker: Callable[[Any], bool] = _structured_ok if structured else _text_ok
        empty = sum(1 for value in values if _is_empty(value))
        failures = sum(1 for value in values if not _is_empty(value) and not checker(value))
        lengths = [len(_as_text(value)) for value in values if not _is_empty(value)]
        empty_cells += empty
        format_failures += failures
        columns.append(
            {
                "name": name,
                "type": column_type,
                "empty": empty,
                "format_failures": failures,
                "format_check": "json" if structured else "leaked_tokens",
                "min_chars": min(lengths) if lengths else None,
                "median_chars": int(statistics.median(lengths)) if lengths else None,
                "max_chars": max(lengths) if lengths else None,
            }
        )

    validator_stats: list[dict[str, Any]] = []
    for name in validators:
        results = [_validator_result(row.get(name)) for row in rows]
        checked = sum(1 for r in results if r is not None)
        passed = sum(1 for r in results if r is True)
        validator_stats.append(
            {
                "name": name,
                "checked": checked,
                "passed": passed,
                "pass_rate": (passed / checked) if checked else None,
            }
        )

    texts = [
        "␞".join(_as_text(row.get(name)) for name in content_columns) for row in rows
    ]
    exact, near, near_checked = _count_duplicates(texts)

    warnings: list[str] = []
    if row_count == 0:
        warnings.append("The run produced no rows.")
    else:
        dup_rate = (exact + near) / row_count
        if dup_rate >= DUPLICATE_WARN_RATE:
            warnings.append(
                f"{exact + near} of {row_count} rows are duplicates or near-duplicates ({_pct(dup_rate)})."
            )
        cells = row_count * max(len(content_columns), 1)
        if empty_cells and empty_cells / cells >= EMPTY_WARN_RATE:
            warnings.append(f"{empty_cells} generated values are empty ({_pct(empty_cells / cells)}).")
        if format_failures and format_failures / cells >= FORMAT_WARN_RATE:
            warnings.append(
                f"{format_failures} values fail the format check ({_pct(format_failures / cells)})."
            )
        for stat in validator_stats:
            rate = stat["pass_rate"]
            if rate is not None and rate < VALIDATOR_WARN_PASS_RATE:
                warnings.append(f"Validator {stat['name']} passes only {_pct(rate)} of rows.")

    return {
        "rows_checked": row_count,
        "rows_total": total,
        "truncated": total > row_count,
        "content_columns": content_columns,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "near_duplicates_checked": near_checked,
        "columns": columns,
        "validators": validator_stats,
        "warnings": warnings,
        "verdict": "warn" if warnings else "ok",
    }
