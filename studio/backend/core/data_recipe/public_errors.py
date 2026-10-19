# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Client text for data recipe failures: the engine's own text carries SQL, pydantic dumps and server paths."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_DUCKDB_MISSING_TABLE = re.compile(r"Table with name (?P<name>\S+) does not exist")
_EMPTY_DATASET = ("all records may have been dropped", "all records were dropped")


def redact_server_paths(text: str) -> str:
    from utils.native_path_leases import redact_native_paths
    from utils.paths import studio_root

    roots: list[tuple[str, str]] = []
    for root, label in ((studio_root(), "<studio>"), (Path(sys.prefix), "<env>"), (Path.home(), "~")):
        value = str(root)
        if len(value) > 3:
            roots += [(value, label), (value.replace("\\", "/"), label)]
    for value, label in sorted(set(roots), key = lambda item: len(item[0]), reverse = True):
        text = text.replace(value, label)
    return redact_native_paths(text)


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def public_error_text(exc: BaseException) -> str:
    from pydantic import ValidationError

    for error in _chain(exc):
        if isinstance(error, ValidationError):
            fields = [
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors()[:3]
            ]
            return redact_server_paths("Invalid recipe settings. " + "; ".join(fields))
        missing = _DUCKDB_MISSING_TABLE.search(str(error))
        if missing:
            return redact_server_paths(
                f"Seed dataset {missing['name']} wasn't found. "
                "Check the dataset path in the source data block."
            )
    text = str(exc).strip()
    cause = exc.__cause__
    # The engine wraps as "Error generating dataset: {cause}", which is bare when the cause has no message.
    if cause is not None and not str(cause).strip() and text.endswith(":"):
        text = f"{text} {type(cause).__name__}"
    return redact_server_paths(text or type(exc).__name__)


def all_rows_failed_error(error: str | None, record_failures: dict[str, dict[str, int]]) -> str | None:
    """The run's own counts instead of the engine's "check the warnings above" when every row dropped."""
    if not error or not record_failures or not any(marker in error for marker in _EMPTY_DATASET):
        return None
    parts = []
    for column, kinds in record_failures.items():
        reasons = ", ".join(f"{count} {kind}" for kind, count in kinds.items())
        parts.append(f"{sum(kinds.values())} in column '{column}' ({reasons})")
    message = f"Every row failed: {'; '.join(parts)}. Nothing was written."
    if any("timeout" in kinds for kinds in record_failures.values()):
        message += " Raise Timeout on that column's model, or ask for shorter answers."
    return message
