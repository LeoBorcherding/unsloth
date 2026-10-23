# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Carve a held-out split out of a recipe dataset before training on the rest."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.data_recipe.huggingface import _resolve_recipe_artifact_path
from utils.paths.path_utils import drop_appledouble_metadata

SPLITS_DIR_NAME = "splits"


class RecipeTrainingSplitError(ValueError):
    """Raised when a recipe dataset cannot be split for training."""


@dataclass(frozen = True)
class RecipeTrainingSplit:
    train_path: str
    holdout_path: str | None
    train_rows: int
    holdout_rows: int
    # Row positions in the run's parquet shards (sorted file order), so a scorer can name the rows.
    holdout_row_ids: tuple[int, ...] = ()


def holdout_row_count(total: int, holdout_percent: int) -> int:
    """Rows to hold out: rounded, but never all of them and at least one when asked for any."""
    if holdout_percent <= 0 or total <= 1:
        return 0
    return min(total - 1, max(1, round(total * holdout_percent / 100)))


def holdout_row_ids(total: int, holdout_percent: int, seed: int) -> list[int]:
    order = list(range(total))
    random.Random(seed).shuffle(order)
    return sorted(order[: holdout_row_count(total, holdout_percent)])


def split_recipe_dataset(
    *, artifact_path: str, holdout_percent: int, seed: int
) -> RecipeTrainingSplit:
    dataset_path = _resolve_recipe_artifact_path(artifact_path)
    parquet_dir = dataset_path / "parquet-files"
    shards = drop_appledouble_metadata(sorted(parquet_dir.glob("*.parquet")))
    if not shards:
        raise RecipeTrainingSplitError("This run has no dataset files to train on.")

    import pyarrow
    import pyarrow.parquet as pyarrow_parquet

    if holdout_percent <= 0:
        rows = sum(pyarrow_parquet.read_metadata(shard).num_rows for shard in shards)
        return RecipeTrainingSplit(str(parquet_dir), None, rows, 0)

    out_dir = dataset_path / SPLITS_DIR_NAME / f"holdout-{holdout_percent}-seed{seed}"
    train_file = out_dir / "train.parquet"
    holdout_file = out_dir / "holdout.parquet"
    if train_file.is_file() and holdout_file.is_file():
        train_rows = pyarrow_parquet.read_metadata(train_file).num_rows
        holdout_rows = pyarrow_parquet.read_metadata(holdout_file).num_rows
        return RecipeTrainingSplit(
            str(train_file),
            str(holdout_file),
            train_rows,
            holdout_rows,
            tuple(holdout_row_ids(train_rows + holdout_rows, holdout_percent, seed)),
        )

    # The worker already unified the batch schemas, so the shards concatenate.
    table = pyarrow.concat_tables([pyarrow_parquet.read_table(shard) for shard in shards])
    total = table.num_rows
    holdout_indices = holdout_row_ids(total, holdout_percent, seed)
    if not holdout_indices:
        raise RecipeTrainingSplitError(
            f"{total} row(s) is too few to hold any out. Generate more rows or set hold out to 0%."
        )
    held = set(holdout_indices)
    train_indices = [index for index in range(total) if index not in held]

    out_dir.mkdir(parents = True, exist_ok = True)
    # Written beside and renamed, so a crash never leaves a half file that the cache check above trusts.
    for path, indices in ((holdout_file, holdout_indices), (train_file, train_indices)):
        staged = path.with_name(path.name + ".partial")
        pyarrow_parquet.write_table(table.take(pyarrow.array(indices)), staged)
        os.replace(staged, path)
    return RecipeTrainingSplit(
        str(train_file),
        str(holdout_file),
        len(train_indices),
        len(holdout_indices),
        tuple(holdout_indices),
    )


PAIR_RULES = ("a_better", "winner", "scores")


@dataclass(frozen = True)
class PreferencePairs:
    """How two answer columns become DPO/ORPO's chosen and rejected."""

    answer_a: str
    answer_b: str
    rule: str = "a_better"
    winner_column: str | None = None
    score_a_column: str | None = None
    score_b_column: str | None = None


@dataclass(frozen = True)
class RecipePreferenceSplit:
    train_path: str
    kept: int
    dropped: int


def _winner_from_label(value: Any, pairs: PreferencePairs) -> str | None:
    words = re.sub(r"[^a-z0-9_]+", " ", str(value or "").lower()).split()
    if words[:1] in (["answer"], ["response"], ["option"]):
        words = words[1:]
    if not words:
        return None
    if words[0] in ("a", "1", pairs.answer_a.lower()):
        return "a"
    if words[0] in ("b", "2", pairs.answer_b.lower()):
        return "b"
    return None


def _score(value: Any) -> float | None:
    """A number, a numeric string, or a judge column's {"rubric": {"score": n, ...}}."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    if isinstance(value, dict):
        if "score" in value:
            return _score(value["score"])
        for nested in value.values():
            found = _score(nested) if isinstance(nested, dict) else None
            if found is not None:
                return found
    return None


def pair_winner(row: dict, pairs: PreferencePairs) -> str | None:
    """"a", "b", or None for a tie or a row the rule cannot read (dropped from training)."""
    if pairs.rule == "a_better":
        return "a"
    if pairs.rule == "winner":
        return _winner_from_label(row.get(pairs.winner_column or ""), pairs)
    score_a = _score(row.get(pairs.score_a_column or ""))
    score_b = _score(row.get(pairs.score_b_column or ""))
    if score_a is None or score_b is None or score_a == score_b:
        return None
    return "a" if score_a > score_b else "b"


def _pair_columns(pairs: PreferencePairs) -> list[str]:
    if pairs.rule not in PAIR_RULES:
        raise RecipeTrainingSplitError(f"Unknown pair rule: {pairs.rule}.")
    columns = [pairs.answer_a, pairs.answer_b]
    if pairs.rule == "winner":
        columns.append(pairs.winner_column or "")
    elif pairs.rule == "scores":
        columns += [pairs.score_a_column or "", pairs.score_b_column or ""]
    if not all(columns):
        raise RecipeTrainingSplitError("Pick every column the pair rule reads.")
    if pairs.answer_a == pairs.answer_b:
        raise RecipeTrainingSplitError("Answer A and answer B must be different columns.")
    return columns


def pair_recipe_split(
    split: RecipeTrainingSplit, *, artifact_path: str, seed: int, pairs: PreferencePairs
) -> RecipePreferenceSplit:
    """Adds chosen/rejected columns to the training rows; ties and unreadable rows are left out."""
    columns = _pair_columns(pairs)
    import pyarrow
    import pyarrow.parquet as pyarrow_parquet

    key = hashlib.sha1(json.dumps(asdict(pairs), sort_keys = True).encode()).hexdigest()[:12]
    train = split.train_path
    out_dir = (
        os.path.dirname(train)
        if split.holdout_path
        else str(
            _resolve_recipe_artifact_path(artifact_path) / SPLITS_DIR_NAME / f"holdout-0-seed{seed}"
        )
    )
    out_file = os.path.join(out_dir, f"pairs-{key}.parquet")
    if os.path.isfile(out_file):
        kept = pyarrow_parquet.read_metadata(out_file).num_rows
        return RecipePreferenceSplit(out_file, kept, split.train_rows - kept)

    if os.path.isdir(train):
        shards = drop_appledouble_metadata(sorted(Path(train).glob("*.parquet")))
        table = pyarrow.concat_tables([pyarrow_parquet.read_table(shard) for shard in shards])
    else:
        table = pyarrow_parquet.read_table(train)
    missing = [column for column in columns if column not in table.column_names]
    if missing:
        raise RecipeTrainingSplitError(f"Column {missing[0]} is not in this run's dataset.")

    keep, chosen, rejected = [], [], []
    for index, row in enumerate(table.select(list(dict.fromkeys(columns))).to_pylist()):
        winner = pair_winner(row, pairs)
        a, b = row.get(pairs.answer_a), row.get(pairs.answer_b)
        if winner is None or a is None or b is None:
            continue
        a, b = str(a), str(b)
        if not (a.strip() and b.strip()) or a.strip() == b.strip():
            continue
        keep.append(index)
        chosen.append(a if winner == "a" else b)
        rejected.append(b if winner == "a" else a)
    if not keep:
        raise RecipeTrainingSplitError(
            "No row has a clear winner, so there are no preference pairs to train on."
        )

    paired = table.take(pyarrow.array(keep))
    for name in ("chosen", "rejected"):
        if name in paired.column_names:
            paired = paired.drop_columns([name])
    paired = paired.append_column("chosen", pyarrow.array(chosen, pyarrow.string()))
    paired = paired.append_column("rejected", pyarrow.array(rejected, pyarrow.string()))
    os.makedirs(out_dir, exist_ok = True)
    staged = out_file + ".partial"
    pyarrow_parquet.write_table(paired, staged)
    os.replace(staged, out_file)
    return RecipePreferenceSplit(out_file, len(keep), table.num_rows - len(keep))


def read_holdout_rows(split: RecipeTrainingSplit, *, offset: int, limit: int) -> list[dict]:
    if not split.holdout_path:
        return []
    import pyarrow.parquet as pyarrow_parquet

    return pyarrow_parquet.read_table(split.holdout_path).slice(offset, limit).to_pylist()

