# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Carve a held-out split out of a recipe dataset before training on the rest."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

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


def read_holdout_rows(split: RecipeTrainingSplit, *, offset: int, limit: int) -> list[dict]:
    if not split.holdout_path:
        return []
    import pyarrow.parquet as pyarrow_parquet

    return pyarrow_parquet.read_table(split.holdout_path).slice(offset, limit).to_pylist()

