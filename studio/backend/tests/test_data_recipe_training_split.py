# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pathlib import Path

import pytest

from core.data_recipe.training_split import (
    RecipeTrainingSplitError,
    holdout_row_count,
    split_recipe_dataset,
)


def _artifact(tmp_path: Path, monkeypatch, batches: list[list[dict]]) -> Path:
    pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")
    dataset_path = tmp_path / "recipe_abc"
    parquet_dir = dataset_path / "parquet-files"
    parquet_dir.mkdir(parents = True)
    for index, rows in enumerate(batches):
        pd.DataFrame(rows).to_parquet(parquet_dir / f"batch_{index:05d}.parquet", index = False)
    monkeypatch.setattr(
        "core.data_recipe.training_split._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )
    return dataset_path


def _ids(path: str) -> list[int]:
    import pyarrow.parquet as pq

    return pq.read_table(path).column("id").to_pylist()


def test_holdout_row_count_keeps_at_least_one_row_each_side():
    assert holdout_row_count(100, 10) == 10
    assert holdout_row_count(5, 10) == 1
    assert holdout_row_count(2, 50) == 1
    assert holdout_row_count(1, 50) == 0
    assert holdout_row_count(100, 0) == 0


def test_zero_percent_trains_on_the_whole_run(tmp_path, monkeypatch):
    dataset_path = _artifact(tmp_path, monkeypatch, [[{"id": i} for i in range(7)]])
    split = split_recipe_dataset(artifact_path = "x", holdout_percent = 0, seed = 1)
    assert split.train_path == str(dataset_path / "parquet-files")
    assert split.holdout_path is None
    assert (split.train_rows, split.holdout_rows) == (7, 0)


def test_holdout_is_disjoint_complete_and_repeatable(tmp_path, monkeypatch):
    _artifact(
        tmp_path,
        monkeypatch,
        [[{"id": i} for i in range(25)], [{"id": i} for i in range(25, 40)]],
    )
    split = split_recipe_dataset(artifact_path = "x", holdout_percent = 20, seed = 3407)
    train, holdout = _ids(split.train_path), _ids(split.holdout_path)
    assert (split.train_rows, split.holdout_rows) == (32, 8)
    assert len(holdout) == 8
    assert not set(train) & set(holdout)
    assert sorted(train + holdout) == list(range(40))

    again = split_recipe_dataset(artifact_path = "x", holdout_percent = 20, seed = 3407)
    assert _ids(again.holdout_path) == holdout

    other = split_recipe_dataset(artifact_path = "x", holdout_percent = 20, seed = 7)
    assert other.holdout_path != split.holdout_path
    assert _ids(other.holdout_path) != holdout


def test_single_row_run_refuses_a_holdout(tmp_path, monkeypatch):
    _artifact(tmp_path, monkeypatch, [[{"id": 0}]])
    with pytest.raises(RecipeTrainingSplitError):
        split_recipe_dataset(artifact_path = "x", holdout_percent = 10, seed = 1)
