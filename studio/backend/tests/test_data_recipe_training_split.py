# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pathlib import Path

import pytest

from core.data_recipe.training_split import (
    RecipeTrainingSplitError,
    holdout_row_count,
    read_holdout_rows,
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
    assert split.holdout_row_ids == ()
    assert read_holdout_rows(split, offset = 0, limit = 5) == []


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


def test_holdout_row_ids_name_the_held_out_rows_in_file_order(tmp_path, monkeypatch):
    _artifact(
        tmp_path,
        monkeypatch,
        [[{"id": i, "q": f"q{i}"} for i in range(30)], [{"id": i, "q": f"q{i}"} for i in range(30, 50)]],
    )
    split = split_recipe_dataset(artifact_path = "x", holdout_percent = 10, seed = 3407)
    assert list(split.holdout_row_ids) == _ids(split.holdout_path)
    assert len(split.holdout_row_ids) == split.holdout_rows == 5

    cached = split_recipe_dataset(artifact_path = "x", holdout_percent = 10, seed = 3407)
    assert cached.holdout_row_ids == split.holdout_row_ids

    page = read_holdout_rows(split, offset = 1, limit = 2)
    assert [row["id"] for row in page] == list(split.holdout_row_ids[1:3])
    assert page[0]["q"] == f"q{split.holdout_row_ids[1]}"


def test_holdout_rows_route_returns_the_rows_and_their_ids(tmp_path, monkeypatch):
    from models.data_recipe import HoldoutRowsRequest
    from routes.data_recipe import jobs

    _artifact(tmp_path, monkeypatch, [[{"id": i} for i in range(20)]])

    class _Manager:
        def get_status(self, job_id):
            return {"status": "completed", "artifact_path": "x"}

    monkeypatch.setattr(jobs, "get_job_manager", lambda: _Manager())
    request = HoldoutRowsRequest(holdout_percent = 20, seed = 3407, offset = 0, limit = 3)
    page = jobs.read_job_holdout_rows("job-1", request)
    assert page.total == 4
    assert [row["id"] for row in page.rows] == page.row_ids
    split = jobs.create_job_training_split("job-1", request)
    assert split.holdout_row_ids[:3] == page.row_ids
