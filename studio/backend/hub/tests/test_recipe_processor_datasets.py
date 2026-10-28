# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A data recipe's schema transform writes its reshaped rows to processors-files/<name>, beside
the untransformed parquet-files. The local dataset list only offered parquet-files, so the
reshaped output (often the training format itself) never reached the training picker."""

from hub.services.datasets import local


def _recipe(
    root,
    name,
    *,
    processors = (),
):
    parquet_dir = root / name / "parquet-files"
    parquet_dir.mkdir(parents = True)
    (parquet_dir / "batch_00000.parquet").write_bytes(b"parquet")
    for processor, files in processors:
        processor_dir = root / name / "processors-files" / processor
        processor_dir.mkdir(parents = True)
        for file_name in files:
            (processor_dir / file_name).write_bytes(b"parquet")


def _items(monkeypatch, tmp_path):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setattr(local, "LOCAL_DATASETS_ROOT", tmp_path / "recipes")
    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", upload_root)
    return {item.id: item for item in local.list_local_datasets_response().datasets}


def test_schema_transform_output_is_a_recipe_dataset(monkeypatch, tmp_path):
    root = tmp_path / "recipes"
    _recipe(root, "recipe_chat", processors = [("to_messages", ["batch_00000.parquet"])])

    items = _items(monkeypatch, tmp_path)

    item = items["recipe_chat:to_messages"]
    assert item.source == "recipe"
    assert item.label == "recipe_chat (to_messages)"
    assert item.path == str((root / "recipe_chat" / "processors-files" / "to_messages").resolve())
    assert items["recipe_chat"].path == str((root / "recipe_chat" / "parquet-files").resolve())


def test_processor_folder_without_parquet_is_not_listed(monkeypatch, tmp_path):
    root = tmp_path / "recipes"
    _recipe(root, "recipe_chat", processors = [("to_messages", ["notes.txt"])])

    assert set(_items(monkeypatch, tmp_path)) == {"recipe_chat"}
