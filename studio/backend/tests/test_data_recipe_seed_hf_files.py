# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _load_seed_route(monkeypatch, tmp_path):
    pytest.importorskip("multipart")
    backend_root = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(backend_root))
    spec = importlib.util.spec_from_file_location(
        "seed_hf_files_under_test", backend_root / "routes" / "data_recipe" / "seed.py"
    )
    seed_route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed_route)
    seed_route.UNSTRUCTURED_UPLOAD_ROOT = tmp_path / "unstructured-uploads"
    return seed_route


def _inspect(monkeypatch, tmp_path, repo_files):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    monkeypatch.setattr(seed_route, "_list_hf_repo_files", lambda **kwargs: repo_files)
    monkeypatch.setattr(seed_route, "_list_hf_dataset_configs", lambda **kwargs: [])
    monkeypatch.setattr(seed_route, "refuse_unauthorized_dataset_preview", lambda *a, **k: None)
    loads = []

    def fake_preview(*, load_dataset_fn, load_kwargs, preview_size):
        # Stands in for `datasets`, which reads a whole JSON array before yielding a row.
        loads.append(load_kwargs)
        return [{"text": "from load_dataset"}]

    monkeypatch.setattr(seed_route, "_load_preview_rows", fake_preview)

    def inspect():
        return seed_route.inspect_seed_dataset(
            SimpleNamespace(
                dataset_name = "org/repo", split = "train", subset = None, hf_token = None, preview_size = 3
            ),
            allow_ambient_token = False,
        )

    return seed_route, inspect, loads


def test_a_compressed_only_repo_is_a_plain_422_not_a_parquet_glob(monkeypatch, tmp_path):
    _seed_route, inspect, loads = _inspect(
        monkeypatch,
        tmp_path,
        [".gitattributes", "README.md", "train.jsonl.gz", "validation.jsonl.gz"],
    )

    with pytest.raises(HTTPException) as refused:
        inspect()

    assert refused.value.status_code == 422
    assert ".jsonl.gz" in refused.value.detail
    assert loads == []


class _CountingFile(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.size = len(data)
        self.bytes_read = 0

    def read(self, n = -1):
        chunk = super().read(n)
        self.bytes_read += len(chunk)
        return chunk


def test_a_large_json_array_previews_from_its_first_rows(monkeypatch, tmp_path):
    seed_route, inspect, loads = _inspect(monkeypatch, tmp_path, ["README.md", "data.json"])
    rows = [{"id": i, "text": "x" * 200, "note": "é"} for i in range(40_000)]
    handle = _CountingFile(json.dumps(rows, ensure_ascii = False, indent = 1).encode("utf-8"))
    monkeypatch.setattr(seed_route, "JSON_HEAD_PREVIEW_MIN_BYTES", 1024, raising = False)
    monkeypatch.setattr(seed_route, "_open_hf_file", lambda **kwargs: handle, raising = False)

    response = inspect()

    assert loads == []
    assert [row["id"] for row in response.preview_rows] == [0, 1, 2]
    assert response.preview_rows[0]["note"] == "é"
    assert response.columns == ["id", "text", "note"]
    assert handle.bytes_read < handle.size / 4


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b'  [ {"a": 1} , {"a": 2},\n{"a": 3}, {"a": 4}]', [{"a": 1}, {"a": 2}, {"a": 3}]),
        (b'[{"a": 1}]', [{"a": 1}]),
        (b'{"data": [{"a": 1}]}', None),
        (b"[1, 2, 3]", None),
        (b'[{"a": ', None),
        # Malformed arrays fall back to the loader, which rejects them as the run would.
        (b'[{"a": 1} {"a": 2}]', None),
        (b'[, {"a": 1}]', None),
        (b'[{"a": 1},, {"a": 2}]', None),
        (b'[{"a": 1},]', None),
    ],
)
def test_json_head_reads_only_an_array_of_objects(monkeypatch, tmp_path, payload, expected):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    monkeypatch.setattr(seed_route, "_JSON_HEAD_CHUNK_BYTES", 5, raising = False)
    assert seed_route._read_json_array_head(io.BytesIO(payload), 3) == expected


def test_a_small_json_file_still_goes_through_the_loader(monkeypatch, tmp_path):
    seed_route, inspect, loads = _inspect(monkeypatch, tmp_path, ["README.md", "data.json"])
    handle = _CountingFile(b'[{"text": "a"}]')
    monkeypatch.setattr(seed_route, "_open_hf_file", lambda **kwargs: handle, raising = False)

    response = inspect()

    assert loads and loads[0]["data_files"] == ["data.json"]
    assert response.preview_rows == [{"text": "from load_dataset"}]
