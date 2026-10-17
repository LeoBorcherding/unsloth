# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Publish runs the real data_designer client and card; only the Hub itself is faked."""

import json
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("data_designer.integrations.huggingface.client")

import huggingface_hub.repocard as repocard
from data_designer.integrations.huggingface import client as dd_client
from huggingface_hub import CommitOperationDelete
from huggingface_hub.errors import HfHubHTTPError

from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)
from core.data_recipe import huggingface as recipe_hf
from routes.data_recipe import jobs as data_recipe_jobs_routes

REPO = "org/dataset"


class FakeHub:
    """A dataset repo in memory. Writes touching ``fail_path`` fail with ``fail_status``."""

    def __init__(self, files = None, *, fail_path = None, fail_status = 403):
        self.files = dict(files or {})
        self.created = bool(files)
        self.fail_path = fail_path
        self.fail_status = fail_status
        self.tokens = []

    def _check(self, paths):
        if self.fail_path and any(p.startswith(self.fail_path) for p in paths):
            response = httpx.Response(
                self.fail_status, request = httpx.Request("POST", "https://hub.invalid/api")
            )
            raise HfHubHTTPError(f"{self.fail_status} Client Error", response = response)

    def repo_exists(self, *, repo_id, repo_type):
        return self.created

    def create_repo(self, *, repo_id, repo_type, exist_ok, private):
        self.created = True

    def list_repo_files(self, *, repo_id, repo_type):
        return list(self.files)

    def create_commit(self, *, repo_id, repo_type, operations, **kwargs):
        self._check([op.path_in_repo for op in operations])
        files = dict(self.files)
        for op in operations:
            if isinstance(op, CommitOperationDelete):
                files.pop(op.path_in_repo, None)
            else:
                with op.as_file() as fh:
                    files[op.path_in_repo] = fh.read()
        self.files = files

    def upload_folder(self, *, repo_id, folder_path, path_in_repo, **kwargs):
        from pathlib import Path

        self._check([path_in_repo + "/"])
        for path in Path(folder_path).rglob("*"):
            if path.is_file():
                rel = path.relative_to(folder_path).as_posix()
                self.files[f"{path_in_repo}/{rel}"] = path.read_bytes()

    def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
        from pathlib import Path

        self._check([path_in_repo])
        self.files[path_in_repo] = Path(path_or_fileobj).read_bytes()


@pytest.fixture
def hub(monkeypatch):
    state = {"hub": FakeHub()}

    def api(token = None):
        state["hub"].tokens.append(token)
        return state["hub"]

    monkeypatch.setattr(dd_client, "HfApi", api)
    # The card's own push validates against huggingface.co and uploads through the module function.
    monkeypatch.setattr(repocard.RepoCard, "validate", lambda self, repo_type = None: None)
    monkeypatch.setattr(
        repocard, "upload_file", lambda **kwargs: state["hub"].upload_file(**kwargs)
    )

    def no_network(*args, **kwargs):
        raise AssertionError("no real Hub call")

    monkeypatch.setattr(repocard, "get_session", no_network)
    return state


def _run(root, name, *, shards, images = 0, processors = (), windows_paths = False):
    run = root / name
    (run / "parquet-files").mkdir(parents = True)
    sep = "\\" if windows_paths else "/"
    paths = []
    for i in range(shards):
        (run / "parquet-files" / f"batch_{i:05d}.parquet").write_bytes(f"{name}-{i}".encode())
        paths.append(f"parquet-files{sep}batch_{i:05d}.parquet")
    for i in range(images):
        (run / "images").mkdir(exist_ok = True)
        (run / "images" / f"img{i}.png").write_bytes(b"png")
    processor_paths = {}
    for proc in processors:
        (run / "processors-files" / proc).mkdir(parents = True)
        (run / "processors-files" / proc / "batch_00000.parquet").write_bytes(b"p")
        processor_paths[proc] = [f"processors-files{sep}{proc}{sep}batch_00000.parquet"]
    metadata = {
        "target_num_records": shards,
        "schema": {"q": "string"},
        "column_statistics": [],
        "file_paths": {"parquet-files": paths},
    }
    if processor_paths:
        metadata["file_paths"]["processor-files"] = processor_paths
    (run / "metadata.json").write_text(json.dumps(metadata), encoding = "utf-8")
    (run / "builder_config.json").write_text(
        json.dumps({"data_designer": {"columns": []}}), encoding = "utf-8"
    )
    return run


def _publish(monkeypatch, run, token = "hf_publish"):
    monkeypatch.setattr(recipe_hf, "_resolve_recipe_artifact_path", lambda _: run)
    return recipe_hf.publish_recipe_dataset(
        artifact_path = str(run),
        repo_id = REPO,
        description = "d",
        hf_token = token,
    )


def test_republish_leaves_only_the_new_runs_files(monkeypatch, tmp_path, hub):
    _publish(monkeypatch, _run(tmp_path, "a", shards = 3, images = 1, processors = ["p1"]))
    hub["hub"].files["LICENSE"] = b"kept"
    _publish(monkeypatch, _run(tmp_path, "b", shards = 1))

    assert sorted(hub["hub"].files) == [
        "LICENSE",
        "README.md",
        "builder_config.json",
        "data/batch_00000.parquet",
        "metadata.json",
    ]
    assert hub["hub"].files["data/batch_00000.parquet"] == b"b-0"


def test_a_failed_data_upload_leaves_no_half_made_repo(monkeypatch, tmp_path, hub):
    hub["hub"] = FakeHub(fail_path = "data/", fail_status = 403)
    with pytest.raises(recipe_hf.RecipeDatasetPublishError) as caught:
        _publish(monkeypatch, _run(tmp_path, "a", shards = 1))

    assert hub["hub"].files == {}
    assert "was created on the Hub, but nothing from this run is in it" in str(caught.value)


@pytest.mark.parametrize(
    "status,expected", [(401, "rejected the token"), (403, "can't write"), (429, "rate limiting")]
)
def test_a_refused_card_push_is_a_curated_error(monkeypatch, tmp_path, hub, status, expected):
    hub["hub"] = FakeHub(
        {"README.md": b"old", "data/batch_00000.parquet": b"old"},
        fail_path = "README.md",
        fail_status = status,
    )
    with pytest.raises(recipe_hf.RecipeDatasetPublishError) as caught:
        _publish(monkeypatch, _run(tmp_path, "a", shards = 1))

    assert expected in str(caught.value)
    assert caught.value.status_code == (429 if status == 429 else 400)
    assert hub["hub"].files == {"README.md": b"old", "data/batch_00000.parquet": b"old"}


def test_published_metadata_uses_hub_paths_on_windows(monkeypatch, tmp_path, hub):
    run = _run(tmp_path, "a", shards = 2, processors = ["p1"], windows_paths = True)
    _publish(monkeypatch, run)

    metadata = json.loads(hub["hub"].files["metadata.json"])
    assert metadata["file_paths"] == {
        "data": ["data/batch_00000.parquet", "data/batch_00001.parquet"],
        "processor-files": {"p1": ["p1/batch_00000.parquet"]},
    }


@pytest.mark.parametrize("name", ["data", "images"])
def test_a_processor_named_like_a_dataset_folder_is_refused(monkeypatch, tmp_path, hub, name):
    with pytest.raises(recipe_hf.RecipeDatasetPublishError, match = "Rename the processor"):
        _publish(monkeypatch, _run(tmp_path, "a", shards = 1, processors = [name]))

    assert hub["hub"].created is False
    assert hub["hub"].files == {}


def _app():
    app = FastAPI()
    app.include_router(data_recipe_jobs_routes.router, prefix = "/api/data-recipe")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[get_current_credential] = lambda: ("alice", "cred-1")
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return app


def _completed_job(monkeypatch, artifact):
    mgr = MagicMock()
    mgr.get_status.return_value = {
        "status": "completed",
        "execution_type": "full",
        "artifact_path": str(artifact),
    }
    monkeypatch.setattr(data_recipe_jobs_routes, "get_job_manager", lambda: mgr)


@pytest.mark.parametrize("status", [401, 403, 429])
def test_the_route_answers_a_refused_card_push_with_a_4xx(monkeypatch, tmp_path, hub, status):
    run = _run(tmp_path, "a", shards = 1)
    _completed_job(monkeypatch, run)
    monkeypatch.setattr(recipe_hf, "_resolve_recipe_artifact_path", lambda _: run)
    hub["hub"] = FakeHub(fail_path = "README.md", fail_status = status)

    response = TestClient(_app()).post(
        "/api/data-recipe/jobs/job-1/publish",
        json = {"repo_id": REPO, "description": "d", "hf_token": "hf_publish"},
    )
    assert response.status_code == (429 if status == 429 else 400)
    assert f"({status})" in response.json()["detail"]


def test_a_managed_account_without_a_token_is_refused_before_any_hub_call(
    monkeypatch, tmp_path, hub
):
    _completed_job(monkeypatch, _run(tmp_path, "a", shards = 1))
    monkeypatch.setattr(data_recipe_jobs_routes, "managed_account", lambda: True, raising = False)
    published = []
    monkeypatch.setattr(
        data_recipe_jobs_routes, "publish_recipe_dataset", lambda **kw: published.append(kw)
    )

    response = TestClient(_app()).post(
        "/api/data-recipe/jobs/job-1/publish",
        json = {"repo_id": REPO, "description": "d", "hf_token": ""},
    )
    assert response.status_code == 403
    assert "your own Hugging Face token" in response.json()["detail"]
    assert published == []
