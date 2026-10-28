# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The GitHub seed plugin against a fake GitHub client: no network, no real token."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SRC = BACKEND_ROOT / "plugins" / "data-designer-github-repo-seed" / "src"
_PLUGIN_MODULES = {"data_designer_github_repo_seed", "gh_client", "scraper", "queries", "state_store"}
_FAKE_TOKEN = "test-token-not-real"


class FakeGitHubClient:
    """Answers like GitHub's GraphQL API: `first: N` caps a connection and nothing paginates."""

    calls: list[str] = []
    graphql_remaining = 5000

    def __init__(self, token, token_source):
        assert token == _FAKE_TOKEN

    def graphql(self, query, variables):
        type(self).calls.append(query)
        if "RepoMeta" in query:
            return {"data": {"repository": {"name": variables["name"], "defaultBranchRef": None}}}
        field = "pullRequests" if "pullRequests(" in query else "issues"
        nodes = []
        for number in range(1, variables["first"] + 1):
            node = {"number": number, "title": f"t{number}", "author": {"login": "a"}}
            if "comments(first: 30)" in query:
                node["comments"] = {
                    "pageInfo": {"hasNextPage": True, "endCursor": "c"},
                    "nodes": [{"body": f"c{i}", "author": {"login": "b"}} for i in range(30)],
                }
            nodes.append(node)
        page = {"pageInfo": {"hasNextPage": False, "endCursor": "e"}, "nodes": nodes}
        return {"data": {"repository": {field: page}}}


def _drop_plugin_modules():
    for name in list(sys.modules):
        if name.split(".")[0] in _PLUGIN_MODULES:
            del sys.modules[name]


@pytest.fixture
def plugin(monkeypatch):
    pytest.importorskip("data_designer")
    monkeypatch.syspath_prepend(str(BACKEND_ROOT))
    monkeypatch.syspath_prepend(str(PLUGIN_SRC))
    for name in list(sys.modules):
        if name.split(".")[0] in _PLUGIN_MODULES:
            monkeypatch.delitem(sys.modules, name)
    package = importlib.import_module("data_designer_github_repo_seed")
    assert Path(package.__file__).is_relative_to(PLUGIN_SRC)
    scraper = importlib.import_module("data_designer_github_repo_seed.scraper")
    _, repo_scraper = scraper._load_impl()
    monkeypatch.setattr(scraper, "_load_impl", lambda: (FakeGitHubClient, repo_scraper))
    FakeGitHubClient.calls = []
    yield package
    _drop_plugin_modules()


def _source(**overrides):
    from data_designer_github_repo_seed.config import GitHubRepoSeedSource

    fields = {"repos": ["octo/repo"], "token": _FAKE_TOKEN, "item_types": ["issues"], "limit": 2}
    return GitHubRepoSeedSource(**{**fields, **overrides})


def test_max_comments_above_what_the_query_reads_is_refused(plugin):
    from pydantic import ValidationError

    assert _source(max_comments_per_item = 30).max_comments_per_item == 30
    with pytest.raises(ValidationError):
        _source(max_comments_per_item = 100)


@pytest.mark.parametrize("item_type", ["issues", "pulls"])
def test_comments_off_sends_no_comments_selection(plugin, tmp_path, item_type):
    from data_designer_github_repo_seed.scraper import ScrapeConfig, scrape

    cfg = ScrapeConfig(["octo/repo"], _FAKE_TOKEN, [item_type], 2, False, 30)
    rows = scrape(cfg, tmp_path / "raw")

    pages = [q for q in FakeGitHubClient.calls if "RepoMeta" not in q]
    assert pages and not any("comments(" in q for q in pages)
    assert [row["comments"] for row in rows] == ["", ""]

    FakeGitHubClient.calls = []
    cfg = ScrapeConfig(["octo/repo"], _FAKE_TOKEN, [item_type], 2, True, 5)
    rows = scrape(cfg, tmp_path / "raw-on")
    assert all("comments(first: 30)" in q for q in FakeGitHubClient.calls if "RepoMeta" not in q)
    assert [row["comments"].count("[b]:") for row in rows] == [5, 5]


_CHILD = """
import json, sys, tempfile
from pathlib import Path

cache = Path(sys.argv[1])
tempfile.tempdir = str(cache)
sys.path.insert(0, sys.argv[2])
from test_data_recipe_github_seed import FakeGitHubClient, _source
from data_designer_github_repo_seed import impl, scraper

_, repo_scraper = scraper._load_impl()
scraper._load_impl = lambda: (FakeGitHubClient, repo_scraper)
if hasattr(impl, "_CACHE_DIR"):
    impl._CACHE_DIR = cache / "github-seed-cache"
reader = impl.GitHubRepoSeedReader()
reader.attach(_source(), None)
path = reader.get_dataset_uri()
print(json.dumps({"scrapes": sum("RepoMeta" in q for q in FakeGitHubClient.calls), "path": path}))
"""


def _run_job_process(tmp_path: Path) -> dict:
    # A fresh interpreter, like the spawned worker each recipe job runs in.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(PLUGIN_SRC), str(BACKEND_ROOT)])}
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, str(tmp_path), str(Path(__file__).parent)],
        capture_output = True,
        text = True,
        env = env,
        cwd = tmp_path,
        timeout = 300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_two_job_processes_scrape_once_and_leave_no_raw_runs(tmp_path):
    pytest.importorskip("data_designer")

    first = _run_job_process(tmp_path)
    second = _run_job_process(tmp_path)

    assert (first["scrapes"], second["scrapes"]) == (1, 0)
    assert second["path"] == first["path"]
    assert [row["number"] for row in map(json.loads, Path(first["path"]).read_text().splitlines())] == [1, 2]
    assert not list(tmp_path.rglob("raw-runs"))


def _age(path: Path, seconds: int) -> None:
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime - seconds - 1))


def _new_job(impl) -> None:
    # Another recipe job is another process, with nothing from this one in memory.
    getattr(impl, "_JOB_SCRAPES", {}).clear()


def test_an_expired_scrape_is_replaced_and_removed(plugin, tmp_path, monkeypatch):
    from data_designer_github_repo_seed import impl

    monkeypatch.setattr(impl, "_CACHE_DIR", tmp_path / "cache")
    reader = impl.GitHubRepoSeedReader()
    reader.attach(_source(), None)
    old = Path(reader.get_dataset_uri())
    _age(old, impl._SCRAPE_TTL_SECONDS)
    _new_job(impl)

    new = Path(reader.get_dataset_uri())

    assert new != old and new.exists() and old.exists()
    assert sum("RepoMeta" in q for q in FakeGitHubClient.calls) == 2

    _age(old, impl._SCRAPE_KEEP_SECONDS)
    _new_job(impl)
    assert Path(reader.get_dataset_uri()) == new and not old.exists()


def test_a_long_run_keeps_reading_its_own_scrape(plugin, tmp_path, monkeypatch):
    from data_designer_github_repo_seed import impl

    monkeypatch.setattr(impl, "_CACHE_DIR", tmp_path / "cache")
    reader = impl.GitHubRepoSeedReader()
    reader.attach(_source(), None)
    first = Path(reader.get_dataset_uri())
    # The engine asks for the URI again when it cycles the seed, maybe hours into the run.
    _age(first, impl._SCRAPE_TTL_SECONDS)

    assert Path(reader.get_dataset_uri()) == first and first.exists()
    assert sum("RepoMeta" in q for q in FakeGitHubClient.calls) == 1


def _github_recipe(**source):
    return {
        "seed_config": {
            "source": {
                "seed_type": "github_repo",
                "repos": ["octo/repo"],
                "item_types": ["issues"],
                "limit": 1,
                **source,
            }
        },
        "columns": [{"column_type": "expression", "name": "x", "expr": "1"}],
    }


@pytest.fixture
def validate_module(monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.syspath_prepend(str(BACKEND_ROOT))
    from routes.data_recipe import validate as module

    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising = False)
    return module


def _check(module, **source):
    from models.data_recipe import RecipePayload

    return module.validate(RecipePayload(recipe = _github_recipe(**source)))


def test_check_fails_with_no_token_anywhere(validate_module):
    response = _check(validate_module, token = "")

    assert response.valid is False
    assert [e.message for e in response.errors] == [validate_module._GITHUB_TOKEN_REQUIRED]


def test_check_passes_with_a_recipe_or_server_token(validate_module, monkeypatch):
    assert _check(validate_module, token = _FAKE_TOKEN).errors == []

    monkeypatch.setenv("GH_TOKEN", _FAKE_TOKEN)
    assert _check(validate_module, token = "").errors == []

    # A managed account's run strips the server's token from its environment.
    monkeypatch.setattr(validate_module, "managed_account", lambda: True)
    assert _check(validate_module, token = "").valid is False


def test_check_refuses_more_comments_than_the_scraper_reads(validate_module):
    response = _check(validate_module, token = _FAKE_TOKEN, max_comments_per_item = 100)

    assert response.valid is False
    assert "0 to 30" in response.errors[0].message
