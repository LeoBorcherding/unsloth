# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio update --ref`: ref validation, the download, and the pin the desktop reads."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli import _studio_source_ref as sr

SHA = "a" * 40


@pytest.mark.parametrize("ref", ["main", "feat/studio-x", "v2026.9.14", "a25b166e30", "nightly_1.2"])
def test_valid_refs(ref):
    assert sr.valid_ref(ref)


@pytest.mark.parametrize("ref", ["", "-x", "--upload-pack=x", "a..b", "x/", "x.lock", "a b", "a;b", "a@{1}", "x" * 201])
def test_invalid_refs(ref):
    assert not sr.valid_ref(ref)
    with pytest.raises(sr.SourceRefError):
        sr.resolve_ref(ref)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _tarball(entries):
    buf = io.BytesIO()
    with tarfile.open(fileobj = buf, mode = "w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_resolve_ref_asks_github_for_the_sha(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        return _Resp(SHA.encode())

    monkeypatch.setattr(sr.urllib.request, "urlopen", fake_urlopen)
    assert sr.resolve_ref("feat/x") == SHA
    assert seen["url"] == "https://api.github.com/repos/unslothai/unsloth/commits/feat/x"


def test_fetch_source_unpacks_one_tree_and_reuses_it(monkeypatch, tmp_path):
    body = _tarball({f"unsloth-{SHA}/pyproject.toml": b"[project]\n", f"unsloth-{SHA}/unsloth/__init__.py": b""})
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return _Resp(body)

    monkeypatch.setattr(sr.urllib.request, "urlopen", fake_urlopen)
    dest = sr.fetch_source(SHA, tmp_path)
    assert dest == tmp_path / "sources" / f"unsloth-{SHA[:12]}"
    assert (dest / "pyproject.toml").is_file()
    assert sr.fetch_source(SHA, tmp_path) == dest
    assert calls == [f"https://codeload.github.com/unslothai/unsloth/tar.gz/{SHA}"]


def test_fetch_source_rejects_an_archive_that_is_not_unsloth(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.urllib.request, "urlopen", lambda url, timeout: _Resp(_tarball({"x/readme": b""})))
    with pytest.raises(sr.SourceRefError):
        sr.fetch_source(SHA, tmp_path)


def test_pin_and_request_round_trip(tmp_path):
    assert sr.read_pin(tmp_path) is None
    sr.write_pin(tmp_path, "feat/x", SHA, tmp_path / "src")
    assert sr.read_pin(tmp_path)["ref"] == "feat/x"
    sr.clear_pin(tmp_path)
    assert sr.read_pin(tmp_path) is None

    sr.write_request(tmp_path, None, None)
    assert sr.read_request(tmp_path)["ref"] is None
    sr.clear_request(tmp_path)
    assert sr.read_request(tmp_path) is None


def test_prune_keeps_only_the_active_tree(tmp_path):
    for name in ("unsloth-old", "unsloth-new"):
        (tmp_path / "sources" / name).mkdir(parents = True)
    sr.prune_sources(tmp_path, keep = tmp_path / "sources" / "unsloth-new")
    assert [p.name for p in (tmp_path / "sources").iterdir()] == ["unsloth-new"]


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


def _stub_install(monkeypatch, studio, tmp_path):
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    seen = {}

    def fake_setup(*, verbose = False, repo_root = None):
        seen["repo_root"] = repo_root
        seen["local"] = studio.os.environ.get("STUDIO_LOCAL_INSTALL")

    class _Txn:
        enabled = True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def validate_launcher(self):
            pass

    monkeypatch.setattr(studio, "_run_setup_script", fake_setup)
    monkeypatch.setattr(studio, "_WindowsLauncherUpdateTransaction", _Txn)
    monkeypatch.setattr(studio._studio_runtime_gate, "ensure_managed_environment_is_idle", lambda home: None)
    monkeypatch.setattr(studio, "_refresh_desktop_shortcuts", lambda **k: None)
    return seen


def test_update_ref_installs_the_download_as_local_and_pins_it(monkeypatch, tmp_path):
    studio = _studio()
    seen = _stub_install(monkeypatch, studio, tmp_path)
    tree = tmp_path / "sources" / f"unsloth-{SHA[:12]}"
    monkeypatch.setattr(sr, "resolve_ref", lambda ref: SHA)
    monkeypatch.setattr(sr, "fetch_source", lambda sha, home: tree)
    sr.write_request(tmp_path, "feat/x", SHA)

    result = CliRunner().invoke(studio.studio_app, ["update", "--ref", "feat/x", "--no-verify"])

    assert result.exit_code == 0, result.output
    assert seen == {"repo_root": tree, "local": "1"}
    pin = json.loads((tmp_path / sr.PIN_FILE).read_text())
    assert (pin["ref"], pin["sha"]) == ("feat/x", SHA)
    assert sr.read_request(tmp_path) is None


def test_plain_update_returns_to_the_release_and_drops_the_pin(monkeypatch, tmp_path):
    studio = _studio()
    _stub_install(monkeypatch, studio, tmp_path)
    sr.write_pin(tmp_path, "feat/x", SHA, tmp_path / "src")

    result = CliRunner().invoke(studio.studio_app, ["update", "--no-verify"])

    assert result.exit_code == 0, result.output
    assert sr.read_pin(tmp_path) is None


def test_ref_and_local_are_exclusive(monkeypatch, tmp_path):
    studio = _studio()
    _stub_install(monkeypatch, studio, tmp_path)
    result = CliRunner().invoke(studio.studio_app, ["update", "--ref", "main", "--local"])
    assert result.exit_code == 2


def test_a_bad_ref_fails_before_installing(monkeypatch, tmp_path):
    studio = _studio()
    seen = _stub_install(monkeypatch, studio, tmp_path)
    result = CliRunner().invoke(studio.studio_app, ["update", "--ref", "--upload-pack=x"])
    assert result.exit_code != 0
    assert "repo_root" not in seen
