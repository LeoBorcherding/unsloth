# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/studio/source: off by default, desktop only, and it only records the request."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parents[1]
for p in (str(_BACKEND), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import get_current_subject
from routes import studio_source
from unsloth_cli import _studio_source_ref as sr

SHA = "b" * 40


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(studio_source, "studio_root", lambda: tmp_path)
    monkeypatch.setattr(sr, "resolve_ref", lambda ref: SHA)
    monkeypatch.delenv(sr.ALLOW_API_ENV, raising = False)
    monkeypatch.delenv(studio_source.DESKTOP_MANAGED_ENV, raising = False)
    app = FastAPI()
    app.include_router(studio_source.router, prefix = "/api/studio")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    app.dependency_overrides[policy.require_owner] = lambda: None
    return TestClient(app), tmp_path


def _enable(monkeypatch, desktop = True):
    monkeypatch.setenv(sr.ALLOW_API_ENV, "1")
    if desktop:
        monkeypatch.setenv(studio_source.DESKTOP_MANAGED_ENV, "1")


def test_status_reports_the_pin(client):
    c, home = client
    sr.write_pin(home, "feat/x", SHA, home / "src")
    body = c.get("/api/studio/source").json()
    assert body["pinned"]["ref"] == "feat/x"
    assert body["pending"] is None
    assert body["switch_allowed"] is False


def test_switch_is_off_by_default(client):
    c, home = client
    assert c.post("/api/studio/source", json = {"ref": "main"}).status_code == 403
    assert sr.read_request(home) is None


def test_switch_outside_the_desktop_is_refused_with_the_command(client, monkeypatch):
    c, home = client
    _enable(monkeypatch, desktop = False)
    r = c.post("/api/studio/source", json = {"ref": "feat/x"})
    assert r.status_code == 409
    assert "unsloth studio update --ref feat/x" in r.json()["detail"]
    assert sr.read_request(home) is None


def test_switch_records_a_request_for_the_desktop(client, monkeypatch):
    c, home = client
    _enable(monkeypatch)
    r = c.post("/api/studio/source", json = {"ref": "feat/x"})
    assert r.status_code == 202
    assert r.json()["pending"]["sha"] == SHA
    assert sr.read_request(home)["ref"] == "feat/x"


def test_null_ref_asks_for_the_release(client, monkeypatch):
    c, home = client
    _enable(monkeypatch)
    assert c.post("/api/studio/source", json = {"ref": None}).status_code == 202
    assert sr.read_request(home)["ref"] is None


def test_unknown_ref_is_a_422(client, monkeypatch):
    c, home = client
    _enable(monkeypatch)

    def missing(ref):
        raise sr.SourceRefError("not on unslothai/unsloth")

    monkeypatch.setattr(sr, "resolve_ref", missing)
    assert c.post("/api/studio/source", json = {"ref": "nope"}).status_code == 422
    assert sr.read_request(home) is None


def test_delete_drops_the_pending_request(client, monkeypatch):
    c, home = client
    sr.write_request(home, "feat/x", SHA)
    assert c.delete("/api/studio/source").json()["pending"] is None
    assert sr.read_request(home) is None
