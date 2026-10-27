# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DELETE /api/models/delete-finetuned must refuse a checkpoint the export worker has loaded and the
folder an in-flight export is writing into. Every other engine was checked; this one was not."""

from __future__ import annotations

import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.models as models_module
from auth.authentication import get_current_subject
from core.export.orchestrator import ExportOrchestrator
from routes.models import router as models_router
from storage import library_db


class _ExportWorker:
    def __init__(self, *paths):
        self._paths = list(paths)

    def paths_in_use(self):
        return self._paths


@pytest.fixture()
def client(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    exports = tmp_path / "exports"
    outputs.mkdir()
    exports.mkdir()
    monkeypatch.setattr(models_module, "outputs_root", lambda: outputs)
    monkeypatch.setattr(models_module, "exports_root", lambda: exports)
    monkeypatch.setattr(library_db, "list_entries", lambda: {})
    monkeypatch.setattr(library_db, "delete_entry", lambda _id: None)
    monkeypatch.setattr(models_module, "_active_diffusion_backend", lambda: None)
    monkeypatch.setattr(models_module, "_active_video_backend", lambda: None)

    app = FastAPI()
    app.include_router(models_router, prefix = "/api/models")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app), outputs, exports


def _use_export_worker(monkeypatch, worker):
    stub = types.ModuleType("core.export")
    stub.get_export_backend = lambda: worker
    monkeypatch.setitem(sys.modules, "core.export", stub)


def _model_dir(root, name):
    d = root / name
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    return d


def _delete(client, path, source):
    return client.request(
        "DELETE",
        "/api/models/delete-finetuned",
        json = {"model_path": str(path), "source": source},
    )


def test_refuses_the_checkpoint_the_export_worker_has_loaded(client, monkeypatch):
    c, outputs, _ = client
    run = _model_dir(outputs, "my-run")
    checkpoint = _model_dir(run, "checkpoint-60")
    _use_export_worker(monkeypatch, _ExportWorker(str(checkpoint)))

    # The run folder holds the loaded checkpoint, and so does the checkpoint itself.
    for target in (run, checkpoint):
        resp = _delete(c, target, "training")
        assert resp.status_code == 409, resp.text
        assert "export" in resp.json()["detail"].lower()
    assert (checkpoint / "model.safetensors").exists()


def test_refuses_the_folder_an_export_is_writing_into(client, monkeypatch):
    c, _, exports = client
    out = _model_dir(exports, "my-export")
    _use_export_worker(monkeypatch, _ExportWorker(str(out)))

    resp = _delete(c, out, "exported")
    assert resp.status_code == 409, resp.text
    assert out.exists()


def test_other_models_still_delete_while_the_worker_is_busy(client, monkeypatch):
    c, outputs, exports = client
    busy = _model_dir(outputs, "busy-run")
    other = _model_dir(outputs, "other-run")
    out = _model_dir(exports, "my-export")
    _use_export_worker(monkeypatch, _ExportWorker(str(busy), str(out)))

    assert _delete(c, other, "training").status_code == 200
    assert not other.exists() and busy.exists() and out.exists()


def test_an_unreadable_export_worker_fails_closed(client, monkeypatch):
    c, outputs, _ = client
    target = _model_dir(outputs, "my-run")

    class _Broken:
        def paths_in_use(self):
            raise RuntimeError("worker state unreadable")

    _use_export_worker(monkeypatch, _Broken())
    assert _delete(c, target, "training").status_code == 503
    assert target.exists()


def test_the_orchestrator_reports_the_in_flight_output_folder(tmp_path, monkeypatch):
    import utils.paths as paths_module

    out = tmp_path / "exports" / "merged"
    monkeypatch.setattr(paths_module, "resolve_export_write_dir", lambda _p: out)
    orch = ExportOrchestrator()
    orch.current_checkpoint = str(tmp_path / "outputs" / "run")
    monkeypatch.setattr(orch, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orch, "_send_cmd", lambda cmd: None)
    seen = {}

    def _wait_response(_kind, timeout = None):
        seen["during"] = orch.paths_in_use()
        return {"success": True, "message": "ok", "output_path": str(out)}

    monkeypatch.setattr(orch, "_wait_response", _wait_response)
    ok, _, _ = orch.export_merged_model(save_directory = "merged")

    assert ok
    assert seen["during"] == [orch.current_checkpoint, str(out)]
    assert orch.paths_in_use() == [orch.current_checkpoint]


def test_a_compressed_merge_also_holds_its_suffixed_sibling(tmp_path, monkeypatch):
    import utils.paths as paths_module

    out = tmp_path / "exports" / "merged"
    sibling = tmp_path / "exports" / "merged-torchao-fp8"
    unrelated = tmp_path / "exports" / "other"
    for d in (sibling, unrelated):
        d.mkdir(parents = True)
    monkeypatch.setattr(paths_module, "resolve_export_write_dir", lambda _p: out)
    orch = ExportOrchestrator()
    monkeypatch.setattr(orch, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(orch, "_send_cmd", lambda cmd: None)
    seen = {}

    def _wait_response(_kind, timeout = None):
        seen["during"] = orch.paths_in_use()
        return {"success": True, "message": "ok", "output_path": str(sibling)}

    monkeypatch.setattr(orch, "_wait_response", _wait_response)
    ok, _, _ = orch.export_merged_model(save_directory = "merged", format_type = "FP8 (torchao)")

    assert ok
    assert seen["during"] == [str(out), str(sibling)]
