# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's /audio/stt/unload releases only a model it can see."""

import asyncio
import threading
import time

import pytest

from core.inference import stt_registry
from core.inference.stt_sidecar import WhisperSttSidecar

OWNER_MODEL = "openai/whisper-small"
MANAGED_MODEL = "openai/whisper-base"


def _sidecar(model_id = None):
    sidecar = WhisperSttSidecar.__new__(WhisperSttSidecar)
    sidecar._engine = None
    sidecar._model_id = model_id
    sidecar._lock = threading.RLock()
    sidecar._load_state_lock = threading.Lock()
    sidecar._loading = False
    sidecar.released = []

    def release():
        sidecar.released.append(sidecar._model_id)
        sidecar._model_id = None

    sidecar._release_engine_locked = release
    return sidecar


@pytest.fixture
def managed(monkeypatch):
    import routes.inference as ri

    monkeypatch.setattr(ri.account_access, "managed_account", lambda: True)
    # Only the managed caller's own model is visible to it.
    monkeypatch.setattr(
        ri.account_access,
        "resident_hidden",
        lambda modality = None, reference = None: reference not in (None, MANAGED_MODEL),
    )
    monkeypatch.setattr(ri, "_resolve_serving_stt_engine", lambda engine: engine)
    monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (stt_registry.load, stt_registry.unload))
    return ri


def _unload(ri, sidecar, monkeypatch, model = None):
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda engine: sidecar)
    return asyncio.run(ri.stt_unload(engine = "transformers", model = model, current_subject = "bob"))


def test_a_managed_unload_during_the_owners_first_load_keeps_the_owners_model(managed, monkeypatch):
    sidecar = _sidecar()
    holding = threading.Event()

    def owner_load():
        # load() holds the lock for the whole load and publishes the model inside it.
        with sidecar._lock:
            with sidecar._load_state_lock:
                sidecar._loading = True
            holding.set()
            time.sleep(1.0)
            sidecar._model_id = OWNER_MODEL
            with sidecar._load_state_lock:
                sidecar._loading = False

    loader = threading.Thread(target = owner_load)
    loader.start()
    assert holding.wait(5)
    try:
        response = _unload(managed, sidecar, monkeypatch)
    finally:
        loader.join(5)
    assert response.status_code == 200
    assert sidecar.released == []
    assert sidecar._model_id == OWNER_MODEL


def test_a_managed_unload_still_releases_its_own_model(managed, monkeypatch):
    sidecar = _sidecar(MANAGED_MODEL)
    response = _unload(managed, sidecar, monkeypatch)
    assert response.status_code == 200
    assert sidecar.released == [MANAGED_MODEL]


def test_a_managed_unload_keeps_an_owner_model_swapped_in_before_the_lock(managed, monkeypatch):
    sidecar = _sidecar(MANAGED_MODEL)

    def swapped_then_unload(*args, **kwargs):
        sidecar._model_id = OWNER_MODEL
        return WhisperSttSidecar.unload(sidecar, *args, **kwargs)

    sidecar.unload = swapped_then_unload
    response = _unload(managed, sidecar, monkeypatch)
    assert response.status_code == 200
    assert sidecar.released == []
    assert sidecar._model_id == OWNER_MODEL
