# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unload() collects a denoiser that only turns into cyclic garbage after _unload_locked's own gc."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest

from core.inference import diffusion
from core.inference.diffusion import DiffusionBackend


class _Denoiser:
    def __init__(self) -> None:
        self.hook = lambda: self  # closure cell back to self, like the fp16 guard's pre-hook


@pytest.fixture
def no_auto_gc():
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


def test_unload_collects_denoiser_pinned_through_teardown_gc(monkeypatch, no_auto_gc):
    engine = DiffusionBackend()
    engine._state = SimpleNamespace(pipe = SimpleNamespace(transformer = _Denoiser()))
    alive = weakref.ref(engine._state.pipe.transformer)
    clears = []
    monkeypatch.setattr(diffusion, "clear_gpu_cache", lambda: clears.append(gc.collect()))

    def teardown():
        held = engine._state  # still referenced while the teardown's own clear runs
        engine._state = None
        diffusion.clear_gpu_cache()
        assert alive() is not None and held is not None

    monkeypatch.setattr(engine, "_unload_locked", teardown)
    engine.unload()
    assert alive() is None
    assert len(clears) == 2


def test_unload_with_nothing_loaded_skips_the_extra_clear(monkeypatch):
    engine = DiffusionBackend()
    clears = []
    monkeypatch.setattr(diffusion, "clear_gpu_cache", lambda: clears.append(1))
    monkeypatch.setattr(engine, "_unload_locked", lambda: None)
    engine.unload()
    assert clears == []
