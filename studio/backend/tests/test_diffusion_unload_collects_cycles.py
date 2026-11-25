# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unload() collects a denoiser that only turns into cyclic garbage after _unload_locked's own gc."""

from __future__ import annotations

import gc
import threading
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


def test_unload_queued_behind_a_teardown_waits_for_its_reclaim(monkeypatch):
    engine = DiffusionBackend()
    engine._state = SimpleNamespace(pipe = None)
    in_reclaim, release = threading.Event(), threading.Event()
    calls = []

    def clear():
        calls.append(threading.current_thread().name)
        if len(calls) == 2:  # the second clear of the first unload: the post-teardown reclaim
            in_reclaim.set()
            assert release.wait(10)

    def teardown():
        if engine._state is not None:
            engine._state = None
            diffusion.clear_gpu_cache()

    monkeypatch.setattr(diffusion, "clear_gpu_cache", clear)
    monkeypatch.setattr(engine, "_unload_locked", teardown)
    first = threading.Thread(target = engine.unload, name = "first")
    first.start()
    assert in_reclaim.wait(10)
    second_done = threading.Event()
    second = threading.Thread(target = lambda: (engine.unload(), second_done.set()), name = "second")
    second.start()
    assert not second_done.wait(0.3)
    release.set()
    first.join(10)
    second.join(10)
    assert second_done.is_set()
    assert calls == ["first", "first"]
