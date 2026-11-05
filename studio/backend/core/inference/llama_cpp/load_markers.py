# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load-in-flight markers shared with the routes."""

import contextlib
import functools
import os
import threading

from typing import (
    Callable,
    Optional,
)
from core.inference.llama_server_args import extra_args_disable_mmproj

from .binary_runtime import _load_probe_memo_scope
from .core_types import GgufLoadIntent
from .gguf_cache import _hub_download_blocks_gguf_load
from .spec_args import _canonicalize_spec_mode

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.

# Active GGUF loads by normalized repo ID.
_LOADS_IN_FLIGHT: dict[str, int] = {}
_LOADS_IN_FLIGHT_LOCK = threading.Lock()


@contextlib.contextmanager
def gguf_load_in_flight(hf_repo: Optional[str]):
    """Track an HF GGUF load until the context exits."""
    key = (hf_repo or "").strip().lower()
    if not key:
        yield
        return
    with _LOADS_IN_FLIGHT_LOCK:
        _LOADS_IN_FLIGHT[key] = _LOADS_IN_FLIGHT.get(key, 0) + 1
    try:
        yield
    finally:
        with _LOADS_IN_FLIGHT_LOCK:
            remaining = _LOADS_IN_FLIGHT.get(key, 1) - 1
            if remaining <= 0:
                _LOADS_IN_FLIGHT.pop(key, None)
            else:
                _LOADS_IN_FLIGHT[key] = remaining


# Chat loads in flight, repo-agnostic (local paths and safetensors included). The repo-keyed counter above answers the download manager; this one answers the GPU arbiter, which must know before llama-server spawns.
_CHAT_LOADS_IN_FLIGHT = 0


@contextlib.contextmanager
def chat_load_in_flight():
    """Track a chat load from before its download until the load call returns."""
    global _CHAT_LOADS_IN_FLIGHT
    with _LOADS_IN_FLIGHT_LOCK:
        _CHAT_LOADS_IN_FLIGHT += 1
    try:
        yield
    finally:
        with _LOADS_IN_FLIGHT_LOCK:
            _CHAT_LOADS_IN_FLIGHT = max(0, _CHAT_LOADS_IN_FLIGHT - 1)


def chat_load_active() -> bool:
    """Whether any chat load is in flight, spawned or not; is_active misses loads still downloading."""
    with _LOADS_IN_FLIGHT_LOCK:
        return _CHAT_LOADS_IN_FLIGHT > 0


def zero_vram_chat_load(
    gpu_memory_mode: str,
    gpu_layers: int,
    extra_args: Optional[list] = None,
    needs_mmproj: bool = False,
    speculative_type: Optional[str] = None,
) -> bool:
    """Mirrors the launch GPU mask so a CPU-only load does not evict an image or video pipeline; mmproj and drafter intent are passed in."""
    if gpu_memory_mode != "manual" or gpu_layers != 0:
        return False
    # Every spec mode but off may launch a GPU drafter: canonicalize first, and an absent mode is auto, not off, or the arbiter is skipped.
    # -- reading the FINISHED argv, drafter included -- kept the GPUs visible: no
    spec_mode = _canonicalize_spec_mode(speculative_type) or "auto"
    if needs_mmproj or spec_mode != "off":
        return False
    if LlamaCppBackend._is_vulkan_backend():
        return False
    return not LlamaCppBackend._zero_offload_keeps_gpu_visible(
        [str(arg) for arg in (extra_args or [])], os.environ
    )


def hf_gguf_load_in_flight(hf_repo: str) -> bool:
    """Return whether a GGUF load is active for hf_repo."""
    key = (hf_repo or "").strip().lower()
    if not key:
        return False
    with _LOADS_IN_FLIGHT_LOCK:
        return _LOADS_IN_FLIGHT.get(key, 0) > 0


def _with_gguf_load_marker(load: Callable):
    """Keep an HF repo marked for the full synchronous load call."""

    @functools.wraps(load)
    def wrapped(
        self,
        intent: GgufLoadIntent,
        load_cancel_event: Optional[threading.Event] = None,
    ):
        hf_repo = intent.hf_repo
        # Must not leak: polling outside a load needs LIVE VRAM, never this snapshot.
        with _load_probe_memo_scope(), gguf_load_in_flight(hf_repo):
            if hf_repo and _hub_download_blocks_gguf_load(
                hf_repo,
                intent.hf_variant,
                # Must mirror the download gate below exactly, or a vision-off load races a download job.
                require_mmproj = bool(
                    intent.is_vision and not extra_args_disable_mmproj(intent.extra_args)
                ),
                hf_token = intent.hf_token,
            ):
                raise RuntimeError(
                    f"'{hf_repo}' is currently being downloaded by the download manager"
                )
            return load(self, intent, load_cancel_event = load_cancel_event)

    return wrapped
