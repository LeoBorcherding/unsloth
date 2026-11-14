# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exceptions, load intents, the serving-backend registry and shared constants."""

import errno
import os
import tempfile
import threading
import time
import weakref

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Literal,
    NamedTuple,
    Optional,
)
import contextlib
import subprocess
from .memory_policy import (
    _APPLE_UNIFIED_MEMORY_FRACTION,
    _APPLE_WIRED_CEILING_FRACTION,
)

# Floor for a GGUF TTS read; the subprocess backend needs a much larger base.
_GGUF_AUDIO_READ_TIMEOUT = 300.0


class LlamaServerNotFoundError(RuntimeError):
    """GGUF model needs the llama.cpp runtime but no llama-server is installed.
    Subclasses RuntimeError so existing handlers still catch it."""

    __slots__ = ()


def _gpu_short_message(
    need_gb: float, free_gb: float, asked_ctx: int, fitting_ctx: int, capped_only: bool
) -> str:
    """What a user reads when a model does not fit next to the loaded ones."""
    message = (
        f"This model needs about {need_gb:.1f} GB of GPU memory at a {asked_ctx} context, "
        f"and {free_gb:.1f} GB is free next to the models already loaded. "
    )
    if capped_only:
        return message + f"It fits with a {fitting_ctx} context."
    try:
        from utils.multi_model_settings import get_multi_model_mode
        always = get_multi_model_mode() == "always"
    except Exception:  # noqa: BLE001 - the copy must never fail the load
        always = False
    if always:
        return message + (
            "Unload one of them or lower the context length. Keep multiple models loaded "
            "is set to Always, so nothing is unloaded to make room."
        )
    return message + (
        "Unload one of them, lower the context length, or set Keep multiple models loaded "
        "to Off to replace the current model."
    )


class GpuMemoryShortError(RuntimeError):
    """A load that must fit next to the loaded models does not. ``capped``: it would, at a
    smaller context than asked for. ``gpu_indices``: the GPUs ``short_mib`` was measured on.
    ``discrete``: none of them shares host RAM, so a server off them or on the CPU frees nothing there.
    ``host``: short on free VRAM and RAM together."""

    __slots__ = ("capped", "short_mib", "gpu_indices", "discrete", "host")

    def __init__(
        self,
        message: str,
        capped: bool = False,
        short_mib: int = 0,
        gpu_indices = None,
        discrete: bool = False,
        host: bool = False,
    ):
        super().__init__(message)
        self.capped = capped
        self.short_mib = short_mib
        self.gpu_indices = gpu_indices
        self.discrete = discrete
        self.host = host


def _apple_budget_beside_media_mib(held_mib: int, unallocated_mib: int) -> int:
    """Unified-memory budget beside resident media: price off the working set, unallocated part off available RAM."""
    from utils.hardware.hardware import _metal_recommended_working_set_bytes
    import psutil

    mib = 1024 * 1024
    working_set = _metal_recommended_working_set_bytes() or int(psutil.virtual_memory().total)
    available = int(psutil.virtual_memory().available)
    left = working_set // mib - held_mib
    if available > 0:
        left = min(left, available // mib - unallocated_mib)
    return max(0, int(left * _APPLE_UNIFIED_MEMORY_FRACTION))


def _apple_wired_headroom_bytes() -> Optional[int]:
    """Wired GPU memory a Metal load may still take, system-wide; None if unknown."""
    from utils.hardware import is_apple_silicon

    if not is_apple_silicon():
        return None
    try:
        probe = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output = True,
            text = True,
            timeout = 5,
            encoding = "utf-8",
            errors = "replace",
        )
        cap_bytes = int(probe.stdout.strip()) * 1024 * 1024
    except Exception:
        return None
    # 0 means the kernel default, Metal's recommended working set.
    if cap_bytes <= 0:
        from utils.hardware.hardware import _metal_recommended_working_set_bytes
        cap_bytes = _metal_recommended_working_set_bytes()
    if cap_bytes <= 0:
        return None
    from utils.hardware.hardware import _read_apple_gpu_stats

    in_use = _read_apple_gpu_stats().get("vram_used_bytes")
    if in_use is None:
        return None
    return max(0, int((cap_bytes - in_use) * _APPLE_WIRED_CEILING_FRACTION))


def _gpu_plan_mib(total_bytes: int, on: list[int], gpus) -> dict[int, int]:
    """MiB this load holds per GPU: an even share, capped at what each card had free, since
    --fit on offloads the rest to CPU."""
    share = total_bytes // (1024 * 1024 * len(on))
    free = dict(gpus)
    return {idx: min(share, free.get(idx, share)) for idx in on}


def _net_of_held_vram(gpu_mem, held: dict[int, int]):
    """Cap each GPU's free MiB at its total less what the other loaded models planned. A total of 0
    is an integrated GPU on shared RAM, whose free reading already counts them: left as is."""
    if not held:
        return gpu_mem
    return [
        (idx, min(free, max(0, total - held.get(idx, 0))) if total > 0 else free, total)
        for idx, free, total in gpu_mem
    ]


def _net_of_other_models_vram(gpu_mem, held: dict[int, int], unallocated: dict[int, int]):
    """Live free less unallocated media VRAM, then capped at total less ``held``. The other order charges the
    headroom twice whenever the total cap binds (nothing else on the card)."""
    return _net_of_held_vram(_net_of_unallocated_vram(gpu_mem, unallocated), held)


def _net_of_unallocated_vram(gpu_mem, unallocated: dict[int, int]):
    """Take from each GPU's live free MiB what a resident image or video model has not allocated yet (its generation
    headroom): the total cap misses it once another process uses the card. Totals of 0 are left as is."""
    if not unallocated:
        return gpu_mem
    return [
        (idx, max(0, free - unallocated.get(idx, 0)) if total > 0 else free, total)
        for idx, free, total in gpu_mem
    ]


_serving_backends: "weakref.WeakSet[LlamaCppBackend]" = weakref.WeakSet()
# A WeakSet is not safe to iterate while another thread adds or drops a backend.
_serving_backends_lock = threading.Lock()


def register_serving_backend(backend) -> None:
    with _serving_backends_lock:
        _serving_backends.add(backend)


def unregister_serving_backend(backend) -> None:
    with _serving_backends_lock:
        _serving_backends.discard(backend)


def _serving_backends_snapshot() -> list:
    with _serving_backends_lock:
        return list(_serving_backends)


class GgufDownloadCancelled(RuntimeError):
    """Expected signal from a cancelled GGUF download."""

    __slots__ = ()


class _LlamaStreamCancelled(Exception):
    """Internal signal for an expected client/request cancellation."""

    __slots__ = ()


# vulkan-hpp formats DeviceLostError as "vk::Queue::submit: ErrorDeviceLost".
_GPU_DEVICE_LOST_MARKERS = ("ErrorDeviceLost", "VK_ERROR_DEVICE_LOST")


_DRAFT_CONTEXT_FAILURE_MARKERS = (
    "llama_process(ctx_dft) failed",
    "failed to process speculative batch",
)


# ggml-metal latches a failed command buffer ("recreate the backend to recover"); the child then
# answers "Compute error.".
_METAL_BACKEND_POISONED_MARKERS = (
    "backend is in error state from a previous command buffer failure",
    "kIOGPUCommandBufferCallbackErrorOutOfMemory",
)
_DEVICE_LOST_LOG_GRACE_S = 2.0


@contextlib.contextmanager
def _automatic_recovery_load():
    prev = getattr(_AUTO_RECOVERY_LOAD, "active", False)
    _AUTO_RECOVERY_LOAD.active = True
    try:
        yield
    finally:
        _AUTO_RECOVERY_LOAD.active = prev


# Set during automatic recovery loads: only those inherit a Metal OOM strike.
_AUTO_RECOVERY_LOAD = threading.local()


# Run after a terminal Metal OOM unload, outside every backend lock (routes/inference.py registers
# them).
_TERMINAL_RETIREMENT_HOOKS: list = []


def _is_gpu_device_lost(exc: BaseException) -> bool:
    """True when llama-server reported a lost GPU device (500 body or SSE error, #11453)."""
    text = str(exc)
    return any(marker in text for marker in _GPU_DEVICE_LOST_MARKERS)


class _CombinedCancelEvent:
    __slots__ = ("_events",)

    def __init__(self, *events: threading.Event):
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)

    def wait(self, timeout: Optional[float] = None) -> bool:
        if self.is_set():
            return True
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return self.is_set()
            pause = 0.05 if remaining is None else min(0.05, remaining)
            self._events[0].wait(pause)
            if self.is_set():
                return True


# Not slots=True: vars(intent) must work or every GGUF load breaks.
@dataclass(frozen = True)
class GgufLoadIntent:
    """Immutable caller intent replayed by retries and recovery."""

    model_identifier: str
    gguf_path: Optional[str] = None
    # A cached file and every shard's byte count, already verified for this repo and variant.
    verified_gguf: Optional[tuple[str, str, str, tuple[tuple[str, int], ...]]] = None
    mmproj_path: Optional[str] = None
    mtp_draft_path: Optional[str] = None
    dspark_draft_path: Optional[str] = None
    dflash_draft_path: Optional[str] = None
    hf_repo: Optional[str] = None
    hf_variant: Optional[str] = None
    hf_token: Optional[str] = None
    audio_codec_path: Optional[str] = None
    is_vision: bool = False
    # Text-only load: no projector on GPU or CPU, freeing its VRAM; image input is off for the session.
    disable_vision: bool = False
    n_ctx: int = 4096
    max_seq_length_auto_derived: bool = False
    chat_template_override: Optional[str] = None
    cache_type_kv: Optional[str] = None
    speculative_type: Optional[str] = None
    spec_draft_n_max: Optional[int] = None
    tensor_parallel: bool = False
    gpu_memory_mode: Literal["auto", "manual"] = "auto"
    gpu_layers: int = -1
    n_cpu_moe: int = 0
    tensor_split: Optional[tuple[float, ...]] = None
    gpu_ids: Optional[tuple[int, ...]] = None
    gpu_ids_are_vulkan_ordinals: Optional[bool] = None
    n_threads: Optional[int] = None
    n_parallel: int = 1
    # Slots left at the default: Auto speculation may lower them to keep a drafter (#8872).
    n_parallel_from_default: bool = False
    # none follows the llama.cpp defaults (2048 / 512)
    n_batch: Optional[int] = None
    n_ubatch: Optional[int] = None
    # llama-server tuning group; none follows the llama.cpp defaults
    # (auto / f16 / 32 / 8192).
    load_mode: Optional[str] = None
    spec_draft_cache_type: Optional[str] = None
    ctx_checkpoints: Optional[int] = None
    cache_ram: Optional[int] = None
    # Sets --reasoning-budget and --reasoning-budget-message; -1 and empty use llama.cpp defaults.
    reasoning_budget: int = -1
    reasoning_budget_message: str = ""
    extra_args: Optional[tuple[str, ...]] = None
    # Inherited extras look like named ones; duplicate-load checks need the difference.
    extra_args_inherited: bool = False
    preserve_multi_gpu_on_layer: bool = False
    refuse_partial_gpu_fit: bool = False
    compare_mtp_draft: bool = False
    force_reload: bool = False

    cpu_fallback: bool = False

    def __post_init__(self):
        for key in ("tensor_split", "gpu_ids", "extra_args"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, tuple(value))
        if self.gpu_ids == ():
            object.__setattr__(self, "gpu_ids", None)


class _CpuFallbackRuntime(NamedTuple):
    tempdir: tempfile.TemporaryDirectory
    source_binary: Path
    staged_binary: Path
    # An update swaps the install dir in place, so the path alone is not enough.
    source_stamp: tuple


# Shared so the from_identifier preflight and the load-time raise stay in sync.
LLAMA_SERVER_NOT_FOUND_DETAIL = (
    "This is a GGUF model, but no executable llama.cpp runtime (llama-server) "
    "is available. Run `unsloth studio setup` to download the prebuilt runtime, "
    "then try again. If you selected a custom runtime, restore its execute "
    "permission first (for example, `chmod +x llama-server` on Unix)."
)

# Every env var that can put a drafter or spec type into the child. Pre-b8955 spellings too: older builds still read them and keep the drafter.
_SPEC_ENV_VARS = (
    "LLAMA_ARG_SPEC_DRAFT_MODEL",
    "LLAMA_ARG_SPEC_DRAFT_HF_REPO",
    "LLAMA_ARG_SPEC_TYPE",
    "LLAMA_ARG_MODEL_DRAFT",
    "LLAMA_ARG_HFD_REPO",
)

# Hub failures that are settled: retrying them buys nothing.
_PERMANENT_HUB_ERRORS = (
    "RepositoryNotFoundError",
    "GatedRepoError",
    "RevisionNotFoundError",
    "EntryNotFoundError",
    "OfflineModeIsEnabled",
)

# By errno, not type: the Hub client raises OSError subclasses for retryable network errors.
_UNRECOVERABLE_LOCAL_ERRNOS = frozenset(
    {
        errno.EACCES,  # cache dir not writable
        errno.EPERM,
        errno.ENOSPC,  # disk full
        errno.EROFS,
    }
)

# Every kind records binary_no_mtp, so the reason alone cannot say which capability to recheck.
_SPEC_KIND_CAPABILITY: dict[str, str] = {
    "dspark": "supports_dspark",
    "dflash": "supports_dflash",
    "eagle3": "supports_eagle3",
    "mtp": "supports_mtp",
}


# llama.h: #define LLAMA_DEFAULT_SEED 0xFFFFFFFF
_LLAMA_RANDOM_SEED = 0xFFFFFFFF


_EMPTY_THINK_PROBE_LOCK = threading.Lock()


def _apply_seeded_llama_request(
    payload: dict,
    seed: Optional[int],
    *,
    reuse_prompt_cache: bool = False,
) -> None:
    """Disable prompt caching for fixed seeds so repeated requests stay reproducible.

    ``reuse_prompt_cache``: a tool-loop continuation of this run; cold, it re-prefills (#10698).
    """
    if seed is None:
        return
    payload["seed"] = seed
    if reuse_prompt_cache:
        return
    # Compared as uint32: the schemas also accept 4294967295, the same "pick at random".
    if (seed & 0xFFFFFFFF) != _LLAMA_RANDOM_SEED:
        payload["cache_prompt"] = False


try:
    _SLOT_SAVE_MAX_BYTES = int(os.environ.get("UNSLOTH_SLOT_SAVE_MAX_BYTES") or (10 << 30))
except ValueError:
    _SLOT_SAVE_MAX_BYTES = 10 << 30

# The idle loop holds the lifecycle gate across a slot save, so bound it.
# Override with UNSLOTH_SLOT_SAVE_TIMEOUT (seconds).
try:
    _SLOT_SAVE_HTTP_TIMEOUT = float(os.environ.get("UNSLOTH_SLOT_SAVE_TIMEOUT") or 30.0)
except ValueError:
    _SLOT_SAVE_HTTP_TIMEOUT = 30.0
if _SLOT_SAVE_HTTP_TIMEOUT <= 0:
    _SLOT_SAVE_HTTP_TIMEOUT = 30.0
