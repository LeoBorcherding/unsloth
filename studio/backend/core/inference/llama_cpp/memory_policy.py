# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VRAM budgets, KV cache sizing and memory policy constants."""

import os

from typing import (
    Callable,
    Collection,
    Iterable,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _flag_name,
    _LAYER_OFFLOAD_FLAGS,
    _MOE_OFFLOAD_FLAGS,
    _SPLIT_MODE_FLAGS,
    _TENSOR_SPLIT_FLAGS,
    parse_cache_override_per_axis,
)
from core.inference.llama_server_args import _last_flag_value
import ctypes
import re

# Below this, draft-mtp regresses vs spec-off; auto mode drops MTP under it.
_MTP_MIN_SIZE_B = 3.0

# Cap GPU occupancy at 97%; the 3% covers fragmentation, per-device CUDA context and MoE routing, not the modelled compute buffer.
_CTX_FIT_VRAM_FRACTION = 0.97

# Apple unified memory is shared with the OS, so tighter than VRAM; matches the 0.85 in mlx_inference.py but is not kept in sync.
_APPLE_UNIFIED_MEMORY_FRACTION = 0.85

# The 5% of wired headroom held back absorbs error in the footprint estimate.
_APPLE_WIRED_CEILING_FRACTION = 0.95

# 8192 floor (llama.cpp 4096 is too short for chat); a result at the search floor is that floor, not a measurement; Metal re-prices from _FIT_FLOOR_MIN_CTX.
_FIT_MIN_CTX = 8192
_FIT_FLOOR_MIN_CTX = 256

# llama.cpp's own --fit floor; without the fitter running, hand over this, not 8192.
_LLAMA_FIT_MIN_CTX = 4096

# _plan_tensor_parallel's no-KV fallback. A GUESS: may shrink Auto, never a request (#9653).
_TP_UNMEASURED_CTX = 4096

# Auto offload context when no GPU subset holds the model; prefers a chat-useful size, kept separate from _FIT_MIN_CTX (search and Apple floor).
_AUTO_OFFLOAD_CTX = 8192


# Auto context on unified memory; 65536 is the floor unsloth start agents need.
_APPLE_AUTO_CTX = 65536

# Keep this at or above the fit search floor. Not a style rule, a correctness one:
# the Auto offload branch re-checks whether a subset holds the model at the reduced
# context, and that re-check can only award residency BELOW the floor, since a subset
# winnable at or above it was already taken by the fit loop that runs first. Moving
# this constant is therefore free of placement changes only while it stays on the dead
# side of that floor. Below it the re-check hands over a device and flips --fit off,
# which is a placement decision rather than a display one.
#
# The floor in question is the ``min_ctx = 4096`` DEFAULT on _fit_context_to_vram and
# _cap_ctx_to_per_device_reserve, not _FIT_MIN_CTX: neither auto call site passes the
# argument, and _FIT_MIN_CTX is only handed in explicitly on the Apple arm. The three
# numbers agree today and nothing here makes them; test_auto_offload_ctx_invariants.py
# pins both defaults to _FIT_MIN_CTX and this constant above it, so the agreement
# cannot lapse silently.

# Must stay >= fit search floor or the Auto re-check flips --fit off (test pins it). amd-smi vs HIP VRAM totals differing over 10% are different scopes.
_AMD_SMI_TOTAL_TOLERANCE = 0.10

# Flat MTP reserve, used only when GGUF dims are too sparse for _estimate_mtp_overhead_bytes; applied to both the fit budget and the pin.
_MTP_VRAM_RESERVE_FRAC = 0.05


# Left on every card even at 100%, matching the --fit-target margin llama.cpp
# keeps for its own fitter.
_VRAM_FLOOR_RESERVE_MIB = 512.0

# llama-server's own --fit-target default ("target margin per device for --fit,
# ... default: 1024"): what the fallback fitter leaves when we pass nothing.
_LLAMA_FIT_TARGET_DEFAULT_MIB = 1024.0

# Windows + NVIDIA since driver 536.40: WDDM spills VRAM overflow to host RAM instead of failing, so the reserve matches llama.cpp fit margin.
_WINDOWS_SYSMEM_FALLBACK_RESERVE_MIB = _LLAMA_FIT_TARGET_DEFAULT_MIB
_WINDOWS_SYSMEM_FALLBACK_MAX_FRACTION = 0.125


def _windows_private_memory(pid: int) -> Optional[tuple[int, int]]:
    """(PrivateUsage, PrivateWorkingSetSize) in bytes; None when unreadable (needs Windows 10 1809+).
    psutil's uss and wset both count mmap'd GGUF pages, which would hide untouched commit."""
    from ctypes import wintypes

    class _COUNTERS_EX2(ctypes.Structure):
        _fields_ = (
            [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)]
            + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                    "PrivateUsage",
                    "PrivateWorkingSetSize",
                )
            ]
            + [("SharedCommitUsage", ctypes.c_ulonglong)]
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        counters = _COUNTERS_EX2()
        counters.cb = ctypes.sizeof(counters)
        if not kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PrivateUsage), int(counters.PrivateWorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


_WINDOWS_GPU_COMMIT_HEADROOM_WARN_MIB = 4096


# WDDM backs GPU allocations with system commit the process never touches (#16015).
_WINDOWS_GPU_UNTOUCHED_COMMIT_MIN_MIB = 2048


def _vram_reserve_floor_mib(total_mib: float, *, sysmem_fallback: bool = False) -> float:
    """Floor 512 MiB, capped at the default reserve to keep the budget monotonic; sysmem_fallback raises it and deliberately overrides a 100% budget."""
    floor = min(_VRAM_FLOOR_RESERVE_MIB, (1.0 - _CTX_FIT_VRAM_FRACTION) * total_mib)
    if not sysmem_fallback:
        return floor
    return max(
        floor,
        min(
            _WINDOWS_SYSMEM_FALLBACK_RESERVE_MIB,
            _WINDOWS_SYSMEM_FALLBACK_MAX_FRACTION * total_mib,
        ),
    )


def _vram_usable_mib(
    free_mib: float,
    total_mib: float,
    frac: float,
    *,
    pooled: bool = False,
    sysmem_fallback: bool = False,
) -> float:
    """Pooled budgets already paid every card reserve, so do not charge it again; a probe with no total still keeps a margin."""
    if total_mib and total_mib > 0:
        reserve = max(
            (1.0 - frac) * total_mib,
            _vram_reserve_floor_mib(total_mib, sysmem_fallback = sysmem_fallback),
        )
        return free_mib - reserve
    if pooled:
        return free_mib * frac
    return min(
        free_mib * frac,
        free_mib - _vram_reserve_floor_mib(free_mib, sysmem_fallback = sysmem_fallback),
    )


def _model_memory_settings_or_none() -> Optional[tuple[bool, bool]]:
    try:
        from utils.model_memory_settings import get_model_memory_settings
        return get_model_memory_settings()
    except Exception:
        return None


def _active_vram_fraction() -> float:
    """User VRAM budget, or _CTX_FIT_VRAM_FRACTION if unset; settings clamp it so no NaN or past 1.0. Read per call, so no restart is needed."""
    try:
        from utils.vram_budget_settings import get_vram_budget_fraction
        return get_vram_budget_fraction()
    except Exception:
        # Settings are a convenience; a load must never fail because of them.
        return _CTX_FIT_VRAM_FRACTION


def _kv_bytes_per_elem(cache_type: Optional[str]) -> float:
    """Bytes per KV-cache element for a llama.cpp cache type (f16 default)."""
    return {
        "f32": 4.0,
        "f16": 2.0,
        "bf16": 2.0,
        "q8_0": 34 / 32,
        "q5_1": 0.75,
        "q5_0": 0.6875,
        "q4_1": 0.625,
        "q4_0": 0.5625,
        "iq4_nl": 0.5625,
    }.get((cache_type or "f16").strip().lower(), 2.0)


# Not a CUDA/HIP FlashAttention KV type (ggml_cuda_fattn_kv_type_supported), so attention runs
# on the CPU. q4_1 / q5_0 / q5_1 did too until ggml-org/llama.cpp#28079 added an f16 fallback.
# Mapped to a same-size type that has the kernel, so the KV budget is unchanged (#10336).
_GPU_UNACCELERATED_CACHE_TYPES = {"iq4_nl": "q4_0"}


def _kv_cache_gpu_fallback_warning(
    cache_type_kv: Optional[str],
    gpu_memory_mode: str,
    gpu_layers: int,
    installed_backends: Callable[[], frozenset[str]],
) -> Optional[str]:
    """Warning for a KV cache type that runs attention on the CPU of a CUDA/HIP launch."""
    if cache_type_kv not in _GPU_UNACCELERATED_CACHE_TYPES:
        return None
    if gpu_memory_mode == "manual" and gpu_layers == 0:
        return None
    if not {"cuda", "hip"} & installed_backends():
        return None
    return (
        f"KV cache type {cache_type_kv} has no CUDA/HIP flash-attention kernel, so attention "
        "would fall back to CPU, causing high CPU load and slow generation. Using "
        f"{_GPU_UNACCELERATED_CACHE_TYPES[cache_type_kv]} instead, the same size on the GPU."
    )


# Upper bound on any current tokenizer vocab (output rows) when a truncated header drops the token array; above the 256k of Llama 4 and Gemma 3.
_ASSUMED_MAX_VOCAB = 262144
# Ceiling on float32 activations per micro-batch token for an unreadable header: 4 * n_ff at n_ff = 65536, above the Llama 3.1 405B figure of 53056.
_ASSUMED_MAX_ACTIVATION_WIDTH = 262144


def _pad_kv_cells(cells: int) -> int:
    return ((cells + 255) // 256) * 256


def _kv_cache_cell_layout(n_ctx: int, n_parallel: int, kv_unified: bool) -> tuple[int, int, int]:
    """Return llama.cpp's slot count, stream count, and cells per stream."""
    slots = max(1, n_parallel)
    padded_ctx = _pad_kv_cells(n_ctx)
    streams = 1 if kv_unified else slots
    if padded_ctx <= 0:
        return slots, streams, 0
    cells_per_stream = padded_ctx if kv_unified else _pad_kv_cells(padded_ctx // slots)
    return slots, streams, cells_per_stream


def _env_main_cache_type_for_budget(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Heavier of LLAMA_ARG_CACHE_TYPE_K/_V when not f16, budgeted because the child gets it unbudgeted (f32 under-reserves, quantized skips scratch)."""
    e = os.environ if env is None else env
    # An unset axis is llama.cpp f16 and joins the max; dropping it lets a lone q4_0 V price K as q4 while the child runs f16, under-reserving.
    axis_types = [
        (e.get(var) or "").strip().lower() or "f16"
        for var in ("LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V")
    ]
    if all(t == "f16" for t in axis_types):
        return None
    heaviest = max(axis_types, key = _kv_bytes_per_elem)
    if _kv_bytes_per_elem(heaviest) == _kv_bytes_per_elem("f16"):
        return None
    return heaviest


def _extra_args_main_cache_type_for_budget(extra_args: Optional[Iterable[str]]) -> Optional[str]:
    """Budget by the heavier axis: resolve_cache_type_kv keeps the last-wins type and under-reserves."""
    k, v = parse_cache_override_per_axis(extra_args)
    candidates = [c for c in (k, v) if c]
    if not candidates:
        return None
    return max(candidates, key = _kv_bytes_per_elem)


def _effective_main_cache_types(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[str, str]:
    """Effective main K/V cache types after environment and CLI precedence."""
    source_env = os.environ if env is None else env
    env_k = (source_env.get("LLAMA_ARG_CACHE_TYPE_K") or "f16").strip().lower()
    env_v = (source_env.get("LLAMA_ARG_CACHE_TYPE_V") or "f16").strip().lower()
    arg_k, arg_v = parse_cache_override_per_axis(args)
    return (
        (arg_k or env_k).strip().lower(),
        (arg_v or env_v).strip().lower(),
    )


def _planned_main_cache_types(
    cache_type_kv: Optional[str],
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
) -> tuple[str, str]:
    """Main K/V types the loader's managed flags and user extras will produce."""
    args = list(extra_args or ())
    emitted_type = _extra_args_main_cache_type_for_budget(args) or cache_type_kv
    if emitted_type:
        args = [
            "--cache-type-k",
            emitted_type,
            "--cache-type-v",
            emitted_type,
            *args,
        ]
    return _effective_main_cache_types(args, env)


# What kv_cache_type_from_str accepts; module scope so the budget can check it early.
_VALID_CACHE_TYPES = frozenset(
    {"f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl", "f32"}
)


def _planned_scratch_cache_type(
    cache_type_kv: Optional[str],
    extra_args: Optional[Iterable[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Scratch is charged only under f16, so take the LIGHTER axis: the heavier one hides a quantized K behind f16 and silently drops the scratch."""
    return min(_planned_main_cache_types(cache_type_kv, extra_args, env), key = _kv_bytes_per_elem)


def _auto_mode_drops_mtp(
    req_mode: Optional[str],
    size_b: Optional[float],
    *,
    has_separate_drafter: bool = False,
) -> bool:
    """Embedded MTP regresses below _MTP_MIN_SIZE_B, so Auto drops it; a separate drafter is kept."""
    if has_separate_drafter:
        return False
    return req_mode == "auto" and size_b is not None and size_b < _MTP_MIN_SIZE_B


# glm5next (upstream glm5-next): MTP covers only the NextN block, so Auto keeps MTP and the fit reserves no copy; old builds retry without speculation.
_MLA_MTP_FAST_ARCHS = frozenset({"glm5next", "glm5-next"})
# Smallest context Auto shrinks to so a fast MLA head still launches.
_FAST_MTP_MIN_CTX = 65536
# Max share of the target-only Auto context a drafter may take at one slot.
_MTP_SLOT_CTX_TRIM = 0.10


def _arch_has_fast_mla_mtp(architecture: Optional[str]) -> bool:
    """Whether this MLA architecture's embedded MTP head is worth promoting in Auto."""
    return bool(architecture) and str(architecture).strip().lower() in _MLA_MTP_FAST_ARCHS


def _arch_mtp_skips_target_kv_copy(architecture: Optional[str]) -> bool:
    """Whether this MLA architecture's MTP context skips the duplicated target KV."""
    return _arch_has_fast_mla_mtp(architecture)


def _mla_mtp_auto_enabled() -> bool:
    """Off by default: MLA MTP runs ~2x slower than no speculation (duplicated KV, indexer recomputed); UNSLOTH_MLA_MTP_ENABLED=1 enables; forced modes ignore it."""
    return os.environ.get("UNSLOTH_MLA_MTP_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _extra_args_set_spec_type(extra_args: Optional[Iterable[str]]) -> bool:
    """User passed --spec-type / --spec-default? llama-server takes one
    --spec-type (comma-separated to chain), so suppress auto-emit."""
    return _extra_args_set_any_flag(extra_args, {"--spec-type", "--spec-default"})


# Shared with llama_server_args so detection and stripping cannot drift.
_GPU_OFFLOAD_OVERRIDE_FLAGS = _LAYER_OFFLOAD_FLAGS
# Extras keeping weights on CPU regardless of -ngl; -ncmoe 0 places nothing, so it is parsed.
_CPU_MOE_COUNT_FLAGS = frozenset({"-ncmoe", "--n-cpu-moe"})
# Dense twin of -ncmoe: a positive -ncffn moves the first N layers' FFN to CPU.
_CPU_FFN_COUNT_FLAGS = frozenset({"-ncffn", "--n-cpu-ffn"})
# User controls over llama.cpp's own fitter. Studio emits neither on Metal, so a launch carrying
# one is placed by a fitter the Metal memory probe did not run.
_FIT_CTX_FLAGS = frozenset({"-fitc", "--fit-ctx"})
_FIT_CONTROL_FLAGS = frozenset({"-fitt", "--fit-target"}) | _FIT_CTX_FLAGS
_FIT_CONTROL_ENV_VARS = ("LLAMA_ARG_FIT_TARGET", "LLAMA_ARG_FIT_CTX")
_CPU_PLACEMENT_PRESENCE_FLAGS = (_MOE_OFFLOAD_FLAGS - _CPU_MOE_COUNT_FLAGS) | frozenset(
    {"-ot", "--override-tensor"}
)
_CPU_PLACEMENT_FLAGS = _MOE_OFFLOAD_FLAGS | frozenset({"-ot", "--override-tensor"})
# Pass-through that re-places what a fit priced: the fitter, split, remote (RPC) devices,
# draft tensors moved onto a card of their own, or an unpriced vocoder GGUF.
_VRAM_FIT_VOIDING_FLAGS = (
    frozenset(
        {
            "-fit",
            "--fit",
            "--rpc",
            "-otd",
            "--override-tensor-draft",
            "--spec-draft-override-tensor",
            "-mv",
            "--model-vocoder",
        }
    )
    | _FIT_CONTROL_FLAGS
    | _TENSOR_SPLIT_FLAGS
    | _SPLIT_MODE_FLAGS
)
_VRAM_FIT_VOIDING_ENV_VARS = (
    "LLAMA_ARG_FIT",
    "LLAMA_ARG_RPC",
    *_FIT_CONTROL_ENV_VARS,
    "LLAMA_ARG_TENSOR_SPLIT",
    "LLAMA_ARG_SPLIT_MODE",
)
_THREAD_OVERRIDE_FLAGS = frozenset({"-t", "--threads"})
_GENERATION_CPU_AFFINITY_FLAGS = frozenset({"-C", "--cpu-mask", "-Cr", "--cpu-range"})
_TENSOR_SPLIT_FLAGS = frozenset({"--tensor-split", "-ts"})


def _extra_args_set_any_flag(extra_args: Optional[Iterable[str]], flags: Collection[str]) -> bool:
    if not extra_args:
        return False
    for raw in extra_args:
        flag = _flag_name(str(raw))
        if flag in flags:
            return True
    return False


def _explicit_device_ordinals_needed(extra_args: Optional[Iterable[str]]) -> Optional[int]:
    """Leading child ordinals the user's device flags name (highest ordinal + 1, tensor-split
    length), 0 when no flag names one; None when a value cannot be read or the pick is CPU only.
    argv wins over LLAMA_ARG_*."""
    try:
        device = _last_flag_value(extra_args, frozenset({"--device", "-dev"}))
        main_gpu = _last_flag_value(extra_args, frozenset({"--main-gpu", "-mg"}))
        split = _last_flag_value(extra_args, _TENSOR_SPLIT_FLAGS)
    except ValueError:
        return None
    device = device or str(os.environ.get("LLAMA_ARG_DEVICE", "")).strip() or None
    main_gpu = main_gpu or str(os.environ.get("LLAMA_ARG_MAIN_GPU", "")).strip() or None
    split = split or str(os.environ.get("LLAMA_ARG_TENSOR_SPLIT", "")).strip() or None
    need = 0
    for token in (device or "").split(","):
        token = token.strip()
        if not token:
            continue
        # CPU only ("none", like "cpu"): the child uses no GPU, so no GPU subset may be priced lighter.
        if token.lower() == "none":
            return None
        match = re.fullmatch(r"[A-Za-z]+(\d+)", token)
        if match is None:
            return None
        need = max(need, int(match.group(1)) + 1)
    if main_gpu is not None:
        try:
            need = max(need, int(main_gpu) + 1)
        except ValueError:
            return None
    if split is not None:
        need = max(need, len([part for part in re.split(r"[,/]", split) if part.strip()]))
    return need
