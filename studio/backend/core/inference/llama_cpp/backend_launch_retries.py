# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, launch planning, retries and fallbacks."""

import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time

from dataclasses import replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
)
from core.inference.llama_server_args import (
    _CACHE_RAM_FLAGS,
    _CTX_CHECKPOINTS_FLAGS,
    _DEVICE_FLAGS,
    _flag_name,
    _PARALLEL_FLAGS,
    fit_is_effectively_on,
    strip_shadowing_flags,
)
from utils.gpu_memory_events import invalidate_gpu_memory as _invalidate_gpu_memory
from utils.log_retention import prune_log_dir
from utils.process_lifetime import (
    child_popen_kwargs as _child_popen_kwargs,
    spawn_on_lifetime_thread as _spawn_on_lifetime_thread,
)
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

from .binary_runtime import (
    _CPU_RUNTIME_OWNER_FILE,
    _GGML_GPU_BACKEND_RE,
    _llama_lib_dir,
    _loader_path_var,
    _resolve_llama_binary,
    _sweep_abandoned_cpu_runtimes,
)
from .core_types import (
    GgufLoadIntent,
    _CpuFallbackRuntime,
)
from .draft_args import (
    _extra_args_mtp_draft_path,
    _extra_args_mtp_draft_source,
    _without_subsequence,
)
from .launch_flags import (
    _LLAMA_ARG_TRUE_FALSE_AUTO_VALUES,
    _LLAMA_ARG_TRUE_OR_AUTO_VALUES,
    _env_fixes_gpu_layers,
    _strip_flag_pairs,
)
from .launch_support import _metal_capable_host
from .log import logger
from .memory_policy import (
    _THREAD_OVERRIDE_FLAGS,
    _TP_UNMEASURED_CTX,
    _active_vram_fraction,
    _extra_args_set_any_flag,
    _vram_usable_mib,
)
from .moe_cache import (
    _MOE_CACHE_USER_OWNED_ENV,
    _MOE_CACHE_USER_OWNED_FLAGS,
)
from .placement_args import (
    _MMPROJ_OFFLOAD_ENV_VAR,
    _OVERRIDE_TENSOR_FLAGS,
    _paravirtual_draft_ngl_flag,
    _paravirtual_mmproj_pinnable,
    _paravirtual_strip_gpu_overrides,
    _resolved_mmproj_offload,
)
from .swa_cache import _swa_cache_path
from utils.child_stdio import utf8_child_env
from .binary_runtime import (
    _parse_listed_devices,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _LaunchRetryMixin:
    def _plan_tensor_parallel(
        self,
        gpus: list[tuple[int, int]],
        model_size: int,
        target_ctx: int,
        cache_type_kv: Optional[str] = None,
        n_parallel: int = 1,
        mtp_engaged: bool = False,
        mtp_overhead_fn: Optional[Callable[[int], int]] = None,
        mtp_flat_reserve_bytes: int = 0,
        max_target_ctx: Optional[int] = None,
        total_by_idx: Optional[dict[int, int]] = None,
        n_ubatch: Optional[int] = None,
        soft_overhead_bytes: int = 0,
        swa_full: bool = False,
        kv_unified: bool = True,
        flash_attn: bool = True,
        vram_fraction: Optional[float] = None,
        scratch_cache_type_kv: Optional[str] = None,
        explicit_ctx: bool = False,
    ) -> tuple[int, int, list[int], Optional[list[int]]]:
        """Plans a --split-mode tensor load, capping context to pooled VRAM; if KV cannot be sized it may shrink Auto but never overrule an explicit context."""
        _tp_frac = vram_fraction if vram_fraction is not None else _active_vram_fraction()

        _tp_sysmem_fallback = LlamaCppBackend._sysmem_fallback_risk()

        def _usable(idx: int, free_mib: int) -> float:
            # Clamped on both branches: this pool is summed, so a negative card must not fund another.
            t = total_by_idx.get(idx, 0) if total_by_idx else 0
            return max(
                0.0,
                _vram_usable_mib(free_mib, t, _tp_frac, sysmem_fallback = _tp_sysmem_fallback),
            )

        # Drop GPUs whose usable budget cannot hold the per-device compute buffer (OOM in TP).
        _reserve_bytes = self._estimate_compute_buffer_bytes(
            n_ubatch = n_ubatch, n_parallel = n_parallel, per_device_tensor = True
        )
        reserve_mib = (
            _reserve_bytes // (1024 * 1024)
            if _reserve_bytes > 0
            else self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
        )
        usable_gpus = [g for g in gpus if _usable(g[0], g[1]) >= reserve_mib]
        gpu_indices = sorted(idx for idx, _ in usable_gpus)
        if len(gpu_indices) < 2:
            # Tensor parallelism is meaningless below 2 GPUs; the caller drops it, so never emit a split.
            return (
                target_ctx if target_ctx > 0 else 4096,
                target_ctx if target_ctx > 0 else 4096,
                gpu_indices,
                None,
            )
        free_by_idx = {idx: free for idx, free in usable_gpus}
        usable_by_idx = {idx: _usable(idx, free_by_idx[idx]) for idx in gpu_indices}
        pool_mib = sum(usable_by_idx.values())
        # Flat MTP cushion for what the byte fn cannot size; without it the search OOMs.
        flat_mtp_bytes = max(0, mtp_flat_reserve_bytes)
        if mtp_engaged and mtp_overhead_fn is None:
            flat_mtp_bytes = max(flat_mtp_bytes, 2 * 1024**3)
        # Tensor mode has no --fit valve, so charge the soft overhead here too (pooled, once).
        kv_budget_b = (
            (pool_mib - len(gpu_indices) * reserve_mib) * 1024 * 1024
            - model_size
            - flat_mtp_bytes
            - max(0, soft_overhead_bytes)
        )

        def _mtp_at(ctx: int) -> int:
            return mtp_overhead_fn(ctx) if mtp_overhead_fn is not None else 0

        def _kv_at(ctx: int) -> int:
            return self._estimate_kv_cache_bytes(
                ctx,
                cache_type_kv,
                n_parallel = n_parallel,
                swa_full = swa_full,
                kv_unified = kv_unified,
                n_ubatch = n_ubatch,
                flash_attn = flash_attn,
            )

        # Tensor mode replicates the compute graph on every device, so growth is n_dev x per-device.
        # Priced from the lighter KV axis so a q4 axis keeps its dequant scratch.
        n_dev = len(gpu_indices)
        cc_cache_type = (
            scratch_cache_type_kv if scratch_cache_type_kv is not None else cache_type_kv
        )

        def _cc_ctx(ctx: int) -> int:
            # Same state _kv_at prices with: the mask is f32 whenever tensor mode does not answer True.
            return n_dev * self._compute_buffer_ctx_bytes(
                ctx, n_ubatch, cc_cache_type, flash_attn = flash_attn
            )

        def _fit_ctx(ctx: int) -> int:
            # Largest ctx whose KV, MTP reserve and compute fit the pooled budget; never above an explicit ctx.
            if self._can_estimate_kv() and ctx > 0:
                ctx_floor = min(2048, ctx)
                if kv_budget_b <= 0:
                    # If weights and buffers exceed the pool, use the floor; the load then falls back to layer split.
                    return ctx_floor

                def _consumer(c: int) -> int:
                    return _kv_at(c) + _mtp_at(c) + _cc_ctx(c)

                if _consumer(ctx) <= kv_budget_b:
                    return ctx
                lo, hi, best = ctx_floor, ctx, ctx_floor
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if _consumer(mid) <= kv_budget_b:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                return best
            # KV size unknown -> can't prove a safe cap; floor.
            return min(_TP_UNMEASURED_CTX, ctx) if ctx > 0 else _TP_UNMEASURED_CTX

        # Hardware ceiling for the UI bound, independent of an explicit small -c.
        max_ctx_target = max_target_ctx if (max_target_ctx and max_target_ctx > 0) else target_ctx
        max_available_ctx = _fit_ctx(max_ctx_target)
        effective_ctx = min(_fit_ctx(target_ctx), max_available_ctx)
        if not self._can_estimate_kv() and explicit_ctx and target_ctx > 0:
            # Never apply the guessed min() to an explicitly typed context.
            effective_ctx = target_ctx
            max_available_ctx = max(max_available_ctx, effective_ctx)

        min_usable_mib = min(usable_by_idx.values())
        kv_bytes = _kv_at(effective_ctx) if (self._can_estimate_kv() and effective_ctx > 0) else 0
        # The MTP reserve must also fit the even split: per-ctx bytes plus the same flat cushion.
        mtp_bytes = (_mtp_at(effective_ctx) if effective_ctx > 0 else 0) + flat_mtp_bytes
        # Context-linear compute is replicated per device, so charge the whole split, as kv_budget_b does.
        cc_bytes = _cc_ctx(effective_ctx) if effective_ctx > 0 else 0
        even_share_mib = (
            (model_size + kv_bytes + mtp_bytes + cc_bytes) / len(gpu_indices) / (1024 * 1024)
        )
        tensor_split: Optional[list[int]] = None
        if even_share_mib > (min_usable_mib - reserve_mib):
            # Subtract the replicated compute share per device, or the smaller card is overweighted.
            cc_per_dev_mib = (cc_bytes // len(gpu_indices)) // (1024 * 1024) if cc_bytes else 0
            adj = [
                max(0, int(usable_by_idx[i] - reserve_mib - cc_per_dev_mib)) for i in gpu_indices
            ]
            if sum(adj) > 0:
                tensor_split = adj
        return effective_ctx, max_available_ctx, gpu_indices, tensor_split

    def _tensor_split_fits_budget(
        self,
        split: list[float],
        gpus: list[tuple[int, int]],
        gpu_indices: list[int],
        model_size: int,
        effective_ctx: int,
        cache_type_kv: Optional[str] = None,
        n_parallel: int = 1,
        n_ubatch: Optional[int] = None,
        kv_unified: bool = True,
        flash_attn: bool = True,
        swa_full: bool = False,
        mtp_engaged: bool = False,
        mtp_overhead_fn: Optional[Callable[[int], int]] = None,
        mtp_flat_reserve_bytes: int = 0,
        total_by_idx: Optional[dict[int, int]] = None,
        vram_fraction: Optional[float] = None,
        soft_overhead_bytes: int = 0,
        scratch_cache_type_kv: Optional[str] = None,
    ) -> bool:
        """Mirrors the planner budget, including soft_overhead and the lighter K/V type, or it is
        optimistic."""
        if not split or sum(split) <= 0 or len(split) != len(gpu_indices):
            return False

        _tp_frac = vram_fraction if vram_fraction is not None else _active_vram_fraction()
        # Must match the planner's floor, or this passes splits the planner rejected.
        _tp_sysmem_fallback = LlamaCppBackend._sysmem_fallback_risk()

        def _usable(idx: int, free_mib: int) -> float:
            t = total_by_idx.get(idx, 0) if total_by_idx else 0
            return max(
                0.0,
                _vram_usable_mib(free_mib, t, _tp_frac, sysmem_fallback = _tp_sysmem_fallback),
            )

        free_by_idx = {idx: free for idx, free in gpus}
        # Fail closed: gpus and gpu_indices can disagree; an unpriceable budget is not a pass.
        if any(idx not in free_by_idx for idx in gpu_indices):
            return False
        usable_by_idx = {idx: _usable(idx, free_by_idx[idx]) for idx in gpu_indices}

        _reserve_bytes = self._estimate_compute_buffer_bytes(
            n_ubatch = n_ubatch, n_parallel = n_parallel, per_device_tensor = True
        )
        reserve_mib = (
            _reserve_bytes // (1024 * 1024)
            if _reserve_bytes > 0
            else self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
        )

        flat_mtp = max(0, mtp_flat_reserve_bytes)
        if mtp_engaged and mtp_overhead_fn is None:
            flat_mtp = max(flat_mtp, 2 * 1024**3)

        kv_bytes = (
            self._estimate_kv_cache_bytes(
                effective_ctx,
                cache_type_kv,
                n_parallel = n_parallel,
                swa_full = swa_full,
                kv_unified = kv_unified,
                n_ubatch = n_ubatch,
                flash_attn = flash_attn,
            )
            if self._can_estimate_kv() and effective_ctx > 0
            else 0
        )
        mtp_bytes = (
            mtp_overhead_fn(effective_ctx) if mtp_overhead_fn is not None else 0
        ) + flat_mtp
        # Replicated, not split: every device allocates the whole buffer, so charge it flat per device.
        cc_per_device_bytes = self._compute_buffer_ctx_bytes(
            effective_ctx,
            n_ubatch,
            scratch_cache_type_kv or cache_type_kv,
            # Same state as the KV above.
            flash_attn = flash_attn,
        )
        total_bytes = model_size + kv_bytes + mtp_bytes + max(0, soft_overhead_bytes)

        total_weight = sum(split)
        for i, idx in enumerate(gpu_indices):
            alloc_bytes = total_bytes * split[i] / total_weight
            capacity_bytes = (usable_by_idx[idx] - reserve_mib) * 1024 * 1024 - cc_per_device_bytes
            if alloc_bytes > capacity_bytes:
                return False
        return True

    @staticmethod
    def _with_mmproj_offload_disabled(
        cmd: Sequence[str], env: Optional[Mapping[str, str]] = None
    ) -> Optional[List[str]]:
        """Returns a retry argv with the projector pinned to CPU at the end, since llama.cpp boolean flags are last-wins; None when no projector or already on CPU."""
        args = [str(arg) for arg in cmd]

        def _flag(token: str) -> str:
            return token.split("=", 1)[0].replace("_", "-").lower()

        if not any(_flag(arg) in ("--mmproj", "-mm") for arg in args):
            return None
        # Only a resolved False means on CPU; argv and env at arg.cpp precedence (LLAMA_ARG_NO_MMPROJ_OFFLOAD, disabled token); explicit --mmproj-offload is overridden.
        if _resolved_mmproj_offload(args, env or {}) is False:
            return None
        placements = [
            _flag(arg) for arg in args if _flag(arg) in ("--mmproj-offload", "--no-mmproj-offload")
        ]
        if not placements:
            # Deliberate superset of the resolution above: arg.cpp rejects 'no', so no retry helps.
            env_value = str((env or {}).get(_MMPROJ_OFFLOAD_ENV_VAR) or "").strip().lower()
            if env_value in ("0", "false", "off", "no"):
                return None
        return args + ["--no-mmproj-offload"]

    @staticmethod
    def _is_gpu_memory_start_failure(output: str) -> bool:
        """Whether startup output ties allocation pressure to a GPU backend."""
        lines = (output or "").lower().splitlines()
        allocation_markers = (
            "out of memory",
            "failed to allocate",
            "cudamalloc failed",
            "hiperroroutofmemory",
            "vk_error_out_of_device_memory",
        )
        gpu_markers = (
            "cuda",
            "hiperror",
            "hipmalloc",
            "ggml_backend_hip",
            "rocm",
            "vulkan",
            "vk_",
            "metal",
            "sycl",
            "oneapi",
            "musa",
            "gpu",
            "device memory",
        )
        return any(
            any(marker in line for marker in allocation_markers)
            and any(marker in line for marker in gpu_markers)
            for line in lines
        )

    @staticmethod
    def _is_projector_incompatibility(output: str) -> bool:
        """Matches only projector-format errors, so OOM or bad-GGUF failures keep their own handling."""
        text = (output or "").lower()
        if any(
            m in text
            for m in (
                "unknown projector type",
                "unsupported projector",
                "unsupported mmproj",
            )
        ):
            return True
        # Builds that phrase it via clip.cpp without the exact words above.
        return (
            "clip" in text
            and "projector" in text
            and ("unknown" in text or "unsupported" in text or "not supported" in text)
        )

    @staticmethod
    def _mmproj_retry_failure_message(*, projector_confirmed: bool, detail: str) -> str:
        """Only a confirmed projector mismatch keeps the historical wording; bare signal crashes do not."""
        if projector_confirmed:
            return (
                "Vision projector incompatible with this llama.cpp "
                "build, and the text-only retry also failed: " + detail
            )
        return (
            "Vision model failed to start (llama-server crashed with "
            "--mmproj), and the text-only retry also failed: " + detail
        )

    @staticmethod
    def _output_has_nonprojector_diagnostic(output: str) -> bool:
        """Output naming a real non-projector cause, like OOM, must surface instead of a text-only retry."""
        text = (output or "").lower()
        return any(
            m in text
            for m in (
                "out of memory",
                "failed to allocate",
                "unknown model architecture",
                "split_mode_tensor not implemented",
                LlamaCppBackend._TENSOR_QUANT_KV_UNSUPPORTED_MARKER,
            )
        )

    # llama.cpp before b9455 refused quantized KV under --split-mode tensor with a clean exit 1;
    # this marker avoids two doomed loads on older binaries.
    _TENSOR_QUANT_KV_UNSUPPORTED_MARKER = (
        "split_mode_tensor and kv cache quantization not implemented"
    )

    @classmethod
    def _is_tensor_quant_kv_unsupported(cls, output: str) -> bool:
        """Matches the quantized-KV tensor-mode refusal; no retry helps, only a different llama.cpp
        build."""
        return cls._TENSOR_QUANT_KV_UNSUPPORTED_MARKER in (output or "").lower()

    @staticmethod
    def _is_tensor_split_assert(output: str) -> bool:
        """True only for the #6415 split-axis warmup assert (GGML_BACKEND_SPLIT_AXIS_*),
        not any ggml assert/abort, so an unrelated invariant isn't cached. stderr is
        merged into output."""
        text = (output or "").lower()
        if "ggml_assert" not in text and "ggml_abort" not in text:
            return False
        # the split-axis enum token, unique to this assert (not the source file).
        return "split_axis" in text

    # Matched on basename so a path component cannot stand in for the object.
    _ROCM_OBJECT_HINTS = ("libamdhip", "libamd_comgr", "libhsa-runtime", "libhip", "libroc")

    # Paired with the binary revision so an old proof does not apply to a newly swapped build.
    _bundle_only_rocm_dirs: dict[str, tuple] = {}
    _bundle_only_rocm_lock = threading.Lock()

    @staticmethod
    def _remember_bundle_only_rocm(binary: str) -> None:
        resolved = _resolve_llama_binary(binary)
        revision = LlamaCppBackend._binary_stamp(resolved)
        with LlamaCppBackend._bundle_only_rocm_lock:
            LlamaCppBackend._bundle_only_rocm_dirs[str(resolved.parent)] = revision

    @staticmethod
    def _prefers_bundle_only_rocm(binary: str) -> bool:
        resolved = _resolve_llama_binary(binary)
        revision = LlamaCppBackend._binary_stamp(resolved)
        with LlamaCppBackend._bundle_only_rocm_lock:
            return LlamaCppBackend._bundle_only_rocm_dirs.get(str(resolved.parent)) == revision

    _system_rocm_probe_lock = threading.Lock()

    # Build dirs whose system ROCm prepend was already probed, keyed like the latch above.
    _system_rocm_probed: dict[str, tuple] = {}

    @staticmethod
    def _system_rocm_hides_gpu(binary: str, system_dirs: "list[str]") -> bool:
        """True when llama-server lists no GPU with system ROCm first but does with its bundle.

        The prepend crash-proofs hosts whose bundled ROCR mismatches amdkfd (#7208), but
        where the system copy is the mismatched one HIP init fails silently and the server
        runs on CPU (#14906). No crash means the load_model retry never fires, so this asks
        the binary directly, once per build. Any other answer keeps the prepend.
        """
        resolved = _resolve_llama_binary(binary)
        key = str(resolved.parent)
        revision = LlamaCppBackend._binary_stamp(resolved)
        with LlamaCppBackend._system_rocm_probe_lock:
            # A caller that waited out the first probe re-reads its verdict from the latch.
            if LlamaCppBackend._system_rocm_probed.get(key) == revision:
                return LlamaCppBackend._prefers_bundle_only_rocm(binary)
            mixed = LlamaCppBackend._list_devices_under(
                binary,
                LlamaCppBackend._llama_server_env_for_binary(binary, verify_system_rocm = False),
            )
            bundled = None
            if mixed == []:
                bundled = LlamaCppBackend._list_devices_under(
                    binary,
                    LlamaCppBackend._llama_server_env_for_binary(binary, use_system_rocm = False),
                )
            LlamaCppBackend._system_rocm_probed[key] = revision
            if not bundled:
                return False
            logger.warning(
                "llama-server lists no GPU with the system ROCm libraries (%s) ahead of its "
                "bundled runtime, but lists %s without them; using the bundled ROCm runtime.",
                ":".join(system_dirs),
                ", ".join(bundled),
            )
            LlamaCppBackend._remember_bundle_only_rocm(binary)
            return True

    @staticmethod
    def _list_devices_under(binary: str, env: Mapping[str, str]) -> Optional[list[str]]:
        try:
            result = subprocess.run(
                [binary, "--list-devices"],
                capture_output = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 30,
                env = utf8_child_env(dict(env)),
                **_windows_hidden_subprocess_kwargs(),
            )
        except Exception as e:
            logger.debug(f"llama-server --list-devices failed: {e}")
            return None
        if result.returncode != 0:
            return None
        return _parse_listed_devices(result.stdout)

    @staticmethod
    def _bundled_hip_symbol_miss(output: str) -> "Optional[tuple[str, str]]":
        """Splits glibc's symbol lookup error at ': undefined symbol:', so paths with spaces stay whole."""
        match = re.search(
            r"symbol lookup error:[ \t]*([^\r\n]{1,4096}?):"
            r"[ \t]*undefined symbol:[ \t]*([^\s,]+)",
            output or "",
        )
        if not match:
            return None
        obj = os.path.basename(match.group(1)).lower()
        if not obj.startswith(LlamaCppBackend._ROCM_OBJECT_HINTS):
            return None
        return match.group(1), match.group(2)

    @staticmethod
    def _is_bundled_hip_rocr_mismatch(output: str) -> bool:
        return LlamaCppBackend._bundled_hip_symbol_miss(output) is not None

    @staticmethod
    def _is_sched_reserve_abort(output: str) -> bool:
        """GGML_ASSERT(*cur_backend_id != -1); matches backtrace frames, which outlive the [New LWP] dump."""
        text = (output or "").lower()
        if "ggml_assert" not in text and "ggml_abort" not in text:
            return False
        # The #6415 split-axis abort shares the frame but has its own latch.
        if "split_axis" in text:
            return False
        if "cur_backend_id" in text:
            return True
        # Never sched_reserve / graph_reserve alone (reserve-time OOM passes through them).
        if "failed to initialize context" in text or "ggml_assert(" in text:
            return False
        return "ggml_backend_sched_split_graph" in text

    @staticmethod
    def _sched_reserve_abort_message() -> str:
        return (
            "llama.cpp aborted while reserving the compute graph "
            "(GGML_ASSERT(*cur_backend_id != -1)): no backend can place part of this model's "
            "graph. Either this llama.cpp build lacks an operation the model needs on this "
            "device (common for new attention types on CPU), or the model does not fit in "
            "memory. Try `unsloth studio update` for a newer llama.cpp, a smaller "
            "quantization, a lower context length, or turning off speculative decoding."
        )

    @staticmethod
    def _is_signal_crash(returncode: Optional[int]) -> bool:
        """Hard faults only: SIGKILL, SIGTERM and SIGINT are excluded since they mean OOM kill or unload."""
        if returncode is None:
            return False
        if returncode >= 0xC0000000:  # Windows access violation / illegal instruction
            return True
        return -returncode in (4, 6, 7, 8, 11)  # SIGILL SIGABRT SIGBUS SIGFPE SIGSEGV

    @staticmethod
    def _is_abort_exit(returncode: Optional[int]) -> bool:
        """Windows CRT abort() exit code (3) from GGML_ASSERT on MSVC -- not a POSIX
        signal or 0xC0000000+ NTSTATUS."""
        return returncode == 3

    @classmethod
    def _should_record_tensor_split_abort(cls, returncode: Optional[int], output: str) -> bool:
        """The #6415 split-axis abort: the marker plus a hard crash (POSIX signal or
        Windows abort exit). Marker required so a generic crash isn't cached."""
        return cls._is_tensor_split_assert(output) and (
            cls._is_signal_crash(returncode) or cls._is_abort_exit(returncode)
        )

    # ROCm arch-mismatch crash: deterministic, only a different device helps.
    # clr reports it with four different codes.
    _KERNEL_IMAGE_INVALID_MARKERS = (
        "device kernel image is invalid",  # hipErrorInvalidImage
        "no kernel image is available for execution",  # hipErrorNoBinaryForGpu
        "invalid kernel file",  # hipErrorInvalidKernelFile
        "invalid device function",  # hipErrorInvalidDeviceFunction
    )

    @classmethod
    def _kernel_image_invalid(cls, output: str) -> bool:
        # Case-folded, since hipGetErrorString is lowercase but the layers that reprint it vary in case.
        text = (output or "").lower()
        return any(marker in text for marker in cls._KERNEL_IMAGE_INVALID_MARKERS)

    # #12842. A HIP build prints "ROCm error:", so the #7624 crash never matches. Every
    # kernel fails alike, so no fit, flash-attn, slot or drafter retry can help.
    _CUDA_KERNEL_IMAGE_RE = re.compile(
        r"\bCUDA error: (device kernel image is invalid|no kernel image is available for execution)",
        re.IGNORECASE,
    )

    @classmethod
    def _cuda_kernel_image_error(cls, output: str) -> Optional[str]:
        """The CUDA kernel-image error text in ``output`` (lowercased), or None."""
        match = cls._CUDA_KERNEL_IMAGE_RE.search(output or "")
        return match.group(1).lower() if match else None

    @staticmethod
    def _cuda_install_driver_version(binary: Optional[str]) -> Optional[tuple[int, int]]:
        """The driver CUDA version the installer recorded for the managed install that
        owns ``binary`` (UNSLOTH_PREBUILT_INFO.json host_profile), or None."""
        try:
            from utils.llama_cpp_update import _llama_install_root

            root = _llama_install_root(binary) if binary else None
            if root is None:
                return None
            marker = json.loads((root / "UNSLOTH_PREBUILT_INFO.json").read_text(encoding = "utf-8"))
            driver = (marker.get("host_profile") or {}).get("driver_cuda_version")
            if isinstance(driver, list) and len(driver) == 2:
                return (int(driver[0]), int(driver[1]))
        except Exception:
            pass
        return None

    @classmethod
    def _cuda_kernel_image_message(
        cls,
        error: str,
        binary: Optional[str],
        log_path: "Optional[Path | str]" = None,
    ) -> str:
        remedy = cls._runtime_remedy(binary)
        log_hint = f" Full log: {log_path}" if log_path else ""
        if error == "no kernel image is available for execution":
            return (
                'llama-server could not load its CUDA kernels ("no kernel image is available '
                "for execution\"): this llama.cpp CUDA build has no kernels for this GPU's "
                "architecture. This is not the GGUF file and not out of memory. "
                f"{remedy[0].upper()}{remedy[1:]}, or use the CPU or Vulkan backend.{log_hint}"
            )
        driver = cls._cuda_install_driver_version(binary)
        if driver is not None and driver >= (12, 4):
            return (
                'llama-server could not load its CUDA kernels ("device kernel image is '
                f'invalid") although this NVIDIA driver supports CUDA {driver[0]}.{driver[1]}: '
                "the llama.cpp CUDA libraries look damaged or mismatched. This is not the "
                f"GGUF file and not out of memory. {remedy[0].upper()}{remedy[1:]}.{log_hint}"
            )
        driver_text = (
            f" (this NVIDIA driver supports CUDA {driver[0]}.{driver[1]})" if driver else ""
        )
        return (
            'llama-server could not load its CUDA kernels ("device kernel image is '
            'invalid"): the NVIDIA driver is most likely too old for this llama.cpp CUDA '
            f"build{driver_text}, which needs CUDA 12.4 or newer. This is not the GGUF file "
            "and not out of memory. Update the NVIDIA driver to R550 or newer (CUDA 12.4+), "
            f"or {remedy}.{log_hint}"
        )

    @classmethod
    def _arch_crash_retry_gpu_ids(cls, selected, enumerated) -> list[int]:
        """Prefers GPUs the failed launch never touched; else drops unified-memory devices, where
        this recurs."""
        _selected = set(selected or ())
        if not _selected:
            return []
        remaining = sorted(i for i in set(enumerated or ()) if i not in _selected)
        if remaining or len(_selected) < 2:
            return remaining
        try:
            discrete = sorted(_selected - cls._rocm_unified_memory_gpu_ids())
        except Exception:
            return []
        return discrete if 0 < len(discrete) < len(_selected) else []

    @staticmethod
    def _cache_tuning_target_unknown(
        extra_args: Optional[Iterable[str]], device_flags_owned: bool, env: Mapping[str, str]
    ) -> bool:
        """Fails closed: a user device pick can redirect the child, so the target device is unknown."""
        if device_flags_owned:
            return False
        return bool(
            _extra_args_set_any_flag(extra_args, _DEVICE_FLAGS)
            or str(env.get("LLAMA_ARG_DEVICE", "")).strip()
        )

    @staticmethod
    def _retry_cache_tuning_flags(
        cmd: list[str],
        *,
        cache_ram: Optional[int],
        ctx_checkpoints: Optional[int],
        server_caps: Mapping[str, Any],
    ) -> list[str]:
        """Skips any cache setting the command already states, since appended flags win the last-
        wins parse."""
        # Short spellings state the setting too; appending after one would clobber the user's value.
        stated = {_flag_name(str(token)) for token in cmd}
        flags: list[str] = []
        if (
            cache_ram is None
            and server_caps.get("supports_cache_ram")
            and not (_CACHE_RAM_FLAGS & stated)
        ):
            flags.extend(["--cache-ram", "0"])
        checkpoints_flag = server_caps.get("ctx_checkpoints_flag")
        if ctx_checkpoints is None and checkpoints_flag and not (_CTX_CHECKPOINTS_FLAGS & stated):
            flags.extend([str(checkpoints_flag), "0"])
        return flags

    @staticmethod
    def _without_flag_pairs(cmd: list[str], pairs: list[str]) -> list[str]:
        """Safe only for pairs this process emitted: a two-token step would swallow a following user
        extra."""
        out = list(cmd)
        for index in range(0, len(pairs) - 1, 2):
            flag, value = pairs[index], pairs[index + 1]
            for at in range(len(out) - 1):
                if out[at] == flag and out[at + 1] == value:
                    del out[at : at + 2]
                    break
        return out

    @staticmethod
    def _without_tensor_split(cmd: list[str]) -> Optional[list[str]]:
        """Drops the flag: split shares are positional over visible devices, and masking re-indexes them."""
        return LlamaCppBackend._without_flags(cmd, ("--tensor-split", "-ts"))

    @staticmethod
    def _without_flags(cmd: list[str], names: Collection[str]) -> Optional[list[str]]:
        """Value-taking flags only: a valueless one such as --mlock would drop the next argv entry."""
        out: list[str] = []
        found = False
        skip = False
        for tok in cmd:
            if skip:
                skip = False
                continue
            token = str(tok)
            if _flag_name(token) in names:
                found = True
                skip = "=" not in token
                continue
            out.append(token)
        return out if found else None

    @staticmethod
    def _requires_symmetric_kv(kv_lora_rank, architecture) -> bool:
        """Whether llama.cpp rejects K != V for a model with this metadata.

        Mirrors ``is_mla() || arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41``:
        kv_lora_rank stands in for is_mla() (the MLA archs' converter writes it
        alongside the MLA head dims), and deepseek4 / deepseek41 are checked by name
        because they set neither.
        """
        if kv_lora_rank is not None:
            return True
        return (architecture or "").strip().lower() in ("deepseek4", "deepseek41")

    def _target_kv_symmetry(self) -> bool:
        """Whether the TARGET model requires K == V."""
        return self._requires_symmetric_kv(self._kv_lora_rank, getattr(self, "_architecture", None))

    def _draft_kv_symmetry(
        self,
        cmd: Optional[Iterable[str]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> Optional[bool]:
        """Reads the drafter the command names, not the stored sidecar; an unreadable one counts as True."""
        path, is_remote = _extra_args_mtp_draft_source(list(cmd or []), env)
        if not path:
            return None
        if is_remote:  # a repo id is not a local file whose metadata can be read
            return True
        try:
            db = self._draft_backend_for(path)
        except Exception:
            db = None
        if db is None:  # named but unreadable -> conservative, not the target
            return True
        return self._requires_symmetric_kv(
            getattr(db, "_kv_lora_rank", None), getattr(db, "_architecture", None)
        )

    @staticmethod
    def _reset_quantized_v_cache(
        cmd: list[str],
        reason: str,
        *,
        mla: bool = False,
        draft_mla: Optional[bool] = None,
    ) -> list[str]:
        """Quantized V needs flash attention, so it goes to f16; MLA and DeepSeek4 reset K to f16 as
        well."""
        out = list(cmd)
        _v_cache_flags = (
            "--cache-type-v",
            "-ctv",
            "--cache-type-v-draft",
            "--spec-draft-type-v",
            "-ctvd",
        )
        _cache_reset = False
        for i, tok in enumerate(out):
            # llama.cpp rewrites '_' to '-' in '--' flags, so match underscore aliases too.
            name = _flag_name(tok)
            if name not in _v_cache_flags:
                continue
            if "=" in tok:
                flag, _, value = tok.partition("=")
                if value.strip().lower() not in LlamaCppBackend._NON_QUANTIZED_KV_TYPES:
                    out[i] = f"{flag}=f16"
                    _cache_reset = True
            elif i + 1 < len(out):
                if out[i + 1].strip().lower() not in LlamaCppBackend._NON_QUANTIZED_KV_TYPES:
                    out[i + 1] = "f16"
                    _cache_reset = True
        # MLA rejects K != V, so K follows V; gated per model, and an unreadable drafter counts as MLA.
        _main_k_flags = ("--cache-type-k", "-ctk")
        _draft_k_flags = ("--cache-type-k-draft", "--spec-draft-type-k", "-ctkd")
        if draft_mla is None:
            draft_mla = mla
        _k_cache_flags = (_main_k_flags if mla else ()) + (_draft_k_flags if draft_mla else ())
        _k_reset = False
        for i, tok in enumerate(out):
            if _flag_name(tok) not in _k_cache_flags:
                continue
            if "=" in tok:
                flag, _, value = tok.partition("=")
                if value.strip().lower() not in LlamaCppBackend._NON_QUANTIZED_KV_TYPES:
                    out[i] = f"{flag}=f16"
                    _k_reset = True
            elif i + 1 < len(out):
                if out[i + 1].strip().lower() not in LlamaCppBackend._NON_QUANTIZED_KV_TYPES:
                    out[i + 1] = "f16"
                    _k_reset = True
        if _cache_reset or _k_reset:
            logger.info(
                "KV cache dtype reset to f16 because flash attention is off (%s): a "
                "quantized V cache requires flash attention in llama.cpp. %s",
                reason,
                (
                    "The K cache went down with it: this model uses MLA, which "
                    "rejects different K and V cache types."
                    if _k_reset
                    else "The K cache is left untouched."
                ),
            )
        return out

    @staticmethod
    def _with_flash_attn_off(
        cmd: list[str],
        *,
        mla: bool = False,
        draft_mla: Optional[bool] = None,
    ) -> Optional[list[str]]:
        """Turns flash attention off; a bare --flash-attn is dropped, since older builds take no value."""
        out = list(cmd)

        def explicit(i):
            nxt = out[i + 1] if i + 1 < len(out) else None
            return nxt if nxt in _LLAMA_ARG_TRUE_FALSE_AUTO_VALUES else None

        effective = None
        for i, tok in enumerate(out):
            name = _flag_name(tok)
            if name in ("--flash-attn", "-fa") and "=" in tok:
                effective = tok.partition("=")[2]
            elif name in ("--flash-attn", "-fa"):
                effective = explicit(i) or "on"
        if effective not in _LLAMA_ARG_TRUE_OR_AUTO_VALUES:
            return None
        # A bare flag cannot take =off, so dropping the token disables FA; remove back to front.
        bare_at: list[int] = []
        for i, tok in enumerate(out):
            name = _flag_name(tok)
            if name in ("--flash-attn", "-fa") and "=" in tok:
                flag, _, value = tok.partition("=")
                if value in _LLAMA_ARG_TRUE_OR_AUTO_VALUES:
                    out[i] = f"{flag}=off"
            elif name in ("--flash-attn", "-fa"):
                if explicit(i) in _LLAMA_ARG_TRUE_OR_AUTO_VALUES:
                    out[i + 1] = "off"
                elif explicit(i) is None:  # bare flag (reads as on) -> drop it
                    bare_at.append(i)
        for i in reversed(bare_at):
            del out[i]

        # Quantized V is legal with FA on but would crash this FA-off retry.
        return LlamaCppBackend._reset_quantized_v_cache(
            out,
            "disabled by the crash-recovery fallback",
            mla = mla,
            draft_mla = draft_mla,
        )

    @staticmethod
    def _drop_env_quantized_v_cache(
        env: MutableMapping[str, str],
        *,
        mla: bool = False,
        draft_mla: Optional[bool] = None,
    ) -> bool:
        """Env-only quantized V caches would abort the FA-off retry; MLA also drops the K env vars."""
        dropped = False
        variables = ["LLAMA_ARG_CACHE_TYPE_V", "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V"]
        if draft_mla is None:
            draft_mla = mla
        if mla:
            variables.append("LLAMA_ARG_CACHE_TYPE_K")
        if draft_mla:
            variables.append("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K")
        for var in variables:
            value = (env.get(var) or "").strip().lower()
            if value and value not in LlamaCppBackend._NON_QUANTIZED_KV_TYPES:
                env.pop(var, None)
                dropped = True
        return dropped

    @staticmethod
    def _drop_env_flash_attn(env: MutableMapping[str, str]) -> bool:
        """Env is read before argv, and older builds cannot negate a bare --flash-attn; drop the env var."""
        return env.pop("LLAMA_ARG_FLASH_ATTN", None) is not None

    # Matched on wording, not architecture name.
    _KV_UNIFIED_REFUSED_MARKERS = (
        "a unified kv cache is only supported with a single sequence",
        "needs one sequence per stream",
    )

    @staticmethod
    def _is_kv_unified_refused(output: str) -> bool:
        low = (output or "").lower()
        return any(marker in low for marker in LlamaCppBackend._KV_UNIFIED_REFUSED_MARKERS)

    @staticmethod
    def _with_single_sequence(cmd: list[str]) -> Optional[list[str]]:
        """Reverses Studio's own --kv-unified choice above one slot; every parallel-flag alias is
        removed."""
        out: list[str] = []
        saw_kv_unified = False
        multi_slot = False
        skip_value = False
        for i, tok in enumerate(cmd):
            if skip_value:
                skip_value = False
                continue
            name = _flag_name(tok)
            if name in ("--kv-unified", "-kvu"):
                saw_kv_unified = True
                # The flag is bare, but an inline "=1" or a separate "1" reach us.
                if (
                    "=" not in tok
                    and cmd[i + 1 : i + 2]
                    # Case-sensitive, like llama.cpp's own bool parse.
                    and cmd[i + 1] in _LLAMA_ARG_TRUE_FALSE_AUTO_VALUES
                ):
                    skip_value = True
                continue
            if name in _PARALLEL_FLAGS:
                if "=" in tok:
                    value = tok.partition("=")[2]
                elif tok != name:
                    # The attached short, -np8, which _flag_name peels to -np.
                    value = tok[len(name) :]
                else:
                    # Valueless --parallel: eating the next token would delete a flag.
                    value = cmd[i + 1] if i + 1 < len(cmd) else ""
                    skip_value = _flag_name(value) is None
                try:
                    multi_slot = multi_slot or int(value.strip()) > 1
                except (AttributeError, TypeError, ValueError):
                    pass
                out.extend([name, "1"])
                continue
            out.append(tok)
        if not saw_kv_unified and not multi_slot:
            return None
        if not any(_flag_name(tok) in _PARALLEL_FLAGS for tok in out):
            out.extend(["--parallel", "1"])
        return out

    @staticmethod
    def _drop_env_single_sequence(env: MutableMapping[str, str]) -> bool:
        """Drops the env twins instead of negating them, since --no-kv-unified needs every build to
        know it."""
        dropped = env.pop("LLAMA_ARG_KV_UNIFIED", None) is not None
        return env.pop("LLAMA_ARG_N_PARALLEL", None) is not None or dropped

    @staticmethod
    def _strip_mmproj_args(cmd: list[str]) -> list[str]:
        """Remove explicit projector paths, including aliases, for managed emission or retry."""
        out: list[str] = []
        skip_value = False
        for tok in cmd:
            if skip_value:
                skip_value = False
                continue
            if _flag_name(tok) in {"--mmproj", "-mm"}:
                skip_value = "=" not in tok
                continue
            out.append(tok)
        return out

    @staticmethod
    def _vulkan_prebuilt_was_auto_selected(binary: Optional[str]) -> bool:
        """Only an auto-detected Vulkan bundle may fall back to CPU on a crash; a user-chosen one
        must not."""
        if not binary:
            return False
        from utils.llama_cpp_update import _llama_install_root
        from utils.prebuilt.llama_backend import (
            environment_backend_override,
            marker_backend_was_chosen,
        )

        # Same precedence the installer used; only a concrete override counts as a choice.
        override = environment_backend_override(
            os.environ.get("UNSLOTH_LLAMA_CPP_BACKEND"),
            os.environ.get("UNSLOTH_FORCE_VULKAN"),
        )
        if override is not None and override != "auto":
            return False
        try:
            install_root = _llama_install_root(binary)
            if install_root is None:
                return False
            marker = json.loads(
                (install_root / "UNSLOTH_PREBUILT_INFO.json").read_text(encoding = "utf-8")
            )
        except (OSError, ValueError, TypeError):
            return False
        # Reads new and pre-#7188 markers alike, including the force_cpu-only ones.
        return isinstance(marker, dict) and not marker_backend_was_chosen(marker)

    @staticmethod
    def _strip_cpu_fallback_main_placement(args: Iterable[str]) -> list[str]:
        """Remove every raw main-model placement choice the CPU replay replaces."""
        stripped = strip_shadowing_flags(
            args,
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = True,
            strip_offload = True,
            strip_device = True,
        )
        return _paravirtual_strip_gpu_overrides(stripped, log_dropped = False) or []

    @classmethod
    def _cpu_fallback_request_eligible(
        cls,
        intent: GgufLoadIntent,
        extra_args: Optional[Iterable[str]],
        env: Optional[Mapping[str, str]] = None,
        *,
        allow_manual_cpu: bool = False,
    ) -> bool:
        """Whether CPU recovery may replace the requested placement."""
        args = [str(arg) for arg in (extra_args or ())]
        companion_placement_flags = {
            *_OVERRIDE_TENSOR_FLAGS,
            "--spec-draft-device",
            "-devd",
            "--device-draft",
            "--spec-draft-ngl",
            "-ngld",
            "--gpu-layers-draft",
            "--n-gpu-layers-draft",
            "--mmproj-offload",
            "--no-mmproj-offload",
        }
        child_env = os.environ if env is None else env
        placement_eligible = intent.gpu_memory_mode == "auto" or (
            allow_manual_cpu and intent.gpu_memory_mode == "manual" and intent.gpu_layers == 0
        )
        return bool(
            placement_eligible
            and not intent.gpu_ids
            and not intent.tensor_parallel
            and not intent.tensor_split
            and intent.n_cpu_moe == 0
            and cls._strip_cpu_fallback_main_placement(args) == args
            and not _extra_args_set_any_flag(args, companion_placement_flags)
            and not any(
                str(child_env.get(name) or "").strip()
                for name in (
                    *cls._CPU_FALLBACK_MAIN_PLACEMENT_ENV_VARS,
                    "LLAMA_ARG_N_GPU_LAYERS_DRAFT",
                    "LLAMA_ARG_MMPROJ_OFFLOAD",
                    "GGML_BACKEND_PATH",
                )
            )
        )

    @classmethod
    def _auto_vulkan_cpu_fallback_eligible(
        cls,
        binary: Optional[str],
        intent: GgufLoadIntent,
        extra_args: Optional[Iterable[str]],
        env: Optional[Mapping[str, str]] = None,
        *,
        allow_manual_cpu: bool = False,
    ) -> bool:
        """Whether a signal may replace automatic Vulkan placement with CPU."""
        return bool(
            cls._cpu_fallback_request_eligible(
                intent, extra_args, env, allow_manual_cpu = allow_manual_cpu
            )
            and cls._is_vulkan_backend(binary)
            and cls._vulkan_prebuilt_was_auto_selected(binary)
        )

    @staticmethod
    def _as_cpu_fallback_intent(intent: GgufLoadIntent) -> GgufLoadIntent:
        return replace(
            intent,
            gpu_memory_mode = "manual",
            gpu_layers = 0,
            n_cpu_moe = 0,
            tensor_parallel = False,
            tensor_split = None,
            gpu_ids = None,
            cpu_fallback = True,
        )

    def _preserve_cpu_fallback_intent(
        self,
        intent: GgufLoadIntent,
        *,
        source_matches: bool = False,
    ) -> GgufLoadIntent:
        if not self._cpu_fallback_reason:
            return intent
        if not source_matches and not self.matches_load_source(intent):
            return intent
        # Decided by this load's own probe: only a zero-device answer takes the CPU route.
        if not source_matches and self._cpu_fallback_reason in (
            "vulkan_no_device",
            "vulkan_startup_stall",
        ):
            return intent
        extras = list(intent.extra_args) if intent.extra_args is not None else None
        # Judge the env the replay runs with (_cpu_isolated_replay pops placement vars).
        replay_env = {
            name: value
            for name, value in os.environ.items()
            if name not in self._CPU_FALLBACK_MAIN_PLACEMENT_ENV_VARS
            and name != "GGML_BACKEND_PATH"
        }
        if not self._cpu_fallback_request_eligible(
            intent,
            extras,
            replay_env,
            allow_manual_cpu = True,
        ):
            return intent
        return self._as_cpu_fallback_intent(intent)

    def _apply_cpu_fallback_state(
        self,
        intent: GgufLoadIntent,
        *,
        is_vision: bool,
        mmproj_has_audio: bool,
        disable_vision: bool,
        vision_disabled_by_user: bool,
        reason: str = "vulkan_startup_crash",
    ) -> GgufLoadIntent:
        intent = self._as_cpu_fallback_intent(intent)
        self._gpu_memory_mode = intent.gpu_memory_mode
        self._gpu_layers = intent.gpu_layers
        self._n_cpu_moe = intent.n_cpu_moe
        self._tensor_split = intent.tensor_split
        self._tensor_parallel = intent.tensor_parallel
        self._gpu_ids = intent.gpu_ids
        self._requested_gpu_ids = None
        self._layer_preserves_tensor_intent = False
        self._is_vision = is_vision
        self._mmproj_has_audio = mmproj_has_audio
        # Caller returns before the load assigns these; without it Vision state stays stale.
        self._disable_vision = bool(disable_vision)
        self._vision_disabled_by_user = bool(vision_disabled_by_user)
        self._cpu_fallback_reason = reason
        return intent

    @staticmethod
    def _launch_has_mmproj(cmd: Iterable[str], env: Mapping[str, str]) -> bool:
        """Whether this launch receives a projector from argv or the environment."""
        return _extra_args_set_any_flag(cmd, {"--mmproj", "-mm"}) or any(
            str(env.get(name) or "").strip()
            for name in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL")
        )

    @classmethod
    def _cpu_isolated_replay(
        cls,
        cmd: Iterable[str],
        env: dict[str, str],
        server_caps: Mapping[str, object],
        *,
        drop_full_offload_threads: bool = False,
    ) -> Optional[list[str]]:
        """Uses --device none, not -ngl 0, and returns None when a companion mmproj cannot be pinned
        to CPU."""
        original = [str(arg) for arg in cmd]
        has_mmproj = cls._launch_has_mmproj(original, env)
        if has_mmproj and not _paravirtual_mmproj_pinnable(server_caps):
            return None
        has_draft = _extra_args_mtp_draft_path(original, env = env) is not None
        draft_ngl_flag = _paravirtual_draft_ngl_flag(server_caps) if has_draft else None
        if has_draft and draft_ngl_flag is None:
            return None

        replay = cls._strip_cpu_fallback_main_placement(original)
        if drop_full_offload_threads:
            # Windows full-offload pins 2 threads; a CPU replay would decode on 2 cores.
            replay = _strip_flag_pairs(replay, _THREAD_OVERRIDE_FLAGS)
            for name in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY"):
                env.pop(name, None)
        replay.extend(["--gpu-layers", "0", "--fit", "off", "--device", "none"])
        if has_mmproj:
            replay.append("--no-mmproj-offload")
        if has_draft:
            replay.extend([str(draft_ngl_flag), "0", "--device-draft", "none"])

        for name in (*cls._CPU_FALLBACK_MAIN_PLACEMENT_ENV_VARS, "GGML_BACKEND_PATH"):
            env.pop(name, None)
        return replay

    def _undo_moe_spill_batch(self, argv: "list[str]") -> "list[str]":
        """Restores only the batch tokens Unsloth itself raised; otherwise argv comes back unchanged."""
        tokens = getattr(self, "_moe_spill_batch_tokens", None)
        if not tokens:
            return list(argv)
        raised, before = tokens
        argv = list(argv)
        for i in range(len(argv) - len(raised) + 1):
            if argv[i : i + len(raised)] == raised:
                return [*argv[:i], *before, *argv[i + len(raised) :]]
        return argv

    def _drop_moe_cache(
        self,
        argv: "list[str]",
        why: str = "",
    ) -> "list[str]":
        """``argv`` without the ``--moe-cache-mib auto`` this launch added; a user's is kept."""
        tokens = list(getattr(self, "_moe_cache_flags", None) or ())
        stripped = _without_subsequence(list(argv), tokens)
        if tokens and stripped != list(argv) and why:
            logger.info("MoE cache: dropping --moe-cache-mib auto for the %s.", why)
        return stripped

    def _moe_cache_auto_eligible(
        self,
        *,
        caps: Optional[Mapping[str, object]],
        gpu_memory_mode: str,
        use_fit: bool,
        experts_on_host: bool,
        discrete_gpu: bool,
        gpu_indices: Optional[Iterable[int]],
        detected_gpus: Optional[Iterable[tuple[int, int]]],
        is_vulkan_backend: bool,
        tensor_parallel: bool,
        extra_args: Optional[Iterable[str]],
        env: Optional[Mapping[str, str]] = None,
    ) -> bool:
        """Whether this launch may get ``--moe-cache-mib auto`` (the caller still checks RAM
        and the final load mode): a build with ``auto``, ``--fit on``, spilling experts, one
        discrete CUDA / ROCm GPU (the cache is single-device), and no user placement."""
        if not (caps or {}).get("supports_moe_cache_auto"):
            return False
        if gpu_memory_mode == "manual" or not use_fit or not experts_on_host:
            return False
        if not discrete_gpu or is_vulkan_backend or tensor_parallel or self.n_moe_layers <= 0:
            return False
        devices = (
            {int(i) for i in gpu_indices}
            if gpu_indices is not None
            else {int(idx) for idx, _free in (detected_gpus or ())}
        )
        if len(devices) != 1:
            return False
        source_env = os.environ if env is None else env
        if _extra_args_set_any_flag(extra_args, _MOE_CACHE_USER_OWNED_FLAGS):
            return False
        if any(str(source_env.get(name) or "").strip() for name in _MOE_CACHE_USER_OWNED_ENV):
            return False
        if _env_fixes_gpu_layers(source_env):
            return False
        try:
            return fit_is_effectively_on(list(extra_args or ()), source_env)
        except ValueError:  # a valueless --fit: llama-server refuses the launch anyway
            return False

    def _discrete_gpu_for_expert_spill(
        self,
        gpu_indices: Optional[Iterable[int]],
        detected_gpus: Optional[Iterable[tuple[int, int]]],
        shared_gpu_ids: Optional[Iterable[int]] = None,
    ) -> bool:
        """Unified memory has nothing to stream and a CPU launch has no GPU, so only a discrete GPU
        counts."""
        detected = list(detected_gpus or ())
        if not detected or _metal_capable_host():
            return False
        on = set(gpu_indices) if gpu_indices else {idx for idx, _free in detected}
        if on and on <= set(shared_gpu_ids or ()):
            return False
        try:
            if self._amd_apu_wants_unified_memory(
                gpu_indices
            ) or self._integrated_cuda_unified_memory(gpu_indices):
                return False
        except Exception as e:
            logger.debug(f"unified-memory probe failed ({e}); not raising the micro-batch")
            return False
        return True

    def _prepare_cpu_fallback_launch(
        self,
        binary: Optional[str],
        cmd: Iterable[str],
        env: dict[str, str],
        server_caps: Mapping[str, object],
        *,
        drop_full_offload_threads: bool = False,
        avail_mib: Optional[int] = None,
    ) -> Optional[tuple[list[str], Optional[str], Optional[str]]]:
        """Returns (argv, cwd, pageable_note) or None; avail_mib is shared so rewrite and repricing
        agree."""
        if not self._is_vulkan_backend(binary):
            return None
        replay = self._cpu_isolated_replay(
            cmd, env, server_caps, drop_full_offload_threads = drop_full_offload_threads
        )
        if replay is None:
            return None
        # Replay runs on no GPU, so give the CPU attempt llama.cpp's auto mmap back.
        if self._fit_load_mode_flags:
            replay = _without_subsequence(replay, self._fit_load_mode_flags)
            logger.info(
                "Load mode: dropping the fit's --load-mode none for the CPU "
                "fallback; it runs entirely from host RAM."
            )
        replay = self._drop_managed_lazy_mode(replay, "CPU fallback")
        # And the managed pair: this replay appends "--device none".
        replay = self._drop_managed_dio(
            replay, "the CPU fallback runs entirely from host RAM", clear_record = False
        )
        # Nothing to stream without a GPU; the expert cache needs one too.
        replay = self._undo_moe_spill_batch(replay)
        replay = self._drop_moe_cache(replay, "CPU fallback")
        # A user's own "--load-mode none" / "--no-mmap" survives that strip, by design,
        # and on this rung it is no longer the mode they were priced for: the replay
        # appends "--gpu-layers 0 --fit off --device none", so nothing credits VRAM and
        # the child holds the WHOLE model in host RAM. Unmapped that is one allocation
        # of the whole file rather than a mapping the kernel pages, i.e. an OOM kill on
        # a host that cannot hold it -- which the main launch path would have rewritten
        # before spawning. So price the replay on its OWN footprint (never the preflight
        # verdict, which credited a GPU placement this argv has just deleted) and apply
        # the same override.
        _priced_replay = replay
        replay, pageable_note = self._page_an_oversized_unmapped_load(
            replay,
            env,
            # UNSLOTH_ALLOW_HOST_OFFLOAD silences a message; it must not withdraw the rewrite.
            lambda: bool(
                self._launch_host_shortfall_message(
                    _priced_replay,
                    (),
                    env,
                    child_has_no_gpu = True,
                    avail_mib = avail_mib,
                    honor_opt_out = False,
                )
            ),
        )
        cpu_binary = self._cpu_isolated_binary(binary)
        if cpu_binary is None:
            return None
        loader_env = self._llama_server_env_for_binary(cpu_binary)
        loader_path = _loader_path_var()
        env[loader_path] = loader_env[loader_path]
        replay[0] = cpu_binary
        # Keep the working directory: relative paths in user extra args resolve against it.
        return replay, None, pageable_note

    def _cpu_isolated_binary(self, binary: Optional[str]) -> Optional[str]:
        """Backend libs load before --device none, so the CPU copy is staged without GPU backend libs."""
        if not binary or not self._is_llama_install_tree(binary):
            return None
        source_binary = _resolve_llama_binary(binary)
        source_stamp = self._binary_stamp(source_binary)
        runtime = self._cpu_fallback_runtime
        if (
            runtime is not None
            and runtime.source_binary == source_binary
            and runtime.source_stamp == source_stamp
        ):
            if runtime.staged_binary.is_file():
                return str(runtime.staged_binary)
        self._cleanup_cpu_fallback_runtime()
        staged_runtime = None
        try:
            runtime_root = _swa_cache_path().parent / "runtime"
            runtime_root.mkdir(parents = True, exist_ok = True)
            _sweep_abandoned_cpu_runtimes(runtime_root)
            staged_runtime = tempfile.TemporaryDirectory(
                prefix = "llama-cpu-",
                dir = runtime_root,
            )
            staged_dir = Path(staged_runtime.name)
            # kill -9 skips the atexit hook, so stamp the owner for later collection.
            (staged_dir / _CPU_RUNTIME_OWNER_FILE).write_text(str(os.getpid()), encoding = "utf-8")
            lib_dir = _llama_lib_dir(str(source_binary))
            gpu_backend = _GGML_GPU_BACKEND_RE
            cpu_library_present = False
            for source in lib_dir.iterdir():
                try:
                    resolved = source.resolve(strict = True)
                except OSError:
                    continue
                name = source.name
                is_shared = name.endswith((".dll", ".dylib")) or ".so" in name
                if resolved != source_binary and not is_shared:
                    continue
                if gpu_backend.match(name.lower()):
                    continue
                destination = staged_dir / name
                if destination.exists():
                    continue
                try:
                    os.link(resolved, destination)
                except OSError:
                    shutil.copy2(resolved, destination)
                if name.lower().startswith(("libggml-cpu", "ggml-cpu")):
                    cpu_library_present = True
            staged_binary = staged_dir / source_binary.name
            if not staged_binary.is_file() or not cpu_library_present:
                staged_runtime.cleanup()
                return None
        except (OSError, ValueError):
            if staged_runtime is not None:
                with contextlib.suppress(Exception):
                    staged_runtime.cleanup()
            return None
        self._cpu_fallback_runtime = _CpuFallbackRuntime(
            tempdir = staged_runtime,
            source_binary = source_binary,
            staged_binary = staged_binary,
            source_stamp = source_stamp,
        )
        return str(staged_binary)

    @staticmethod
    def _binary_stamp(path: Path) -> tuple:
        try:
            stat = path.stat()
        except OSError:
            return ()
        return (stat.st_ino, stat.st_size, stat.st_mtime_ns)

    @staticmethod
    def _binary_revision(binary: Optional[str]) -> tuple:
        """Path cannot tell the build, as updates swap the file in place; () when unreadable means
        unchanged."""
        if not binary:
            return ()
        stamp = LlamaCppBackend._binary_stamp(Path(binary))
        return (str(binary),) + stamp if stamp else ()

    def _cleanup_cpu_fallback_runtime(self) -> None:
        runtime = getattr(self, "_cpu_fallback_runtime", None)
        self._cpu_fallback_runtime = None
        # A failed cleanup() detaches its finalizer, so retry it on the next call.
        pending = getattr(self, "_pending_cpu_fallback_cleanups", [])
        if runtime is not None:
            pending.append(runtime.tempdir)
        retry = []
        for tempdir in pending:
            try:
                tempdir.cleanup()
            except Exception:
                retry.append(tempdir)
        self._pending_cpu_fallback_cleanups = retry

    def _cleanup_failed_cpu_fallback(self) -> None:
        """Remove every process and staged artifact from a terminal CPU replay."""
        self._kill_process()
        self._cpu_fallback_reason = None
        self._cleanup_cpu_fallback_runtime()

    @staticmethod
    def _gpu_init_crash_message(binary: Optional[str] = None) -> str:
        """Return backend-specific advice for a GPU initialization crash."""
        base = (
            "llama-server crashed at startup on both the vision and text-only "
            "attempts -- a GPU driver/runtime initialization crash, not a model "
            "or vision-projector problem. "
        )
        backends = LlamaCppBackend._installed_ggml_backends(binary)
        backend = next((name for name in ("cuda", "hip", "vulkan") if name in backends), None)
        if backend == "vulkan":
            return base + (
                "This build uses Vulkan, which Unsloth selects when it detects an "
                "Intel or AMD GPU. Older integrated GPUs and their drivers can fault "
                "inside Vulkan device initialization. Update your Vulkan driver (Mesa "
                "on Linux), or reinstall llama.cpp on CPU with "
                "UNSLOTH_LLAMA_CPP_BACKEND=cpu unsloth studio update."
            )
        if backend == "hip":
            return base + (
                "This often means an unsupported secondary GPU; on AMD/ROCm, hide it "
                "with ROCR_VISIBLE_DEVICES (e.g. ROCR_VISIBLE_DEVICES=0 exposes only "
                "the first GPU) before launching Unsloth Studio."
            )
        if backend == "cuda":
            return base + (
                "This often means an unsupported secondary GPU or a driver/CUDA "
                "runtime mismatch; hide the extra GPU with CUDA_VISIBLE_DEVICES (e.g. "
                "CUDA_VISIBLE_DEVICES=0 exposes only the first GPU) before launching "
                "Unsloth Studio, or update the NVIDIA driver."
            )
        return base + (
            "Check the llama-server log and your GPU driver, and make sure the "
            "installed llama.cpp build matches this machine's GPU."
        )

    @staticmethod
    def _redacted_cmd_for_log(cmd: "list[str]") -> "list[str]":
        """Copy of cmd with the value after --api-key replaced by <redacted>."""
        out = list(cmd)
        if "--api-key" in out:
            ki = out.index("--api-key") + 1
            if ki < len(out):
                out[ki] = "<redacted>"
        return out

    def _start_llama_process(
        self, cmd: list[str], env: dict, *, child_gpu_physical_ids: Optional[tuple[int, ...]]
    ) -> bool:
        """Returns False without spawning once teardown has begun, so callers stop instead of
        health-waiting."""
        # Drop an orphan Popen a concurrent load may have stored past Phase 1.
        self._kill_process()
        self._child_gpu_physical_ids = child_gpu_physical_ids

        self._stdout_lines = []
        # Tees llama-server output to a log file for post-mortems; the parent keeps only the last 50 lines.
        self._llama_log_fh = None
        try:
            log_dir = _swa_cache_path().parent / "logs" / "llama-server"
            log_dir.mkdir(parents = True, exist_ok = True)
            self._llama_log_path = log_dir / f"llama-{int(time.time())}-port-{self._port}.log"
            self._llama_log_fh = open(
                self._llama_log_path,
                "w",
                encoding = "utf-8",
                buffering = 1,
            )
            # After the open, so the cap counts this file and can never delete it.
            prune_log_dir(log_dir, "llama-*.log", protect = self._llama_log_path)
            logger.info(f"llama-server stdout/stderr -> {self._llama_log_path}")
        except (OSError, UnicodeDecodeError) as e:
            # Best-effort; never block the load on logging.
            logger.warning(f"Could not open llama-server log file: {e}")
            self._llama_log_path = None

        # Log the argv per attempt (the text-only mmproj retry re-enters here
        # with --mmproj stripped), redacting the API key.
        logger.info(f"Starting llama-server: {' '.join(self._redacted_cmd_for_log(cmd))}")

        # The mmproj text-only retry reaches a spawn without passing _spawn_and_wait.
        with self._spawn_lock:
            if self._spawn_is_stale():
                logger.info("app is shutting down; not starting llama-server")
                self._close_attempt_log()
                self._health_wait_cancelled = True
                return False
            # PDEATHSIG follows the forking thread; the MTP-crash reload's thread exits.
            _spawned = _spawn_on_lifetime_thread(
                lambda: subprocess.Popen(
                    cmd,
                    stdout = subprocess.PIPE,
                    stderr = subprocess.STDOUT,
                    text = True,
                    encoding = "utf-8",
                    errors = "replace",
                    env = env,
                    **_windows_hidden_subprocess_kwargs(),
                    **_child_popen_kwargs(),
                )
            )
            self._process = _spawned
            # The env picks companions too (LLAMA_ARG_MMPROJ, ...): load_progress sizes them from it.
            self._process_env = (_spawned, env)
            _invalidate_gpu_memory("llama-server spawned")
            # Serialise loads so two llama-servers never coexist; /unload is not blocked; lifecycle read first.
            self._note_server_pid(_spawned.pid)

        # A helper load's stale check is not atomic with the latch; recheck and reap.
        if self._spawn_is_stale():
            logger.info("shutdown began during the spawn; killing the new llama-server")
            self._kill_process()
            self._close_attempt_log()
            self._health_wait_cancelled = True
            return False

        # Start background thread to drain stdout and prevent pipe deadlock
        self._stdout_thread = threading.Thread(
            target = self._drain_stdout, daemon = True, name = "llama-stdout"
        )
        self._stdout_thread.start()
        return True

    @contextlib.contextmanager
    def _serial_load_scope(self):
        """Clears the pending fraction on leaving the lock, so a queued load's freshly armed value
        survives."""
        with self._serial_load_lock:
            try:
                yield
            finally:
                self._vram_fraction_pending = None
                self._binary_revision_pending = None
                # Cleared here, or a finished load could blank a queued load's marker.
                self._memory_pending_launch = None
