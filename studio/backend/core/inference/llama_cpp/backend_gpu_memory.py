# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: GPU allocation, memory probes and host budgets."""

import json
import math
import os
import re
import subprocess
import sys

from pathlib import Path
from typing import (
    Callable,
    Collection,
    Iterable,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _GPU_LAYER_FLAGS,
    apply_load_mode_policy,
    apply_model_memory_policy,
    fit_is_effectively_on,
    fit_is_enabled_in,
    parse_gpu_layers_override,
    resolve_effective_load_state,
    scrub_memory_env,
    strip_shadowing_flags,
)
from utils import host_memory as _host_memory
from utils.child_stdio import utf8_child_env
from utils.native_path_leases import child_env_without_native_path_secret
from utils.paths.path_utils import _is_wsl
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

from .binary_runtime import (
    _CARVEOUT_ADVICE_HOST_FRACTION,
    _CARVEOUT_ADVICE_MIN_HOST_GB,
    _CARVEOUT_NOMINAL_SLACK_GB,
    _CGROUP_ROOT,
    _GGML_GPU_BACKEND_RE,
    _HOST_RAM_HEADROOM_MIB,
    _IGPU_HOST_RESERVE_MIB,
    _PROC_SELF_CGROUP,
    _apply_igpu_host_reserve_mib,
    _lib_dir_has_ggml_backend,
    _llama_lib_dir,
    _load_probe_memo_get,
    _load_probe_memo_put,
    _loader_path_var,
)
from .draft_args import (
    _device_selection_is_cpu,
    _extra_args_device,
    _extra_args_main_device,
    _parse_metal_memory_breakdown,
    _sidecar_adapter_paths,
    _split_mode_confines_to_one_device,
)
from .launch_flags import (
    _args_place_tensors_on_cpu,
    _env_fixes_gpu_layers,
    _env_places_tensors_on_cpu,
)
from .launch_support import _metal_capable_host
from .log import logger
from .memory_policy import (
    _AMD_SMI_TOTAL_TOLERANCE,
    _APPLE_UNIFIED_MEMORY_FRACTION,
    _APPLE_WIRED_CEILING_FRACTION,
    _FIT_CONTROL_ENV_VARS,
    _FIT_CONTROL_FLAGS,
    _FIT_MIN_CTX,
    _LLAMA_FIT_MIN_CTX,
    _TENSOR_SPLIT_FLAGS,
    _extra_args_set_any_flag,
)
import time
from .binary_runtime import (
    _LOAD_PROBE_STATE,
    _VULKAN_INVENTORY_TTL_S,
    _info_once_per_load,
    _load_probe_memo_armed,
    _vulkan_inventory_cache,
    _vulkan_inventory_lock,
)
from .core_types import (
    _apple_wired_headroom_bytes,
)
from .draft_args import (
    _METAL_FIT_PROBE_TIMEOUT_S,
)
from .moe_cache import (
    _GPU_MEMORY_SKIP_TORCH,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _GpuMemoryMixin:
    @staticmethod
    def _backend_lacks_gpu_lib(binary: Optional[str] = None) -> bool:
        """Unknown and static layouts return False, so custom GPU builds are never rejected as CPU-only."""
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return False
        backends = LlamaCppBackend._installed_ggml_backends(binary)
        return bool(backends.intersection({"cpu", "base"})) and not backends.intersection(
            {"cuda", "hip", "vulkan"}
        )

    def is_vulkan_build(self) -> bool:
        """Whether the resolved llama-server uses Vulkan ordinals."""
        try:
            return self._is_vulkan_backend(self._find_llama_server_binary())
        except Exception:
            return False

    def _gpu_ids_own_placement(self, gpu_ids) -> bool:
        """True when a gpu_ids pick overrules a user --device or --main-gpu; one predicate for all sites so the launch argv and loaded check cannot drift."""
        return gpu_ids is not None

    @staticmethod
    def _strip_device_extra_args(extra_args):
        """Drop device and main-GPU flags so gpu_ids remains authoritative."""
        return strip_shadowing_flags(
            extra_args,
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = False,
            strip_device = True,
        )

    # Concrete gfx arch with trailing hex letter (gfx90a); generic or family labels are rejected on purpose, and the gate fails open on them.
    _CONCRETE_GFX_ARCH = re.compile(r"^gfx[0-9][0-9a-f]{2,4}$")

    @staticmethod
    def _installed_llama_gfx_archs(binary: Optional[str] = None) -> Optional[frozenset]:
        """Concrete gfx archs the installed ROCm prebuilt covers (mapped_targets in UNSLOTH_PREBUILT_INFO.json); None when unknown, so callers keep every device."""
        try:
            from utils.llama_cpp_freshness import read_install_marker

            marker = read_install_marker(binary or LlamaCppBackend._find_llama_server_binary())
            if not marker:
                return None
            targets = marker.get("mapped_targets")
            if not isinstance(targets, list):
                return None
            archs = frozenset(
                str(t).split(":")[0].strip().lower() for t in targets if str(t).strip()
            )
            if not archs:
                return None
            # Exact-set match only works if every token is concrete; one unknown token would drop every GPU,
            # so an unrecognised list fails open as a whole.
            unknown = sorted(a for a in archs if not LlamaCppBackend._CONCRETE_GFX_ARCH.match(a))
            if unknown:
                logger.warning(
                    f"install marker mapped_targets has non-concrete arch tokens "
                    f"{unknown}; skipping the GPU arch gate"
                )
                return None
            return archs
        except Exception as e:
            logger.debug(f"install marker arch read failed: {e}")
            return None

    @staticmethod
    def _installed_llama_cuda_sms(binary: Optional[str] = None) -> Optional[frozenset]:
        """SM targets the installed CUDA prebuilt was built for (``supported_sms``
        in UNSLOTH_PREBUILT_INFO.json). None when unknown (no marker, older
        install, non-CUDA bundle), so callers fail open. Never raises."""
        try:
            from utils.llama_cpp_freshness import read_install_marker

            marker = read_install_marker(binary or LlamaCppBackend._find_llama_server_binary())
            if not marker:
                return None
            sms = marker.get("supported_sms")
            if not isinstance(sms, list) or not sms:
                return None
            if not all(str(s).strip().isdigit() for s in sms):
                return None
            return frozenset(int(s) for s in sms)
        except Exception as e:
            logger.debug(f"install marker supported_sms read failed: {e}")
            return None

    @staticmethod
    def _cuda_compute_caps() -> dict:
        """Physical id -> SM (90 for sm_90) via nvidia-smi, honoring
        CUDA_VISIBLE_DEVICES; {} when unknown. Never raises."""
        mig_rows = LlamaCppBackend._mig_snapshot_rows()
        if mig_rows is not None:
            caps = {}
            for row in mig_rows:
                try:
                    major, minor = row.get("compute_cap", "").split(".")
                    caps[int(row["index"])] = int(major) * 10 + int(minor)
                except (TypeError, ValueError):
                    pass
            return caps
        try:
            from utils.hardware import gpu_query

            result = gpu_query.run_nvidia_smi(
                ["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                env = child_env_without_native_path_secret(),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode != 0:
                return {}
            allowed = LlamaCppBackend._visible_devices_mask("CUDA_VISIBLE_DEVICES")
            if allowed is not None and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
                # Numeric mask entries are CUDA ordinals, not PCI indices; without a shared order, fail open.
                return {}
            caps = {}
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0])
                    major, minor = parts[1].split(".")
                    sm = int(major) * 10 + int(minor)
                except ValueError:
                    continue
                if allowed is not None and idx not in allowed:
                    continue
                caps[idx] = sm
            return caps
        except Exception as e:
            logger.debug(f"compute_cap probe failed: {e}")
            return {}

    @classmethod
    def _cuda_sm_gate_error(
        cls,
        binary: Optional[str] = None,
        gpu_indices: Optional[Iterable[int]] = None,
    ) -> Optional[str]:
        """Error message when every visible GPU is OLDER than the oldest arch the
        installed CUDA bundle was built for, or None to proceed. Fails open on
        unknown coverage or caps.

        ``gpu_indices`` (physical ids the child is pinned to) tightens "every
        visible" to "any pinned": the child places work on each pinned card, so one
        too-old card aborts the load even beside a supported one.

        Only the too-old direction is broken. The bundles compile plain arch
        numbers, so each fatbin carries PTX beside its cubins, and the driver JITs
        that PTX forward onto a newer card; ggml_cuda_highest_compiled_arch picks
        the highest compiled arch <= the device and only aborts ("not compiled with
        any CUDA arch <= N") when the device is below every one of them. So an
        exact-SM test would refuse working installs: the legacy PTX-only bundle
        (sm_50-61) drives an sm_86/sm_89 host at full speed. Missing native cubins
        cost a one-time JIT, not a launch. The installer's exact-membership check
        is a different question -- which bundle is BEST to install -- not whether
        the installed one can run."""
        # Resolved once so the marker read and the remedy name the same binary.
        binary = binary or cls._find_llama_server_binary()
        supported = cls._installed_llama_cuda_sms(binary)
        if supported is None:
            return None
        caps = cls._cuda_compute_caps()
        oldest = min(supported)
        if gpu_indices is not None:
            pinned = {int(i) for i in gpu_indices}
            caps = {idx: sm for idx, sm in caps.items() if idx in pinned and sm < oldest}
        elif any(sm >= oldest for sm in caps.values()):
            return None
        if not caps:
            return None
        present = ", ".join(f"GPU {idx} is sm_{sm}" for idx, sm in sorted(caps.items()))
        # The updater cannot replace a pinned LLAMA_SERVER_PATH or a PATH llama-server, so remedy follows provenance.
        return (
            f"The installed llama.cpp build only has GPU code for sm_{oldest} and "
            f"newer, but {present} -- it was likely installed on a machine with a "
            f"newer GPU, so {cls._runtime_remedy(binary)}."
        )

    @classmethod
    def _cuda_sm_gate_keep(
        cls, binary: Optional[str], gpus: list[tuple[int, int, int]]
    ) -> list[tuple[int, int, int]]:
        """``gpus`` minus the CUDA cards older than the installed bundle's oldest
        arch, so auto placement cannot pin one beside a supported card (RTX 4090 +
        GTX 1060 under an sm_75+ build). Unchanged on unknown coverage or caps, and
        when no card survives: the gate then refuses with its own message."""
        supported = cls._installed_llama_cuda_sms(binary)
        if supported is None or not gpus:
            return gpus
        caps = cls._cuda_compute_caps()
        oldest = min(supported)
        kept = [g for g in gpus if caps.get(g[0], oldest) >= oldest]
        if not kept or len(kept) == len(gpus):
            return gpus
        for idx, _free, _total in gpus:
            if caps.get(idx, oldest) < oldest:
                logger.warning(
                    f"Skipping GPU {idx} (sm_{caps[idx]}): installed llama.cpp "
                    f"build only has GPU code for sm_{oldest} and newer"
                )
        return kept

    @classmethod
    def _arch_gate_survivors(cls, binary: Optional[str] = None) -> list[int]:
        """Physical ids the arch gate keeps, for pinning: an uncovered arch crashes HSA enumeration."""
        try:
            # Coverage first: the cached marker read avoids importing torch (~1.3s) on non-ROCm bundles.
            if cls._installed_llama_gfx_archs(binary) is None:
                return []  # unknown coverage: fail open, same as the probe
            if not cls._host_torch_is_rocm():
                return []
            if cls._visibility_mask_is_unmappable():
                # A UUID/MIG mask leaves ordinals unmapped; pinning could re-expose the dropped card. Fail open.
                logger.info(
                    "Not pinning the arch gate's survivors: the inherited GPU "
                    "visibility mask is not index based, so they cannot be named "
                    "to the child."
                )
                return []
            gated = [
                idx for idx, _free, _total in cls._get_gpu_memory(binary, for_llama_server = True)
            ]
            everything = [idx for idx, _free, _total in cls._get_gpu_memory(binary)]
            return gated if gated and len(gated) < len(everything) else []
        except Exception as e:
            logger.debug(f"arch gate survivor probe failed: {e}")
            return []

    @staticmethod
    def _get_gpu_free_memory(
        binary: Optional[str] = None, *, for_llama_server: bool = False
    ) -> list[tuple[int, int]]:
        """Free memory per GPU as ``(gpu_index, free_mib)`` sorted by index; empty
        if no supported GPU is reachable. Thin wrapper over ``_get_gpu_memory``,
        which also documents ``for_llama_server`` (forwarded verbatim)."""
        return [
            (idx, free)
            for idx, free, _total in LlamaCppBackend._get_gpu_memory(
                binary, for_llama_server = for_llama_server
            )
        ]

    @staticmethod
    def _metal_zero_ctx_floor(
        effective_ctx: int,
        auto_fit: bool,
        caller_owns_budget: bool,
        native_ctx: Optional[int],
        max_available_ctx: Optional[int] = None,
        *,
        fitter_runs: bool = True,
    ) -> int:
        """Floors a zero Metal context (-c 0 pins native length, disables --fit): _FIT_MIN_CTX if fitter runs, else _LLAMA_FIT_MIN_CTX; auto_fit and caller-owned exempt."""
        if effective_ctx > 0 or (auto_fit and fitter_runs) or caller_owns_budget:
            return 0
        if not LlamaCppBackend._apple_metal_memory_budget_bytes():
            return 0
        floor = _FIT_MIN_CTX if fitter_runs else _LLAMA_FIT_MIN_CTX
        # Never above the ceiling the cap computed; floating back up would re-create the over-commit.
        return min(floor, native_ctx or floor, max_available_ctx or floor)

    @staticmethod
    def _metal_drops_zero_ctx_override(
        ctx_override: Optional[int], caller_owns_budget: bool
    ) -> bool:
        """Drops a pass-through -c 0, since llama.cpp is last-wins and would re-pin the native length."""
        if ctx_override != 0 or caller_owns_budget:
            return False
        return bool(LlamaCppBackend._apple_metal_memory_budget_bytes())

    @staticmethod
    def _apple_metal_memory_budget_bytes() -> int:
        """Unified-memory budget for GGUF context fitting on Apple Silicon.

        No GPU is enumerated on Metal, so the context would default to native and
        over-commit unified memory ("Compute error." at decode, #5118/#6529). Use a
        fraction of the Metal working set, else total RAM; 0 off Apple Silicon or
        when unresolvable, so callers skip the cap.

        Both of those inputs describe the MACHINE, not the moment. MLX's
        max_recommended_working_set_size is a static device property, and
        virtual_memory().total obviously is, so on a 16 GB Mac the budget came
        out around 9 GB whether the machine was idle or already holding several
        gigabytes. Unsloth's own idle footprint alone is over a gigabyte once the
        warm thread has imported torch and, on Apple Silicon, MLX. The fit then
        sized a context against headroom that was not there, and llama-server
        died in KV or compute allocation.

        So take whichever is smaller: the device ceiling, or what is actually
        available right now. The ceiling still applies on a machine with plenty
        free, which is the case the working-set number was chosen for.

        _APPLE_UNIFIED_MEMORY_FRACTION then earns a second job it did not have
        over a static ceiling: available is a snapshot, and llama-server spends
        seconds loading weights, so another app can take memory inside that
        window. Budgeting all of it would re-create the failure this avoids.
        """
        from utils.hardware import is_apple_silicon

        if not is_apple_silicon():
            return 0
        from utils.hardware.hardware import _metal_recommended_working_set_bytes

        rec_bytes = _metal_recommended_working_set_bytes()
        vm = None
        try:
            import psutil
            vm = psutil.virtual_memory()
        except Exception:
            vm = None
        if rec_bytes <= 0:
            if vm is None:
                return 0
            try:
                rec_bytes = int(vm.total)
            except Exception:
                return 0
        # psutil available, not free: macOS free omits the reclaimable inactive queue. Only lowers the budget.
        try:
            available = int(getattr(vm, "available", 0) or 0) if vm is not None else 0
        except Exception:
            available = 0
        if available > 0:
            rec_bytes = min(rec_bytes, available)
        return int(rec_bytes * _APPLE_UNIFIED_MEMORY_FRACTION)

    def _metal_measured_model_mib(
        self, binary: Optional[str], model_path: Optional[str]
    ) -> Optional[tuple[int, int, int]]:
        """Measures full-offload model buffers with the sibling llama-fit-params; None if that probe
        fails, with the reason in ``_metal_probe_failure``."""
        self._metal_probe_failure = None
        if not binary or not model_path:
            self._metal_probe_failure = "no binary or model path"
            return None
        try:
            probe = _llama_lib_dir(binary) / "llama-fit-params"
        except OSError:
            self._metal_probe_failure = "llama-fit-params not found"
            return None
        try:
            probe_stat = probe.stat()
        except OSError:
            self._metal_probe_failure = f"llama-fit-params not found at {probe}"
            return None
        try:
            model_stat = os.stat(model_path)
        except OSError:
            self._metal_probe_failure = "model file unreadable"
            return None
        if os.name != "nt" and not os.access(probe, os.X_OK):
            self._metal_probe_failure = f"llama-fit-params at {probe} is not executable"
            return None
        key = (
            str(probe),
            probe_stat.st_size,
            probe_stat.st_mtime_ns,
            probe_stat.st_mode,
            model_path,
            model_stat.st_size,
            model_stat.st_mtime_ns,
        )
        cached = getattr(self, "_metal_model_mib_cache", None)
        if cached is not None and cached[0] == key:
            self._metal_probe_failure = cached[2]
            return cached[1]
        transient = getattr(_LOAD_PROBE_STATE, "metal_transient", None)
        if transient is not None and key in transient:
            self._metal_probe_failure = transient[key]
            return None
        measured: Optional[tuple[int, int, int]] = None
        failure: Optional[str] = None
        cacheable = True
        try:
            env = self._llama_server_env_for_binary(str(binary))
            for name in tuple(env):
                if name.startswith("LLAMA_ARG_"):
                    env.pop(name, None)
            result = subprocess.run(
                [str(probe), "-m", model_path, "-ngl", "999", "-c", "512", "-lv", "4"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                # A cold Metal shader compile lands here instead of at the launch.
                timeout = _METAL_FIT_PROBE_TIMEOUT_S,
                check = False,
                env = env,
                **_windows_hidden_subprocess_kwargs(),
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode != 0:
                failure = f"llama-fit-params exited with code {result.returncode}"
                cacheable = False
            else:
                measured = _parse_metal_memory_breakdown(output)
                if measured is None:
                    failure = "llama-fit-params printed no Metal memory breakdown"
            if failure:
                tail = " | ".join(
                    line.strip() for line in output.strip().splitlines()[-3:] if line.strip()
                )
                if tail:
                    failure = f"{failure}: {tail[-300:]}"
        except subprocess.TimeoutExpired:
            failure = f"llama-fit-params timed out after {_METAL_FIT_PROBE_TIMEOUT_S} s"
            cacheable = False
        except Exception as e:
            failure = f"llama-fit-params could not run: {e}"
            cacheable = False
        if failure:
            logger.debug(f"llama-fit-params memory probe failed: {failure}")
        self._metal_probe_failure = failure
        if cacheable:
            self._metal_model_mib_cache = (key, measured, failure)
        elif transient is not None:
            transient[key] = failure
        return measured

    def _metal_demand_paged_embedding_bytes(
        self,
        model_path: Optional[str],
        extra_args: Optional[Iterable[str]],
        *,
        binary: Optional[str],
        requested_load_mode: Optional[str],
        supports_load_mode: bool,
        settings: tuple[bool, bool],
        layers_fixed: bool,
        mtp_may_engage: bool,
        env: Optional[Mapping[str, str]] = None,
    ) -> int:
        """Embedding bytes proven absent from the Metal working set; fails closed, CPU weights stay
        charged."""
        source_env = os.environ if env is None else env
        if not self._metal_placement_unset(extra_args, source_env, layers_fixed = layers_fixed):
            return 0
        env_view = dict(source_env)
        scrub_memory_env(env_view, settings)
        managed, extras = apply_model_memory_policy(
            extra_args,
            supports_load_mode = supports_load_mode,
            weights_in_host_memory = True,
            gpu_offload_confirmed = False,
            env = env_view,
            settings = settings,
        )
        selected, extras = apply_load_mode_policy(
            extras,
            supports_load_mode = supports_load_mode,
            weights_in_host_memory = True,
            requested_load_mode = requested_load_mode,
            settings = settings,
        )
        mlock, reserves_ram, direct_io = resolve_effective_load_state(
            [*managed, *selected, *extras], env_view
        )
        if mlock or reserves_ram or direct_io:
            return 0
        layout = self._tensor_spill_layout(model_path, all_shards = True)
        if layout is None or not layout.complete:
            return 0
        embeddings = int(layout.token_embd_bytes) + int(layout.per_layer_embd_bytes)
        if embeddings <= 0 or (mtp_may_engage and layout.has_excluded_blocks):
            return 0
        measured = self._metal_measured_model_mib(binary, model_path)
        if measured is None:
            _info_once_per_load(
                "Metal fit: could not measure how llama.cpp maps this model (%s), so its "
                "%.2f GB of input embeddings stay charged.",
                getattr(self, "_metal_probe_failure", None) or "llama-fit-params unavailable",
                embeddings / (1024**3),
            )
            return 0
        metal_mib, host_mib, rows = measured
        mib = 1024 * 1024
        charged = (metal_mib + rows) * mib + max(0, host_mib * mib - embeddings)
        # TENSOR_SKIP blocks never reach a buffer: their bytes are in the file but not in the probe.
        mapped = int(layout.tensor_bytes) - int(layout.excluded_block_bytes)
        unmapped = min(embeddings, max(0, mapped - charged))
        if not unmapped:
            _info_once_per_load(
                "Metal fit: llama.cpp maps this model's %.2f GB of input embeddings into Metal, "
                "so they stay charged. A llama.cpp update may remove that.",
                embeddings / (1024**3),
            )
        return unmapped

    @staticmethod
    def _apple_metal_wired_ceiling_bytes() -> int:
        """Wired GPU memory a Metal load may still take, system-wide; 0 if unknown or used up."""
        return _apple_wired_headroom_bytes() or 0

    @staticmethod
    def _apple_metal_wired_headroom_bytes() -> Optional[int]:
        """The same headroom, but None when unknown, so 0 means used up."""
        return _apple_wired_headroom_bytes()

    @staticmethod
    def _metal_fit_target_floor_mib() -> float:
        """Smallest ``--fit-target`` that keeps llama.cpp's Metal fit inside the wired headroom.

        ggml-metal reports free as recommendedMaxWorkingSetSize minus this process's own
        allocations (ggml-metal-device.m, ggml_metal_device_get_memory), blind to the GPU
        memory WindowServer and other apps already hold, so --fit packs to within 1 GiB of
        the static ceiling and over-commits the wired limit (#9279). 0 when either side is
        unreadable, which leaves llama.cpp's own margin.
        """
        headroom = LlamaCppBackend._apple_metal_wired_ceiling_bytes()
        if headroom <= 0:
            return 0.0
        from utils.hardware.hardware import _metal_recommended_working_set_bytes

        seen = _metal_recommended_working_set_bytes()
        if seen <= 0:
            return 0.0
        return max(0.0, (seen - headroom) / (1024 * 1024))

    @staticmethod
    def _rocm_arch_gate_keep(
        binary: Optional[str], torch_mod, for_llama_server: bool
    ) -> Callable[[int], bool]:
        """Predicate over PHYSICAL ids: keeps a device unless ROCm with a gfx arch the installed prebuilt lacks; shared by amd-smi and torch paths."""
        if not for_llama_server or not LlamaCppBackend._torch_is_rocm(torch_mod):
            return lambda _idx: True
        supported_archs = LlamaCppBackend._installed_llama_gfx_archs(binary)
        if supported_archs is None:
            return lambda _idx: True
        arch_by_id = LlamaCppBackend._rocm_arch_by_physical_id()

        def _keep(idx: int) -> bool:
            _base = arch_by_id.get(idx, "")
            if _base and _base not in supported_archs:
                logger.warning(
                    f"Skipping GPU {idx} ({_base}): installed llama.cpp "
                    f"prebuilt only covers {sorted(supported_archs)}"
                )
                return False
            return True

        return _keep

    @staticmethod
    def _rocm_hip_is_reachable() -> bool:
        """Whether HIP can open a device; amd-smi may list cards HIP cannot open, e.g. without /dev/kfd."""
        try:
            import torch
            return bool(hasattr(torch, "cuda") and torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _rocm_hip_device_count() -> int:
        """HIP's device count, the inventory the child sees; amd-smi reads the whole host instead."""
        try:
            import torch
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return 0
            return int(torch.cuda.device_count())
        except Exception:
            return 0

    @staticmethod
    def _gpu_device_ordinal_active() -> bool:
        """Whether GPU_DEVICE_ORDINAL filters devices; treated as a mask, as AMD and clr disagree on HIP."""
        return bool(os.environ.get("GPU_DEVICE_ORDINAL", "").strip())

    @staticmethod
    def _rocm_visibility_masks_are_stacked() -> bool:
        """True when ROCR and HIP masks both filter, so HIP ordinals are not physical; callers mapping onto amd-smi ids must decline. Empty ROCR still counts as set."""
        if sys.platform == "win32":
            return False
        if os.environ.get("ROCR_VISIBLE_DEVICES") is None:
            return False
        return (
            os.environ.get("HIP_VISIBLE_DEVICES") is not None
            or os.environ.get("CUDA_VISIBLE_DEVICES") is not None
        )

    @staticmethod
    def _rocm_total_memory_mib_by_physical_id() -> dict[int, int]:
        """Total MiB per PHYSICAL id via get_device_properties, which creates no HIP context."""
        totals: dict[int, int] = {}
        try:
            import torch

            if not LlamaCppBackend._torch_is_rocm(torch):
                return totals
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return totals
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            for ordinal in range(torch.cuda.device_count()):
                try:
                    total_bytes = int(
                        getattr(torch.cuda.get_device_properties(ordinal), "total_memory", 0) or 0
                    )
                except Exception:
                    continue
                if total_bytes <= 0:
                    continue
                pid = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                totals[pid] = total_bytes // (1024 * 1024)
        except Exception:
            return totals
        return totals

    @staticmethod
    def _amd_smi_hip_id_map(
        enumerated: list[int], mask: Optional[list[int]]
    ) -> Optional[dict[int, int]]:
        """Maps amd-smi ids to HIP ids; None when unprovable, as an untranslated id pins the wrong card."""
        if len(enumerated) == 1 and mask in (None, [0]):
            # One card: HIP can only see it as id 0.
            return {enumerated[0]: 0}
        from utils.hardware.amd import get_hip_id_by_gpu_index

        mapping = get_hip_id_by_gpu_index()
        if mapping is None or not all(_g in mapping for _g in enumerated):
            logger.debug(
                "amd-smi VRAM probe deferring to torch: no amd-smi to HIP id mapping "
                "for this host (amd-smi list -e needs ROCm 6.4.0+)"
            )
            return None
        return mapping

    @staticmethod
    def _get_gpu_memory_amd_smi(
        binary: Optional[str] = None, *, for_llama_server: bool = False
    ) -> list[tuple[int, int, int]]:
        """ROCm VRAM via amd-smi per HIP id to avoid HIP context leaks; [] when partial, off ROCm, on unified APUs, or filtered"""
        try:
            import torch
            if not LlamaCppBackend._torch_is_rocm(torch):
                return []
        except Exception:
            return []
        if not LlamaCppBackend._rocm_hip_is_reachable():
            # HIP cannot open the hardware amd-smi lists; let torch answer.
            logger.debug(
                "amd-smi VRAM probe deferring to torch: this ROCm torch cannot open "
                "a device, so amd-smi's inventory is not usable by llama-server"
            )
            return []
        try:
            from utils.hardware.amd import get_gpu_vram_report

            if LlamaCppBackend._visibility_mask_is_unmappable():
                # A UUID mask cannot be applied here and amd-smi ignores masks; let torch answer.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: the GPU visibility mask "
                    "is not index based, so amd-smi's device list cannot be filtered"
                )
                return []
            if LlamaCppBackend._gpu_device_ordinal_active():
                # _resolve_visible_physical_ids does not read this variable, so amd-smi cannot be filtered.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: GPU_DEVICE_ORDINAL is "
                    "set, so amd-smi's device list cannot be filtered"
                )
                return []
            if LlamaCppBackend._rocm_visibility_masks_are_stacked():
                # A ROCr mask under a HIP mask re-indexes it, so the resolved ids would not be physical ids.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: ROCR_VISIBLE_DEVICES is "
                    "stacked under a HIP-layer mask, so the visible ids are positions "
                    "in the ROCr-filtered set rather than physical devices"
                )
                return []
            mask = LlamaCppBackend._resolve_visible_physical_ids()
            if mask is not None and not mask:
                return []  # every device hidden
            vram, enumerated = get_gpu_vram_report()
            if not enumerated:
                return []
            # amd-smi ids are not HIP ids; translate before comparing with the mask or passing them to the child.
            hip_by_gpu = LlamaCppBackend._amd_smi_hip_id_map(enumerated, mask)
            if hip_by_gpu is None:
                return []
            vram = {hip_by_gpu[_g]: _v for _g, _v in vram.items()}
            # amd-smi lists all cards regardless of mask; drop those this process cannot see; no mask keeps all.
            visible = set(mask) if mask is not None else set(hip_by_gpu.values())
            if not visible.issubset(vram):
                # A device that is visible and unmeasured: all or nothing.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: no usable VRAM reading "
                    f"for visible GPU(s) {sorted(visible - set(vram))}"
                )
                return []
            hip_count = LlamaCppBackend._rocm_hip_device_count()
            if len(visible) != hip_count:
                # The amd-smi inventory must equal HIP's (device-cgroup containers filter outside env vars),
                # otherwise the numbers are not comparable.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: amd-smi reports "
                    f"{len(visible)} visible GPU(s) and HIP opens {hip_count}"
                )
                return []
            hip_totals = LlamaCppBackend._rocm_total_memory_mib_by_physical_id()
            disagreeing = sorted(
                _idx
                for _idx in visible
                if _idx in hip_totals
                and abs(vram[_idx][1] - hip_totals[_idx])
                > _AMD_SMI_TOTAL_TOLERANCE * hip_totals[_idx]
            )
            if disagreeing:
                # amd-smi and HIP may measure different memory scopes (APU carve-out vs GTT); defer on mismatch; undescribed devices fail open.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: amd-smi and HIP report "
                    f"different total memory for GPU(s) {disagreeing}"
                )
                return []
            unified_ids = LlamaCppBackend._rocm_unified_memory_gpu_ids()
            if unified_ids:
                # On a unified-memory APU amd-smi reports only the BIOS carve-out; defer the whole host to torch
                # so both branches of _get_gpu_memory agree.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: unified-memory APU(s) "
                    f"{sorted(unified_ids)} report only their dedicated VRAM slice"
                )
                return []
            arch_keeps = LlamaCppBackend._rocm_arch_gate_keep(binary, torch, for_llama_server)
            gpus: list[tuple[int, int, int]] = []
            for idx in sorted(visible):
                if not arch_keeps(idx):
                    continue
                free_mib, total_mib = vram[idx]
                gpus.append((idx, free_mib, total_mib))
            return gpus
        except Exception as e:
            logger.debug(f"amd-smi GPU probe failed: {e}")
            return []

    @staticmethod
    def _explain_empty_gpu_probe(binary: Optional[str] = None) -> str:
        """Says why _get_gpu_memory came back empty; never raises, never returns an empty string; no mem_get_info."""
        try:
            binary = binary or LlamaCppBackend._find_llama_server_binary()
            if _metal_capable_host():
                # Same check as the load site: an Intel Mac wants its real reason.
                return "this probe reads CUDA and HIP only; Apple Silicon offloads through Metal"
            # Checked first: an unopenable render node leaves HIP and Vulkan empty; Vulkan asks only the render node; unreadable installs stay eligible.
            _is_vulkan = LlamaCppBackend._is_vulkan_backend(binary)
            _backends = LlamaCppBackend._installed_ggml_backends(binary)
            _amd_capable = not LlamaCppBackend._backend_lacks_gpu_lib(binary) and (
                _is_vulkan or "hip" in _backends or not _backends
            )

            # KFD ordinal space the device lists index; None if unreadable, which leaves selectors alone.
            try:
                from utils.hardware.amd import amd_kfd_gpu_node_count
                _amd_gpu_count = amd_kfd_gpu_node_count()
            except Exception:  # noqa: BLE001
                _amd_gpu_count = None

            def _post_rocr_device_count() -> "int | None":
                # ROCr filters the physical list first and renumbers, so HIP ordinals are judged against
                # the post-ROCr count. None when survivors cannot be counted.
                if not _rocr_filters:
                    return _amd_gpu_count
                _raw = (os.environ.get("ROCR_VISIBLE_DEVICES") or "").strip()
                if not _raw or not _amd_gpu_count:
                    return None
                # Mirrors ROCr RvdFilter (amd_filter_device.h): out-of-range, repeated or illegal tokens end
                # the list, so every ending is a known prefix; only a UUID is unknowable.
                _survivors = 0
                _selected: "set[int]" = set()
                for _entry in _raw.split(","):
                    _entry = _entry.strip()
                    if not _entry.isdigit():
                        # Only the GPU-XX UUID form is unknowable; other non-indices are Illegal to ROCr.
                        return None if _entry.lower().startswith("gpu-") else _survivors
                    _idx = int(_entry)
                    if _idx >= _amd_gpu_count or _idx in _selected:
                        return _survivors
                    _selected.add(_idx)
                    _survivors += 1
                return _survivors

            def _hides_every_device(
                value: str,
                count: "int | None" = None,
                *,
                strict: bool = False,
            ) -> bool:
                # CUDA/HIP stop at the first entry naming no device; an empty or negative first
                # entry exposes nothing.
                first = value.split(",")[0].strip()
                if first == "" or first.startswith("-"):
                    return True
                # clr rejects a token unless atoi round-trips it, so 'garbage', '0x1', '00' terminate like -1.
                if strict and not first.lower().startswith("gpu-"):
                    try:
                        _index = int(first)
                    except ValueError:
                        return True
                    if str(_index) != first:
                        return True
                # An out-of-range ordinal also ends the list; only judged against a count actually read.
                _bound = _amd_gpu_count if count is None else count
                if not _bound or not first.isdigit():
                    return False
                return int(first) >= _bound

            def _cannot_be_resolved(value: str) -> bool:
                # UUID tokens cannot be resolved here, so they are unresolved rather than blockers.
                first = value.split(",")[0].strip()
                return first.lower().startswith("gpu-")

            def _vk_selects_no_device(value: str) -> bool:
                # ggml stops at the first token with no integer prefix; a sign counts as a prefix.
                _first = value.replace(",", " ").split()
                if not _first:
                    return True
                _token = _first[0]
                _digits = _token[1:] if _token[:1] in ("+", "-") else _token
                return not _digits[:1].isdigit()

            def _vk_ordinal_always_throws(value: str) -> bool:
                # A negative ordinal throws once ggml reaches it (size_t wraps -1 to 2**64-1); earlier tokens must still extract; -0 is 0.
                for _token in value.replace(",", " ").split():
                    _signed = _token[:1] in ("+", "-")
                    _digits = _token[1:] if _signed else _token
                    if not _digits[:1].isdigit():
                        return False  # extraction stops here, so nothing after it is read
                    if _token[:1] == "-" and _digits.lstrip("0"):
                        return True
                return False

            def _is_an_illegal_rocr_selector(value: str) -> bool:
                # An Illegal first ROCr token leaves zero survivors; no group membership fixes it.
                first = value.split(",")[0].strip()
                return (
                    bool(first)
                    and not first.startswith("-")
                    and not first.isdigit()
                    and not first.lower().startswith("gpu-")
                )

            # Per-variable precedence as HIP reads it: clr prefers a non-empty HIP mask over CUDA, ROCr
            # composes below it (an empty ROCr mask blinds the runtime), Windows has no ROCr layer.
            _hip_layer_var = (
                "HIP_VISIBLE_DEVICES"
                if os.environ.get("HIP_VISIBLE_DEVICES", "")
                else "CUDA_VISIBLE_DEVICES"
            )
            _rocr_filters = (
                sys.platform != "win32" and os.environ.get("ROCR_VISIBLE_DEVICES") is not None
            )
            _ordinal_filters = LlamaCppBackend._gpu_device_ordinal_active()

            _post_rocr_count = _post_rocr_device_count()

            masks = []
            blocking = []
            unresolved = []
            for var in (
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
            ):
                raw = os.environ.get(var)
                if raw is None:
                    continue
                phrase = f"{var}={raw!r}" if raw.strip() else f"{var} is empty"
                masks.append(phrase)
                if var == "GPU_DEVICE_ORDINAL":
                    _consulted = _ordinal_filters
                elif var == "ROCR_VISIBLE_DEVICES":
                    _consulted = _rocr_filters
                else:
                    _consulted = var == _hip_layer_var
                # A Vulkan build reads none of the four, so none of them blocks it.
                if _is_vulkan or not _consulted:
                    continue
                # ROCr indexes the physical list, the HIP layer indexes ROCr's survivors.
                _rocr = var == "ROCR_VISIBLE_DEVICES"
                _bound = _amd_gpu_count if _rocr else _post_rocr_count
                if _hides_every_device(raw, _bound, strict = not _rocr) or (
                    _rocr and _is_an_illegal_rocr_selector(raw)
                ):
                    blocking.append(phrase)
                elif _cannot_be_resolved(raw):
                    unresolved.append(phrase)
            # A Vulkan build reads only GGML_VK_VISIBLE_DEVICES; without the raw device count only a
            # no-integer first token or a negative ordinal is decidable.
            if _is_vulkan:
                _vk_raw = os.environ.get("GGML_VK_VISIBLE_DEVICES")
                if _vk_raw is not None:
                    _vk_phrase = (
                        f"GGML_VK_VISIBLE_DEVICES={_vk_raw!r}"
                        if _vk_raw.strip()
                        else "GGML_VK_VISIBLE_DEVICES is empty"
                    )
                    masks.append(_vk_phrase)
                    if _vk_selects_no_device(_vk_raw) or _vk_ordinal_always_throws(_vk_raw):
                        blocking.append(_vk_phrase)
                    else:
                        unresolved.append(_vk_phrase)

            mask_note = f" ({', '.join(masks)})" if masks else ""

            node_hint = None
            if _amd_capable:
                try:
                    from utils.hardware.amd import amd_node_permission_hint
                    node_hint = amd_node_permission_hint(needs_kfd = not _is_vulkan)
                except Exception:  # noqa: BLE001
                    node_hint = None

            def _closed_nodes_block_the_runtime() -> bool:
                # An unreadable host answers True, keeping the closed node as the reason, as before the sibling check.
                try:
                    from utils.hardware.amd import amd_closed_nodes_block_the_runtime
                    return amd_closed_nodes_block_the_runtime(needs_kfd = not _is_vulkan)
                except Exception:  # noqa: BLE001
                    return True

            def _another_vendor_has_an_open_node() -> bool:
                # Vulkan enumerates any vendor, so an open non-AMD render node is a complete path; False on a host this cannot read.
                if not _is_vulkan:
                    return False
                # Decided by which drivers the loader can load: an AMD-only loader ignores other vendor open nodes; one that loads them must NOT suppress.
                try:
                    from utils.hardware.amd import (
                        a_non_amd_render_node_is_open,
                        the_vulkan_loader_can_only_load_amd,
                    )
                    if the_vulkan_loader_can_only_load_amd():
                        return False
                    return a_non_amd_render_node_is_open()
                except Exception:  # noqa: BLE001
                    return False

            # Closed node, if it does not explain the empty probe, is appended to the reason, not in place of it.
            _second_finding = ""
            # Sufficient alone, so it rides on the PRIMARY reason, not on whatever gets demoted.
            _loader_finding = ""

            def _the_vulkan_loader_has_no_driver() -> bool:
                try:
                    from utils.hardware.amd import the_vulkan_loader_has_no_usable_driver
                    return the_vulkan_loader_has_no_usable_driver()
                except Exception:  # noqa: BLE001
                    return False

            def _the_loader_override_to_blame() -> "str | None":
                # Which of the three causes it is, since only two are repaired by
                # installing anything.
                try:
                    from utils.hardware.amd import the_vulkan_loader_override_to_blame
                    return the_vulkan_loader_override_to_blame()
                except Exception:  # noqa: BLE001
                    return None

            def _reason(text: str) -> str:
                return f"{text}{_loader_finding}{_second_finding}"

            if node_hint:
                # Only a mask that can hide every device counts as a second blocker beside a closed node.
                if blocking:
                    node_hint = (
                        f"{node_hint} A device visibility mask is also in force "
                        f"({', '.join(blocking)}), which the groups do not clear."
                    )
                elif unresolved:
                    node_hint = (
                        f"{node_hint} {', '.join(unresolved)} names a device this cannot "
                        f"resolve, so whether it also hides the card is unknown; check it "
                        f"if the groups do not help."
                    )
                # A closed node explains the empty probe only if the runtime would have used it (no
                # open sibling).
                _node_is_why = (
                    _closed_nodes_block_the_runtime() and not _another_vendor_has_an_open_node()
                )
                # No loadable manifest is a blocker the node repair cannot clear; keep it out of node_hint.
                _also = "also " if _node_is_why else ""
                if _is_vulkan and _the_vulkan_loader_has_no_driver():
                    # Name the loader override env var so the user knows what to unset.
                    _override = _the_loader_override_to_blame()
                    if _override:
                        _loader_finding = (
                            f" The Vulkan loader {_also}has no driver it can load "
                            f"here, and what leaves it with none is {_override}: clear or "
                            f"correct that, since reinstalling the driver does not change "
                            f"an environment override."
                        )
                    else:
                        _loader_finding = (
                            f" The Vulkan loader {_also}has no driver it can load "
                            f"here: every ICD manifest it would read is missing its library "
                            f"or is 32-bit, so reinstall the Vulkan driver as well."
                        )
                if _node_is_why:
                    return f"{node_hint}{_loader_finding}"
                _second_finding = f" Separately, and not why the probe is empty: {node_hint}"
            if _is_vulkan:
                return _reason("the Vulkan probe reported no device")

            try:
                import torch
            except Exception:  # noqa: BLE001
                return _reason(f"torch is not importable, so no GPU could be enumerated{mask_note}")
            if not hasattr(torch, "cuda") or not torch.cuda.is_available():
                return _reason(f"torch reports no usable CUDA or HIP device{mask_note}")
            # Counting devices does not create a context; reading their memory would.
            count = torch.cuda.device_count()
            if not count:
                return _reason(f"torch enumerated 0 devices{mask_note}")

            if LlamaCppBackend._torch_is_rocm(torch):
                coverage = LlamaCppBackend._installed_llama_gfx_archs(binary)
                if coverage:
                    present = sorted(set(LlamaCppBackend._rocm_arch_by_physical_id().values()))
                    if present and not (set(present) & set(coverage)):
                        return _reason(
                            f"the installed llama.cpp build covers {sorted(coverage)} but this "
                            f"host has {present}, so the arch gate dropped every device"
                        )
                return _reason(
                    f"torch sees {count} ROCm device(s) but the probe returned none, so "
                    f"amd-smi and the torch fallback both declined{mask_note}"
                )
            return _reason(f"torch sees {count} device(s) but the probe returned none{mask_note}")
        except Exception as e:  # noqa: BLE001 -- diagnostics must not break a load
            return f"the reason could not be determined ({type(e).__name__})"

    @staticmethod
    def _nvidia_probe_script() -> Optional[Path]:
        """studio/nvidia_probe.py, the installers' NVML / CUDA driver reader (UNSLOTH_NVIDIA_PROBE wins)."""
        from utils.prebuilt.update_flow import find_installer_script
        return find_installer_script(env_var = "UNSLOTH_NVIDIA_PROBE", script_name = "nvidia_probe.py")

    @staticmethod
    def _nvml_rows_visible(rows: list[dict]) -> list[dict]:
        """The NVML rows CUDA_VISIBLE_DEVICES permits, in mask order: an index or a GPU-/MIG-
        uuid prefix per entry, as the CUDA runtime reads it. An entry naming no single device
        ends the mask there: "0,2,-1,1" exposes 0 and 2, "-1" alone hides every GPU, and a
        repeated index empties it, as CUDA does. A GPU in MIG mode is one row backed by one
        slice, which the child is pinned to: the first slice the mask names, else the one with
        the least free memory, so the budget never exceeds the device the child gets (R570+
        enumerates every slice, but one per card is sized and pinned)."""
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
        parents = [r for r in rows if not r.get("mig")]

        def free_mib(r: dict) -> int:
            try:
                return int(r.get("memory_free_mib") or 0)
            except (TypeError, ValueError):
                return 0

        def as_cuda_sees(r: dict) -> dict:
            if r.get("mig"):
                return r  # a slice the mask named stays that slice
            slices = [
                s for s in rows if s.get("mig") and str(s.get("index")) == str(r.get("index"))
            ]
            return min(slices, key = free_mib) if slices else r

        tokens = [t.strip() for t in (raw or "").split(",")] if raw else []
        picked: list[dict] = []
        if raw is None:
            picked = [as_cuda_sees(r) for r in parents]
        seen: set[int] = set()
        for token in tokens:
            if not token:
                break  # an empty entry ends the mask, as CUDA reads it
            prefix = re.match(r"\+?\d+", token)
            # Case-insensitive, as the nvidia-smi uuid matcher reads the same mask.
            low = token.lower()
            if low.startswith("gpu-"):
                hits = [r for r in parents if str(r.get("uuid", "")).lower().startswith(low)]
            elif low.startswith("mig-"):
                # A MIG- entry names a slice row the probe lists under its parent.
                hits = [
                    r
                    for r in rows
                    if r.get("mig") and str(r.get("uuid", "")).lower().startswith(low)
                ]
            elif prefix:
                # CUDA reads an index strtoul-style: "1gpu2" is GPU 1.
                index = int(prefix.group())
                if index in seen:
                    return []
                seen.add(index)
                hits = [r for r in parents if str(r.get("index")) == str(index)]
            else:
                hits = []
            if len(hits) != 1:
                break
            if str(hits[0].get("index")) not in {str(p.get("index")) for p in picked}:
                picked.append(as_cuda_sees(hits[0]))
        return picked

    @staticmethod
    def _uuid_by_index(rows: list[dict]) -> dict[int, str]:
        """The uuid each visible index stands for: every row when the mask named rows by
        uuid, else only a slice standing in for its MIG parent (a slice has no index of its
        own). Derived from the rows and the mask on each call, so it belongs to no probe."""
        tokens = [t.strip() for t in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")]
        by_uuid = any(t and not t.isdigit() for t in tokens)
        return {
            int(r["index"]): str(r["uuid"])
            for r in LlamaCppBackend._nvml_rows_visible(rows)
            if str(r.get("index", "")).isdigit() and (by_uuid or r.get("mig"))
        }

    @staticmethod
    def _child_visibility_for(gpu_indices) -> str:
        """The mask for these indices: the inherited uuids when the mask named the rows by
        uuid (a MIG slice must stay a MIG- entry), else the indices themselves."""
        ids = [int(i) for i in gpu_indices]
        by_index = LlamaCppBackend._uuid_by_index(LlamaCppBackend._NVML_ROWS)
        if by_index and all(i in by_index for i in ids):
            return ",".join(by_index[i] for i in ids)
        return ",".join(str(i) for i in ids)

    @staticmethod
    def _mig_tokens_for_ordinals(cvd: str, gpu_indices) -> Optional[str]:
        """The inherited MIG- entries behind torch ordinals, or None (keep the whole mask). Only
        when torch enumerated every entry, so ordinal i is entry i in mask order."""
        tokens = [t.strip() for t in cvd.split(",") if t.strip()]
        if LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is not False or not all(
            t.startswith("MIG-") for t in tokens
        ):
            return None
        try:
            import torch
            if torch.cuda.device_count() != len(tokens):
                return None
            ids = [int(i) for i in gpu_indices]
        except Exception:
            return None
        if not ids or len(set(ids)) != len(ids) or not all(0 <= i < len(tokens) for i in ids):
            return None
        return ",".join(tokens[i] for i in ids)

    @staticmethod
    def _smi_uuids_for(cvd: str, gpu_indices) -> Optional[str]:
        """The uuids of these physical ids when the current whole-GPU uuid mask names each
        of them in the last nvidia-smi rows, else None. Never for a MIG card: CUDA hands a
        GPU- entry on one a slice, which only the NVML rows name."""
        if LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is not True:
            return None
        from utils.hardware import nvidia

        rows = LlamaCppBackend._SMI_UUID_ROWS
        visible = nvidia._uuid_visible_ordinal_map(cvd, rows)
        try:
            ids = [int(i) for i in gpu_indices]
        except (TypeError, ValueError):
            return None
        if not visible or not ids or len(set(ids)) != len(ids):
            return None
        if any(i not in visible for i in ids):
            return None
        # The probe that chose these rows classified them; a later recheck failing must
        # not widen the child to the whole inherited mask.
        mig = LlamaCppBackend._SMI_MIG_INDICES
        if mig is None:
            # Unknown then too: NVML rows from that probe still name MIG cards.
            mig = {
                int(r["index"])
                for r in LlamaCppBackend._NVML_ROWS
                if r.get("mig") and str(r.get("index", "")).isdigit()
            }
        if set(ids) & mig:
            return None
        by_index = dict(rows)
        return ",".join(by_index[i] for i in ids)

    @staticmethod
    def _smi_uuid_mask_order() -> Optional[list[int]]:
        """Physical ids of a whole-GPU uuid mask in mask order, from the last nvidia-smi
        rows; None unless the probe's ids are physical and every token matched a row."""
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is not True or not (cvd or "").strip():
            return None
        from utils.hardware import nvidia

        visible = nvidia._uuid_visible_ordinal_map(cvd, LlamaCppBackend._SMI_UUID_ROWS)
        return sorted(visible, key = visible.__getitem__) if visible else None

    @staticmethod
    def _pinned_child_visibility(gpu_indices) -> Optional[str]:
        """``_child_visibility_for``, or None when the inherited CUDA mask must stand: a MIG
        or unresolvable UUID mask the ids cannot be mapped back to, where a numeric pin would
        be re-read against the whole host (a card the parent hid, or the whole MIG card)."""
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        mig_rows = LlamaCppBackend._mig_snapshot_rows()
        if mig_rows is not None:
            from utils.hardware import mig
            return mig.visibility_for(mig_rows, gpu_indices)
        if (
            LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is False
            and cvd is not None
            and cvd.strip()
            and LlamaCppBackend._visibility_mask_is_unmappable()
        ):
            # Torch ordinals: map through the mask itself, never through NVML rows cached by
            # an earlier probe (their indices are physical, so ordinal 0 could name another slice).
            return LlamaCppBackend._mig_tokens_for_ordinals(cvd, gpu_indices)
        mask = LlamaCppBackend._child_visibility_for(gpu_indices)
        if (
            cvd is not None
            and cvd.strip()
            and mask == ",".join(str(int(i)) for i in gpu_indices)
            and LlamaCppBackend._visible_devices_mask("CUDA_VISIBLE_DEVICES") is None
            and LlamaCppBackend._visibility_mask_is_unmappable()
        ):
            # Physical rows the memory probe matched by uuid: hand the child those uuids.
            uuids = LlamaCppBackend._smi_uuids_for(cvd, gpu_indices)
            if uuids is not None:
                return uuids
            return LlamaCppBackend._mig_tokens_for_ordinals(cvd, gpu_indices)
        return mask

    @staticmethod
    def _get_gpu_memory_nvml() -> list[tuple[int, int, int]]:
        """Free/total per NVIDIA GPU via NVML in a child for hosts whose nvidia-smi is absent or hung; a row without memory voids the answer."""
        if sys.platform == "darwin" or os.environ.get("UNSLOTH_NVIDIA_LIBRARY_PROBE", "1") == "0":
            return []
        script = LlamaCppBackend._nvidia_probe_script()
        if script is None:
            return []
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script), "--json"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                env = utf8_child_env(child_env_without_native_path_secret()),
                **_windows_hidden_subprocess_kwargs(),
            )
            payload = json.loads(result.stdout or "null")
        except Exception as e:
            logger.debug(f"NVML probe failed: {e}")
            return []
        if not isinstance(payload, dict) or payload.get("source") != "nvml":
            return []
        LlamaCppBackend._NVML_ROWS = list(payload.get("devices") or [])
        gpus: list[tuple[int, int, int]] = []
        for row in LlamaCppBackend._nvml_rows_visible(LlamaCppBackend._NVML_ROWS):
            try:
                idx = int(row["index"])
                free_mib = int(row.get("memory_free_mib") or 0)
                total_mib = int(row.get("memory_total_mib") or 0)
            except (KeyError, TypeError, ValueError):
                return []
            if total_mib <= 0:
                # One visible GPU without a reading voids the answer; a full card (free 0) is still a reading.
                return []
            gpus.append((idx, free_mib, total_mib))
        gpus.sort(key = lambda g: g[0])
        return gpus

    @staticmethod
    def _whole_gpu_mask_rows(
        cvd: str, uuid_rows: list[tuple[int, str]], physical_ids: set[int]
    ) -> Optional[set[int]]:
        """Physical rows behind a whole-GPU uuid mask the ordinal resolvers left open.
        A uuid names its card whatever CUDA_DEVICE_ORDER says, so only membership is
        needed. None for a MIG mask or a token no row verifies, so the caller keeps
        its conservative path."""
        from utils.hardware import nvidia

        tokens = [t.strip().lower() for t in cvd.split(",") if t.strip()]
        if not tokens or any(not t.startswith("gpu-") for t in tokens):
            return None
        if uuid_rows:
            visible = nvidia._uuid_visible_ordinal_map(cvd, uuid_rows)
            return set(visible) if visible is not None else None
        # No uuid column: verify with the separate uuid query, never assume. Only a
        # one-card mask: these rows carry no uuids to order or pin a multi-card subset,
        # while the inherited mask already pins a single card.
        ids = nvidia.resolve_uuid_mask(cvd)
        return set(ids) if ids and len(ids) == 1 and set(ids) <= physical_ids else None

    @staticmethod
    def _get_gpu_memory(
        binary: Optional[str] = None, *, for_llama_server: bool = False
    ) -> list[tuple[int, int, int]]:
        """Order nvidia-smi, amd-smi, then torch last (permanent CUDA/HIP context); Vulkan probe on Vulkan builds; for_llama_server adds ROCm arch gate."""
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if LlamaCppBackend._is_vulkan_backend(binary):
            return LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
        from utils.hardware import mig

        mig_rows = mig.visible_devices(fresh = True, listing = True)
        LlamaCppBackend._MIG_DEVICE_SNAPSHOT = None
        if mig_rows is not None:
            # CUDA device ids; unknown memory stays zero rather than dropping a device.
            LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = False
            LlamaCppBackend._MIG_DEVICE_SNAPSHOT = (
                os.environ.get("CUDA_VISIBLE_DEVICES"),
                os.environ.get("CUDA_DEVICE_ORDER"),
                mig_rows,
            )
            gpus = sorted(
                (
                    int(r["index"]),
                    int(r.get("memory_free_mib") or 0),
                    int(r.get("memory_total_mib") or 0),
                )
                for r in mig_rows
            )
            return LlamaCppBackend._cuda_sm_gate_keep(binary, gpus) if for_llama_server else gpus
        # ── NVIDIA via nvidia-smi ────────────────────────────────────
        unsized_ids: list[int] = []
        mig_card_rows: list[tuple[int, int, int]] = []
        try:
            from utils.hardware import gpu_query

            # Placement / fit: never a cached reading. A hung CLI raises so the MIG-aware NVML branch runs.
            result = gpu_query.run_nvidia_smi(
                [
                    "nvidia-smi",
                    # uuid in the same call: a whole-GPU uuid mask resolves without
                    # CUDA_DEVICE_ORDER or a second query.
                    "--query-gpu=index,memory.free,memory.total,uuid",
                    "--format=csv,noheader,nounits",
                ],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                env = child_env_without_native_path_secret(),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                from utils.hardware import nvidia

                allowed = LlamaCppBackend._visible_devices_mask("CUDA_VISIBLE_DEVICES")
                _cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
                rows: list[tuple[int, int, int]] = []
                unsized_rows: list[int] = []
                uuid_rows: list[tuple[int, str]] = []
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 2:
                        continue
                    if len(parts) >= 4 and parts[0].isdigit() and parts[3]:
                        uuid_rows.append((int(parts[0]), parts[3]))
                    # Index and free required; skip a bad line rather than abandon
                    # the probe to the torch fallback.
                    try:
                        idx = int(parts[0])
                        free_mib = int(parts[1])
                    except ValueError:
                        # Jetson answers "[N/A]": no dedicated VRAM (#4862).
                        if parts[0].isdigit():
                            unsized_rows.append(int(parts[0]))
                        continue
                    # Total parsed separately: N/A on MIG/vGPU keeps the GPU at total 0 instead of dropping it.
                    total_mib = 0
                    if len(parts) >= 3 and parts[2]:
                        try:
                            total_mib = int(parts[2])
                        except ValueError:
                            total_mib = 0
                    rows.append((idx, free_mib, total_mib))
                LlamaCppBackend._SMI_UUID_ROWS = uuid_rows
                LlamaCppBackend._SMI_MIG_INDICES = None
                if allowed is None and _cvd:
                    allowed = LlamaCppBackend._whole_gpu_mask_rows(
                        _cvd, uuid_rows, {r[0] for r in rows} | set(unsized_rows)
                    )
                if allowed is None and _cvd:
                    # CUDA's numeric-prefix reading ("1gpu2" is 1, "0,1,abc" is 0,1).
                    _ids = LlamaCppBackend._resolve_visible_physical_ids()
                    allowed = set(_ids) if _ids is not None else None
                unsized_ids = [i for i in unsized_rows if allowed is None or i in allowed]
                gpus = [r for r in rows if allowed is None or r[0] in allowed]
                # Match the docstring's sort-by-id guarantee (driver order isn't).
                gpus.sort(key = lambda g: g[0])
                if allowed is None and _cvd:
                    # A MIG / unresolvable UUID mask names devices these rows cannot be
                    # matched to: listing every card sizes against GPUs the parent hid.
                    # The NVML branch names slices and uuids.
                    gpus = []
                elif gpus:
                    mig_known = nvidia.get_mig_gpu_indices_or_none()
                    if mig_known is not None:
                        LlamaCppBackend._SMI_MIG_INDICES = frozenset(mig_known)
                    mig = (mig_known or set()) & {g[0] for g in gpus}
                    if mig_known is None:
                        # MIG discovery failed: a row may be a whole MIG card. NVML names
                        # slices and whole cards alike, so its reading wins; without one
                        # the CLI rows stand, as before MIG support (no CUDA context).
                        nvml = {g[0]: g for g in LlamaCppBackend._get_gpu_memory_nvml()}
                        if nvml:
                            gpus = [nvml.get(g[0], g) for g in gpus]
                        else:
                            # No current NVML inventory: an older one must not name devices.
                            LlamaCppBackend._NVML_ROWS = []
                    elif not mig:
                        # The CLI now answers for these cards: an earlier NVML inventory
                        # (slices since removed) must not name the child's devices.
                        LlamaCppBackend._NVML_ROWS = []
                    if mig:
                        # A MIG card's row is the whole card: swap in its NVML slice reading,
                        # keeping the whole cards. No reading at all: the later probes decide,
                        # then these rows as before.
                        slices = {g[0]: g for g in LlamaCppBackend._get_gpu_memory_nvml()}
                        mig_card_rows = gpus
                        gpus = [slices.get(g[0], g) if g[0] in mig else g for g in gpus]
                        gpus = [g for g in gpus if g[0] not in mig or g[0] in slices]
                if gpus:
                    LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = True
                    if for_llama_server:
                        gpus = LlamaCppBackend._cuda_sm_gate_keep(binary, gpus)
                    # The CLI row is the carve-out, not the pool. Discrete rows unchanged.
                    return LlamaCppBackend._widen_integrated_cuda_rows(gpus)
        except Exception as e:
            logger.debug(f"nvidia-smi probe failed: {e}")

        # ── NVIDIA via NVML, when nvidia-smi is absent, stale or hung ────
        nvml_gpus = LlamaCppBackend._get_gpu_memory_nvml()
        if nvml_gpus:
            LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = True
            if for_llama_server:
                nvml_gpus = LlamaCppBackend._cuda_sm_gate_keep(binary, nvml_gpus)
            # nvidia-smi IS NVML: same carve-out, same PCI indexing, same correction.
            return LlamaCppBackend._widen_integrated_cuda_rows(nvml_gpus)

        # ── AMD ROCm via amd-smi ─────────────────────────────────────
        rocm_gpus = LlamaCppBackend._get_gpu_memory_amd_smi(
            binary, for_llama_server = for_llama_server
        )
        if rocm_gpus:
            return rocm_gpus

        # ── Torch fallback (covers AMD ROCm and missing nvidia-smi) ──
        gpus = (
            []
            if _GPU_MEMORY_SKIP_TORCH.get()
            else LlamaCppBackend._get_gpu_memory_torch(binary, for_llama_server)
        )
        if not gpus and mig_card_rows:
            LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = True
            return LlamaCppBackend._widen_integrated_cuda_rows(mig_card_rows)
        if gpus or not (for_llama_server and unsized_ids):
            return gpus
        return LlamaCppBackend._tegra_unified_memory_rows(unsized_ids)

    @staticmethod
    def _tegra_unified_memory_rows(gpu_ids: list[int]) -> list[tuple[int, int, int]]:
        """Price a Jetson's GPU from system RAM when nvidia-smi answers ``[N/A]`` and
        torch cannot see it (a generic aarch64 wheel), so the context is still capped
        (#4862). llama-server only: it runs on JetPack's CUDA, torch-backed callers do not."""
        from utils.tegra import is_tegra

        if not is_tegra():
            return []
        avail = LlamaCppBackend._available_system_memory_mib()
        if avail is None:
            return []
        cgroup_mib = LlamaCppBackend._cgroup_available_memory_mib()
        # Same policy as the torch integrated branch: a cgroup-bound pool publishes no total.
        total = (
            0
            if cgroup_mib is not None and cgroup_mib <= avail
            else LlamaCppBackend._total_system_memory_mib() or 0
        )
        free = _apply_igpu_host_reserve_mib(avail, True)
        LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = True
        # Torch never saw these, so record them as integrated or the pool is credited twice.
        LlamaCppBackend._INTEGRATED_CUDA_IDS[LlamaCppBackend._integrated_cuda_mask_key()] = set(
            gpu_ids
        )
        return [(idx, free, total) for idx in sorted(gpu_ids)]

    @staticmethod
    def _get_gpu_memory_torch(
        binary: Optional[str], for_llama_server: bool
    ) -> list[tuple[int, int, int]]:
        """The torch arm of ``_get_gpu_memory``."""
        try:
            import torch

            if not hasattr(torch, "cuda") or not torch.cuda.is_available():
                return []
            if not hasattr(torch.cuda, "mem_get_info"):
                return []
            # CUDA ordinals from here on, not nvidia-smi PCI indices.
            LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = False
            # torch ordinals are relative to the mask: translate to physical ids before CVD; an empty mask means no GPUs, so never use or to fall back.
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            # Shared-pool APU: report total 0 since that total is system RAM.
            unified_ids = LlamaCppBackend._rocm_unified_memory_gpu_ids()
            # Integrated CUDA SoCs share one pool too, and their free reading errs the opposite way to ROCm's.
            integrated_ids = LlamaCppBackend._integrated_cuda_gpu_ids()
            # Same #7624 arch gate the amd-smi branch applies, from the one helper.
            arch_keeps = LlamaCppBackend._rocm_arch_gate_keep(binary, torch, for_llama_server)
            # Read once from the set the loop tests, so count and decision agree (no division by zero).
            shared_cuda_count = sum(1 for _i in integrated_ids if arch_keeps(_i))
            gpus = []
            # Windows ROCm over-reports free on discrete cards too; cap against this process's allocator.
            from utils.hardware import trusted_mem_get_info

            for ordinal in range(torch.cuda.device_count()):
                free_bytes, total_bytes = trusted_mem_get_info(ordinal)
                idx = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                if not arch_keeps(idx):
                    continue
                shared = idx in unified_ids
                integrated = idx in integrated_ids
                cgroup_bound = False
                raw_mib = free_bytes // (1024 * 1024)
                total_mib = total_bytes // (1024 * 1024)
                if shared:
                    # ROCm free is unreliable on a shared pool (Windows free==total); cap at system RAM before taking the reserve.
                    avail = LlamaCppBackend._available_system_memory_mib()
                    if avail is not None:
                        raw_mib = min(raw_mib, avail)
                elif integrated:
                    # cudaMemGetInfo free is MemFree, which counts page cache as used; MemAvailable
                    # adds it back.
                    avail = LlamaCppBackend._available_system_memory_mib()
                    # A zero total is a probe that could not size the pool, not a pool
                    # of zero: capping against it would take the device to nothing.
                    if avail is not None and total_mib > 0:
                        raw_mib = min(total_mib, max(raw_mib, avail))
                    # cgroup available is a ceiling too, since allocations are charged to the cgroup.
                    # `<=` on purpose: equality is normal when the cgroup caps MemAvailable.
                    cgroup_mib = LlamaCppBackend._cgroup_available_memory_mib()
                    if cgroup_mib is not None and cgroup_mib <= raw_mib:
                        raw_mib = cgroup_mib
                        # The pool is the container's: publish no total so the reserve is not taken
                        # from host memory.
                        cgroup_bound = True
                    # One pool shared by all integrated devices: split it so sums do not double count.
                    if shared_cuda_count > 1:
                        raw_mib //= shared_cuda_count
                        total_mib //= shared_cuda_count
                free_mib = _apply_igpu_host_reserve_mib(raw_mib, shared or integrated)
                if free_mib < raw_mib:
                    logger.info(
                        f"{'CUDA' if integrated else 'ROCm'} device {idx} is a "
                        f"unified-memory {'SoC' if integrated else 'APU'} sharing system "
                        f"RAM; reserving {raw_mib - free_mib}MiB host headroom "
                        f"({raw_mib}->{free_mib}MiB usable)"
                    )
                # ROCm publishes 0 for a shared pool because that total is system RAM of unknown scope; zeroing it would drop to free*frac.
                gpus.append((idx, free_mib, 0 if shared or cgroup_bound else total_mib))
            # Match the nvidia-smi path's docstring guarantee of sorted-by-id.
            return sorted(gpus, key = lambda g: g[0])
        except Exception as e:
            logger.debug(f"torch GPU probe failed: {e}")
            return []

    @staticmethod
    def _vulkan_auto_gpu_memory(gpus: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        """Prefer discrete devices for automatic Vulkan placement."""
        discrete = [gpu for gpu in gpus if gpu[2] > 0]
        return discrete or gpus

    @staticmethod
    def _run_vulkan_probe(binary: Optional[str] = None) -> list[dict]:
        """Indices are ggml's compact Vulkan ordinals, as --device Vulkan<i> selects, not raw
        enumeration."""
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return []
        # Memoised per load (15s subprocess); cleared by _load_probe_memo_scope.
        _memo = _load_probe_memo_get("vulkan", binary)
        if _memo is not None:
            return _memo
        binary_dir = _llama_lib_dir(binary)
        if not _lib_dir_has_ggml_backend(binary_dir, "vulkan"):
            _load_probe_memo_put("vulkan_outcome", binary, "error")
            _load_probe_memo_put("vulkan", binary, [])
            return []

        env = child_env_without_native_path_secret()
        # Let ggml apply GGML_VK_VISIBLE_DEVICES; Python sees compact ordinals.
        if sys.platform != "win32":
            # _loader_path_var keeps macOS off LD_LIBRARY_PATH, which dyld ignores.
            _loader_var = _loader_path_var()
            existing_ld = env.get(_loader_var, "")
            env[_loader_var] = (
                f"{binary_dir}{os.pathsep}{existing_ld}" if existing_ld else str(binary_dir)
            )
        probe_script = Path(__file__).resolve().parent.with_name("_vulkan_probe.py")
        try:
            # UTF-8: device names can be non-ASCII and a cp1252 decode could throw.
            result = subprocess.run(
                [sys.executable, str(probe_script), str(binary_dir)],
                capture_output = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 15,
                env = utf8_child_env(env),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode != 0:
                logger.debug(
                    f"vulkan GPU probe exited {result.returncode}: {result.stderr.strip()}"
                )
                # Suspect only, never proof: a driver can fault on exit after listing a device.
                _crashed = LlamaCppBackend._is_signal_crash(result.returncode) or (
                    sys.platform == "win32" and LlamaCppBackend._is_abort_exit(result.returncode)
                )
                _load_probe_memo_put("vulkan_outcome", binary, "crashed" if _crashed else "error")
                _load_probe_memo_put("vulkan", binary, [])
                return []
        except subprocess.TimeoutExpired as e:
            # Suspect, not absent: a healthy discrete card can take longer than this.
            logger.info(f"vulkan GPU probe timed out: {e}")
            _load_probe_memo_put("vulkan_outcome", binary, "timeout")
            _load_probe_memo_put("vulkan", binary, [])
            return []
        except Exception as e:
            logger.debug(f"vulkan GPU probe failed: {e}")
            # Memoised too: an unanswered probe comes straight back here, so an uncached
            # timeout is paid twice over and again per rung.
            _load_probe_memo_put("vulkan_outcome", binary, "error")
            _load_probe_memo_put("vulkan", binary, [])
            return []

        rows: list[dict] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            # 4 columns from an older probe (no name); 5 with the name column, 7 with the PCI id.
            if len(parts) not in (4, 5, 6, 7):
                continue
            try:
                free_b, total_b = int(parts[1]), int(parts[3])
                # Intel's Windows driver can report budget > heap size (ggml-org/llama.cpp#29277).
                if 0 < total_b < free_b:
                    free_b = total_b
                rows.append(
                    {
                        "index": int(parts[0]),
                        "free_mib": free_b // (1024 * 1024),
                        "is_igpu": parts[2] == "1",
                        "total_mib": total_b // (1024 * 1024),
                        "name": parts[4].strip() if len(parts) >= 5 else "",
                        # Absent on an older staged probe, so it reads unknown and declines.
                        "type_known": len(parts) >= 6 and parts[5].strip() == "1",
                        "pci_id": parts[6].strip().lower() if len(parts) >= 7 else "",
                    }
                )
            except ValueError:
                continue
        rows.sort(key = lambda r: r["index"])
        if rows:
            _outcome = "devices"
        else:
            # Unparseable output is not proof of zero devices.
            _outcome = "error" if result.stdout.strip() else "none"
        _load_probe_memo_put("vulkan_outcome", binary, _outcome)
        _load_probe_memo_put("vulkan", binary, rows)
        return rows

    @staticmethod
    def _vulkan_probe_outcome(binary: Optional[str] = None) -> str:
        """``devices``, ``none`` (enumerated zero), ``timeout``, ``crashed`` or ``error``,
        from this load's probe memo. Unarmed (outside a load) only ``devices`` / ``error``.
        """
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return "error"
        if LlamaCppBackend._run_vulkan_probe(binary):
            return "devices"
        outcome = _load_probe_memo_get("vulkan_outcome", binary)
        return outcome if outcome in ("none", "timeout", "crashed") else "error"

    @staticmethod
    def vulkan_device_inventory(binary: Optional[str] = None) -> list[dict]:
        """UI-facing Vulkan device list: the devices llama-server will actually
        use, with real totals (an iGPU keeps its shared-RAM total here -- the
        caller labels it, unlike the fit which zeroes it). Same rows as
        ``_run_vulkan_probe``; names fall back to ``Vulkan<i>``.
        """
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return []
        from utils import gpu_memory_events
        from utils.hardware import gpu_query

        use_cache = gpu_query.display_cache_allowed() and not _load_probe_memo_armed()
        key = ("vulkan", binary)
        if use_cache:
            with _vulkan_inventory_lock:
                entry = _vulkan_inventory_cache.get(key)
            if (
                entry is not None
                and time.monotonic() - entry[0] < _VULKAN_INVENTORY_TTL_S
                and entry[1] == gpu_memory_events.generation()
            ):
                return [dict(row) for row in entry[2]]
        generation = gpu_memory_events.generation()
        rows = LlamaCppBackend._run_vulkan_probe(binary)
        for row in rows:
            if not row["name"]:
                row["name"] = f"Vulkan{row['index']}"
        # An empty answer may be a failed probe: never served, like gpu_query's failures.
        with _vulkan_inventory_lock:
            if rows:
                _vulkan_inventory_cache[key] = (
                    time.monotonic(),
                    generation,
                    [dict(row) for row in rows],
                )
            else:
                _vulkan_inventory_cache.pop(key, None)
        return rows

    @staticmethod
    def _get_gpu_free_memory_vulkan(binary: Optional[str] = None) -> list[tuple[int, int, int]]:
        """Fit view of Vulkan probe: (index, free, total) sorted; iGPU total 0 so fit uses free*frac; [] when no device reachable."""
        gpus: list[tuple[int, int, int]] = []
        for row in LlamaCppBackend._run_vulkan_probe(binary):
            idx, free_mib, is_igpu = row["index"], row["free_mib"], row["is_igpu"]
            # iGPU total is shared RAM: keep 0 so the fit uses free*frac.
            total_mib = 0 if is_igpu else row["total_mib"]
            capped = _apply_igpu_host_reserve_mib(free_mib, is_igpu)
            if capped < free_mib:
                logger.info(
                    f"Vulkan device VK{idx} is an integrated GPU sharing system "
                    f"RAM; reserving {free_mib - capped}MiB host headroom "
                    f"({free_mib}->{capped}MiB usable)"
                )
            gpus.append((idx, capped, total_mib))
        if gpus:
            logger.info(
                "Vulkan GPU memory detected: "
                + ", ".join(f"VK{idx}={free}MiB" for idx, free, _total in gpus)
            )
        return gpus

    @staticmethod
    def _igpu_backed_free_mib(free_mib: int, headroom_mib: int = 0) -> int:
        """Bounds a raw iGPU free reading by host RAM; not applicable on WSL, whose VM and adapter
        disagree."""
        available = LlamaCppBackend._available_system_memory_mib()
        total_ram = LlamaCppBackend._total_system_memory_mib()
        if available is None or total_ram is None:
            return free_mib
        unseen = 0 if _is_wsl() else max(0, free_mib - total_ram)
        host_backed = max(0, available - max(0, headroom_mib))
        return min(free_mib, unseen + host_backed)

    @staticmethod
    def _igpu_backed_usable_mib(usable_mib: int) -> int:
        """Admission bound for post-reserve iGPU budget; placement keeps the raw reading to avoid
        demotion."""
        if usable_mib <= 0:
            return 0
        raw_free_mib = usable_mib + _IGPU_HOST_RESERVE_MIB
        backed_mib = LlamaCppBackend._igpu_backed_free_mib(
            raw_free_mib, headroom_mib = _HOST_RAM_HEADROOM_MIB
        )
        return min(usable_mib, backed_mib)

    @staticmethod
    def _cgroup_memory_budgets() -> list[tuple[int, int]]:
        """Memory this process may still charge to an enforcing cgroup; empty when no finite limit."""
        return _host_memory.cgroup_memory_budgets(_CGROUP_ROOT, _PROC_SELF_CGROUP)

    @staticmethod
    def _cgroup_available_memory_mib() -> Optional[int]:
        """What this process can still charge, i.e. the tightest REMAINDER."""
        return _host_memory.cgroup_headroom_mib(LlamaCppBackend._cgroup_memory_budgets())

    @staticmethod
    def _cgroup_memory_limit_mib() -> Optional[int]:
        """Tightest cgroup LIMIT, i.e. capacity; never the remainder, which shrinks as the container
        fills."""
        return _host_memory.cgroup_limit_mib(LlamaCppBackend._cgroup_memory_budgets())

    @staticmethod
    def _available_system_memory_mib() -> Optional[int]:
        """Available system RAM MiB (psutil, then /proc/meminfo), None if unreadable, capped by cgroup remainder; the APU ceiling."""
        return _host_memory.usable_mib(
            _host_memory.system_available_mib(), LlamaCppBackend._cgroup_available_memory_mib()
        )

    @staticmethod
    def _total_system_memory_mib() -> Optional[int]:
        """Total system RAM MiB (psutil, then /proc/meminfo), None if unreadable; the ceiling MemAvailable can reach."""
        return _host_memory.system_total_mib()

    @staticmethod
    def _host_memory_capacity_mib() -> Optional[int]:
        """Host RAM this process may charge: MemTotal narrowed by a cgroup LIMIT, not the remainder."""
        return _host_memory.usable_mib(
            LlamaCppBackend._total_system_memory_mib(), LlamaCppBackend._cgroup_memory_limit_mib()
        )

    _ARGV_MODEL = frozenset({"-m", "--model"})
    _ARGV_RPC = frozenset({"--rpc"})

    @staticmethod
    def _binary_ships_no_gpu_backend(
        binary: Optional[str] = None, env: Optional[Mapping[str, str]] = None
    ) -> bool:
        """True when no GPU backend; SYCL, MUSA, CANN, OpenCL count as GPU; static or unreadable layouts and GGML_BACKEND_PATH answer False."""
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return False
        source = os.environ if env is None else env
        if str(source.get("GGML_BACKEND_PATH", "") or "").strip():
            return False
        try:
            files = tuple(path.name for path in _llama_lib_dir(binary).iterdir() if path.is_file())
        except OSError:
            return False
        cpu_stem = "ggml-cpu" if sys.platform == "win32" else "libggml-cpu"
        base_stem = "ggml-base" if sys.platform == "win32" else "libggml-base"
        split_library = any(name.startswith((cpu_stem, base_stem)) for name in files)
        return split_library and not any(_GGML_GPU_BACKEND_RE.match(name) for name in files)

    def _argv_offloads_every_layer(
        self,
        argv: Iterable[str],
        env: Optional[Mapping[str, str]] = None,
    ) -> bool:
        """Unknown placement counts as partial offload, so unverified GPU memory is never credited."""
        args = [str(a) for a in argv or ()]
        if _device_selection_is_cpu(args, env):
            return False
        if _args_place_tensors_on_cpu(args) or _env_places_tensors_on_cpu(env):
            return False
        # The fitter owns placement unless it is explicitly disabled.
        if fit_is_effectively_on(args, env):
            return False
        try:
            requested = parse_gpu_layers_override(args)
        except ValueError:
            return False
        if requested is None:
            return False
        # -1 means every layer; otherwise the count must exceed the block count.
        n_layers = self.n_layers
        return requested == -1 or (bool(n_layers) and requested > n_layers)

    def _launch_forces_full_offload(
        self,
        argv: Iterable[str],
        env: Optional[Mapping[str, str]] = None,
    ) -> bool:
        """Whether every layer is on GPU: a user count above block count stands under --fit on; no count with fitter off (default -1) too."""
        args = [str(a) for a in argv or ()]
        if self._argv_offloads_every_layer(args, env):
            return True
        if _device_selection_is_cpu(args, env):
            return False
        if _args_place_tensors_on_cpu(args) or _env_places_tensors_on_cpu(env):
            return False
        try:
            requested = parse_gpu_layers_override(args)
        except ValueError:
            return False
        if requested is None and env and _env_fixes_gpu_layers(env):
            raw = str(env.get("LLAMA_ARG_N_GPU_LAYERS", "")).strip().lower()
            if raw == "all":
                return True
            try:
                requested = int(raw)
            except ValueError:
                return False
        if requested is None:
            return not fit_is_effectively_on(args, env)
        n_layers = self.n_layers
        if not n_layers:
            return False
        return requested > n_layers

    @staticmethod
    def _rows_the_child_can_reach(detected_gpus, pinned_ids) -> list:
        """Narrows detected GPUs to the launch pin set, not the inherited mask (already applied); None means no narrowing, not an empty pin."""
        rows = list(detected_gpus or ())
        if pinned_ids is None:
            return rows
        visible = set(pinned_ids)
        return [row for row in rows if row[0] in visible]

    def _shared_heap_budget(
        self,
        gpus: Iterable[tuple],
        shared_gpu_ids: Collection[int],
        model_bytes: int,
        argv: list[str],
        env: Optional[Mapping[str, str]] = None,
    ) -> tuple[int, int]:
        """Largest shared iGPU pool only, when a pin names it or unpinned with no discrete card; selected discrete cards reduce its bytes."""
        rows = [(row[0], max(0, row[1])) for row in gpus]
        pinned = _extra_args_main_device(argv)
        if pinned is None and env:
            pinned = env.get("LLAMA_ARG_DEVICE")
        if pinned is None:
            reachable = {idx for idx, _free in rows}
            # Unpinned, llama.cpp adds integrated GPUs only when it finds no discrete one.
            if any(idx not in shared_gpu_ids for idx in reachable):
                reachable -= set(shared_gpu_ids)
        else:
            # Match the ggml registry names emitted by _vulkan_pin_args.
            names = {device.strip().lower() for device in str(pinned).split(",")}
            reachable = {idx for idx, _free in rows if f"vulkan{idx}" in names}
        pool = 0
        if (
            shared_gpu_ids
            and self._argv_offloads_every_layer(argv, env)
            # Tensor-split positions cannot be mapped reliably to probe indices.
            and not _extra_args_set_any_flag(argv, _TENSOR_SPLIT_FLAGS)
            and not str((env or {}).get("LLAMA_ARG_TENSOR_SPLIT") or "").strip()
            # With multiple devices, split-mode none does not identify the recipient.
            and not (len(reachable) > 1 and _split_mode_confines_to_one_device(argv, env))
        ):
            budgets = []
            for idx, free in rows:
                if idx not in shared_gpu_ids or idx not in reachable:
                    continue
                backed = self._igpu_backed_usable_mib(free)
                if backed < free:
                    logger.info(
                        f"Vulkan device VK{idx} offers {free}MiB to placement; "
                        f"host admission credits {backed}MiB"
                    )
                budgets.append(backed)
            pool = max(budgets, default = 0)
        on_cards = sum(free for idx, free in rows if idx not in shared_gpu_ids and idx in reachable)
        return pool, model_bytes - on_cards * 1024 * 1024

    def _launch_host_shortfall_message(
        self,
        cmd: Iterable[str],
        detected_gpus: Iterable[tuple],
        env: Optional[Mapping[str, str]] = None,
        *,
        child_has_no_gpu: bool = False,
        avail_mib: Optional[int] = None,
        shared_gpu_ids: Iterable[int] = (),
        honor_opt_out: bool = True,
        integrated_gpu_ids: Iterable[int] = (),
    ) -> Optional[str]:
        """Advisory shortfall warning, never blocks; UNSLOTH_ALLOW_HOST_OFFLOAD=1 silences only the message, never the pageable-load override."""
        argv = [str(a) for a in cmd or ()]
        if not argv:
            return None
        model_path = _extra_args_device(argv, self._ARGV_MODEL)
        if not model_path:
            return None
        if honor_opt_out and self._host_offload_warning_opted_out():
            logger.info("UNSLOTH_ALLOW_HOST_OFFLOAD set: skipping the host-RAM preflight.")
            return None
        # rpc places layers on remote devices this cannot size, in either spelling
        _env = os.environ if env is None else env
        if (_extra_args_device(argv, self._ARGV_RPC) or "").strip() or str(
            _env.get("LLAMA_ARG_RPC", "") or ""
        ).strip():
            return None
        try:
            model_bytes = self._get_gguf_size_bytes(model_path)
        except Exception:
            return None
        gpus = [] if child_has_no_gpu else list(detected_gpus or ())
        # _get_gpu_memory returns [] on a failed probe, which looks like a GPU-less host.
        if not model_bytes or (not gpus and not child_has_no_gpu):
            return None
        shared = set(shared_gpu_ids or ())
        # Same lazy-read discount as the fit, or this rewrite undoes its pinned verdict.
        try:
            model_bytes = max(
                1,
                model_bytes
                - self._lazy_read_host_bytes(
                    model_path,
                    caps = self.probe_server_capabilities(argv[0]),
                    extra_args = argv,
                    env = _env,
                    auto_resolves_off = self._lazy_auto_resolves_off(
                        gpu_indices = [row[0] for row in gpus] if gpus else None,
                        detected_gpus = gpus,
                        shared_gpu_ids = shared,
                        is_vulkan_backend = bool(shared) or self._is_vulkan_backend(argv[0]),
                        probe_integrated_cuda = False,
                        extra_args = argv,
                        env = _env,
                        device_list = False,
                        integrated_gpu_ids = integrated_gpu_ids,
                    ),
                ),
            )
        except Exception as e:  # noqa: BLE001 -- unverified discount: price the whole file
            logger.debug("Lazy-read discount skipped in the host preflight: %s", e)
        free_vram_mib = sum(max(0, row[1]) for row in gpus if row[0] not in shared)
        heap_free_mib, heap_bytes = self._shared_heap_budget(gpus, shared, model_bytes, argv, _env)
        offload_bytes = model_bytes - free_vram_mib * 1024 * 1024
        # Skip card-resident loads so a cgroup budget never applies to a nonpositive need.
        if 0 < offload_bytes and heap_bytes <= heap_free_mib * 1024 * 1024:
            # A finite cgroup still limits host-backed GPU allocations.
            cgroup_mib = self._cgroup_available_memory_mib()
            if cgroup_mib is None:
                return None
            return self._apu_ram_shortfall_message(heap_bytes, cgroup_mib)
        return self._host_offload_shortfall_message(
            offload_bytes,
            self._available_system_memory_mib() if avail_mib is None else avail_mib,
        )

    @staticmethod
    def _apu_ram_shortfall_message(
        model_size_bytes: int,
        avail_mib: Optional[int],
        headroom_mib: int = _HOST_RAM_HEADROOM_MIB,
        *,
        part: str = "APU",
    ) -> Optional[str]:
        """Weights-only RAM shortfall on unified-memory parts; KV not counted; None if avail unknown; advisory, never a refusal."""
        if avail_mib is None:
            return None
        need_mib = model_size_bytes / (1024 * 1024)
        if need_mib <= avail_mib - headroom_mib:
            return None
        free_hint = " (on WSL, raise the memory limit in .wslconfig)" if part == "APU" else ""
        return (
            f"This model needs about {need_mib / 1024:.0f} GB but only about "
            f"{avail_mib / 1024:.0f} GB of memory is available. On a unified-memory "
            f"{part} the weights load into system RAM, so the OS may stop the load. "
            "Loading anyway. If it does not complete, use a smaller or more "
            f"quantized GGUF, or free memory{free_hint}."
        )

    @staticmethod
    def _igpu_dedicated_memory_bytes(
        gpu_indices = None, *, ordinals_are_vulkan = False
    ) -> Optional[int]:
        """Dedicated iGPU bytes: Windows registry first (cheap), then the ROCm pool; None rather
        than guess."""
        # Every listed adapter counts, readable or not: the count IS the attribution test.
        answers: list[Optional[dict]] = []
        try:
            from utils.hardware.hardware import (
                _AMD_PCI_VENDOR_ID,
                _INTEL_PCI_VENDOR_ID,
                _windows_amd_adapter_records_by_luid,
            )
            for vendor_id in (_AMD_PCI_VENDOR_ID, _INTEL_PCI_VENDOR_ID):
                answers.append(
                    _windows_amd_adapter_records_by_luid(vendor_id, distinguish_failure = True)
                )
        except Exception:
            answers = []
        if any(answer is not None for answer in answers):
            # An unreadable vendor makes the inventory incomplete, so fail closed.
            if any(answer is None for answer in answers):
                return None
            adapters = sum(len(answer) for answer in answers)
            # AMD only: on Intel UMA the DirectX value is a small dedicated block, not a carve-out.
            amd_sizes = [
                int(record["dedicated_memory_bytes"])
                for record in (answers[0] or {}).values()
                if record.get("dedicated_memory_bytes")
            ]
            if adapters == 1 and len(amd_sizes) == 1 and amd_sizes[0] > 0:
                return amd_sizes[0]
            if adapters:
                # Two adapters need a LUID join to attribute; staying silent beats advising the wrong GPU.
                return None
        if ordinals_are_vulkan:
            # No Linux reading for Vulkan: gpu_indices are Vulkan ordinals, not HIP ids, and probing
            # would create a HIP context (~800 MiB) in this process.
            return None
        pool_mib = LlamaCppBackend._rocm_selected_pool_mib(gpu_indices)
        return int(pool_mib) * 1024 * 1024 if pool_mib and pool_mib > 0 else None

    @staticmethod
    def _igpu_carveout_ladder_gb(cap_gb: float) -> list[int]:
        """Ascending carve-out sizes up to cap_gb, in firmware-menu steps (powers of two and halves)."""
        # A non-finite cap would hang the loop below; reject it first.
        if not isinstance(cap_gb, (int, float)) or not math.isfinite(cap_gb):
            return []
        rungs: set[int] = set()
        # From 1 GB: an APU on its automatic setting reports a few hundred MB.
        step = 1
        while step <= cap_gb:
            rungs.add(step)
            if step * 1.5 <= cap_gb:
                rungs.add(int(step * 1.5))
            step *= 2
        return sorted(rungs)

    @staticmethod
    def _igpu_carveout_advice(
        model_size_bytes: Optional[int],
        carve_out_bytes: Optional[int],
        host_total_bytes: Optional[int],
        *,
        is_igpu: bool,
        min_host_gb: int = _CARVEOUT_ADVICE_MIN_HOST_GB,
        host_fraction: float = _CARVEOUT_ADVICE_HOST_FRACTION,
        nominal_slack_gb: float = _CARVEOUT_NOMINAL_SLACK_GB,
    ) -> Optional[dict]:
        """Advice to raise an iGPU dedicated allocation when weights exceed it; silent on discrete GPUs and when no allocation fits; only advises."""
        if not is_igpu:
            return None
        # Strictly positive, not truthy: -1 would yield confident wrong advice.
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in (model_size_bytes, carve_out_bytes, host_total_bytes)
        ):
            return None
        if model_size_bytes <= carve_out_bytes:
            return None  # already fits: nothing to advise

        gb = float(1024**3)
        # Windows subtracts the carve-out from reported RAM, so add it back.
        machine_gb = (host_total_bytes + carve_out_bytes) / gb
        need_gb = model_size_bytes / gb
        current_gb = carve_out_bytes / gb

        # Whatever is suggested, the rest of the system still has to run.
        reserve_gb = max(float(min_host_gb), machine_gb * host_fraction)
        cap_gb = machine_gb - reserve_gb
        if need_gb > cap_gb:
            # No allocation this machine can offer holds the weights, so any advice would be unactionable.
            return None

        suggested = next(
            (rung for rung in LlamaCppBackend._igpu_carveout_ladder_gb(cap_gb) if rung >= need_gb),
            None,
        )
        # Not `<= current_gb`: the current rung sits just above the reading. See _CARVEOUT_NOMINAL_SLACK_GB.
        if suggested is None or suggested <= current_gb + nominal_slack_gb:
            return None

        return {
            "current_gb": round(current_gb, 1),
            "needed_gb": round(need_gb, 1),
            "suggested_gb": int(suggested),
            "machine_gb": round(machine_gb, 1),
            "host_left_gb": round(machine_gb - suggested, 1),
        }

    def _record_carveout_advice(
        self,
        gpu_indices,
        need_bytes,
        *,
        is_vulkan_backend = False,
        shared_gpu_ids = None,
        detected_gpus = None,
        target_unknown = False,
        forced_cpu = False,
    ) -> None:
        """Sets _last_carveout_advice; never raises, runs the cheapest test first; declines if target unknown or load is forced to CPU."""
        self._last_carveout_advice = None
        try:
            if not need_bytes or target_unknown or forced_cpu:
                return
            # Gated here: on Vulkan gpu_indices are Vulkan ordinals, which the ROCm gate would misread
            # as HIP ids. Also cheap, so dGPU-only Vulkan launches return early.
            if is_vulkan_backend and not self._offload_target_shares_system_memory(
                is_vulkan_backend = True,
                shared_gpu_ids = shared_gpu_ids,
                detected_gpus = detected_gpus,
                gpu_indices = gpu_indices,
            ):
                return
            carve_out = self._igpu_dedicated_memory_bytes(
                gpu_indices, ordinals_are_vulkan = is_vulkan_backend
            )
            if not carve_out or need_bytes <= carve_out:
                return  # fits, or nothing to compare it against
            total_mib = self._total_system_memory_mib()
            advice = self._igpu_carveout_advice(
                need_bytes,
                carve_out,
                int(total_mib) * 1024 * 1024 if total_mib else None,
                is_igpu = True,
            )
            if advice is None:
                return
            # Probe only after a confirmed shortfall. AMD only (CUDA would create a context per device),
            # and not on Vulkan, which was classified above.
            if not is_vulkan_backend and not self._amd_apu_wants_unified_memory(gpu_indices):
                return
            # Asked last, so the common path never pays a database round trip.
            from utils.igpu_carveout_notice_settings import notice_already_dismissed

            if notice_already_dismissed(advice.get("current_gb")):
                return
            advice["message"] = self._igpu_carveout_advice_message(advice)
            self._last_carveout_advice = advice
            logger.info(
                "Integrated GPU has %.0f GB dedicated but this model needs about "
                "%.0f GB; suggesting %d GB.",
                advice["current_gb"],
                advice["needed_gb"],
                advice["suggested_gb"],
            )
        except Exception:
            logger.debug("Carve-out advice declined", exc_info = True)
            self._last_carveout_advice = None

    @property
    def last_carveout_advice(self) -> Optional[dict]:
        """Advice from the most recent load, or None. Read by the route."""
        return getattr(self, "_last_carveout_advice", None)

    @staticmethod
    def _fmt_gb(value: float) -> str:
        """Whole numbers above 10 GB, one decimal below, so small allocations do not print as 0 GB."""
        if value < 10:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{value:.0f}"

    @staticmethod
    def _igpu_carveout_advice_message(advice: dict) -> str:
        """Two short sentences, since a longer toast covers the controls under it; names no vendor
        or menu."""
        fmt = LlamaCppBackend._fmt_gb
        return (
            f"Weights need about {fmt(advice['needed_gb'])} GB but only "
            f"{fmt(advice['current_gb'])} GB is allocated to the integrated GPU, so the rest "
            f"runs from slower shared memory. Raising it to {advice['suggested_gb']} GB in "
            "your firmware or GPU control panel leaves about "
            f"{fmt(advice['host_left_gb'])} GB for the system."
        )

    @staticmethod
    def _host_offload_shortfall_message(
        offload_bytes: int,
        avail_mib: Optional[int],
        headroom_mib: int = _HOST_RAM_HEADROOM_MIB,
    ) -> Optional[str]:
        """Discrete-GPU spill warning vs free VRAM, no margin; None if avail unknown or nothing spills; advisory, never a refusal."""
        if offload_bytes <= 0 or avail_mib is None:
            return None
        need_mib = offload_bytes / (1024 * 1024)
        if need_mib <= avail_mib - headroom_mib:
            return None
        # need up, usable down, so the printed pair cannot round into a tie
        need_gb = math.ceil(need_mib / 1024)
        usable_gb = math.floor(max(0, avail_mib - headroom_mib) / 1024)
        return (
            f"About {need_gb} GB of this model does not fit in GPU memory and would run "
            f"from system RAM. Only about {avail_mib / 1024:.0f} GB is available and "
            f"{headroom_mib / 1024:.0f} GB of that is kept free for the rest of the "
            f"system, leaving about {usable_gb} GB usable. The weights are memory-mapped, "
            "so the machine pages them in from disk as it runs. Loading anyway: expect "
            "slow generation, and slower still on a hard disk than on an SSD. A smaller "
            "or more quantized GGUF, or more free memory, would run faster."
        )

    @staticmethod
    def _host_offload_warning_opted_out() -> bool:
        """UNSLOTH_ALLOW_HOST_OFFLOAD (deprecated) may only silence the warning, never change the child."""
        return os.environ.get("UNSLOTH_ALLOW_HOST_OFFLOAD", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
