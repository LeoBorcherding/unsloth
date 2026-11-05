# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, tensor spill planning."""

import os

from typing import (
    Iterable,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _DEVICE_FLAGS,
    _GPU_LAYER_FLAGS,
    fit_target_margin_in,
)

from .batch_args import (
    _extra_args_n_parallel,
    _extra_args_split_mode,
    _extra_args_tensor_split,
)
from .binary_runtime import _HOST_RAM_HEADROOM_MIB
from .draft_args import (
    _extra_args_device,
    _extra_args_draft_device_pin,
    _sidecar_adapter_bytes,
    _without_subsequence,
)
from .launch_flags import (
    _args_place_tensors_on_cpu,
    _env_places_tensors_on_cpu,
    _kv_offload_from_args,
    _spilled_decode_threads,
)
from .log import logger
from .memory_policy import (
    _TENSOR_SPLIT_FLAGS,
    _extra_args_set_any_flag,
)


class _TensorSpillMixin:
    def _planned_tensor_spill(
        self,
        inputs: Optional[dict],
        *,
        extra_args: Optional[Iterable[str]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "Optional[SpillPlan]":
        """Abstains (None) on any uncertain input, so argv stays byte-identical to today's --fit on."""
        from core.inference.offload_cost_model import HostProfile
        from core.inference.offload_planner import (
            PlanOptions,
            plan_placement,
            smart_offload_enabled,
        )

        source_env = os.environ if env is None else env
        if not smart_offload_enabled(source_env) or not inputs:
            return None

        # Integrated CUDA SoCs (Jetson, DGX Spark) are unified memory (cudaDeviceProp::integrated); no other check sees them, so a spill would count one pool twice
        from utils.hardware import is_apple_silicon

        if is_apple_silicon():
            return None

        # Integrated CUDA SoCs (Jetson, DGX Spark) are unified memory too; a spill gains nothing.
        if self._integrated_cuda_unified_memory(inputs.get("gpu_indices")):
            logger.debug("Tensor spill: declined, an integrated CUDA device is unified memory")
            return None

        # Someone else owns the placement -- decline. Each of these re-places the
        # model out from under a plan made here: -ot / --cpu-moe (or its env twin)
        # already moves tensors, a layer count overrides -ngl -1, a device list
        # changes which cards the resident half was sized against.
        if (
            _args_place_tensors_on_cpu(extra_args)
            or _env_places_tensors_on_cpu(source_env)
            or _extra_args_set_any_flag(extra_args, _GPU_LAYER_FLAGS)
            or str(source_env.get("LLAMA_ARG_N_GPU_LAYERS", "")).strip()
            or _extra_args_set_any_flag(extra_args, _DEVICE_FLAGS)
            or str(source_env.get("LLAMA_ARG_DEVICE", "")).strip()
            # A pinned draft device would be booked on device 0 by the planner; decline.
            or _extra_args_draft_device_pin(extra_args)
            # -sm row/tensor split within tensors, which the row model cannot describe.
            or (_extra_args_split_mode(extra_args, source_env) in ("row", "tensor"))
            # Inherited -sm env may be stripped by load_model later; only argv provably survives.
            or (
                _extra_args_split_mode(extra_args, source_env) == "none"
                and _extra_args_split_mode(extra_args, {}) != "none"
            )
            # LLAMA_ARG_MAIN_GPU still reaches the child and changes which card -sm none keeps.
            or str(source_env.get("LLAMA_ARG_MAIN_GPU", "")).strip()
            # Any explicit --fit: the retry's appended '--fit on' would override a user '--fit off'.
            or any(
                str(a) in ("-fit", "--fit") or str(a).startswith(("-fit=", "--fit="))
                for a in (extra_args or ())
            )
            or str(source_env.get("LLAMA_ARG_FIT", "")).strip()
            # --rpc devices are remote and go first in the list, so local VRAM numbers are wrong.
            or (_extra_args_device(extra_args, {"--rpc"}) or "").strip()
            or str(source_env.get("LLAMA_ARG_RPC", "")).strip()
            # --fit off would silently drop the user's --fit-target margin.
            or fit_target_margin_in(extra_args, source_env) is not None
        ):
            logger.debug("Tensor spill: declined, pass-through arguments own the placement")
            return None

        # Slots are SIZING, not placement, and these numbers were priced at Unsloth's
        # own --parallel. A pass-through is appended after it and llama.cpp is
        # last-wins, so a larger one grows the attention cache and the recurrent
        # state under a deficit computed for the smaller count. Only larger matters:
        # a smaller value leaves the plan over-reserved, the safe direction.
        priced_parallel = int(inputs.get("n_parallel") or 1)
        override_parallel = _extra_args_n_parallel(extra_args, source_env)
        if override_parallel is not None and override_parallel > priced_parallel:
            logger.debug(
                "Tensor spill: declined, --parallel %d overrides the %d slots this was priced at",
                override_parallel,
                priced_parallel,
            )
            return None

        model_size = int(inputs.get("model_size") or 0)
        kv_cache_bytes = int(inputs.get("kv_cache_bytes") or 0)
        rows = list(inputs.get("gpus") or ())
        if not model_size or not kv_cache_bytes or not rows:
            return None
        if not self._can_estimate_kv():
            return None

        layout = self._tensor_spill_layout(inputs.get("model_path"))
        if layout is None or not layout.complete:
            return None

        # Shared rows are dropped: an iGPU reports host RAM as VRAM, double counting the pool.
        pinned = inputs.get("gpu_indices")
        shared = set(inputs.get("shared_gpu_ids") or ())
        pinned_set = None if pinned is None else set(pinned)
        # The child may still use the shared device and an iGPU takes most layers; decline.
        if shared and (pinned_set is None or (shared & pinned_set)):
            logger.debug(
                "Tensor spill: declined, a shared-memory GPU stays in the child's device list"
            )
            return None
        # Usable budget from Unsloth's fit, falling back to raw free.
        usable_mib = inputs.get("gpu_usable_mib") or {}
        kept = [
            (idx, free_mib)
            for idx, free_mib in rows
            if idx not in shared and (pinned_set is None or idx in pinned_set)
        ]
        # -sm none is not a reason to decline, it is a device list of one:
        # llama.cpp keeps only devices[main_gpu] (llama.cpp:288-299), the
        # single-GPU case this planner has the most evidence for. -mg and
        # LLAMA_ARG_MAIN_GPU are declined above, so main_gpu is llama.cpp's default
        # of 0 -- the first kept row. ARGV only, deliberately: the env spelling is
        # declined above because load_model may strip it before launch, so it does
        # not describe the child's device list.
        if _extra_args_split_mode(extra_args, {}) == "none" and kept:
            kept = kept[:1]
        vram_per_device = [
            int(usable_mib.get(idx, free_mib) * 1024 * 1024) for idx, free_mib in kept
        ]
        # -ts replaces the free-memory split, one entry per kept device, so lengths must match; read the env the child gets since load_model may pop it
        split_weights = [int(free_mib * 1024 * 1024) for _idx, free_mib in kept]
        # -ts replaces the free-memory split. Read the env the child gets: load_model may pop it.
        _inherited_sm = str(source_env.get("LLAMA_ARG_SPLIT_MODE", "")).strip().lower()
        _ts_env = source_env if _inherited_sm in ("", "layer") else {}
        _ts = _extra_args_tensor_split(extra_args, _ts_env)
        if _ts is None and (
            _extra_args_set_any_flag(extra_args, _TENSOR_SPLIT_FLAGS)
            or str(_ts_env.get("LLAMA_ARG_TENSOR_SPLIT", "")).strip()
        ):
            # An unparseable -ts still reaches the child, so decline.
            logger.debug("Tensor spill: declined, -ts is present but could not be parsed")
            return None
        if _ts is not None:
            if len(_ts) < len(kept):
                logger.debug(
                    "Tensor spill: declined, -ts names %d shares for %d devices",
                    len(_ts),
                    len(kept),
                )
                return None
            split_weights = _ts[: len(kept)]
        if not vram_per_device:
            return None
        avail_mib = self._available_system_memory_mib()
        if avail_mib is None:
            return None

        # Charge back the dropped MTP blocks when a draft will engage; the layout omits them, so the deficit would be short by the whole MTP head
        if inputs.get("separate_draft_on_gpu") and len(vram_per_device) > 1:
            logger.debug(
                "Tensor spill: declined, a separate drafter's bytes cannot be placed "
                "across %d devices",
                len(vram_per_device),
            )
            return None

        # Charge back the dropped MTP blocks when a draft will engage.
        extra_gpu_bytes = int(inputs.get("extra_gpu_bytes") or 0)
        if inputs.get("mtp_will_engage") and layout.excluded_block_bytes:
            extra_gpu_bytes += layout.excluded_block_bytes

        # Flat compute buffer, charged once on device 0.
        extra_gpu_bytes += int(inputs.get("compute_buffer_flat") or 0)

        # A projector set via env that Studio never sized still loads on GPU; decline.
        if inputs.get("env_mmproj_unsized"):
            logger.debug("Tensor spill: declined, an inherited projector could not be sized")
            return None
        extra_gpu_bytes += int(inputs.get("env_mmproj_bytes") or 0)

        # Pass-through LoRA / control-vector weights are GPU resident and never mmapped.
        sidecar_bytes = _sidecar_adapter_bytes(extra_args)
        if sidecar_bytes is None:
            logger.debug("Tensor spill: declined, a pass-through adapter could not be sized")
            return None
        # ...but a flat term is only right on ONE device. extra_resident_bytes is
        # charged entirely to device 0 (_per_device_shortfall), while a LoRA tensor
        # is allocated on its BASE tensor's buffer (llama-adapter.cpp:335) and a
        # control-vector row on model.select_buft(il) (:67) -- so on a layer split
        # they follow the layers onto every card. Booking them all on the main GPU
        # under-books the others, and --fit off means common/fit.cpp never runs to
        # catch the overflow. The layout is built from the base GGUF alone and holds
        # no adapter tensor table, so the per-device share cannot be derived here.
        # Decline, like every other placement this planner cannot price.
        if sidecar_bytes and len(vram_per_device) > 1:
            logger.debug(
                "Tensor spill: declined, a pass-through adapter's bytes cannot be placed "
                "across %d devices",
                len(vram_per_device),
            )
            return None
        extra_gpu_bytes += sidecar_bytes

        decode_threads = _spilled_decode_threads(
            inputs.get("n_threads"),
            extra_args,
        )
        if decode_threads is None:
            logger.debug("Tensor spill: declined, decode CPU settings cannot be priced")
            return None

        return plan_placement(
            layout,
            vram_per_device,
            avail_mib * 1024 * 1024,
            int(inputs.get("n_ctx") or 0),
            split_weights_per_device = split_weights,
            kv_layer_weights = list(inputs.get("kv_layer_weights") or ()),
            opts = PlanOptions(
                overhead_bytes_per_device = (
                    int(inputs.get("soft_overhead") or 0)
                    + int(inputs.get("ctx_compute_per_device") or 0)
                ),
                # Charged only for devices after the first, like elsewhere in this file.
                pipeline_overhead_bytes = self._PIPELINE_PER_DEVICE_OVERHEAD_MIB * 1024 * 1024,
                # Module-level, unlike the line above, so not reachable through self.
                host_ram_headroom_bytes = _HOST_RAM_HEADROOM_MIB * 1024 * 1024,
                # -nkvo moves cache out of VRAM, so it shrinks the deficit.
                kv_on_host = not _kv_offload_from_args(extra_args, env = source_env),
                lazy_read_bytes = int(inputs.get("lazy_read_bytes") or 0),
                # GPU-resident allocations the layout cannot see, since it is built
                # from the target GGUF alone: the vision projector and the MTP draft
                # reserve. Both are in the footprint that produced the use_fit
                # verdict, so omitting them makes the deficit too small.
                extra_resident_bytes = extra_gpu_bytes,
                # Spilled decode runs on the CPU backend (ggml migrates an op to the
                # GPU only at batch >= 32, and decode is batch 1), so the penalty
                # tracks core count. Read the real one, not a default.
                host = HostProfile(
                    threads = decode_threads,
                    # On unified-memory APUs a spill frees no VRAM; the planner abstains only if told.
                    unified_memory = self._amd_apu_wants_unified_memory(inputs.get("gpu_indices")),
                ),
            ),
            # Plan has no __bool__, so emit flags only if it spills; one -ot per pattern, not semicolon-joined; --fit off is required since -ngl -1 does not hold it off
            kv_bytes_floor = kv_cache_bytes,
        )

    @staticmethod
    def _spill_plan_flags_for(plan: "Optional[SpillPlan]") -> "list[str]":
        """Plan has no __bool__, so every instance is truthy; only a plan that spills anything earns
        flags."""
        if plan is None or not plan.spills_anything:
            return []
        tokens = [tok for pat in plan.ot_patterns for tok in ("-ot", f"{pat}=CPU")]
        if not tokens:
            return []
        return ["-ngl", "-1", "--fit", "off", *tokens]

    def _drop_managed_lazy_mode(self, run_cmd: "list[str]", why: str) -> "list[str]":
        """Take Unsloth's ``--lazy-mode off`` back out of an argv for a retry."""
        if not getattr(self, "_lazy_mode_flags", None):
            return run_cmd
        stripped = _without_subsequence(run_cmd, self._lazy_mode_flags)
        if stripped == run_cmd:
            return run_cmd
        logger.info(
            "Lookup tables: dropping %s for the %s retry; llama.cpp streams the table.",
            " ".join(self._lazy_mode_flags),
            why,
        )
        return stripped

    def _drop_tensor_spill(self, run_cmd: "list[str]", why: str) -> "list[str]":
        """Retries invalidate the plan, so placement returns to --fit on; the record stays for CPU
        fallback."""
        if not self._spill_plan_flags:
            return run_cmd
        stripped = _without_subsequence(run_cmd, self._spill_plan_flags)
        if stripped == run_cmd:
            return run_cmd
        logger.info("Tensor spill: dropping the plan for the %s retry; %s", why, "using --fit on")
        return [*stripped, "--fit", "on"]

    def _tensor_spill_layout(
        self,
        model_path: "Optional[str]",
        *,
        all_shards: bool = False,
    ) -> "Optional[ModelLayout]":
        """Per block so the planner spills the minimum set; a read failure returns None and abstains."""
        from core.inference.offload_layout import layout_from_gguf

        if not model_path:
            return None
        # Keyed on (size, mtime_ns), not name, so a replaced file is re-read.
        try:
            st = os.stat(model_path)
            key = (model_path, st.st_size, st.st_mtime_ns, all_shards)
        except OSError:
            key = (model_path, None, None, all_shards)
        cached = getattr(self, "_spill_layout_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            layout = (
                layout_from_gguf(model_path, all_shards = True)
                if all_shards
                else layout_from_gguf(model_path)
            )
        except Exception as e:  # unreadable, truncated, or an arch we cannot bucket
            logger.debug("Tensor spill: cannot read layout from %s (%s)", model_path, e)
            layout = None
        self._spill_layout_cache = (key, layout)
        return layout

    @staticmethod
    def _fit_off_retry_eligible(cmd: "list[str]", use_fit: bool) -> bool:
        """Only when Unsloth placed the model and no -fit/--fit flag chose the mode; --fit-ctx does
        not block."""
        if use_fit:
            return False
        for a in cmd:
            if a in ("-fit", "--fit") or a.startswith(("-fit=", "--fit=")):
                return False
        return True
