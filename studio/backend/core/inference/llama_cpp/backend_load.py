# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, load_model and load intent matching."""

import contextlib
import functools
import os
import re
import subprocess
import sys
import threading
import time

from dataclasses import replace
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Optional,
    Union,
)
from core.inference.llama_server_args import (
    _CACHE_TYPE_K_FLAGS,
    _CACHE_TYPE_V_FLAGS,
    _CTX_CHECKPOINTS_FLAGS,
    _effective_tensor_parallel,
    _flag_name,
    _GPU_LAYER_FLAGS,
    _last_flag_value,
    apply_load_mode_policy,
    apply_model_memory_policy,
    extra_args_disable_mmproj,
    extra_args_mmproj_auto,
    extra_args_select_load_mode,
    fit_is_effectively_on,
    fit_is_enabled_in,
    fit_target_margin_in,
    MANAGED_DIO_FLAGS,
    memory_env_selects_load_mode,
    memory_state_satisfies_settings,
    no_reserve_requires_dio,
    parse_cache_override,
    parse_cache_override_per_axis,
    parse_ctx_override,
    parse_split_mode_override,
    resolve_ctx_checkpoints,
    resolve_launch_load_mode,
    resolve_reasoning_budget,
    resolve_reasoning_budget_message,
    resolve_reasoning_budget_message_with_env,
    resolve_reasoning_budget_with_env,
    resolve_requested_ctx,
    scrub_denied_env,
    scrub_memory_env,
    strip_context_only,
    strip_shadowing_flags,
    strip_split_mode_only,
)
from utils.gpu_memory_events import (
    invalidate_gpu_memory as _invalidate_gpu_memory,
    invalidates_gpu_memory as _invalidates_gpu_memory,
)
from utils.log_retention import prune_log_dir
from utils.process_lifetime import (
    child_popen_kwargs as _child_popen_kwargs,
    spawn_on_lifetime_thread as _spawn_on_lifetime_thread,
)
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

from .batch_args import (
    _batch_ubatch_for_mmproj,
    _embedding_batch_ubatch,
    _emitted_n_batch,
    _expert_spill_places_tensors_on_cpu,
    _extra_args_n_parallel,
    _extra_args_n_ubatch,
    _launch_required_ubatch,
    _moe_spill_batch_ubatch,
    _named_batch_sizes,
    _read_gguf_embedding_length,
    _with_parallel_slots,
)
from .binary_runtime import (
    _DEVICE_VISIBILITY_ENV,
    _arm_load_probe_memo,
)
from .core_types import (
    GgufLoadIntent,
    GpuMemoryShortError,
    LLAMA_SERVER_NOT_FOUND_DETAIL,
    LlamaServerNotFoundError,
    _CombinedCancelEvent,
    _SPEC_ENV_VARS,
    _gpu_plan_mib,
    _gpu_short_message,
    _net_of_held_vram,
)
from .draft_args import (
    _COMPANION_DEVICE_FLAG_GROUPS,
    _LOCAL_DRAFT_FLAGS,
    _VALID_KV_CACHE_TYPES,
    _device_selection_is_cpu,
    _draft_is_split_across_host,
    _extra_args_device,
    _extra_args_draft_cache_types,
    _extra_args_draft_offloaded_to_cpu,
    _extra_args_main_device,
    _extra_args_mtp_draft_path,
    _extra_args_requests_separate_draft,
    _extra_args_spec_draft_n_max,
    _pass_through_overrides_placement,
    _sidecar_adapter_bytes,
    _subsequence_index,
    _widen_pin_ids_for_companion_devices,
    _without_subsequence,
)
from .drafters import (
    _DRAFTER_DISPLAY_LABELS,
    _is_gemma_mtp_name,
    _is_mtp_model_name,
)
from .gguf_cache import (
    _mtp_drafter_loads_standalone,
    _resolve_repo_id_casing,
)
from .hf_offline import (
    _extract_model_size_b,
    _hf_offline_if_unreachable,
)
from .launch_flags import (
    _DEFAULT_LLAMA_N_UBATCH,
    _TARGET_ROLLBACK_SPEC_TYPES,
    _accumulated_spec_types,
    _architecture_forces_flash_attn_off,
    _args_place_tensors_on_cpu,
    _child_spec_env,
    _dflash_drafter_breaks_on_images,
    _env_asks_for_the_native_context,
    _env_ctx_checkpoints_override,
    _env_fixes_gpu_layers,
    _env_places_tensors_on_cpu,
    _extra_args_requests_dflash,
    _extra_args_requests_dspark,
    _extra_args_requests_mtp,
    _flash_attn_enabled_from_args,
    _is_positive_int,
    _kv_offload_from_args,
    _kv_unified_from_args,
    _pipeline_parallel_disabled_by_args,
    _placement_is_fitter_proof,
    _planned_flash_attn_state,
    _reasoning_effort_from_env,
    _reasoning_from_env,
    _reserved_flash_attn_state,
    _swa_full_from_args_or_env,
    _user_fit_disabled,
)
from .launch_support import (
    _extra_args_have_tensor_split,
    _llama_server_api_key_enabled,
    _llama_server_key_launch,
    _metal_capable_host,
)
from .load_markers import _with_gguf_load_marker
from .log import logger
from .memory_policy import (
    _AUTO_OFFLOAD_CTX,
    _CTX_FIT_VRAM_FRACTION,
    _FAST_MTP_MIN_CTX,
    _FIT_CTX_FLAGS,
    _FIT_FLOOR_MIN_CTX,
    _FIT_MIN_CTX,
    _GPU_OFFLOAD_OVERRIDE_FLAGS,
    _GPU_UNACCELERATED_CACHE_TYPES,
    _LLAMA_FIT_MIN_CTX,
    _LLAMA_FIT_TARGET_DEFAULT_MIB,
    _MTP_MIN_SIZE_B,
    _MTP_SLOT_CTX_TRIM,
    _MTP_VRAM_RESERVE_FRAC,
    _THREAD_OVERRIDE_FLAGS,
    _VALID_CACHE_TYPES,
    _VRAM_FIT_VOIDING_ENV_VARS,
    _VRAM_FIT_VOIDING_FLAGS,
    _active_vram_fraction,
    _arch_has_fast_mla_mtp,
    _effective_main_cache_types,
    _env_main_cache_type_for_budget,
    _extra_args_main_cache_type_for_budget,
    _extra_args_set_any_flag,
    _extra_args_set_spec_type,
    _kv_cache_gpu_fallback_warning,
    _mla_mtp_auto_enabled,
    _model_memory_settings_or_none,
    _planned_main_cache_types,
    _planned_scratch_cache_type,
    _vram_usable_mib,
)
from .moe_cache import (
    _MOE_CACHE_AUTO_TOKENS,
    _MOE_CACHE_ENV,
    _MOE_CACHE_ERROR_RE,
    _MOE_CACHE_WITH_DRAFTER,
    _MOE_SPILL_UBATCH_ENV,
    _argv_loads_pinned,
    _moe_cache_share_admits,
    _moe_resident_expert_share,
    _moe_spill_raise_keeps_experts,
    _moe_switch,
    _prompt_cache_host_bytes,
)
from .placement_args import (
    _argv_keeps_projector_on_gpu,
    _env_sets_mmproj_offload,
    _extra_args_set_mmproj_offload,
    _idle_slot_clearing_active,
    _metal_device_is_paravirtual,
    _mmproj_env_is_audio_only,
    _paravirtual_draft_ngl_flag,
    _paravirtual_mmproj_pinnable,
    _paravirtual_probe_answered,
    _paravirtual_split_mode_pin,
    _paravirtual_strip_gpu_overrides,
    _resolved_mmproj_offload,
    _video_fps_flags,
    effective_ctx_checkpoints_for_caps,
    paravirtual_normalized_request,
)
from .reasoning import detect_reasoning_flags
from .server_log import (
    _PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR,
    _VULKAN_DIFFUSION_GPU_IDS_ERROR,
    _trace_verbosity_args,
    classify_gpu_offload_lines,
    llama_saw_gpu_device,
    parse_gpu_offload_counts,
)
from .spec_args import (
    _build_launch_reasoning_args,
    _build_ngram_mod_flags,
    _build_reasoning_budget_flags,
    _canonicalize_spec_mode,
    _trained_drafter_sidecar,
)
from .swa_cache import _swa_cache_path
from core.inference.inference_gate import holds_inference
from core.inference.llama_server_args import _DEVICE_FLAGS
from core.inference.llama_server_args import keep_resident_lock_applicable
from core.inference.llama_server_args import keep_resident_lock_flags
from core.inference.llama_server_args import load_mode_support
from core.inference.llama_server_args import translate_ngram_mod_args
from core.inference.lookup_tables import is_per_layer_embedding as _is_per_layer_embedding
import bisect
import itertools
import math
from .batch_args import (
    _capped_draft_n_ubatch,
    _extra_args_split_mode,
    _extra_args_tensor_split,
    _pass_through_reprices_cache,
)
from .core_types import (
    _AUTO_RECOVERY_LOAD,
    _apple_budget_beside_media_mib,
    _net_of_other_models_vram,
)
from .draft_args import (
    _CPU_DEVICE_VALUES,
    _draft_backend_sampling_off,
    _draft_card_share_unknown,
    _drop_sidecar_margins,
    _extra_args_draft_device_pin,
    _fit_margins_by_device,
    _margin_mib_for,
    _sidecar_child_ctx,
    _sidecar_reserve_margin_flags,
    _spawn_draft_on_target,
    _split_mode_confines_to_one_device,
)
from .gguf_cache import (
    _gguf_set_bytes,
)
from .launch_flags import (
    _CUDA_LIKE_DEVICE_RE,
    _distinct_device_count,
    _draft_pin_device_index,
    _reasoning_preserve_from_args_or_env,
    _sidecar_borrows_output,
    _sidecar_borrows_target_tensors,
    _template_preserves_thinking_by_default,
)
from .memory_policy import (
    _APPLE_AUTO_CTX,
    _APPLE_WIRED_CEILING_FRACTION,
    _VRAM_FLOOR_RESERVE_MIB,
    _explicit_device_ordinals_needed,
)
from .moe_cache import (
    _LAZY_RESIDENT_MARGIN_FRAC,
    _effective_lazy_mode,
    _gpu_device_node_present,
    _host_pinned_input_bytes,
    _override_places_tensor_on_device,
)
from .placement_args import (
    _video_ffmpeg_dir_flags,
    _video_tools_reachable,
    _video_tools_search,
)
from .server_log import (
    parse_weight_placement,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _LoadMixin:
    @_invalidates_gpu_memory("llama.cpp load")
    @_with_gguf_load_marker
    @holds_inference
    def load_model(
        self,
        intent: GgufLoadIntent,
        load_cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        """Start llama-server from one immutable load intent."""

        def _load_cancelled() -> bool:
            return self._cancel_event.is_set() or bool(
                load_cancel_event is not None and load_cancel_event.is_set()
            )

        download_cancel_event = (
            _CombinedCancelEvent(self._cancel_event, load_cancel_event)
            if load_cancel_event is not None
            else self._cancel_event
        )

        gguf_path = intent.gguf_path
        mmproj_path = intent.mmproj_path
        mtp_draft_path = intent.mtp_draft_path
        dspark_draft_path = intent.dspark_draft_path
        dflash_draft_path = intent.dflash_draft_path
        hf_repo = intent.hf_repo
        hf_variant = intent.hf_variant
        hf_token = intent.hf_token
        model_identifier = intent.model_identifier
        is_vision = intent.is_vision
        disable_vision = intent.disable_vision
        n_ctx = intent.n_ctx
        _replayed_ctx_refit = False
        chat_template_override = intent.chat_template_override
        cache_type_kv = intent.cache_type_kv
        speculative_type = intent.speculative_type
        spec_draft_n_max = intent.spec_draft_n_max
        tensor_parallel = intent.tensor_parallel
        gpu_memory_mode = intent.gpu_memory_mode
        gpu_layers = intent.gpu_layers
        n_cpu_moe = intent.n_cpu_moe
        tensor_split = list(intent.tensor_split) if intent.tensor_split is not None else None
        gpu_ids = list(intent.gpu_ids) if intent.gpu_ids is not None else None
        gpu_ids_are_vulkan_ordinals = intent.gpu_ids_are_vulkan_ordinals
        n_threads = intent.n_threads
        n_parallel = intent.n_parallel
        n_batch = intent.n_batch
        n_ubatch = intent.n_ubatch
        load_mode = intent.load_mode
        spec_draft_cache_type = intent.spec_draft_cache_type
        ctx_checkpoints = intent.ctx_checkpoints
        cache_ram = intent.cache_ram
        extra_args = list(intent.extra_args) if intent.extra_args is not None else None
        custom_mmproj = _extra_args_device(extra_args, {"--mmproj", "-mm"})
        if custom_mmproj is not None:
            mmproj_path = self._resolve_launch_mmproj_path(
                model_path = gguf_path or "", mmproj_path = mmproj_path, extra_args = extra_args
            )
            is_vision = True
        preserve_multi_gpu_on_layer = intent.preserve_multi_gpu_on_layer
        reasoning_budget = intent.reasoning_budget
        reasoning_budget_message = intent.reasoning_budget_message
        # Refuse fast during an in-app binary swap; the flag is set under this lock so loads have drained.
        self._codec_failure.message = None
        with self._serial_load_scope():
            # Here, not at the spawn: the duplicate-adoption phase below kills whatever is loaded.
            with self._spawn_lock:
                _stale_load = self._spawn_is_stale()
            if _stale_load:
                logger.info("dropping a load left over from the previous server lifecycle")
                return False
            # In-app update swapping binaries: refuse fast (set under this lock,
            # so any in-flight load has drained) instead of using a half-swapped one.
            if getattr(self, "_llama_update_in_progress", False):
                raise RuntimeError("llama.cpp is updating; try again in a moment.")

            intent = self._preserve_cpu_fallback_intent(intent)
            tensor_parallel = intent.tensor_parallel
            gpu_memory_mode = intent.gpu_memory_mode
            gpu_layers = intent.gpu_layers
            n_cpu_moe = intent.n_cpu_moe
            tensor_split = list(intent.tensor_split) if intent.tensor_split is not None else None
            gpu_ids = list(intent.gpu_ids) if intent.gpu_ids is not None else None

            # Read before the pin below rewrites every placement to manual/0.
            _caller_owns_budget = gpu_memory_mode == "manual" and gpu_layers >= 0

            # A virtualised Apple GPU corrupts offloaded layers, so pin to CPU before the duplicate
            # check so the compared placement is the one that launches.
            _paravirtual_cpu_forced = _metal_device_is_paravirtual()
            if _paravirtual_cpu_forced:
                _pv_placement = paravirtual_normalized_request(
                    gpu_memory_mode = gpu_memory_mode,
                    gpu_layers = gpu_layers,
                    tensor_parallel = tensor_parallel,
                    tensor_split = tensor_split,
                    n_cpu_moe = n_cpu_moe,
                    extra_args = extra_args,
                )
                if (gpu_memory_mode, gpu_layers) != (
                    _pv_placement.gpu_memory_mode,
                    _pv_placement.gpu_layers,
                ):
                    logger.warning(
                        "Forcing gpu_layers=0 for %s: this Mac's Metal device is "
                        "virtualised and offloaded layers produce corrupt output.",
                        model_identifier or gguf_path,
                    )
                if extra_args and _pv_placement.extra_args != list(extra_args):
                    logger.warning(
                        "Dropping offload flags from extra args on a "
                        "virtualised Metal device: %s -> %s",
                        list(extra_args),
                        _pv_placement.extra_args,
                    )
                if tensor_parallel or tensor_split:
                    logger.info(
                        "Virtualised Metal device: dropping tensor split/parallel "
                        "flags (nothing to split on the GPU)"
                    )
                gpu_memory_mode = _pv_placement.gpu_memory_mode
                gpu_layers = _pv_placement.gpu_layers
                n_cpu_moe = _pv_placement.n_cpu_moe
                extra_args = _pv_placement.extra_args
                # Spelled out (not read off _pv_placement, which returns exactly these)
                # so tests/test_tp_vision_regression.py's TP-drop allowlist sees this site.
                tensor_parallel = False
                tensor_split = None
                # Signed shift to llama.cpp's fit margin: lowering holds more back, raising packs closer; default 0.
                intent = replace(
                    intent,
                    gpu_memory_mode = gpu_memory_mode,
                    gpu_layers = gpu_layers,
                    n_cpu_moe = n_cpu_moe,
                    extra_args = tuple(extra_args) if extra_args is not None else None,
                    tensor_parallel = False,
                    tensor_split = None,
                )

            # Resolved once under the lock so ranking, fitting and the duplicate check agree.
            _vram_frac = _active_vram_fraction()
            # How far this budget moves llama.cpp's own fitter margin, set once the
            # devices are known. Signed: lowering holds more back, raising packs
            # closer to the card. Zero at the default.
            _fit_target_delta_mib = 0.0
            # The same delta per physical GPU id, when the per-device list applies.
            _fit_target_delta_by_idx: Optional[dict[int, float]] = None
            # Studio's emitted margins per card, spelled into --fit-target at spawn.
            _fit_target_margins_by_idx: Optional[dict[int, float]] = None
            # Set only if that margin actually reached the command line, which is
            # what makes the child budget-priced in a mode the planner skipped.
            _fit_target_priced = False

            def _budget_priced_placement() -> Optional[float]:
                """Read at call time: the fraction this child was sized against; None for manual or
                CPU-fallback."""
                if intent.cpu_fallback:
                    return None
                if gpus:
                    return _vram_frac
                # Manual+Auto bypasses the planner but --fit-target still prices the child.
                return _vram_frac if _fit_target_priced else None

            # Armed before the duplicate check so a racing save sees the pending value.
            self._vram_fraction_pending = _vram_frac
            # _vram_fraction_launched is committed at the launch site, not here.
            if self.adopt_load_intent_if_matched(intent):
                logger.info(
                    f"load_model: backend already in target state for "
                    f"'{model_identifier}', skipping reload"
                )
                # Retry probe only if a prior attempt didn't finish.
                if not self._audio_probed:
                    try:
                        detected = self._detect_audio_type_strict()
                        self._audio_probed = True
                    except Exception as exc:
                        logger.debug("Fast-path audio probe failed: %s", exc)
                        detected = None
                    applied_audio = (
                        self._apply_detected_audio(detected)
                        if intent.audio_codec_path is None
                        else self._apply_detected_audio(detected, intent.audio_codec_path)
                    )
                    if not applied_audio:
                        return False
                if not self._healthy:
                    return False
                return True

            # Stays armed through the download and planning, which can take minutes.
            # Placement is undecided until the spawn site narrows it: a needless
            # reload prompt beats sizing a child against a replaced budget.
            self._cancel_event.clear()
            # Belongs to this attempt only; a guard that runs before this load's first
            # health wait would otherwise read the previous request's cancellation.
            self._health_wait_cancelled = False
            if _load_cancelled():
                logger.info("Load cancelled before GGUF resolution")
                return False

            # Defer a not-found error: diffusion GGUFs use another runner.
            # Snapshot under the settings guard so a racing save asks for a reload.
            from utils.llama_cpp_path_settings import llama_cpp_path_selection_guard

            with llama_cpp_path_selection_guard():
                binary = self._find_llama_server_binary()
                # macOS: launch the resolved server, never a shell wrapper; SIP purges DYLD_* through /bin/sh.
                binary = self._exec_path_for_launch(binary)
                self._binary_revision_pending = self._binary_revision(binary)

            # Accumulated, not sampled: a later probe success must not erase an earlier guess.
            _launch_probe_inconclusive = False
            # Set when FA ends up off without the argv saying so.
            _flash_attn_known_off = False

            def _launch_caps(bin_path):
                nonlocal _launch_probe_inconclusive
                caps = self.probe_server_capabilities(bin_path)
                if caps.get("mtp_probe_inconclusive"):
                    _launch_probe_inconclusive = True
                return caps

            _reasoning_requested = any(
                self.reasoning_budget_settings_requested(
                    extra_args = extra_args,
                    reasoning_budget = reasoning_budget,
                    reasoning_budget_message = reasoning_budget_message,
                )
            )
            if (
                _reasoning_requested
                and gguf_path
                and Path(gguf_path).is_file()
                and self._gguf_path_is_diffusion(gguf_path, model_identifier)
            ):
                raise ValueError(
                    "Reasoning Budget settings are not supported for DiffusionGemma models."
                )
            # Nothing configured means nothing to reject; _launch_caps probes the binary anyway.
            if _reasoning_requested:
                self.validate_reasoning_budget_capabilities(
                    binary,
                    extra_args = extra_args,
                    reasoning_budget = reasoning_budget,
                    reasoning_budget_message = reasoning_budget_message,
                )
            is_vulkan_backend = self._is_vulkan_backend(binary)
            # Computed once so the reload comparator matches the argv.
            _gpu_ids_own_device_flags = self._gpu_ids_own_placement(gpu_ids)
            _vulkan_ordinal_pin = (
                is_vulkan_backend and bool(gpu_ids) and gpu_ids_are_vulkan_ordinals is not False
            )
            _cpu_only_pin = (
                bool(gpu_ids) and not is_vulkan_backend and self._backend_lacks_gpu_lib(binary)
            )

            # Without --kv-unified, --parallel N splits -c into N windows; fall back to one slot.
            if n_parallel > 1 and binary and not _launch_caps(binary).get("supports_kv_unified"):
                logger.warning(
                    "llama-server at %s has no --kv-unified, so %d parallel slots would "
                    "split the context window %d ways. Using 1 slot instead; update "
                    "llama.cpp to run chats in parallel.",
                    binary,
                    n_parallel,
                    n_parallel,
                )
                n_parallel = 1  # allow-slot-clamp: the build cannot serve more

            # ── Vulkan-ordinal preflight (BEFORE the Phase 1 kill) ────────
            # Vulkan-ordinal preflight, above the kill so an invalid pin leaves the live model untouched.
            # Vulkan ordinals are not range-checked at the route, unlike CUDA ids.
            if _vulkan_ordinal_pin and binary:
                _pf_wanted = {int(x) for x in gpu_ids}
                _pf_probed = {g[0] for g in self._get_gpu_memory(binary)}
                if not _pf_wanted.issubset(_pf_probed):
                    raise ValueError(
                        f"Requested Vulkan GPU ordinal(s) {sorted(_pf_wanted)} not "
                        f"present. Available Vulkan devices: {sorted(_pf_probed)}."
                    )

            # Classify before killing the healthy server; diffusion changes placement validity.
            _pv_diffusion_unpinnable = bool(
                _paravirtual_cpu_forced and not self.diffusion_split_supported()
            )
            if hf_repo:
                # Resolve repo casing once so the GGUF and companions share one cache entry.
                _resolved_repo = _resolve_repo_id_casing(hf_repo)
                if _resolved_repo != hf_repo:
                    logger.info(
                        "Using cached repo_id casing '%s' for requested '%s'",
                        _resolved_repo,
                        hf_repo,
                    )
                    hf_repo = _resolved_repo
            _preflight_model_path = None
            self._pending_variant_fallback = None
            if hf_repo and (_vulkan_ordinal_pin or _cpu_only_pin or _pv_diffusion_unpinnable):
                with _hf_offline_if_unreachable():
                    _preflight_model_path = self._download_gguf(
                        hf_repo = hf_repo,
                        hf_variant = hf_variant,
                        hf_token = hf_token,
                        cancel_event = download_cancel_event,
                    )
                _preflight_is_diffusion = self._gguf_path_is_diffusion(
                    _preflight_model_path, model_identifier
                )
                if _preflight_is_diffusion:
                    if _vulkan_ordinal_pin:
                        raise ValueError(_VULKAN_DIFFUSION_GPU_IDS_ERROR)
                    if _pv_diffusion_unpinnable:
                        raise ValueError(_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR)
                elif _cpu_only_pin:
                    raise ValueError(
                        f"Requested gpu_ids {list(gpu_ids)} but the llama.cpp build has "
                        "no GPU backend (CPU-only build); it would ignore the pin and run "
                        "on CPU. Omit gpu_ids to run on CPU."
                    )
            elif is_vulkan_backend and gpu_ids and gguf_path and not hf_repo:
                if not Path(gguf_path).is_file():
                    raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

            # Settle an unpinnable diffusion split here before teardown; _start_diffusion_server re-checks it.
            if _vulkan_ordinal_pin and gguf_path and not hf_repo:
                self._reject_vulkan_diffusion_gpu_ids_before_teardown(
                    gguf_path,
                    model_identifier,
                )
            # Same for the unpinnable diffusion split: _start_diffusion_server enforces
            # it everywhere, but only after the teardown, so settle the local case here
            # and leave the running server alone.
            if (
                _pv_diffusion_unpinnable
                and gguf_path
                and not hf_repo
                and self._gguf_path_is_diffusion(gguf_path, model_identifier)
            ):
                raise ValueError(_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR)

            # Refuse an on-disk image/video GGUF before the teardown (header answers it).
            if gguf_path and not hf_repo and Path(gguf_path).is_file():
                _early_non_chat = self._non_chat_gguf_refusal_for_path(gguf_path, model_identifier)
                if _early_non_chat:
                    logger.error(
                        "Refusing non-chat GGUF before teardown: %s (%s)",
                        _early_non_chat,
                        gguf_path,
                    )
                    raise ValueError(_early_non_chat)

            # Same refusal for repo loads (what the Model Hub sends), via a header byte range.
            # Fails open into the post-download check.
            if hf_repo:
                # Offline guard: the probe hits the Hub twice, each with retry backoff.
                with _hf_offline_if_unreachable():
                    _early_non_chat = (
                        self._non_chat_gguf_refusal_for_path(
                            _preflight_model_path, model_identifier
                        )
                        if _preflight_model_path
                        else self._remote_non_chat_gguf_refusal(
                            hf_repo = hf_repo,
                            hf_variant = hf_variant,
                            hf_token = hf_token,
                            model_identifier = model_identifier,
                        )
                    )
                if _early_non_chat:
                    logger.error(
                        "Refusing non-chat GGUF before teardown: %s (%s)",
                        _early_non_chat,
                        hf_repo,
                    )
                    raise ValueError(_early_non_chat)

            # Resolve the canonical MTP mode once; it decides which drafter is downloaded and launched.
            _spec_canon = _canonicalize_spec_mode(speculative_type) or "auto"
            # #11308: DFlash2 + --split-mode tensor aborts at startup (ggml-org/llama.cpp#27819) and the
            # route falls back to layer split, so Auto prefers a loadable MTP sidecar on tensor instead.
            # _estimate_gguf_required_gb mirrors this.
            _auto_tensor_split = _spec_canon == "auto" and _effective_tensor_parallel(
                extra_args, tensor_parallel
            )

            def _auto_dflash_blocked_by_tensor() -> bool:
                if (
                    _auto_tensor_split
                    and mtp_draft_path
                    and not _extra_args_mtp_draft_path(extra_args, env = _child_spec_env(extra_args))
                ):
                    try:
                        from utils.models.gguf_metadata import read_gguf_nextn_predict_layers
                        return bool(
                            _launch_caps(binary).get("mtp_token")
                            and not (read_gguf_nextn_predict_layers(model_path) or 0) > 0
                            and _mtp_drafter_loads_standalone(mtp_draft_path)
                        )
                    except Exception:
                        return False
                return False

            def _dflash_breaks_on_images() -> bool:
                if not dflash_draft_path or disable_vision or extra_args_disable_mmproj(extra_args):
                    return False
                try:
                    projector = self._resolve_launch_mmproj_path(
                        model_path = model_path, mmproj_path = mmproj_path, extra_args = extra_args
                    )
                except Exception:
                    return False
                return _dflash_drafter_breaks_on_images(model_path, dflash_draft_path, projector)

            _unloadable_mtp_draft_path: Optional[str] = None
            # Downloads run before the teardown so a failed or cancelled fetch keeps the resident model;
            # sidecar fetches write flags on self, so the resident's come back until the teardown commits.
            _resident_sidecar_flags = (
                self._dspark_sidecar_absent,
                self._dflash_sidecar_absent,
                self._dflash_retry_needed,
            )
            self._dflash_retry_needed = False
            model_path = _preflight_model_path
            try:
                if hf_repo:
                    # Scope HF_HUB_OFFLINE to the download block only when DNS is dead.
                    with _hf_offline_if_unreachable():
                        if model_path is None:
                            # Revalidate the carried file before skipping download resolution.
                            model_path = self._verified_cached_gguf(intent, hf_repo, hf_variant)
                            if model_path:
                                logger.info("Reusing verified cached GGUF: %s", model_path)
                        if model_path is None:
                            model_path = self._download_gguf(
                                hf_repo = hf_repo,
                                hf_variant = hf_variant,
                                hf_token = hf_token,
                                cancel_event = download_cancel_event,
                            )
                        # Vision off does not skip the fetch: an mmproj may be an audio encoder instead.
                        if (
                            is_vision
                            and not mmproj_path
                            and not extra_args_disable_mmproj(extra_args)
                        ):
                            mmproj_path = self._download_mmproj(
                                hf_repo = hf_repo,
                                hf_token = hf_token,
                                cancel_event = download_cancel_event,
                                near_path = model_path,
                            )
                        # Fetch the separate MTP drafter; no size gate (helps even sub-3B).
                        # The MTP/ fallback is qwen4exp only, whose GGUFs carry no head.
                        if (
                            not mtp_draft_path
                            and _spec_canon in ("auto", "mtp", "mtp+ngram")
                            and not _extra_args_set_spec_type(extra_args)
                        ):
                            mtp_draft_path = self._download_mtp(
                                hf_repo = hf_repo,
                                hf_token = hf_token,
                                cancel_event = download_cancel_event,
                                near_path = model_path,
                                allow_nested = self._gguf_path_wants_nested_mtp(model_path),
                            )
                        # "auto" is included: DSpark is the default whenever the repo
                        # ships a sidecar. Repos without one no-op, exactly like the
                        # MTP fetch above, so this costs one listing and no download.
                        if (
                            not dspark_draft_path
                            and _spec_canon in ("auto", "dspark")
                            and not _extra_args_set_spec_type(extra_args)
                        ):
                            dspark_draft_path = self._download_dspark(
                                hf_repo = hf_repo,
                                hf_token = hf_token,
                                cancel_event = download_cancel_event,
                                near_path = model_path,
                                binary = binary,
                                caps_probe = _launch_caps,
                            )
                        # Auto fetch stands down once DSpark resolves (it wins the promotion), using the same
                        # capability-aware 'resolved'. An explicit 'dflash' still fetches.
                        if (
                            not dflash_draft_path
                            and _spec_canon in ("auto", "dflash")
                            and not _auto_dflash_blocked_by_tensor()
                            and not self._dspark_wins_auto(
                                binary = binary,
                                dspark_draft_path = dspark_draft_path,
                                spec_canon = _spec_canon,
                                extra_args = extra_args,
                            )
                            and not _extra_args_set_spec_type(extra_args)
                        ):
                            dflash_draft_path = self._download_dflash(
                                hf_repo = hf_repo,
                                hf_token = hf_token,
                                cancel_event = download_cancel_event,
                                near_path = model_path,
                                binary = binary,
                            )
                elif gguf_path:
                    if not Path(gguf_path).is_file():
                        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
                    model_path = gguf_path
                else:
                    raise ValueError("Either gguf_path or hf_repo must be provided")
            finally:
                _load_sidecar_flags = (
                    self._dspark_sidecar_absent,
                    self._dflash_sidecar_absent,
                    self._dflash_retry_needed,
                )
                (
                    self._dspark_sidecar_absent,
                    self._dflash_sidecar_absent,
                    self._dflash_retry_needed,
                ) = _resident_sidecar_flags

            # The probe can take a moment on a slow Hub, so re-read the cancel flag rather
            # than tearing down a healthy server for a load that was called off.
            if _load_cancelled():
                logger.info("Load cancelled before teardown")
                return False

            # Fail fast before killing the live server; an explicit reload retries.
            _abort_memo_model = repr(
                (
                    replace(intent, hf_token = None, force_reload = False, verified_gguf = None),
                    _vram_frac,
                    _model_memory_settings_or_none(),
                )
            )
            if intent.force_reload:
                LlamaCppBackend._forget_sched_reserve_abort(binary, _abort_memo_model)
            elif LlamaCppBackend._sched_reserve_aborts(binary, _abort_memo_model):
                logger.warning(
                    "Skipping reload of '%s': it already aborted in the llama.cpp graph "
                    "scheduler this session.",
                    model_identifier,
                )
                raise RuntimeError(self._sched_reserve_abort_message())

            # ── Phase 1: kill old process (under lock, fast) ──────────
            # Drop the previous advisory here, where the old child is committed to dying, so every
            # stand-down above keeps it.
            _replaying_cpu_fallback = intent.cpu_fallback
            with self._lock:
                self._begin_load_warnings()
                self._kill_process()
                if not _replaying_cpu_fallback:
                    self._cpu_fallback_reason = None
                    self._cleanup_cpu_fallback_runtime()

                self._mmproj_fallback_reason = None
                self._variant_fallback_warning = None
            # Reset at the teardown, not where the binary resolves: early bails keep the old server.
            self._launch_binary_revision = self._binary_revision(binary)
            (
                self._dspark_sidecar_absent,
                self._dflash_sidecar_absent,
                self._dflash_retry_needed,
            ) = _load_sidecar_flags
            if hf_repo and self._pending_variant_fallback:
                hf_variant, self._variant_fallback_warning = self._pending_variant_fallback
                # Replayed by the respawn and a variant-less Apply, which would re-download.
                intent = replace(intent, hf_variant = hf_variant)

            # Set identifier early so _read_gguf_metadata can use it (DeepSeek).
            self._model_identifier = model_identifier

            # Capability gate shared with the DFlash fetch so fetch and promotion agree.
            if self._dspark_wins_auto(
                binary = binary,
                dspark_draft_path = dspark_draft_path,
                spec_canon = _spec_canon,
                extra_args = extra_args,
                caps = _launch_caps,
            ):
                _spec_canon = "dspark"
                logger.info("Auto: DSpark sidecar available, using draft-dspark.")

            # DSpark keeps first refusal, matching llama.cpp's own downloader.
            if (
                _spec_canon == "auto"
                and dflash_draft_path
                and not _extra_args_set_spec_type(extra_args)
            ):
                try:
                    if not _launch_caps(binary).get("supports_dflash"):
                        pass
                    elif _auto_dflash_blocked_by_tensor():
                        logger.info(
                            "Auto: DFlash sidecar available but tensor split is on; "
                            "keeping tensor split with the MTP drafter "
                            "(llama.cpp cannot run DFlash2 with --split-mode tensor yet)."
                        )
                    elif _dflash_breaks_on_images():
                        logger.info(
                            "Auto: DFlash sidecar %s has no M-RoPE sections, so llama.cpp "
                            "cannot carry it past an image on this vision model; keeping "
                            "vision and skipping DFlash. Choose DFlash in the Speculative "
                            "Decoding dropdown to force it, or turn Vision off.",
                            dflash_draft_path,
                        )
                    else:
                        _spec_canon = "dflash"
                        logger.info("Auto: DFlash sidecar available, using draft-dflash.")
                except Exception as exc:
                    logger.debug("DFlash capability probe failed during Auto: %s", exc)
            elif (
                _spec_canon == "dflash"
                and not _extra_args_set_spec_type(extra_args)
                and _dflash_breaks_on_images()
            ):
                logger.warning(
                    "DFlash forced with sidecar %s, which has no M-RoPE sections: llama.cpp "
                    "will fail the first image sent after text in a chat. Use Auto or turn "
                    "Vision off to keep images working.",
                    dflash_draft_path,
                )

            # From here mtp_draft_path is whichever drafter launches; _spec_canon says which.
            if _spec_canon == "dspark":
                mtp_draft_path = dspark_draft_path
            elif _spec_canon == "dflash":
                mtp_draft_path = dflash_draft_path

            # Read GGUF metadata (context_length, chat_template); header-only.
            self._read_gguf_metadata(model_path)

            if (
                self._nextn_predict_layers
                and mtp_draft_path
                and _spec_canon not in ("dspark", "dflash")
            ):
                # A root mtp-*.gguf may mirror the embedded head; -md would replace it.
                logger.info("Main GGUF contains an embedded MTP head; ignoring separate drafter.")
                mtp_draft_path = None

            # Drop before the fit prices it, but not when extras name their own drafter (last-wins).
            if (
                mtp_draft_path
                and _spec_canon not in ("dspark", "dflash")
                and not _extra_args_mtp_draft_path(extra_args, env = _child_spec_env(extra_args))
                and not _mtp_drafter_loads_standalone(mtp_draft_path)
            ):
                logger.warning(
                    "Dropping MTP drafter %s: it carries neither token_embd.weight nor "
                    "nextn_shared_target_tensors, so llama-server cannot load it as "
                    "--model-draft; loading without it.",
                    mtp_draft_path,
                )
                _unloadable_mtp_draft_path = mtp_draft_path
                mtp_draft_path = None

            if _load_cancelled():
                logger.info("Load cancelled after download phase")
                return False

            # Whether the user set the batch pair, read before the embedding and projector
            # raises below overwrite n_batch / n_ubatch. Gates the expert-spill raise.
            _user_named_batch = any(_named_batch_sizes(extra_args, None, n_batch, n_ubatch)[2:])

            # MEAN/CLS inputs must fit one micro-batch (LAST splits). Before the projector raise,
            # which would otherwise read as a user-set micro-batch.
            if self._pooling_type in (1, 2):
                n_batch, n_ubatch = _embedding_batch_ubatch(
                    resolve_requested_ctx(extra_args, n_ctx) or self._context_length or 0,
                    n_batch,
                    n_ubatch,
                    extra_args,
                )

            # Decide after downloading the projector and before pricing the load.
            n_batch, n_ubatch = _batch_ubatch_for_mmproj(
                _launch_required_ubatch(
                    None
                    if (disable_vision or not is_vision)
                    else self._resolve_launch_mmproj_path(
                        model_path = model_path,
                        mmproj_path = mmproj_path,
                        extra_args = extra_args,
                    ),
                    # Use the same order-independent metadata read as the estimators.
                    _read_gguf_embedding_length(model_path),
                    extra_args,
                    is_vision = is_vision,
                    vision_off = disable_vision,
                ),
                n_batch,
                n_ubatch,
                extra_args,
            )

            # Backstop for what the pre-teardown probes fail open on.
            non_chat = self._non_chat_gguf_refusal(model_path)
            if non_chat:
                logger.error("Refusing non-chat GGUF: %s (%s)", non_chat, model_path)
                raise ValueError(non_chat)

            # Block-diffusion GGUFs (DiffusionGemma) cannot run on llama-server;
            # serve them with the diffusion runner (same OpenAI-compat interface).
            if self._is_diffusion:
                # The diffusion runner cannot translate Vulkan ordinals to CUDA IDs.
                if is_vulkan_backend and gpu_ids and gpu_ids_are_vulkan_ordinals is not False:
                    raise ValueError(_VULKAN_DIFFUSION_GPU_IDS_ERROR)
                # This path skips the command builder that clears these.
                self._layer_preserves_tensor_intent = False
                # This path must reset the drafter verdict like _build_speculative_flags, or a stale one relaunches.
                self._mtp_draft_path = None
                self._mtp_draft_suppressed_path = None
                self._mtp_draft_suppressed_reason = None
                # And the verdict on why the last load had no drafter, for the same
                # reason once more: _build_speculative_flags clears it at the top of
                # every load that reaches it, and this path never does. Both retry
                # rules in the dedupe read it -- the drafter one and the binary one --
                # and a stale answer would relaunch this server on every Apply for a
                # drafter it was never going to carry.
                self._spec_fallback_reason = None
                self._spec_drafter_kind = None
                # Discovery runs before the metadata read that says diffusion, so a
                # transient sidecar failure can set this for a drafterless server.
                self._dflash_retry_needed = False
                # Same reason: this path returns before the launch block, and the
                # diffusion runner ignores the budget, so a previous llama-server's
                # fraction would relaunch a healthy diffusion server on every Apply.
                self._vram_fraction_launched = None
                # Outside the lock, as for llama-server: the Auto layer fit reads free VRAM,
                # which a just-killed runner still holds until the driver reclaims it.
                self._wait_for_vram_settle(since_kill = self._last_kill_monotonic)
                with self._lock:
                    if _load_cancelled():
                        logger.info("Load cancelled before diffusion server start")
                        return False
                    return self._start_diffusion_server(
                        model_path = model_path,
                        gguf_path = gguf_path,
                        hf_repo = hf_repo,
                        hf_variant = hf_variant,
                        model_identifier = model_identifier,
                        n_ctx = n_ctx,
                        extra_args = extra_args,
                        gpu_ids = gpu_ids,
                        gpu_memory_mode = gpu_memory_mode,
                        gpu_layers = gpu_layers,
                        cancelled = _load_cancelled,
                    )

            if not binary:
                # Locked binary (antivirus / in-flight install) means retry, not reinstall.
                locked = self._find_llama_server_binary(include_denied = True)
                if locked:
                    raise RuntimeError(
                        f"llama-server at {locked} is temporarily unavailable "
                        "(access-denied; antivirus or an in-flight install). "
                        "Retry the load once it is released."
                    )
                # Typed error so /load returns an actionable 400, not a 500.
                raise LlamaServerNotFoundError(LLAMA_SERVER_NOT_FOUND_DETAIL)

            server_caps = _launch_caps(binary)
            self._reserves_micro_batch_outputs = self.reserves_micro_batch_outputs(binary)
            self._resumes_thoughts = self.resumes_thoughts(binary)

            # Outside self._lock so /unload, /cancel, /status aren't blocked.
            self._wait_for_vram_settle(since_kill = self._last_kill_monotonic)

            # ── Phase 3: start llama-server (under lock) ──────────────
            with self._lock:
                # Re-check cancel inside lock
                if _load_cancelled():
                    logger.info("Load cancelled before server start")
                    return False

                self._port = self._find_free_port()

                # Seed safe defaults before probing so the except path has valid state.
                ctx_override = parse_ctx_override(extra_args)
                requested_ctx = resolve_requested_ctx(extra_args, n_ctx)
                swa_full = _swa_full_from_args_or_env(extra_args)

                # Size from the emitted batch (floor max(slots, 2)), not the requested one.
                def _ubatch_for_slots(slots: int) -> Optional[int]:
                    return _extra_args_n_ubatch(
                        extra_args,
                        n_ctx = (requested_ctx if requested_ctx > 0 else self._context_length),
                        n_batch = _emitted_n_batch(n_batch, slots),
                        n_ubatch = n_ubatch,
                    )

                _effective_ubatch = _ubatch_for_slots(n_parallel)

                # The budget is per slot and the fit can still cut n_parallel, so both
                # counts are taken after it settles, as _effective_ubatch is below.
                def _ctx_checkpoints_for_final_slots() -> int:
                    return effective_ctx_checkpoints_for_caps(
                        server_caps,
                        extra_args,
                        ctx_checkpoints,
                        per_checkpoint_bytes = self._ctx_checkpoint_bytes(
                            cache_type_kv, swa_full = swa_full, flash_attn = planned_flash_attn
                        ),
                        n_parallel = n_parallel,
                        total_host_bytes = ((self._host_memory_capacity_mib() or 0) * 1024 * 1024)
                        or None,
                    )

                _requested_ctx_checkpoints = (
                    resolve_ctx_checkpoints(extra_args, ctx_checkpoints)
                    if server_caps.get("ctx_checkpoints_flag")
                    else 0
                )
                # Snapshots are host RAM (copied device-to-host): only a host-RAM pool pays.
                _fit_ctx_checkpoints = (
                    0 if self._rollback_state_bytes(1) > 0 else _requested_ctx_checkpoints
                )
                # --embedding forces n_batch = n_ubatch, which aborts a load below the slots.
                if (
                    self.is_embedding_gguf
                    and _effective_ubatch is not None
                    and _effective_ubatch < n_parallel
                ):
                    # max(): '-b 0' resolves to 0, and --parallel 0 is rejected.
                    _embedding_slots = max(1, _effective_ubatch)
                    logger.warning(
                        "Reducing serving slots from %d to %d: llama-server caps the batch "
                        "at the %d-token micro-batch under --embedding and aborts when that "
                        "is below the slot count.",
                        n_parallel,
                        _embedding_slots,
                        _effective_ubatch,
                    )
                    n_parallel = _embedding_slots  # allow-slot-clamp: llama-server would abort
                    _effective_ubatch = _ubatch_for_slots(n_parallel)
                planned_kv_unified = _kv_unified_from_args(
                    extra_args,
                    default = n_parallel > 1 and server_caps.get("supports_kv_unified", False),
                )
                _pipeline_parallel_off = _pipeline_parallel_disabled_by_args(
                    extra_args, n_layers = self._n_layers
                )
                cache_override = parse_cache_override(extra_args)
                # Budget the heavier of asymmetric K/V extras; this only affects the reserve.
                _extras_cache = _extra_args_main_cache_type_for_budget(extra_args)
                cache_type_kv = _extras_cache if _extras_cache is not None else cache_type_kv
                _cache_type_from_env = False
                # An unknown type is never emitted, so treat it as unset and adopt the env type.
                if cache_type_kv is not None and (
                    cache_type_kv.strip().lower() not in _VALID_CACHE_TYPES
                ):
                    cache_type_kv = None
                if cache_type_kv is None:
                    # Adopt an inherited env type for the reserve only; not re-emitted.
                    cache_type_kv = _env_main_cache_type_for_budget()
                    _cache_type_from_env = cache_type_kv is not None
                # Dequant scratch keys off the lighter axis, not the heavier one.
                _scratch_cache_type_kv = _planned_scratch_cache_type(
                    None if _cache_type_from_env else cache_type_kv,
                    extra_args,
                )
                # The (K, V) pair this launch will run, for the tensor-abort latch:
                # since #23792 the type reaches the child, so an abort can be specific
                # to it and the latch must not generalise across types.
                _planned_cache_pair = _planned_main_cache_types(
                    None if _cache_type_from_env else cache_type_kv,
                    extra_args,
                )
                # Auto leaves this negative, which is llama.cpp's default and no override.
                _manual_gpu_layers = gpu_layers if gpu_memory_mode == "manual" else None
                # After the cache pair: a quantized V forces flash attention on.
                planned_flash_attn = _reserved_flash_attn_state(
                    _planned_flash_attn_state(
                        extra_args,
                        planned_cache_types = _planned_cache_pair,
                        supports_flash_attn = bool(server_caps.get("supports_flash_attn", True)),
                        tensor_parallel = tensor_parallel,
                        architecture = self._architecture,
                    ),
                    extra_args,
                    tensor_parallel = tensor_parallel,
                    gpu_layers = _manual_gpu_layers,
                )

                def _replan_env(_current_tp: bool) -> "Optional[Mapping[str, str]]":
                    """The environment the CHILD will see: the launch clears a non-layer
                    inherited LLAMA_ARG_SPLIT_MODE, which otherwise turns tensor mode back
                    on inside the helper."""
                    if _current_tp:
                        return None
                    inherited = (os.environ.get("LLAMA_ARG_SPLIT_MODE") or "").strip().lower()
                    if not inherited or inherited == "layer":
                        return None
                    return {
                        key: value
                        for key, value in os.environ.items()
                        if key not in ("LLAMA_ARG_SPLIT_MODE", "LLAMA_ARG_TENSOR_SPLIT")
                    }

                def _replanned_flash_attn(_current_tp: bool) -> bool:
                    """Re-resolve the planned attention after a tensor-mode downgrade: AUTO
                    plans it ON under tensor, so a stale plan budgets a V layer never gets."""
                    _env = _replan_env(_current_tp)
                    return _reserved_flash_attn_state(
                        _planned_flash_attn_state(
                            extra_args,
                            planned_cache_types = _planned_cache_pair,
                            supports_flash_attn = bool(server_caps.get("supports_flash_attn", True)),
                            tensor_parallel = _current_tp,
                            architecture = self._architecture,
                            env = _env,
                        ),
                        extra_args,
                        tensor_parallel = _current_tp,
                        gpu_layers = _manual_gpu_layers,
                        env = _env,
                    )

                # Extras --split-mode wins and inherited tensor env flips it on.
                split_mode_override = parse_split_mode_override(extra_args)
                tensor_parallel = _effective_tensor_parallel(extra_args, tensor_parallel)
                # Nothing to split; tensor mode under the CPU-only mask would abort.
                if gpu_memory_mode == "manual" and gpu_layers == 0:
                    if tensor_parallel or tensor_split:
                        logger.info(
                            "Manual gpu_layers=0: dropping tensor split/parallel "
                            "flags (nothing to split on the GPU)"
                        )
                    tensor_parallel = False
                    tensor_split = None
                    # The plan was made under tensor, the one state forcing AUTO on. Same
                    # for every later downgrade.
                    planned_flash_attn = _replanned_flash_attn(tensor_parallel)
                # Auto ratio is set only on the auto tensor branch; reset so a manual load cannot carry it forward.
                self._gpu_memory_mode = gpu_memory_mode
                # Reset before the branch so an earlier launch's recorded drop cannot excuse this one.
                self._arch_gate_dropped_tensor_split = None
                self._arch_gate_dropped_tensor_parallel = False
                # Same reason, same place: the auto ratio is recorded only on the
                # auto tensor branch, so without this a manual load would carry a
                # previous auto load's ratio forward into the next comparison.
                self._auto_tensor_split = None
                self._auto_tensor_split_emitted = None
                if gpu_memory_mode == "manual" and gpu_layers >= 0:
                    self._gpu_layers = gpu_layers
                    self._n_cpu_moe = n_cpu_moe
                    self._tensor_split = tensor_split
                else:
                    self._gpu_layers = -1
                    self._n_cpu_moe = 0
                    self._tensor_split = None
                    self._auto_tensor_split = None
                    self._auto_tensor_split_emitted = None
                self._mixed_split_flags = []
                self._requested_gpu_ids = [int(i) for i in gpu_ids] if gpu_ids else None
                self._gpu_ids = list(self._requested_gpu_ids) if self._requested_gpu_ids else None
                # Tensor split on <2 GPUs is a no-op and aborts on some architectures.
                if (
                    tensor_parallel
                    and gpu_memory_mode == "manual"
                    and gpu_layers >= 0
                    and self._effective_gpu_count(sorted(gpu_ids) if gpu_ids else None) < 2
                ):
                    logger.info(
                        "Tensor parallelism requested in manual mode but fewer "
                        "than 2 GPUs are in use; ignoring (needs >= 2)."
                    )
                    tensor_parallel = False
                    planned_flash_attn = _replanned_flash_attn(tensor_parallel)
                if tensor_parallel and gpu_memory_mode == "manual" and gpu_layers < 0:
                    logger.info(
                        "Manual mode with Auto layers hands memory management to "
                        "llama.cpp --fit, which is incompatible with tensor "
                        "parallelism; ignoring the tensor split."
                    )
                    tensor_parallel = False
                    planned_flash_attn = _replanned_flash_attn(tensor_parallel)
                if ctx_override is not None and ctx_override > 0:
                    logger.info(f"User --ctx-size {ctx_override} honored; skipping auto-reduce")
                if cache_override is not None:
                    _ck, _cv = parse_cache_override_per_axis(extra_args)
                    logger.info(
                        f"User --cache-type-k/-v (k={_ck}, v={_cv}) honored; "
                        "KV estimate budgets the heavier axis"
                    )
                if split_mode_override is not None:
                    logger.info(
                        f"User --split-mode {split_mode_override} honored; "
                        "reconciled into tensor_parallel state"
                    )
                effective_ctx = requested_ctx if requested_ctx > 0 else (self._context_length or 0)
                max_available_ctx = self._context_length or effective_ctx
                gpus: list[tuple[int, int]] = []
                # Keep fit-budget and launch-flag mmproj resolution in sync.
                launch_mmproj_path = None
                # Only where the switch dropped a usable projector; None also means a missing file or family mismatch, so do not tell the user to re-enable Vision.
                _dv_dropped_image_projector = False
                if not extra_args_disable_mmproj(extra_args):
                    launch_mmproj_path = self._resolve_launch_mmproj_path(
                        model_path = model_path,
                        mmproj_path = mmproj_path,
                        extra_args = extra_args,
                    )
                    # Audio-only projector (ultravox, Voxtral, Qwen3-ASR) stays when Vision is off: suppressing it drops audio input and frees no image VRAM.
                    if disable_vision and launch_mmproj_path:
                        _dv_has_audio, _dv_accepts_image = False, True
                        try:
                            from utils.models.gguf_metadata import mmproj_capabilities
                            _dv_has_audio, _dv_accepts_image = mmproj_capabilities(
                                launch_mmproj_path
                            )
                        except Exception as e:
                            # Unknown reads image-capable upstream, so honor the switch
                            # rather than quietly keep a projector the user turned off.
                            logger.debug(f"mmproj capability read failed: {e}")
                        if _dv_accepts_image:
                            if _dv_has_audio:
                                # One projector serves both modalities (Qwen2.5-Omni) and llama.cpp cannot load half of it, so audio goes with vision.
                                logger.warning(
                                    "Vision is off for this load, so this model's audio "
                                    "input goes too: its projector serves both and "
                                    "llama.cpp cannot load one modality without the "
                                    "other. Turn Vision back on to send audio."
                                )
                            launch_mmproj_path = None
                            _dv_dropped_image_projector = True
                        else:
                            logger.info(
                                "Vision is off for this load, but this model's projector "
                                "is audio-only (no vision tower), so it stays loaded: "
                                "dropping it would remove audio input and free no image "
                                "VRAM."
                            )
                # Virtualised Metal corrupts the encoder: drop the projector unless --no-mmproj-offload exists, else images encode from garbage; only if the probe answered.
                _pv_mmproj_unpinnable = bool(
                    launch_mmproj_path
                    and _paravirtual_cpu_forced
                    and _paravirtual_probe_answered(server_caps)
                    and not server_caps.get("supports_no_mmproj_offload")
                )
                if _pv_mmproj_unpinnable:
                    logger.warning(
                        "Disabling image/audio input for this session: this Mac's "
                        "Metal device is virtualised and this llama-server build has "
                        "no --no-mmproj-offload, so the vision projector would keep "
                        "running on the device whose output is corrupt. Run 'unsloth "
                        "studio update' to install a build that supports it."
                    )
                    launch_mmproj_path = None
                # A stray mmproj must not flip a non-VLM GGUF into vision mode.
                effective_is_vision = bool(launch_mmproj_path) and bool(is_vision)
                if is_vision and not effective_is_vision and not _pv_mmproj_unpinnable:
                    logger.warning(
                        "Vision-capable GGUF loaded without a usable mmproj; "
                        "image input will be disabled for this session"
                    )
                # Seed before the try (except falls through to the launch; the probe may throw first). Capture before manual empties gpus so defaults stay GPU-aware.
                _detected_gpus: list[tuple[int, int]] = []
                _shared_gpu_ids: set[int] = set()
                _vulkan_igpu_ids: frozenset[int] = frozenset()
                _placement_ctx_checkpoints = 0
                # Set when the arch gate emptied a non-empty GPU pool, so the env
                # block below masks the child onto the CPU. Bound before the try for
                # the same reason as _detected_gpus: the except path (--fit on) falls
                # through to the launch, which reads it.
                _arch_gate_forced_cpu = False
                # Not derived from env: an inherited mask would be subtracted twice.
                _launch_pinned_ids: Optional[list] = None
                # The pageable rewrite a CPU replay applied, kept across attempts: the
                # env is shared, so only the first replay to need it produces a note.
                _replay_pageable_override: Optional[str] = None
                # Set by the restored cpu_fallback path, consumed once the launch is
                # healthy: str carries that replay's override note, True means none.
                _restored_cpu_reprice: Union[str, bool, None] = None
                # Whether the planner ran and could not fit; use_fit alone cannot tell.
                _placement_verdict_partial = False
                _ctx_capped_for_vram = False
                # Probed GPUs reporting total 0: a pool shared with host RAM (APU, iGPU, SoC).
                _shared_pool_gpus: set[int] = set()
                _metal_alongside_shortfall: Optional[tuple[int, int]] = None
                # A path that never prices the launch must not commit the previous one's plan.
                self._pending_plan_mib = {}
                # Set by the expert-spill raise after the fit.
                self._moe_spill_batch_restore: Optional[tuple[Optional[int], Optional[int]]] = None
                _moe_cache_candidate = False
                _moe_cache_ram_ok = False
                _moe_table_bytes = 0
                _moe_cache_lazy_table = False
                _shared_gpus = frozenset()
                _integrated_gpus = frozenset()
                # The discrete-first tier; accounting keeps _integrated_gpus.
                _tier_gpus = frozenset()
                # dGPU subsets an explicit pick may price without host-only bytes.
                _discount_prefixes = frozenset()
                # Sized inputs for the tensor-spill planner, None when the fit never
                # priced them. Bound before the try like _placement_verdict_partial:
                # the except arm restores use_fit=True without rebinding
                # kv_cache_bytes, so reading it from the spill arm would be an
                # UnboundLocalError on the path that reaches it most often.
                _spill_inputs: Optional[dict] = None
                # Read after the try: the except arm leaves the planner's locals unbound.
                _tp_planned = False
                # Held, not recorded here: _record_load_warning is first-notice-wins.
                _unmeasured_ctx_notice: Optional[str] = None
                _cuda_ctx_notice: Optional[str] = None
                _ctx_cap_fits = False
                _vram_fit_ctx: Optional[int] = None
                _flat_mtp_engages = False
                total_by_idx: dict[int, int] = {}
                _gpu_mem: list[tuple[int, int, int]] = []
                model_size = None  # set in the fit try; used by the APU RAM guard
                _mtp_will_engage = False
                _separate_draft_launches = False  # a sidecar displaces an embedded head
                # "none" once the fit proves the load needs no demand paging, else None
                # for llama.cpp's own default. Bound before the try like the verdict
                # flags above: the except path falls through to the launch, which reads it.
                _fit_load_mode: Optional[str] = None
                _managed_lazy_flags: Optional[list[str]] = None
                _host_input_embd = 0
                # Layer-fallback min GPUs; raised below on a tensor downgrade. Bound
                # before the try so the --fit-on except path still has it (no UnboundLocal).
                _layer_min_gpus = 1
                # Reject after the try; raising inside would be swallowed into the --fit-on fallback.
                _vulkan_explicit_unmatched = False
                _vulkan_requested_ids: list[int] = []
                _vulkan_available_ordinals: list[int] = []
                # Auto dropped the drafter because only the target fit. Bound before the
                # try: the launch below reads it either way.
                _spec_dropped_no_vram = False
                _mtp_slots_asked: Optional[int] = None
                _mtp_slots_target_ctx: Optional[int] = None
                # The projector runs on the CPU and so allocates no VRAM:
                # --no-mmproj-offload clears mmproj_use_gpu and clip.cpp gates its whole
                # GPU backend on that, leaving no partial offload. Only the paravirtual
                # reason is known this early; the fit below may add the automatic one.
                # Bound before the try, like _spec_dropped_no_vram.
                _mmproj_cpu_pinned = bool(
                    launch_mmproj_path
                    and _paravirtual_cpu_forced
                    and _paravirtual_mmproj_pinnable(server_caps)
                )
                # The APU RAM guard prices system RAM, where a CPU-pinned projector still lives.
                _mmproj_pinned_bytes = 0
                # Raised after this block so the except below cannot swallow it; the handler log line is kept out of this comment because a test string-searches for it.
                _metal_ctx_refusal: Optional[str] = None
                _metal_ctx_warning: Optional[str] = None
                # A discounted Metal ceiling must launch the measured full-offload placement.
                _metal_full_offload_pinned = False
                # --fit-target floor from the Metal wired headroom; 0 off Apple Silicon.
                _metal_fit_target_floor_mib = 0.0
                # --fit-target headroom for a DSpark/DFlash sidecar the child's fitter cannot size.
                _sidecar_fit_reserve_mib = 0
                _sidecar_fit_reserve_eligible_mib = 0
                _sidecar_compute_mib = 0
                _fit_device_count = 1
                _sidecar_reserve_graph_mib = [0]
                _sidecar_draft_layers: Optional[int] = None
                _sidecar_owns_output = False
                _sidecar_override_flags: List[str] = []
                _sidecar_override_pairs: List[List[str]] = []
                _sidecar_reserve_logged: list = []
                _sidecar_reserve_reprice: Optional[Callable[..., int]] = None
                _sidecar_kind_for_fit = None
                # The fit and launch must use the same Model Memory snapshot.
                from utils.model_memory_settings import capture_model_memory_settings

                _mem_settings = capture_model_memory_settings(
                    lambda pair: setattr(self, "_memory_pending_launch", pair)
                )
                try:
                    gguf_size = self._get_gguf_size_bytes(model_path)
                    # Only a GPU-loaded mmproj counts; the resolved value (env or argv) decides.
                    _user_mmproj_offload = _resolved_mmproj_offload(extra_args)
                    mmproj_size = (
                        self._mmproj_vram_bytes(launch_mmproj_path)
                        if (effective_is_vision and _user_mmproj_offload is not False)
                        else 0
                    )
                    if _user_mmproj_offload is False and launch_mmproj_path:
                        # Still in system RAM, which the APU shortfall guard prices.
                        _mmproj_pinned_bytes = self._mmproj_vram_bytes(launch_mmproj_path)
                    model_size = gguf_size + mmproj_size
                    # 2-tuple gpus for existing logic + a total map for the absolute
                    # per-GPU headroom (correct when the GPU is already partly used).
                    # Pass binary so a Vulkan build probes ggml's Vulkan ordinals.
                    # AUTOMATIC placement only. Gating an explicit pin would leave
                    # `gpus` empty, skipping the fit and layer plan and forcing
                    # `--fit on`, yet `gpu_indices` is restored from `gpu_ids` below
                    # so the pin happens anyway. A pinned uncovered GPU is the user's
                    # call and already reports "device kernel image is invalid".
                    # After the download, so the launch reuses fresh rows instead of re-probing (15 s).
                    if self._is_vulkan_backend(binary):
                        _arm_load_probe_memo()
                    _gpu_mem = self._get_gpu_memory(binary, for_llama_server = not gpu_ids)
                    _held_by_others = self._other_planned_vram_mib()
                    _shared_gpus = frozenset(_held_by_others)
                    _gpu_mem = _net_of_other_models_vram(
                        _gpu_mem, _held_by_others, self._media_unallocated_vram_mib()
                    )
                    # Every present device gated out (#7624). Left alone the launch
                    # takes the `--fit on` arm with `gpu_indices` still None, so no
                    # mask is written, the child enumerates every unsupported card and
                    # dies, and the reactive retry cannot help (its guard needs a
                    # truthy `gpu_indices`). CPU is the only placement that runs here.
                    # ROCm auto only; the guard spares CUDA and CPU-only hosts, where
                    # an empty probe just means "no GPU".
                    if (
                        not gpu_ids
                        and not _gpu_mem
                        and not is_vulkan_backend
                        and self._host_torch_is_rocm()
                    ):
                        _ungated = self._get_gpu_memory(binary)
                        # Re-derive: _get_gpu_memory turns any probe error into [], so do not infer from empty.
                        _coverage = self._installed_llama_gfx_archs(binary)
                        _arch_by_id = self._rocm_arch_by_physical_id() if _ungated else {}
                        _all_uncovered = (
                            bool(_ungated)
                            and _coverage is not None
                            and all(
                                (_arch_by_id.get(idx) or "") not in ("", *_coverage)
                                for idx, _free, _total in _ungated
                            )
                        )
                        if _all_uncovered:
                            _arch_gate_forced_cpu = True
                            _present = ", ".join(
                                f"{idx} ({_arch_by_id.get(idx) or 'unknown arch'})"
                                for idx, _free, _total in _ungated
                            )
                            logger.warning(
                                "The installed llama.cpp build has no kernels for "
                                "any GPU on this host (%s), so inference falls back "
                                "to CPU. Reinstall llama.cpp for these devices to "
                                "get GPU inference back.",
                                _present,
                            )
                    _vulkan_igpu_ids = (
                        frozenset(idx for idx, _free, total in _gpu_mem if total <= 0)
                        if is_vulkan_backend
                        else frozenset()
                    )
                    if is_vulkan_backend and not gpu_ids and gpu_memory_mode != "manual":
                        _gpu_mem = self._vulkan_auto_gpu_memory(_gpu_mem)
                    gpus = [(idx, free) for idx, free, _t in _gpu_mem]
                    # Fail-open if none match so a stale UI choice can't strand the load on CPU.
                    if gpu_ids:
                        # Restrict probed pool to selection; if none match, fall back to all GPUs rather than strand on CPU.
                        _wanted_ids = {int(x) for x in gpu_ids}
                        # Reject if ANY requested ordinal is absent, not only when none match.
                        _probed_ordinals = {g[0] for g in gpus}
                        if is_vulkan_backend and not _wanted_ids.issubset(_probed_ordinals):
                            _vulkan_explicit_unmatched = True
                            _vulkan_requested_ids = sorted(_wanted_ids)
                            _vulkan_available_ordinals = sorted(_probed_ordinals)
                        # Restrict the probed pool to the selection; fail-open (keep the
                        # full pool) if none match so a stale UI choice can't strand the
                        # load on CPU (issue #7164).
                        _sel_gpus = [g for g in gpus if g[0] in _wanted_ids]
                        gpus = _sel_gpus if _sel_gpus else gpus
                    total_by_idx = {idx: total for idx, _f, total in _gpu_mem}
                    _shared_pool_gpus = {idx for idx, total in total_by_idx.items() if total <= 0}
                    # GPU picker: restrict every mode to the chosen devices, so
                    # auto selection only considers them and manual mask to
                    # them (the env block below pins CUDA/HIP_VISIBLE_DEVICES).
                    if gpu_ids:
                        _picked = set(gpu_ids)
                        gpus = [g for g in gpus if g[0] in _picked]

                    # GPUs the model will run on -- captured before manual
                    # empty `gpus` to bypass the planner. bool() drives the
                    # GPU-aware speculative defaults; the list feeds the
                    # CPU-fallback check.
                    _detected_gpus = list(gpus)
                    # Auto placement only; Vulkan auto already dropped its iGPUs above.
                    if not gpu_ids and not is_vulkan_backend and len(gpus) > 1:
                        _unified = self._rocm_unified_memory_gpu_ids()
                        # CUDA ids are ordinals; nvidia-smi rows are PCI indices. Join only
                        # where they agree, as _widen_integrated_cuda_rows does.
                        if LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is False or (
                            (os.environ.get("CUDA_DEVICE_ORDER") or "").strip().upper()
                            == "PCI_BUS_ID"
                            and not self._visibility_mask_is_unmappable()
                        ):
                            _unified |= set(
                                LlamaCppBackend._INTEGRATED_CUDA_IDS.get(
                                    LlamaCppBackend._integrated_cuda_mask_key(), ()
                                )
                            )
                        _integrated_gpus = frozenset(idx for idx, _free in gpus if idx in _unified)
                        # A user device flag names ordinals of the base pick's mask, and an
                        # unmappable mask re-emits ordinals as physical ids: keep the base pick.
                        # Host-only bytes still never land on a discrete-only subset.
                        _tier_gpus = _integrated_gpus
                        if _integrated_gpus and (
                            _extra_args_set_any_flag(extra_args, _DEVICE_FLAGS)
                            or any(
                                str(os.environ.get(_v, "")).strip()
                                for _v in ("LLAMA_ARG_DEVICE", "LLAMA_ARG_MAIN_GPU")
                            )
                            or self._visibility_mask_is_unmappable()
                        ):
                            _tier_gpus = frozenset()
                        # Under that pick, host-only bytes may skip only a dGPU subset whose pinned
                        # mask (emitted in ascending ids) keeps every ordinal the flags name on the
                        # same card: a leading run of the inherited order, already ascending.
                        if (
                            _integrated_gpus
                            and not _tier_gpus
                            and not self._visibility_mask_is_unmappable()
                        ):
                            _need = _explicit_device_ordinals_needed(extra_args)
                            if _need is not None:
                                _ids = {idx for idx, _free in gpus}
                                _mask = self._resolve_visible_physical_ids()
                                _order = (
                                    [i for i in _mask if i in _ids]
                                    if _mask is not None
                                    else sorted(_ids)
                                )
                                _discount_prefixes = frozenset(
                                    frozenset(_order[:k])
                                    for k in range(max(1, _need), len(_order) + 1)
                                    if _order[:k] == sorted(_order[:k])
                                )
                    # Vulkan reports total 0 only for integrated GPUs. Their
                    # free "VRAM" is the same host pool the RAM guard prices.
                    _shared_gpu_ids = (
                        {idx for idx, _free in _detected_gpus if total_by_idx.get(idx, 1) <= 0}
                        if is_vulkan_backend
                        else set()
                    )
                    # Checkpoints are host RAM: a discrete card never holds them, but a
                    # unified-memory GPU's "VRAM" is that same pool, so it still pays. ANY
                    # shared candidate keeps the charge: Auto may later pick that one alone.
                    _gpu_ids_now = [idx for idx, _free in gpus]
                    _placement_ctx_checkpoints = (
                        _fit_ctx_checkpoints
                        if _fit_ctx_checkpoints
                        and _gpu_ids_now
                        and (
                            bool(_shared_gpu_ids)
                            or (
                                not is_vulkan_backend
                                and (
                                    self._amd_apu_wants_unified_memory(_gpu_ids_now)
                                    or (
                                        self._integrated_cuda_probe_is_free()
                                        and self._integrated_cuda_unified_memory(_gpu_ids_now)
                                    )
                                )
                            )
                        )
                        else 0
                    )

                    def _subset_ckpts(subset) -> int:
                        # A discrete-tier subset holds no unified-memory GPU, so no checkpoints.
                        if _integrated_gpus and _integrated_gpus.isdisjoint(i for i, _f in subset):
                            return 0
                        return _placement_ctx_checkpoints

                    def _tiered_subsets(
                        ranked: list,
                        floor: int,
                        score_of = None,
                    ) -> list:
                        # (count, prefix) in _select_gpus order: unshared discrete first.
                        tier = LlamaCppBackend._discrete_tier(ranked, _tier_gpus, _shared_gpus)
                        base = [(n, ranked[:n]) for n in range(floor, len(ranked) + 1)]
                        if tier is None or floor > len(tier):
                            return base
                        first = [(n, tier[:n]) for n in range(floor, len(tier) + 1)]
                        if score_of is None:
                            return first + base
                        # Auto: the discrete tier wins only where it gives up nothing the base walk
                        # (the pick before #14372) got. ``score_of`` is None for no fit, else a tuple
                        # (context, drafter kept, ...) the tier must match in every term.
                        base_score = next(
                            (sc for n, sub in base for sc in (score_of(n, sub),) if sc), None
                        )
                        if base_score is None:
                            return first + base
                        kept = []
                        for n, sub in first:
                            sc = score_of(n, sub)
                            if sc and all(t >= b for t, b in zip(sc, base_score)):
                                kept.append((n, sub))
                        return kept + base

                    # The --fit fallback is llama.cpp's own fitter, which knows nothing
                    # about this budget: it keeps its own margin and packs the rest on,
                    # so the slider never reached the path that runs when the fit is
                    # tightest. Hand it the difference, both directions, measured from
                    # the default so an untouched slider leaves the command alone.
                    # --fit-target takes a per-device list in llama.cpp's enumeration
                    # order. Where CUDA's PCI order makes that order provable, each card
                    # gets its own margin, spelled at spawn against the final mask. A
                    # single broadcast value would otherwise have to be sized by the
                    # largest card when lowering, which on mixed cards can exceed a
                    # smaller card's whole memory and leave it unused. Elsewhere (Vulkan,
                    # ROCm, UUID/MIG masks, --device/--rpc) the direction picks the card:
                    # lowering scales by the largest, so no device ends up under the
                    # margin it asked for, raising by the smallest, so none is taken
                    # below its own.
                    if _vram_frac != _CTX_FIT_VRAM_FRACTION and gpus:
                        # Free stands in for an unreported total (MIG/vGPU); deriving from a zero total would silently drop the budget where headroom is hardest to see.
                        _scales = [total_by_idx.get(_idx) or _free for _idx, _free in gpus]
                        _scale = (
                            max(_scales) if _vram_frac < _CTX_FIT_VRAM_FRACTION else min(_scales)
                        )
                        _fit_target_delta_mib = (_CTX_FIT_VRAM_FRACTION - _vram_frac) * _scale
                        # Where llama.cpp's device order is provable, each card gets
                        # its own margin instead (_per_device_fit_target_argv). Every
                        # probed row, not just `gpus`: an unpinned child sees them all.
                        if self._fit_target_per_device_ok(
                            is_vulkan_backend = is_vulkan_backend,
                            extra_args = extra_args,
                            env = os.environ,
                        ):
                            _fit_target_delta_by_idx = {
                                int(_idx): (_CTX_FIT_VRAM_FRACTION - _vram_frac)
                                * (_total if _total and _total > 0 else _free)
                                for _idx, _free, _total in _gpu_mem
                            }

                    # Each card keeps its own margin plus the headroom a mid-session OOM earned.
                    _oom_headroom_mib = self._gpu_oom_headroom_mib(intent)
                    if _oom_headroom_mib:
                        _min_delta = _VRAM_FLOOR_RESERVE_MIB - (
                            _VRAM_FLOOR_RESERVE_MIB
                            if gpu_memory_mode == "manual" and gpu_layers < 0
                            else _LLAMA_FIT_TARGET_DEFAULT_MIB
                        )
                        _fit_target_delta_mib = (
                            max(_min_delta, _fit_target_delta_mib) + _oom_headroom_mib
                        )
                        if _fit_target_delta_by_idx:
                            _fit_target_delta_by_idx = {
                                _idx: max(_min_delta, _delta) + _oom_headroom_mib
                                for _idx, _delta in _fit_target_delta_by_idx.items()
                            }

                    _sysmem_fallback = self._sysmem_fallback_risk(binary)

                    def _gpu_usable(g, frac = _vram_frac):
                        # Callers pass the ACTIVE fraction so the ranking matches the
                        # budget the fit then tests, else mixed totals mis-order.
                        idx, free = g
                        return _vram_usable_mib(
                            free,
                            total_by_idx.get(idx, 0),
                            frac,
                            sysmem_fallback = _sysmem_fallback,
                        )

                    def _gpu_rank(
                        g,
                        frac = _vram_frac,
                        floor_mib = 0.0,
                    ):
                        # iGPUs after every discrete card that holds its split buffers (_select_gpus).
                        usable = _gpu_usable(g, frac)
                        return (g[0] not in _shared_gpu_ids and usable > floor_mib, usable)

                    def _pool_budget_mib(subset, frac):
                        # Per-GPU sum, so an unknown-total GPU cannot add free with no cushion.
                        return sum(max(0.0, _gpu_usable(g, frac)) for g in subset)

                    # Resolve effective context: 0 means let llama-server use
                    # the model's native length. Only expand to a known native
                    # length if metadata exists; else keep 0 as a sentinel.
                    if requested_ctx > 0:
                        effective_ctx = requested_ctx
                    elif self._context_length is not None:
                        effective_ctx = self._context_length
                    else:
                        effective_ctx = 0
                    original_ctx = effective_ctx
                    # Default UI ceiling to the native context length;
                    # GPU/VRAM-fit logic below may shrink it on limited HW.
                    max_available_ctx = self._context_length or effective_ctx

                    # Manual + Auto layers: hand memory to llama.cpp --fit; an explicit gpu_ids pick still pins, and a context of 0 lets --fit size it.
                    if gpu_memory_mode == "manual" and gpu_layers < 0:
                        gpus = []
                        effective_ctx = requested_ctx if requested_ctx > 0 else 0
                        original_ctx = effective_ctx
                        # --fit aborts under --split-mode tensor.
                        extra_args = strip_split_mode_only(extra_args)
                        planned_flash_attn = _replanned_flash_attn(tensor_parallel)
                    elif gpu_memory_mode == "manual":
                        # Manual offload: the user owns masking and context; an explicit gpu_ids pick still pins below.
                        gpus = []
                        effective_ctx = (
                            requested_ctx if requested_ctx > 0 else (self._context_length or 0)
                        )
                        original_ctx = effective_ctx
                        # Single-GPU tensor aborts; non-tensor user modes are kept.
                        if tensor_parallel or split_mode_override == "tensor":
                            extra_args = strip_split_mode_only(extra_args)
                            planned_flash_attn = _replanned_flash_attn(tensor_parallel)

                    # Mirrors _build_speculative_flags; uses _spec_canon since Auto may be dspark/dflash.
                    _mtp_effective = _spec_canon
                    _mtp_size_for_fit = _extract_model_size_b(model_identifier)
                    # Draft-simple/draft-eagle3 in extras load a separate model; reserve only when extras name a drafter.
                    _mtp_sub_3b_for_fit = (
                        _mtp_size_for_fit is not None
                        and _mtp_size_for_fit < _MTP_MIN_SIZE_B
                        and not bool(mtp_draft_path)
                    )
                    # Inherited spec env reaches the child only when extras own --spec-type.
                    _spec_env: Mapping[str, str] = _child_spec_env(extra_args)
                    # Extras can run MTP even when Unsloth suppresses its own emission.
                    _user_mtp_via_extras = _extra_args_requests_mtp(extra_args, env = _spec_env)
                    # A non-MTP model-based draft mode (draft-simple/draft-eagle3) in
                    # extras also loads a separate draft model that needs reserving;
                    # engage only when extras actually name a drafter for it.
                    _user_draft_via_extras = _extra_args_requests_separate_draft(
                        extra_args, env = _spec_env
                    ) and bool(_extra_args_mtp_draft_path(extra_args))
                    # Mirrors _build_speculative_flags: reserve only for MTP the launch will emit, which needs a head or drafter and a binary that supports --spec-type mtp.
                    _mtp_model_for_fit = bool(
                        self._nextn_predict_layers
                        or _is_mtp_model_name(model_identifier, model_path)
                        or bool(mtp_draft_path)
                    ) and not (
                        # Drafterless Gemma falls back to ngram-mod; reserve no
                        # drafter VRAM for it (mirrors the launch resolver).
                        _is_gemma_mtp_name(model_identifier, model_path)
                        and not mtp_draft_path
                        and not self._nextn_predict_layers
                    )
                    _mtp_binary_ok = True
                    _mtp_probe_raised = False
                    if not _user_mtp_via_extras:
                        try:
                            _fit_caps = _launch_caps(binary) or {}
                            _sidecar_cap = {
                                "dspark": "supports_dspark",
                                "dflash": "supports_dflash",
                            }.get(_mtp_effective)
                            _mtp_binary_ok = bool(
                                _fit_caps.get(_sidecar_cap)
                                if _sidecar_cap
                                else _fit_caps.get("mtp_token")
                            )
                        except Exception:
                            _mtp_binary_ok = False
                            _mtp_probe_raised = True
                    _auto_studio_mtp = (
                        not _extra_args_set_spec_type(extra_args)
                        and _mtp_model_for_fit
                        and (
                            _mtp_effective in ("mtp", "mtp+ngram", "dspark", "dflash")
                            or (_mtp_effective == "auto" and not _mtp_sub_3b_for_fit)
                        )
                        and (
                            _mtp_binary_ok
                            # Reserve on a raised (uncached) probe too: it re-probes in
                            # _build_speculative_flags and may still engage MTP (embedded
                            # head or separate drafter -- _mtp_model_for_fit covers both).
                            or _mtp_probe_raised
                        )
                    )
                    _mtp_will_engage = bool(
                        _user_mtp_via_extras or _user_draft_via_extras or _auto_studio_mtp
                    )
                    # The duplicated target-KV copy is MTP-only; separate drafters keep no such copy.
                    _engaged_is_mtp = bool(
                        _user_mtp_via_extras
                        or (_auto_studio_mtp and _mtp_effective not in ("dspark", "dflash"))
                    )
                    # Without draft-mtp the embedded head is never loaded, so its blocks
                    # leave the weights the fit prices (about 3 GB on Nemotron 3.5 Super).
                    if not _engaged_is_mtp:
                        _nextn_unloaded = self._unloaded_nextn_bytes(model_path)
                        gguf_size -= _nextn_unloaded
                        model_size -= _nextn_unloaded
                    # The recurrent rollback copies are a WIDER set than that copy
                    # (_TARGET_ROLLBACK_SPEC_TYPES): DSpark and DFlash pay them too,
                    # and draft-simple is the one engaged draft mode that does not,
                    # which is why the pass-through arm reads the types rather than
                    # assuming every extras-owned drafter qualifies.
                    _target_rollback = bool(
                        _auto_studio_mtp
                        or _user_mtp_via_extras
                        or (
                            _user_draft_via_extras
                            and _accumulated_spec_types(extra_args, env = _spec_env)
                            & _TARGET_ROLLBACK_SPEC_TYPES
                        )
                    )

                    # Effective draft depth, by what the CHILD will run on rather than
                    # what was asked for: an extras depth wins (last-wins at launch),
                    # else the field, else the platform default (2 GPU / 3 CPU).
                    _extra_n_max = _extra_args_spec_draft_n_max(extra_args)
                    _mtp_eff_n_max = _extra_n_max
                    if _mtp_eff_n_max is None and _extra_args_set_spec_type(extra_args):
                        # Extras own the spec block, so the child drafts at the env or build default depth;
                        # pricing a request field of 2 would under-reserve a build that drafts 16.
                        try:
                            _depth_caps = _launch_caps(binary) or {}
                        except Exception:
                            _depth_caps = {}
                        # Only the env twin of the flag this build carries; both names only where the probe named no flag, since a post-rename build ignores the legacy name.
                        _depth_flag = _depth_caps.get("spec_draft_n_max_flag")
                        if _depth_flag == "--draft-max":
                            _env_names = ("LLAMA_ARG_DRAFT_MAX",)
                        elif _depth_flag:
                            _env_names = ("LLAMA_ARG_SPEC_DRAFT_N_MAX",)
                        else:
                            _env_names = ("LLAMA_ARG_SPEC_DRAFT_N_MAX", "LLAMA_ARG_DRAFT_MAX")
                        _env_n_max = next(
                            (v for v in map(_spec_env.get, _env_names) if _is_positive_int(v)),
                            None,
                        )
                        if _env_n_max is not None:
                            _mtp_eff_n_max = int(str(_env_n_max).strip())
                        else:
                            # Probe gave no default: never fall back to the platform default (the child drafts at the build depth); assume the deepest shipped llama.cpp.
                            _mtp_eff_n_max = (
                                _depth_caps.get("spec_draft_n_max_default")
                                or self._UNKNOWN_SPEC_DRAFT_N_MAX
                            )
                    elif _mtp_eff_n_max is None:
                        _mtp_eff_n_max = spec_draft_n_max
                    if _mtp_eff_n_max is None:
                        # _detected_gpus, not gpus: manual empties gpus but still uses the GPU draft depth.
                        _mtp_eff_n_max = (
                            3 if _mtp_effective == "dspark" else (2 if _detected_gpus else 3)
                        )
                    # Size the drafter the launch loads: extras --model-draft (last wins), else Unsloth mtp_draft_path, else env. A wrong one under-reserves and OOMs.
                    _cli_draft_for_budget = _extra_args_mtp_draft_path(extra_args, env = {})
                    _studio_draft_for_budget = (
                        mtp_draft_path
                        if (
                            _mtp_will_engage
                            and mtp_draft_path
                            and not _extra_args_set_spec_type(extra_args)
                        )
                        else None
                    )
                    # Same scrubbed view: reserving for an inherited drafter the child
                    # never loads shrinks the context or rejects a placement that fits.
                    _env_draft_for_budget = _extra_args_mtp_draft_path([], env = _spec_env)
                    _mtp_draft_for_budget = (
                        _cli_draft_for_budget or _studio_draft_for_budget or _env_draft_for_budget
                    )
                    # Whether a separate drafter is emitted, taken before the CPU nulling below. Not mtp_draft_path: extras owning --spec-type bypass the Unsloth sidecar.
                    _separate_draft_launches = bool(_mtp_draft_for_budget)
                    # The child honors LLAMA_ARG_N_GPU_LAYERS_DRAFT, so consult the env too.
                    _draft_on_cpu = _extra_args_draft_offloaded_to_cpu(extra_args, env = os.environ)
                    # Kept for the load-mode fit: a CPU drafter's bytes move to host RAM.
                    _cpu_draft_path = _mtp_draft_for_budget if _draft_on_cpu else None
                    if _draft_on_cpu:
                        _mtp_draft_for_budget = None

                    def _launch_split_env() -> dict:
                        e = dict(os.environ)
                        if not tensor_parallel:
                            sm = (e.get("LLAMA_ARG_SPLIT_MODE") or "").strip().lower()
                            if sm and sm != "layer":
                                e.pop("LLAMA_ARG_SPLIT_MODE", None)
                                e.pop("LLAMA_ARG_TENSOR_SPLIT", None)
                        return e

                    def _sidecar_planned_device_count() -> int:
                        count = len(gpu_indices if gpu_indices is not None else _detected_gpus)
                        if not _gpu_ids_own_device_flags:
                            named = _extra_args_main_device(extra_args) or os.environ.get(
                                "LLAMA_ARG_DEVICE"
                            )
                            if named:
                                count = _distinct_device_count(named)
                        if count > 1 and _split_mode_confines_to_one_device(
                            extra_args, _launch_split_env()
                        ):
                            return 1
                        return count

                    def _draft_on_target_device_now(apple_budget_mib: Optional[int] = None) -> bool:
                        # A draft pinned to another device allocates there (on Apple Silicon only
                        # MTL0 / Vulkan0 is the target's).
                        pin = _extra_args_draft_device_pin(extra_args)
                        if pin is None:
                            return True
                        distinct = {d.strip() for d in pin.split(",") if d.strip()}
                        pin = distinct.pop() if len(distinct) == 1 else pin
                        apple = _apple_budget_mib if apple_budget_mib is None else apple_budget_mib
                        # The picker strips competing device extras and env twins at emission, so
                        # this is the launch's placement.
                        _pl_args = (
                            self._strip_device_extra_args(extra_args)
                            if _gpu_ids_own_device_flags
                            else extra_args
                        )
                        _pl_env = _launch_split_env()
                        if gpu_memory_mode == "manual":
                            self._clear_manual_placement_env(_pl_env)
                        if _gpu_ids_own_device_flags:
                            self._clear_device_placement_env(_pl_env)
                        main = _extra_args_main_device(_pl_args)
                        if main is None and not _gpu_ids_own_device_flags:
                            main = _pl_env.get("LLAMA_ARG_DEVICE") or None
                        if main is None and is_vulkan_backend:
                            ids = (
                                gpu_indices
                                if gpu_indices is not None
                                else [g[0] for g in _detected_gpus]
                            )
                            main = ",".join(f"Vulkan{i}" for i in ids) or (
                                "Vulkan0" if apple > 0 else None
                            )
                        elif main is None and apple > 0:
                            main = "MTL0"
                        elif (main is None and pin.strip()) and all(
                            _CUDA_LIKE_DEVICE_RE.match(d.strip())
                            for d in pin.split(",")
                            if d.strip()
                        ):
                            # CUDA / ROCm pins are remapped into the child's mask: only the spawned
                            # argv can tell (_spawn_draft_on_target).
                            return True

                        if main and _split_mode_confines_to_one_device(_pl_args, _pl_env):
                            listed = [d.strip() for d in str(main).split(",") if d.strip()]
                            if len(listed) > 1:
                                at = self._effective_main_gpu(_pl_args, _pl_env)
                                at = 0 if at is None else at
                                main = listed[at] if 0 <= at < len(listed) else None

                        def _names(value: str) -> set:
                            return {d.strip().lower() for d in str(value).split(",") if d.strip()}

                        return bool(main) and bool(_names(pin)) and _names(pin) <= _names(main)

                    _draft_share_unknown = bool(
                        _mtp_draft_for_budget
                        and _draft_card_share_unknown(
                            extra_args,
                            os.environ,
                            n_draft_layers = getattr(
                                self._draft_backend_for(_mtp_draft_for_budget),
                                "_n_layers",
                                None,
                            ),
                        )
                    )
                    _mtp_draft_weights = 0
                    if _mtp_draft_for_budget:
                        try:
                            _mtp_draft_weights = self._get_gguf_size_bytes(_mtp_draft_for_budget)
                        except Exception:
                            _mtp_draft_weights = 0
                    # Draft K/V types (f16 by default; independent extras overrides).
                    _mtp_draft_ck, _mtp_draft_cv = _extra_args_draft_cache_types(extra_args)
                    # Emitted before the extras, so an extra still last-wins.
                    _budget_draft_cache = (
                        spec_draft_cache_type.strip().lower() if spec_draft_cache_type else None
                    )
                    if _budget_draft_cache in _VALID_KV_CACHE_TYPES:
                        _mtp_draft_ck = _mtp_draft_ck or _budget_draft_cache
                        _mtp_draft_cv = _mtp_draft_cv or _budget_draft_cache

                    # Byte-accurate reserve when dims allow, else None -> flat fallback.
                    mtp_overhead_fn: Optional[Callable[[int], int]] = None
                    # True when the byte reserve is weights-only; the flat cushion then covers the draft KV.
                    _mtp_kv_unsized = False
                    if _mtp_will_engage:
                        _probe_ctx = self._context_length or (
                            effective_ctx if effective_ctx > 0 else 4096
                        )
                        _draft_kv_probe = self._mtp_draft_kv_bytes(
                            _probe_ctx,
                            drafter_path = _mtp_draft_for_budget,
                            draft_cache_type_k = _mtp_draft_ck,
                            draft_cache_type_v = _mtp_draft_cv,
                            n_parallel = n_parallel,
                            swa_full = swa_full,
                            kv_unified = planned_kv_unified,
                            n_ubatch = _effective_ubatch,
                            flash_attn = planned_flash_attn,
                        )
                        if (
                            self._estimate_mtp_overhead_bytes(
                                _probe_ctx,
                                spec_draft_n_max = _mtp_eff_n_max,
                                draft_cache_type_k = _mtp_draft_ck,
                                draft_cache_type_v = _mtp_draft_cv,
                                drafter_path = _mtp_draft_for_budget,
                                draft_weights_bytes = _mtp_draft_weights,
                                n_parallel = n_parallel,
                                mtp_keeps_target_ctx = _engaged_is_mtp,
                                target_rollback = _target_rollback,
                                swa_full = swa_full,
                                kv_unified = planned_kv_unified,
                                n_ubatch = _effective_ubatch,
                                flash_attn = planned_flash_attn,
                            )
                            is not None
                        ):
                            # Reserve is weights-only when the draft KV is unsizable.
                            _mtp_kv_unsized = _draft_kv_probe is None

                            # Closure binding this load's draft params; ctx varies.
                            def mtp_overhead_fn(
                                ctx: int,
                                _n: int = _mtp_eff_n_max,
                                _ck: Optional[str] = _mtp_draft_ck,
                                _cv: Optional[str] = _mtp_draft_cv,
                                _dp: Optional[str] = _mtp_draft_for_budget,
                                _w: int = _mtp_draft_weights,
                                _np: int = n_parallel,
                                _mtp: bool = _engaged_is_mtp,
                                _rollback: bool = _target_rollback,
                                _swa_full: bool = swa_full,
                                _kv_unified: bool = planned_kv_unified,
                                _n_ubatch: Optional[int] = _effective_ubatch,
                            ) -> int:
                                # Applied here, not in a wrapper, so every gate taking this callback
                                # prices the draft micro-batch.
                                _n_ubatch = (
                                    _capped_draft_n_ubatch(
                                        extra_args,
                                        os.environ,
                                        ctx,
                                        _emitted_n_batch(n_batch, _np),
                                        n_ubatch,
                                    )
                                    or _n_ubatch
                                )
                                v = self._estimate_mtp_overhead_bytes(
                                    ctx,
                                    spec_draft_n_max = _n,
                                    draft_cache_type_k = _ck,
                                    draft_cache_type_v = _cv,
                                    drafter_path = _dp,
                                    draft_weights_bytes = _w,
                                    n_parallel = _np,
                                    mtp_keeps_target_ctx = _mtp,
                                    target_rollback = _rollback,
                                    swa_full = _swa_full,
                                    kv_unified = _kv_unified,
                                    n_ubatch = _n_ubatch,
                                    # Read at call time: a downgrade re-plans attention, and a
                                    # default arg would go stale.
                                    flash_attn = planned_flash_attn,
                                )
                                return v if v is not None else 0

                    def _mtp_bytes(
                        ctx: int,
                        slots: Optional[int] = None,
                        ubatch: Optional[int] = None,
                    ) -> int:
                        # Reserve scales with slots/ubatch for compact SWA and MLA+recurrent.
                        if mtp_overhead_fn is None:
                            return 0
                        if slots is None:
                            if ubatch:
                                return mtp_overhead_fn(ctx, _n_ubatch = ubatch)
                            return mtp_overhead_fn(ctx)
                        return mtp_overhead_fn(
                            ctx, _np = slots, _n_ubatch = ubatch if ubatch else _effective_ubatch
                        )

                    def _kv_bytes(ctx: int, ctx_checkpoints: int = 0) -> int:
                        # Checkpoints default OFF: placement prices them itself, and only
                        # on unified memory. The load-mode host figure passes a count.
                        return self._estimate_kv_cache_bytes(
                            ctx,
                            cache_type_kv,
                            n_parallel = n_parallel,
                            swa_full = swa_full,
                            kv_unified = planned_kv_unified,
                            n_ubatch = _effective_ubatch,
                            ctx_checkpoints = ctx_checkpoints,
                            flash_attn = planned_flash_attn,
                        )

                    def _cc_bytes(
                        ctx: int,
                        n_gpus: int = 1,
                        slots: int = 0,
                        cache_type: Optional[str] = None,
                    ) -> int:
                        # Compute growth priced from the LIGHTER KV axis (the heavier scalar hides a paired quantized one); replicated per device; read at call time.
                        return max(1, n_gpus) * self._compute_buffer_ctx_bytes(
                            ctx,
                            _ubatch_for_slots(slots) if slots else _effective_ubatch,
                            cache_type or _scratch_cache_type_kv,
                            layer_split = n_gpus > 1 and not _pipeline_parallel_off,
                            flash_attn = planned_flash_attn,
                            n_parallel = slots or n_parallel,
                        )

                    def _cc_split_extra(
                        ctx: int,
                        slots: int = 0,
                        cache_type: Optional[str] = None,
                    ) -> int:
                        # Per-device step from the single-device to split compute rate, for paths choosing GPUs before the count; 0 when llama.cpp declines pipeline parallelism.
                        return max(
                            0,
                            _cc_bytes(ctx, 2, slots, cache_type) // 2
                            - _cc_bytes(ctx, 1, slots, cache_type),
                        )

                    def _layer_compute_buffer(slots: int) -> int:
                        cb = self._estimate_compute_buffer_bytes(
                            n_ubatch = _ubatch_for_slots(slots),
                            n_parallel = slots,
                            per_device_tensor = False,
                        )
                        return (
                            cb if cb > 0 else self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024
                        )

                    # Layer-split compute buffer (one lump; tensor mode reserves it
                    # per device in _plan_tensor_parallel). Context-independent, so
                    # fold it into the model footprint for the branches below. Falls
                    # back to the flat reserve when dims are missing (returns 0), a
                    # safe upper bound since the tensor buffer >= the layer one.
                    _compute_buffer_pipeline = self._estimate_compute_buffer_bytes(
                        n_ubatch = _effective_ubatch,
                        n_parallel = n_parallel,
                        per_device_tensor = False,
                    )
                    if _compute_buffer_pipeline <= 0:
                        _compute_buffer_pipeline = (
                            self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024
                        )

                    # Tensor mode reserves the whole buffer on every selected device.
                    _compute_buffer_tensor = self._estimate_compute_buffer_bytes(
                        n_ubatch = _effective_ubatch,
                        n_parallel = n_parallel,
                        per_device_tensor = True,
                    )
                    if _compute_buffer_tensor <= 0:
                        _compute_buffer_tensor = (
                            self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024
                        )

                    # Reserve extra devices' fixed overhead so a k-GPU split cannot OOM a device.
                    _pipeline_overhead_bytes = self._PIPELINE_PER_DEVICE_OVERHEAD_MIB * 1024 * 1024

                    # Explicit n_ctx: honor it, capping only if no GPU combination fits. Auto: prefer fewer GPUs with reduced context (multi-GPU is slower).
                    gpu_indices, use_fit = None, True
                    # Per-GPU weight proportions for tensor mode (None lets
                    # llama.cpp split by free VRAM).
                    tp_tensor_split: Optional[list[int]] = None
                    # Whether the TP branch planned this load. Not readable from tp_tensor_split (None when llama.cpp sizes it); re-set per retry so a retry starts clean.
                    _tp_planned = False
                    explicit_ctx = requested_ctx > 0
                    # No speculative GPU load when the drafter is CPU-offloaded; a sidecar wins over an embedded head (has_dft), and a head with no sidecar stays reserved.
                    _draft_cpu_no_embedded = _draft_on_cpu and (
                        _separate_draft_launches or not self._nextn_predict_layers
                    )
                    # A replay was fitted without the drafter: re-fit it like a fresh MTP load, before any explicit_ctx reader.
                    if (
                        explicit_ctx
                        and ctx_override is None
                        and intent.max_seq_length_auto_derived
                        # CPU-pinned drafters too: the target still pays rollback state for them.
                        and _mtp_will_engage
                        # Forced = anything that bypasses the Auto drop probe, advanced arguments included.
                        and (
                            (_canonicalize_spec_mode(speculative_type) or "auto") != "auto"
                            or _user_mtp_via_extras
                            or _user_draft_via_extras
                            or _extra_args_set_spec_type(extra_args)
                            or _extra_args_mtp_draft_path(extra_args, env = _spec_env)
                        )
                    ):
                        explicit_ctx = False
                        _replayed_ctx_refit = True

                    # Run before the probe, which is gated on not tensor_parallel but must answer for any load that ends layer-split, so the downgrades must come first.
                    if preserve_multi_gpu_on_layer:
                        _layer_min_gpus = max(_layer_min_gpus, len(gpus))

                    if tensor_parallel and self._tensor_quant_kv_unsupported_binary(
                        binary, _planned_cache_pair
                    ):
                        # This build refused quantized KV under tensor: go straight to layer, which keeps the quantized cache, instead of reloading just to hear it again.
                        logger.info(
                            "Tensor parallelism skipped: this llama.cpp build cannot "
                            "use a quantized KV cache with --split-mode tensor "
                            "(needs b9455+); loading with a layer split."
                        )
                        tensor_parallel = False
                        # A missing capability says nothing about capacity, so keep multi-GPU.
                        _layer_min_gpus = max(_layer_min_gpus, len(gpus))
                        extra_args = strip_split_mode_only(extra_args)
                        planned_flash_attn = _replanned_flash_attn(tensor_parallel)
                    if tensor_parallel and self._tensor_split_aborts(
                        binary, model_identifier, _planned_cache_pair
                    ):
                        # Aborted on tensor for this model this session; skip tensor upfront.
                        logger.info(
                            "Tensor parallelism skipped: this llama.cpp build aborted "
                            "on --split-mode tensor for this model earlier this "
                            "session; using layer split across %d GPU(s).",
                            len(gpus),
                        )
                        tensor_parallel = False
                        # The abort says nothing about capacity, so keep multi-GPU.
                        _layer_min_gpus = max(_layer_min_gpus, len(gpus))
                        # An extras --split-mode tensor would re-engage tensor mode.
                        extra_args = strip_split_mode_only(extra_args)
                        planned_flash_attn = _replanned_flash_attn(tensor_parallel)

                    # Tensor mode replicates a compute buffer on every GPU, so drop
                    # GPUs below that reserve from the set up front (gpu_indices
                    # becomes the CUDA_VISIBLE_DEVICES mask, fully excluding them).
                    tp_gpus = gpus
                    # Manual mode owns the layer count and context, so it skips
                    # the memory-based planner; its toggle still emits
                    # --split-mode tensor below (split by free VRAM, or by the
                    # Split ratio if set). auto plans here.
                    plan_tp = tensor_parallel and gpu_memory_mode != "manual"
                    if plan_tp:
                        # Downstream, incl. the drop probe, prices the projector off these two; drafter is dropped last.
                        _tp_reserve_bytes = self._estimate_compute_buffer_bytes(
                            n_ubatch = _effective_ubatch,
                            n_parallel = n_parallel,
                            per_device_tensor = True,
                        )
                        reserve_mib = (
                            _tp_reserve_bytes // (1024 * 1024)
                            if _tp_reserve_bytes > 0
                            else self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
                        )
                        # Admit by usable budget, not raw free (a partly-used big card has no budget left).
                        tp_gpus = [g for g in gpus if _gpu_usable(g) >= reserve_mib]

                    if plan_tp and len(tp_gpus) < 2:
                        # Tensor needs >= 2 usable GPUs; otherwise drop --split-mode tensor (a single GPU is a no-op and zero GPUs must not reach llama-server) and use layer or CPU.
                        logger.info(
                            "Tensor parallelism requested but only %d of %d GPU(s) "
                            "have enough free VRAM for the compute buffer; "
                            "ignoring (needs >= 2).",
                            len(tp_gpus),
                            len(gpus),
                        )
                        tensor_parallel = False
                        # Cards below tensor's reserve can still do layer split, so keep multi-GPU.
                        if len(gpus) >= 2:
                            _layer_min_gpus = max(_layer_min_gpus, len(gpus))
                        extra_args = strip_split_mode_only(extra_args)
                        planned_flash_attn = _replanned_flash_attn(tensor_parallel)

                    # Normalize before anything prices it: _mtp_bytes reads mtp_overhead_fn at call time.
                    if _draft_cpu_no_embedded and mtp_overhead_fn is not None:
                        # Drafter on CPU: only the target's rollback snapshots and verification rows
                        # stay on GPU.
                        def _cpu_draft_target_state(
                            _ctx: int,
                            _np: int = n_parallel,
                            _n_ubatch: Optional[int] = _effective_ubatch,
                            _n: int = _mtp_eff_n_max,
                            _rollback: bool = _target_rollback,
                        ) -> int:
                            if _n <= 0:
                                return 0
                            rollback = self._rollback_state_bytes(_np) * _n if _rollback else 0
                            return rollback + self._spec_verify_rows_bytes(_np, _n)

                        mtp_overhead_fn = (
                            _cpu_draft_target_state
                            if _cpu_draft_target_state(effective_ctx) > 0
                            else None
                        )
                        _mtp_kv_unsized = False

                    # Projector goes to host RAM first when the load does not fit, but never drops vision; pin only helps where a device has its own budget (total > 0).
                    _mm_pin_deferred = False

                    def _apply_mmproj_cpu_pin(floor_ctx: int) -> None:
                        nonlocal _mmproj_cpu_pinned, _mmproj_pinned_bytes
                        nonlocal mmproj_size, model_size
                        _mmproj_cpu_pinned = True
                        _mmproj_pinned_bytes = mmproj_size
                        # No KV estimate: the fit never applies a floor, so pricing the reserve at one promises unseen room.
                        mmproj_size = 0
                        model_size = gguf_size
                        # Announce a CPU-resident projector: set after the per-load reset and before recovery paths, which override it with a more specific reason.
                        self._mmproj_fallback_reason = "cpu_offload"
                        logger.info(
                            "Auto: running the vision projector on the CPU "
                            "(--no-mmproj-offload); it does not fit in VRAM "
                            "alongside the model at %d context. Image encoding "
                            "gets slower, text generation is unaffected. Pass "
                            "--mmproj-offload in the advanced arguments to keep it "
                            "on the GPU.",
                            floor_ctx,
                        )

                    def _host_input_bytes_now(weights: int) -> int:
                        if not (
                            gpus
                            and not _integrated_gpus
                            and not _shared_gpu_ids
                            and not _shared_pool_gpus
                            and not _metal_capable_host()
                            and not self._amd_apu_wants_unified_memory([i for i, _f in gpus])
                            and not set(
                                LlamaCppBackend._INTEGRATED_CUDA_IDS.get(
                                    LlamaCppBackend._integrated_cuda_mask_key(), ()
                                )
                            ).intersection(i for i, _f in gpus)
                        ):
                            return 0
                        # A streamed table stays on the host whatever -ot says.
                        _inp_extras = (
                            self._strip_device_extra_args(extra_args)
                            if _gpu_ids_own_device_flags
                            else extra_args
                        )
                        _inp_env = dict(os.environ)
                        if gpu_memory_mode == "manual":
                            self._clear_manual_placement_env(_inp_env)
                        if _gpu_ids_own_device_flags:
                            self._clear_device_placement_env(_inp_env)
                        self._reconcile_split_mode_env(_inp_env, tensor_parallel = tensor_parallel)
                        try:
                            _inp_lazy = "off"
                            if server_caps.get("supports_lazy_mode"):
                                _inp_lazy = _effective_lazy_mode(
                                    _inp_extras,
                                    _inp_env,
                                    flag = str(server_caps.get("lazy_mode_flag") or "--lazy-mode"),
                                )
                            if _inp_lazy == "auto" and self._lazy_auto_resolves_off(
                                gpu_indices = (
                                    gpu_indices
                                    if gpu_indices is not None or not gpu_ids
                                    else [int(i) for i in gpu_ids]
                                ),
                                detected_gpus = _detected_gpus,
                                shared_gpu_ids = _shared_gpu_ids,
                                is_vulkan_backend = is_vulkan_backend,
                                extra_args = _inp_extras,
                                env = _inp_env,
                                tensor_parallel = tensor_parallel,
                                integrated_gpu_ids = _vulkan_igpu_ids,
                            ):
                                _inp_lazy = "off"
                            found = _host_pinned_input_bytes(
                                self._gguf_tensor_scan(model_path),
                                _inp_extras,
                                _inp_env,
                                lazy_mode = _inp_lazy,
                                host_forced = self._host_forced_table_archs(binary),
                            )
                        except Exception as e:  # unread header: charge the whole file
                            logger.debug("Host-pinned input embeddings unpriced: %s", e)
                            return 0
                        return min(max(0, found), max(0, int(weights) - 1))

                    _mm_budgeted_gpus = [
                        (_idx, _free) for _idx, _free in gpus if total_by_idx.get(_idx, 0) > 0
                    ]
                    _discrete_vram = bool(_mm_budgeted_gpus)
                    if (
                        effective_is_vision
                        and mmproj_size > 0
                        and _discrete_vram
                        and not _mmproj_cpu_pinned
                        and gpu_memory_mode != "manual"
                        # Not gated on tensor_parallel: the answer is never applied to a tensor load, but the tensor weight-budget check may downgrade one to layer.
                        and _paravirtual_mmproj_pinnable(server_caps)
                        # arg.cpp applies env before argv, so a user's spelling in either owns the placement.
                        and not _extra_args_set_mmproj_offload(extra_args)
                        and not _env_sets_mmproj_offload()
                    ):
                        # Is model+drafter+projector VRAM-resident? Auto asks at the floor (shrinks context first); explicit asks at its requested length, since --fit spills layers.
                        _mm_mtp_on_gpu = _mtp_will_engage and not _draft_cpu_no_embedded
                        _mm_frac = _vram_frac - (
                            _MTP_VRAM_RESERVE_FRAC
                            if (_mm_mtp_on_gpu and (mtp_overhead_fn is None or _mtp_kv_unsized))
                            else 0.0
                        )
                        _mm_floor_ctx = (
                            effective_ctx
                            if explicit_ctx
                            else min(self._MMPROJ_FIT_FLOOR_CTX, effective_ctx)
                        )
                        _mm_need = (
                            gguf_size
                            - _host_input_bytes_now(gguf_size)
                            + mmproj_size
                            + _compute_buffer_pipeline
                            + self._CUDA_CONTEXT_RESERVE_BYTES
                            + int(mmproj_size * (self._MMPROJ_VRAM_SAFETY - 1.0))
                            + (self._MTP_DRAFT_COMPUTE_BYTES if _mm_mtp_on_gpu else 0)
                        )
                        if _mm_floor_ctx > 0 and self._can_estimate_kv():
                            _mm_need += (
                                _kv_bytes(_mm_floor_ctx)
                                + _cc_bytes(_mm_floor_ctx)
                                + _mtp_bytes(_mm_floor_ctx)
                            )
                        else:
                            # Flat fraction is the reserve when _mtp_bytes is 0; the probe must test the budget the fit uses.
                            _mm_floor_ctx = self._context_length or effective_ctx or 4096
                            _mm_need += _mtp_bytes(_mm_floor_ctx)
                        # Use the same selector, fraction and split-aware overhead as the placement,
                        # so this probe
                        # is never more optimistic than the placement it gates.
                        _mm_split_aware = self._can_estimate_kv()

                        def _mm_probe_needs_fit(need: int, devices: list) -> bool:
                            return (
                                self._select_gpus_split_aware(
                                    need,
                                    devices,
                                    usable_fraction = _mm_frac,
                                    total_by_idx = total_by_idx,
                                    per_device_overhead_bytes = _pipeline_overhead_bytes
                                    + _cc_bytes(_mm_floor_ctx),
                                    min_gpus = _layer_min_gpus,
                                    split_extra_bytes = _cc_split_extra(_mm_floor_ctx),
                                    shared_gpu_ids = _shared_gpu_ids,
                                )
                                if _mm_split_aware
                                else self._select_gpus(
                                    need,
                                    devices,
                                    usable_fraction = _mm_frac,
                                    total_by_idx = total_by_idx,
                                    per_device_overhead_bytes = _pipeline_overhead_bytes,
                                    min_gpus = _layer_min_gpus,
                                    shared_gpu_ids = _shared_gpu_ids,
                                )
                            )[1]

                        _mm_needs_fit = _mm_probe_needs_fit(_mm_need, _mm_budgeted_gpus)
                        # An APU reporting a total is budgeted above, but its "VRAM" is host
                        # RAM, so moving the projector frees nothing there. Also pin when that
                        # alone lets the discrete cards hold the model (#7624).
                        _mm_discrete = LlamaCppBackend._discrete_tier(
                            _mm_budgeted_gpus, _tier_gpus, _shared_gpus
                        )
                        if (
                            not _mm_needs_fit
                            and _mm_discrete
                            and _layer_min_gpus <= len(_mm_discrete)
                        ):
                            _mm_needs_fit = _mm_probe_needs_fit(
                                _mm_need, _mm_discrete
                            ) and not _mm_probe_needs_fit(
                                _mm_need
                                - mmproj_size
                                - int(mmproj_size * (self._MMPROJ_VRAM_SAFETY - 1.0)),
                                _mm_discrete,
                            )
                        if _mm_needs_fit:
                            if tensor_parallel:
                                # Held, not applied here: layer-split numbers would misprice a tensor load. The tensor downgrade below needs this verdict.
                                _mm_pin_deferred = True
                            else:
                                _apply_mmproj_cpu_pin(_mm_floor_ctx)

                    # Auto drops the drafter rather than trade context or --fit offload for it, priced over
                    # the same subsets the placement loop walks. Skipped under tensor parallelism.
                    if (
                        _mtp_will_engage
                        and (_canonicalize_spec_mode(speculative_type) or "auto") == "auto"
                        and not _user_mtp_via_extras
                        and not _user_draft_via_extras
                        and not _extra_args_set_spec_type(extra_args)
                        # llama.cpp loads a bare --model-draft anyway (has_dft()), so keep the reserve.
                        and not _extra_args_mtp_draft_path(extra_args, env = _spec_env)
                        and not _draft_cpu_no_embedded
                        and not _draft_share_unknown
                        and _draft_on_target_device_now(apple_budget_mib = 0)
                        and not tensor_parallel
                        and gpus
                        and effective_ctx > 0
                        and self._can_estimate_kv()
                    ):

                        def _probe_frac(drafter: bool) -> float:
                            # The flat fraction is the reserve whenever _mtp_bytes is 0.
                            # Same budget the fit uses, else the probe answers for a
                            # card size the fit never tests.
                            return _vram_frac - (
                                _MTP_VRAM_RESERVE_FRAC
                                if (drafter and (mtp_overhead_fn is None or _mtp_kv_unsized))
                                else 0.0
                            )

                        def _probe_base(
                            drafter: bool,
                            n: int,
                            slots: int = 0,
                        ) -> int:
                            # model_size_fit's terms, before it is built below.
                            _soft = self._CUDA_CONTEXT_RESERVE_BYTES
                            if effective_is_vision and mmproj_size > 0:
                                _soft += int(mmproj_size * (self._MMPROJ_VRAM_SAFETY - 1.0))
                            if drafter:
                                _soft += self._MTP_DRAFT_COMPUTE_BYTES
                            return (
                                model_size
                                - _host_input_bytes_now(model_size)
                                + (
                                    _layer_compute_buffer(slots)
                                    if slots
                                    else _compute_buffer_pipeline
                                )
                                + _soft
                                + max(0, n - 1) * _pipeline_overhead_bytes
                            )

                        # Same order and budget as the auto placement loop.
                        def _probe_rank(drafter: bool) -> list:
                            return sorted(
                                gpus,
                                key = lambda g: _gpu_rank(
                                    g,
                                    _probe_frac(drafter),
                                    _pipeline_overhead_bytes / (1024 * 1024),
                                ),
                                reverse = True,
                            )

                        _probe_overhead_mib = _pipeline_overhead_bytes / (1024 * 1024)

                        def _probe_floor(ranked: list, drafter: bool) -> int:
                            # Same floor as the placement loop _auto_min_gpus, so a one-GPU placement the loop forbids cannot be the one that saves the drafter.
                            return max(
                                1,
                                min(
                                    _layer_min_gpus,
                                    sum(
                                        1
                                        for g in ranked
                                        if _gpu_usable(g, _probe_frac(drafter))
                                        > _probe_overhead_mib
                                    )
                                    or 1,
                                ),
                            )

                        # Both rankings: the unsized-drafter fraction can reorder heterogeneous cards.
                        _probe_orders = [_probe_rank(False)]
                        _ranked_w = _probe_rank(True)
                        if [id(g) for g in _ranked_w] != [id(g) for g in _probe_orders[0]]:
                            _probe_orders.append(_ranked_w)
                        _target_fits_somewhere = False
                        _both_fit_somewhere = False
                        _probe_ctx = 0
                        _probe_need = _probe_have = 0.0
                        # (GPU count, target-only ctx, budget with drafter, checkpoints) per
                        # drafter-ranked subset; a discrete-tier subset pays no checkpoints.
                        _target_placements: list[tuple[int, int, float, int]] = []
                        _drafter_order = _probe_orders[-1]

                        def _probe_target_ctx(_n: int, _subset) -> Optional[int]:
                            # The target-only context the placement loop gives ``_subset``, or None.
                            _cc_n = lambda c, _k = _n: _cc_bytes(c, _k)
                            _base_wo = _probe_base(False, _n)
                            _budget_wo = _pool_budget_mib(_subset, _probe_frac(False))
                            # Explicit context is honored verbatim, so that is what the
                            # drafter has to fit alongside; Auto gets its own best cap.
                            _ctx_wo = (
                                effective_ctx
                                if explicit_ctx
                                else self._fit_context_to_vram(
                                    effective_ctx,
                                    _budget_wo,
                                    _base_wo,
                                    cache_type_kv,
                                    swa_full = swa_full,
                                    n_parallel = n_parallel,
                                    kv_unified = planned_kv_unified,
                                    n_ubatch = _effective_ubatch,
                                    flash_attn = planned_flash_attn,
                                    mtp_engaged = False,
                                    mtp_overhead_fn = None,
                                    compute_ctx_bytes_fn = _cc_n,
                                    budget_frac = 1.0,
                                    pooled = True,
                                    total_mib = None,
                                    ctx_checkpoints = _subset_ckpts(_subset),
                                )
                            )
                            if _ctx_wo <= 0:
                                return None
                            _shared = _kv_bytes(_ctx_wo) + _cc_n(_ctx_wo)
                            _foot_wo = (_base_wo + _shared) / (1024 * 1024)
                            if _foot_wo > _budget_wo:
                                return None
                            # The pooled figure hides the compute buffer every device
                            # replicates, so the weakest card may not hold the context
                            # the pool priced and the placement loop caps to what it
                            # does. Charging the drafter at the uncapped context
                            # condemns it at a context this load never reaches. Same
                            # reserve and cap the auto-context loop below applies, so
                            # the two cannot disagree. Auto only: an explicit context
                            # is honored verbatim and overflows to --fit.
                            _usable_wo = [_gpu_usable(g, _probe_frac(False)) for g in _subset]
                            _probe_reserve_at = lambda c, _k = _n: (
                                (_pipeline_overhead_bytes if _k > 1 else 0) + _cc_bytes(c, _k) // _k
                            )
                            if not self._every_gpu_holds_reserve(
                                _usable_wo, _probe_reserve_at(_ctx_wo)
                            ):
                                if explicit_ctx:
                                    # Nothing to cap: this subset cannot hold it, and
                                    # _select_gpus_split_aware will go to --fit too.
                                    # Calling it a fit would drop the drafter and then
                                    # report a pin that never happened.
                                    return None
                                _ctx_wo = self._cap_ctx_to_per_device_reserve(
                                    _ctx_wo, _usable_wo, _probe_reserve_at
                                )
                                if _ctx_wo <= 0:
                                    return None
                                # Every pooled term shrinks with the context, so this
                                # cannot newly fail; re-price rather than lean on it.
                                _shared = _kv_bytes(_ctx_wo) + _cc_n(_ctx_wo)
                                _foot_wo = (_base_wo + _shared) / (1024 * 1024)
                                if _foot_wo > _budget_wo:
                                    return None
                            return _ctx_wo

                        _slots_reducible = (
                            n_parallel > 1
                            and intent.n_parallel_from_default
                            and _extra_args_n_parallel(extra_args, env = {}) is None
                        )

                        def _mtp_foot_mib(n: int, ctx: int, s: int) -> float:
                            ub = _ubatch_for_slots(s)
                            return (
                                _probe_base(True, n, s)
                                + self._estimate_kv_cache_bytes(
                                    ctx,
                                    cache_type_kv,
                                    n_parallel = s,
                                    swa_full = swa_full,
                                    kv_unified = planned_kv_unified,
                                    n_ubatch = ub,
                                    flash_attn = planned_flash_attn,
                                )
                                + _cc_bytes(ctx, n, s)
                                + _mtp_bytes(ctx, s, ub)
                            ) / (1024 * 1024)

                        def _ctx_with_drafter(
                            n: int, ctx: int, budget: float, s: int, ckpts: int
                        ) -> int:
                            if explicit_ctx:
                                return ctx if _mtp_foot_mib(n, ctx, s) <= budget else 0
                            ub = _ubatch_for_slots(s)
                            fitted = self._fit_context_to_vram(
                                ctx,
                                budget,
                                _probe_base(True, n, s),
                                cache_type_kv,
                                swa_full = swa_full,
                                n_parallel = s,
                                kv_unified = planned_kv_unified,
                                n_ubatch = ub,
                                flash_attn = planned_flash_attn,
                                mtp_engaged = True,
                                mtp_overhead_fn = (
                                    functools.partial(mtp_overhead_fn, _np = s, _n_ubatch = ub)
                                    if mtp_overhead_fn is not None
                                    else None
                                ),
                                compute_ctx_bytes_fn = lambda c: _cc_bytes(c, n, s),
                                budget_frac = 1.0,
                                pooled = True,
                                total_mib = None,
                                ctx_checkpoints = ckpts,
                            )
                            return (
                                fitted
                                if fitted > 0 and _mtp_foot_mib(n, fitted, s) <= budget
                                else 0
                            )

                        def _probe_slots(_n: int, _subset, _ctx: int) -> int:
                            # Most slots the slot search below can keep the drafter at here (ctx
                            # kept, less the trim at one slot), 0 if none or slots are pinned.
                            if not _slots_reducible:
                                return 0
                            _budget = _pool_budget_mib(_subset, _probe_frac(True))
                            _ckpts = _subset_ckpts(_subset)
                            for _s in range(n_parallel - 1, 0, -1):
                                _keep = 1 - (
                                    _MTP_SLOT_CTX_TRIM if _s == 1 and not explicit_ctx else 0
                                )
                                _c = _ctx_with_drafter(_n, _ctx, _budget, _s, _ckpts)
                                if _c > 0 and _c >= _ctx * _keep:
                                    return _s
                            return 0

                        def _probe_score(_n: int, _subset):
                            # (target context, drafter fits beside it at that context, slots it
                            # keeps); explicit context is the same everywhere.
                            _ctx = _probe_target_ctx(_n, _subset)
                            if _ctx is None:
                                return None
                            _need = (
                                _probe_base(True, _n)
                                + _kv_bytes(_ctx)
                                + _cc_bytes(_ctx, _n)
                                + _mtp_bytes(_ctx)
                            ) / (1024 * 1024)
                            if _need <= _pool_budget_mib(_subset, _probe_frac(True)):
                                return (_ctx, 1, n_parallel)
                            # No drafter at the requested slots: the placement keeping more
                            # slots for it wins, so the tier cannot cost the base walk slots.
                            return (_ctx, 0, _probe_slots(_n, _subset, _ctx))

                        for _probe_ranked in _probe_orders:
                            if _both_fit_somewhere:
                                break
                            _probe_min_gpus = _probe_floor(_probe_ranked, False)
                            # The placement loop's subsets in its order: discrete first.
                            for _n, _subset in _tiered_subsets(
                                _probe_ranked,
                                _probe_min_gpus,
                                score_of = _probe_score,
                            ):
                                _ctx_wo = _probe_target_ctx(_n, _subset)
                                if _ctx_wo is None:
                                    continue
                                _cc_n = lambda c, _k = _n: _cc_bytes(c, _k)
                                _shared = _kv_bytes(_ctx_wo) + _cc_n(_ctx_wo)
                                _foot_w = (
                                    _probe_base(True, _n) + _shared + _mtp_bytes(_ctx_wo)
                                ) / (1024 * 1024)
                                _budget_w = _pool_budget_mib(_subset, _probe_frac(True))
                                if not _target_fits_somewhere:
                                    _target_fits_somewhere = True
                                    _probe_ctx, _probe_need, _probe_have = (
                                        _ctx_wo,
                                        _foot_w,
                                        _budget_w,
                                    )
                                if _probe_ranked is _drafter_order:
                                    _target_placements.append(
                                        (_n, _ctx_wo, _budget_w, _subset_ckpts(_subset))
                                    )
                                if _foot_w <= _budget_w:
                                    _both_fit_somewhere = True
                                    break
                                # The loop would re-cap context with the drafter, which is the trade
                                # being refused.
                                _ctx_w = (
                                    0
                                    if explicit_ctx
                                    else self._fit_context_to_vram(
                                        _ctx_wo,
                                        _budget_w,
                                        _probe_base(True, _n),
                                        cache_type_kv,
                                        swa_full = swa_full,
                                        n_parallel = n_parallel,
                                        kv_unified = planned_kv_unified,
                                        n_ubatch = _effective_ubatch,
                                        flash_attn = planned_flash_attn,
                                        mtp_engaged = True,
                                        mtp_overhead_fn = mtp_overhead_fn,
                                        compute_ctx_bytes_fn = _cc_n,
                                        budget_frac = 1.0,
                                        pooled = True,
                                        total_mib = None,
                                        ctx_checkpoints = _subset_ckpts(_subset),
                                    )
                                )
                                if (
                                    _ctx_w > 0
                                    and (
                                        _probe_base(True, _n)
                                        + _kv_bytes(_ctx_w)
                                        + _cc_n(_ctx_w)
                                        + _mtp_bytes(_ctx_w)
                                    )
                                    / (1024 * 1024)
                                    <= _budget_w
                                ):
                                    # A NextN-only head is the main decode speedup: pay context.
                                    if (
                                        _engaged_is_mtp
                                        and _arch_has_fast_mla_mtp(
                                            getattr(self, "_architecture", None)
                                        )
                                        and _ctx_w >= min(_ctx_wo, _FAST_MTP_MIN_CTX)
                                    ):
                                        _both_fit_somewhere = True
                                        logger.info(
                                            "Auto: keeping the embedded MTP head; context %d -> %d "
                                            "so it fits (%.1f GB budget).",
                                            _ctx_wo,
                                            _ctx_w,
                                            _budget_w / 1024,
                                        )
                                    break
                        # Fewer slots before no drafter. The walk takes the first subset holding the
                        # drafter, so it must keep its target-only ctx (less the trim at one slot).
                        # An extras --parallel wins at launch, so it is kept; the env twin is scrubbed.
                        if _target_fits_somewhere and not _both_fit_somewhere and _slots_reducible:
                            _mtp_slots = _mtp_slots_ctx = 0
                            for _s in range(n_parallel - 1, 0, -1):
                                _keep = 1 - (
                                    _MTP_SLOT_CTX_TRIM if _s == 1 and not explicit_ctx else 0
                                )
                                for _n, _ctx_c, _budget_c, _ckpts_c in _target_placements:
                                    _c = _ctx_with_drafter(_n, _ctx_c, _budget_c, _s, _ckpts_c)
                                    if _c <= 0:
                                        continue
                                    if _c >= _ctx_c * _keep:
                                        _mtp_slots, _mtp_slots_ctx = _s, _c
                                        _mtp_slots_target_ctx = _ctx_c
                                    break
                                if _mtp_slots:
                                    break
                            if _mtp_slots:
                                logger.info(
                                    "Auto: serving slots reduced %d -> %d so the drafter fits "
                                    "in VRAM at context %d (target alone: %d). Set Parallel "
                                    "Slots to keep %d without the drafter.",
                                    n_parallel,
                                    _mtp_slots,
                                    _mtp_slots_ctx,
                                    _probe_ctx,
                                    n_parallel,
                                )
                                _mtp_slots_asked = n_parallel
                                n_parallel = _mtp_slots  # allow-slot-clamp: the drafter fits in VRAM only at fewer slots
                                _effective_ubatch = _ubatch_for_slots(n_parallel)
                                _compute_buffer_pipeline = _layer_compute_buffer(n_parallel)
                                if mtp_overhead_fn is not None:
                                    # Its defaults were bound at the requested count.
                                    mtp_overhead_fn = functools.partial(
                                        mtp_overhead_fn,
                                        _np = n_parallel,
                                        _n_ubatch = _effective_ubatch,
                                    )
                                _both_fit_somewhere = True
                        if _target_fits_somewhere and not _both_fit_somewhere:
                            _spec_dropped_no_vram = True
                            _mtp_will_engage = False
                            # Clearing the flag alone leaves the reserve (the fit calls any non-None mtp_overhead_fn): clear that and _mtp_kv_unsized too, or the flat fraction returns.
                            mtp_overhead_fn = None
                            _mtp_kv_unsized = False
                            logger.warning(
                                "Speculative decoding disabled for this load: the model "
                                "fits in VRAM at context %d but its drafter does not "
                                "(needs %.1f GB of a %.1f GB budget, on any GPU subset). "
                                "Auto keeps the context rather than shrink it for a speed "
                                "option. Select the drafter in Settings to force it.",
                                _probe_ctx,
                                _probe_need / 1024,
                                _probe_have / 1024,
                            )

                    # Flat fallback only when the draft KV is unsized; an embedded head keeps its reserve.
                    _flat_mtp_engages = _mtp_will_engage and (
                        mtp_overhead_fn is None or _mtp_kv_unsized
                    )
                    # MTP reserves GPU VRAM unless its only drafter is a separate
                    # CPU-offloaded one (an embedded head stays on GPU). The tensor
                    # path reserves like the layer path; gate both on this.
                    _mtp_reserves_gpu = _mtp_will_engage and not _draft_cpu_no_embedded
                    # Whether any byte reserve survives on GPU (a CPU drafter keeps target rollback state). Kept apart from _mtp_reserves_gpu, which also charges the draft graph.
                    _mtp_gpu_bytes_remain = _mtp_reserves_gpu or mtp_overhead_fn is not None
                    _flat_mtp_reserve = (
                        _MTP_VRAM_RESERVE_FRAC
                        if (_flat_mtp_engages and not _draft_cpu_no_embedded)
                        else 0.0
                    )
                    _pin_fraction = _vram_frac - _flat_mtp_reserve

                    # CUDA/cuBLAS context is discrete-GPU only (not Metal); mmproj and MTP draft-graph buffers exist on every backend, so charge them everywhere.
                    _soft_overhead = self._CUDA_CONTEXT_RESERVE_BYTES if gpus else 0
                    if effective_is_vision and mmproj_size > 0:
                        _soft_overhead += int(mmproj_size * (self._MMPROJ_VRAM_SAFETY - 1.0))
                    if _mtp_reserves_gpu:
                        _soft_overhead += self._MTP_DRAFT_COMPUTE_BYTES
                    # On shared memory (APU/iGPU, unified, unknown total) a CPU-pinned projector still sits in
                    # the measured pool, so charge it when shared memory is present.
                    _shared_pool_mmproj = (
                        _mmproj_pinned_bytes
                        if (not gpus or len(_mm_budgeted_gpus) < len(gpus))
                        else 0
                    )
                    _host_input_embd = _host_input_bytes_now(model_size)
                    if _host_input_embd:
                        logger.info(
                            "GPU fit: %.2f GB of input embeddings and lookup tables stay "
                            "in host memory; not charged to VRAM.",
                            _host_input_embd / (1024**3),
                        )
                    model_size_fit = (
                        model_size
                        - _host_input_embd
                        + _compute_buffer_pipeline
                        + _soft_overhead
                        + _shared_pool_mmproj
                    )

                    def _subset_model_size(n_gpus: int) -> int:
                        return model_size_fit + max(0, n_gpus - 1) * _pipeline_overhead_bytes

                    def _discrete_host_only(subset = None) -> int:
                        # The shared-pool projector charge a discrete tier drops: only when
                        # every discrete total is known, and none for a subset holding an
                        # iGPU. Read by name so a later rebind of _shared_pool_mmproj carries.
                        if subset is not None and not _integrated_gpus.isdisjoint(
                            i for i, _f in subset
                        ):
                            return 0
                        if _integrated_gpus and all(
                            (total_by_idx or {}).get(i, 0) > 0
                            for i, _f in gpus
                            if i not in _integrated_gpus
                        ):
                            return _shared_pool_mmproj
                        return 0

                    # Unified-memory budget (0 off Apple Silicon) for the no-GPU Metal cap below.
                    _apple_budget_mib = self._apple_metal_memory_budget_bytes() // (1024 * 1024)
                    _metal_media_mib, _metal_media_unallocated_mib = (
                        self._metal_media_mib() if _apple_budget_mib > 0 else (0, 0)
                    )
                    if _metal_media_mib:
                        # 0 would mean "not Apple Silicon" and skip every Metal check.
                        _apple_budget_mib = max(
                            1,
                            min(
                                _apple_budget_mib,
                                _apple_budget_beside_media_mib(
                                    _metal_media_mib, _metal_media_unallocated_mib
                                ),
                            ),
                        )
                    _metal_media_wired_mib = int(
                        _metal_media_unallocated_mib * _APPLE_WIRED_CEILING_FRACTION
                    )

                    def _apple_wired_left_mib(wired_bytes: int) -> int:
                        return max(0, wired_bytes // (1024 * 1024) - _metal_media_wired_mib)

                    if tensor_parallel and tp_gpus:
                        # The planner can shrink ctx/KV, not weights or the MTP reserve.
                        _tp_weight_budget_mib = (
                            sum(_gpu_usable(g) for g in tp_gpus) - len(tp_gpus) * reserve_mib
                        )
                        _tp_flat_mtp = 2 * 1024**3  # flat reserve when dims unavailable
                        if not _mtp_gpu_bytes_remain:
                            # No MTP, or its only drafter is CPU-offloaded and the
                            # target keeps no state of its own (no GPU bytes at all).
                            _tp_mtp_floor = 0
                        elif mtp_overhead_fn is not None and not _mtp_kv_unsized:
                            _tp_mtp_floor = _mtp_bytes(
                                min(2048, effective_ctx) if effective_ctx > 0 else 2048
                            )
                        else:
                            # Tensor mode has no --fit valve: keep the flat cushion for unsized KV, never below the known byte reserve.
                            _tp_mtp_floor = max(
                                _tp_flat_mtp,
                                _mtp_bytes(min(2048, effective_ctx) if effective_ctx > 0 else 2048),
                            )
                        _tp_required_mib = (
                            model_size - _host_input_embd + _tp_mtp_floor + _soft_overhead
                        ) / (1024 * 1024)
                        if _tp_weight_budget_mib <= _tp_required_mib:
                            logger.info(
                                "Tensor parallelism requested but the pooled VRAM "
                                "budget cannot hold the weights, MTP reserve, and "
                                "per-device compute buffers; falling back to layer split."
                            )
                            tensor_parallel = False
                            # Weights needed >1 card, so keep multi-GPU across the
                            # usable tensor GPUs.
                            if len(tp_gpus) >= 2:
                                _layer_min_gpus = max(_layer_min_gpus, len(tp_gpus))
                            extra_args = strip_split_mode_only(extra_args)
                            planned_flash_attn = _replanned_flash_attn(tensor_parallel)
                            # Apply the deferred pin only without a GPU drafter: the drafter-drop
                            # probe did not run.
                            if _mm_pin_deferred and not _mtp_reserves_gpu:
                                _apply_mmproj_cpu_pin(_mm_floor_ctx)
                                _mm_pin_deferred = False
                                # _subset_model_size closes over model_size_fit by name, so
                                # rebinding carries through.
                                _soft_overhead = self._CUDA_CONTEXT_RESERVE_BYTES if gpus else 0
                                if _mtp_reserves_gpu:
                                    _soft_overhead += self._MTP_DRAFT_COMPUTE_BYTES
                                _shared_pool_mmproj = (
                                    _mmproj_pinned_bytes
                                    if (not gpus or len(_mm_budgeted_gpus) < len(gpus))
                                    else 0
                                )
                                model_size_fit = (
                                    model_size
                                    - _host_input_embd
                                    + _compute_buffer_pipeline
                                    + _soft_overhead
                                    + _shared_pool_mmproj
                                )

                    if tensor_parallel and tp_gpus:
                        # Tensor-parallel allocation; see _plan_tensor_parallel.
                        target_ctx = (
                            effective_ctx
                            if explicit_ctx
                            else (self._context_length or effective_ctx)
                        )
                        # Weights-only mtp_overhead_fn: pass the flat cushion for unsized KV, or the binary search spends that cushion on context.
                        _tp_unsized_mtp_reserve = (
                            2 * 1024**3 if (_mtp_reserves_gpu and _mtp_kv_unsized) else 0
                        )
                        (
                            effective_ctx,
                            max_available_ctx,
                            gpu_indices,
                            tp_tensor_split,
                        ) = self._plan_tensor_parallel(
                            tp_gpus,
                            max(1, model_size - _host_input_embd),
                            target_ctx,
                            cache_type_kv = cache_type_kv,
                            scratch_cache_type_kv = _scratch_cache_type_kv,
                            n_parallel = n_parallel,
                            mtp_engaged = _mtp_reserves_gpu,
                            mtp_overhead_fn = mtp_overhead_fn,
                            mtp_flat_reserve_bytes = _tp_unsized_mtp_reserve,
                            # Report the UI ceiling from native ctx, not the
                            # explicit small request.
                            max_target_ctx = self._context_length or target_ctx,
                            total_by_idx = total_by_idx,
                            n_ubatch = _effective_ubatch,
                            soft_overhead_bytes = _soft_overhead,
                            swa_full = swa_full,
                            kv_unified = planned_kv_unified,
                            flash_attn = planned_flash_attn,
                            # Already resolved: a mid-load save must not change the fraction here.
                            vram_fraction = _vram_frac,
                            explicit_ctx = explicit_ctx,
                        )
                        use_fit = False
                        _tp_planned = True
                        if explicit_ctx and not self._can_estimate_kv():
                            # Tensor mode emits --fit off, so nothing downstream catches it.
                            _unmeasured_ctx_notice = self._unmeasured_context_notice(
                                effective_ctx, cache_type_kv
                            )
                    elif gpus and self._can_estimate_kv() and effective_ctx > 0:
                        # Hardware cap for UI bounds, independent of the requested context.
                        native_ctx_for_cap = self._context_length or effective_ctx
                        if native_ctx_for_cap > 0:
                            ranked_for_cap = sorted(
                                gpus,
                                key = lambda g: _gpu_rank(
                                    g,
                                    _vram_frac - _flat_mtp_reserve,
                                    _pipeline_overhead_bytes / (1024 * 1024),
                                ),
                                reverse = True,
                            )
                            best_cap = 0
                            _cap_fraction = _vram_frac - _flat_mtp_reserve
                            # Every subset Auto may take, discrete tier included, priced as Auto prices it.
                            # With an integrated GPU, start at Auto's GPU floor: a forced split never takes
                            # a checkpoint-free dGPU-only subset, explicit device pick or not.
                            _cap_floor = 1
                            if _integrated_gpus:
                                _cap_floor = max(
                                    1,
                                    min(
                                        _layer_min_gpus,
                                        sum(
                                            1
                                            for g in ranked_for_cap
                                            if _gpu_usable(g, _cap_fraction)
                                            > _pipeline_overhead_bytes / (1024 * 1024)
                                        )
                                        or 1,
                                    ),
                                )

                            def _cap_fit(n_gpus: int, subset) -> Optional[int]:
                                # The context Auto would cap ``subset`` to for the UI ceiling, or None.
                                # Per-GPU-consistent pool budget (fixes mixed
                                # known/unknown totals); pass it as an absolute
                                # budget so the fit and the check below agree.
                                pool_budget = _pool_budget_mib(subset, _cap_fraction)
                                _ms = _subset_model_size(n_gpus) - _discrete_host_only(subset)
                                # Compute buffer is replicated per device in a layer
                                # split, so scale the context term by the subset size.
                                _cc_sub = lambda c, n = n_gpus: _cc_bytes(c, n)
                                capped = self._fit_context_to_vram(
                                    native_ctx_for_cap,
                                    pool_budget,
                                    _ms,
                                    cache_type_kv,
                                    swa_full = swa_full,
                                    n_parallel = n_parallel,
                                    kv_unified = planned_kv_unified,
                                    n_ubatch = _effective_ubatch,
                                    flash_attn = planned_flash_attn,
                                    mtp_engaged = _mtp_reserves_gpu,
                                    mtp_overhead_fn = mtp_overhead_fn,
                                    compute_ctx_bytes_fn = _cc_sub,
                                    budget_frac = 1.0,
                                    pooled = True,
                                    total_mib = None,
                                    ctx_checkpoints = _subset_ckpts(subset),
                                )
                                kv = _kv_bytes(capped)
                                footprint_mib = (
                                    _ms + kv + _mtp_bytes(capped) + _cc_sub(capped)
                                ) / (1024 * 1024)
                                if footprint_mib > pool_budget:
                                    return None
                                if (
                                    n_gpus > 1
                                    and _integrated_gpus
                                    and _integrated_gpus.isdisjoint(i for i, _f in subset)
                                ):
                                    # A dGPU-only tier is not a usable-descending prefix, so pooling can
                                    # hide a small card's replicated reserve: cap as Auto's walk does.
                                    _usable_mib = [_gpu_usable(g, _cap_fraction) for g in subset]
                                    _reserve_at = lambda c, n = n_gpus: (
                                        _pipeline_overhead_bytes + _cc_bytes(c, n) // n
                                    )
                                    if not self._every_gpu_holds_reserve(
                                        _usable_mib, _reserve_at(capped)
                                    ):
                                        capped = self._cap_ctx_to_per_device_reserve(
                                            capped, _usable_mib, _reserve_at
                                        )
                                return capped

                            def _cap_score(n_gpus: int, subset):
                                capped = _cap_fit(n_gpus, subset)
                                return None if capped is None else (capped,)

                            for n_gpus, subset in _tiered_subsets(
                                ranked_for_cap, _cap_floor, score_of = _cap_score
                            ):
                                capped = _cap_fit(n_gpus, subset)
                                if capped is not None:
                                    best_cap = max(best_cap, capped)
                            if best_cap > 0:
                                max_available_ctx = best_cap
                                _ctx_cap_fits = True
                                # A forced GPU floor plans subsets this sweep did not hold to it.
                                _vram_fit_ctx = best_cap if _layer_min_gpus <= 1 else None
                            else:
                                # No context fits: anchor the UI safe zone at the Auto offload fallback so Auto
                                # never exceeds its own published ceiling.
                                max_available_ctx = min(_AUTO_OFFLOAD_CTX, native_ctx_for_cap)

                        if explicit_ctx:
                            # Honor the requested context; if it does not fit, ship --fit on and let -ngl flex.
                            requested_total = (
                                model_size_fit
                                + _kv_bytes(effective_ctx)
                                + _mtp_bytes(effective_ctx)
                                + _cc_bytes(effective_ctx)
                            )
                            # Compute buffer is replicated on every device in a layer split: priced single-device, then re-checked at the split rate, falling back to --fit on.
                            gpu_indices, use_fit = self._select_gpus_split_aware(
                                requested_total,
                                gpus,
                                usable_fraction = _pin_fraction,
                                total_by_idx = total_by_idx,
                                per_device_overhead_bytes = _pipeline_overhead_bytes
                                + _cc_bytes(effective_ctx),
                                min_gpus = _layer_min_gpus,
                                split_extra_bytes = _cc_split_extra(effective_ctx),
                                shared = _shared_gpus,
                                integrated = _integrated_gpus,
                                prefer_discrete = bool(_tier_gpus),
                                discount_prefixes = _discount_prefixes,
                                discrete_size_bytes = requested_total - _discrete_host_only(),
                                shared_gpu_ids = _shared_gpu_ids,
                            )
                            # No silent shrink: effective_ctx stays requested_ctx; use_fit means --fit on will offload. -nkvo keeps the cache on the host, so overflow is not real.
                            if (
                                use_fit
                                and _ctx_cap_fits
                                and not _cuda_ctx_notice
                                and _kv_offload_from_args(extra_args)
                                and not _args_place_tensors_on_cpu(extra_args)
                                and not _env_places_tensors_on_cpu()
                                and not _device_selection_is_cpu(extra_args, os.environ)
                                and not _extra_args_set_any_flag(extra_args, _GPU_LAYER_FLAGS)
                                and not _env_fixes_gpu_layers()
                                # A trailing user --fit off disables the fitter that would offload.
                                and (
                                    fit_is_enabled_in(extra_args)
                                    or not _extra_args_set_any_flag(extra_args, {"-fit", "--fit"})
                                )
                                # At the native ceiling the true max above it was never searched.
                                and max_available_ctx < native_ctx_for_cap
                            ):
                                _q8_fits = False
                                if planned_flash_attn and (
                                    cache_type_kv or "f16"
                                ).strip().lower() in (
                                    "f16",
                                    "fp16",
                                    "",
                                ):
                                    _q8_kv = self._estimate_kv_cache_bytes(
                                        effective_ctx,
                                        "q8_0",
                                        n_parallel = n_parallel,
                                        swa_full = swa_full,
                                        kv_unified = planned_kv_unified,
                                        n_ubatch = _effective_ubatch,
                                        flash_attn = planned_flash_attn,
                                    )
                                    if _q8_kv > 0:
                                        _, _q8_use_fit = self._select_gpus_split_aware(
                                            model_size_fit
                                            + _q8_kv
                                            + _mtp_bytes(effective_ctx)
                                            + _cc_bytes(effective_ctx, cache_type = "q8_0"),
                                            gpus,
                                            usable_fraction = _pin_fraction,
                                            total_by_idx = total_by_idx,
                                            per_device_overhead_bytes = (
                                                _pipeline_overhead_bytes
                                                + _cc_bytes(effective_ctx, cache_type = "q8_0")
                                            ),
                                            min_gpus = _layer_min_gpus,
                                            split_extra_bytes = _cc_split_extra(
                                                effective_ctx, cache_type = "q8_0"
                                            ),
                                            shared = _shared_gpus,
                                            integrated = _integrated_gpus,
                                            prefer_discrete = bool(_tier_gpus),
                                            discount_prefixes = _discount_prefixes,
                                            discrete_size_bytes = model_size_fit
                                            + _q8_kv
                                            + _mtp_bytes(effective_ctx)
                                            + _cc_bytes(effective_ctx, cache_type = "q8_0")
                                            - _discrete_host_only(),
                                            shared_gpu_ids = _shared_gpu_ids,
                                        )
                                        _q8_fits = not _q8_use_fit
                                _cuda_ctx_notice = self._cuda_context_overcommit_notice(
                                    effective_ctx,
                                    max_available_ctx,
                                    cache_type_kv,
                                    quantised_ctx_fits = _q8_fits,
                                )
                        else:
                            # Rank by the active pin fraction so the order matches the fit budget.
                            pin_fraction = _pin_fraction
                            # Max _reserve_at ask: a short card must not hide an iGPU that fits.
                            _rank_floor_mib = (
                                _pipeline_overhead_bytes + _cc_bytes(effective_ctx, 2) // 2
                            ) / (1024 * 1024)
                            ranked = sorted(
                                gpus,
                                key = lambda g: _gpu_rank(g, pin_fraction, _rank_floor_mib),
                                reverse = True,
                            )
                            # Skips _select_gpus, so apply its per-device overhead cap.
                            _pipeline_overhead_mib = _pipeline_overhead_bytes / (1024 * 1024)
                            _auto_min_gpus = max(
                                1,
                                min(
                                    _layer_min_gpus,
                                    sum(
                                        1
                                        for g in ranked
                                        if _gpu_usable(g, pin_fraction) > _pipeline_overhead_mib
                                    )
                                    or 1,
                                ),
                            )
                            _ctx_before_fit = effective_ctx
                            if _shared_gpus and _auto_min_gpus == 1 and effective_ctx > 0:
                                # Sharing a card splits its compute between the models.
                                _alone_mib = (
                                    _subset_model_size(1)
                                    + _kv_bytes(effective_ctx)
                                    + _mtp_bytes(effective_ctx)
                                    + _cc_bytes(effective_ctx, 1)
                                ) / (1024 * 1024)
                                _unshared = next(
                                    (
                                        g
                                        for g in ranked
                                        if g[0] not in _shared_gpus
                                        and _gpu_usable(g, pin_fraction) >= _alone_mib
                                    ),
                                    None,
                                )
                                if _unshared is not None:
                                    ranked.remove(_unshared)
                                    ranked.insert(0, _unshared)

                            def _subset_ms(n_gpus: int, subset) -> int:
                                # A CPU-pinned projector is charged for a shared pool; a
                                # discrete-tier subset draws on none.
                                return _subset_model_size(n_gpus) - _discrete_host_only(subset)

                            def _auto_fit(n_gpus: int, subset, ctx: int) -> Optional[int]:
                                # The context ``subset`` holds when Auto caps ``ctx`` to it, or None.
                                pool_budget = _pool_budget_mib(subset, pin_fraction)
                                _ms = _subset_ms(n_gpus, subset)
                                # Compute buffer is replicated per device in a layer
                                # split, so scale the context term by the subset size.
                                _cc_sub = lambda c, n = n_gpus: _cc_bytes(c, n)
                                capped = self._fit_context_to_vram(
                                    ctx,
                                    pool_budget,
                                    _ms,
                                    cache_type_kv,
                                    swa_full = swa_full,
                                    n_parallel = n_parallel,
                                    kv_unified = planned_kv_unified,
                                    n_ubatch = _effective_ubatch,
                                    flash_attn = planned_flash_attn,
                                    mtp_engaged = _mtp_reserves_gpu,
                                    mtp_overhead_fn = mtp_overhead_fn,
                                    compute_ctx_bytes_fn = _cc_sub,
                                    budget_frac = 1.0,
                                    pooled = True,
                                    total_mib = None,
                                    ctx_checkpoints = _subset_ckpts(subset),
                                )
                                kv = _kv_bytes(capped)
                                footprint_mib = (
                                    _ms + kv + _mtp_bytes(capped) + _cc_sub(capped)
                                ) / (1024 * 1024)
                                if footprint_mib > pool_budget:
                                    return None
                                # Pooling admits a subset whose per-device reserve
                                # does not fit the smallest card, which then OOMs at
                                # launch; only knowable once `capped` is chosen.
                                _usable_mib = [_gpu_usable(g, pin_fraction) for g in subset]
                                _reserve_at = lambda c, n = n_gpus: (
                                    (_pipeline_overhead_bytes if n > 1 else 0)
                                    + _cc_bytes(c, n) // n
                                )
                                if not self._every_gpu_holds_reserve(
                                    _usable_mib, _reserve_at(capped)
                                ):
                                    # Keep the largest context the smallest card holds
                                    # rather than lose the subset to the 4096 drop below.
                                    # Costs no context: a longer prefix never has a larger
                                    # smallest card, and the reserve is one function for n > 1.
                                    capped = self._cap_ctx_to_per_device_reserve(
                                        capped, _usable_mib, _reserve_at
                                    )
                                    if capped <= 0:
                                        return None
                                    # Pooled terms only shrink with the context, so this
                                    # cannot newly fail; re-price rather than lean on that
                                    # across the KV and MTP branches.
                                    kv = _kv_bytes(capped)
                                    footprint_mib = (
                                        _ms + kv + _mtp_bytes(capped) + _cc_sub(capped)
                                    ) / (1024 * 1024)
                                    if footprint_mib > pool_budget:
                                        return None
                                return capped

                            def _auto_score(
                                n_gpus: int,
                                subset,
                                ctx = effective_ctx,
                            ):
                                capped = _auto_fit(n_gpus, subset, ctx)
                                return None if capped is None else (capped,)

                            _subsets = _tiered_subsets(ranked, _auto_min_gpus, score_of = _auto_score)
                            for n_gpus, subset in _subsets:
                                capped = _auto_fit(n_gpus, subset, effective_ctx)
                                if capped is None:
                                    continue
                                effective_ctx = capped
                                _ctx_capped_for_vram = 0 < capped < _ctx_before_fit
                                if (
                                    _ctx_capped_for_vram
                                    and _tier_gpus
                                    and _tier_gpus.isdisjoint(i for i, _f in subset)
                                ):
                                    # Capped only by the discrete preference: the base walk
                                    # holds the full context, so nothing evicted would help.
                                    _base_ctx = next(
                                        (
                                            c
                                            for n in range(_auto_min_gpus, len(ranked) + 1)
                                            for c in (_auto_fit(n, ranked[:n], _ctx_before_fit),)
                                            if c is not None
                                        ),
                                        None,
                                    )
                                    _ctx_capped_for_vram = not (
                                        _base_ctx is not None and _base_ctx >= _ctx_before_fit
                                    )
                                gpu_indices = sorted(idx for idx, _ in subset)
                                use_fit = False
                                break
                            else:
                                # No GPU subset (discrete or full pool) holds the model at a
                                # fitted context. Prefer a useful chat context before
                                # handing placement to --fit on and host offload.
                                effective_ctx = min(_AUTO_OFFLOAD_CTX, effective_ctx)
                                if effective_ctx > 0:
                                    for n_gpus, subset in _subsets:
                                        kv = _kv_bytes(effective_ctx)
                                        footprint_mib = (
                                            _subset_ms(n_gpus, subset)
                                            + kv
                                            + _mtp_bytes(effective_ctx)
                                            + _cc_bytes(effective_ctx, n_gpus)
                                        ) / (1024 * 1024)
                                        if footprint_mib > _pool_budget_mib(subset, pin_fraction):
                                            continue
                                        # Pooling hides the per-device reserve here too.
                                        if not self._every_gpu_holds_reserve(
                                            (_gpu_usable(g, pin_fraction) for g in subset),
                                            (_pipeline_overhead_bytes if n_gpus > 1 else 0)
                                            + _cc_bytes(effective_ctx, n_gpus) // n_gpus,
                                        ):
                                            continue
                                        gpu_indices = sorted(idx for idx, _ in subset)
                                        use_fit = False
                                        break

                    elif gpus:
                        # Without KV metadata llama.cpp owns the fit; keep the Auto default of the measured offload path.
                        logger.debug(
                            "Falling back to file-size-only GPU selection",
                            model_size_gb = round(model_size / (1024**3), 2),
                        )
                        # Byte-accurate MTP reserve and the flat fallback are mutually exclusive by design.
                        _fs_total = model_size_fit + _mtp_bytes(
                            self._context_length or effective_ctx or 4096
                        )
                        gpu_indices, use_fit = self._select_gpus(
                            _fs_total,
                            gpus,
                            usable_fraction = _pin_fraction,
                            total_by_idx = total_by_idx,
                            per_device_overhead_bytes = _pipeline_overhead_bytes,
                            min_gpus = _layer_min_gpus,
                            shared = _shared_gpus,
                            integrated = _integrated_gpus,
                            prefer_discrete = bool(_tier_gpus),
                            discount_prefixes = _discount_prefixes,
                            discrete_size_bytes = _fs_total - _discrete_host_only(),
                            shared_gpu_ids = _shared_gpu_ids,
                        )
                        if use_fit and not explicit_ctx:
                            # Without KV metadata, llama.cpp owns the fit. Keep the
                            # same useful Auto default as the measured offload path.
                            effective_ctx = (
                                min(_AUTO_OFFLOAD_CTX, effective_ctx)
                                if effective_ctx > 0
                                else _AUTO_OFFLOAD_CTX
                            )

                    elif _apple_budget_mib > 0 and effective_ctx > 0:
                        # No GPU on Metal: cap here, or the context stays native and over-commits
                        # unified memory.
                        # llama.cpp's --fit is blind to Unsloth's own footprint and the wired limit.
                        native_ctx_for_cap = self._context_length or effective_ctx
                        # With --fit off (argv or LLAMA_ARG_FIT env) nothing reduces what we emit, so hand over the floor the llama.cpp fitter itself would reduce to.
                        _metal_floor_ctx = (
                            _FIT_MIN_CTX
                            if fit_is_effectively_on(extra_args, os.environ)
                            else _LLAMA_FIT_MIN_CTX
                        )
                        # Only a ceiling backed by a real KV estimate may refuse; the 4096 fallback is a guess.
                        _apple_measured_ceiling: Optional[int] = None
                        # The other measured verdict: nothing fits, so there is no ceiling
                        # to name and every explicit request over-commits.
                        _apple_nothing_fits = False
                        # No measurement: the floor is a guess and may not refuse or be published.
                        _apple_ctx_unmeasured = False
                        # Reserve the flat MTP fraction up front so an unsized draft cannot over-commit; exclusive with the byte-accurate _mtp_bytes reserve.
                        _apple_fit_budget_mib = int(
                            _apple_budget_mib * max(0.0, 1.0 - _flat_mtp_reserve)
                        )
                        # A CPU-pinned projector is already in model_size_fit here; do not charge it twice.
                        _apple_host_embd = self._metal_demand_paged_embedding_bytes(
                            model_path,
                            extra_args,
                            binary = binary,
                            requested_load_mode = load_mode,
                            supports_load_mode = load_mode_support(server_caps),
                            settings = _mem_settings,
                            layers_fixed = _caller_owns_budget or _paravirtual_cpu_forced,
                            mtp_may_engage = _mtp_will_engage,
                        )
                        _apple_probe_embd = _apple_host_embd
                        # Metal refuses past the wired limit; None is unknown, 0 used up.
                        _apple_wired = self._apple_metal_wired_headroom_bytes()
                        _apple_resident_budget_mib = (
                            _apple_fit_budget_mib
                            if _apple_wired is None
                            else min(
                                _apple_fit_budget_mib,
                                int(
                                    _apple_wired
                                    // (1024 * 1024)
                                    * max(0.0, 1.0 - _flat_mtp_reserve)
                                ),
                            )
                        )
                        _apple_extras = (
                            self._strip_device_extra_args(extra_args)
                            if _gpu_ids_own_device_flags
                            else extra_args
                        )
                        _apple_env = dict(os.environ)
                        if gpu_memory_mode == "manual":
                            self._clear_manual_placement_env(_apple_env)
                        if _gpu_ids_own_device_flags:
                            self._clear_device_placement_env(_apple_env)
                        self._reconcile_split_mode_env(_apple_env, tensor_parallel = tensor_parallel)
                        _apple_host_embd = self._apple_lookup_table_discount(
                            _apple_host_embd,
                            model_path,
                            caps = server_caps,
                            extra_args = _apple_extras,
                            env = _apple_env,
                            model_size_fit = model_size_fit,
                            need_at_ctx = (
                                (
                                    _kv_bytes(effective_ctx, _fit_ctx_checkpoints)
                                    + _cc_bytes(effective_ctx)
                                    + _mtp_bytes(effective_ctx)
                                    + _apple_cpu_draft
                                )
                                if effective_ctx > 0
                                and not _pass_through_reprices_cache(extra_args, n_parallel)
                                and not _flat_mtp_engages
                                and (
                                    _apple_cpu_draft := self._cpu_resident_draft_bytes(
                                        effective_ctx,
                                        drafter_path = _cpu_draft_path,
                                        draft_cache_type_k = _mtp_draft_ck,
                                        draft_cache_type_v = _mtp_draft_cv,
                                        n_parallel = n_parallel,
                                        swa_full = swa_full,
                                        kv_unified = planned_kv_unified,
                                        n_ubatch = _effective_ubatch,
                                        flash_attn = planned_flash_attn,
                                    )
                                )
                                is not None
                                else None
                            ),
                            budget_mib = _apple_resident_budget_mib,
                            cache_ram = cache_ram,
                            binary = binary,
                            cpu_forced = _paravirtual_cpu_forced or _arch_gate_forced_cpu,
                        )
                        _managed_lazy_flags = list(self._apple_lazy_flags)
                        _apple_model_size_fit = model_size_fit - _apple_host_embd
                        if _apple_host_embd:
                            logger.info(
                                "Metal fit: %.2f GB of input embeddings stay in the CPU "
                                "file mapping; charging %.2f GB of weights to unified memory.",
                                _apple_host_embd / (1024**3),
                                _apple_model_size_fit / (1024**3),
                            )

                        def _apple_ctx_fit(
                            target: int,
                            min_ctx: int,
                            budget_mib: Optional[int] = None,
                        ) -> int:
                            return self._fit_context_to_vram(
                                target,
                                _apple_fit_budget_mib if budget_mib is None else budget_mib,
                                _apple_model_size_fit,
                                cache_type_kv,
                                min_ctx = min_ctx,
                                swa_full = swa_full,
                                n_parallel = n_parallel,
                                kv_unified = planned_kv_unified,
                                n_ubatch = _effective_ubatch,
                                flash_attn = planned_flash_attn,
                                mtp_engaged = _mtp_reserves_gpu,
                                mtp_overhead_fn = mtp_overhead_fn,
                                compute_ctx_bytes_fn = _cc_bytes,
                                budget_frac = 1.0,
                                pooled = True,
                                total_mib = None,
                                ctx_checkpoints = _fit_ctx_checkpoints,
                            )

                        def _apple_footprint_mib(ctx: int, ctx_checkpoints: int = 0) -> float:
                            return (
                                _apple_model_size_fit
                                + _kv_bytes(ctx, ctx_checkpoints)
                                + _mtp_bytes(ctx)
                                + _cc_bytes(ctx)
                            ) / (1024 * 1024)

                        # Auto only: drop the drafter when the target fits Metal alone but not with
                        # the sidecar reserve.
                        if (
                            use_fit
                            and _mtp_will_engage
                            and _mtp_reserves_gpu
                            and _mtp_effective in ("dspark", "dflash")
                            and (_canonicalize_spec_mode(speculative_type) or "auto") == "auto"
                            and not _extra_args_set_spec_type(extra_args)
                            # A bare --model-draft in extras still loads (has_dft()).
                            and not _extra_args_mtp_draft_path(extra_args, env = _spec_env)
                            and not _draft_share_unknown
                            and _draft_on_target_device_now()
                            # Zero KV would read as a target that fits alone.
                            and self._can_estimate_kv()
                        ):
                            # _apple_fit_budget_mib already holds back a flat fraction for this
                            # drafter.
                            _sc_ctx = (
                                effective_ctx
                                if explicit_ctx
                                else min(_metal_floor_ctx, native_ctx_for_cap or _metal_floor_ctx)
                            )
                            _sc_target_mib = (
                                _apple_model_size_fit
                                + _kv_bytes(_sc_ctx, _fit_ctx_checkpoints)
                                + _cc_bytes(_sc_ctx)
                            ) / (1024 * 1024)
                            _sc_draft_ub = _capped_draft_n_ubatch(
                                extra_args,
                                os.environ,
                                _sc_ctx,
                                _emitted_n_batch(n_batch, n_parallel),
                                n_ubatch,
                            )
                            _sc_with_draft_mib = _sc_target_mib + self._sidecar_fit_reserve_mib(
                                _mtp_bytes(
                                    _sc_ctx,
                                    slots = _extra_args_n_parallel(extra_args, os.environ),
                                    ubatch = _sc_draft_ub,
                                ),
                                mtp_draft_path,
                                flat_mib = math.ceil(_flat_mtp_reserve * max(0, _apple_budget_mib)),
                            )
                            # An explicit context is refused past the wired limit when that is
                            # readable.
                            _sc_wired_mib = (
                                self._apple_metal_wired_ceiling_bytes() // (1024 * 1024)
                                if explicit_ctx
                                and not _caller_owns_budget
                                and not _paravirtual_cpu_forced
                                else 0
                            )
                            _sc_over_limit = _apple_budget_mib < _sc_with_draft_mib or (
                                0 < _sc_wired_mib < _sc_with_draft_mib
                            )
                            # Only when both do not fit: the draft-free embedding discount can cost
                            # a cold llama-fit-params run.
                            _sc_host_embd_without_draft = _apple_host_embd
                            _sc_target_only_mib = _sc_target_mib
                            if _sc_over_limit:
                                _sc_host_embd_without_draft = (
                                    self._metal_demand_paged_embedding_bytes(
                                        model_path,
                                        extra_args,
                                        binary = binary,
                                        requested_load_mode = load_mode,
                                        supports_load_mode = load_mode_support(server_caps),
                                        settings = _mem_settings,
                                        layers_fixed = _caller_owns_budget or _paravirtual_cpu_forced,
                                        mtp_may_engage = False,
                                    )
                                )
                                _sc_target_only_mib = _sc_target_mib - (
                                    self._MTP_DRAFT_COMPUTE_BYTES
                                    + _sc_host_embd_without_draft
                                    - _apple_host_embd
                                ) / (1024 * 1024)
                            if _sc_target_only_mib <= _apple_budget_mib < _sc_with_draft_mib or (
                                _sc_wired_mib > 0
                                and _sc_target_only_mib <= _sc_wired_mib < _sc_with_draft_mib
                            ):
                                logger.info(
                                    "Auto: the %s sidecar does not fit in unified memory "
                                    "alongside the model, so it is dropped.",
                                    "DSpark" if _mtp_effective == "dspark" else "DFlash",
                                )
                                _spec_dropped_no_vram = True
                                _mtp_will_engage = False
                                _mtp_reserves_gpu = False
                                mtp_overhead_fn = None
                                _mtp_kv_unsized = False
                                # Derived after the discrete drop: undo them, or Auto keeps a
                                # smaller context than Off.
                                _mtp_gpu_bytes_remain = False
                                _flat_mtp_engages = False
                                _flat_mtp_reserve = 0.0
                                _pin_fraction = _vram_frac
                                _apple_fit_budget_mib = _apple_budget_mib
                                _soft_overhead -= self._MTP_DRAFT_COMPUTE_BYTES
                                model_size_fit -= self._MTP_DRAFT_COMPUTE_BYTES
                                _apple_embd_logged = bool(_apple_host_embd)
                                _apple_host_embd = _sc_host_embd_without_draft
                                _apple_model_size_fit = model_size_fit - _apple_host_embd
                                if _apple_host_embd and not _apple_embd_logged:
                                    logger.info(
                                        "Metal fit: %.2f GB of input embeddings stay in the CPU "
                                        "file mapping; charging %.2f GB of weights to unified "
                                        "memory.",
                                        _apple_host_embd / (1024**3),
                                        _apple_model_size_fit / (1024**3),
                                    )

                        if self._can_estimate_kv():
                            cap = _apple_ctx_fit(native_ctx_for_cap, _FIT_MIN_CTX)
                            _cap_footprint_mib = _apple_footprint_mib(cap)
                            # Fit returns the request unchanged when it fits or weights exceed budget; only the latter over-commits, so floor to _FIT_MIN_CTX.
                            if _cap_footprint_mib <= _apple_fit_budget_mib:
                                max_available_ctx = cap
                                _apple_measured_ceiling = cap
                            else:
                                # Weights fit but _FIT_MIN_CTX may not: re-price under the floor
                                # before concluding nothing
                                # was measured (llama.cpp's own --fit stops at 4096 and cannot rescue it).
                                _floor_cap = _apple_ctx_fit(native_ctx_for_cap, _FIT_FLOOR_MIN_CTX)
                                # Ask the budget directly: both fits tie when native <= floor, and a tie reads as priced nothing and refuses nothing (native 256 hits it)
                                _weights_over_budget = (
                                    _apple_model_size_fit / (1024 * 1024)
                                ) > _apple_fit_budget_mib
                                if _weights_over_budget:
                                    # Weights alone miss the budget: floor for the UI but do not
                                    # refuse on an unvouched number.
                                    max_available_ctx = min(_metal_floor_ctx, native_ctx_for_cap)
                                elif _apple_footprint_mib(_floor_cap) <= _apple_fit_budget_mib:
                                    max_available_ctx = _floor_cap
                                    _apple_measured_ceiling = _floor_cap
                                else:
                                    # Nothing fits even at the smallest context: every explicit
                                    # request over-commits.
                                    max_available_ctx = min(_metal_floor_ctx, native_ctx_for_cap)
                                    _apple_nothing_fits = True
                        else:
                            # No KV estimate: floor to _FIT_MIN_CTX rather than launch at native, which would over-commit memory
                            max_available_ctx = min(_metal_floor_ctx, native_ctx_for_cap)
                            _apple_ctx_unmeasured = True
                        if not explicit_ctx:
                            effective_ctx = min(max_available_ctx, _APPLE_AUTO_CTX)
                        elif _apple_ctx_unmeasured:
                            # Never publish a ceiling below the launched context.
                            max_available_ctx = max(max_available_ctx, effective_ctx)
                            _unmeasured_ctx_notice = self._unmeasured_context_notice(
                                effective_ctx, cache_type_kv
                            )
                        elif (
                            self._can_estimate_kv()
                            and not _caller_owns_budget
                            and not _paravirtual_cpu_forced
                        ):
                            # Exempt manual caller-owned loads and the paravirtual CPU pin (no Metal memory, RAM priced elsewhere). Re-price past native; adopt only if budget vouches
                            if _apple_measured_ceiling is not None and (
                                effective_ctx > native_ctx_for_cap
                            ):
                                _extended_ceiling = _apple_ctx_fit(effective_ctx, _FIT_MIN_CTX)
                                if _apple_footprint_mib(_extended_ceiling) > _apple_fit_budget_mib:
                                    # _FIT_MIN_CTX is a floor, not a measurement: when native is below it, a refusal wrongly names the native-sized cap as the largest that fits
                                    _extended_ceiling = _apple_ctx_fit(
                                        effective_ctx, _FIT_FLOOR_MIN_CTX
                                    )
                                if (
                                    _extended_ceiling > _apple_measured_ceiling
                                    and _apple_footprint_mib(_extended_ceiling)
                                    <= _apple_fit_budget_mib
                                ):
                                    _apple_measured_ceiling = _extended_ceiling
                                    # Publish it: the UI reads max_context_length as the largest context that fits; left at native, an allowed load would show a does-not-fit warning
                                    max_available_ctx = max(
                                        max_available_ctx, _apple_measured_ceiling
                                    )
                            # Past free memory macOS swaps; past the wired limit the machine panics.
                            _apple_wired_mib = int(
                                _apple_wired_left_mib(self._apple_metal_wired_ceiling_bytes())
                                * max(0.0, 1.0 - _flat_mtp_reserve)
                            )
                            if (
                                _apple_wired_mib <= 0
                                or _apple_model_size_fit / (1024 * 1024) > _apple_wired_mib
                            ):
                                _metal_ctx_refusal = self._metal_context_overcommit_message(
                                    effective_ctx,
                                    _apple_measured_ceiling or 0,
                                    cache_type_kv,
                                    nothing_fits = _apple_nothing_fits,
                                )
                            else:
                                # Priced like the fit (SWA checkpoints included) so admission matches the refusal ceiling.
                                _requested_mib = _apple_footprint_mib(
                                    effective_ctx, _fit_ctx_checkpoints
                                )
                                if _requested_mib > _apple_wired_mib:
                                    _wired_ctx = _apple_ctx_fit(
                                        effective_ctx, _FIT_FLOOR_MIN_CTX, _apple_wired_mib
                                    )
                                    _wired_fits = (
                                        _apple_footprint_mib(_wired_ctx, _fit_ctx_checkpoints)
                                        <= _apple_wired_mib
                                    )
                                    _metal_ctx_refusal = self._metal_context_overcommit_message(
                                        effective_ctx,
                                        _wired_ctx if _wired_fits else 0,
                                        cache_type_kv,
                                        nothing_fits = not _wired_fits,
                                        wired_limit = True,
                                    )
                                elif _requested_mib > _apple_fit_budget_mib:
                                    _metal_ctx_warning = self._metal_context_pressure_message(
                                        effective_ctx, _requested_mib, _apple_fit_budget_mib
                                    )
                        # The paravirtual pin loads on CPU, so no Metal budget held anything.
                        _vram_fit_ctx = None if _paravirtual_cpu_forced else _apple_measured_ceiling
                        # The measured ceiling was priced without SWA checkpoints; re-fit with them.
                        if _vram_fit_ctx and (
                            _apple_footprint_mib(_vram_fit_ctx, _fit_ctx_checkpoints)
                            > _apple_fit_budget_mib
                        ):
                            _ckpt_cap = _apple_ctx_fit(_vram_fit_ctx, _FIT_FLOOR_MIN_CTX)
                            _vram_fit_ctx = (
                                _ckpt_cap
                                if _apple_footprint_mib(_ckpt_cap, _fit_ctx_checkpoints)
                                <= _apple_fit_budget_mib
                                else None
                            )
                        # Past the wired limit Metal refuses, so it is the harder ceiling when readable.
                        _fit_wired_mib = (
                            int(
                                _apple_wired_left_mib(self._apple_metal_wired_ceiling_bytes())
                                * max(0.0, 1.0 - _flat_mtp_reserve)
                            )
                            if _vram_fit_ctx
                            else 0
                        )
                        if (
                            _fit_wired_mib > 0
                            and _apple_footprint_mib(_vram_fit_ctx, _fit_ctx_checkpoints)
                            > _fit_wired_mib
                        ):
                            _wired_cap = _apple_ctx_fit(
                                _vram_fit_ctx, _FIT_FLOOR_MIN_CTX, _fit_wired_mib
                            )
                            _vram_fit_ctx = (
                                _wired_cap
                                if _apple_footprint_mib(_wired_cap, _fit_ctx_checkpoints)
                                <= _fit_wired_mib
                                else None
                            )
                            # Auto launches a fixed -c, so llama.cpp's fitter could only move
                            # layers to CPU; give up context first, down to the Auto floor,
                            # so every layer stays on Metal.
                            if (
                                not explicit_ctx
                                and _vram_fit_ctx
                                and min(_FIT_MIN_CTX, effective_ctx)
                                <= _vram_fit_ctx
                                < max_available_ctx
                            ):
                                max_available_ctx = _vram_fit_ctx
                                effective_ctx = min(effective_ctx, _vram_fit_ctx)
                        # Unmeasured floors still rely on llama.cpp's fitter.
                        _metal_full_offload_pinned = bool(
                            _apple_measured_ceiling is not None
                            and (
                                _apple_probe_embd
                                or (
                                    _apple_host_embd
                                    and not _paravirtual_cpu_forced
                                    and self._metal_placement_unset(
                                        extra_args, os.environ, layers_fixed = _caller_owns_budget
                                    )
                                )
                            )
                        )
                        # Weights over the budget launch here too and the fitter places
                        # them, so it must see the same wired limit the refusals do.
                        if not _caller_owns_budget and not _paravirtual_cpu_forced:
                            _metal_fit_target_floor_mib = self._metal_fit_target_floor_mib()
                            if _metal_fit_target_floor_mib > 0:
                                _metal_fit_target_floor_mib += _metal_media_wired_mib

                        if (
                            intent.refuse_partial_gpu_fit
                            and gpu_memory_mode != "manual"
                            and not _paravirtual_cpu_forced
                            # A CPU-only child takes no Metal memory; the host-RAM guard prices it.
                            and not _device_selection_is_cpu(extra_args, os.environ)
                        ):
                            # No discrete GPU row on Metal, so the partial-placement verdict
                            # cannot refuse this; CPU offload adds no unified memory.
                            budget_mib = _apple_fit_budget_mib
                            # Unknown headroom skips the wired cap, a known 0 is exhausted. The
                            # reserve matches the wired refusals: the footprint omits an unsized MTP draft.
                            wired_bytes = self._apple_metal_wired_headroom_bytes()
                            if wired_bytes is not None:
                                wired_mib = int(
                                    _apple_wired_left_mib(wired_bytes)
                                    * max(0.0, 1.0 - _flat_mtp_reserve)
                                )
                                budget_mib = min(budget_mib, wired_mib)
                            need_mib = math.ceil(
                                _apple_footprint_mib(effective_ctx, _fit_ctx_checkpoints)
                            )
                            if need_mib > budget_mib:
                                _metal_alongside_shortfall = (need_mib, budget_mib)

                    # Prefer fewer serving slots on GPU over --fit on offload: when the extra
                    # --parallel slots push the footprint past the pin budget, llama-server
                    # offloads layers to host and decode collapses ~3x (#6718). Retry the fit
                    # at fewer slots, keeping the largest count that stays fully on GPU and the
                    # chosen context. Skips tensor mode / Metal / KV-inestimable paths.
                    if (
                        use_fit
                        and n_parallel > 1
                        and gpus
                        and self._can_estimate_kv()
                        and effective_ctx > 0
                    ):
                        # Probe at _FIT_MIN_CTX, not _AUTO_OFFLOAD_CTX (a higher probe fails the slot reduction); must match _largest_ctx floor. Explicit contexts priced as given
                        _reduce_ctx = (
                            effective_ctx if explicit_ctx else min(effective_ctx, _FIT_MIN_CTX)
                        )
                        # Slot-independent footprint (folded compute buffer and the MTP
                        # reserve swapped out so the helper re-prices both per candidate).
                        _base_footprint = (
                            model_size_fit - _compute_buffer_pipeline + _cc_bytes(_reduce_ctx)
                        )
                        _gi_slots, _uf_slots, _slots = self._slots_that_fit_on_gpu(
                            n_parallel,
                            _reduce_ctx,
                            gpus,
                            total_by_idx,
                            _base_footprint,
                            cache_type_kv,
                            _pin_fraction,
                            _pipeline_overhead_bytes + _cc_bytes(_reduce_ctx),
                            _layer_min_gpus,
                            _effective_ubatch,
                            swa_full = swa_full,
                            kv_unified = planned_kv_unified,
                            flash_attn = planned_flash_attn,
                            split_extra_for_slots = lambda s: _cc_split_extra(_reduce_ctx, s),
                            ubatch_for_slots = _ubatch_for_slots,
                            mtp_bytes_for_slots = lambda s, ub: _mtp_bytes(_reduce_ctx, s, ub),
                            ctx_checkpoints = _placement_ctx_checkpoints,
                            # An explicit device pick skips the discrete tier; host-only accounting still applies.
                            integrated = _integrated_gpus,
                            prefer_discrete = bool(_tier_gpus),
                            discount_prefixes = _discount_prefixes,
                            discrete_host_only_bytes = _discrete_host_only(),
                            shared_gpu_ids = _shared_gpu_ids,
                        )
                        if not _uf_slots:
                            _slots_asked = n_parallel
                            gpu_indices, use_fit, n_parallel = _gi_slots, False, _slots
                            # Fewer slots lower the batch floor; re-derive the micro-batch, or the recorded _n_ubatch describes a graph never built
                            _effective_ubatch = _ubatch_for_slots(n_parallel)
                            # Re-fit at the final slot count; the KV and compute closures
                            # read the values just rebound above.

                            def _slots_hold(
                                ctx: int, devices: list[tuple[int, int]]
                            ) -> Optional[list[int]]:
                                """Return the GPU subset if the final slot count fits."""
                                _gi, _uf, _got = self._slots_that_fit_on_gpu(
                                    n_parallel,
                                    ctx,
                                    devices,
                                    total_by_idx,
                                    model_size_fit - _compute_buffer_pipeline + _cc_bytes(ctx),
                                    cache_type_kv,
                                    _pin_fraction,
                                    _pipeline_overhead_bytes + _cc_bytes(ctx),
                                    _layer_min_gpus,
                                    _effective_ubatch,
                                    swa_full = swa_full,
                                    kv_unified = planned_kv_unified,
                                    flash_attn = planned_flash_attn,
                                    split_extra_bytes = _cc_split_extra(ctx),
                                    ubatch_for_slots = _ubatch_for_slots,
                                    mtp_bytes_for_slots = (lambda s, ub, c = ctx: _mtp_bytes(c, s, ub)),
                                    include_requested = True,
                                    exact = True,  # only n_parallel is accepted below
                                    ctx_checkpoints = _placement_ctx_checkpoints,
                                    integrated = _integrated_gpus,
                                    prefer_discrete = bool(_tier_gpus),
                                    discount_prefixes = _discount_prefixes,
                                    discrete_host_only_bytes = _discrete_host_only(),
                                    shared_gpu_ids = _shared_gpu_ids,
                                )
                                return _gi if not _uf and _got == n_parallel else None

                            def _largest_ctx(
                                devices: list[tuple[int, int]],
                            ) -> Optional[tuple[int, list[int]]]:
                                """Largest 256-aligned context the cards hold; footprint grows monotonically with context, and a missing native length (0 sentinel) would search nothing"""
                                _native = self._context_length or _reduce_ctx
                                _lo = max(1, min(_FIT_MIN_CTX, _native) // 256)
                                _hi = _native // 256
                                _best = None
                                while _lo <= _hi:
                                    _mid = (_lo + _hi) // 2
                                    _held = _slots_hold(_mid * 256, devices)
                                    if _held is None:
                                        _hi = _mid - 1
                                    else:
                                        _best = (_mid * 256, _held)
                                        _lo = _mid + 1
                                return _best

                            # Only the chosen cards; searching all GPUs would reintroduce the layer split.
                            _kept = set(_gi_slots or ())
                            _plan_gpus = [g for g in gpus if g[0] in _kept]
                            _refit = _largest_ctx(_plan_gpus)
                            # Explicit contexts are never rewritten; otherwise the re-fit is the answer, up or down, since any other context was never priced for the final slots
                            if _refit is not None and not explicit_ctx:
                                if _refit[0] != effective_ctx:
                                    logger.info(
                                        "Context re-fitted %d -> %d for %d serving slot(s).",
                                        effective_ctx,
                                        _refit[0],
                                        n_parallel,
                                    )
                                effective_ctx, gpu_indices = _refit
                            # After the re-fit, so it names the context that launches
                            # rather than one sized for slots this load dropped.
                            logger.info(
                                "Serving slots reduced %d -> %d to keep the model on GPU "
                                "(avoid --fit offload) at context %d.",
                                _slots_asked,
                                n_parallel,
                                effective_ctx,
                            )
                            # Largest context across every card at the final slot count, matching best_cap; not a hardware max, since a larger request can reduce slots further
                            _ceiling = (
                                _refit if len(_plan_gpus) == len(gpus) else _largest_ctx(gpus)
                            )
                            if _ceiling is not None:
                                # Replace, not max(): the old anchor belongs to a superseded plan.
                                max_available_ctx = _ceiling[0]
                                _vram_fit_ctx = _ceiling[0]

                    # Single-GPU VRAM shortfall vs free less --fit-target (None = unpriceable).
                    _model_size_fit_cb = _compute_buffer_pipeline

                    def _fit_ctx_overridden(
                        child_extras: Optional[Iterable[str]], child_env: Mapping[str, str]
                    ) -> bool:
                        """A --fit-ctx / -fitc (argv, else LLAMA_ARG_FIT_CTX) the child
                        runs with that is not the value _ctx_integrity_flags emits."""
                        _studio_fitc: Optional[int] = None
                        if (
                            use_fit
                            and not (gpu_memory_mode == "manual" and gpu_layers >= 0)
                            and server_caps.get("supports_fit_ctx")
                        ):
                            if requested_ctx > 0 and effective_ctx > 0:
                                _studio_fitc = int(effective_ctx)
                            elif gpu_memory_mode == "manual" and gpu_layers < 0:
                                _studio_fitc = 8192
                        try:
                            _raw = _last_flag_value(child_extras, _FIT_CTX_FLAGS)
                        except ValueError:
                            return True
                        # The emitted flag beats the env; argv after it beats both.
                        if _raw is None and _studio_fitc is None:
                            _raw = child_env.get("LLAMA_ARG_FIT_CTX")
                        if _raw is None or not str(_raw).strip():
                            return False
                        try:
                            return int(str(_raw).strip()) != _studio_fitc
                        except ValueError:
                            return True

                    def _fitter_vram_shortfall() -> Optional[float]:
                        if not (gpus and effective_ctx > 0 and self._can_estimate_kv()):
                            return None
                        # Smart offload may still pin experts to CPU against Unsloth's budget.
                        from core.inference.offload_planner import smart_offload_enabled

                        if smart_offload_enabled(os.environ):
                            return None
                        # A drafter priced with a flat cushion has no draft-KV term here.
                        if _flat_mtp_engages:
                            return None
                        _on = [g for g in gpus if gpu_indices is None or g[0] in gpu_indices]
                        # Per-device placement (common/fit.cpp) can spill despite a pooled fit.
                        if len(_on) != 1:
                            return None
                        _child_extras = (
                            self._strip_device_extra_args(extra_args)
                            if _gpu_ids_own_device_flags
                            else extra_args
                        )
                        _child_env = dict(os.environ)
                        if _gpu_ids_own_device_flags:
                            self._clear_device_placement_env(_child_env)
                        self._reconcile_split_mode_env(_child_env, tensor_parallel = tensor_parallel)
                        if _pass_through_overrides_placement(
                            _child_extras,
                            _child_env,
                            len(_on),
                            is_vulkan_backend = is_vulkan_backend,
                            gpu_indices = gpu_indices,
                        ):
                            return None
                        # Priced at Unsloth's --parallel; any other pass-through one wins.
                        _np_override = _extra_args_n_parallel(_child_extras, {})
                        if _np_override is not None and _np_override != n_parallel:
                            return None
                        # A later pass-through -c 0 lets llama.cpp pick the context.
                        if parse_ctx_override(_child_extras) == 0:
                            return None
                        # A --fit-ctx Studio did not emit moves llama.cpp's fit.
                        if _fit_ctx_overridden(_child_extras, _child_env):
                            return None
                        # A GPU projector Studio never sized; Vision off keeps only audio-only.
                        _env_mmproj = _child_env.get("LLAMA_ARG_MMPROJ")
                        # The emitted --mmproj replaces an inherited path (argv after env).
                        if effective_is_vision:
                            _env_mmproj = None
                        # --mmproj-device / MTMD_BACKEND_DEVICE re-enable the GPU projector.
                        _mmproj_dev = _extra_args_device(
                            _child_extras, {"--mmproj-device", "-mmdev"}
                        )
                        if _mmproj_dev is None:
                            _mmproj_dev = _child_env.get("MTMD_BACKEND_DEVICE")
                        _mmproj_dev_gpu = bool(
                            _mmproj_dev is not None and str(_mmproj_dev).strip().lower() != "none"
                        )
                        if _resolved_mmproj_offload(_child_extras, _child_env) is False:
                            if _mmproj_dev_gpu:
                                return None
                        elif disable_vision:
                            if _mmproj_env_is_audio_only(_env_mmproj):
                                return None
                        elif (
                            _env_mmproj
                            or _child_env.get("LLAMA_ARG_MMPROJ_URL")
                            or (
                                not launch_mmproj_path
                                and extra_args_mmproj_auto(_child_extras, _child_env)
                            )
                        ):
                            return None
                        # --fit-target precedence: argv, Studio's emitted value, env, default.
                        _supports_fit_target = bool(server_caps.get("supports_fit_target"))
                        _auto_fit = gpu_memory_mode == "manual" and gpu_layers < 0
                        _managed_mib = self._fit_target_margin_mib(
                            auto_fit = _auto_fit,
                            fit_target_delta_mib = _fit_target_delta_mib,
                            supports_fit_target = _supports_fit_target,
                        )
                        # A per-device list reads as its largest entry (the larger shortfall).
                        _margin_mib = fit_target_margin_in(_child_extras or (), None)
                        # Studio's own per-card margins, when it spells the list itself.
                        _margins_by_idx: Optional[dict[int, float]] = None
                        if _margin_mib is None:
                            _emits = _supports_fit_target and (
                                _auto_fit or _managed_mib != _LLAMA_FIT_TARGET_DEFAULT_MIB
                            )
                            _env_mib = None if _emits else fit_target_margin_in((), _child_env)
                            # An inherited 0 is a valid margin, not "unset".
                            _margin_mib = _managed_mib if _env_mib is None else _env_mib
                            if _emits:
                                _margins_by_idx = self._fit_target_margins_mib(
                                    auto_fit = _auto_fit,
                                    fit_target_delta_by_idx = _fit_target_delta_by_idx,
                                    supports_fit_target = _supports_fit_target,
                                )
                        # A lazily read table stays mmap'd on the host, never on the GPU.
                        _child_order, _child_mask = self._planned_child_gpu_order(
                            gpu_ids = gpu_ids,
                            gpu_indices = gpu_indices,
                            tp_tensor_split = tp_tensor_split,
                            extra_args = _child_extras,
                            env = _child_env,
                            is_vulkan_backend = is_vulkan_backend,
                            cpu_forced = _paravirtual_cpu_forced or _arch_gate_forced_cpu,
                        )
                        _lazy_host = self._lazy_read_host_bytes(
                            model_path,
                            caps = server_caps,
                            extra_args = _child_extras,
                            env = _child_env,
                            auto_resolves_off = self._lazy_auto_resolves_off(
                                gpu_indices = _child_order,
                                detected_gpus = _detected_gpus,
                                shared_gpu_ids = _shared_gpu_ids,
                                is_vulkan_backend = is_vulkan_backend,
                                extra_args = _child_extras,
                                env = _child_env,
                                tensor_parallel = tensor_parallel,
                                integrated_gpu_ids = _vulkan_igpu_ids,
                                child_gpu_order = _child_mask,
                            ),
                        )
                        _need = (
                            model_size_fit
                            - max(0, _lazy_host - _host_input_embd)
                            - _model_size_fit_cb
                            + _compute_buffer_pipeline
                            + (
                                _kv_bytes(effective_ctx)
                                if _kv_offload_from_args(_child_extras, _child_env)
                                else 0
                            )
                            + (
                                _mtp_bytes(effective_ctx, n_parallel, _effective_ubatch)
                                if _mtp_will_engage
                                else 0
                            )
                            + _cc_bytes(effective_ctx, len(_on))
                        )
                        _free = sum(
                            max(0.0, free - _margin_mib_for(_margins_by_idx or _margin_mib, _idx))
                            for _idx, free in _on
                        )
                        return _need - _free * 1024 * 1024

                    def _moe_resident_share_now() -> Optional[float]:
                        if self.n_moe_layers <= 0:
                            return None
                        try:
                            _short = _fitter_vram_shortfall()
                            if _short is None:
                                return None
                            _scan = self._gguf_tensor_scan(model_path) or (None, {}, 0)
                            return _moe_resident_expert_share(_short, _scan[2])
                        except Exception as e:
                            logger.debug(f"MoE resident share unpriceable ({e})")
                            return None

                    # MoE experts in host RAM: a larger micro-batch amortises the expert
                    # copies of prompt processing (see _MOE_SPILL_N_UBATCH). The placement
                    # verdict above was priced at the default micro-batch, so a model that
                    # fits keeps today's launch. This runs before every term that reads the
                    # micro-batch (MTP reserve, KV, compute buffer, spill planner, load mode),
                    # so the larger buffer is priced: the spill planner keeps fewer experts on
                    # the GPU, and llama.cpp's fitter measures a --fit on launch at the emitted
                    # value. Manual pinned layers (--fit off) are the user's placement, where an
                    # unpriced larger buffer could OOM, so they are left alone. A GPU-only -ot
                    # does not count for the same reason. A user --device none / cpu runs on
                    # the CPU with nothing to stream, like the CPU fallback.
                    _moe_experts_on_host = bool(
                        not (gpu_memory_mode == "manual" and gpu_layers >= 0)
                        and not intent.cpu_fallback
                        and not _device_selection_is_cpu(extra_args, os.environ)
                        and (use_fit or _expert_spill_places_tensors_on_cpu(extra_args, os.environ))
                    )
                    _moe_spill_discrete_gpu = bool(
                        _moe_experts_on_host
                        and self._discrete_gpu_for_expert_spill(
                            gpu_indices, _detected_gpus, _shared_gpu_ids
                        )
                    )
                    _spill_n_batch, _spill_n_ubatch = _moe_spill_batch_ubatch(
                        n_batch,
                        n_ubatch,
                        n_moe_layers = self.n_moe_layers,
                        experts_on_host = _moe_experts_on_host,
                        discrete_gpu = _moe_spill_discrete_gpu,
                        user_named_batch = _user_named_batch,
                    )
                    _moe_ub_switch = _moe_switch(_MOE_SPILL_UBATCH_ENV)
                    if _spill_n_ubatch != n_ubatch and _moe_ub_switch != "off":
                        _share_before = (
                            _moe_resident_share_now() if _moe_ub_switch == "auto" else None
                        )
                        _pre_raise = (
                            n_batch,
                            n_ubatch,
                            _effective_ubatch,
                            _compute_buffer_pipeline,
                            _compute_buffer_tensor,
                        )
                        n_batch, n_ubatch = _spill_n_batch, _spill_n_ubatch
                        _effective_ubatch = _ubatch_for_slots(n_parallel)
                        # Same calls and fallbacks as above, at the new micro-batch.
                        _compute_buffer_pipeline = self._estimate_compute_buffer_bytes(
                            n_ubatch = _effective_ubatch,
                            n_parallel = n_parallel,
                            per_device_tensor = False,
                        ) or (self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024)
                        _compute_buffer_tensor = self._estimate_compute_buffer_bytes(
                            n_ubatch = _effective_ubatch,
                            n_parallel = n_parallel,
                            per_device_tensor = True,
                        ) or (self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024)
                        _share_after = (
                            _moe_resident_share_now()
                            if _moe_ub_switch == "auto" and _share_before is not None
                            else None
                        )
                        if _moe_spill_raise_keeps_experts(
                            _share_before, _share_after, switch = _moe_ub_switch
                        ):
                            logger.info(
                                "MoE experts are offloaded to system RAM; raising "
                                "--ubatch-size %s -> %s for faster prompt processing.",
                                (
                                    _pre_raise[1]
                                    if _pre_raise[1] is not None
                                    else _DEFAULT_LLAMA_N_UBATCH
                                ),
                                _spill_n_ubatch,
                            )
                            self._moe_spill_batch_restore = (_pre_raise[0], _pre_raise[1])
                        else:
                            (
                                n_batch,
                                n_ubatch,
                                _effective_ubatch,
                                _compute_buffer_pipeline,
                                _compute_buffer_tensor,
                            ) = _pre_raise
                            logger.info(
                                "MoE experts: keeping the default --ubatch-size; %s would "
                                "move the GPU's resident share of the routed experts "
                                "%.0f%% -> %.0f%%.",
                                _spill_n_ubatch,
                                100 * (_share_before or 0.0),
                                100 * (_share_after or 0.0),
                            )

                    _moe_cache_candidate = self._moe_cache_auto_eligible(
                        caps = server_caps,
                        gpu_memory_mode = gpu_memory_mode,
                        use_fit = use_fit,
                        experts_on_host = _moe_experts_on_host,
                        discrete_gpu = _moe_spill_discrete_gpu,
                        gpu_indices = gpu_indices,
                        detected_gpus = _detected_gpus,
                        is_vulkan_backend = is_vulkan_backend,
                        tensor_parallel = tensor_parallel,
                        extra_args = extra_args,
                        env = os.environ,
                    )

                    # Pass the final slot and micro-batch values instead of the defaults
                    # captured before slot reduction.
                    _mtp_reserve_bytes = (
                        _mtp_bytes(effective_ctx, n_parallel, _effective_ubatch)
                        if _mtp_will_engage
                        else 0
                    )
                    # DSpark/DFlash borrow the target's token_embd/output, so llama.cpp's fitter
                    # cannot measure the draft
                    # (common/fit.cpp "fitting without it"); carry the planner's reserve in the
                    # child's --fit-target.
                    _sidecar_kind_for_fit = (
                        _mtp_effective if _mtp_effective in ("dspark", "dflash") else None
                    )
                    if _extra_args_set_spec_type(extra_args):
                        _child_spec_types = _accumulated_spec_types(extra_args, env = _spec_env)
                        _sidecar_kind_for_fit = (
                            "dspark"
                            if "draft-dspark" in _child_spec_types
                            else ("dflash" if "draft-dflash" in _child_spec_types else None)
                        )
                    _fit_device_count = _sidecar_planned_device_count()
                    _draft_on_target_device = _draft_on_target_device_now()
                    if (
                        _mtp_will_engage
                        and _mtp_reserves_gpu
                        and _sidecar_kind_for_fit is not None
                        and _draft_on_target_device
                        # Builds whose fitter measures borrowed-tensor drafts need no reserve.
                        and not server_caps.get("fit_measures_borrowed_drafts")
                        and _sidecar_borrows_target_tensors(_mtp_draft_for_budget or mtp_draft_path)
                    ):
                        # Priced even under --fit off: a startup-OOM retry turns the fitter on with
                        # the drafter loaded.
                        _sidecar_auto_ctx = effective_ctx <= 0
                        _sidecar_reserve_ctx = (
                            _sidecar_child_ctx(
                                extra_args,
                                os.environ,
                                self._context_length,
                                n_parallel,
                                planned_kv_unified,
                            )
                            if _sidecar_auto_ctx
                            else effective_ctx
                        )
                        _sidecar_compute_mib = math.ceil(
                            self._MTP_DRAFT_COMPUTE_BYTES / (1024 * 1024)
                        )
                        _sidecar_reserve_path = _mtp_draft_for_budget or mtp_draft_path
                        try:
                            _sidecar_draft_layers = getattr(
                                self._draft_backend_for(_sidecar_reserve_path), "_n_layers", None
                            )
                        except Exception:  # noqa: BLE001 - an unread drafter shares by memory alone
                            _sidecar_draft_layers = None
                        _sidecar_owns_output = not _sidecar_borrows_output(_sidecar_reserve_path)
                        # The fitter measures the target's rollback copies itself.
                        _sidecar_only_bytes = _mtp_bytes(
                            _sidecar_reserve_ctx, n_parallel, _effective_ubatch
                        )
                        if _target_rollback and _mtp_eff_n_max:
                            _sidecar_only_bytes -= (
                                self._rollback_state_bytes(n_parallel) * _mtp_eff_n_max
                            )
                        # An unsized sidecar carries the planner's flat fraction, or the fitter
                        # spends it on target layers.
                        _sidecar_flat_mib = 0
                        if _flat_mtp_reserve > 0:
                            _flat_pool_mib = (
                                _apple_budget_mib
                                if _apple_budget_mib > 0
                                else max(
                                    (
                                        free
                                        for idx, free in (gpus or _detected_gpus)
                                        if gpu_indices is None or idx in gpu_indices
                                    ),
                                    default = 0,
                                )
                            )
                            _sidecar_flat_mib = math.ceil(
                                _flat_mtp_reserve * max(0, _flat_pool_mib)
                            )
                        _sidecar_fit_reserve_eligible_mib = (
                            self._sidecar_fit_reserve_mib(
                                max(0, _sidecar_only_bytes),
                                _sidecar_reserve_path,
                                flat_mib = _sidecar_flat_mib,
                            )
                            + _sidecar_compute_mib
                        )

                        def _sidecar_child_on_discrete_gpus(
                            run_args: List[str], child_env: Mapping[str, str]
                        ) -> bool:
                            """Whether every card the drafter runs on is discrete; positive evidence
                            only."""
                            if _apple_budget_mib:
                                return False
                            active = gpu_indices
                            if (
                                not is_vulkan_backend
                                and _child_gpu_physical_ids is not None
                                and _child_gpu_physical_ids is not _parent_view_gpu_ids
                            ):
                                active = list(_child_gpu_physical_ids)
                                if not active:
                                    return False
                            override = self._effective_device_ids(run_args, child_env)
                            if override is not None:
                                if not override or any(
                                    d.lower() in _CPU_DEVICE_VALUES for d in override
                                ):
                                    return False
                                # This backend's own names only: Vulkan0 is not CUDA0.
                                if not all(
                                    (
                                        re.fullmatch(r"(?i)vulkan\d+", d)
                                        if is_vulkan_backend
                                        else _CUDA_LIKE_DEVICE_RE.match(d)
                                    )
                                    for d in override
                                ):
                                    return False
                                studio_pin = (
                                    [f"vulkan{i}" for i in _vulkan_pin_ids]
                                    if is_vulkan_backend and _vulkan_pin_ids
                                    else None
                                )
                                if [d.lower() for d in override] != studio_pin:
                                    listed = {
                                        d.lower()
                                        for d in (
                                            self._enumerated_gpu_devices(binary, child_env) or []
                                        )
                                    }
                                    if not {d.lower() for d in override} <= listed:
                                        return False
                            seen = {idx for idx, _free in (_detected_gpus or ())}
                            shared = set(_shared_gpu_ids or ())
                            if (
                                is_vulkan_backend
                                and override
                                and not {int(d[6:]) for d in override} <= seen
                            ):
                                # Vulkan Auto drops its iGPUs from the probe.
                                return False
                            if (
                                seen
                                and not (seen & shared)
                                and self._discrete_gpu_for_expert_spill(
                                    None, _detected_gpus, _shared_gpu_ids
                                )
                            ):
                                return True
                            # A mixed host: narrow only through an order the child provably shares.
                            if not is_vulkan_backend and (
                                self._positional_child_order_unproven(child_env)
                                or any(
                                    not all(
                                        t.strip().isdigit()
                                        for t in str(child_env.get(name)).split(",")
                                        if t.strip()
                                    )
                                    for name in (
                                        "CUDA_VISIBLE_DEVICES",
                                        "HIP_VISIBLE_DEVICES",
                                        "ROCR_VISIBLE_DEVICES",
                                    )
                                    if child_env.get(name) is not None
                                )
                            ):
                                return False
                            active, compact = self._effective_gpu_selection(
                                binary, child_env, active, run_args, is_vulkan = is_vulkan_backend
                            )
                            pos = _draft_pin_device_index(
                                _extra_args_draft_device_pin(run_args),
                                ",".join(override) if override else None,
                                len(compact or active or _detected_gpus or ()),
                            )
                            if (
                                pos is None
                                and _extra_args_draft_device_pin(run_args) is None
                                and _split_mode_confines_to_one_device(run_args, child_env)
                            ):
                                main = self._effective_main_gpu(run_args, child_env)
                                pos = 0 if main is None else (main if override else None)
                            if pos is not None:
                                if override:
                                    if not (compact and 0 <= pos < len(compact)):
                                        return False
                                    active = self._physical_ordinals(
                                        [compact[pos]], child_env, is_vulkan_backend
                                    )
                                elif is_vulkan_backend or not self._positional_child_order_unproven(
                                    child_env
                                ):
                                    active = self._physical_ordinals(
                                        [pos], child_env, is_vulkan_backend
                                    )
                            on = set(active) if active else seen
                            if not on or not on <= seen - shared:
                                return False
                            return self._discrete_gpu_for_expert_spill(
                                sorted(on), _detected_gpus, _shared_gpu_ids
                            )

                        def _sidecar_reserve_reprice(
                            run_args: List[str],
                            child_env: Mapping[str, str],
                            _ctx: int = _sidecar_reserve_ctx,
                            _n: int = _mtp_eff_n_max,
                            _dp: Optional[str] = _sidecar_reserve_path,
                            _w: int = _mtp_draft_weights,
                            _np: int = n_parallel,
                            _mtp: bool = _engaged_is_mtp,
                            _swa_full: bool = swa_full,
                            _kv_unified: bool = planned_kv_unified,
                            _ub: Optional[int] = _effective_ubatch,
                            _floor: int = _sidecar_compute_mib,
                            _flat: int = _sidecar_flat_mib,
                            _native: Optional[int] = self._context_length,
                        ) -> int:
                            # Read off the argv: a --flash-attn off respawn pads the draft cache to
                            # f16, extras can change slots.
                            _ck, _cv = _extra_args_draft_cache_types(run_args, child_env)
                            _np = _extra_args_n_parallel(run_args, child_env) or _np
                            _kv_unified = _kv_unified_from_args(
                                run_args, default = _kv_unified, env = child_env
                            )
                            try:
                                _ctx = _sidecar_child_ctx(
                                    run_args, child_env, _native, _np, _kv_unified
                                )
                            except ValueError:
                                pass  # a malformed -c is the launch's error to report, not this
                            # llama_context caps the micro-batch at the batch, and a causal batch at
                            # the context.
                            _ub = _capped_draft_n_ubatch(run_args, child_env, _ctx) or _ub
                            if _ub and _ctx and _ctx > 0:
                                _ub = min(_ub, int(_ctx))
                            _kv_kw = dict(
                                drafter_path = _dp,
                                draft_cache_type_k = _ck,
                                draft_cache_type_v = _cv,
                                n_parallel = _np,
                                swa_full = _swa_full,
                                kv_unified = _kv_unified,
                                n_ubatch = _ub,
                                flash_attn = _flash_attn_enabled_from_args(
                                    run_args, default = planned_flash_attn, env = child_env
                                ),
                            )
                            _v = self._estimate_mtp_overhead_bytes(
                                _ctx,
                                spec_draft_n_max = _n,
                                draft_weights_bytes = _w,
                                mtp_keeps_target_ctx = _mtp,
                                target_rollback = False,
                                **_kv_kw,
                            )
                            # The draft context copies -nkvo, so its KV stays in host buffers.
                            if (
                                _v
                                and not _kv_offload_from_args(run_args, child_env)
                                and _sidecar_child_on_discrete_gpus(run_args, child_env)
                            ):
                                _host_kv = self._mtp_draft_kv_bytes(_ctx, **_kv_kw)
                                if _host_kv:
                                    _v = max(0, _v - _host_kv)
                            try:
                                _sidecar_reserve_graph_mib[0] = math.ceil(
                                    self._mtp_draft_compute_bytes(
                                        _ctx,
                                        drafter_path = _dp,
                                        n_ubatch = _ub,
                                        n_parallel = _np,
                                        spec_draft_n_max = _n,
                                    )
                                    / (1024 * 1024)
                                )
                            except Exception:  # noqa: BLE001 - unsized: the shares carry it
                                _sidecar_reserve_graph_mib[0] = 0
                            return (
                                self._sidecar_fit_reserve_mib(_v or 0, _dp, flat_mib = _flat) + _floor
                            )

                        if _draft_share_unknown:
                            # Drafter partly off the card: its weight share is unpriced, keep only
                            # the compute floor.
                            _sidecar_fit_reserve_eligible_mib = _sidecar_compute_mib
                            _sidecar_reserve_reprice = None
                        # A scalar --fit-target suits only a single-device launch; priced anyway,
                        # since a retry can narrow to one card.
                        if use_fit and (_apple_budget_mib > 0 or _fit_device_count == 1):
                            _sidecar_fit_reserve_mib = _sidecar_fit_reserve_eligible_mib
                    if _mtp_will_engage:
                        _mtp_note = self._mtp_reserve_note(
                            _mtp_reserve_bytes,
                            n_ctx = effective_ctx,
                            n_parallel = n_parallel or 1,
                            n_ubatch = _effective_ubatch,
                            n_max = _mtp_eff_n_max,
                            target_rollback = _target_rollback,
                            flat_fallback = mtp_overhead_fn is None,
                            reprice = lambda slots, ub: _mtp_bytes(effective_ctx, slots, ub),
                        )
                    else:
                        _mtp_note = ""

                    if effective_ctx < original_ctx:
                        kv_est = _kv_bytes(effective_ctx)
                        logger.info(
                            f"Context auto-reduced: {original_ctx} -> {effective_ctx} "
                            f"(model: {model_size / (1024**3):.1f} GB, "
                            f"est. KV cache: {kv_est / (1024**3):.1f} GB, "
                            f"{_mtp_note}".rstrip(", ")
                            + ")"
                        )

                    kv_cache_bytes = _kv_bytes(effective_ctx)
                    # Decide before the log line so it matches the argv.
                    if gpu_memory_mode == "manual" and gpu_layers >= 0:
                        use_fit = False
                    # Separate GPU drafter is a second model; None once CPU-offloaded (_mtp_draft_for_budget nulled).
                    _spill_inputs = {
                        "model_size": model_size,
                        "kv_cache_bytes": kv_cache_bytes,
                        "gpus": list(gpus or ()),
                        # Usable budget per device, not raw free_mib (would under-spill then pin
                        # with --fit off).
                        # _pin_fraction keeps the unsized-drafter cushion free.
                        "gpu_usable_mib": {
                            _idx: max(0.0, _gpu_usable((_idx, _free), _pin_fraction))
                            for _idx, _free in (gpus or ())
                        },
                        "gpu_indices": gpu_indices,
                        "soft_overhead": _soft_overhead,
                        # The planner rebuilds from the GGUF tensor table, so projector and MTP are absent.
                        "extra_gpu_bytes": (mmproj_size + _shared_pool_mmproj + _mtp_reserve_bytes),
                        # Not double-charged; omitting it would OOM in graph_reserve under --fit off.
                        "compute_buffer_flat": int(_compute_buffer_pipeline),
                        "ctx_compute_per_device": (
                            _cc_bytes(effective_ctx, max(1, len(gpus or ())))
                            // max(1, len(gpus or ()))
                        ),
                        # An engaged draft of any kind makes nextn/MTP blocks resident in the target load, so charge them back; over-charging only spills more
                        "mtp_will_engage": bool(_mtp_will_engage),
                        # A SEPARATE drafter that stays on the GPU: a second model,
                        # whose weights and KV llama.cpp distributes like any other.
                        # None once the drafter is CPU-offloaded (_mtp_draft_for_budget
                        # is nulled there), which is the case that really is off-GPU.
                        "separate_draft_on_gpu": bool(_mtp_will_engage and _mtp_draft_for_budget),
                        # Claims no placement (llama.cpp enumerates devices); _detected_gpus, since manual modes empty gpus.
                        "model_path": model_path,
                        "n_ctx": effective_ctx,
                        "n_threads": n_threads,
                        # Slot count the KV and recurrent state were priced at; a pass-through --parallel is appended last and wins, and both caches scale with it
                        "n_parallel": int(n_parallel or 1),
                        # Only ratios are used (scaled to kv_cache_bytes) to show where big caches land; must use the same launch settings kv_cache_bytes was priced at
                        "kv_layer_weights": self._kv_layer_weights(
                            effective_ctx,
                            swa_full = swa_full,
                            n_parallel = n_parallel,
                            kv_unified = planned_kv_unified,
                            n_ubatch = _effective_ubatch,
                            flash_attn = planned_flash_attn,
                            cache_type_kv = cache_type_kv,
                        ),
                        # An iGPU or APU reports host RAM as VRAM; do not count it twice.
                        "shared_gpu_ids": set(_shared_gpu_ids or ()),
                    }
                    mmproj_note = (
                        f"mmproj: {mmproj_size / (1024**3):.1f} GB, " if mmproj_size else ""
                    )
                    logger.info(
                        f"GGUF size: {gguf_size / (1024**3):.1f} GB, "
                        f"{mmproj_note}"
                        f"est. KV cache: {kv_cache_bytes / (1024**3):.1f} GB, "
                        f"{_mtp_note}"
                        f"context: {effective_ctx}, "
                        # Reports the --fit flag state, not whether the model fits: off means this subset provably fits; manual modes empty gpus on purpose, not a failed probe
                        f"GPUs free: {gpus}"
                        f"{' (manual mode)' if not gpus and gpu_memory_mode == 'manual' else ''}, "
                        f"selected: {gpu_indices}, --fit: {'on' if use_fit else 'off'}"
                    )
                    _on = list(gpu_indices or [idx for idx, _ in gpus])
                    if _on:
                        self._pending_plan_mib = _gpu_plan_mib(
                            max(1, gguf_size - _host_input_embd) + mmproj_size + kv_cache_bytes,
                            _on,
                            gpus,
                        )
                    if (
                        not gpus
                        and not _detected_gpus
                        and not gpu_ids
                        and not _arch_gate_forced_cpu
                        and not _metal_capable_host()
                    ):
                        # Claims no placement: llama.cpp enumerates devices itself.
                        # _detected_gpus, not gpus: Manual modes empty gpus on purpose.
                        # gpu_ids is pinned for the child below; the arch gate warns better.
                        logger.warning(
                            "Unsloth could not enumerate any GPU, so this load was planned "
                            "without device information: %s",
                            LlamaCppBackend._explain_empty_gpu_probe(binary),
                        )
                    # Set last, so any early return or raise leaves it False.
                    _placement_verdict_partial = bool(_detected_gpus) and bool(use_fit)

                    # Load mode: mmap demand-pages what RAM cannot hold, so a fitting load wants none. Use gpus not _detected_gpus: arch gate empties it on --device none
                    _fit_devices = max(1, len(gpu_indices or ()) or len(gpus))
                    # A CPU-pinned drafter is still in host RAM; None means abstain.
                    _cpu_draft_fit_bytes = self._cpu_resident_draft_bytes(
                        effective_ctx,
                        drafter_path = _cpu_draft_path,
                        draft_cache_type_k = _mtp_draft_ck,
                        draft_cache_type_v = _mtp_draft_cv,
                        n_parallel = n_parallel,
                        swa_full = swa_full,
                        kv_unified = planned_kv_unified,
                        n_ubatch = _effective_ubatch,
                        flash_attn = planned_flash_attn,
                    )
                    # Classify on what the child gets: a gpu_ids pin drops device flags and env twins; unpinned, the child inherits LLAMA_ARG_DEVICE verbatim
                    _fit_extras = (
                        self._strip_device_extra_args(extra_args)
                        if _gpu_ids_own_device_flags
                        else extra_args
                    )
                    _fit_env = dict(os.environ)
                    if gpu_memory_mode == "manual":
                        self._clear_manual_placement_env(_fit_env)
                    if _gpu_ids_own_device_flags:
                        self._clear_device_placement_env(_fit_env)
                    self._reconcile_split_mode_env(_fit_env, tensor_parallel = tensor_parallel)
                    # What the placement branch below will emit, so the effective
                    # (last-wins) fitter state is readable before the final argv exists.
                    # Every reachable branch is fixed by use_fit.
                    _fitter_runs = fit_is_effectively_on(
                        ["--fit", "on" if use_fit else "off", *(_fit_extras or [])],
                        _fit_env,
                    )
                    # Drafter split card/host by fixed -ngld is unsized, so abstain: llama.cpp keeps layers past the count in host RAM. Pass-through --fit-target wins
                    _draft_split_across_host = bool(
                        _mtp_draft_for_budget
                        and _draft_is_split_across_host(
                            _fit_extras,
                            _fit_env,
                            n_draft_layers = getattr(
                                self._draft_backend_for(_mtp_draft_for_budget),
                                "_n_layers",
                                None,
                            ),
                        )
                    )
                    _fit_margin_mib = (
                        max(
                            self._fit_target_margin_mib(
                                auto_fit = (gpu_memory_mode == "manual" and gpu_layers < 0),
                                fit_target_delta_mib = _fit_target_delta_mib if use_fit else 0.0,
                                supports_fit_target = use_fit
                                and bool(server_caps.get("supports_fit_target")),
                                floor_mib = _metal_fit_target_floor_mib,
                            ),
                            fit_target_margin_in(_fit_extras, _fit_env) or 0.0,
                        )
                        if _fitter_runs
                        else 0.0
                    )
                    # Studio's per-card list replaces the single value only when Studio
                    # owns the margin: a pass-through one wins outright (above).
                    _fit_margins_by_idx = (
                        self._fit_target_margins_mib(
                            auto_fit = (gpu_memory_mode == "manual" and gpu_layers < 0),
                            fit_target_delta_by_idx = _fit_target_delta_by_idx,
                            supports_fit_target = bool(server_caps.get("supports_fit_target")),
                        )
                        if _fitter_runs
                        and use_fit
                        and not _metal_fit_target_floor_mib
                        and fit_target_margin_in(_fit_extras, _fit_env) is None
                        else None
                    )
                    # A projector Unsloth never resolved but the child loads anyway:
                    # LLAMA_ARG_MMPROJ / _URL are read before argv and only the vision
                    # switch and the paravirtual pin clear them, while model_size carries
                    # none of those bytes. A URL names a download that has not happened
                    # (and outranks even the --mmproj this launch emits), and an
                    # unreadable path cannot be sized, so both abstain.
                    _fit_env_mmproj_scrubbed = bool(
                        _pv_mmproj_unpinnable or _paravirtual_cpu_forced
                    )
                    _fit_env_mmproj = (
                        ""
                        if (
                            launch_mmproj_path
                            or _fit_env_mmproj_scrubbed
                            or (
                                disable_vision
                                # The scrub's own audio-only reprieve: that projector
                                # survives the switch, so it is loaded.
                                and not _mmproj_env_is_audio_only(_fit_env.get("LLAMA_ARG_MMPROJ"))
                            )
                        )
                        else (_fit_env.get("LLAMA_ARG_MMPROJ") or "").strip()
                    )
                    _fit_env_mmproj_url = (
                        ""
                        if (_fit_env_mmproj_scrubbed or disable_vision)
                        else (_fit_env.get("LLAMA_ARG_MMPROJ_URL") or "").strip()
                    )
                    _fit_env_mmproj_bytes = self._mmproj_vram_bytes(_fit_env_mmproj or None)
                    _fit_env_mmproj_unsized = bool(_fit_env_mmproj_url) or (
                        bool(_fit_env_mmproj) and _fit_env_mmproj_bytes <= 0
                    )
                    # Ungated like _subset_model_size: CUDA context and scratch exist regardless of the pipeline.
                    _fit_env_mmproj_on_host = (
                        _resolved_mmproj_offload(_fit_extras, _fit_env) is False
                    )
                    _fit_model_size = model_size
                    if _fit_env_mmproj_unsized:
                        # Same abstain the other unreadable terms take.
                        _fit_model_size = None
                    elif _fit_model_size and not _fit_env_mmproj_on_host:
                        _fit_model_size += _fit_env_mmproj_bytes
                    _fit_soft_overhead = _soft_overhead + self._inherited_mmproj_soft_overhead(
                        _fit_env_mmproj_bytes,
                        on_host = _fit_env_mmproj_on_host,
                    )
                    # Pass the inherited projector to the spill planner, or its VRAM goes unpaid: the plan is pinned with --fit off, so llama.cpp cannot absorb it
                    _spill_inputs["env_mmproj_unsized"] = _fit_env_mmproj_unsized
                    _spill_inputs["host_mmproj_bytes"] = _mmproj_pinned_bytes + (
                        _fit_env_mmproj_bytes if _fit_env_mmproj_on_host else 0
                    )
                    _spill_inputs["env_mmproj_bytes"] = (
                        0
                        if _fit_env_mmproj_on_host
                        else _fit_env_mmproj_bytes
                        + self._inherited_mmproj_soft_overhead(_fit_env_mmproj_bytes, on_host = False)
                    )
                    # Derive the host-only share from the same estimator as the total.
                    _effective_ctx_checkpoints = _ctx_checkpoints_for_final_slots()
                    _ckpt_host_bytes = max(
                        0,
                        _kv_bytes(effective_ctx, _effective_ctx_checkpoints)
                        - _kv_bytes(effective_ctx, 0),
                    )
                    # A lazily read table stays mmap'd under every load mode: not pinned RAM.
                    # An empty inventory beside a GPU may hide an iGPU, where auto is off: priced
                    # resident.
                    _fit_child_order, _fit_child_mask = self._planned_child_gpu_order(
                        gpu_ids = gpu_ids,
                        gpu_indices = gpu_indices,
                        tp_tensor_split = tp_tensor_split,
                        extra_args = _fit_extras,
                        env = _fit_env,
                        is_vulkan_backend = is_vulkan_backend,
                        cpu_forced = _paravirtual_cpu_forced or _arch_gate_forced_cpu,
                    )
                    _fit_lazy_auto_off = (
                        not _detected_gpus
                        and (is_vulkan_backend or _gpu_device_node_present())
                        and not self._child_runs_on_cpu(
                            binary,
                            _fit_extras,
                            _fit_env,
                            cpu_forced = _paravirtual_cpu_forced or _arch_gate_forced_cpu,
                            tensor_parallel = tensor_parallel,
                        )
                    ) or self._lazy_auto_resolves_off(
                        gpu_indices = _fit_child_order,
                        detected_gpus = _detected_gpus,
                        shared_gpu_ids = _shared_gpu_ids,
                        is_vulkan_backend = is_vulkan_backend,
                        extra_args = _fit_extras,
                        env = _fit_env,
                        tensor_parallel = tensor_parallel,
                        integrated_gpu_ids = _vulkan_igpu_ids,
                        child_gpu_order = _fit_child_mask,
                    )
                    _fit_lazy_bytes = self._lazy_read_host_bytes(
                        model_path,
                        caps = server_caps,
                        extra_args = _fit_extras,
                        env = _fit_env,
                        auto_resolves_off = _fit_lazy_auto_off,
                    )
                    _fit_load_mode_kwargs = dict(
                        model_size = _fit_model_size,
                        mmproj_pinned_bytes = _mmproj_pinned_bytes
                        + (_fit_env_mmproj_bytes if _fit_env_mmproj_on_host else 0),
                        # Checkpoints are split into host_only_bytes below.
                        kv_cache_bytes = _kv_bytes(effective_ctx, 0),
                        kv_sized = self._can_estimate_kv(),
                        # mtp_overhead_fn bound the pre-raise micro-batch; charge the raised one.
                        mtp_bytes = _mtp_bytes(
                            effective_ctx,
                            ubatch = (
                                _effective_ubatch
                                if self._moe_spill_batch_restore is not None
                                else None
                            ),
                        ),
                        # MTP engages whole: its unsized arm prices weights but no draft KV, which a
                        # flat cushion covers.
                        mtp_unsized = bool(
                            _flat_mtp_engages
                            or _cpu_draft_fit_bytes is None
                            or _draft_split_across_host
                        ),
                        # Neither a CPU drafter nor checkpoints can be covered by free VRAM.
                        host_only_bytes = (_cpu_draft_fit_bytes or 0) + _ckpt_host_bytes,
                        # A tensor split replicates the graph buffer on every device.
                        compute_buffer_flat = (
                            _fit_devices * _compute_buffer_tensor
                            if _tp_planned
                            else _compute_buffer_pipeline
                        ),
                        compute_buffer_ctx = _cc_bytes(effective_ctx, _fit_devices),
                        # Ungated, like _subset_model_size: the per-device CUDA context
                        # and scratch are there whether or not llama.cpp keeps the
                        # pipeline (_cc_bytes is the term that gates on it).
                        pipeline_overhead_bytes = (
                            max(0, _fit_devices - 1) * _pipeline_overhead_bytes
                        ),
                        soft_overhead = _fit_soft_overhead,
                        gpus = gpus,
                        gpu_indices = gpu_indices,
                        shared_gpu_ids = _shared_gpu_ids,
                        # A running fitter keeps --fit-target free per device, so that
                        # margin is not VRAM this fit may spend. Zero only when the
                        # fitter is provably off; see _fitter_runs above.
                        fit_margin_mib = _fit_margins_by_idx or _fit_margin_mib,
                        is_vulkan_backend = is_vulkan_backend,
                        # Never stripped, and appended last, so they beat this
                        # launch's own -ngl.
                        extra_args = _fit_extras,
                        env = _fit_env,
                        lazy_read_bytes = _fit_lazy_bytes,
                    )
                    _resident_inputs = max(0, _host_input_embd - _fit_lazy_bytes)
                    if _resident_inputs and _fit_load_mode_kwargs["model_size"]:
                        _fit_load_mode_kwargs["model_size"] = max(
                            1, _fit_load_mode_kwargs["model_size"] - _resident_inputs
                        )
                        _fit_load_mode_kwargs["host_only_bytes"] += _resident_inputs
                    if _managed_lazy_flags is None:
                        _lazy_prompt_cache = _prompt_cache_host_bytes(
                            cache_ram,
                            _fit_extras,
                            _fit_env,
                            supports_cache_ram = bool(server_caps.get("supports_cache_ram")),
                        )

                        def _table_resident_fits(table: int) -> Optional[bool]:
                            if (
                                _lazy_prompt_cache is None
                                or not _fit_load_mode_kwargs["model_size"]
                                or effective_ctx <= 0
                                or _pass_through_reprices_cache(_fit_extras, n_parallel)
                                # A CPU child cannot use the cards this fit credits.
                                or self._child_runs_on_cpu(
                                    binary,
                                    _fit_extras,
                                    _fit_env,
                                    cpu_forced = _paravirtual_cpu_forced or _arch_gate_forced_cpu,
                                    tensor_parallel = tensor_parallel,
                                )
                            ):
                                return None
                            return (
                                self._fit_derived_load_mode(
                                    **{
                                        **_fit_load_mode_kwargs,
                                        "lazy_read_bytes": table,
                                        "host_only_bytes": _fit_load_mode_kwargs["host_only_bytes"]
                                        + table
                                        + _lazy_prompt_cache
                                        + int(table * _LAZY_RESIDENT_MARGIN_FRAC),
                                        "announce": False,
                                    }
                                )
                                == self._FIT_LOAD_MODE
                            )

                        _managed_lazy_flags = self._managed_lazy_mode_flags(
                            model_path,
                            caps = server_caps,
                            extra_args = _fit_extras,
                            env = _fit_env,
                            auto_resolves_off = _fit_lazy_auto_off,
                            resident_fits = _table_resident_fits,
                            binary = binary,
                        )
                    if _managed_lazy_flags and _fit_lazy_bytes:
                        # Kept resident by --lazy-mode off; the MoE cache below may want it lazy.
                        _moe_table_bytes = _fit_lazy_bytes
                        if _fit_load_mode_kwargs["model_size"]:
                            _fit_load_mode_kwargs["model_size"] = max(
                                1, _fit_load_mode_kwargs["model_size"] - _fit_lazy_bytes
                            )
                        _fit_load_mode_kwargs["host_only_bytes"] += _fit_lazy_bytes
                        _fit_load_mode_kwargs["lazy_read_bytes"] = 0
                        _fit_lazy_bytes = 0
                    _spill_inputs["lazy_read_bytes"] = _fit_lazy_bytes
                    _fit_load_mode = self._fit_derived_load_mode(**_fit_load_mode_kwargs)
                    if _moe_cache_candidate:
                        _moe_cache_switch = _moe_switch(_MOE_CACHE_ENV)
                        _moe_share = (
                            _moe_resident_share_now() if _moe_cache_switch == "auto" else None
                        )
                        if not _moe_cache_share_admits(
                            _moe_share, drafter = _mtp_will_engage, switch = _moe_cache_switch
                        ):
                            _moe_cache_candidate = False
                            if _moe_cache_switch == "off":
                                _why = f"{_MOE_CACHE_ENV} is off"
                            elif _moe_share is None:
                                _why = "the share of routed experts left on the GPU is unknown"
                            elif _mtp_will_engage and not _MOE_CACHE_WITH_DRAFTER:
                                _why = "a drafter is engaged"
                            else:
                                _why = (
                                    f"an estimated {100 * _moe_share:.0f}% of the routed "
                                    "experts stay on the GPU"
                                )
                            logger.info(
                                "MoE cache: %s; keeping the --fit split without "
                                "--moe-cache-mib.",
                                _why,
                            )
                    if _moe_cache_candidate:
                        # The cache frees no host RAM: every routed expert plus the prompt
                        # cache must fit pinned (moved out of model_size, not counted twice).
                        _moe_expert_bytes = (self._gguf_tensor_scan(model_path) or (None, {}, 0))[2]
                        _moe_prompt_cache_bytes = _prompt_cache_host_bytes(
                            cache_ram,
                            _fit_extras,
                            _fit_env,
                            supports_cache_ram = bool(server_caps.get("supports_cache_ram")),
                        )
                        # An unknown size (unsized inherited projector) admits nothing.
                        _moe_cache_ram_ok = bool(
                            _fit_load_mode_kwargs["model_size"]
                            and _moe_expert_bytes > 0
                            and _moe_prompt_cache_bytes is not None
                            and self._fit_derived_load_mode(
                                **{
                                    **_fit_load_mode_kwargs,
                                    "model_size": max(
                                        1, _fit_load_mode_kwargs["model_size"] - _moe_expert_bytes
                                    ),
                                    "host_only_bytes": _fit_load_mode_kwargs["host_only_bytes"]
                                    + _moe_expert_bytes
                                    + _moe_prompt_cache_bytes,
                                    "announce": False,
                                }
                            )
                            == self._FIT_LOAD_MODE
                        )
                        if not _moe_cache_ram_ok and _moe_table_bytes > 0 and _moe_expert_bytes > 0:
                            # Reading the resident table lazily frees the RAM to pin the experts; the cache beats the --fit split.
                            _moe_cache_ram_ok = bool(
                                _moe_prompt_cache_bytes is not None
                                and self._fit_derived_load_mode(
                                    **{
                                        **_fit_load_mode_kwargs,
                                        "model_size": max(
                                            1,
                                            _fit_load_mode_kwargs["model_size"] - _moe_expert_bytes,
                                        ),
                                        "host_only_bytes": max(
                                            0,
                                            _fit_load_mode_kwargs["host_only_bytes"]
                                            - _moe_table_bytes,
                                        )
                                        + _moe_expert_bytes
                                        + _moe_prompt_cache_bytes,
                                        "announce": False,
                                    }
                                )
                                == self._FIT_LOAD_MODE
                            )
                            # --lazy-mode off goes only where the cache is really added below.
                            _moe_cache_lazy_table = _moe_cache_ram_ok
                        if not _moe_cache_ram_ok:
                            logger.info(
                                "MoE cache: system RAM cannot hold the %.1f GB of routed "
                                "experts pinned; launching without --moe-cache-mib.",
                                _moe_expert_bytes / (1024**3),
                            )
                except Exception as e:
                    logger.warning(f"GPU selection failed ({e}), using --fit on")
                    gpu_indices, use_fit = None, True
                    _metal_full_offload_pinned = False
                    # Not a verdict: this arm never priced anything, so the flag is set explicitly and cannot survive from a previous load
                    _placement_verdict_partial = False
                    # Same reasoning: an unpriced load keeps llama.cpp's default.
                    _fit_load_mode = None
                    _managed_lazy_flags = None
                    _moe_cache_candidate = _moe_cache_ram_ok = _moe_cache_lazy_table = False
                    # And a half-built snapshot is not a price either: the throw can
                    # land after the dict exists but before the later terms are added
                    # to it, which would plan against a footprint missing them.
                    _spill_inputs = None
                    # Nor is a ceiling the fallback's own fitter may spill past.
                    _vram_fit_ctx = None
                    tp_tensor_split = None
                    effective_ctx = requested_ctx  # fall back to original
                    # Nor is a slot cut for the drafter: requested count, no drafter, as before.
                    if _mtp_slots_asked is not None:
                        n_parallel = _mtp_slots_asked
                        _mtp_slots_asked = _mtp_slots_target_ctx = None
                        _effective_ubatch = _ubatch_for_slots(n_parallel)
                        _compute_buffer_pipeline = _layer_compute_buffer(n_parallel)
                        _spec_dropped_no_vram = True
                        _mtp_will_engage = False
                        mtp_overhead_fn = None
                        _mtp_kv_unsized = False

                # Outside the placement catch, which would fall back to --fit on; the route
                # evicts an idle resident and retries.
                if _metal_alongside_shortfall is not None:
                    need_mib, budget_mib = _metal_alongside_shortfall
                    raise GpuMemoryShortError(
                        f"This model and its context need about {need_mib / 1024:.1f} GiB, "
                        f"but only {budget_mib / 1024:.1f} GiB remains in this Mac's "
                        "unified-memory budget. Free memory or use a smaller model.",
                        short_mib = need_mib - budget_mib,
                    )
                # A hand-set context unified memory cannot hold. Raised here so the handler
                # above cannot turn it back into the launch it refuses, and ahead of the
                # Vulkan and APU checks, which describe hardware this branch has ruled out.
                if _metal_ctx_refusal:
                    raise RuntimeError(_metal_ctx_refusal)
                self._record_load_warning(_metal_ctx_warning)

                if (
                    intent.refuse_partial_gpu_fit
                    and (_placement_verdict_partial or _ctx_capped_for_vram)
                    and gpu_memory_mode != "manual"
                ):
                    capped_only = _ctx_capped_for_vram and not _placement_verdict_partial
                    asked_ctx = _ctx_before_fit if capped_only else effective_ctx
                    need_mib = (
                        max(1, gguf_size - _host_input_embd) + mmproj_size + _kv_bytes(asked_ctx)
                    ) // (1024 * 1024)
                    free_mib = sum(free for _, free in gpus)
                    need_gb, free_gb = need_mib / 1024, free_mib / 1024
                    # The probe's own classifiers: an integrated CUDA SoC can report a real total.
                    try:
                        _shared = (
                            _shared_pool_gpus
                            | self._rocm_unified_memory_gpu_ids()
                            | self._integrated_cuda_gpu_ids()
                        )
                    except Exception:
                        _shared = None
                    raise GpuMemoryShortError(
                        _gpu_short_message(need_gb, free_gb, asked_ctx, effective_ctx, capped_only),
                        capped = capped_only,
                        short_mib = max(0, need_mib - free_mib),
                        gpu_indices = tuple(idx for idx, _ in gpus),
                        discrete = _shared is not None and not _shared & {idx for idx, _ in gpus},
                    )

                # Unpinnable explicit Vulkan ordinal: fail loudly rather than fit onto an unselected device; clear raw selection so it never leaks into gpu_ids
                if _vulkan_explicit_unmatched:
                    self._gpu_ids = None
                    self._requested_gpu_ids = None
                    raise ValueError(
                        f"Requested Vulkan GPU ordinal(s) {_vulkan_requested_ids} not "
                        f"present. Available Vulkan devices: {_vulkan_available_ordinals}."
                    )

                # Pin the whole picked set so the model can't spill onto an unpicked GPU.
                if gpu_ids:
                    gpu_indices, tp_tensor_split = self._picker_ordered_gpu_indices(
                        gpu_ids, gpu_indices, tp_tensor_split
                    )
                # Auto Vulkan fit prefers discrete GPUs and keeps that pool pinned.
                elif (
                    is_vulkan_backend
                    and gpu_memory_mode != "manual"
                    and use_fit
                    and gpu_indices is None
                    and _detected_gpus
                ):
                    gpu_indices = sorted(idx for idx, _free in _detected_gpus)

                # Reload dedupe compares requested_gpu_ids; a forced-CPU launch pins no GPU.
                if gpu_ids and not _paravirtual_cpu_forced:
                    effective_pin = gpu_indices if gpu_indices is not None else gpu_ids
                    # Survivor order, not sorted: _runtime_matches_intent compares positionally.
                    self._gpu_ids = [int(idx) for idx in effective_pin]
                else:
                    self._gpu_ids = None

                # Carried to the launch site, the only place the pageable override can apply; without it an oversized APU load under none or mlock is an OOM kill
                _apu_ram_oversized = False
                # model_size less a lazily read table; the text-only reprice charges the same.
                _apu_resident_model_size = model_size
                # The two shortfall notices this load recorded, kept verbatim (and
                # un-amended) so the text-only fallback further down can re-price them
                # once it drops the CPU-pinned projector whose bytes they charged. Both
                # are re-stated by the arch-crash retry, which starts its own warning
                # scope, so what these hold is always the pair behind the notice the
                # route would hand back right now.
                _apu_ram_msg: Optional[str] = None
                _host_ram_msg: Optional[str] = None
                # The RAM figure those notices were priced against, kept with them.
                _apu_avail_mib: Optional[int] = None
                # ...and which part they describe, for the same reason.
                _apu_ram_part = "APU"

                # Unified-memory APUs load weights into system RAM (WSL: VM cap), so refuse oversize loads the OS would kill mid-flight; base model only. CUDA/ROCm ids only
                if (
                    model_size is not None
                    and not is_vulkan_backend
                    # An integrated CUDA SoC loads into system RAM; _shared_gpu_ids is Vulkan-only, so without this its pool is credited as VRAM and the spill prices at zero
                    and (
                        self._amd_apu_wants_unified_memory(gpu_indices)
                        or (
                            self._integrated_cuda_probe_is_free()
                            and self._integrated_cuda_selection_is_all_shared(gpu_indices)
                        )
                    )
                ):
                    # Read once: a later reading would already exclude the resident model.
                    _apu_avail_mib = self._available_system_memory_mib()
                    # Kept beside the notice: the text-only fallback rebuilds it later
                    # and would otherwise turn a Spark's into an APU's, .wslconfig hint
                    # and all.
                    _apu_ram_part = (
                        "APU" if self._amd_apu_wants_unified_memory(gpu_indices) else "SoC"
                    )
                    _apu_lazy_extras = (
                        self._strip_device_extra_args(extra_args)
                        if _gpu_ids_own_device_flags
                        else extra_args
                    )
                    _apu_lazy_env = dict(os.environ)
                    if gpu_memory_mode == "manual":
                        self._clear_manual_placement_env(_apu_lazy_env)
                    if _gpu_ids_own_device_flags:
                        self._clear_device_placement_env(_apu_lazy_env)
                    self._reconcile_split_mode_env(_apu_lazy_env, tensor_parallel = tensor_parallel)
                    _apu_child_order, _apu_child_mask = self._planned_child_gpu_order(
                        gpu_ids = gpu_ids,
                        gpu_indices = gpu_indices,
                        tp_tensor_split = tp_tensor_split,
                        extra_args = _apu_lazy_extras,
                        env = _apu_lazy_env,
                        is_vulkan_backend = False,
                        cpu_forced = _paravirtual_cpu_forced or _arch_gate_forced_cpu,
                    )
                    _apu_resident_model_size = max(
                        1,
                        model_size
                        - self._lazy_read_host_bytes(
                            model_path,
                            caps = server_caps,
                            extra_args = _apu_lazy_extras,
                            env = _apu_lazy_env,
                            # A failed GPU probe still launches on the APU, where auto is off.
                            auto_resolves_off = (
                                not _detected_gpus
                                and not self._child_runs_on_cpu(
                                    binary,
                                    _apu_lazy_extras,
                                    _apu_lazy_env,
                                    cpu_forced = _paravirtual_cpu_forced or _arch_gate_forced_cpu,
                                    tensor_parallel = tensor_parallel,
                                )
                            )
                            or self._lazy_auto_resolves_off(
                                gpu_indices = _apu_child_order,
                                detected_gpus = _detected_gpus,
                                shared_gpu_ids = _shared_gpu_ids,
                                is_vulkan_backend = False,
                                probe_integrated_cuda = False,
                                extra_args = _apu_lazy_extras,
                                env = _apu_lazy_env,
                                tensor_parallel = tensor_parallel,
                                child_gpu_order = _apu_child_mask,
                            ),
                        ),
                    )
                    _ram_msg = self._apu_ram_shortfall_message(
                        # A pinned projector left model_size but not system RAM, and
                        # this guard exists to stop an oversize load being OOM-killed
                        # mid-read, so it has to weigh the projector either way.
                        _apu_resident_model_size + _mmproj_pinned_bytes,
                        _apu_avail_mib,
                        part = _apu_ram_part,
                    )
                    # Re-ask for the set the child sees: the guard priced every visible card, incl. an APU the gate drops. Warned branch only; an empty answer keeps the warning
                    if _ram_msg and gpu_indices is None and not gpu_ids:
                        _gated_pin = self._arch_gate_survivors(binary)
                        if _gated_pin and not self._amd_apu_wants_unified_memory(_gated_pin):
                            logger.info(
                                "Skipping the APU RAM warning: the arch gate pins "
                                "this launch to discrete GPU(s) %s.",
                                _gated_pin,
                            )
                            _ram_msg = None
                    # Read after the arch-gate skip, so a launch the gate moves onto discrete cards is not remapped; independent of UNSLOTH_ALLOW_HOST_OFFLOAD
                    _apu_ram_oversized = bool(_ram_msg)
                    # The opt-out may only silence the message, never the rewrite: read it after the verdict, so the rewrite the verdict drives is untouched
                    if _ram_msg and self._host_offload_warning_opted_out():
                        logger.info(
                            "UNSLOTH_ALLOW_HOST_OFFLOAD set: not recording the APU RAM advisory."
                        )
                        _ram_msg = None
                    _apu_ram_msg = _ram_msg
                    self._record_load_warning(_ram_msg)

                # Probe LLAMA_ARG_MMPROJ too, since launch_mmproj_path misses it, but never a projector the unpinnable guard dropped
                self._mmproj_has_audio = False
                self._mmproj_accepts_image = True
                # Skip a projector the vision switch or the Metal scrub will remove, unless the switch keeps
                # an audio-only encoder; otherwise we'd advertise audio the child does not have.
                _dv_env_mmproj_kept = (
                    disable_vision
                    and not _paravirtual_cpu_forced
                    and _mmproj_env_is_audio_only(os.environ.get("LLAMA_ARG_MMPROJ"))
                )
                _mmproj_probe = launch_mmproj_path or (
                    ""
                    if (_pv_mmproj_unpinnable or (disable_vision and not _dv_env_mmproj_kept))
                    else (os.environ.get("LLAMA_ARG_MMPROJ") or "")
                )
                self._mmproj_projector_type = None
                if launch_mmproj_path or os.path.isfile(_mmproj_probe):
                    try:
                        from utils.models.gguf_metadata import (
                            mmproj_capabilities,
                            read_mmproj_projector_type,
                        )

                        has_audio, accepts_image = mmproj_capabilities(_mmproj_probe)
                        self._mmproj_has_audio = has_audio
                        self._mmproj_accepts_image = accepts_image
                        # KV admission needs the projector family to look up per-image token cost.
                        self._mmproj_projector_type = read_mmproj_projector_type(_mmproj_probe)
                    except Exception as e:
                        logger.debug(f"mmproj capability read failed: {e}")

                auto_fit = gpu_memory_mode == "manual" and gpu_layers < 0
                # Keep the unstripped request: it is the reuse comparator, stripping it forced reloads.
                _extra_args_as_requested = list(extra_args) if extra_args is not None else None
                if self._metal_drops_zero_ctx_override(ctx_override, _caller_owns_budget):
                    extra_args = strip_context_only(extra_args)
                    logger.warning(
                        "Dropping the pass-through zero context: on Metal it pins the "
                        "model's native length and disables --fit."
                    )
                _metal_floor = self._metal_zero_ctx_floor(
                    effective_ctx,
                    auto_fit,
                    _caller_owns_budget,
                    self._context_length,
                    max_available_ctx,
                    # Whether the child fitter runs decides the Auto-layers exemption and floor reach; only the user can turn it off, via later extras or LLAMA_ARG_FIT
                    fitter_runs = fit_is_effectively_on(extra_args, os.environ),
                )
                if _metal_floor:
                    effective_ctx = _metal_floor
                    # Replace, not max(): on failure this still holds the native length, which never fit.
                    max_available_ctx = _metal_floor
                    logger.warning(
                        "No GPU is enumerated on Metal and the context cap did not "
                        f"run, so starting at {effective_ctx} rather than the model's "
                        "native length."
                    )

                # Fail open: these flags are emitted every launch and an unknown arg exits a stale or user LLAMA_SERVER_PATH, so only a build positively lacking one drops it
                _caps = _launch_caps(binary)
                cmd = [
                    binary,
                    "-m",
                    model_path,
                    "--port",
                    str(self._port),
                    "--parallel",
                    str(n_parallel),
                ]
                # Gate on whether the build has -fa and whether it takes a value (old builds: bare bool).
                if _caps.get("supports_flash_attn", True):
                    if not _extra_args_set_any_flag(extra_args, {"-fa", "--flash-attn"}):
                        cmd.append("--flash-attn")
                        if _caps.get("flash_attn_takes_value", True):
                            cmd.append("on")
                else:
                    # Only ever reached on an authoritative --help that has no
                    # flash attention to lose, so there is no speed to give up.
                    _flash_attn_known_off = True
                # Error out at n_ctx instead of silently rotating the KV cache; frontend catches it and points the user at "Context Length".
                if _caps.get("supports_no_context_shift", True):
                    if not _extra_args_set_any_flag(
                        extra_args, {"--context-shift", "--no-context-shift"}
                    ):
                        cmd.append("--no-context-shift")
                # Omit -c for auto-fit with no context: -c 0 pins native ctx and disables --fit reduction.
                if effective_ctx > 0:
                    cmd.extend(["-c", str(effective_ctx)])
                elif not auto_fit:
                    cmd.extend(["-c", "0"])

                # llama-server 501s on /v1/embeddings without this; pooling stays model-default.
                if self.is_embedding_gguf:
                    cmd.append("--embedding")

                # emitted before user extras so a pass-through -b / -ub still last-wins-overrides
                if n_batch is not None:
                    # -b < --parallel and -b 1 both GGML_ASSERT, so the batch floor is max(slots, 2).
                    _emit_batch = _emitted_n_batch(n_batch, n_parallel)
                    if _emit_batch != n_batch:
                        logger.warning(
                            "Raising --batch-size from %s to %s: llama-server aborts below "
                            "2 and below the %s parallel slot(s) it serves.",
                            n_batch,
                            _emit_batch,
                            n_parallel,
                        )
                    cmd.extend(["--batch-size", str(_emit_batch)])
                if n_ubatch is not None:
                    cmd.extend(["--ubatch-size", str(n_ubatch)])
                # The expert-spill raise, as emitted and as it was before, so a CPU
                # replay (no GPU, nothing to stream) can hand back today's pair.
                self._moe_spill_batch_tokens = None
                if self._moe_spill_batch_restore is not None:
                    _orig_batch, _orig_ubatch = self._moe_spill_batch_restore
                    _raised = ["--ubatch-size", str(n_ubatch)]
                    _before = [] if _orig_ubatch is None else ["--ubatch-size", str(_orig_ubatch)]
                    if n_batch is not None and n_batch != _orig_batch:
                        _raised = ["--batch-size", str(_emit_batch), *_raised]
                        _before = [
                            *(
                                []
                                if _orig_batch is None
                                else [
                                    "--batch-size",
                                    str(_emitted_n_batch(_orig_batch, n_parallel)),
                                ]
                            ),
                            *_before,
                        ]
                    self._moe_spill_batch_tokens = (_raised, _before)

                server_caps = _launch_caps(binary)

                # Before extras so a hand-typed flag wins; gated since older builds exit on unknown flags.
                if ctx_checkpoints is not None:
                    _ctxcp_flag = server_caps.get("ctx_checkpoints_flag")
                    if _ctxcp_flag:
                        cmd.extend([str(_ctxcp_flag), str(int(ctx_checkpoints))])
                    else:
                        logger.info(
                            "llama-server has no --ctx-checkpoints; skipping the requested %s.",
                            ctx_checkpoints,
                        )
                if cache_ram is not None:
                    if server_caps.get("supports_cache_ram"):
                        cmd.extend(["--cache-ram", str(int(cache_ram))])
                    else:
                        logger.info(
                            "llama-server has no --cache-ram; skipping the requested %s MiB.",
                            cache_ram,
                        )
                # Read early so the effective budget can fold in an inherited
                # LLAMA_ARG_THINK_BUDGET*; the launch env is rebuilt below before spawn.
                env = self._llama_server_env_for_binary(binary)
                # On every launch, not only where Studio emits --mmproj: a projector
                # from the extras or LLAMA_ARG_MMPROJ takes video too.
                cmd.extend(_video_fps_flags(server_caps, extra_args = extra_args))
                _ffmpeg_flags, self._video_ffmpeg_dir = _video_ffmpeg_dir_flags(
                    server_caps, extra_args = extra_args
                )
                cmd.extend(_ffmpeg_flags)

                # Report the public model id, not the raw -m path.
                from core.inference.model_ids import public_model_id

                _alias = public_model_id(self._model_identifier or model_path)
                if _alias:
                    cmd.extend(["--alias", _alias])

                fully_gpu_offloaded = False
                # Set when a positional --tensor-split is emitted, so the env block
                # can pin CUDA to PCI order even without a GPU subset (see below).
                manual_tensor_split_emitted = False
                # The MoE cache's "auto" sizing runs only under llama.cpp's fitter.
                _placed_by_fitter = False
                if gpu_memory_mode == "manual" and gpu_layers >= 0:
                    # --fit off: _ctx_integrity_flags must not add --fit-ctx.
                    use_fit = False
                    cmd.extend(["--gpu-layers", str(gpu_layers), "--fit", "off"])
                    # Keep the first n_cpu_moe MoE layers' experts on CPU.
                    moe_flag = self._resolve_cpu_moe_flag(
                        n_cpu_moe,
                        self.n_moe_layers,
                        self._leading_dense_block_count or 0,
                    )
                    if moe_flag is not None:
                        cmd.extend(["--n-cpu-moe", str(moe_flag)])
                    elif n_cpu_moe:
                        # Requested on a dense model: nothing was emitted, so
                        # don't report a count llama-server never received.
                        self._n_cpu_moe = 0
                    # Emit only with >1 GPU and a matching length: llama.cpp zero-pads short lists silently.
                    _split_gpus = self._effective_gpu_count(gpu_indices)
                    if tensor_split and _split_gpus > 1:
                        # An all-zero/non-positive sanitized split assigns nothing
                        # anywhere, so fall through to the free-VRAM default in
                        # that case.
                        _sanitized_split = self._sanitize_tensor_split(tensor_split)
                        _split_total = sum(_sanitized_split)
                        if len(_sanitized_split) == _split_gpus and _split_total > 0:
                            cmd.extend(
                                ["--tensor-split", ",".join(f"{x:g}" for x in _sanitized_split)]
                            )
                            self._tensor_split = _sanitized_split
                            manual_tensor_split_emitted = True
                        else:
                            logger.warning(
                                "Dropping manual --tensor-split (%d entries for "
                                "%d GPUs, sanitized total %s); llama.cpp's "
                                "free-VRAM split applies instead",
                                len(tensor_split),
                                _split_gpus,
                                _split_total,
                            )
                            self._tensor_split = None
                    elif tensor_split:
                        # Single effective GPU: the split is never emitted, so
                        # don't report it as active via /status and /load.
                        self._tensor_split = None
                elif use_fit and _metal_full_offload_pinned:
                    # Keep the launch on the full-offload mapping the discount measured.
                    cmd.extend(["-ngl", "-1", "--fit", "off"])
                    # -1 is the llama.cpp default, so the --fit on retry does not abort on a user-set -ngl; the retry is gated on fully_gpu_offloaded, so set it
                    fully_gpu_offloaded = True
                elif use_fit:
                    # Spill weight tensors, not whole layers: a spilled layer takes its KV cache with it.
                    _spill = self._planned_tensor_spill(
                        _spill_inputs,
                        extra_args = extra_args,
                        env = os.environ,
                    )
                    # Do not test the spill plan for truthiness: Plan is a dataclass and always truthy, even when it could not place anything
                    _spill_flags = self._spill_plan_flags_for(_spill)
                    if _spill_flags:
                        self._spill_plan_flags = _spill_flags
                        cmd.extend(self._spill_plan_flags)
                        logger.info(
                            "Tensor spill: %s (resident %.1f GB, host %.1f GB)",
                            " ".join(_spill.ot_patterns),
                            _spill.vram_bytes / (1024**3),
                            _spill.host_bytes / (1024**3),
                        )
                    else:
                        cmd.extend(["--fit", "on"])
                        _placed_by_fitter = True
                elif gpu_indices is not None:
                    # --fit off required: otherwise --fit on still offloads ~1 GB at --parallel 4.
                    cmd.extend(["-ngl", "-1", "--fit", "off"])
                    fully_gpu_offloaded = True
                    _mixed_split = self._mixed_pin_split(
                        list(gpu_indices),
                        _spill_inputs,
                        _shared_gpu_ids,
                        extra_args,
                        env,
                        layer_min_gpus = _layer_min_gpus,
                        tensor_parallel = tensor_parallel,
                        pin_owns_devices = _gpu_ids_own_device_flags,
                    )
                    if _mixed_split is not None:
                        self._mixed_split_flags = [
                            "--tensor-split",
                            self._format_tensor_split(_mixed_split),
                        ]
                        cmd.extend(self._mixed_split_flags)
                        if gpu_memory_mode != "manual":
                            self._auto_tensor_split_emitted = self._auto_split_fingerprint(
                                _mixed_split
                            )
                        logger.info(
                            "Filling discrete GPU(s) before the shared-memory iGPU: "
                            "--tensor-split %s over %s",
                            cmd[-1],
                            list(gpu_indices),
                        )

                # Expose Prometheus /metrics for the engine-stats logger, only
                # when the binary advertises it (older/custom binaries may not).
                if server_caps.get("supports_metrics"):
                    if not _extra_args_set_any_flag(extra_args, {"--metrics", "--no-metrics"}):
                        cmd.append("--metrics")
                self._slot_save_dir = None
                self._slot_save_binary = None
                self._prompt_cache_disabled = False
                if server_caps.get("supports_slot_save"):
                    try:
                        from utils.paths.storage_roots import (  # noqa: WPS433
                            llama_slot_cache_root,
                        )

                        slot_dir = llama_slot_cache_root()
                        slot_dir.mkdir(parents = True, exist_ok = True)
                        # Saved KV encodes chat content; keep it from other local users.
                        with contextlib.suppress(OSError):
                            os.chmod(slot_dir, 0o700)
                        slot_path = str(slot_dir)
                        # llama.cpp rejects non-ASCII paths on Windows; skip optional slot persistence until the managed runtime carries the upstream UTF-8 fix
                        if sys.platform == "win32" and not slot_path.isascii():
                            logger.warning(
                                "Disabling llama.cpp slot persistence because its "
                                "Windows path contains non-ASCII characters: %s",
                                slot_path,
                            )
                        else:
                            cmd.extend(["--slot-save-path", slot_path])
                            self._slot_save_dir = slot_path
                            self._slot_save_binary = (binary, Path(binary).stat().st_mtime_ns)
                    except OSError:
                        self._slot_save_dir = None
                        self._slot_save_binary = None
                _integrity_args = (
                    n_parallel,
                    use_fit,
                    auto_fit,
                    requested_ctx,
                    effective_ctx,
                    server_caps,
                )
                _integrity_flags = self._ctx_integrity_flags(
                    *_integrity_args,
                    fit_target_delta_mib = _fit_target_delta_mib,
                )
                # From the flags, not the inputs: --fit off, an older server without
                # the capability and a budget at the default all leave the child
                # unpriced, and each is decided inside that call.
                _fit_target_priced = "--fit-target" in _integrity_flags
                # Spelled per device at spawn, against the child's final mask.
                _fit_target_margins_by_idx = (
                    self._fit_target_margins_mib(
                        auto_fit = auto_fit,
                        fit_target_delta_by_idx = _fit_target_delta_by_idx,
                        supports_fit_target = bool(server_caps.get("supports_fit_target")),
                    )
                    if _fit_target_priced and use_fit and not _metal_fit_target_floor_mib
                    else None
                )
                if _metal_fit_target_floor_mib or _sidecar_fit_reserve_mib:
                    # Rebuilt after the price is read: the floor is the machine's
                    # margin and the sidecar reserve the drafter's, not the budget
                    # slider's.
                    _integrity_flags = self._ctx_integrity_flags(
                        *_integrity_args,
                        fit_target_delta_mib = _fit_target_delta_mib,
                        fit_target_floor_mib = _metal_fit_target_floor_mib,
                        # The reserve rides on the spawn's trailing pair, which retry cleanup must
                        # tell apart from a user's value.
                        fit_target_reserve_mib = 0,
                    )
                    if _sidecar_fit_reserve_mib and not server_caps.get("supports_fit_target"):
                        _sidecar_fit_reserve_mib = 0
                cmd.extend(_integrity_flags)
                offload_overridden = _extra_args_set_any_flag(
                    extra_args, _GPU_OFFLOAD_OVERRIDE_FLAGS
                )
                threads_overridden = _extra_args_set_any_flag(extra_args, _THREAD_OVERRIDE_FLAGS)
                full_offload_tuning_active = fully_gpu_offloaded and not offload_overridden
                # The CPU replay must undo what this launch pins for an absent GPU;
                # a user's own --threads still wins.
                _drop_full_offload_threads = (
                    sys.platform == "win32"
                    and full_offload_tuning_active
                    and not threads_overridden
                )

                # Omit --threads when unset: -1 means all hyperthreads, which slows decode.
                if (
                    sys.platform == "win32"
                    and full_offload_tuning_active
                    and not threads_overridden
                ):
                    cmd.extend(["--threads", "2"])
                elif n_threads is not None and n_threads > 0:
                    cmd.extend(["--threads", str(n_threads)])

                # Enable Jinja chat template rendering
                if _caps.get("supports_jinja", True):
                    if not _extra_args_set_any_flag(extra_args, {"--jinja", "--no-jinja"}):
                        cmd.extend(["--jinja"])
                cmd.extend(_trace_verbosity_args(_caps, extra_args, os.environ))

                # KV cache data type
                _valid_cache_types = _VALID_CACHE_TYPES
                # Normalize case like the budget does: a case-sensitive match drops Q8_0 with no flag, so llama.cpp runs f16 while the estimate priced q8_0
                cache_type_kv = cache_type_kv.strip().lower() if cache_type_kv else cache_type_kv
                if (
                    cache_type_kv
                    and cache_type_kv in _valid_cache_types
                    and not _cache_type_from_env
                ):
                    gpu_fallback_warning = (
                        None
                        if intent.cpu_fallback
                        or _arch_gate_forced_cpu
                        or not _detected_gpus
                        or _extras_cache is not None
                        else _kv_cache_gpu_fallback_warning(
                            cache_type_kv,
                            gpu_memory_mode,
                            gpu_layers,
                            lambda: self._installed_ggml_backends(binary),
                        )
                    )
                    if gpu_fallback_warning:
                        logger.warning(gpu_fallback_warning)
                        cache_type_kv = _GPU_UNACCELERATED_CACHE_TYPES[cache_type_kv]
                    if not _extra_args_set_any_flag(extra_args, _CACHE_TYPE_K_FLAGS):
                        cmd.extend(["--cache-type-k", cache_type_kv])
                    if not _extra_args_set_any_flag(extra_args, _CACHE_TYPE_V_FLAGS):
                        cmd.extend(["--cache-type-v", cache_type_kv])
                    self._cache_type_kv = cache_type_kv
                    logger.info(f"KV cache type: {cache_type_kv}")
                else:
                    # An env-only type is left inherited (untouched) so an
                    # asymmetric K/V env reaches the child as set.
                    self._cache_type_kv = None

                # Single-GPU tensor mode is redundant; unsupported archs abort and /load retries layer split.
                if tensor_parallel:
                    cmd.extend(["--split-mode", "tensor"])
                    _emitted_tensor_split: Optional[str] = None
                    if tp_tensor_split and len(tp_tensor_split) > 1:
                        _emitted_tensor_split = ",".join(str(int(x)) for x in tp_tensor_split)
                    elif (
                        gpu_memory_mode != "manual" and tensor_split and _tp_planned and gpu_indices
                    ):
                        # Fall back to the user ratio only if the planner emitted nothing and it fits budget; gate on _tp_planned, not tp_tensor_split, to avoid a KeyError
                        _split_gpus = self._effective_gpu_count(gpu_indices)
                        _sanitized_split = self._sanitize_tensor_split(tensor_split)
                        if (
                            len(_sanitized_split) == _split_gpus
                            and sum(_sanitized_split) > 0
                            and self._tensor_split_fits_budget(
                                _sanitized_split,
                                tp_gpus,
                                gpu_indices,
                                max(1, model_size - _host_input_embd),
                                effective_ctx,
                                cache_type_kv = cache_type_kv,
                                n_parallel = n_parallel,
                                n_ubatch = _effective_ubatch,
                                kv_unified = planned_kv_unified,
                                flash_attn = planned_flash_attn,
                                swa_full = swa_full,
                                mtp_engaged = _mtp_reserves_gpu,
                                mtp_overhead_fn = mtp_overhead_fn,
                                mtp_flat_reserve_bytes = (
                                    2 * 1024**3 if (_mtp_reserves_gpu and _mtp_kv_unsized) else 0
                                ),
                                total_by_idx = total_by_idx,
                                vram_fraction = _vram_frac,
                                # The planner's own operands, so "the same budget"
                                # in the docstring is true. Both are bound by the
                                # time _tp_planned is set.
                                soft_overhead_bytes = _soft_overhead,
                                scratch_cache_type_kv = _scratch_cache_type_kv,
                            )
                        ):
                            _emitted_tensor_split = self._format_tensor_split(_sanitized_split)
                    _emitted_values: Optional[list[float]] = None
                    if _emitted_tensor_split is not None:
                        cmd.extend(["--tensor-split", _emitted_tensor_split])
                        _emitted_values = [float(x) for x in _emitted_tensor_split.split(",")]
                    self._tensor_parallel = True
                    self._layer_preserves_tensor_intent = False
                    # Record the requested ratio, not the emitted one, so identical repeats are reused.
                    if gpu_memory_mode != "manual":
                        self._auto_tensor_split = self._auto_split_fingerprint(tensor_split)
                        self._auto_tensor_split_emitted = self._auto_split_fingerprint(
                            _emitted_values
                        )
                    logger.info(
                        "Tensor parallelism: --split-mode tensor, --tensor-split %s",
                        _emitted_tensor_split,
                    )
                else:
                    self._tensor_parallel = False
                    # > 1 only when a tensor request was downgraded but kept multi-GPU.
                    self._layer_preserves_tensor_intent = _layer_min_gpus > 1

                # Speculative decoding. See _build_speculative_flags for the
                # mode resolution, benchmarks, and llama.cpp references.
                _vulkan_pin_ids = gpu_indices if gpu_indices is not None else (gpu_ids or None)
                _draft_device = (
                    _extra_args_main_device(extra_args) if not _gpu_ids_own_device_flags else None
                )
                if _draft_device is None and is_vulkan_backend and _vulkan_pin_ids:
                    _draft_device = ",".join(f"Vulkan{i}" for i in _vulkan_pin_ids)
                launch_mtp_draft_path = self._resolve_launch_mtp_path(
                    mtp_draft_path = mtp_draft_path,
                    drafter_label = _DRAFTER_DISPLAY_LABELS.get(_spec_canon, "MTP"),
                )
                _suppressed_draft_path: Optional[str] = _unloadable_mtp_draft_path
                _suppressed_draft_reason: Optional[str] = (
                    "unloadable" if _unloadable_mtp_draft_path else None
                )
                _pv_suppressed_spec_extra_args: Optional[List[str]] = None
                # Drop an unpinnable drafter on virtualised Metal (throughput only, never correctness); a drafter-dependent --spec-type would abort the server, so drop it too
                _pv_draft_unpinnable = bool(
                    _paravirtual_cpu_forced
                    and not _paravirtual_draft_ngl_flag(server_caps)
                    and (
                        _extra_args_mtp_draft_path(extra_args, env = _child_spec_env(extra_args))
                        or (launch_mtp_draft_path and not _extra_args_set_spec_type(extra_args))
                    )
                    and not _extra_args_draft_offloaded_to_cpu(extra_args)
                )
                if _pv_draft_unpinnable:
                    logger.warning(
                        "Dropping the speculative drafter for this session: this "
                        "Mac's Metal device is virtualised and this llama-server "
                        "build advertises no draft-layer flag, so the drafter would "
                        "keep running on the device whose output is corrupt. Output "
                        "stays correct either way; this only costs speed. Run "
                        "'unsloth studio update' to install a build that supports it."
                    )
                    # Remember the suppressed drafter, else every repeat Apply reloads; record only what this drop removed, since None would clear an earlier drop record
                    if launch_mtp_draft_path:
                        _suppressed_draft_path = launch_mtp_draft_path
                        # A property of the build, not the file, so a repair must not reload.
                        _suppressed_draft_reason = "paravirtual"
                    launch_mtp_draft_path = None
                    if extra_args:
                        # Record extras as requested, or repeat Applies never match and reload.
                        _pv_suppressed_spec_extra_args = list(extra_args)
                        extra_args = strip_shadowing_flags(
                            extra_args,
                            strip_context = False,
                            strip_cache = False,
                            strip_spec = True,
                            strip_template = False,
                            strip_split_mode = False,
                        )
                # Classify using the same argv/env views as the launch so they cannot drift.
                _spec_placement_env = dict(os.environ)
                if gpu_memory_mode == "manual":
                    self._clear_manual_placement_env(_spec_placement_env)
                _spec_placement_extras = extra_args
                if _gpu_ids_own_device_flags:
                    _spec_placement_extras = self._strip_device_extra_args(extra_args)
                    self._clear_device_placement_env(_spec_placement_env)
                _spec_placement_argv = [*cmd, *(_spec_placement_extras or [])]
                # A device the launch really carries, and not the CPU spelling of it.
                # Only consulted where the probe found nothing: it is the one remaining
                # way the child can still be pointed at a GPU.
                _spec_device_pick = _extra_args_main_device(
                    _spec_placement_argv
                ) or _spec_placement_env.get("LLAMA_ARG_DEVICE")
                _spec_extras_pick_a_gpu = bool(_spec_device_pick) and not _device_selection_is_cpu(
                    _spec_placement_argv, _spec_placement_env
                )
                # Log-only; a priced reserve is reported by the spawn (_log_sidecar_fit_reserve).
                _sidecar_fit_sized = (
                    not fit_is_effectively_on(_spec_placement_argv, _spec_placement_env)
                    or bool(_sidecar_fit_reserve_eligible_mib)
                    or bool(server_caps.get("fit_measures_borrowed_drafts"))
                    or (
                        bool(launch_mtp_draft_path)
                        and not _sidecar_borrows_target_tensors(launch_mtp_draft_path)
                    )
                )
                spec_flags = self._build_speculative_flags(
                    speculative_type = speculative_type,
                    spec_draft_n_max = spec_draft_n_max,
                    extra_args = extra_args,
                    model_identifier = model_identifier,
                    model_path = model_path,
                    gpus = bool(_detected_gpus),
                    binary = binary,
                    mtp_draft_path = (
                        None if _spec_canon in ("dspark", "dflash") else launch_mtp_draft_path
                    ),
                    dspark_draft_path = (launch_mtp_draft_path if _spec_canon == "dspark" else None),
                    dspark_fit_sized = _sidecar_fit_sized,
                    dflash_draft_path = (launch_mtp_draft_path if _spec_canon == "dflash" else None),
                    dflash_fit_sized = _sidecar_fit_sized,
                    # So the fallback says unopenable, not missing. DSpark/DFlash never
                    # reach the drop.
                    mtp_drafter_unloadable = bool(_unloadable_mtp_draft_path),
                    drafter_no_vram = _spec_dropped_no_vram,
                    embedded_mtp_partial_offload = bool(
                        _spec_canon == "auto"
                        and self._nextn_predict_layers
                        and self._ssm_inner_size is not None
                        and self._full_attention_interval is not None
                        and not launch_mtp_draft_path
                        and not _extra_args_set_spec_type(extra_args)
                        # Only when no GPU is offloaded to: an explicit pick or a device in extras counts even if the probe missed it; CPU-only and Metal also emit --fit on
                        and (_detected_gpus or gpu_indices or _spec_extras_pick_a_gpu)
                        and self._partially_offloads_layers(
                            [*cmd, *(_spec_placement_extras or [])],
                            _spec_placement_env,
                            # Manual --fit on is a default, not evidence of a failed fit; only Auto fitter that ran and failed counts, not an empty VRAM probe with a hand-pinned device
                            fit_implies_partial = (
                                gpu_memory_mode != "manual" and _placement_verdict_partial
                            ),
                        )
                    ),
                    draft_device = _draft_device,
                )
                # Restore the requested spec view, or the spec-mode compare never matches.
                if _extra_args_set_spec_type(_pv_suppressed_spec_extra_args):
                    self._requested_spec_mode = None
                # Draft KV dtype applies only with --model-draft, judged on flags actually built, not the requested mode; unknown values are dropped since llama-server exits
                _draft_cache_type = (
                    spec_draft_cache_type.strip().lower() if spec_draft_cache_type else None
                )
                if _draft_cache_type and _draft_cache_type not in _VALID_KV_CACHE_TYPES:
                    logger.warning(
                        "Ignoring unsupported draft KV cache type %r", spec_draft_cache_type
                    )
                    _draft_cache_type = None
                if _draft_cache_type and any(
                    _flag_name(tok) in _LOCAL_DRAFT_FLAGS for tok in spec_flags
                ):
                    _draft_k_flag = server_caps.get("spec_draft_cache_k_flag")
                    _draft_v_flag = server_caps.get("spec_draft_cache_v_flag")
                    if _draft_k_flag and _draft_v_flag:
                        # llama.cpp refuses K != V on an MLA draft context.
                        spec_flags.extend(
                            [
                                str(_draft_k_flag),
                                _draft_cache_type,
                                str(_draft_v_flag),
                                _draft_cache_type,
                            ]
                        )
                        logger.info("Draft KV cache dtype: %s", _draft_cache_type)
                    else:
                        logger.info(
                            "llama-server has no draft KV cache flags; skipping the requested %s.",
                            _draft_cache_type,
                        )
                _no_backend_draft = _draft_backend_sampling_off(
                    spec_flags, server_caps, self._vocab_size, self._is_vulkan_backend(binary)
                )
                if _no_backend_draft:
                    logger.info(
                        "Vulkan: sampling drafts on the CPU, the %s-token vocab misaligns "
                        "the logits rows the backend sampler reads.",
                        self._vocab_size or "unknown",
                    )
                    spec_flags.extend(_no_backend_draft)
                # Remember where the spec block sits so a drafter-load failure
                # can be retried with these flags swapped out (see below).
                _spec_start = len(cmd)
                cmd.extend(spec_flags)

                # Separate drafter ignores main --gpu-layers (draft default -1 is auto), so pin it; emit after the pass-through extras, since the parser is last-wins
                _pv_draft_cpu_pin: List[str] = []
                _pv_mmproj_cpu_pin: List[str] = []
                _pv_split_mode_pin: List[str] = []
                # --gpu-layers 0 is not enough: op_offload still sends ops to Metal; --device none drops it.
                _pv_device_pin: List[str] = ["--device", "none"] if _paravirtual_cpu_forced else []
                if (
                    _paravirtual_cpu_forced
                    # preserve_thinking is independent of the thinking gate; Qwen3.8 defaults on, other families off.
                    and _extra_args_mtp_draft_path(
                        [*spec_flags, *(extra_args or [])], env = _child_spec_env(extra_args)
                    )
                    is not None
                    and _paravirtual_draft_ngl_flag(server_caps)
                ):
                    # The draft context has its own device list, so main --device none does not reach it.
                    _pv_draft_cpu_pin = [
                        str(_paravirtual_draft_ngl_flag(server_caps)),
                        "0",
                        "--device-draft",
                        "none",
                    ]
                    logger.warning(
                        "Pinning the speculative drafter to CPU: this Mac's Metal "
                        "device is virtualised."
                    )

                # Emitted after the pass-through extras too, same last-wins reason.
                if _paravirtual_cpu_forced:
                    _pv_split_mode_pin = _paravirtual_split_mode_pin(extra_args)

                self._chat_template_override = chat_template_override
                self._empty_chat_render_refused = False
                _effective_template = self._effective_chat_template(chat_template_override)
                if _effective_template:
                    import tempfile

                    flags = detect_reasoning_flags(
                        _effective_template,
                        self._model_identifier,
                        log_source = (
                            "GGUF chat template override"
                            if chat_template_override
                            else "repaired GGUF chat template"
                        ),
                    )
                    self._supports_reasoning = flags["supports_reasoning"]
                    self._reasoning_style = flags["reasoning_style"]
                    self._reasoning_effort_levels = flags.get("reasoning_effort_levels", [])
                    self._reasoning_always_on = flags["reasoning_always_on"]
                    self._supports_preserve_thinking = flags["supports_preserve_thinking"]
                    self._preserve_thinking_default = flags["preserve_thinking_default"]
                    self._supports_tools = flags["supports_tools"]

                    self._chat_template_file = tempfile.NamedTemporaryFile(
                        mode = "w",
                        encoding = "utf-8",
                        suffix = ".jinja",
                        delete = False,
                        prefix = "unsloth_chat_template_",
                    )
                    self._chat_template_file.write(_effective_template)
                    self._chat_template_file.close()
                    cmd.extend(["--chat-template-file", self._chat_template_file.name])
                    logger.info(f"Using custom chat template file: {self._chat_template_file.name}")

                # Qwen3.5 Small tier (<=9B) defaults reasoning off, and the templates match so it is a launch flag; always-on templates ignore the kwarg, so skip them
                if self._supports_reasoning and not self._reasoning_always_on:
                    thinking_default = True
                    mid = (model_identifier or "").lower()
                    if "qwen3.5" in mid or "qwen3.6" in mid:
                        # Match total params (35B-A3B is 35B), not the active count; scan segments right to left so the leaf size beats a size-like parent dir
                        size_re = r"(?:^|[-_.])(\d+\.?\d*)\s*([bm])(?:$|[-_.])"
                        size_match = None
                        for seg in reversed(mid.replace("\\", "/").split("/")):
                            size_match = re.search(size_re, seg)
                            if size_match:
                                break
                        if size_match:
                            size_b = float(size_match.group(1))
                            if size_match.group(2) == "m":
                                size_b /= 1000.0
                            # <= 9, not < 9: 9B is the top of the Small tier, so it is off too.
                            if size_b <= 9:
                                thinking_default = False
                    # argv beats the env `unsloth start` pins these through, so repeat it.
                    _env_reasoning = _reasoning_from_env(env)
                    if _env_reasoning is not None:
                        thinking_default = _env_reasoning
                    reasoning_kw = self._reasoning_kwargs(
                        thinking_default, _reasoning_effort_from_env(env)
                    )
                    # An effort ladder thinks at every level but "none" (Inkling: 0), low included.
                    self._reasoning_default = reasoning_kw.get(
                        "enable_thinking", reasoning_kw.get("reasoning_effort") not in ("none", 0)
                    )
                    # preserve_thinking is independent of the thinking gate.
                    # Qwen3.8 defaults it on; Qwen3.6, Gemma 4, and every other
                    # supporting family keep the existing off default. The
                    # frontend receives the same resolved value via load/status.
                    # An override takes the default its own text renders, and an explicit
                    # --[no-]reasoning-preserve beats both: llama.cpp applies this kwarg
                    # after that flag, so it would otherwise undo it (#15115).
                    if self._supports_preserve_thinking:
                        if chat_template_override:
                            self._preserve_thinking_default = (
                                _template_preserves_thinking_by_default(
                                    chat_template_override, reasoning_kw
                                )
                            )
                        _preserve = _reasoning_preserve_from_args_or_env(extra_args, env)
                        if _preserve is not None:
                            self._preserve_thinking_default = _preserve
                        reasoning_kw["preserve_thinking"] = self._preserve_thinking_default
                    _reasoning_args = _build_launch_reasoning_args(server_caps, reasoning_kw)
                    cmd.extend(_reasoning_args)
                    logger.info(
                        f"Reasoning model: {reasoning_kw} by default "
                        f"(launched as {' '.join(_reasoning_args)})"
                    )

                reasoning_budget = resolve_reasoning_budget(extra_args, reasoning_budget)
                reasoning_budget_message = resolve_reasoning_budget_message(
                    extra_args, reasoning_budget_message
                )
                cmd.extend(
                    _build_reasoning_budget_flags(
                        server_caps, reasoning_budget, reasoning_budget_message
                    )
                )
                # Check extra_args and env, not the later policy list; only explicit gpu_ids clears LLAMA_ARG_DEVICE.
                self._requested_reasoning_budget = reasoning_budget
                self._requested_reasoning_budget_message = reasoning_budget_message
                # Only a conclusive missing flag means the child ignores LLAMA_ARG_THINK_BUDGET*.
                _budget_env_applies = server_caps.get(
                    "supports_reasoning_budget"
                ) or server_caps.get("reasoning_budget_probe_inconclusive")
                reasoning_budget = (
                    resolve_reasoning_budget_with_env(extra_args, reasoning_budget, env)
                    if _budget_env_applies
                    else reasoning_budget
                )
                reasoning_budget_message = (
                    resolve_reasoning_budget_message_with_env(
                        extra_args, reasoning_budget_message, env
                    )
                    if server_caps.get("supports_reasoning_budget_message")
                    or server_caps.get("reasoning_budget_probe_inconclusive")
                    else reasoning_budget_message
                )
                self._reasoning_budget = reasoning_budget
                self._reasoning_budget_message = reasoning_budget_message

                if launch_mmproj_path and effective_is_vision:
                    cmd.extend(["--mmproj", launch_mmproj_path])
                    logger.info(f"Using mmproj for vision: {launch_mmproj_path}")
                    # The projector ignores --gpu-layers, so pin it to CPU; last flag wins, so emit
                    # after extras.
                    if _mmproj_cpu_pinned:
                        _pv_mmproj_cpu_pin = ["--no-mmproj-offload"]
                        if _paravirtual_cpu_forced:
                            logger.warning(
                                "Disabling mmproj GPU offload: this Mac's Metal "
                                "device is virtualised."
                            )

                # llama-server sends permissive CORS, so without a key any open web page could drive it.
                import secrets as _secrets

                _key_on = _llama_server_api_key_enabled()
                if _key_on and not server_caps.get("supports_api_key_file", True):
                    # A custom build predating --api-key-file would exit on the flag; run it keyless as before.
                    logger.warning(
                        "This llama-server build has no --api-key-file; starting it without an API key."
                    )
                    _key_on = False
                _key_env: dict[str, str] = {}
                if _key_on:
                    self._api_key = _secrets.token_urlsafe(32)
                    # Through a file or the child env, not argv: a command line is readable by every process of this Unix user, and the auth directory is not.
                    _key_argv, _key_env, self._api_key_file = _llama_server_key_launch(
                        self._api_key, self._api_key_file
                    )
                    cmd.extend(_key_argv)
                else:
                    self._api_key = None

                # Windows discrete full offload: drop host-RAM KV checkpoints but keep prompt cache; not on iGPUs, where --cache-ram 0 also drops --cache-idle-slots
                _cache_flags_emitted: list[str] = []
                # llama.cpp reads env before argv, so an appended 0 would override the operator's value.
                _ctx_checkpoints_owned = (
                    ctx_checkpoints
                    if ctx_checkpoints is not None
                    else _env_ctx_checkpoints_override()
                )
                # extra_args, not the memory policy's list: that one is built later, and
                # a gpu_ids pin is the case _strip_device_extra_args removes the flag in.
                # The env twin counts too: only an explicit gpu_ids clears
                # LLAMA_ARG_DEVICE, and llama.cpp reads it before argv, so an automatic
                # load can place against a target the generated pin never names.
                _cache_target_unknown = self._cache_tuning_target_unknown(
                    extra_args, _gpu_ids_own_device_flags, os.environ
                )
                _shared_memory_offload = (
                    sys.platform == "win32"
                    and full_offload_tuning_active
                    and (
                        _cache_target_unknown
                        or self._offload_target_shares_system_memory(
                            is_vulkan_backend = is_vulkan_backend,
                            shared_gpu_ids = _shared_gpu_ids,
                            detected_gpus = _detected_gpus,
                            gpu_indices = gpu_indices,
                        )
                    )
                )
                if _cache_target_unknown and sys.platform == "win32" and full_offload_tuning_active:
                    logger.info(
                        "Keeping the llama-server prompt cache: a user --device in the extra "
                        "arguments overrides the automatic placement, so the offload target "
                        "this tuning would be chosen against is not the one the child gets."
                    )
                elif _shared_memory_offload:
                    logger.info(
                        "Keeping the llama-server prompt cache: this load offloads to a GPU "
                        "that shares system memory, so --cache-ram 0 and --ctx-checkpoints 0 "
                        "would remove prefix reuse without saving a transfer."
                    )
                elif sys.platform == "win32" and full_offload_tuning_active:
                    unsupported_cache_flags: list[str] = []
                    # An explicit control wins over platform tuning (llama.cpp is last-wins).
                    if cache_ram is not None:
                        pass
                    elif server_caps.get("supports_cache_ram"):
                        _cache_flags_emitted.extend(["--cache-ram", "0"])
                    else:
                        unsupported_cache_flags.append("--cache-ram")
                    if _ctx_checkpoints_owned is not None:
                        pass
                    elif server_caps.get("ctx_checkpoints_flag"):
                        _cache_flags_emitted.extend([str(server_caps["ctx_checkpoints_flag"]), "0"])
                    else:
                        unsupported_cache_flags.append("--ctx-checkpoints")
                    cmd.extend(_cache_flags_emitted)
                    if unsupported_cache_flags:
                        logger.info(
                            "Skipping unsupported Windows cache flags for llama-server: %s",
                            ", ".join(unsupported_cache_flags),
                        )

                def _decide_auto_ctx_checkpoints() -> Optional[int]:
                    """The cap for the slots n_parallel currently names, or None to stand down."""
                    return (
                        self._bounded_ctx_checkpoints(
                            n_parallel,
                            server_caps,
                            extra_args,
                            cache_type_kv = cache_type_kv,
                            swa_full = swa_full,
                            flash_attn = planned_flash_attn,
                        )
                        if ctx_checkpoints is None
                        else None
                    )

                _auto_ctx_checkpoints = _decide_auto_ctx_checkpoints()
                # Tracked apart from _cache_flags_emitted, which means "the Windows tuning
                # ran", so the arch-crash respawn can re-decide each one on its own.
                _auto_ckpt_emitted: list[str] = []

                def _emit_auto_ctx_checkpoints(command: list[str]) -> list[str]:
                    """Append the automatic cap unless the Windows tuning already zeroed it."""
                    if _auto_ctx_checkpoints is None:
                        return []
                    if any(
                        _flag_name(str(token)) in _CTX_CHECKPOINTS_FLAGS
                        for token in _cache_flags_emitted
                    ):
                        return []
                    pair = [
                        str(server_caps["ctx_checkpoints_flag"]),
                        str(_auto_ctx_checkpoints),
                    ]
                    command.extend(pair)
                    return pair

                _auto_ckpt_emitted = _emit_auto_ctx_checkpoints(cmd)

                # Record the effective pin so /status never echoes an ordinal the child never saw.
                if _paravirtual_cpu_forced:
                    # --device none pins nothing, so record no effective pin here either.
                    # Ahead of the backend branches: this is the authoritative record and
                    # would otherwise re-echo the device the child never saw.
                    self._gpu_ids = None
                elif is_vulkan_backend:
                    # Only record an explicit Vulkan pin, mirroring CUDA/ROCm; recording an auto pick would make dedupe miss the loaded server
                    self._gpu_ids = (
                        [int(x) for x in _vulkan_pin_ids] if (gpu_ids and _vulkan_pin_ids) else None
                    )
                elif gpu_ids:
                    # Classify on the same stripped view the launch uses; a gpu_ids pin drops the device flags.
                    _effective_pin_ids = (
                        [int(x) for x in gpu_indices]
                        if gpu_indices is not None
                        else [int(x) for x in gpu_ids]
                    )
                    # Keep survivor order, not sorted, so /status round-trips the pin.
                    self._gpu_ids = (
                        [int(x) for x in _effective_pin_ids] if _effective_pin_ids else None
                    )
                else:
                    self._gpu_ids = None

                # Raw requested pin for load dedupe; /status echoes the effective one.
                self._requested_gpu_ids = [int(x) for x in gpu_ids] if gpu_ids else None

                if is_vulkan_backend and _vulkan_pin_ids is not None:
                    cmd += LlamaCppBackend._vulkan_pin_args(_vulkan_pin_ids)

                # Managed flag goes before extras; vetoed flags drop from a launch-only copy. mlock applies only where weights sit in host RAM; asked on every launch
                _mem_env = dict(os.environ)
                if gpu_memory_mode == "manual":
                    self._clear_manual_placement_env(_mem_env)
                # A gpu_ids pin drops the device flags from argv and env below,
                # so classifying on them would pin for a --device cpu the child
                # never sees. Same helpers the launch uses, so they cannot drift.
                _mem_extra_args = extra_args
                if _gpu_ids_own_device_flags:
                    _mem_extra_args = self._strip_device_extra_args(extra_args)
                    self._clear_device_placement_env(_mem_env)
                _mem_keep_resident, _mem_no_reserve = _mem_settings
                _mem_should_mlock = _mem_keep_resident and not _mem_no_reserve
                # Armed here, not at load start: rows from before the Hub download are stale.
                _arm_load_probe_memo()
                _mem_dio_possible = no_reserve_requires_dio(
                    supports_load_mode = load_mode_support(server_caps),
                    gpu_offload_confirmed = True,
                )
                # NOT gated on the toggle: the answer is this launch's recorded placement
                # and a LATER save is compared against it, so gating made the record mean
                # "we did not look". Costs one subprocess, on Windows Vulkan builds only.
                _mem_probe_for_dio = _mem_dio_possible
                # Ordinals the child uses: a pass-through --device wins.
                _mem_physical_indices, _mem_compact_indices = self._effective_gpu_selection(
                    binary,
                    _mem_env,
                    gpu_indices,
                    _mem_extra_args,
                    _mem_dio_possible,
                    is_vulkan_backend,
                )
                _mem_host_resident = self._weights_in_host_memory(
                    fully_gpu_offloaded = fully_gpu_offloaded,
                    gpu_memory_mode = gpu_memory_mode,
                    gpu_layers = gpu_layers,
                    extra_args = _mem_extra_args,
                    gpu_indices = _mem_physical_indices,
                    is_vulkan_backend = is_vulkan_backend,
                    binary = binary,
                    env = _mem_env,
                    # Gate masks every device, so host RAM is used; a skipped page-lock is recorded as deliberate.
                    probe_vulkan = _mem_should_mlock or _mem_probe_for_dio,
                    # Over cmd and extras so a later user --fit wins by last-arg.
                    fit_active = fit_is_effectively_on([*cmd, *(_mem_extra_args or [])], _mem_env),
                )
                # Env as it would be under no-reserve, since the scrub follows the settings, not the live view.
                if _arch_gate_forced_cpu:
                    _mem_host_resident = True
                # Moved up: the managed DirectIO defers to an inherited mode and runs first.
                _fit_load_mode_env_view = dict(_mem_env)
                scrub_memory_env(_fit_load_mode_env_view, _mem_settings)
                # Must be positive: the predicate errs True when unprobed; dio is not gated on projector placement.
                _mem_env_view_no_reserve = dict(_mem_env)
                scrub_memory_env(_mem_env_view_no_reserve, (_mem_keep_resident, True))
                # Whether a relaunch would stream: placement alone is not enough, and a per-model mmap or extra wins.
                _mem_gpu_offload_confirmed = self._gpu_offload_confirmed(
                    binary,
                    _mem_env,
                    _mem_compact_indices,
                    _mem_host_resident,
                    _mem_dio_possible,
                    _mem_extra_args,
                )
                _mem_managed, _mem_extras = apply_model_memory_policy(
                    extra_args,
                    supports_load_mode = load_mode_support(server_caps),
                    weights_in_host_memory = _mem_host_resident,
                    gpu_offload_confirmed = _mem_gpu_offload_confirmed,
                    env = _fit_load_mode_env_view,
                    settings = _mem_settings,
                )
                # What the user wrote, so the strip can tell their pair from ours.
                self._memory_dio_user_tokens = list(_mem_extras or [])
                self._memory_dio_flags = (
                    list(_mem_managed) if tuple(_mem_managed) == MANAGED_DIO_FLAGS else []
                )
                # Defers to Model Memory and extras; an inherited LLAMA_ARG_LOAD_MODE is an explicit choice that would beat the managed block, so only the fit mode yields
                if (
                    _fit_load_mode
                    and not load_mode
                    and memory_env_selects_load_mode(_fit_load_mode_env_view)
                ):
                    logger.info(
                        "Load mode: the environment already selects a loader mode; "
                        "leaving the fit's --load-mode %s off this launch.",
                        _fit_load_mode,
                    )
                    _fit_load_mode = None
                # Retry rungs strip the first matching pair, so a user's duplicate pair must be detected.
                if _fit_load_mode and not load_mode and extra_args_select_load_mode(_mem_extras):
                    logger.info(
                        "Load mode: the extra arguments already select a loader mode; "
                        "leaving the fit's --load-mode %s off this launch.",
                        _fit_load_mode,
                    )
                    _fit_load_mode = None
                _resolved_load_mode = load_mode or _fit_load_mode
                _load_mode_managed, _mem_extras = apply_load_mode_policy(
                    _mem_extras,
                    supports_load_mode = load_mode_support(server_caps),
                    weights_in_host_memory = _mem_host_resident,
                    requested_load_mode = _resolved_load_mode,
                    settings = _mem_settings,
                )

                # Whether a relaunch under no-reserve would really stream. Placement alone
                # is not enough: a per-model mmap, an extra or an inherited choice lands
                # after the managed pair and wins by last-arg, so it demanded a reload no
                # relaunch could satisfy. Toggle-independent, so a LATER save is compared
                # against this launch instead of reading as satisfied.
                def _ask(pair, env_view):
                    return resolve_launch_load_mode(
                        extra_args,
                        supports_load_mode = load_mode_support(server_caps),
                        weights_in_host_memory = _mem_host_resident,
                        gpu_offload_confirmed = _mem_gpu_offload_confirmed,
                        requested_load_mode = _resolved_load_mode,
                        env = env_view,
                        settings = pair,
                    )

                # Would a relaunch ADD the pair. The POLICY emitting it is the question,
                # not the effective mode: a user's own "dio" already streams, and a reload
                # to reach the mode the child is in is a notice that never clears.
                _hypo_emitted, _hypo_effective = _ask(
                    (_mem_keep_resident, True), _mem_env_view_no_reserve
                )
                self._memory_dio_applicable = _hypo_emitted and _hypo_effective

                # The child with the policy off entirely, which is what activity asks.
                _off_view = dict(_mem_env)
                scrub_memory_env(_off_view, (False, False))

                def _mem_env_for(child_env):
                    """`_mem_env`'s placement scrubbing with the CHILD's current
                    visibility. `_mem_env` predates the gate, so probing with it reused
                    the unnarrowed set's verdict."""
                    if child_env is None:
                        return _mem_env
                    view = dict(_mem_env)
                    for _name in _DEVICE_VISIBILITY_ENV:
                        if _name in child_env:
                            view[_name] = child_env[_name]
                        else:
                            view.pop(_name, None)
                    return view

                def _dio_decision_for(
                    devices,
                    *,
                    fully_offloaded,
                    child_env = None,
                ):
                    """Returns (pair, applicable, active) for a changed device set, re-asking every launch question instead of a subset each."""
                    _rung_env = _mem_env_for(child_env)
                    _rung_physical, _rung_compact = self._effective_gpu_selection(
                        binary,
                        _rung_env,
                        devices,
                        _mem_extra_args,
                        _mem_dio_possible,
                        is_vulkan_backend,
                    )
                    host_resident = self._weights_in_host_memory(
                        fully_gpu_offloaded = fully_offloaded,
                        gpu_memory_mode = gpu_memory_mode,
                        gpu_layers = gpu_layers,
                        extra_args = _mem_extra_args,
                        gpu_indices = _rung_physical,
                        is_vulkan_backend = is_vulkan_backend,
                        binary = binary,
                        env = _rung_env,
                        probe_vulkan = _mem_probe_for_dio or _mem_should_mlock,
                        fit_active = fit_is_effectively_on(
                            [*cmd, *(_mem_extra_args or [])], _rung_env
                        ),
                    )
                    confirmed = self._gpu_offload_confirmed(
                        binary,
                        _rung_env,
                        _rung_compact,
                        host_resident,
                        _mem_dio_possible,
                        _mem_extra_args,
                    )

                    def _for(pair, env_view):
                        return resolve_launch_load_mode(
                            extra_args,
                            supports_load_mode = load_mode_support(server_caps),
                            weights_in_host_memory = host_resident,
                            gpu_offload_confirmed = confirmed,
                            requested_load_mode = _resolved_load_mode,
                            env = env_view,
                            settings = pair,
                        )

                    live_emitted, live_effective = _for(_mem_settings, _fit_load_mode_env_view)
                    hypo_emitted, hypo_effective = _for(
                        (_mem_keep_resident, True), _mem_env_view_no_reserve
                    )
                    pair = list(MANAGED_DIO_FLAGS) if (live_emitted and live_effective) else []
                    # Redundant with the user's own loader: nothing a relaunch could undo.
                    active = bool(pair) and live_effective != _for((False, False), _off_view)[1]
                    return pair, (hypo_emitted and hypo_effective), active

                # Only when FIT chose the load mode: a user's own pick survives every fallback, but a placement conclusion must go when the placement does
                self._fit_load_mode_flags = (
                    list(_load_mode_managed) if (_fit_load_mode and not load_mode) else []
                )
                # Remembered so the reload hint and the duplicate-load comparator do
                # not demand an mlock this launch deliberately skipped.
                # A probe forced for the DirectIO decision must not make the page-lock
                # bookkeeping LESS conservative: an unreadable Vulkan probe answers "not an
                # iGPU", which recorded the lock inapplicable on maybe-host-backed weights.
                self._memory_mlock_applicable = _mem_host_resident or (
                    _mem_probe_for_dio
                    and not _mem_should_mlock
                    and is_vulkan_backend
                    and not self._vulkan_probe_answered(binary)
                )
                # An unknown build gets no lock, so none is owed either.
                self._memory_mlock_applicable = keep_resident_lock_applicable(
                    self._memory_mlock_applicable, load_mode_support(server_caps)
                )
                if (
                    _mem_should_mlock
                    and _mem_host_resident
                    and load_mode_support(server_caps) is None
                ):
                    logger.info(
                        "Model Memory: skipping page-lock, the llama-server --help "
                        "probe was inconclusive and newer builds reject --mlock."
                    )
                if _mem_should_mlock and not _mem_host_resident:
                    logger.info(
                        "Model Memory: skipping page-lock, the weights are fully "
                        "offloaded to a discrete GPU; residency is kept by not "
                        "unloading the model."
                    )
                if _mem_managed:
                    cmd.extend(_mem_managed)
                    logger.info(
                        "Model Memory: applying loading policy (%s)",
                        " ".join(_mem_managed),
                    )
                if _load_mode_managed:
                    cmd.extend(_load_mode_managed)
                    # Protected like the extras: the per-model selection is the user's too,
                    # so a rung that stripped a COPY left it behind for the next strip.
                    self._memory_dio_user_tokens = [
                        *self._memory_dio_user_tokens,
                        *_load_mode_managed,
                    ]
                    logger.info("Load mode: %s", " ".join(_load_mode_managed))

                # Before the extras, so a user's own lazy mode wins by last-arg.
                self._lazy_mode_flags = list(_managed_lazy_flags or [])
                if self._lazy_mode_flags:
                    cmd.extend(self._lazy_mode_flags)
                    logger.info("Lookup tables: %s", " ".join(self._lazy_mode_flags))

                # A flag validate_extra_args refuses has an env twin llama.cpp reads before argv, so both must go.
                if _mem_extras:
                    _emit_extra_args = self._strip_mmproj_args(list(_mem_extras))
                    if _gpu_ids_own_device_flags:
                        # gpu_ids owns placement, so remove competing device flags.
                        _before_device_strip = list(_emit_extra_args)
                        _emit_extra_args = self._strip_device_extra_args(_emit_extra_args)
                        if _emit_extra_args != _before_device_strip:
                            logger.info(
                                "Dropped user device placement flags from extra args: "
                                "explicit gpu_ids owns placement."
                            )
                    _emit_extra_args = translate_ngram_mod_args(
                        _emit_extra_args,
                        _caps.get("ngram_mod_flavor"),
                        chain_with_mtp = self._speculative_type == "draft-mtp"
                        or _extra_args_requests_mtp(
                            _emit_extra_args, env = _child_spec_env(_emit_extra_args)
                        ),
                    )
                    cmd.extend(str(a) for a in _emit_extra_args)
                    logger.info(
                        f"Appending user extra args to llama-server: {list(_emit_extra_args)}"
                    )
                # Reconcile the reserve with the child as launched: extras can turn the fitter on or
                # add a margin.
                _sidecar_user_tokens = list(_emit_extra_args) if _mem_extras else []
                if (
                    _sidecar_fit_reserve_eligible_mib
                    and server_caps.get("supports_fit_target")
                    # Re-read: a failed GPU selection falls back to every card.
                    and (_apple_budget_mib > 0 or _sidecar_planned_device_count() == 1)
                ):
                    _sidecar_child_env = dict(os.environ)
                    if gpu_memory_mode == "manual":
                        self._clear_manual_placement_env(_sidecar_child_env)
                    _sidecar_override_flags = _sidecar_reserve_margin_flags(
                        cmd,
                        _emit_extra_args if _mem_extras else (),
                        _sidecar_child_env,
                        max(
                            _VRAM_FLOOR_RESERVE_MIB,
                            self._fit_target_margin_mib(
                                auto_fit = auto_fit,
                                fit_target_delta_mib = _fit_target_delta_mib,
                                floor_mib = _metal_fit_target_floor_mib,
                            ),
                        ),
                        _sidecar_fit_reserve_eligible_mib,
                    )
                    if _sidecar_override_flags:
                        _sidecar_fit_reserve_mib = _sidecar_fit_reserve_eligible_mib
                        _sidecar_override_pairs.append(list(_sidecar_override_flags))
                        cmd.extend(_sidecar_override_flags)
                # Last so it wins: the Metal drafter CPU pin is a correctness fix.
                if _pv_draft_cpu_pin:
                    cmd.extend(_pv_draft_cpu_pin)
                if _pv_mmproj_cpu_pin:
                    cmd.extend(_pv_mmproj_cpu_pin)
                # Last too: a remembered --mmproj-auto would rediscover the projector and restore vision on a text-only load; --no-mmproj-auto does not unload it.
                if disable_vision and not launch_mmproj_path and not _dv_env_mmproj_kept:
                    cmd.append("--no-mmproj-auto")
                # Must be in cmd before the env block reads _zero_offload_keeps_gpu_visible.
                if _pv_split_mode_pin:
                    cmd.extend(_pv_split_mode_pin)
                if _pv_device_pin:
                    cmd.extend(_pv_device_pin)
                # Pinned loads only: under mmap the cache measured slower than none.
                self._moe_cache_flags = []
                if (
                    _moe_cache_candidate
                    and _moe_cache_ram_ok
                    and _placed_by_fitter
                    and _argv_loads_pinned(cmd, _fit_load_mode_env_view)
                ):
                    if _moe_cache_lazy_table and self._lazy_mode_flags:
                        # These flags sit after the spec block, so removing them leaves _spec_start valid.
                        cmd = _without_subsequence(cmd, self._lazy_mode_flags)
                        self._lazy_mode_flags = []
                        logger.info(
                            "MoE cache: reading the %.1f GB lookup table lazily so system "
                            "RAM can pin the %.1f GB of routed experts.",
                            _moe_table_bytes / (1024**3),
                            _moe_expert_bytes / (1024**3),
                        )
                    self._moe_cache_flags = list(_MOE_CACHE_AUTO_TOKENS)
                    cmd.extend(self._moe_cache_flags)
                    logger.info(
                        "MoE experts spill to pinned system RAM on one GPU; adding "
                        "--moe-cache-mib auto so llama.cpp caches hot experts in VRAM."
                    )

                # Without --flash-attn a build cannot run a quantized V cache; reset V before the logged launch, keeping length so _spec_start still indexes the tokens.
                if _flash_attn_known_off:
                    cmd = self._reset_quantized_v_cache(
                        cmd,
                        "this build has no --flash-attn",
                        mla = self._target_kv_symmetry(),
                        # Use os.environ: the child env is not built until below (reading it here is an UnboundLocalError), and Unsloth never rewrites LLAMA_ARG_SPEC_DRAFT_MODEL.
                        draft_mla = self._draft_kv_symmetry(cmd),
                    )

                kv_cache_unified = _kv_unified_from_args(cmd)

                # Library paths so llama-server finds its shared libs and CUDA DLLs.
                env = self._llama_server_env_for_binary(binary)
                # The GGUF header controls pooling; ignore inherited defaults.
                for _pooling_var in (
                    "LLAMA_ARG_POOLING",
                    "LLAMA_ARG_RERANKING",
                    "LLAMA_ARG_EMBEDDINGS",
                ):
                    env.pop(_pooling_var, None)
                if gpu_memory_mode == "manual":
                    self._clear_manual_placement_env(env)
                # Scrub inherited LLAMA_ARG_MLOCK/_MMAP/_LOAD_MODE (read before argv); use the _mem_settings snapshot, not live toggles, or env and argv diverge.
                _mem_scrubbed = scrub_memory_env(env, _mem_settings)
                if _mem_scrubbed:
                    logger.info(
                        "Model Memory owns placement; dropped inherited %s",
                        ", ".join(_mem_scrubbed),
                    )
                # Tracks whether the policy changed this launch; turning the toggles off relaunches only if it did.
                _denied_scrubbed = scrub_denied_env(env)
                if _denied_scrubbed:
                    logger.info(
                        "dropped inherited %s: managed by Unsloth Studio",
                        ", ".join(_denied_scrubbed),
                    )
                # After the scrub, which drops an inherited LLAMA_API_KEY.
                env.update(_key_env)
                # Records what the child will run (env plus last-wins argv, load-mode pair included: none holds an anonymous copy) so user --mlock and --no-mmap count.
                self._record_memory_state(
                    list(_mem_managed) + list(_load_mode_managed) + list(_mem_extras), env
                )
                # Activity is whether turning the toggles off changes this child; shadowed _mem_managed does not.
                _mem_policy_touched_extras = bool(_mem_scrubbed) or _mem_extras != list(
                    extra_args or []
                )
                # Snapshot of what cmd carries, taken before respawns edit it; the arch-crash retry restores it.
                _mem_managed_is_effective = bool(_mem_managed) and (
                    tuple(_mem_managed) != MANAGED_DIO_FLAGS
                    or self._memory_direct_io != _ask((False, False), _off_view)[1]
                )
                self._memory_policy_active = _mem_managed_is_effective or _mem_policy_touched_extras
                self._memory_policy_extras_touched = _mem_policy_touched_extras

                # No --flash-attn in this build: LLAMA_ARG_FLASH_ATTN is unread; keeping it under-sizes the V cache.
                def _snapshot_policy_for_cmd():
                    """Read at call time, not from a snapshot, since a stale snapshot restored an
                    outdated cmd."""
                    return (
                        _mem_host_resident,
                        self._memory_state,
                        self._memory_direct_io,
                        self._memory_dio_applicable,
                        list(self._memory_dio_flags),
                        self._memory_policy_active,
                        self._memory_mlock_applicable,
                    )

                _mem_policy_for_cmd = _snapshot_policy_for_cmd()
                # Drop inherited LLAMA_ARG_THREADS, which would force hardware_concurrency().
                if "--threads" not in cmd:
                    env.pop("LLAMA_ARG_THREADS", None)
                # Drop an inherited LLAMA_ARG_CTX_SIZE=0: it pins native ctx and disables --fit.
                if (
                    not any(token in cmd for token in ("-c", "--ctx-size"))
                    and _env_asks_for_the_native_context(env)
                    and self._metal_drops_zero_ctx_override(0, _caller_owns_budget)
                ):
                    env.pop("LLAMA_ARG_CTX_SIZE", None)
                    logger.warning(
                        "Dropping the inherited LLAMA_ARG_CTX_SIZE=0: on Metal it pins "
                        "the model's native length and disables --fit."
                    )
                # A build whose help has no --flash-attn never registers the option,
                # so it never reads LLAMA_ARG_FLASH_ATTN either (llama.cpp resolves
                # each env var through the arg that declares it). Leaving an
                # inherited one in place would only fool the accounting below, which
                # reads env plus argv and would then record flash attention as on for
                # a server running without it, under-sizing the padded V cache.
                if _flash_attn_known_off and self._drop_env_flash_attn(env):
                    logger.info(
                        "Dropped inherited LLAMA_ARG_FLASH_ATTN: this build has no --flash-attn."
                    )
                # Same for an env-only quantized V cache, which the argv reset
                # above cannot reach.
                if _flash_attn_known_off and self._drop_env_quantized_v_cache(
                    env,
                    mla = self._target_kv_symmetry(),
                    draft_mla = self._draft_kv_symmetry(cmd, env),
                ):
                    logger.info(
                        "Dropped inherited quantized V-cache env: this build has no --flash-attn."
                    )

                # Drop an unparsable inherited KV type: it aborts both the launch and the layer retry; apply in every split mode, since llama.cpp parses it the same way.
                for _ct_var in ("LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V"):
                    # Compare against the ORIGINAL, not a stripped copy: kv_cache_type_from_str needs an exact match, and a stripped compare leaves the whitespace on the child.
                    _ct_orig = env.get(_ct_var) or ""
                    _ct_norm = _ct_orig.strip().lower()
                    if not _ct_norm:
                        continue
                    if _ct_norm not in _VALID_CACHE_TYPES:
                        env.pop(_ct_var, None)
                        logger.info(
                            "Dropped inherited %s=%s: llama.cpp does not accept that "
                            "KV cache type, and it would abort the load.",
                            _ct_var,
                            _ct_orig,
                        )
                    elif _ct_norm != _ct_orig:
                        # Clears an inherited non-layer LLAMA_ARG_SPLIT_MODE so the child cannot override the layer plan.
                        env[_ct_var] = _ct_norm

                # Reconcile the inherited LLAMA_ARG_* split env with Unsloth's final
                # decision (the previews above apply the same helper).
                self._reconcile_split_mode_env(env, tensor_parallel = tensor_parallel)

                # A gpu_ids pin also owns inherited device placement.
                if _gpu_ids_own_device_flags:
                    self._clear_device_placement_env(env)

                # Point at the Vision switch only for an existing mmproj file; a stale path drops nothing to restore.
                _cpu_fallback_eligible = self._auto_vulkan_cpu_fallback_eligible(
                    binary,
                    intent,
                    extra_args,
                    env,
                )
                _vulkan_suspect = False

                # Argv drop cannot reach an env drafter; spec types accumulate, so scrub both.
                if _pv_draft_unpinnable or not _extra_args_set_spec_type(extra_args):
                    for _pv_spec_var in _SPEC_ENV_VARS:
                        env.pop(_pv_spec_var, None)

                # Also clear LLAMA_ARG_MMPROJ(_URL): the guard only cleared our resolved path.
                if disable_vision:
                    # Keep an audio-only encoder: a projector is not always a vision tower, and clearing it loses audio input; unknown still counts as image-capable.
                    _dv_env_mmproj = (env.get("LLAMA_ARG_MMPROJ") or "").strip()
                    if not _mmproj_env_is_audio_only(_dv_env_mmproj):
                        # Same signal the resolved path records, for the same reason:
                        # this one IS restored by turning Vision back on, so the composer
                        # must point at the switch rather than at a missing mmproj. Only
                        # for a file that exists -- a stale path drops nothing, and
                        # blaming the switch there sends the user to a control that
                        # cannot help. Unknown-but-present reads image-capable, matching
                        # the rule the scrub itself just used.
                        if _dv_env_mmproj and Path(_dv_env_mmproj).is_file():
                            _dv_dropped_image_projector = True
                        env.pop("LLAMA_ARG_MMPROJ", None)
                    # No such reprieve for the URL: it names a download that has not
                    # happened, so there is no file to classify and nothing to do but
                    # honor the switch.
                    env.pop("LLAMA_ARG_MMPROJ_URL", None)
                if _paravirtual_cpu_forced:
                    for _pv_mmproj_var in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"):
                        env.pop(_pv_mmproj_var, None)
                    # LLAMA_ARG_OVERRIDE_TENSOR puts weights back on Metal regardless of --gpu-layers 0; strip only GPU-bound overrides, keeping inherited CPU-target ones.
                    _pv_env_ot = env.get("LLAMA_ARG_OVERRIDE_TENSOR")
                    if _pv_env_ot and not _paravirtual_strip_gpu_overrides(
                        ["-ot", str(_pv_env_ot)]
                    ):
                        env.pop("LLAMA_ARG_OVERRIDE_TENSOR", None)

                # Windows full offload: PASSIVE OMP + 2 threads stop spin-wait CPU burn.
                if sys.platform == "win32" and full_offload_tuning_active:
                    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
                    if not threads_overridden:
                        env.setdefault("OMP_NUM_THREADS", "2")

                # Managed allocations trade ROCm APU carve-out capacity for host RAM.
                # Vulkan indices are not ROCm ids. Track ownership so later retries
                # preserve inherited values.
                _unified_env_applied = False
                _unified_opt_out = self._unified_memory_opted_out(env)
                _unified_opt_in = self._unified_memory_opted_in(env)
                # The arch gate and arch-crash retry re-select without rebinding gpu_indices.
                _unified_gpu_indices = gpu_indices

                def _unified_need_now(argv = None, mtp_engages = None):
                    """Bytes a forced full offload puts on the device, or None."""
                    run_argv = cmd if argv is None else argv
                    # The trailing blocks load only when the EMBEDDED head drafts: a
                    # sidecar wins over it (llama.cpp has_dft()), wherever it lands.
                    if mtp_engages is None:
                        engages = _mtp_will_engage and not _separate_draft_launches
                    else:
                        engages = mtp_engages
                    if model_size is None or not self._launch_forces_full_offload(run_argv, env):
                        return None
                    need = int(model_size)
                    if mmproj_size and not _argv_keeps_projector_on_gpu(run_argv, env):
                        need -= int(mmproj_size)
                    # Re-priced rather than cleared, so a spill that still stands is still reported.
                    layout = self._tensor_spill_layout(model_path, all_shards = True)
                    if layout is None or not getattr(layout, "complete", True):
                        return None
                    if int(getattr(layout, "lm_head_bytes", 0) or 0) and not (
                        _override_places_tensor_on_device("token_embd.weight", run_argv, env)
                    ):
                        need -= int(getattr(layout, "token_embd_bytes", 0) or 0)
                    per_layer = int(getattr(layout, "per_layer_embd_bytes", 0) or 0)
                    scan = self._gguf_tensor_scan(model_path) if per_layer else None
                    if scan is not None and any(
                        _override_places_tensor_on_device(name, run_argv, env)
                        for name in scan[1]
                        if _is_per_layer_embedding(name)
                    ):
                        per_layer = 0
                    need -= per_layer
                    if not engages and self._nextn_predict_layers:
                        need -= int(getattr(layout, "excluded_block_bytes", 0) or 0)
                    return need

                def _unified_withdraw_if_unneeded(
                    run_cmd,
                    *,
                    mtp_engages = None,
                    why = "",
                ):
                    """Drop the variable THIS launch set once a respawn stops needing it."""
                    nonlocal _unified_env_applied
                    # Whether the user could stop the spill at all, independent of the managed-memory decision above.
                    try:
                        _carveout_need = _unified_need_now(argv = run_cmd, mtp_engages = mtp_engages)
                    except Exception:
                        _carveout_need = None
                    self._record_carveout_advice(
                        _unified_gpu_indices,
                        _carveout_need,
                        is_vulkan_backend = is_vulkan_backend,
                        shared_gpu_ids = _shared_gpu_ids,
                        detected_gpus = _detected_gpus,
                        target_unknown = _cache_target_unknown,
                        forced_cpu = _arch_gate_forced_cpu,
                    )
                    if not _unified_env_applied:
                        return
                    if self._unified_memory_for_launch(
                        _unified_gpu_indices,
                        _unified_need_now(argv = run_cmd, mtp_engages = mtp_engages),
                        opted_in = _unified_opt_in,
                    ):
                        return
                    env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
                    _unified_env_applied = False
                    logger.info(
                        "%s no longer needs managed memory; dropped "
                        "GGML_CUDA_ENABLE_UNIFIED_MEMORY.",
                        why or "The retry",
                    )

                _unified_need = _unified_need_now()
                if _unified_opt_out:
                    # ggml tests presence, so a user's "0" would enable it; only absence is off.
                    if env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None) is not None:
                        logger.info(
                            "Unified memory opted out: unset GGML_CUDA_ENABLE_UNIFIED_MEMORY"
                        )
                elif not is_vulkan_backend and self._unified_memory_for_launch(
                    gpu_indices, _unified_need, opted_in = _unified_opt_in
                ):
                    _unified_env_applied = "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env
                    env.setdefault("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1")
                    logger.info(
                        "AMD unified-memory APU: set GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 (%s)",
                        "UNSLOTH_ENABLE_UNIFIED_MEMORY=1"
                        if _unified_opt_in
                        else "the weights outgrow the carve-out and host RAM is the larger pool",
                    )

                # Computed here: zero-offload is masked to CUDA_VISIBLE_DEVICES=-1 and moves no tensors between GPUs.
                self._record_carveout_advice(
                    gpu_indices,
                    _unified_need,
                    is_vulkan_backend = is_vulkan_backend,
                    shared_gpu_ids = _shared_gpu_ids,
                    detected_gpus = _detected_gpus,
                    target_unknown = _cache_target_unknown,
                    forced_cpu = _arch_gate_forced_cpu,
                )

                # GGML_CUDA_P2P=0 still enables P2P (presence, not value); env builder strips it already, so this covers envs assembled another way.
                self._sanitize_p2p_env(env)

                # gpu_indices are PCI ids only when PCI_BUS_ID is pinned, else FASTEST_FIRST may name other cards.
                _cpu_only_zero_offload = (
                    gpu_memory_mode == "manual"
                    and gpu_layers == 0
                    and not is_vulkan_backend
                    and not self._zero_offload_keeps_gpu_visible(cmd, env)
                )

                # Whether the child will resolve gpu_indices as the nvidia-smi ids
                # the P2P gate verifies. The launch below pins PCI_BUS_ID for an
                # explicit pick, and for an auto-fit selection only when NO mask was
                # inherited (nothing to reinterpret); anything else leaves CUDA on
                # FASTEST_FIRST, where those ids may name other cards.
                # That auto-fit arm also requires the ids to BE PCI indices: when the
                # nvidia-smi query fails _get_gpu_memory falls back to torch CUDA
                # ordinals, and pinning PCI order re-emits those numbers as a
                # different set of cards. An explicit pick is unaffected, its ids
                # come from the UI's own PCI-ordered picker.
                _p2p_launch_order_pinned = bool(gpu_ids) or (
                    os.environ.get("CUDA_VISIBLE_DEVICES") is None
                    and LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is True
                )
                # Only for an unconfirmed fabric; datacenter boxes are left to _apply_datacenter_env, which warns.
                _p2p_ids_are_pci = (
                    LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is True
                    or os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
                )

                # When the arch gate empties the pool, mask every GPU: the child would enumerate and abort on them.
                if (
                    env.get("GGML_CUDA_P2P")
                    and not _cpu_only_zero_offload
                    and not is_vulkan_backend
                    and not self._is_datacenter_gpu(gpu_indices)
                    and self._effective_gpu_count(gpu_indices) > 1
                    and not LlamaCppBackend._warned_no_nvlink
                ):
                    _p2p_veto = self._p2p_veto_reason(
                        gpu_indices, _p2p_launch_order_pinned, _p2p_ids_are_pci
                    )
                    if _p2p_veto is not None:
                        LlamaCppBackend._warned_no_nvlink = True
                        logger.warning(
                            "GGML_CUDA_P2P=%s is set in the environment but this "
                            "selection has no confirmed NVLink fabric: %s. Peer copies "
                            "over PCIe can be discarded silently while CUDA still "
                            "reports success, which surfaces as garbled output rather "
                            "than an error. Verify with scripts/p2p_integrity_probe.py "
                            "and unset the variable entirely (not =0, which reads as ON "
                            "upstream) if it fails (#10613).",
                            env["GGML_CUDA_P2P"],
                            _p2p_veto,
                        )

                # DC NVIDIA: FP32 accum, launch queues, P2P once NVLink is confirmed. Opt out: UNSLOTH_DISABLE_DC_TUNING=1, or UNSLOTH_DISABLE_DC_P2P=1 for P2P only.
                if not (is_vulkan_backend or _cpu_only_zero_offload):
                    self._apply_datacenter_env(
                        env,
                        gpu_indices,
                        p2p_opted_out = self._p2p_user_opted_out(),
                        launch_order_pinned = _p2p_launch_order_pinned,
                        ids_are_pci_indices = _p2p_ids_are_pci,
                    )

                # ROCm needs HIP_VISIBLE_DEVICES too. Zero-offload hides GPUs (a visible CUDA device costs ~0.5 GB reported free); user device pins keep control.
                _child_gpu_physical_ids: Optional[tuple[int, ...]] = None
                # companion flags fitted to the child's mask, refitted by the arch-crash retry
                _companion_fit_mask: Optional[list[int]] = None
                _companion_added_device: Optional[list[str]] = None
                if not is_vulkan_backend and _gpu_mem:
                    _child_gpu_physical_ids = self._unmasked_child_gpu_physical_ids()
                _parent_view_gpu_ids = _child_gpu_physical_ids

                # The CUDA SM gate goes here, where the child's GPU visibility is
                # finally known: both arms below write CUDA_VISIBLE_DEVICES=-1, so no
                # kernel image loads and the abort this pre-empts cannot happen.
                # Deciding it from the request over-refused -- speculative "auto", a
                # --no-mmproj-offload projector and a --spec-draft-ngl 0 drafter all
                # read as GPU-bearing there, only the resolved argv knows. Still
                # before every spawn below and outside the fit's try/except, so the
                # refusal cannot be swallowed into the --fit on fallback.
                # Pinned ids are checked against nvidia-smi's caps only in its index space
                # (torch-fallback ordinals are not); otherwise "any visible", as before.
                _sm_gate_ids_are_pci = not is_vulkan_backend and _p2p_ids_are_pci
                if not (_cpu_only_zero_offload or _arch_gate_forced_cpu):
                    # Pinned launches check the pinned cards, not "any visible".
                    _sm_gate = self._cuda_sm_gate_error(
                        binary, gpu_indices if _sm_gate_ids_are_pci else None
                    )
                    if _sm_gate:
                        raise RuntimeError(_sm_gate)
                # The arch gate emptying the pool lands here for the same reason
                # (#7624): no device set this binary can launch on, so the child must
                # not see the cards it would enumerate and abort on. gpu_indices is
                # None there, so this branch writes the mask.
                if _cpu_only_zero_offload or _arch_gate_forced_cpu:
                    self._emit_child_gpu_visibility(env, "-1")
                    _child_gpu_physical_ids = ()
                    # Tensor mode with no visible device fails in llama_prepare_model_devices; strip it.
                    _cpu_cmd = (
                        self._without_flags(cmd, ("--split-mode", "-sm", "--tensor-split", "-ts"))
                        if _arch_gate_forced_cpu
                        else None
                    )
                    if _cpu_cmd is not None:
                        logger.info(
                            "Forced-CPU launch: dropped the split-mode/tensor-split "
                            "flags, which abort a server with no visible device."
                        )
                        cmd = _cpu_cmd
                        # Recorded, or the next Apply reads the normalized server as a
                        # different one and rebuilds a multi-GB model every time.
                        if self._tensor_parallel:
                            self._arch_gate_dropped_tensor_parallel = True
                        if self._tensor_split:
                            self._arch_gate_dropped_tensor_split = tuple(self._tensor_split)
                        self._tensor_parallel = False
                        self._tensor_split = None
                    if _arch_gate_forced_cpu:
                        # Not gated on the argv strip: llama.cpp reads the LLAMA_ARG_*
                        # twins whether or not we emitted the flags, so an inherited
                        # tensor mode aborts a child with no visible device on its own.
                        self._clear_split_placement_env(env)
                elif gpu_indices is not None and not is_vulkan_backend:
                    # Pin PCI_BUS_ID so UI GPU 1 is llama.cpp GPU 1; auto-fit only when no inherited mask exists, since a scheduler mask would be re-read in new order.
                    if _p2p_launch_order_pinned:
                        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                    # Keep a mask that deliberately reorders the same cards.
                    _pin_ids = list(gpu_indices)
                    # Record the rewritten --tensor-split, or /status pairs shares with the wrong device.
                    _inherited_order = (
                        None if gpu_ids else self._inherited_child_gpu_order(_pin_ids)
                    )
                    if _inherited_order is not None:
                        # A user --tensor-split is positional; decline to reorder under it.
                        if _extra_args_have_tensor_split(extra_args, env):
                            logger.info(
                                "Keeping ascending GPU order: a pass-through "
                                "--tensor-split is positional over it."
                            )
                        elif self._repoint_emitted_tensor_split(cmd, _pin_ids, _inherited_order):
                            logger.info(
                                "Pinning the child's GPU order to the inherited "
                                "visibility mask: %s (ascending would be %s).",
                                _inherited_order,
                                _pin_ids,
                            )
                            _pin_ids = _inherited_order
                            # --fit on or a manual all-GPU ratio leaves the child unpinned, so an uncovered card can reach it.
                            if "--tensor-split" in cmd and gpu_memory_mode != "manual":
                                _rewritten = cmd[cmd.index("--tensor-split") + 1]
                                try:
                                    self._auto_tensor_split_emitted = self._auto_split_fingerprint(
                                        [float(x) for x in str(_rewritten).split(",")]
                                    )
                                except (TypeError, ValueError):
                                    self._auto_tensor_split_emitted = None
                    # Mask at ROCr/HSA: HIP-only masking still enumerates and segfaults on unsupported GPUs. Skipped for uuid/MIG masks and an inherited LLAMA_ARG_DEVICE.
                    _companion_widen = ""
                    if not self._visibility_mask_is_unmappable() and not (
                        _extra_args_main_device(cmd) is None
                        and str(env.get("LLAMA_ARG_DEVICE", "")).strip()
                    ):
                        _had_main_device = _extra_args_main_device(cmd) is not None
                        _pin_ids, _companion_widen = _widen_pin_ids_for_companion_devices(
                            cmd,
                            _pin_ids,
                            self._resolve_visible_physical_ids(),
                            allowed_ids = gpu_ids or None,
                        )
                        if _companion_widen:
                            _companion_fit_mask = list(_pin_ids)
                            if not _had_main_device and _extra_args_main_device(cmd) is not None:
                                _companion_added_device = list(cmd[-2:])
                            if gpu_ids:
                                self._adopt_widened_pin(_pin_ids)
                    if _companion_widen:
                        logger.info(
                            "Companion device flags name GPUs by their unpinned "
                            "numbering; fitted the pinned mask to them: %s",
                            _companion_widen,
                        )
                        # The widened mask can add a card the gate above never saw.
                        if _sm_gate_ids_are_pci:
                            _sm_gate = self._cuda_sm_gate_error(binary, _pin_ids)
                            if _sm_gate:
                                raise RuntimeError(_sm_gate)
                    _child_mask = LlamaCppBackend._pinned_child_visibility(_pin_ids)
                    if _child_mask is None:
                        logger.info(
                            "Keeping the inherited CUDA_VISIBLE_DEVICES=%s: GPU ids %s "
                            "cannot be mapped to it.",
                            os.environ.get("CUDA_VISIBLE_DEVICES"),
                            _pin_ids,
                        )
                    else:
                        self._emit_child_gpu_visibility(env, _child_mask, prefer_rocr = True)
                    _child_gpu_physical_ids = tuple(int(i) for i in _pin_ids)
                    _launch_pinned_ids = list(_pin_ids)
                elif not is_vulkan_backend and not gpu_ids:
                    # Record it, or the duplicate-load check sees a different server and rebuilds on every Apply.
                    _survivors = self._arch_gate_survivors(binary)
                    if _survivors:
                        # Masking re-indexes survivors, so drop the full-count ratio.
                        _gated_cmd = self._without_tensor_split(cmd)
                        if _gated_cmd is not None:
                            logger.warning(
                                "Dropping the manual --tensor-split: it was sized "
                                "for every visible GPU, and the installed build "
                                "has no kernels for some of them."
                            )
                            cmd = _gated_cmd
                            # With one survivor, --split-mode tensor is a no-op but still reported active, so drop the mode too.
                            self._arch_gate_dropped_tensor_split = (
                                tuple(self._tensor_split) if self._tensor_split else None
                            )
                            self._tensor_split = None
                        # Narrowing to one survivor leaves --split-mode tensor a no-op
                        # still REPORTED as active (tensor_parallel drives the UI and
                        # arms the MTP crash watchdog), manual mode having admitted it
                        # on the FULL device count. _without_tensor_split takes only the
                        # ratio, so drop the mode too, as the arch-crash retry does.
                        if len(_survivors) < 2 and self._tensor_parallel:
                            _no_tensor_mode = self._without_flags(cmd, ("--split-mode", "-sm"))
                            if _no_tensor_mode is not None:
                                cmd = _no_tensor_mode
                            self._arch_gate_dropped_tensor_parallel = True
                            self._tensor_parallel = False
                            logger.info(
                                "The arch gate leaves one GPU; dropped tensor "
                                "parallelism, which needs at least two."
                            )
                        logger.info(
                            "The installed llama.cpp build has no kernels for GPU(s) "
                            "left visible; pinning the child to %s.",
                            _survivors,
                        )
                        # Re-asked on the narrowed set: an unsupported APU answered host-resident for the unnarrowed launch.
                        self._clear_split_placement_env(env)
                        # Process-wide, so recompute after changing devices; an inherited value is never overwritten.
                        _unified_gpu_indices = _survivors
                        _survivors_gain_unified = self._unified_memory_for_launch(
                            _survivors, _unified_need_now(), opted_in = _unified_opt_in
                        )
                        if _unified_env_applied and not _survivors_gain_unified:
                            env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
                            _unified_env_applied = False
                            logger.info(
                                "Arch gate narrowed the launch onto GPU(s) the "
                                "unified-memory swap cannot help; dropped "
                                "GGML_CUDA_ENABLE_UNIFIED_MEMORY."
                            )
                        elif (
                            _survivors_gain_unified
                            and not _unified_opt_out
                            and "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env
                        ):
                            env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
                            _unified_env_applied = True
                            logger.info(
                                "Arch gate narrowed the launch onto a unified-memory "
                                "APU the weights outgrow; set "
                                "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1."
                            )
                        self._emit_child_gpu_visibility(
                            env, ",".join(str(i) for i in _survivors), prefer_rocr = True
                        )
                        _child_gpu_physical_ids = tuple(int(i) for i in _survivors)
                        # Narrower than any pin above, so it replaces it.
                        _launch_pinned_ids = list(_survivors)
                        # Taken against the UNNARROWED set, where an unsupported APU
                        # answered host-resident for the whole launch; this gate removes
                        # it. fully_offloaded False: the fitter still owns placement.
                        _gate_dio, _gate_applicable, _gate_active = _dio_decision_for(
                            _survivors, fully_offloaded = False, child_env = env
                        )
                        self._memory_dio_applicable = _gate_applicable
                        # BOTH directions: narrowing can also take the offload away, and
                        # the pair then reads CPU-resident layers into allocated buffers.
                        if self._memory_dio_flags and not _gate_dio:
                            cmd = self._drop_managed_dio(
                                cmd,
                                "the arch gate's surviving GPU(s) no longer confirm a "
                                "full offload",
                            )
                            self._memory_dio_applicable = _gate_applicable
                            self._record_memory_state(cmd, env)
                            _mem_policy_for_cmd = _snapshot_policy_for_cmd()
                        elif _gate_dio and not self._memory_dio_flags:
                            cmd = [*cmd, *_gate_dio]
                            self._memory_dio_flags = list(_gate_dio)
                            self._memory_policy_active = _gate_active or self._memory_policy_active
                            self._record_memory_state(cmd, env)
                            _mem_policy_for_cmd = _snapshot_policy_for_cmd()
                            logger.info(
                                "Model Memory: applying %s; the arch gate pins this "
                                "launch to discrete GPU(s) %s.",
                                " ".join(_gate_dio),
                                _survivors,
                            )
                        # And the carve-out advice with it: upstream priced the
                        # UNNARROWED set, which _rocm_selected_pool_mib declines on a
                        # mixed host, so a model outgrowing the surviving APU's
                        # carve-out was never advised about.
                        try:
                            _gated_carveout_need = _unified_need_now(argv = cmd)
                        except Exception:
                            _gated_carveout_need = None
                        self._record_carveout_advice(
                            _survivors,
                            _gated_carveout_need,
                            is_vulkan_backend = is_vulkan_backend,
                            shared_gpu_ids = _shared_gpu_ids,
                            detected_gpus = _detected_gpus,
                            target_unknown = _cache_target_unknown,
                            forced_cpu = _arch_gate_forced_cpu,
                        )
                    elif manual_tensor_split_emitted:
                        # A manual per-GPU ratio across ALL GPUs (no explicit pick, so
                        # no mask above): the UI built --tensor-split in ascending
                        # physical/PCI order, so pin the child's enumeration to match.
                        # The whole visible set stays in use, only its order is fixed.
                        _child_gpu_physical_ids = self._pin_visible_gpu_order_for_split(env)

                _child_has_no_gpu = (
                    # each names a device present but unusable, so it owns the empty pool
                    _arch_gate_forced_cpu
                    or _paravirtual_cpu_forced
                    # neither says a device exists, so an empty pool stays unreadable
                    or (
                        bool(_detected_gpus)
                        and (
                            _cpu_only_zero_offload or self._binary_ships_no_gpu_backend(binary, env)
                        )
                    )
                )
                # Pool read before anything spawns (a later reading subtracts the model being priced); the replay and guard must share it, so only the VRAM credit differs.
                _preflight_avail_mib = (
                    self._available_system_memory_mib() if _cpu_fallback_eligible else None
                )
                # Only count VRAM on cards this child can reach.
                _reachable_gpus = self._rows_the_child_can_reach(_detected_gpus, _launch_pinned_ids)
                # reads the argv the child gets, not the mid-fit state that produced it
                _offload_msg = self._launch_host_shortfall_message(
                    cmd,
                    _reachable_gpus,
                    env,
                    child_has_no_gpu = _child_has_no_gpu,
                    avail_mib = _preflight_avail_mib,
                    shared_gpu_ids = _shared_gpu_ids,
                    integrated_gpu_ids = _vulkan_igpu_ids,
                )
                if (
                    _offload_msg
                    and intent.refuse_partial_gpu_fit
                    and gpu_memory_mode != "manual"
                    and self._other_servers_running()
                ):
                    raise GpuMemoryShortError(_offload_msg, host = True)
                # Snapshotted BEFORE the override note is appended below, so a re-price
                # that promotes this notice appends that note once rather than twice.
                _host_ram_msg = _offload_msg

                def _launch_is_oversized() -> bool:
                    """Oversized by APU preflight, the warning just raised, or pricing with the opt-out ignored; the opt-out hides a message and must not withdraw the rewrite."""
                    if _apu_ram_oversized or _offload_msg:
                        return True
                    if not self._host_offload_warning_opted_out():
                        return False
                    return bool(
                        self._launch_host_shortfall_message(
                            cmd,
                            _reachable_gpus,
                            env,
                            child_has_no_gpu = _child_has_no_gpu,
                            avail_mib = _preflight_avail_mib,
                            shared_gpu_ids = _shared_gpu_ids,
                            honor_opt_out = False,
                            integrated_gpu_ids = _vulkan_igpu_ids,
                        )
                    )

                cmd, _pageable_note = self._page_an_oversized_unmapped_load(
                    cmd, env, _launch_is_oversized
                )
                if _pageable_note:
                    # Record the argv that actually launches, and snapshot it too, since arch-crash retry restores it.
                    if _offload_msg:
                        _offload_msg += _pageable_note
                    # First notice wins, so amend whatever is already recorded.
                    self._amend_load_warning(_pageable_note)
                    # The recorded state has to describe the argv that launches, or the
                    # reload comparator judges this child against a mode it no longer
                    # runs. The snapshot too: the arch-crash retry restores it over
                    # `cmd`, which now carries the override.
                    self._record_memory_state(cmd, env)
                    _mem_policy_for_cmd = _snapshot_policy_for_cmd()
                self._record_load_warning(_offload_msg)

                # Captured before any text-only fallback strips it from cmd.
                launched_with_mmproj = "--mmproj" in cmd

                # One-shot --fit off retry (fit aborts on some ROCm), never when fit was requested or passed in extras. Argv launched; text-only retry strips --mmproj.
                _last_spawn_cmd = list(cmd)
                _spawn_cwd: Optional[str] = None
                # An inherited LLAMA_ARG_FIT=off belongs to the user, and the flip would override it on the CLI.
                _did_rocm_retry = False

                def _enable_managed_fit_for_no_flash(fa_cmd: list) -> list:
                    """Restores Unsloth's own --fit on for a --flash-attn off respawn, which is re-
                    placed and needs it."""
                    if "--fit" not in fa_cmd:
                        return fa_cmd
                    if any(
                        _flag_name(str(token)) in {"-fit", "--fit"} for token in (extra_args or ())
                    ):
                        return fa_cmd
                    # Record from the argv, not the managed parts: dropping --load-mode none returns the child to auto.
                    if _user_fit_disabled(extra_args, env = env):
                        return fa_cmd
                    if _placement_is_fitter_proof(fa_cmd, gpu_layers = _manual_gpu_layers, env = env):
                        return fa_cmd
                    index = fa_cmd.index("--fit")
                    if index + 1 >= len(fa_cmd) or str(fa_cmd[index + 1]).strip().lower() == "on":
                        return fa_cmd
                    flipped = list(fa_cmd)
                    flipped[index + 1] = "on"
                    logger.info(
                        "Re-enabling the fitter for the --flash-attn off retry: the padded "
                        "V cache the retry runs with is bigger than the placement the fit "
                        "priced with flash attention on."
                    )
                    return flipped

                def _drop_fit_load_mode_for_no_flash(fa_cmd: list) -> list:
                    """Drops Unsloth's --load-mode on a --flash-attn off respawn (OOM instead of paging); do NOT clear the record, since the CPU fallback replays these tokens."""
                    if not self._fit_load_mode_flags:
                        return fa_cmd
                    stripped = _without_subsequence(fa_cmd, self._fit_load_mode_flags)
                    if stripped != fa_cmd:
                        # Tee llama-server output to a log; label and attempt index stop same-second respawns clobbering it.
                        self._record_memory_state(stripped, env)
                        logger.info(
                            "Load mode: dropping the fit's --load-mode none for the "
                            "--flash-attn off retry; the no-flash launch has a "
                            "footprint the fit did not price."
                        )
                    return stripped

                # Memoed only when the load ends terminally, so a recovering fallback is not blocked.
                _sched_abort_seen = False

                def _vulkan_wait_args(label):
                    """(timeout, kwargs) for a health wait: a stuck Vulkan driver burns CPU
                    and touches pages forever (#142), so auto-Vulkan launches get a ceiling."""
                    timeout = self._HEALTH_STALL_TIMEOUT_S
                    if not _cpu_fallback_eligible or intent.cpu_fallback or label == "-cpu":
                        return timeout, {}
                    per_gib = self._AUTO_VULKAN_START_S_PER_GIB
                    floor = timeout
                    if _vulkan_suspect and not label:
                        timeout = min(timeout, self._SUSPECT_VULKAN_STALL_TIMEOUT_S)
                        floor = timeout
                    elif not _vulkan_suspect:
                        # Listed device: ends with advice only, so CPU work (shader compile) is progress.
                        per_gib *= self._LISTED_VULKAN_CEILING_FACTOR
                        floor = timeout * self._LISTED_VULKAN_CEILING_FACTOR
                    ceiling = (model_size or 0) / (1 << 30) * per_gib
                    return timeout, {
                        "cpu_counts_as_progress": not _vulkan_suspect,
                        "max_total_s": max(floor, ceiling),
                    }

                def _spawn_and_wait(run_cmd, *, label = ""):
                    """Start llama-server with run_cmd and wait for health.

                    Up to four launches: the first, one ROCm env correction, one
                    retry without the MoE expert cache, and one --fit recovery.
                    Separate flags, so a library mix does not spend the fit slot
                    and a VRAM crash after the correction can still fit-retry.
                    """
                    # _mem_host_resident too: the --fit on retry re-arms the
                    # page-lock and writes it back, which without this makes the
                    # read below an UnboundLocalError instead.
                    nonlocal _last_spawn_cmd, _mem_host_resident, _did_rocm_retry
                    nonlocal _sched_abort_seen, _sidecar_fit_reserve_mib
                    # Single revocation point for the tensor-spill plan; label is empty only on the first spawn.
                    if label:
                        run_cmd = self._drop_tensor_spill(run_cmd, label.lstrip("-") or "retry")
                        run_cmd = self._drop_managed_lazy_mode(
                            run_cmd, label.lstrip("-") or "retry"
                        )
                    # The cache only rides the first pinned launch: every retry re-places.
                    if label or not _argv_loads_pinned(run_cmd, env):
                        run_cmd = self._drop_moe_cache(
                            run_cmd, f"{label.lstrip('-') or 'unpinned'} launch"
                        )
                    _fit_retry_allowed = self._fit_off_retry_eligible(run_cmd, use_fit)
                    _did_fit_retry = False
                    _did_moe_cache_retry = False
                    for _spawn_attempt in (0, 1, 2, 3):
                        if _spawn_attempt:
                            run_cmd = self._drop_managed_lazy_mode(run_cmd, "startup")
                        # Defensive kill: drop an orphan Popen a concurrent load may
                        # have stored before we overwrite the reference (#5161).
                        # Also reaps the crashed first attempt on the retry pass.
                        self._kill_process()
                        self._child_gpu_physical_ids = _child_gpu_physical_ids

                        self._stdout_lines = []
                        # Tee llama-server output to a dedicated log file so a
                        # post-mortem has the full trail even when the parent only
                        # kept the last 50 lines. Path is under the studio home.
                        # ``label`` (MTP fallback) and the attempt index (--fit
                        # off retry) keep a respawn within the same epoch second
                        # from truncating the crash log a retry warning just
                        # pointed the user at.
                        self._llama_log_fh = None
                        try:
                            log_dir = _swa_cache_path().parent / "logs" / "llama-server"
                            log_dir.mkdir(parents = True, exist_ok = True)
                            self._llama_log_path = log_dir / (
                                f"llama-{int(time.time())}{label}-port-{self._port}"
                                f"-try{_spawn_attempt}.log"
                            )
                            self._llama_log_fh = open(
                                self._llama_log_path,
                                "w",
                                encoding = "utf-8",
                                buffering = 1,
                            )
                            # After the open, so the cap counts this file and never deletes
                            # it. On the retry path the earlier attempts' logs are what a
                            # reader wants, and keep-newest holds on to them.
                            prune_log_dir(log_dir, "llama-*.log", protect = self._llama_log_path)
                            logger.info(f"llama-server stdout/stderr -> {self._llama_log_path}")
                        except (OSError, UnicodeDecodeError) as e:
                            # Best-effort; never block the load on logging.
                            logger.warning(f"Could not open llama-server log file: {e}")
                            self._llama_log_path = None
                        # Every launch: a retry can turn the fitter on or re-mask onto several
                        # cards.
                        _spawn_devices = _extra_args_main_device(run_cmd) or env.get(
                            "LLAMA_ARG_DEVICE"
                        )
                        _spawn_device_count = (
                            _distinct_device_count(_spawn_devices)
                            if _spawn_devices
                            else max(len(_child_gpu_physical_ids or ()), len(_detected_gpus))
                            if _child_gpu_physical_ids is None
                            or _child_gpu_physical_ids is _parent_view_gpu_ids
                            else len(_child_gpu_physical_ids)
                        )
                        if _sidecar_fit_reserve_eligible_mib:
                            # Re-derived each spawn: drop Studio's earlier reserve-bearing margins,
                            # keep the user's.
                            run_cmd = _drop_sidecar_margins(
                                list(run_cmd),
                                _integrity_flags,
                                self._ctx_integrity_flags(
                                    *_integrity_args,
                                    fit_target_delta_mib = _fit_target_delta_mib,
                                    fit_target_floor_mib = _metal_fit_target_floor_mib,
                                ),
                                _sidecar_override_pairs,
                                user_args = _sidecar_user_tokens,
                            )
                            _sidecar_fit_reserve_mib = 0
                        run_cmd = self._per_device_fit_target_argv(
                            run_cmd, env, _fit_target_margins_by_idx
                        )
                        _spawn_base_by_dev: Optional[list[float]] = None
                        if _fit_target_margins_by_idx:
                            try:
                                _spelled = [
                                    float(v)
                                    for v in str(
                                        _last_flag_value(run_cmd, {"-fitt", "--fit-target"})
                                    ).split(",")
                                ]
                            except ValueError:
                                _spelled = []
                            if len(_spelled) == 1:
                                _spelled = _spelled * _spawn_device_count
                            if len(_spelled) == _spawn_device_count:
                                _spawn_base_by_dev = _spelled
                        # --rpc devices come first in the child's list, so local positional shares
                        # would land on the wrong cards.
                        _spawn_rpc = not _spawn_devices and bool(
                            (_extra_args_device(run_cmd, {"--rpc"}) or "").strip()
                            or str(env.get("LLAMA_ARG_RPC", "")).strip()
                        )
                        # Several cards: each keeps its share of the drafter (whole draft layers) in
                        # a per-device list.
                        _spawn_shares: Optional[list[float]] = [1.0]
                        _spawn_output_dev: Optional[int] = None
                        _spawn_pinned_at: Optional[int] = None
                        _spawn_cpu_only = _device_selection_is_cpu(run_cmd, env) or (
                            _child_gpu_physical_ids is not None
                            and _child_gpu_physical_ids is not _parent_view_gpu_ids
                            and not _child_gpu_physical_ids
                        )
                        if _spawn_cpu_only:
                            _spawn_shares = None
                        elif (
                            _spawn_device_count > 1
                            and _extra_args_split_mode(run_cmd, env) == "tensor"
                        ):
                            # --split-mode tensor folds the cards into one device: no per-device
                            # reserve.
                            _spawn_shares = None
                        elif _spawn_rpc:
                            _spawn_shares = None
                        elif _spawn_device_count > 1 and _split_mode_confines_to_one_device(
                            run_cmd, env
                        ):
                            if _spawn_base_by_dev and len(set(_spawn_base_by_dev)) > 1:
                                # llama.cpp keeps only devices[main_gpu] (src/llama.cpp): the
                                # reserve goes in entry 0 of a list that keeps the other margins.
                                _main = self._effective_main_gpu(run_cmd, env) or 0
                                if 0 <= _main < _spawn_device_count:
                                    _spawn_shares = [
                                        1.0 if i == 0 else 0.0 for i in range(_spawn_device_count)
                                    ]
                                    _spawn_output_dev = 0
                                else:
                                    _spawn_shares = None
                        elif _spawn_device_count > 1:
                            _spawn_shares = None
                            _spawn_pinned_at = _draft_pin_device_index(
                                _extra_args_draft_device_pin(run_cmd),
                                _spawn_devices,
                                _spawn_device_count,
                            )
                            if _spawn_pinned_at is not None:
                                _spawn_shares = [
                                    1.0 if i == _spawn_pinned_at else 0.0
                                    for i in range(_spawn_device_count)
                                ]
                                _spawn_output_dev = _spawn_pinned_at
                            # Positional shares need the child to enumerate in the ids' order.
                            _spawn_order_unproven = self._positional_child_order_unproven(env)
                            if (
                                not is_vulkan_backend
                                and not _spawn_devices
                                and _extra_args_draft_device_pin(run_cmd) is None
                                and not _spawn_order_unproven
                            ):
                                _free_by_idx = dict(gpus or _detected_gpus or ())
                                if (
                                    _child_gpu_physical_ids is None
                                    or _child_gpu_physical_ids is _parent_view_gpu_ids
                                ):
                                    _det_ids = [idx for idx, _free in _detected_gpus]
                                    _spawn_ids = (
                                        self._inherited_child_gpu_order(_det_ids) or _det_ids
                                    )
                                else:
                                    _spawn_ids = list(_child_gpu_physical_ids)
                                _spawn_free = [_free_by_idx.get(i) for i in _spawn_ids]
                                # The drafter inherits the target's --tensor-split
                                # (common_base_params_to_speculative).
                                _spawn_base = self._fit_target_margin_mib(
                                    auto_fit = auto_fit,
                                    fit_target_delta_mib = _fit_target_delta_mib,
                                    floor_mib = _metal_fit_target_floor_mib,
                                )
                                _weights = None
                                if _spawn_free and all(f and f > 0 for f in _spawn_free):
                                    _spawn_margins = _fit_margins_by_device(
                                        extra_args, env, 0.0, len(_spawn_free)
                                    )
                                    _bases = (
                                        _spawn_base_by_dev
                                        if _spawn_base_by_dev
                                        and len(_spawn_base_by_dev) == len(_spawn_free)
                                        else [_spawn_base] * len(_spawn_free)
                                    )
                                    _weights = [
                                        max(1.0, float(f) - max(b, m))
                                        for f, m, b in zip(_spawn_free, _spawn_margins, _bases)
                                    ]
                                _ts = _extra_args_tensor_split(run_cmd, env)
                                if _ts and _spawn_ids:
                                    _ts_w = [
                                        float(_ts[i]) if i < len(_ts) else 0.0
                                        for i in range(len(_spawn_ids))
                                    ]
                                    if sum(_ts_w) > 0 and all(w >= 0 for w in _ts_w):
                                        _weights = _ts_w
                                if _weights:
                                    _layers = _sidecar_draft_layers or 0
                                    _cum = list(
                                        itertools.accumulate(w / sum(_weights) for w in _weights)
                                    )
                                    # "row" assigns blocks like "layer" (CUDA / ROCm register no
                                    # split buffer type).
                                    if _layers > 0:
                                        # llama.cpp's layer split: block il goes to the first device
                                        # whose cumulative share exceeds il / (n_layer + 1).
                                        _counts = [0] * len(_weights)
                                        for _il in range(_layers):
                                            _dev = bisect.bisect_right(_cum, _il / (_layers + 1))
                                            _counts[min(_dev, len(_weights) - 1)] += 1
                                        _spawn_shares = [c / _layers for c in _counts]
                                    else:
                                        _spawn_shares = [w / sum(_weights) for w in _weights]
                                    # Logits over a borrowed output run on the card holding the
                                    # target's layer n_layer.
                                    _t_layers = (
                                        _layers
                                        if _sidecar_owns_output
                                        else int(getattr(self, "_n_layers", 0) or 0) or _layers
                                    )
                                    if _t_layers > 0:
                                        _spawn_output_dev = min(
                                            bisect.bisect_right(_cum, _t_layers / (_t_layers + 1)),
                                            len(_weights) - 1,
                                        )
                        if (
                            _sidecar_fit_reserve_eligible_mib
                            and _spawn_shares is not None
                            and (
                                _spawn_pinned_at is not None
                                or _spawn_draft_on_target(run_cmd, _spawn_devices, env)
                            )
                            and server_caps.get("supports_fit_target")
                        ):
                            _spawn_total = (
                                _sidecar_reserve_reprice(list(run_cmd), env)
                                if _sidecar_reserve_reprice is not None
                                else _sidecar_fit_reserve_eligible_mib
                            )
                            # The vocabulary-wide graph stays whole on the output card.
                            _spawn_graph = min(
                                _sidecar_reserve_graph_mib[0] if len(_spawn_shares) > 1 else 0,
                                max(0, _spawn_total - _sidecar_compute_mib),
                            )
                            _spawn_body = max(0, _spawn_total - _sidecar_compute_mib - _spawn_graph)
                            _last = (
                                _spawn_output_dev
                                if _spawn_output_dev is not None
                                else max(i for i, f in enumerate(_spawn_shares) if f > 0)
                                if any(f > 0 for f in _spawn_shares)
                                else len(_spawn_shares) - 1
                            )
                            _spawn_head = 0
                            if (
                                _sidecar_owns_output
                                and len(_spawn_shares) > 1
                                and not _draft_share_unknown
                            ):
                                _head_layout = self._tensor_spill_layout(
                                    _sidecar_reserve_path, all_shards = True
                                )
                                if _head_layout is not None and _head_layout.complete:
                                    _spawn_head = min(
                                        _spawn_body,
                                        math.ceil(_head_layout.lm_head_bytes / (1024 * 1024)),
                                    )
                                    _spawn_body -= _spawn_head
                                else:
                                    _spawn_head = _spawn_body
                            _spawn_per_device = [
                                math.ceil(_spawn_body * f)
                                + (_sidecar_compute_mib if f > 0 or i == _last else 0)
                                + (_spawn_graph + _spawn_head if i == _last else 0)
                                for i, f in enumerate(_spawn_shares)
                            ]
                            _spawn_reserve = max(_spawn_per_device)
                            _spawn_flags = _sidecar_reserve_margin_flags(
                                run_cmd,
                                extra_args,
                                env,
                                self._fit_target_margin_mib(
                                    auto_fit = auto_fit,
                                    fit_target_delta_mib = _fit_target_delta_mib,
                                    floor_mib = _metal_fit_target_floor_mib,
                                ),
                                _spawn_reserve,
                                per_device = _spawn_per_device
                                if len(_spawn_per_device) > 1
                                else None,
                                base_by_device = _spawn_base_by_dev,
                            )
                            if _spawn_flags:
                                run_cmd = [*run_cmd, *_spawn_flags]
                                if _spawn_total != _sidecar_fit_reserve_eligible_mib:
                                    logger.info(
                                        "Sidecar reserve repriced for this launch's argv: "
                                        "%.2f GB.",
                                        _spawn_total / 1024,
                                    )
                                _sidecar_fit_reserve_mib = _spawn_reserve
                                if list(_spawn_flags) not in _sidecar_override_pairs:
                                    _sidecar_override_pairs.append(list(_spawn_flags))
                        if (
                            _sidecar_fit_reserve_eligible_mib
                            and _sidecar_kind_for_fit
                            and not _spawn_cpu_only
                        ):
                            self._log_sidecar_fit_reserve(
                                run_cmd,
                                env,
                                _sidecar_kind_for_fit,
                                _sidecar_reserve_path,
                                _sidecar_fit_reserve_mib,
                                _sidecar_reserve_logged,
                            )
                        logger.info(
                            f"Starting llama-server: {' '.join(self._redacted_cmd_for_log(run_cmd))}"
                        )
                        _last_spawn_cmd = list(run_cmd)
                        # Read off the argv actually spawned rather than the intent, so
                        # every emitter is covered by construction.
                        self._idle_slot_clearing_active = _idle_slot_clearing_active(
                            run_cmd,
                            supports_cache_ram = bool(server_caps.get("supports_cache_ram")),
                        )
                        # Check and publish under one lock, across Popen only, never the wait.
                        with self._spawn_lock:
                            if self._spawn_is_stale():
                                logger.info("app is shutting down; not starting llama-server")
                                self._close_attempt_log()
                                self._health_wait_cancelled = True
                                return False
                            _spawned = _spawn_on_lifetime_thread(
                                lambda: subprocess.Popen(
                                    run_cmd,
                                    stdout = subprocess.PIPE,
                                    stderr = subprocess.STDOUT,
                                    text = True,
                                    encoding = "utf-8",
                                    errors = "replace",
                                    env = env,
                                    cwd = _spawn_cwd,
                                    **_windows_hidden_subprocess_kwargs(),
                                    **_child_popen_kwargs(),
                                )
                            )
                            self._process = _spawned
                            self._process_env = (_spawned, env)
                            _invalidate_gpu_memory("llama-server spawned")
                            # Inside the lock: written after a sweep reaped the child,
                            # _pid_start_identity yields no start time, and the bare pid
                            # left behind is one a later launch kills blind.
                            self._note_server_pid(_spawned.pid)
                        # mark_process_shutting_down skips _spawn_lock, so recheck after spawn.
                        if self._spawn_is_stale():
                            logger.info(
                                "shutdown began during the spawn; killing the new llama-server"
                            )
                            self._kill_process()
                            self._close_attempt_log()
                            self._health_wait_cancelled = True
                            return False
                        # is_active covers it from here, so drop the pre-spawn flag.
                        # Re-armed below if this attempt fails: the retry rungs respawn from the same settings.
                        self._memory_pending_launch = None

                        # Background thread to drain stdout (prevents pipe deadlock)
                        self._stdout_thread = threading.Thread(
                            target = self._drain_stdout, daemon = True, name = "llama-stdout"
                        )
                        self._stdout_thread.start()
                        _wait_timeout, _wait_kwargs = _vulkan_wait_args(label)
                        if self._wait_for_health(
                            timeout = _wait_timeout,
                            cancelled = _load_cancelled,
                            **_wait_kwargs,
                        ):
                            if _did_rocm_retry:
                                # Proved on this host: every later child skips the
                                # prepend instead of crashing into it first.
                                self._remember_bundle_only_rocm(binary)
                            return True
                        if getattr(self, "_health_wait_cancelled", False):
                            return False
                        # The window is open again: the rungs below respawn from the SAME
                        # captured settings while a crashed child leaves is_active false.
                        # Here rather than per rung, so a later rung cannot forget it.
                        self._memory_pending_launch = _mem_settings
                        # Read once, like the wait itself: a cleared reference is a
                        # teardown, not a startup crash, and re-reading the
                        # attribute per term would race the shutdown thread again.
                        _crashed_proc = self._process
                        _startup_crashed = (
                            _crashed_proc is not None
                            and _crashed_proc.poll() is not None
                            and _crashed_proc.returncode != 0
                        )
                        # Split-axis aborts and old builds refusing tensor-mode quantized KV are not
                        # fit problems.
                        _startup_output = "\n".join(self._stdout_lines[-50:])
                        # Wider tail: the GGML_ASSERT line scrolls past the [New LWP] dump.
                        if _startup_crashed and self._is_sched_reserve_abort(
                            "\n".join(self._stdout_lines[-200:])
                        ):
                            _sched_abort_seen = True
                        # Scan the full buffer: gdb output can bury the assert.
                        _tensor_capability_crash = self._is_tensor_split_assert(
                            "\n".join(self._stdout_lines)
                        ) or self._is_tensor_quant_kv_unsupported(_startup_output)
                        _hip_rocr_mismatch = self._is_bundled_hip_rocr_mismatch(_startup_output)
                        # A startup crash means the spill prediction was wrong, so placement returns to llama.cpp.
                        _capability_crash = _tensor_capability_crash or self._is_kv_unified_refused(
                            "\n".join(self._stdout_lines)
                        )
                        _capability_crash = _capability_crash or (
                            self._cuda_kernel_image_error("\n".join(self._stdout_lines)) is not None
                        )
                        if (
                            not _did_rocm_retry
                            and _startup_crashed
                            and not _tensor_capability_crash
                            and _hip_rocr_mismatch
                        ):
                            # Drop only the prepended lib dirs; the rest of env holds the scrubs.
                            _bundle_only = self._llama_server_env_for_binary(
                                binary, use_system_rocm = False
                            )
                            _retry_ld = _bundle_only.get("LD_LIBRARY_PATH", "")
                            # "" on a host that never had the var: nothing to undo.
                            if _retry_ld and _retry_ld != env.get("LD_LIBRARY_PATH"):
                                logger.warning(
                                    "llama-server crashed during startup (exit code %s) "
                                    "because a bundled ROCm library could not resolve a "
                                    "symbol against the system ROCm; retrying once "
                                    "with the bundled runtime only. Crash log: %s",
                                    _crashed_proc.returncode,
                                    self._llama_log_path,
                                )
                                env["LD_LIBRARY_PATH"] = _retry_ld
                                _did_rocm_retry = True
                                continue
                        if (
                            not _did_moe_cache_retry
                            and _startup_crashed
                            and not _capability_crash
                            and not _hip_rocr_mismatch
                        ):
                            # OOM or a cache error with the cache on: same placement without it.
                            _uncached = self._drop_moe_cache(run_cmd)
                            _crash_text = "\n".join(self._stdout_lines[-200:])
                            if _uncached != run_cmd and (
                                _MOE_CACHE_ERROR_RE.search(_crash_text)
                                or self._is_gpu_memory_start_failure(_crash_text)
                            ):
                                logger.warning(
                                    "llama-server crashed during startup (exit code %s) "
                                    "with --moe-cache-mib auto; retrying once without "
                                    "the MoE expert cache. Crash log: %s",
                                    _crashed_proc.returncode,
                                    self._llama_log_path,
                                )
                                run_cmd = _uncached
                                _did_moe_cache_retry = True
                                continue
                        if (
                            not _did_fit_retry
                            and _startup_crashed
                            and not _capability_crash
                            and not _hip_rocr_mismatch
                        ):
                            # A spill-planned launch that crashed on startup. The
                            # two recoveries below cannot reach it: the spill arm
                            # leaves fully_gpu_offloaded False and runs under
                            # use_fit=True with an explicit --fit, so
                            # _fit_off_retry_eligible is False on both counts and
                            # the load would skip to the terminal/CPU fallbacks
                            # even though the --fit on placement it replaced would
                            # have worked. A crash means the prediction was wrong,
                            # so hand placement back to llama.cpp. Self-gating:
                            # _drop_tensor_spill returns run_cmd unchanged when no
                            # plan is present.
                            _reverted = self._drop_tensor_spill(run_cmd, "startup failure")
                            if _reverted != run_cmd:
                                logger.warning(
                                    "llama-server crashed during startup (exit code %s) "
                                    "with a tensor-spill plan; the plan was optimistic, "
                                    "retrying once with --fit on so llama.cpp can place "
                                    "the model itself. Crash log: %s",
                                    _crashed_proc.returncode,
                                    self._llama_log_path,
                                )
                                run_cmd = _reverted
                                self._record_memory_state(run_cmd, env)
                                _did_fit_retry = True
                                self._vram_fit_context_length = None
                                continue
                        if (
                            not _did_fit_retry
                            and fully_gpu_offloaded
                            and _startup_crashed
                            and not _capability_crash
                            and not _hip_rocr_mismatch
                        ):
                            # Our VRAM math was optimistic: fall back to --fit on and let llama.cpp offload; DSpark takes this path too (fit only skips its sidecar reserve).
                            logger.warning(
                                "llama-server crashed during startup (exit code %s) "
                                "with forced --fit off; the fit estimate was optimistic, "
                                "retrying once with --fit on so it can offload. "
                                "Crash log: %s",
                                _crashed_proc.returncode,
                                self._llama_log_path,
                            )
                            # The fit's --load-mode none rested on a proof that just failed; drop only Unsloth's own tokens.
                            _run = list(run_cmd)
                            if "--fit" in _run:
                                _run[_run.index("--fit") + 1] = "on"
                            if self._mixed_split_flags:
                                # common/fit.cpp aborts a multi-device fit on a set split.
                                _run = _without_subsequence(_run, self._mixed_split_flags)
                                self._auto_tensor_split_emitted = None
                            # The --fit off launch emitted no margin; the next spawn adds the
                            # reserve.
                            # Same reasoning as the page-lock below, one step
                            # earlier: the fit picked "none" because it had proved
                            # the load fits, and that proof is what just failed, so
                            # give the retry llama.cpp's auto back. Only Unsloth's
                            # own tokens; a user's --load-mode is untouched.
                            if self._fit_load_mode_flags:
                                _run = _without_subsequence(_run, self._fit_load_mode_flags)
                                # The record is NOT cleared: it describes `cmd`, which
                                # this retry never touched and the arch-crash and CPU
                                # fallbacks respawn from, so clearing here would leave
                                # them carrying a --load-mode none they can no longer
                                # name. Stripping an already-stripped argv is a no-op.
                                logger.info(
                                    "Load mode: dropping the fit's --load-mode none "
                                    "for the --fit on retry; the fit it was derived "
                                    "from did not hold."
                                )
                            # The full-offload guess that skipped the page-lock just failed; appending it wins as last arg.
                            _run = self._drop_managed_dio(
                                _run,
                                "the --fit on retry gives up the confirmed full offload",
                                clear_record = False,
                            )
                            # With --fit off, -ngl defaults to every layer, so the host-resident verdict may no longer hold.
                            if _mem_should_mlock and not _mem_host_resident:
                                _retry_lock = keep_resident_lock_flags(
                                    load_mode_support(server_caps)
                                )
                                _run.extend(_retry_lock)
                                _mem_host_resident = True
                                # An unknown build gets no lock, so none is owed either.
                                self._memory_mlock_applicable = bool(_retry_lock)
                                if _retry_lock:
                                    self._memory_policy_active = True
                                    logger.info(
                                        "Model Memory: re-applying the page-lock for "
                                        "the --fit on retry; the full-offload "
                                        "prediction did not hold."
                                    )
                            run_cmd = _run
                            self._record_memory_state(_run, env)
                            _did_fit_retry = True
                            # The fit that vouched for the ceiling is what just failed.
                            self._vram_fit_context_length = None
                            continue
                        if (
                            not _did_fit_retry
                            and _fit_retry_allowed
                            and _startup_crashed
                            and not _capability_crash
                            and not _hip_rocr_mismatch
                        ):
                            logger.warning(
                                "llama-server crashed during startup (exit code %s) "
                                "with the default memory-fit step enabled; Unsloth "
                                "already verified the model fits, retrying once "
                                "with --fit off. Crash log: %s",
                                _crashed_proc.returncode,
                                self._llama_log_path,
                            )
                            run_cmd = [*run_cmd, "--fit", "off"]
                            # The mirror of the --fit on retry above. Turning the
                            # fitter off leaves -ngl at its default, which llama.cpp
                            # resolves to every layer, so the fitted attempt's
                            # host-resident verdict can stop holding. Drop a
                            # page-lock the retry no longer needs rather than
                            # reserving a full host copy of a fully offloaded model.
                            if _mem_host_resident and not self._weights_in_host_memory(
                                fully_gpu_offloaded = True,
                                gpu_memory_mode = gpu_memory_mode,
                                gpu_layers = gpu_layers,
                                extra_args = _mem_extra_args,
                                gpu_indices = gpu_indices,
                                is_vulkan_backend = is_vulkan_backend,
                                binary = binary,
                                env = _mem_env,
                            ):
                                _mem_host_resident = False
                                # So a later save is not compared against a lock this
                                # launch dropped.
                                self._memory_mlock_applicable = False
                                if _mem_managed:
                                    run_cmd = _without_subsequence(run_cmd, _mem_managed)
                                    # The managed flag was the policy's only mark on
                                    # this child unless it also scrubbed or stripped,
                                    # and a child equal to an unmanaged one must not
                                    # be torn down when the toggles go off.
                                    self._memory_policy_active = _mem_policy_touched_extras
                                    logger.info(
                                        "Model Memory: dropping the page-lock for "
                                        "the --fit off retry; it offloads every layer."
                                    )

                                # The other direction, which the page-lock arm cannot
                                # reach: nothing was emitted BECAUSE the fitted attempt
                                # read host-resident, the verdict this retry overturns.
                                # Appended, so a hand-typed flag still wins last. The
                                # fitter is OFF here, so -ngl is every layer.
                                _retry_dio, _retry_applicable, _retry_active = _dio_decision_for(
                                    gpu_indices, fully_offloaded = True, child_env = env
                                )
                                self._memory_dio_applicable = _retry_applicable
                                if _retry_dio and not self._memory_dio_flags:
                                    run_cmd = [*run_cmd, *_retry_dio]
                                    self._memory_dio_flags = list(_retry_dio)
                                    self._memory_policy_active = (
                                        _retry_active or self._memory_policy_active
                                    )
                                    logger.info(
                                        "Model Memory: applying %s for the --fit off "
                                        "retry; it offloads every layer.",
                                        " ".join(_retry_dio),
                                    )
                                self._record_memory_state(run_cmd, env)
                            _did_fit_retry = True
                            continue
                        return False

                _launched_mmproj_has_audio = self._mmproj_has_audio

                def _raise_terminal_load_failure(detail: str) -> NoReturn:
                    if _sched_abort_seen and not _load_cancelled():
                        LlamaCppBackend._record_sched_reserve_abort(binary, _abort_memo_model)
                    if intent.cpu_fallback:
                        self._cleanup_failed_cpu_fallback()
                    # No child will carry this budget; left set, the route reads it
                    # ahead of is_active and asks for a reload with nothing loaded.
                    self._vram_fraction_pending = None
                    raise RuntimeError(detail)

                def _vulkan_stall_pending() -> bool:
                    return (
                        getattr(self, "_health_wait_stalled", False)
                        and _cpu_fallback_eligible
                        and not intent.cpu_fallback
                        and not _load_cancelled()
                    )

                def _cleanup_cancelled_load(message: str) -> None:
                    logger.info(message)
                    self._cleanup_failed_cpu_fallback()
                    self._unload_audio_codec()
                    self._healthy = False
                    self._vram_fraction_pending = None

                def _finish_cancelled_health_wait(message: str) -> bool:
                    if not getattr(self, "_health_wait_cancelled", False):
                        return False
                    _cleanup_cancelled_load(message)
                    return True

                def _try_auto_vulkan_cpu_fallback(
                    failed_cmd: list[str],
                    failed_rc: Optional[int],
                    *,
                    terminal: bool = True,
                    stalled: bool = False,
                ) -> bool:
                    """Replay an eligible signal crash without devices.

                    ``terminal`` off reports a failed CPU spawn as "not recovered"
                    instead of raising, for a caller that still holds another argv
                    worth replaying. ``stalled`` replays a child that never became
                    healthy and made no loading progress (already reaped by the
                    caller), the hang a crash gate cannot see.
                    """
                    nonlocal intent, gpu_memory_mode, gpu_layers, n_cpu_moe
                    nonlocal _replay_pageable_override
                    nonlocal tensor_parallel, tensor_split, gpu_indices, use_fit, _spawn_cwd
                    # GGML_ASSERT is a signal on POSIX and a CRT abort (exit 3) on
                    # MSVC, so Windows needs both. Cancel-checked like every retry
                    # here: an /unload racing the crash must not leave a server up.
                    _crashed = stalled or (
                        self._is_signal_crash(failed_rc)
                        or (sys.platform == "win32" and self._is_abort_exit(failed_rc))
                    )
                    if not _crashed or not _cpu_fallback_eligible or _load_cancelled():
                        return False
                    fallback_has_mmproj = self._launch_has_mmproj(failed_cmd, env)
                    prepared = self._prepare_cpu_fallback_launch(
                        binary,
                        failed_cmd,
                        env,
                        server_caps,
                        drop_full_offload_threads = _drop_full_offload_threads,
                        # The same host pool the preflight and the reprice below read,
                        # so the rewrite and the advisory answer one question once.
                        avail_mib = _preflight_avail_mib,
                    )
                    if prepared is None:
                        return False
                    if _load_cancelled():
                        # This argv's drafter may be the cause, which a GPU crash does not show; keep the staged runtime.
                        _cleanup_cancelled_load("Load cancelled while staging the CPU fallback")
                        return False
                    replay, _spawn_cwd, _cpu_pageable_note = prepared
                    # The rewrite mutates the shared env in place, so it runs once across replays; carry its note forward, or a later replay reports no override.
                    if _cpu_pageable_note:
                        _replay_pageable_override = _cpu_pageable_note
                    _cpu_pageable_note = _cpu_pageable_note or _replay_pageable_override

                    logger.warning(
                        "The auto-selected Vulkan backend %s during startup; "
                        "retrying once with llama.cpp devices disabled.",
                        "stopped making progress" if stalled else "hard-crashed",
                    )
                    if not _spawn_and_wait(replay, label = "-cpu"):
                        if _finish_cancelled_health_wait(
                            "Load cancelled during the staged CPU replay health wait"
                        ):
                            return False
                        if not terminal:
                            # Record _last_spawn_cmd, the argv that started, since the replay can drop load-mode and DirectIO.
                            self._kill_process()
                            return False
                        _proc_snap1 = self._process  # snapshot: re-reading races the teardown
                        cpu_rc = _proc_snap1.poll() if _proc_snap1 is not None else None
                        detail = self._classify_llama_start_failure(
                            "\n".join(self._stdout_lines[-50:]),
                            gguf_path,
                            self._model_identifier,
                            cpu_rc,
                            binary,
                            self._llama_log_path,
                            (self._api_key,),
                            self._extra_args,
                        )
                        if _sched_abort_seen:
                            LlamaCppBackend._record_sched_reserve_abort(binary, _abort_memo_model)
                        self._cleanup_failed_cpu_fallback()
                        self._vram_fraction_pending = None
                        raise RuntimeError(detail)

                    # The replay dropped the fit's --load-mode none, so a record taken
                    # off the launch argv describes a reservation this child does not
                    # make, and a later "Don't reserve system RAM" would demand a
                    # pointless reload. From _last_spawn_cmd, the argv that really
                    # started, so this cannot undo a record a --fit retry wrote.
                    #
                    # Or the pageable override rewrote a mode this replay no longer runs
                    # with: the record has to describe the argv that started, whichever
                    # of the two changed it, or the reload comparator judges this child
                    # against a mode it does not have.
                    #
                    # Unconditional now: the replay also withdraws the managed DirectIO, a
                    # third way to go stale, and all of them land in _last_spawn_cmd.
                    self._record_memory_state(_last_spawn_cmd, env)
                    intent = self._apply_cpu_fallback_state(
                        intent,
                        reason = "vulkan_startup_stall" if stalled else "vulkan_startup_crash",
                        is_vision = fallback_has_mmproj,
                        mmproj_has_audio = (
                            _launched_mmproj_has_audio if fallback_has_mmproj else False
                        ),
                        disable_vision = disable_vision,
                        # The same expression the load's own assignment uses, since the
                        # replay carries the same switch and the same dropped projector.
                        vision_disabled_by_user = bool(
                            is_vision and disable_vision and _dv_dropped_image_projector
                        ),
                    )
                    gpu_memory_mode = intent.gpu_memory_mode
                    gpu_layers = intent.gpu_layers
                    n_cpu_moe = intent.n_cpu_moe
                    tensor_parallel = intent.tensor_parallel
                    tensor_split = intent.tensor_split
                    gpu_indices = None
                    use_fit = False
                    # Appended after the host-RAM and offload notices so it never replaces a memory warning.
                    self._reprice_after_cpu_only_fallback(
                        host_msg = _host_ram_msg,
                        cpu_cmd = _last_spawn_cmd,
                        env = env,
                        avail_mib = _preflight_avail_mib,
                        pageable_note = _cpu_pageable_note,
                    )
                    # This return skips the commit block, so record drafter state here.
                    self._mtp_draft_path = launch_mtp_draft_path
                    self._mtp_draft_suppressed_path = _suppressed_draft_path
                    self._mtp_draft_suppressed_reason = _suppressed_draft_reason
                    logger.warning(
                        "llama-server loaded successfully on CPU after the "
                        "auto-selected Vulkan backend %s. GPU acceleration "
                        "is disabled for this model session.",
                        "stalled" if stalled else "crashed",
                    )
                    return True

                def _recover_vulkan_stall(stalled_cmd: list[str]) -> bool:
                    """A probe that could not enumerate plus no loading work replays on CPU (True
                    once up, False if it could not start); any other stall fails with advice."""
                    if (
                        _vulkan_suspect
                        and getattr(self, "_health_wait_stall_kind", None) == "no_progress"
                    ):
                        return _try_auto_vulkan_cpu_fallback(stalled_cmd, None, stalled = True)
                    _why = (
                        "startup time limit reached"
                        if getattr(self, "_health_wait_stall_kind", None) == "ceiling"
                        else "no loading progress"
                    )
                    _raise_terminal_load_failure(
                        "llama-server did not finish starting on the auto-selected Vulkan "
                        f"backend ({_why}). If this keeps happening, choose CPU "
                        "in Settings > GGUF inference engine to run models without the "
                        "Vulkan driver."
                    )

                # Store the resolved on-disk path, not the caller's kwarg: in
                # HF mode gguf_path is None and ``model_path`` is what
                # llama-server mmap's, which downstream consumers need. Must be
                # set BEFORE the spawn: load_progress() reads _gguf_path for
                # the mmap progress total while the health wait runs.
                self._gguf_path = model_path
                self._gguf_load_identity = self._gguf_load_source_identity(model_path, mmproj_path)
                self._hf_repo = hf_repo
                self._mtp_draft_path = launch_mtp_draft_path
                self._mtp_draft_suppressed_path = _suppressed_draft_path
                self._mtp_draft_suppressed_reason = _suppressed_draft_reason
                # For local GGUF files, extract variant from filename if absent
                if hf_variant:
                    self._hf_variant = hf_variant
                elif gguf_path:
                    try:
                        from utils.models.model_config import _extract_quant_label
                        self._hf_variant = _extract_quant_label(gguf_path)
                    except Exception:
                        self._hf_variant = None
                else:
                    self._hf_variant = None
                self._is_vision = effective_is_vision
                self._disable_vision = bool(disable_vision)
                # is_vision, not effective_is_vision (this load detached the projector on purpose); text-only, audio-only and arg-dropped cases must not blame the switch.
                self._vision_disabled_by_user = bool(
                    is_vision and disable_vision and _dv_dropped_image_projector
                )
                self._model_identifier = model_identifier

                # Keep _context_length native for display.
                self._effective_context_length = (
                    effective_ctx if effective_ctx > 0 else self._context_length
                )
                self._max_context_length = (
                    max_available_ctx if max_available_ctx > 0 else self._effective_context_length
                )
                # The fit priced Unsloth's own placement and the base weights alone: a
                # pass-through placement override, adapter or inherited projector is not.
                if _vram_fit_ctx is not None:
                    _child_extras = (
                        self._strip_device_extra_args(extra_args)
                        if _gpu_ids_own_device_flags
                        else extra_args
                    )
                    _child_env = dict(os.environ)
                    if gpu_memory_mode == "manual":
                        self._clear_manual_placement_env(_child_env)
                    if _gpu_ids_own_device_flags:
                        self._clear_device_placement_env(_child_env)
                    if (
                        _pass_through_overrides_placement(
                            _child_extras,
                            _child_env,
                            len(gpu_indices) if gpu_indices is not None else len(gpus),
                            is_vulkan_backend = is_vulkan_backend,
                            gpu_indices = gpu_indices,
                        )
                        or _sidecar_adapter_bytes(extra_args) != 0
                        # A drafter the planner could only cover with a flat cushion.
                        or _flat_mtp_engages
                        or _child_env.get("LLAMA_ARG_MMPROJ")
                        or _child_env.get("LLAMA_ARG_MMPROJ_URL")
                        or (
                            not launch_mmproj_path
                            and extra_args_mmproj_auto(_child_extras, _child_env)
                        )
                        # A hand-set fitter or split ratio re-places what the fit priced.
                        or _extra_args_set_any_flag(_child_extras, _VRAM_FIT_VOIDING_FLAGS)
                        or any(_child_env.get(name) for name in _VRAM_FIT_VOIDING_ENV_VARS)
                        # Any surviving --device: the cap sweep may have pooled cards it leaves out.
                        or _extra_args_main_device(_child_extras) is not None
                        # And any companion (projector, drafter) pinned to a card of its own.
                        or any(
                            _extra_args_device(_child_extras, flags) is not None
                            for flags in _COMPANION_DEVICE_FLAG_GROUPS
                        )
                        or _child_env.get("LLAMA_ARG_DEVICE")
                        # --no-kv-offload keeps the whole cache in host RAM at any context.
                        or not _kv_offload_from_args(_child_extras, _child_env)
                    ):
                        _vram_fit_ctx = None
                self._vram_fit_context_length = _vram_fit_ctx

                # Past the host-RAM and offload advisories, so it can say the ceiling is
                # the request without costing the user a memory warning it does not
                # replace: appended when one of those spoke, recorded when none did.
                if _unmeasured_ctx_notice:
                    if self._last_load_warning:
                        self._amend_load_warning(" " + _unmeasured_ctx_notice)
                    else:
                        self._record_load_warning(_unmeasured_ctx_notice)

                if _cuda_ctx_notice and use_fit:  # slot reduction may have re-pinned it
                    if self._last_load_warning:
                        self._amend_load_warning(" " + _cuda_ctx_notice)
                    else:
                        self._record_load_warning(_cuda_ctx_notice)

                # Zero Vulkan devices: the bundle would still init a driver that can hang (#7213).
                _cpu_route_reason = None
                if not intent.cpu_fallback and _cpu_fallback_eligible:
                    _vk_outcome = self._vulkan_probe_outcome(binary)
                    _vulkan_suspect = _vk_outcome in ("timeout", "crashed")
                    # Only when a replay can be built, so the block below never fails the load.
                    if (
                        _vk_outcome == "none"
                        and self._cpu_isolated_replay(cmd, dict(env), server_caps) is not None
                        and self._cpu_isolated_binary(binary) is not None
                    ):
                        logger.warning(
                            "The auto-selected Vulkan backend has no usable device "
                            "(probe: %s); starting llama.cpp on CPU without loading the "
                            "Vulkan runtime.",
                            _vk_outcome,
                        )
                        intent = replace(intent, cpu_fallback = True)
                        _cpu_route_reason = "vulkan_no_device"

                # LoadRequest carries this flag, so a stale rollback, an API caller or
                # a swapped-out runtime can ask for a replay that never happened. Hold
                # it to the crash path's bar (managed auto Vulkan, no explicit
                # placement) and otherwise run the request as sent.
                if intent.cpu_fallback:
                    if self._auto_vulkan_cpu_fallback_eligible(
                        binary, intent, extra_args, env, allow_manual_cpu = True
                    ):
                        prepared = self._prepare_cpu_fallback_launch(
                            binary,
                            cmd,
                            env,
                            server_caps,
                            drop_full_offload_threads = _drop_full_offload_threads,
                            avail_mib = _preflight_avail_mib,
                        )
                        if prepared is None:
                            if _load_cancelled():
                                _cleanup_cancelled_load(
                                    "Load cancelled while reconstructing the CPU fallback"
                                )
                                return False
                            _raise_terminal_load_failure(
                                "The prior Vulkan CPU fallback could not be reconstructed; run "
                                "`unsloth studio update` to repair the managed llama.cpp runtime."
                            )
                        cmd, _spawn_cwd, _replay_pageable_note = prepared
                        # The override is new since the replay's argv was built, so it is appended to the recorded notice.
                        self._record_memory_state(cmd, env)
                        _mem_policy_for_cmd = _snapshot_policy_for_cmd()
                        # The --device none replay is CPU-only, so it needs a host-RAM advisory; an empty probe leaves none.
                        self._amend_load_warning(_replay_pageable_note)
                        # This replay is CPU-only by construction (--device none), so
                        # the advisory has to describe the whole model in host RAM. The
                        # amend above can only add to a notice that already exists, and
                        # on a restored fallback there often is none: an empty Vulkan
                        # probe leaves _child_has_no_gpu False, so the preflight
                        # abstained and memory_warning was null for a session paging
                        # everything from disk. Deferred rather than priced here,
                        # because only a child that came up is evidence of what runs.
                        _restored_cpu_reprice = _replay_pageable_note or True
                        # Same normalization the crash path applies, so a client that
                        # sent only the flag cannot leave a CPU-only server reporting
                        # an Auto/GPU placement (which then fails holds_no_vram).
                        intent = self._apply_cpu_fallback_state(
                            intent,
                            is_vision = self._is_vision,
                            mmproj_has_audio = self._mmproj_has_audio,
                            disable_vision = disable_vision,
                            vision_disabled_by_user = bool(
                                is_vision and disable_vision and _dv_dropped_image_projector
                            ),
                            reason = (
                                _cpu_route_reason
                                or self._cpu_fallback_reason
                                or "vulkan_startup_crash"
                            ),
                        )
                        gpu_memory_mode = intent.gpu_memory_mode
                        gpu_layers = intent.gpu_layers
                        n_cpu_moe = intent.n_cpu_moe
                        tensor_parallel = intent.tensor_parallel
                        tensor_split = intent.tensor_split
                        gpu_indices = None
                        use_fit = False
                    else:
                        # Clear only once a child is up, where is_active takes over; an unconditional clear undid the re-arm.
                        intent = replace(intent, cpu_fallback = False)
                        self._cpu_fallback_reason = None
                        self._cleanup_cpu_fallback_runtime()
                # _process is unset until Popen, so mark the load committed for saves; Popen and every exit below clear it, so a failed spawn cannot leave it stuck.
                self._memory_pending_launch = _mem_settings
                self._vram_fraction_pending = _budget_priced_placement()
                healthy = False
                try:
                    healthy = _spawn_and_wait(cmd)
                    if not healthy:
                        # Every recovery below relaunches a plan the fit never priced.
                        self._vram_fit_context_length = None
                finally:
                    # Old llama.cpp refuses quantized KV in tensor mode; retries cannot fix it, so latch on first spawn.
                    if healthy:
                        self._memory_pending_launch = None
                if not healthy and _finish_cancelled_health_wait(
                    "Load cancelled during the llama-server health wait"
                ):
                    return False
                # Stalled auto-Vulkan launch (#142): every rung below relaunches the same driver.
                if not healthy and _vulkan_stall_pending():
                    _stalled_cmd = list(_last_spawn_cmd)
                    self._kill_process()
                    if _recover_vulkan_stall(_stalled_cmd):
                        healthy = True
                    elif _finish_cancelled_health_wait(
                        "Load cancelled during the CPU replay health wait"
                    ):
                        return False
                # #6415 split-mode tensor warmup abort. Latch it on THIS first spawn:
                # the flash-attn-off retry below can't run tensor (needs flash_attn),
                # so its output drops the marker and recording later would miss it,
                # looping every load. Record and raise to the route's layer fallback,
                # skipping the futile flash-attn/MTP retries.
                if not healthy and self._tensor_parallel and not _load_cancelled():
                    _ts_out = "\n".join(self._stdout_lines[-50:])
                    _proc_snap2 = self._process  # snapshot: re-reading races the teardown
                    _ts_rc = _proc_snap2.poll() if _proc_snap2 is not None else None
                    if self._should_record_tensor_split_abort(
                        _ts_rc, "\n".join(self._stdout_lines)
                    ):
                        # A drafter crash must not disable tensor split for drafterless loads.
                        if not _extra_args_mtp_draft_path(
                            _last_spawn_cmd, env = _child_spec_env(extra_args)
                        ):
                            LlamaCppBackend._record_tensor_split_abort(
                                binary, model_identifier, _planned_cache_pair
                            )
                        self._kill_process()
                        raise RuntimeError(
                            "llama-server aborted on --split-mode tensor "
                            "(split-axis geometry); retrying with layer split."
                        )
                    # Pre-b9455 llama.cpp refusing a quantized KV cache in tensor mode.
                    # Latched on this first spawn for the same reason as the abort
                    # above: the retries below cannot fix a capability the binary
                    # lacks, and only the first spawn carries the marker.
                    if self._is_tensor_quant_kv_unsupported(_ts_out):
                        # Use _detected_gpus, not gpus: it survives manual mode, which empties the pool to skip the planner.
                        LlamaCppBackend._record_tensor_quant_kv_unsupported(binary)
                        self._kill_process()
                        raise RuntimeError(
                            "This llama.cpp build does not support a quantized KV "
                            "cache with --split-mode tensor (added in b9455, "
                            "ggml-org/llama.cpp#23792). Retrying with layer split; "
                            "update llama.cpp to use tensor parallelism with a "
                            "quantized cache."
                        )
                # Auto-pinned device has no kernels in this binary: retry once on a narrowed set, since the gate cannot see custom builds; explicit picks keep their error.
                if (
                    not healthy
                    and not _load_cancelled()
                    and not gpu_ids
                    and not is_vulkan_backend
                    and gpu_indices
                    and self._kernel_image_invalid("\n".join(self._stdout_lines[-80:]))
                ):
                    _crashed = sorted(set(gpu_indices))
                    # _detected_gpus, not gpus: the post-picker enumeration this launch
                    # planned against, and unlike `gpus` it survives the manual mode that
                    # empties the pool to bypass the planner.
                    _remaining = self._arch_crash_retry_gpu_ids(
                        gpu_indices, [i for i, _free in _detected_gpus]
                    )
                    if _remaining:
                        # Earlier notices priced the crashed selection; re-price on the respawn set, since first notice wins.
                        _retry_wants_unified = self._amd_apu_wants_unified_memory(_remaining)
                        _retry_unified_helps = self._unified_memory_for_launch(
                            _remaining, _unified_need_now(), opted_in = _unified_opt_in
                        )
                        # Everything recorded so far priced the CRASHED selection, and
                        # that placement is gone: the canonical #7624 shape pins the APU
                        # whose shared-pool "free memory" outranked the dGPU, warns that
                        # system RAM cannot hold the weights, then crashes and respawns
                        # on the discrete sibling that holds them in VRAM. First notice
                        # wins, so without this the served child reports a shortfall on
                        # a device it never touches. The two checks below price exactly
                        # the set the respawn reaches, and both are wider than their
                        # upstream twins (the same weights against a NARROWED pool), so
                        # a shortfall that still stands is re-recorded here.
                        #
                        # Both are also kept in a variable: the pageable-load override
                        # at the END of this block, after every rewrite `cmd` still has
                        # coming, reads them as its shortfall condition and amends
                        # whichever one first-notice-wins recorded.
                        self._begin_load_warnings()
                        # ...and the re-price snapshots go with the scope they describe:
                        # the preflight's pair priced the crashed placement, so leaving
                        # them standing would let the text-only fallback re-price the
                        # notice this block is about to record against the wrong pool.
                        _apu_ram_msg = None
                        _host_ram_msg = None
                        _apu_avail_mib = None
                        # Read before the kill frees the crashed child's memory, which would inflate the reading.
                        _retry_apu_avail_mib = (
                            self._available_system_memory_mib()
                            if model_size is not None and _retry_wants_unified
                            else None
                        )
                        _retry_host_avail_mib = self._available_system_memory_mib()
                        # host guard credited the whole pool; the respawn reaches only _remaining
                        _retry_rows = [row for row in _detected_gpus if row[0] in set(_remaining)]
                        logger.warning(
                            f"llama-server crashed with a HIP kernel-image error on "
                            f"GPU(s) {_crashed} -- the llama.cpp build has no kernels "
                            f"for that arch. Retrying on GPU(s) {_remaining}."
                        )
                        self._kill_process()
                        gpu_indices = _remaining
                        _unified_gpu_indices = _remaining
                        # The tuning was decided against the devices that just crashed,
                        # and this respawn reuses `cmd`, so a discrete-to-APU retry would
                        # carry --cache-ram 0 onto a shared pool and the reverse would
                        # reach a discrete card without it. Re-decide against the retry.
                        if sys.platform == "win32" and full_offload_tuning_active:
                            _retry_shared = _cache_target_unknown or (
                                self._offload_target_shares_system_memory(
                                    is_vulkan_backend = is_vulkan_backend,
                                    shared_gpu_ids = _shared_gpu_ids,
                                    detected_gpus = _retry_rows or _detected_gpus,
                                    gpu_indices = _remaining,
                                )
                            )
                            # Off first, or _retry_cache_tuning_flags reads the cap as a
                            # count the user typed and declines to zero it.
                            if _auto_ckpt_emitted:
                                cmd = self._without_flag_pairs(cmd, _auto_ckpt_emitted)
                                _auto_ckpt_emitted = []
                            if _retry_shared and _cache_flags_emitted:
                                # Exactly the tokens this policy appended, each a flag
                                # with its value, so the strip cannot eat a user extra.
                                cmd = self._without_flag_pairs(cmd, _cache_flags_emitted)
                                _cache_flags_emitted = []
                                logger.info(
                                    "Retry lands on a GPU that shares system memory: dropped "
                                    "--cache-ram 0 and --ctx-checkpoints 0 so the prompt cache "
                                    "survives."
                                )
                            elif not _retry_shared and not _cache_flags_emitted:
                                _retry_flags = self._retry_cache_tuning_flags(
                                    cmd,
                                    cache_ram = cache_ram,
                                    ctx_checkpoints = _ctx_checkpoints_owned,
                                    server_caps = server_caps,
                                )
                                if _retry_flags:
                                    cmd.extend(_retry_flags)
                                    _cache_flags_emitted = list(_retry_flags)
                                    logger.info(
                                        "Retry lands on a GPU with its own memory: applied the "
                                        "Windows full-offload cache tuning."
                                    )
                            _auto_ckpt_emitted = _emit_auto_ctx_checkpoints(cmd)
                        # Withdraw GGML_CUDA_ENABLE_UNIFIED_MEMORY decided for the crashed set (only our value).
                        if _unified_env_applied and not _retry_unified_helps:
                            env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
                            _unified_env_applied = False
                            logger.info(
                                "Arch-crash retry targets GPU(s) the unified-memory "
                                "swap cannot help; dropped "
                                "GGML_CUDA_ENABLE_UNIFIED_MEMORY for the respawn."
                            )
                        elif (
                            _retry_unified_helps
                            and not _unified_opt_out  # the opt-out outlives the respawn
                            and "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env
                        ):
                            env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
                            _unified_env_applied = True
                            logger.info(
                                "Arch-crash retry targets a unified-memory APU the "
                                "weights outgrow; set "
                                "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 for the respawn."
                            )
                        _retry_mask = list(_remaining)
                        if _companion_fit_mask is not None:
                            # tokens are positions in the crashed launch's mask
                            if _companion_added_device is not None:
                                for _k in range(len(cmd) - 1, 0, -1):
                                    if cmd[_k - 1 : _k + 1] == _companion_added_device:
                                        del cmd[_k - 1 : _k + 1]
                                        break
                            _had_main_device = _extra_args_main_device(cmd) is not None
                            _retry_mask, _refit = _widen_pin_ids_for_companion_devices(
                                cmd,
                                list(_remaining),
                                list(_companion_fit_mask),
                                allowed_ids = gpu_ids or None,
                            )
                            _companion_added_device = (
                                list(cmd[-2:])
                                if not _had_main_device and _extra_args_main_device(cmd) is not None
                                else None
                            )
                            _companion_fit_mask = list(_retry_mask)
                            if _refit:
                                logger.info(
                                    "Refitted companion device flags to the retry's mask: %s",
                                    _refit,
                                )
                        self._emit_child_gpu_visibility(
                            env, ",".join(str(i) for i in _retry_mask), prefer_rocr = True
                        )
                        _child_gpu_physical_ids = tuple(int(i) for i in _retry_mask)
                        # No ratio record here: _tensor_split is set only in manual mode, where gpu_indices stays None.
                        self._clear_split_placement_env(env)
                        # The mask re-indexes survivors, so drop a --tensor-split sized for the crashed set.
                        _arch_retry_cmd = self._without_tensor_split(cmd)
                        if _arch_retry_cmd is not None:
                            logger.info(
                                "Dropped --tensor-split for the arch-crash retry: "
                                "its shares are positional over the visible "
                                "devices and the narrowed set re-indexes them."
                            )
                            cmd = _arch_retry_cmd
                            # No ratio record, unlike the proactive gate: _tensor_split
                            # is non-None only in manual mode, which leaves gpu_indices
                            # None, and this arm needs it truthy.
                            self._tensor_split = None
                            # The auto twin, which this arm CAN reach: an auto
                            # tensor-parallel fallback emits its ratio with
                            # gpu_indices pinned. Left set it feeds the
                            # `tensor_split` property a ratio the respawned child
                            # does not have.
                            self._auto_tensor_split_emitted = None
                        # The fit's none assumed the pinned devices, and KV can exceed RAM with mmap off; drop the none.
                        if len(_remaining) < 2 and self._tensor_parallel:
                            _no_tensor_mode = self._without_flags(cmd, ("--split-mode", "-sm"))
                            if _no_tensor_mode is not None:
                                cmd = _no_tensor_mode
                            self._arch_gate_dropped_tensor_parallel = True
                            self._tensor_parallel = False
                            logger.info(
                                "Arch-crash retry leaves one GPU; dropped tensor "
                                "parallelism, which needs at least two."
                            )
                        cmd = self._drop_managed_lazy_mode(cmd, "device fallback")
                        _retry_apu_msg = None
                        if model_size is not None and _retry_wants_unified:
                            _apu_avail_mib = _retry_apu_avail_mib
                            _apu_resident_model_size = max(
                                1,
                                model_size
                                - self._lazy_read_host_bytes(
                                    model_path,
                                    caps = server_caps,
                                    extra_args = cmd,
                                    env = env,
                                    auto_resolves_off = self._lazy_auto_resolves_off(
                                        gpu_indices = _remaining,
                                        detected_gpus = _detected_gpus,
                                        shared_gpu_ids = _shared_gpu_ids,
                                        is_vulkan_backend = is_vulkan_backend,
                                        probe_integrated_cuda = False,
                                        extra_args = cmd,
                                        env = env,
                                    ),
                                ),
                            )
                            _retry_apu_msg = self._apu_ram_shortfall_message(
                                _apu_resident_model_size + _mmproj_pinned_bytes,
                                _apu_avail_mib,
                            )
                            if not self._host_offload_warning_opted_out():
                                _apu_ram_msg = _retry_apu_msg
                                self._record_load_warning(_retry_apu_msg)
                        _retry_host_msg = self._launch_host_shortfall_message(
                            cmd, _retry_rows, env, avail_mib = _retry_host_avail_mib
                        )
                        _host_ram_msg = _retry_host_msg
                        self._record_load_warning(_retry_host_msg)
                        # Restore what cmd means: a --fit retry may have recorded a page-lock that cmd never carried.
                        _fit_mode_left_cmd = bool(self._fit_load_mode_flags)
                        if self._fit_load_mode_flags:
                            cmd = _without_subsequence(cmd, self._fit_load_mode_flags)
                            self._fit_load_mode_flags = []
                            logger.info(
                                "Load mode: dropping the fit's --load-mode none for "
                                "the arch-crash retry; it runs on a different device "
                                "set than the fit was proved against."
                            )
                        # Re-asked on the narrowed set both ways, since narrowing can gain an offload the full set lacked.
                        (
                            _mem_host_resident,
                            self._memory_state,
                            self._memory_direct_io,
                            self._memory_dio_applicable,
                            self._memory_dio_flags,
                            self._memory_policy_active,
                            self._memory_mlock_applicable,
                        ) = _mem_policy_for_cmd
                        # Excluding the load-mode pair removed; record the stripped argv, not _mem_managed + _mem_extras.
                        _arch_dio, _arch_applicable, _arch_active = _dio_decision_for(
                            _remaining, fully_offloaded = fully_gpu_offloaded, child_env = env
                        )
                        self._memory_dio_applicable = _arch_applicable
                        _dio_left_cmd = False
                        if self._memory_dio_flags and not _arch_dio:
                            _dio_left_cmd = True
                            cmd = self._drop_managed_dio(
                                cmd,
                                "the arch-crash retry's remaining devices no longer "
                                "confirm a full offload",
                            )
                            self._memory_dio_applicable = _arch_applicable
                        elif _arch_dio and not self._memory_dio_flags:
                            _dio_left_cmd = True
                            cmd = [*cmd, *_arch_dio]
                            self._memory_dio_flags = list(_arch_dio)
                            self._memory_policy_active = _arch_active or self._memory_policy_active
                            logger.info(
                                "Model Memory: applying %s; the arch-crash retry's "
                                "remaining GPU(s) %s confirm a full offload.",
                                " ".join(_arch_dio),
                                _remaining,
                            )
                        # Residency follows the devices the retry changed, so re-decide the page-lock with the same helpers.
                        if _fit_mode_left_cmd or _dio_left_cmd:
                            self._record_memory_state(cmd, env)
                        # Residency is a property of the DEVICES, which the retry just
                        # changed: crash on the discrete card, land on the
                        # unified-memory APU, and the weights are host-backed after
                        # all, so the lock the first launch skipped is the one the user
                        # asked for. Left alone the respawn runs unlocked AND records
                        # _memory_mlock_applicable False, reading the missing lock as
                        # deliberate and deduping away the reload that would apply it.
                        # Through the same helpers, so the two cannot drift.
                        _retry_host_resident = self._weights_in_host_memory(
                            fully_gpu_offloaded = fully_gpu_offloaded,
                            gpu_memory_mode = gpu_memory_mode,
                            gpu_layers = gpu_layers,
                            extra_args = _mem_extra_args,
                            gpu_indices = _remaining,
                            is_vulkan_backend = is_vulkan_backend,
                            binary = binary,
                            env = env,
                            # Only adds a lock: scanning to remove valueless --mlock/--no-mmap would eat the user's extra args.
                            probe_vulkan = _mem_should_mlock or _mem_probe_for_dio,
                            fit_active = fit_is_effectively_on([*cmd, *(_mem_extra_args or [])], env),
                        )
                        # Lock-ADDING direction only: `cmd` carries a policy-emitted
                        # lock only when the first launch found the weights
                        # host-resident, so this is the one direction that adds without
                        # taking anything back off argv. Removing flags would mean
                        # scanning for --mlock/--no-mmap, VALUELESS in llama.cpp's
                        # parser, so the scan would eat the user extra after them. The
                        # reverse direction leaves the crashed launch's lock in place,
                        # which stays truthful: the child does hold it.
                        if _retry_host_resident and not _mem_host_resident:
                            _retry_managed, _ = apply_model_memory_policy(
                                extra_args,
                                supports_load_mode = load_mode_support(server_caps),
                                # Not the requested load mode: the first launch
                                # emitted it and this call only ADDS, so a second
                                # copy would land after the extras and mark
                                # _memory_policy_active for a launch it never touched.
                                weights_in_host_memory = _retry_host_resident,
                                # The launch's own pair: a save landing before this retry
                                # would otherwise pick flags this launch never saw.
                                settings = _mem_settings,
                            )
                            if _retry_managed:
                                # After the user extras, so llama.cpp's last-wins parse
                                # gives it the final say over any hand-written memory
                                # flag. Nothing is stripped, so kept extras survive.
                                cmd.extend(_retry_managed)
                            _mem_host_resident = _retry_host_resident
                            self._memory_mlock_applicable = keep_resident_lock_applicable(
                                _retry_host_resident, load_mode_support(server_caps)
                            )
                            # Stays active if it already was: the flag also
                            # covers extras this policy touched and env it
                            # scrubbed, neither of which the respawn undoes.
                            self._memory_policy_active = (
                                bool(_retry_managed) or self._memory_policy_active
                            )
                            # Runs last, after all `cmd` rewrites: an unmapped respawn on the narrowed pool OOMs in host RAM.
                            self._record_memory_state(cmd, env)
                            logger.info(
                                "Arch-crash retry changed where the weights live; "
                                "recomputed Model Memory (%s).",
                                " ".join(_retry_managed) if _retry_managed else "no page-lock",
                            )

                        # Last, because everything above still rewrites `cmd`: the
                        # tensor-split drop, the split-mode drop, the fit's load-mode
                        # strip and the re-armed page-lock. The narrowed pool is where a
                        # load that FIT its first placement can meet its first shortfall,
                        # and nothing upstream remapped it, so an unmapped respawn would
                        # allocate the whole model in host RAM and be OOM-killed. Same
                        # condition, same helper, before the respawn rather than after.
                        def _retry_is_oversized() -> bool:
                            if _retry_apu_msg or _retry_host_msg:
                                return True
                            # The opt-out silences the message, never the rewrite.
                            if not self._host_offload_warning_opted_out():
                                return False
                            return bool(
                                self._launch_host_shortfall_message(
                                    cmd,
                                    _retry_rows,
                                    env,
                                    honor_opt_out = False,
                                    avail_mib = _retry_host_avail_mib,
                                )
                            )

                        cmd, _retry_pageable_note = self._page_an_oversized_unmapped_load(
                            cmd, env, _retry_is_oversized
                        )
                        if _retry_pageable_note:
                            # Onto whichever notice first-notice-wins kept above; a
                            # suppressed one leaves the log line as the only record,
                            # which is the intended contract for the opt-out.
                            self._amend_load_warning(_retry_pageable_note)
                            # Re-price carve-out advice (dropped by _begin_load_warnings()); respawn may land on a tighter APU.
                            self._record_memory_state(cmd, env)
                        # And the carve-out advice with them. _begin_load_warnings()
                        # dropped the one priced for the crashed placement, but the
                        # respawn can land on a unified-memory APU whose allocation the
                        # same weights outgrow. Priced against `cmd` and _remaining, so
                        # the spill reported is this placement's.
                        try:
                            _retry_carveout_need = _unified_need_now(argv = cmd)
                        except Exception:
                            _retry_carveout_need = None
                        self._record_carveout_advice(
                            _remaining,
                            _retry_carveout_need,
                            is_vulkan_backend = is_vulkan_backend,
                            shared_gpu_ids = _shared_gpu_ids,
                            detected_gpus = _retry_rows or _detected_gpus,
                            target_unknown = _cache_target_unknown,
                        )
                        healthy = _spawn_and_wait(cmd, label = "-archfallback")

                # #12842: after the #7624 respawn every rung below keeps the same kernels.
                if not healthy and not _load_cancelled():
                    # Whole buffer: a Linux abort appends a debugger backtrace.
                    _cuda_image_out = "\n".join(self._stdout_lines)
                    if self._cuda_kernel_image_error(_cuda_image_out) is not None:
                        _proc_snap_ki = self._process  # snapshot: re-reading races the teardown
                        _ki_rc = _proc_snap_ki.poll() if _proc_snap_ki is not None else None
                        self._kill_process()
                        _raise_terminal_load_failure(
                            self._classify_llama_start_failure(
                                _cuda_image_out,
                                gguf_path,
                                self._model_identifier,
                                _ki_rc,
                                binary,
                                self._llama_log_path,
                                (self._api_key,),
                                self._extra_args,
                            )
                        )

                # Retry at one slot on a --kv-unified refusal, which must run ahead of the flash-
                # attn rung.
                if not healthy and not _load_cancelled():
                    _kvu_cmd = (
                        self._with_single_sequence(_last_spawn_cmd)
                        if self._is_kv_unified_refused("\n".join(self._stdout_lines))
                        else None
                    )
                    if _kvu_cmd is not None:
                        logger.warning(
                            "llama-server refused a unified KV cache with more than "
                            "one sequence; retrying with one slot and --kv-unified "
                            "dropped. Concurrent requests will queue."
                        )
                        self._kill_process()
                        if self._drop_env_single_sequence(env):
                            logger.info(
                                "Dropped inherited LLAMA_ARG_KV_UNIFIED / "
                                "LLAMA_ARG_N_PARALLEL for the single-sequence retry."
                            )
                        cmd = _kvu_cmd
                        # Read by admission control; left as-is Studio over-admits.
                        n_parallel = 1  # allow-slot-clamp: llama-server refused more
                        kv_cache_unified = False
                        # The reserve was priced for the dropped slots: reprice for one sequence.
                        if (
                            _sidecar_fit_reserve_eligible_mib
                            and _sidecar_reserve_reprice is not None
                        ):
                            cmd = _drop_sidecar_margins(
                                cmd,
                                _integrity_flags,
                                self._ctx_integrity_flags(
                                    *_integrity_args,
                                    fit_target_delta_mib = _fit_target_delta_mib,
                                    fit_target_floor_mib = _metal_fit_target_floor_mib,
                                ),
                                _sidecar_override_pairs,
                                user_args = _sidecar_user_tokens,
                            )
                            _sidecar_override_pairs.clear()
                            _sidecar_reserve_reprice = functools.partial(
                                _sidecar_reserve_reprice, _np = 1, _kv_unified = False
                            )
                            _sidecar_fit_reserve_eligible_mib = _sidecar_reserve_reprice(cmd, env)
                            _sidecar_fit_reserve_mib = 0
                        # LLAMA_ARG_SPEC_TYPE reaches the child only if extras set --spec-type; else the launch scrubs it.
                        if _auto_ckpt_emitted:
                            cmd = self._without_flag_pairs(cmd, _auto_ckpt_emitted)
                            _auto_ckpt_emitted = []
                        _auto_ctx_checkpoints = _decide_auto_ctx_checkpoints()
                        _auto_ckpt_emitted = _emit_auto_ctx_checkpoints(cmd)
                        healthy = _spawn_and_wait(cmd, label = "-single-seq")

                # FA kernels hard-crash on some ROCm builds; retry FA-off before dropping vision or MTP.
                if not healthy and not _load_cancelled():
                    _proc_snap3 = self._process  # snapshot: re-reading races the teardown
                    _fa_rc = _proc_snap3.poll() if _proc_snap3 is not None else None
                    _fa_cmd = (
                        self._with_flash_attn_off(
                            _last_spawn_cmd,
                            mla = self._target_kv_symmetry(),
                            draft_mla = self._draft_kv_symmetry(_last_spawn_cmd, env),
                        )
                        if self._is_signal_crash(_fa_rc)
                        else None
                    )
                    if _fa_cmd is not None:
                        logger.warning(
                            "llama-server hard-crashed at startup (exit %s) with "
                            "flash attention on; retrying once with --flash-attn "
                            "off (keeps vision and MTP).",
                            _fa_rc,
                        )
                        self._kill_process()
                        # Drop an env-only quantized V cache, or the FA-off child aborts.
                        if self._drop_env_quantized_v_cache(
                            env,
                            mla = self._target_kv_symmetry(),
                            draft_mla = self._draft_kv_symmetry(_last_spawn_cmd, env),
                        ):
                            logger.info(
                                "Dropped inherited quantized V-cache env for the "
                                "--flash-attn off retry (requires flash attention)."
                            )
                        # Also drop the env flash-attn flag; the argv rewrite cannot reach it.
                        if self._drop_env_flash_attn(env):
                            logger.info(
                                "Dropped inherited LLAMA_ARG_FLASH_ATTN for the "
                                "--flash-attn off retry."
                            )
                        _fa_cmd = _drop_fit_load_mode_for_no_flash(_fa_cmd)
                        _fa_cmd = _enable_managed_fit_for_no_flash(_fa_cmd)
                        _flash_attn_known_off = True
                        cmd = _fa_cmd
                        healthy = _spawn_and_wait(_fa_cmd, label = "-noflash")

                # MTP from Unsloth's spec flags or the user's (extra_args
                # --spec-type / LLAMA_ARG_SPEC_TYPE). The env survives to the child only
                # when the extras own --spec-type (the launch scrubs it otherwise), so
                # anything else would judge a server it is not starting.
                _launch_spec_env: Mapping[str, str] = _child_spec_env(extra_args)
                _spec_requested_mtp = any(
                    "mtp" in str(t).lower() for t in spec_flags
                ) or _extra_args_requests_mtp(extra_args, env = _launch_spec_env)
                _spec_requested_dspark = any(
                    "draft-dspark" in str(t).lower() for t in spec_flags
                ) or _extra_args_requests_dspark(extra_args, env = _launch_spec_env)
                _spec_requested_dflash = any(
                    "draft-dflash" in str(t).lower() for t in spec_flags
                ) or _extra_args_requests_dflash(extra_args, env = _launch_spec_env)
                # A paired EAGLE-3 drafter (Studio export) can abort the server the same way.
                _spec_requested_eagle3 = any("draft-eagle3" in str(t).lower() for t in spec_flags)
                # Any MTP launch, not only tensor: a respawn would replay a drafter that
                # aborts every decode (#13253). Cleared if the MTP-drop fallback wins.
                _mtp_active_for_launched_server = bool(_spec_requested_mtp)
                # MTP can pass /health then crash the flash-attn kernel on the
                # first decode under tensor; probe one generation so the fallback
                # catches that too. Tensor-only, so ordinary MTP stays probe-free.
                if (
                    healthy
                    and self._tensor_parallel
                    and _spec_requested_mtp
                    and not _load_cancelled()
                    and not self._probe_mtp_decode()
                ):
                    # A first-decode hard fault is usually the FA kernel: retry
                    # FA-off (keeps MTP) before dropping speculative decoding below.
                    _proc_snap4 = self._process  # snapshot: re-reading races the teardown
                    _probe_rc = _proc_snap4.poll() if _proc_snap4 is not None else None
                    _fa_cmd = (
                        self._with_flash_attn_off(
                            _last_spawn_cmd,
                            mla = self._target_kv_symmetry(),
                            draft_mla = self._draft_kv_symmetry(_last_spawn_cmd, env),
                        )
                        if self._is_signal_crash(_probe_rc)
                        else None
                    )
                    healthy = False
                    if _fa_cmd is not None:
                        logger.warning(
                            "MTP first-decode hard-crashed (exit %s) with flash "
                            "attention on; retrying with --flash-attn off.",
                            _probe_rc,
                        )
                        self._kill_process()
                        # Same reach gap: argv drops a bare --flash-attn, but an env LLAMA_ARG_FLASH_ATTN still enables it.
                        if self._drop_env_quantized_v_cache(
                            env,
                            mla = self._target_kv_symmetry(),
                            draft_mla = self._draft_kv_symmetry(_last_spawn_cmd, env),
                        ):
                            logger.info(
                                "Dropped inherited quantized V-cache env for the "
                                "--flash-attn off retry (requires flash attention)."
                            )
                        # Same reach problem for the flag itself: where the rewrite
                        # could only drop a bare --flash-attn, the env var alone
                        # still enables it.
                        if self._drop_env_flash_attn(env):
                            logger.info(
                                "Dropped inherited LLAMA_ARG_FLASH_ATTN for the "
                                "--flash-attn off retry."
                            )
                        _fa_cmd = _drop_fit_load_mode_for_no_flash(_fa_cmd)
                        _fa_cmd = _enable_managed_fit_for_no_flash(_fa_cmd)
                        _flash_attn_known_off = True
                        cmd = _fa_cmd
                        healthy = (
                            _spawn_and_wait(_fa_cmd, label = "-noflash-mtp")
                            and self._probe_mtp_decode()
                        )
                    if not healthy:
                        logger.warning(
                            "MTP speculative decoding crashed on the first decode "
                            "under tensor parallelism; retrying without it."
                        )
                # Retry once with --spec-default on MTP abort; keep requested argv (GPU init crash is not drafter's fault) and _requested_spec_mode to stop /load thrash.
                _spec_cpu_replay_cmd: Optional[List[str]] = None
                if (
                    not healthy
                    and (
                        _spec_requested_mtp
                        or _spec_requested_dspark
                        or _spec_requested_dflash
                        or _spec_requested_eagle3
                    )
                    and not _load_cancelled()
                ):
                    _spec_cpu_replay_cmd = list(_last_spawn_cmd)
                    # Launched without spec group; keep MTP only in requested state or the next Apply repeats the crash.
                    _lo = "\n".join(self._stdout_lines).lower()
                    # Only an unknown arch proves the binary is outdated; other failures are runtime.
                    _arch_unsupported = "unknown model architecture" in _lo
                    if (
                        _arch_unsupported
                        or "failed to measure draft model memory" in _lo
                        or "failed to measure mtp context memory" in _lo
                        or "failed to create mtp context" in _lo
                        or "failed to create llama_context" in _lo
                    ):
                        _retry_reason = (
                            "the prebuilt may predate it; retrying without "
                            "speculative decoding -- run `unsloth studio "
                            "update` for this drafter"
                        )
                        self._spec_fallback_reason = (
                            "binary_outdated" if _arch_unsupported else "runtime_error"
                        )
                    else:
                        _retry_reason = (
                            "retrying without speculative decoding in case the drafter is the cause"
                        )
                        self._spec_fallback_reason = "runtime_error"
                    _drafter = (
                        Path(launch_mtp_draft_path).name
                        if launch_mtp_draft_path
                        else "embedded head"
                    )
                    logger.warning(
                        "llama-server failed to start with speculative drafter (%s); %s.",
                        _drafter,
                        _retry_reason,
                    )
                    self._kill_process()
                    # Re-locate the spec block: the arch gate may have shifted it.
                    _spec_at = _subsequence_index(cmd, spec_flags, _spec_start)
                    _fb_tail = cmd[_spec_at + len(spec_flags) :]
                    _fb_stripped_extras: Optional[List[str]] = None
                    # Spec types accumulate, so the extras/env MTP must be removed, not overridden.
                    if (
                        _extra_args_requests_mtp(extra_args, env = _launch_spec_env)
                        or _extra_args_requests_dspark(extra_args, env = _launch_spec_env)
                        or _extra_args_requests_dflash(extra_args, env = _launch_spec_env)
                    ):
                        _fb_tail = strip_shadowing_flags(
                            _fb_tail,
                            strip_context = False,
                            strip_cache = False,
                            strip_spec = True,
                            strip_template = False,
                            strip_split_mode = False,
                        )
                        for _fb_spec_var in _SPEC_ENV_VARS:
                            env.pop(_fb_spec_var, None)
                        _fb_stripped_extras = strip_shadowing_flags(
                            extra_args or [],
                            strip_context = False,
                            strip_cache = False,
                            strip_spec = True,
                            strip_template = False,
                            strip_split_mode = False,
                        )
                    fallback_cmd = cmd[:_spec_at] + ["--spec-default"] + _fb_tail
                    # A user-owned drafter in the extras or env still loads and keeps its reserve.
                    if _sidecar_fit_reserve_mib and not _extra_args_mtp_draft_path(
                        fallback_cmd, env = env
                    ):
                        _fb_integrity = self._ctx_integrity_flags(
                            *_integrity_args,
                            fit_target_delta_mib = _fit_target_delta_mib,
                            fit_target_floor_mib = _metal_fit_target_floor_mib,
                        )
                        fallback_cmd = _drop_sidecar_margins(
                            fallback_cmd,
                            _integrity_flags,
                            _fb_integrity,
                            _sidecar_override_pairs,
                            user_args = _sidecar_user_tokens,
                        )
                    # Restore the cut slots; if they fail (unified KV refused), retry as cut.
                    _fb_cut = (
                        (
                            fallback_cmd,
                            n_parallel,
                            effective_ctx,
                            (self._effective_context_length, self._max_context_length),
                            _auto_ctx_checkpoints,
                            _auto_ckpt_emitted,
                        )
                        if _mtp_slots_asked is not None
                        else None
                    )
                    if _fb_cut is not None:
                        n_parallel = _mtp_slots_asked
                        fallback_cmd = _with_parallel_slots(
                            fallback_cmd,
                            n_parallel,
                            n_batch = n_batch,
                            supports_kv_unified = bool(server_caps.get("supports_kv_unified")),
                        )
                        # The cut may have bought context; the restored count was priced at this.
                        if _mtp_slots_target_ctx and "-c" in fallback_cmd:
                            effective_ctx = _mtp_slots_target_ctx
                            fallback_cmd[fallback_cmd.index("-c") + 1] = str(effective_ctx)
                            # The ceiling was priced at the cut count; this is the one proven here.
                            self._effective_context_length = self._max_context_length = (
                                effective_ctx
                            )
                        # The checkpoint cap is per slot; same re-decide as the single-seq rung.
                        if _auto_ckpt_emitted:
                            fallback_cmd = self._without_flag_pairs(
                                fallback_cmd, _auto_ckpt_emitted
                            )
                        _auto_ctx_checkpoints = _decide_auto_ctx_checkpoints()
                        _auto_ckpt_emitted = _emit_auto_ctx_checkpoints(fallback_cmd)
                        kv_cache_unified = _kv_unified_from_args(fallback_cmd)
                        # The plan other loads place against was priced at the cut count.
                        _fb_cut_plan = dict(self._pending_plan_mib)
                        if self._pending_plan_mib and self._can_estimate_kv():
                            kv_cache_bytes = self._estimate_kv_cache_bytes(
                                effective_ctx,
                                cache_type_kv,
                                n_parallel = n_parallel,
                                swa_full = swa_full,
                                kv_unified = bool(kv_cache_unified),
                                n_ubatch = _ubatch_for_slots(n_parallel),
                                flash_attn = planned_flash_attn,
                            )
                            self._pending_plan_mib = _gpu_plan_mib(
                                max(1, gguf_size - _host_input_embd) + mmproj_size + kv_cache_bytes,
                                list(self._pending_plan_mib),
                                gpus,
                            )
                    _unified_withdraw_if_unneeded(
                        fallback_cmd,
                        mtp_engages = False,
                        why = "The retry without speculative decoding",
                    )
                    healthy = _spawn_and_wait(fallback_cmd, label = "-retry")
                    if not healthy and _fb_cut is not None and not _load_cancelled():
                        logger.warning(
                            "The retry without speculative decoding failed at %d slots; "
                            "retrying at the %d the drafter launch ran.",
                            n_parallel,
                            _fb_cut[1],
                        )
                        self._kill_process()
                        (
                            fallback_cmd,
                            _fb_slots,
                            effective_ctx,
                            (self._effective_context_length, self._max_context_length),
                            _auto_ctx_checkpoints,
                            _auto_ckpt_emitted,
                        ) = _fb_cut
                        self._pending_plan_mib = _fb_cut_plan
                        n_parallel = (
                            _fb_slots  # allow-slot-clamp: the restored count failed to start
                        )
                        kv_cache_unified = _kv_unified_from_args(fallback_cmd)
                        healthy = _spawn_and_wait(fallback_cmd, label = "-retry-cut")
                    if healthy:
                        self._speculative_type = "default"
                        _mtp_active_for_launched_server = False
                        # The record below stores extra_args as what LAUNCHED, but this
                        # server launched without the spec group. Leaving the MTP list
                        # there would have the next inheriting Apply pick it back up and
                        # repeat the crash. Keep it as the requested state only, the way
                        # the runtime-crash replay does.
                        if _fb_stripped_extras is not None:
                            _pv_suppressed_spec_extra_args = (
                                _pv_suppressed_spec_extra_args
                                if _pv_suppressed_spec_extra_args is not None
                                else list(extra_args or [])
                            )
                            extra_args = _fb_stripped_extras

                # A stalled later retry: a live child has no exit code for the crash gate.
                if not healthy and _vulkan_stall_pending():
                    _stalled_cmd = list(_last_spawn_cmd)
                    self._kill_process()
                    if _recover_vulkan_stall(_stalled_cmd):
                        healthy = True
                    elif _finish_cancelled_health_wait(
                        "Load cancelled during the CPU replay health wait"
                    ):
                        return False

                # Keep a multimodal model multimodal when only its GPU projector
                # placement fails. llama.cpp owns mmproj offload separately from
                # --gpu-layers, so retry it on CPU before removing --mmproj.
                if not healthy:
                    out = "\n".join(self._stdout_lines[-50:])
                    # Read the crash code before _kill_process() clears _process.
                    _proc_snap5 = self._process  # snapshot: re-reading races the teardown
                    _crash_rc = _proc_snap5.poll() if _proc_snap5 is not None else None
                    self._kill_process()
                    # Only when the wait itself was cancelled; a later cancel must still run the CPU fallback.
                    if _finish_cancelled_health_wait(
                        "Load cancelled during the llama-server health wait"
                    ):
                        return False
                    # Projector moves to host RAM, outside the fit proofs; strip Unsloth's --load-mode only, keep record.
                    _projector_msg = self._is_projector_incompatibility(out)
                    _projector_memory = self._is_gpu_memory_start_failure(out)
                    _signal_mmproj_guess = self._is_signal_crash(
                        _crash_rc
                    ) and not self._output_has_nonprojector_diagnostic(out)
                    if (
                        launched_with_mmproj
                        and not _load_cancelled()
                        and (_projector_msg or _projector_memory or _signal_mmproj_guess)
                    ):
                        _vision_gpu_cmd = list(_last_spawn_cmd)
                        _cpu_projector_cmd = None
                        if not _projector_msg and _paravirtual_mmproj_pinnable(server_caps):
                            _cpu_projector_cmd = self._with_mmproj_offload_disabled(
                                _vision_gpu_cmd, env
                            )

                        if _cpu_projector_cmd is not None:
                            # The fit priced the projector as GPU-eligible, and this
                            # retry moves those bytes into host RAM, so neither the
                            # VRAM-only proof (which never asked what RAM holds) nor the
                            # VRAM-plus-RAM one (answered with the projector on the card)
                            # covers the launch this respawns. Only Unsloth's own tokens;
                            # a user's --load-mode is theirs. The record is NOT cleared:
                            # the fallbacks below respawn from an argv that carries these
                            # tokens and has to be able to name them.
                            if self._fit_load_mode_flags:
                                _stripped_cpu_projector_cmd = _without_subsequence(
                                    _cpu_projector_cmd, self._fit_load_mode_flags
                                )
                                if _stripped_cpu_projector_cmd != _cpu_projector_cmd:
                                    # Same as the no-flash rung: the record follows the
                                    # argv, or a stale "reserving" spends a full reload
                                    # the running server already satisfies.
                                    self._record_memory_state(_stripped_cpu_projector_cmd, env)
                                    logger.info(
                                        "Load mode: dropping the fit's --load-mode none "
                                        "for the CPU-projector retry; it moves the "
                                        "projector into host RAM the fit credited to VRAM."
                                    )
                                _cpu_projector_cmd = _stripped_cpu_projector_cmd
                            # The managed pair stays: this retry moves the PROJECTOR, which
                            # clip.cpp loads through its own ifstream, and the main model is
                            # still fully offloaded. The block above is the fit's VRAM
                            # accounting, which the move really does void.
                            logger.warning(
                                "llama-server failed while loading this model's GPU "
                                "vision projector (--mmproj); retrying with the "
                                "projector on CPU to preserve image input."
                            )
                            cmd = _cpu_projector_cmd
                            _unified_withdraw_if_unneeded(
                                cmd, why = "The retry with the projector on CPU"
                            )
                            healthy = _spawn_and_wait(cmd, label = "-mmproj-cpu")
                            if healthy:
                                self._mmproj_fallback_reason = "cpu_offload"
                                logger.warning(
                                    "Vision projector loaded on CPU after GPU startup "
                                    "failed; image input remains available for this session."
                                )
                            else:
                                _cpu_projector_out = "\n".join(self._stdout_lines[-50:])
                                # Snapshot: re-reading races the teardown.
                                _proc_snap6 = self._process
                                _cpu_projector_rc = (
                                    _proc_snap6.poll() if _proc_snap6 is not None else None
                                )
                                self._kill_process()
                                if _finish_cancelled_health_wait(
                                    "Load cancelled during the CPU-projector health wait"
                                ):
                                    return False
                                if _load_cancelled():
                                    _cleanup_cancelled_load(
                                        "Load cancelled after the CPU-projector health wait"
                                    )
                                    return False
                                if self._is_projector_incompatibility(_cpu_projector_out):
                                    _projector_msg = True
                                elif self._is_gpu_memory_start_failure(_cpu_projector_out):
                                    # The projector was already off the GPU, so removing it
                                    # cannot repair this allocation failure; keep the real error.
                                    _raise_terminal_load_failure(
                                        self._classify_llama_start_failure(
                                            _cpu_projector_out,
                                            gguf_path,
                                            self._model_identifier,
                                            _cpu_projector_rc,
                                            binary,
                                            self._llama_log_path,
                                            (self._api_key,),
                                            self._extra_args,
                                        )
                                    )
                        elif _projector_memory and _paravirtual_mmproj_pinnable(server_caps):
                            # An env/argv pin already put mmproj on CPU. Text-only
                            # cannot free additional GPU memory, so surface the OOM.
                            _raise_terminal_load_failure(
                                self._classify_llama_start_failure(
                                    out,
                                    gguf_path,
                                    self._model_identifier,
                                    _crash_rc,
                                    binary,
                                    self._llama_log_path,
                                    (self._api_key,),
                                    self._extra_args,
                                )
                            )

                        if not healthy:
                            if _projector_msg:
                                logger.warning(
                                    "llama-server could not load this model's vision "
                                    "projector (--mmproj). The installed llama.cpp build is "
                                    "likely too old for it. Loading text-only for this "
                                    "session; run 'unsloth studio update' to enable vision."
                                )
                                self._mmproj_fallback_reason = "projector_incompatible"
                            else:
                                logger.warning(
                                    "llama-server could not start with this model's vision "
                                    "projector (--mmproj), including the CPU-projector "
                                    "recovery when available. Retrying text-only for this "
                                    "session; check memory, GPU/driver logs, or update Unsloth."
                                )
                                self._mmproj_fallback_reason = "projector_startup_failure"
                            cmd = self._strip_mmproj_args(_vision_gpu_cmd)
                            cmd = self._drop_managed_lazy_mode(cmd, "text-only retry")
                            # Keep a trailing sidecar pair: rewriting it would drop the drafter's
                            # reserve.
                            if [
                                "--fit-target",
                                str(_last_flag_value(cmd, {"-fitt", "--fit-target"})),
                            ] not in _sidecar_override_pairs:
                                cmd = self._per_device_fit_target_argv(
                                    cmd, env, _fit_target_margins_by_idx
                                )
                            _unified_withdraw_if_unneeded(cmd, why = "The text-only retry")
                            # This retry bypasses _spawn_and_wait, so refresh the launched-argv snapshot here.
                            _last_spawn_cmd = list(cmd)
                            self._is_vision = False
                            self._mmproj_has_audio = False
                            if not self._start_llama_process(
                                cmd,
                                env,
                                child_gpu_physical_ids = _child_gpu_physical_ids,
                            ):
                                # Repriced here: a failed text-only retry can replay the vision argv on CPU, loading the projector.
                                _cleanup_cancelled_load("App shut down during the text-only retry")
                                return False
                            _textonly_timeout, _textonly_kwargs = _vulkan_wait_args("-textonly")
                            if self._wait_for_health(
                                timeout = _textonly_timeout,
                                cancelled = _load_cancelled,
                                **_textonly_kwargs,
                            ):
                                healthy = True
                                # The child that serves this session never read the
                                # CPU-pinned projector, so the host-memory advisory the
                                # APU preflight charged it against is now describing a
                                # load nothing is running. Priced again here, and not at
                                # the strip above, because a failed text-only retry can
                                # still replay the ORIGINAL vision argv on CPU further
                                # down -- projector and all -- and that child does load
                                # the bytes. Only a healthy text-only server is evidence.
                                self._reprice_after_dropping_pinned_projector(
                                    apu_msg = _apu_ram_msg,
                                    host_msg = _host_ram_msg,
                                    model_size = _apu_resident_model_size,
                                    pinned_bytes = _mmproj_pinned_bytes,
                                    avail_mib = _apu_avail_mib,
                                    part = _apu_ram_part,
                                )
                            else:
                                # Read the exit code before _kill_process() clears it, so
                                # an OS-killed text-only retry still gets the OOM message.
                                # Snapshot: re-reading races the teardown.
                                _retry_proc = self._process
                                _retry_rc = _retry_proc.poll() if _retry_proc is not None else None
                                # This retry bypasses _spawn_and_wait's abort detection.
                                if _retry_rc not in (None, 0) and self._is_sched_reserve_abort(
                                    "\n".join(self._stdout_lines[-200:])
                                ):
                                    _sched_abort_seen = True
                                self._kill_process()
                                if _finish_cancelled_health_wait(
                                    "Load cancelled during the text-only retry health wait"
                                ):
                                    return False
                                if _vulkan_stall_pending():
                                    if _recover_vulkan_stall(
                                        _last_spawn_cmd if _projector_msg else _vision_gpu_cmd
                                    ):
                                        healthy = True
                                        # The replay carries the original vision argv, as below.
                                        if not _projector_msg:
                                            self._mmproj_fallback_reason = None
                                    elif _finish_cancelled_health_wait(
                                        "Load cancelled during the CPU replay health wait"
                                    ):
                                        return False
                                # A text-only signal crash is independent evidence of a GPU
                                # startup fault. Keep a confirmed bad projector out of the
                                # replay; a guessed one still gets a CPU try with vision.
                                if not healthy and self._is_signal_crash(_retry_rc):
                                    _cpu_replay_cmd = (
                                        _last_spawn_cmd if _projector_msg else _vision_gpu_cmd
                                    )
                                    if _try_auto_vulkan_cpu_fallback(
                                        _cpu_replay_cmd,
                                        _retry_rc,
                                    ):
                                        healthy = True
                                        # A whole-runtime CPU replay with the original
                                        # vision argv supersedes the text-only diagnosis.
                                        if not _projector_msg:
                                            self._mmproj_fallback_reason = None
                                    else:
                                        if _finish_cancelled_health_wait(
                                            "Load cancelled during the CPU replay health wait"
                                        ):
                                            return False
                                        _raise_terminal_load_failure(
                                            self._gpu_init_crash_message(binary)
                                        )
                                if not healthy:
                                    _retry_detail = self._classify_llama_start_failure(
                                        "\n".join(self._stdout_lines[-50:]),
                                        gguf_path,
                                        self._model_identifier,
                                        _retry_rc,
                                        binary,
                                        self._llama_log_path,
                                        (self._api_key,),
                                        self._extra_args,
                                    )
                                    if _finish_cancelled_health_wait(
                                        "Load cancelled during the text-only recovery"
                                    ):
                                        return False
                                    _raise_terminal_load_failure(
                                        self._mmproj_retry_failure_message(
                                            projector_confirmed = _projector_msg,
                                            detail = _retry_detail,
                                        )
                                    )
                    else:
                        # Try the drafter launch non-terminally so the drafterless argv can still recover; snapshot that argv, since a spawned replay rebinds _last_spawn_cmd.
                        _drafterless_cpu_replay_cmd = list(_last_spawn_cmd)
                        _replayed = _spec_cpu_replay_cmd is not None and (
                            _try_auto_vulkan_cpu_fallback(
                                _spec_cpu_replay_cmd, _crash_rc, terminal = False
                            )
                        )
                        if _replayed:
                            # The drafter came back with the replay, so the
                            # speculative-disable diagnosis no longer holds.
                            self._spec_fallback_reason = None
                        elif _try_auto_vulkan_cpu_fallback(_drafterless_cpu_replay_cmd, _crash_rc):
                            _replayed = True
                        if _replayed:
                            healthy = True
                        else:
                            if _finish_cancelled_health_wait(
                                "Load cancelled during the CPU replay health wait"
                            ):
                                return False
                            _raise_terminal_load_failure(
                                self._classify_llama_start_failure(
                                    out,
                                    gguf_path,
                                    self._model_identifier,
                                    _crash_rc,
                                    binary,
                                    self._llama_log_path,
                                    (self._api_key,),
                                    self._extra_args,
                                )
                            )

                if _restored_cpu_reprice is not None:
                    # Micro-batch uses the LAUNCHED slot count: the drafter drop and non-MTP retry hand slots back later.
                    self._reprice_after_cpu_only_fallback(
                        host_msg = _host_ram_msg,
                        cpu_cmd = _last_spawn_cmd or cmd,
                        env = env,
                        avail_mib = _preflight_avail_mib,
                        pageable_note = (
                            _restored_cpu_reprice
                            if isinstance(_restored_cpu_reprice, str)
                            else None
                        ),
                    )
                elif not intent.cpu_fallback:
                    _gpu_commit_notice = self._windows_gpu_commit_notice(
                        getattr(self._process, "pid", None), binary
                    )
                    if _gpu_commit_notice:
                        if self._last_load_warning:
                            self._amend_load_warning(" " + _gpu_commit_notice)
                        else:
                            self._record_load_warning(_gpu_commit_notice)
                if not self._publish_healthy():
                    # Teardown began after the 200; publishing would report a model that is gone.
                    _cleanup_cancelled_load("App shut down as the load was completing")
                    return False
                self._commit_effective_parallel_slots(n_parallel)
                self._swa_full = swa_full
                self._kv_cache_unified = kv_cache_unified
                # Request's pair beside the launched one: the matcher must use it, else repeat loads are rejected.
                _launched_ubatch = _ubatch_for_slots(n_parallel)
                self._n_ubatch = max(
                    0,
                    int(self._DEFAULT_N_UBATCH if _launched_ubatch is None else _launched_ubatch),
                )
                # What the child is RUNNING: the same architecture rule the plan above used.
                self._flash_attn_enabled = _flash_attn_enabled_from_args(
                    _last_spawn_cmd, default = not _flash_attn_known_off, env = env
                ) and not _architecture_forces_flash_attn_off(self._architecture)
                self._effective_cache_types = _effective_main_cache_types(
                    _last_spawn_cmd,
                    env,
                )
                # The pair the REQUEST resolved to, kept beside the pair that launched.
                # A launch-time rewrite makes them differ (no --flash-attn resets a
                # quantized V cache to f16, as does the flash-attn crash recovery), and
                # the matcher wants the request: comparing the running pair against an
                # unrewritten request rejects every identical repeat load.
                self._requested_cache_types = _planned_cache_pair
                self._kv_cache_context_total = effective_ctx if effective_ctx > 0 else None

                # -- the length --fit chose, or a --parallel slot split -- so the
                # Adopt the context the server allocated (--fit or slot split); clear video state first, or a swap to a model without video inherits the old answer.
                self._has_video_input = False
                self._video_needs_ffmpeg = False
                # Use _extra_args_as_requested: the drafter-drop snapshot would restore the Metal zero-context strip.
                self._reconcile_effective_ctx_with_server(ctx_override or 0)
                if self._has_video_input:
                    _supports_dir = server_caps.get("supports_video_ffmpeg_dir") is not False
                    self._has_video_input = _video_tools_reachable(
                        _last_spawn_cmd, env, _spawn_cwd, supports_ffmpeg_dir = _supports_dir
                    )
                    self._video_needs_ffmpeg = not self._has_video_input
                    if self._has_video_input:
                        try:
                            self._video_ffmpeg_dir = _video_tools_search(
                                _last_spawn_cmd, env, _spawn_cwd, supports_ffmpeg_dir = _supports_dir
                            )
                        except Exception:
                            pass
                    if not self._has_video_input:
                        logger.warning(
                            "Video input disabled: llama-server decodes video with ffmpeg and "
                            "ffprobe, and neither --video-ffmpeg-dir nor PATH provides both."
                        )
                if self._kv_cache_context_total is not None:
                    self._n_ubatch = min(
                        self._n_ubatch,
                        self._kv_cache_context_total,
                    )

                # Commit caller intent only after _healthy, so a failed start cannot poison the next inheritance check. None keeps prior, [] clears, list sets.
                if extra_args is not None:
                    # Persist device-stripped extras so a later reload cannot resurrect the dropped --device.
                    self._extra_args = (
                        self._strip_device_extra_args(extra_args)
                        if _gpu_ids_own_device_flags
                        else list(extra_args)
                    )
                    # Budget this child was sized for; None if placement never used it. Committed with known-good state.
                    _pv_requested = (
                        _extra_args_as_requested
                        if _extra_args_as_requested is not None
                        else (
                            _pv_suppressed_spec_extra_args
                            if _pv_suppressed_spec_extra_args is not None
                            else extra_args
                        )
                    )
                    self._requested_extra_args = (
                        self._strip_device_extra_args(_pv_requested)
                        if _gpu_ids_own_device_flags
                        else list(_pv_requested)
                    )
                    self._extra_args_source = (model_identifier, hf_variant)
                # A re-fit replay records what launched, which is what the client replays next.
                self._requested_n_ctx = int(effective_ctx if _replayed_ctx_refit else n_ctx)
                # Local n_parallel may have been reduced above; the snapshot has the ask.
                self._requested_n_parallel = max(1, int(intent.n_parallel))
                self._slots_cut_for_drafter = (
                    _mtp_slots_asked is not None and n_parallel < _mtp_slots_asked
                )
                # Commit with the rest of the known-good state: only a launch that got
                # this far replaced the runtime, so a rejected preflight leaves the old
                # marker intact, and the diffusion runner returns well before it without
                # consuming, or paying for, any llama-server capability.
                self._capability_probe_inconclusive = _launch_probe_inconclusive
                # The budget this child was sized against, committed with the rest of
                # the known-good state so a duplicate-load fast path or failed launch
                # leaves the previous marker intact and the route keeps asking for the
                # reload. None when placement never used it (_budget_priced_placement).
                self._vram_fraction_launched = _budget_priced_placement()
                self._vram_fraction_pending = None
                self._requested_n_batch = intent.n_batch
                self._requested_n_ubatch = intent.n_ubatch
                self._requested_load_mode = intent.load_mode
                self._requested_spec_draft_cache_type = intent.spec_draft_cache_type
                self._requested_ctx_checkpoints = intent.ctx_checkpoints
                self._requested_cache_ram = intent.cache_ram
                # Commit the known-good snapshot + whether MTP is live, then
                # watch this load for a mid-generation crash. The verified-cache hint is
                # dropped: a respawn replays this snapshot arbitrarily later, so recovery
                # re-resolves the file instead of trusting a path this request verified.
                self._last_load_intent = replace(intent, verified_gguf = None)
                if not getattr(_AUTO_RECOVERY_LOAD, "active", False):
                    # A caller's load starts a fresh OOM episode; only automatic recovery carries
                    # the strike count.
                    self._metal_oom_strikes = None
                    self._metal_oom_terminal_key = None
                    self._gpu_oom_strikes = None
                self._mtp_runtime_fallback_active = _mtp_active_for_launched_server
                self._start_mtp_crash_watchdog()

                # Catch silent CPU fallback when GPU was intended; a deliberate zero-offload load must read False, not None, or training needlessly unloads it.
                _deliberate_cpu_only = (
                    gpu_memory_mode == "manual" and gpu_layers == 0
                ) or _arch_gate_forced_cpu
                # Publish on every load so a stale value is never read for the new child.
                self._arch_gate_forced_cpu = bool(_arch_gate_forced_cpu)
                if _arch_gate_forced_cpu:
                    # False, never None: training spares only an exactly-False server.
                    self._gpu_offload_active = False
                elif _deliberate_cpu_only:
                    self._gpu_offload_active = self._zero_offload_gpu_flag(
                        _last_spawn_cmd, _detected_gpus, env
                    )
                else:
                    self._gpu_offload_active = self._classify_gpu_offload(
                        gpu_indices is not None or use_fit or gpu_memory_mode == "manual",
                        _detected_gpus,
                    )
                # CPU-only hosts log 0/N too, so a zero needs an expected GPU; a positive count stands.
                _offload_counts = (
                    None
                    if _arch_gate_forced_cpu or _deliberate_cpu_only
                    else parse_gpu_offload_counts(self._stdout_lines)
                )
                if (
                    _offload_counts is not None
                    and _offload_counts[0] <= 0
                    and not (
                        (_detected_gpus or _metal_capable_host())
                        and (gpu_indices is not None or use_fit or gpu_memory_mode == "manual")
                    )
                ):
                    _offload_counts = None
                self._gpu_offload_layers = _offload_counts
                self._weight_placement = parse_weight_placement(
                    self._stdout_lines,
                    _gguf_set_bytes(Path(self._gguf_path)) if self._gguf_path else None,
                )
                self._gpu_backend_unavailable = bool(_detected_gpus or _metal_capable_host()) and (
                    llama_saw_gpu_device(self._stdout_lines) is False
                )
                # Not _GPU_OFFLOAD_OVERRIDE_FLAGS: --fit on is exactly the split to report.
                self._offload_overridden = (
                    _extra_args_set_any_flag(extra_args, _GPU_LAYER_FLAGS)
                    or _device_selection_is_cpu(
                        self._strip_device_extra_args(extra_args)
                        if _gpu_ids_own_device_flags
                        else extra_args,
                        env,
                    )
                    or _env_fixes_gpu_layers(env)
                )
                if (
                    self._gpu_offload_layers is not None
                    and 0 < self._gpu_offload_layers[0] < self._gpu_offload_layers[1]
                ):
                    logger.warning(
                        "llama-server offloaded %d of %d layers to GPU; the rest run on "
                        "CPU, which is much slower for a dense model.",
                        self._gpu_offload_layers[0],
                        self._gpu_offload_layers[1],
                    )
                if self._gpu_offload_active is False and not _deliberate_cpu_only:
                    logger.warning(
                        "llama-server appears to have loaded the model entirely "
                        "on CPU even though Unsloth detected at least one GPU. "
                        "This usually means the prebuilt binary's GPU backend "
                        "failed to load -- on Windows, cudart64_X.dll / "
                        "cublas64_X.dll could not be resolved. Reinstall the "
                        "Unsloth llama.cpp prebuilt or install a matching CUDA "
                        "toolkit (issue unslothai/unsloth#5106).",
                    )

                logger.info(
                    f"llama-server ready on port {self._port} for model '{model_identifier}'"
                )
                # Poll llama-server /metrics -> vLLM-style engine_stats logs
                # (only when the binary exposes /metrics).
                if server_caps.get("supports_metrics"):
                    try:
                        from core.inference.llama_stats import maybe_start_stats_logger
                        if self._stats_logger is not None:
                            self._stats_logger.stop()
                        self._stats_logger = maybe_start_stats_logger(
                            self.base_url, logger, headers = self._auth_headers
                        )
                    except Exception as e:
                        logger.debug(f"engine-stats logger not started: {e}")
                else:
                    self._stats_logger = None

            # Probe outside _lock (interruptible by /unload); init inside.
            self._is_audio = False
            self._audio_type = None
            self._audio_probed = False
            self._has_audio_input = False
            try:
                detected = self._detect_audio_type_strict()
                self._audio_probed = True
            except Exception as exc:
                logger.debug("Audio probe failed: %s", exc)
                detected = None
            applied_audio = (
                self._apply_detected_audio(detected)
                if intent.audio_codec_path is None
                else self._apply_detected_audio(detected, intent.audio_codec_path)
            )
            if not applied_audio:
                return False

            if not self._healthy:
                return False
            if _load_cancelled():
                # Cancelled during the post-health setup or the audio probe. Publishing
                # now leaves the child resident while the cancel route sees is_loaded.
                _cleanup_cancelled_load(
                    "Load cancelled after llama-server became healthy; unwinding"
                )
                return False
            # Snapshot loaded files so save_slots_for_resume() can refuse after on-disk swaps.
            if self._slot_save_dir:
                self._slot_loaded_identity = (
                    self._gguf_file_identity(self._gguf_path),
                    self._slot_launch_fingerprint(),
                )
            return True

    def _build_speculative_flags(
        self,
        *,
        speculative_type: Optional[str],
        spec_draft_n_max: Optional[int],
        extra_args: Optional[List[str]],
        model_identifier: str,
        model_path: Optional[str],
        gpus: bool,
        binary: Optional[str],
        mtp_draft_path: Optional[str] = None,
        dspark_draft_path: Optional[str] = None,
        dspark_fit_sized: bool = True,
        dflash_draft_path: Optional[str] = None,
        dflash_fit_sized: bool = True,
        drafter_no_vram: bool = False,
        embedded_mtp_partial_offload: bool = False,
        draft_device: Optional[str] = None,
        mtp_drafter_unloadable: bool = False,
        eagle3_draft_path: Optional[str] = None,
        mtp_trained: Optional[bool] = None,
    ) -> List[str]:
        """Returns spec flags; sets speculative type, requested mode, draft n-max. Auto uses ngram-mod for sub-3B embedded MTP (regresses); separate drafters exempt."""
        flags: List[str] = []
        # Reset; emit branches re-set on the resolved emission.
        self._spec_draft_n_max = None
        self._speculative_type = None
        self._spec_fallback_reason = None
        # A drafter trained with the fine-tune and exported beside this GGUF (drafter.json).
        trained_steps = None
        if eagle3_draft_path is None or mtp_trained is None:
            sidecar_eagle3, sidecar_trained, trained_steps = _trained_drafter_sidecar(model_path)
            eagle3_draft_path = eagle3_draft_path or sidecar_eagle3
            mtp_trained = sidecar_trained if mtp_trained is None else mtp_trained

        # Canonical UI-facing requested mode (legacy values mapped via
        # _canonicalize_spec_mode).
        canonical_mode = _canonicalize_spec_mode(speculative_type)
        user_owns_spec_type = _extra_args_set_spec_type(extra_args)
        if (
            not user_owns_spec_type
            and canonical_mode in ("auto", "mtp", "mtp+ngram")
            and self._nextn_predict_layers
            and mtp_draft_path
        ):
            # Backstop for callers that bypass load_model's discovery normalization.
            mtp_draft_path = None
        # A dropped sidecar is this quant's only MTP signal; keep it so it reaches the fallback.
        is_mtp_model = (
            bool(self._nextn_predict_layers)
            or _is_mtp_model_name(model_identifier, model_path)
            or bool(mtp_draft_path)
            # Drafter kind is set before fallbacks erase it; under Auto, no later field keeps it.
            or mtp_drafter_unloadable
        )
        _mtp_size_b = _extract_model_size_b(model_identifier)
        # The sub-3B regression is an embedded-head cost; a separate drafter
        # (Gemma) is a cheap standalone model that wins below 3B, so exempt it.
        # A head the user trained with this fine-tune is what they asked to draft with, so the
        # size floor (a stock-head cost measurement) and the MLA default-off below do not apply.
        _mtp_too_small = (
            _mtp_size_b is not None
            and _mtp_size_b < _MTP_MIN_SIZE_B
            and not bool(mtp_draft_path)
            and not mtp_trained
        )
        # Drafterless MTP (a sidecar dropped as unopenable included) would abort llama-server, so fall back; without that case MTP is silently disabled.
        _mtp_drafter_missing = (
            not mtp_draft_path
            and not self._nextn_predict_layers
            and (_is_gemma_mtp_name(model_identifier, model_path) or mtp_drafter_unloadable)
        )
        # MLA embedded MTP is ~2x slower than no spec, so Auto drops it; override with forced mtp or UNSLOTH_MLA_MTP_ENABLED=1. Separate drafters are unaffected.
        _auto_mla_embedded_mtp = (
            bool(self._nextn_predict_layers)
            and self._kv_lora_rank is not None
            and not bool(mtp_draft_path)
            and not _mla_mtp_auto_enabled()
            and not _arch_has_fast_mla_mtp(getattr(self, "_architecture", None))
            and not mtp_trained
        )

        if user_owns_spec_type:
            # User --spec-type wins; skip auto-emit to avoid a duplicate spec block.
            self._requested_spec_mode = None
            return flags

        effective_mode = canonical_mode or "auto"
        self._requested_spec_mode = effective_mode
        if effective_mode == "off":
            return flags
        if effective_mode == "ngram-simple":
            flags.extend(["--spec-type", "ngram-simple"])
            self._speculative_type = "ngram-simple"
            return flags
        caps = self.probe_server_capabilities(binary)

        # Which drafter every branch below is about. Set once here, before any
        # fallback can erase the evidence: the UI names the recovery from this,
        # and under Auto neither the requested mode nor the resolved
        # _speculative_type ("default" after a fallback) still carries the kind.
        _auto_mode = (canonical_mode or "auto") == "auto"
        if canonical_mode == "eagle3" or (
            _auto_mode and eagle3_draft_path and caps.get("supports_eagle3")
        ):
            self._spec_drafter_kind = "eagle3"
            if not caps.get("supports_eagle3"):
                logger.warning(
                    "EAGLE-3 requested but llama-server lacks --spec-type draft-eagle3; "
                    "loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "binary_no_mtp"
                return flags
            if not eagle3_draft_path:
                logger.warning(
                    "EAGLE-3 requested but no trained eagle3-*.gguf drafter sits next to this "
                    "model; loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "drafter_not_found"
                return flags
            draft_n_max = (
                int(spec_draft_n_max) if spec_draft_n_max is not None else (trained_steps or 3)
            )
            if spec_draft_n_max is not None:
                self._spec_draft_n_max = draft_n_max
            n_max_flag = caps.get("spec_draft_n_max_flag") or "--spec-draft-n-max"
            flags.extend(
                [
                    "--model-draft",
                    eagle3_draft_path,
                    "--spec-type",
                    "draft-eagle3",
                    str(n_max_flag),
                    str(draft_n_max),
                ]
            )
            if draft_device:
                flags.extend(["--spec-draft-device", draft_device])
            self._speculative_type = "draft-eagle3"
            logger.info("Spec decoding: draft-eagle3 using %s", eagle3_draft_path)
            return flags
        if canonical_mode == "dspark" or (
            _auto_mode and dspark_draft_path and caps.get("supports_dspark")
        ):
            self._spec_drafter_kind = "dspark"
        elif canonical_mode == "dflash" or (
            _auto_mode and dflash_draft_path and caps.get("supports_dflash")
        ):
            self._spec_drafter_kind = "dflash"
        else:
            self._spec_drafter_kind = "mtp"

        def _resolved_draft_n_max() -> int:
            # Like DSpark, the sidecar borrows target token_embd/output, so llama.cpp cannot size its ~1.5 GB.
            if spec_draft_n_max is not None:
                n = int(spec_draft_n_max)
                self._spec_draft_n_max = n
                return n
            if mtp_trained and trained_steps and self._spec_drafter_kind == "mtp":
                return trained_steps  # the depth the exported head was trained for
            return 2 if gpus else 3

        def _emit_dspark() -> None:
            """Appends --model-draft <sidecar> with draft-dspark; callers check the sidecar and
            capability first."""
            if not dspark_fit_sized:
                # Not a blocker: the sidecar borrows the target's embeddings, so fit cannot reserve for it; the load still works, with ~11 GB missing from the fit budget.
                logger.info(
                    "DSpark under --fit on: llama.cpp cannot size the sidecar during "
                    "fitting, so its ~%.1f GB is not reserved. Pin GPU Layers in "
                    "Manual mode if the load runs out of VRAM.",
                    (self._get_gguf_size_bytes(dspark_draft_path) or 0) / 1e9,
                )
            draft_n_max = int(spec_draft_n_max) if spec_draft_n_max is not None else 3
            if spec_draft_n_max is not None:
                self._spec_draft_n_max = draft_n_max
            n_max_flag = caps.get("spec_draft_n_max_flag") or "--spec-draft-n-max"
            flags.extend(
                [
                    "--model-draft",
                    dspark_draft_path,
                    "--spec-type",
                    "draft-dspark",
                    str(n_max_flag),
                    str(draft_n_max),
                ]
            )
            self._speculative_type = "draft-dspark"
            logger.info("Spec decoding: draft-dspark using %s", dspark_draft_path)

        if effective_mode == "dspark":
            # Check capability first: a missing sidecar is usually the binary's fault.
            if not caps.get("supports_dspark"):
                logger.warning(
                    "DSpark requested but llama-server lacks --spec-type "
                    "draft-dspark; loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "binary_no_mtp"
                return flags
            if not dspark_draft_path:
                logger.warning(
                    "DSpark requested but no matching dspark-*.gguf sidecar was found; "
                    "loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "drafter_not_found"
                return flags
            _emit_dspark()
            return flags

        def _emit_dflash() -> None:
            """Appends --model-draft <sidecar> with draft-dflash; Auto picks it whenever the model
            ships one."""
            if not dflash_fit_sized:
                # Same shape as DSpark: the sidecar ships no token_embd/output and
                # borrows the target's, so llama.cpp cannot build a standalone
                # draft context to measure it (llama-context.cpp:153-160) and
                # skips the reserve (server-context.cpp:1190-1193). The load still
                # works; only the ~1.5 GB is missing from the fit budget, which is
                # an order of magnitude less than the DSpark case.
                logger.info(
                    "DFlash under --fit on: llama.cpp cannot size the sidecar during "
                    "fitting, so its ~%.1f GB is not reserved. Pin GPU Layers in "
                    "Manual mode if the load runs out of VRAM.",
                    (self._get_gguf_size_bytes(dflash_draft_path) or 0) / 1e9,
                )
            draft_n_max = _resolved_draft_n_max()
            n_max_flag = caps.get("spec_draft_n_max_flag") or "--spec-draft-n-max"
            flags.extend(
                [
                    "--model-draft",
                    dflash_draft_path,
                    "--spec-type",
                    "draft-dflash",
                    str(n_max_flag),
                    str(draft_n_max),
                ]
            )
            if draft_device:
                # Check capability first, as DSpark does: a missing sidecar is usually the binary's fault.
                flags.extend(["--spec-draft-device", draft_device])
            self._speculative_type = "draft-dflash"
            logger.info("Spec decoding: draft-dflash using %s", dflash_draft_path)

        if effective_mode == "dflash":
            # Separate-file drafter (Gemma) gets --model-draft path; baked-in heads (Qwen) pass none.
            if not caps.get("supports_dflash"):
                logger.warning(
                    "DFlash requested but llama-server lacks --spec-type "
                    "draft-dflash; loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "binary_no_mtp"
                return flags
            if not dflash_draft_path:
                logger.warning(
                    "DFlash requested but no matching dflash-*.gguf sidecar was found; "
                    "loading without speculative decoding."
                )
                flags.append("--spec-default")
                self._speculative_type = "default"
                self._spec_fallback_reason = "drafter_not_found"
                return flags
            _emit_dflash()
            return flags

        def _emit_mtp(*, chain_ngram: bool) -> bool:
            """Append --spec-type mtp[/draft-mtp][,ngram-mod] + n-max."""
            mtp_token = caps.get("mtp_token") if caps else None
            if not mtp_token:
                inconclusive = bool(caps.get("mtp_probe_inconclusive")) if caps else True
                if inconclusive:
                    logger.info(
                        "Requested MTP speculative decoding but llama-server MTP "
                        "capability probe was inconclusive; loading without "
                        "speculative decoding."
                    )
                else:
                    logger.warning(
                        "Requested MTP speculative decoding but "
                        "llama-server lacks --spec-type mtp/draft-mtp; "
                        "run `unsloth studio update`. Loading without "
                        "speculative decoding."
                    )
                # Override inherited LLAMA_ARG_SPEC_TYPE=draft-mtp (CLI wins over env) so the child matches the capability gate and the no-MTP budget.
                flags.append("--spec-default")
                self._speculative_type = "default"
                if inconclusive:
                    self._spec_fallback_reason = None
                else:
                    self._spec_fallback_reason = "binary_no_mtp"
                return False
            draft_n_max = _resolved_draft_n_max()
            n_max_flag = caps.get("spec_draft_n_max_flag") or "--spec-draft-n-max"
            # Record the stand-down so it takes; reason binary_outdated, since ngram-mod has no drafter kind.
            if mtp_draft_path:
                flags.extend(["--model-draft", mtp_draft_path])
                if draft_device:
                    flags.extend(["--spec-draft-device", draft_device])
                logger.info(f"Using separate MTP drafter: {mtp_draft_path}")
            spec_value = mtp_token
            ngram_knobs: list[str] = []
            if chain_ngram:
                ngram_knobs = _build_ngram_mod_flags(caps, chain_with_mtp = True)
                if ngram_knobs:
                    spec_value = f"ngram-mod,{mtp_token}"
                else:
                    logger.warning(
                        "llama-server lacks ngram-mod tuning "
                        "flags; loading MTP only (no ngram chain)"
                    )
            flags.extend(["--spec-type", spec_value, n_max_flag, str(draft_n_max)])
            flags.extend(ngram_knobs)
            self._speculative_type = "draft-mtp"
            chain_label = "chained ngram-mod" if chain_ngram else "MTP-only"
            logger.info(f"Spec decoding: {mtp_token} ({chain_label})")
            return True

        def _emit_ngram_mod() -> bool:
            """Append --spec-type ngram-mod + flag-set knobs."""
            if not caps.get("supports_ngram_mod"):
                logger.warning(
                    "Requested ngram-mod speculative decoding but llama-server "
                    "does not advertise ngram-mod support; run `unsloth studio "
                    "update`. Loading without speculative decoding."
                )
                # The stand-down the warning tells the user to fix, recorded so the
                # update actually takes: an unrecorded one leaves the picker adopting
                # this process forever. "binary_outdated" rather than "binary_no_mtp"
                # because the retry rule for that one asks _SPEC_KIND_CAPABILITY about
                # _spec_drafter_kind, which ngram-mod never sets, having no drafter.
                self._spec_fallback_reason = "binary_outdated"
                return False
            ngram_knobs = _build_ngram_mod_flags(caps)
            flags.extend(["--spec-type", "ngram-mod"])
            if not ngram_knobs:
                logger.warning(
                    "llama-server lacks ngram-mod tuning "
                    "flags; loading without --spec-ngram-mod-* knobs"
                )
            flags.extend(ngram_knobs)
            self._speculative_type = "ngram-mod"
            logger.info("Spec decoding: ngram-mod")
            return True

        def _fallback_drafter_not_found() -> None:
            """Drafterless Gemma: use ngram-mod (or spec-default) and record why."""
            if mtp_drafter_unloadable:
                logger.warning(
                    "Model %s had an MTP drafter that llama-server cannot open as "
                    "--model-draft; falling back without it.",
                    model_identifier,
                )
            else:
                logger.warning(
                    "Model %s is MTP-capable but no drafter or head was found; "
                    "falling back. Check network or run `unsloth studio update`.",
                    model_identifier,
                )
            if caps.get("supports_ngram_mod"):
                _emit_ngram_mod()
            else:
                flags.append("--spec-default")
                self._speculative_type = "default"
            # Auto stands down embedded Hybrid Mamba MTP on partial offload: rollback pushes layers to CPU.
            self._spec_fallback_reason = (
                "drafter_unloadable" if mtp_drafter_unloadable else "drafter_not_found"
            )

        if effective_mode == "ngram":
            _emit_ngram_mod()
            return flags
        if effective_mode in ("mtp", "mtp+ngram"):
            chain_ngram = effective_mode == "mtp+ngram"
            if not is_mtp_model:
                if chain_ngram:
                    logger.warning(
                        "MTP+Ngram requested but this GGUF has no MTP head or "
                        "drafter; loading ngram-mod only."
                    )
                    _emit_ngram_mod()
                else:
                    logger.warning(
                        "MTP requested but this GGUF has no MTP head or drafter; "
                        "loading without speculative decoding."
                    )
                    flags.append("--spec-default")
                    self._speculative_type = "default"
                return flags
            if _mtp_drafter_missing:
                _fallback_drafter_not_found()
                return flags
            if _mtp_too_small:
                logger.warning(
                    f"Forcing {effective_mode.upper()} on a {_mtp_size_b:.1f}B model; "
                    "the bench shows MTP regresses below 3B. "
                    "Engaging anyway (user override)."
                )
            _emit_mtp(chain_ngram = chain_ngram)
            return flags

        # No vision gate (DFlash measured with mmproj); MLA with no sidecar gets its own reason, not VRAM.
        if embedded_mtp_partial_offload and caps.get("mtp_token"):
            logger.info(
                "Auto: embedded MTP is disabled because the Hybrid Mamba target "
                "requires partial CPU offload; ordinary decoding is faster at this "
                "placement. Choose MTP explicitly to force it."
            )
            flags.extend(["--spec-type", "none"])
            self._speculative_type = "none"
            self._spec_fallback_reason = "mtp_partial_offload"
            # Depth changes the rollback copies priced into the fit, so it must trigger a reload.
            if spec_draft_n_max is not None:
                self._spec_draft_n_max = int(spec_draft_n_max)
            return flags

        # DSpark first: it beats other Auto outcomes for this architecture; without it, --spec-default.
        _no_vram_drops_a_real_drafter = (
            bool(dspark_draft_path and caps.get("supports_dspark"))
            or bool(dflash_draft_path and caps.get("supports_dflash"))
            or not _auto_mla_embedded_mtp
        )
        if drafter_no_vram and _no_vram_drops_a_real_drafter:
            # The fit reserved nothing for the drafter, so emitting one would OOM.
            self._spec_fallback_reason = "drafter_no_vram"
            if caps.get("supports_ngram_mod"):
                logger.info(
                    "Auto: the drafter does not fit in VRAM alongside the model at this "
                    "context, so it is dropped and ngram-mod (zero-VRAM) takes its "
                    "place. Choose the drafter in the Speculative Decoding dropdown to "
                    "force it at a smaller context."
                )
                _emit_ngram_mod()
            else:
                # ngram-mod is not in this build's enum and llama-server aborts on unknown values.
                logger.info(
                    "Auto: the drafter does not fit in VRAM alongside the model at this "
                    "context, so speculative decoding is disabled (this llama-server "
                    "does not advertise ngram-mod). Choose the drafter in the "
                    "Speculative Decoding dropdown to force it at a smaller context."
                )
        elif dspark_draft_path and caps.get("supports_dspark"):
            # DFlash needs no opt-in unlike DSpark: its sidecar ships in the model repo; DSpark is tried first.
            _emit_dspark()
        elif dflash_draft_path and caps.get("supports_dflash"):
            # MLA embedded MTP regresses, so Auto uses ngram-mod; UNSLOTH_MLA_MTP_ENABLED=1 re-enables it.
            _emit_dflash()
        elif _auto_mla_embedded_mtp:
            # MLA embedded-MTP (GLM-5.2 et al.): the MTP path regresses vs spec-off
            # on llama.cpp today, so Auto drops it and falls back to ngram-mod (or
            # spec-off if unsupported), mirroring the sub-3B branch. Forced mtp /
            # mtp+ngram (handled above) still engage; UNSLOTH_MLA_MTP_ENABLED=1
            # re-enables this promotion once upstream optimizes the path.
            self._spec_fallback_reason = "mla_mtp_disabled"
            if caps.get("supports_ngram_mod"):
                logger.info(
                    "Auto: MLA embedded-MTP model detected; llama.cpp's MLA/DSA "
                    "MTP path is slower than no speculation, so using ngram-mod "
                    "instead. Override via the Unsloth Speculative Decoding "
                    "dropdown or UNSLOTH_MLA_MTP_ENABLED=1."
                )
                _emit_ngram_mod()
            else:
                logger.info(
                    "Auto: MLA embedded-MTP model detected; disabling speculative "
                    "decoding (this llama-server does not advertise ngram-mod). "
                    "Override via the dropdown or UNSLOTH_MLA_MTP_ENABLED=1."
                )
                # spec-off: emit nothing, mirroring the sub-3B no-ngram path.
        elif is_mtp_model and not _mtp_too_small:
            if _mtp_drafter_missing:
                # Name-only MTP, drafter did not resolve (download failed/absent).
                _fallback_drafter_not_found()
            else:
                # GPU: MTP-only. CPU/Mac: chain ngram-mod + MTP.
                _emit_mtp(chain_ngram = not gpus)
        elif is_mtp_model and _mtp_too_small:
            # Sub-3B fallback: drop the MTP draft head, keep ngram-mod when
            # the binary supports it.
            if _mtp_drafter_missing:
                _fallback_drafter_not_found()
            elif caps.get("supports_ngram_mod"):
                logger.info(
                    f"MTP GGUF detected but model size {_mtp_size_b:.1f}B "
                    "is below the 3B speedup threshold; using ngram-mod "
                    "only (zero-VRAM, no draft head). Override via "
                    "--spec-type or the Unsloth Speculative Decoding "
                    "dropdown."
                )
                _emit_ngram_mod()
            else:
                logger.info(
                    f"MTP GGUF detected but model size {_mtp_size_b:.1f}B "
                    "is below the 3B speedup threshold and the bundled "
                    "llama-server does not advertise ngram-mod; "
                    "auto-disabling speculative decoding."
                )
        else:
            # Non-MTP model: let llama-server choose its default strategy.
            flags.append("--spec-default")
            self._speculative_type = "default"
        return flags

    def adopt_load_intent_if_matched(self, intent: GgufLoadIntent) -> bool:
        """Match live state and adopt the caller's compatible GPU placement."""
        if not self.matches_load_source(intent):
            return False
        # A custom-path save leaves the model intent unchanged, but the
        # resident process still belongs to the previously selected runtime.
        if self._binary_changed_since_launch():
            logger.info("llama-server selection changed since launch; forcing a reload")
            return False
        intent = self._preserve_cpu_fallback_intent(intent, source_matches = True)
        # Budget is server-wide, so a save with same intent must reload; None if placement never used it.
        if _metal_device_is_paravirtual():
            (
                _pv_mode,
                _pv_layers,
                _pv_tp,
                _pv_split,
                _pv_cpu_moe,
                _pv_extras,
            ) = paravirtual_normalized_request(
                gpu_memory_mode = intent.gpu_memory_mode,
                gpu_layers = intent.gpu_layers,
                tensor_parallel = intent.tensor_parallel,
                tensor_split = list(intent.tensor_split) if intent.tensor_split else None,
                n_cpu_moe = intent.n_cpu_moe,
                extra_args = list(intent.extra_args) if intent.extra_args is not None else None,
                log_dropped = False,
            )
            intent = replace(
                intent,
                gpu_memory_mode = _pv_mode,
                gpu_layers = _pv_layers,
                tensor_parallel = _pv_tp,
                tensor_split = tuple(_pv_split) if _pv_split else None,
                n_cpu_moe = _pv_cpu_moe,
                extra_args = tuple(_pv_extras) if _pv_extras is not None else None,
            )
        effective_extra_args = (
            list(intent.extra_args) if intent.extra_args is not None else self._extra_args
        )
        candidate_extra_args = list(effective_extra_args) if effective_extra_args else []
        if intent.extra_args is not None and self._gpu_ids_own_placement(intent.gpu_ids):
            candidate_extra_args = self._strip_device_extra_args(candidate_extra_args)
        # Model Memory toggles change only launch flags, so compare them explicitly.
        if not memory_state_satisfies_settings(
            self._memory_state,
            self._memory_policy_active,
            self._memory_mlock_applicable,
            self._memory_direct_io,
            self._memory_dio_applicable,
            bool(self._memory_dio_flags),
        ):
            logger.info("Model Memory policy changed since launch; forcing a reload")
            return False
        # Use the pending marker: an in-flight load's budget was captured under the lock, not re-read.
        _launched_frac = self._vram_fraction_launched
        # The pending marker, not a fresh read: a load in flight captured its budget
        # under the lock, and re-reading could decide against a different number than
        # the plan that follows. The route's fast path finds it unset and resolves
        # live as before. Reading the marker rather than taking a parameter keeps the
        # signature every caller and test double already has.
        _pending_frac = self._vram_fraction_pending
        _active_frac = _pending_frac if _pending_frac is not None else _active_vram_fraction()
        if _launched_frac is not None and float(_launched_frac) != _active_frac:
            logger.info("VRAM budget changed since launch; forcing a reload")
            return False
        if self._hand_added_projector_since_launch(candidate_extra_args):
            logger.info("A projector was added beside the cached weight; forcing a reload")
            return False
        if not self._runtime_matches_intent(intent, candidate_extra_args):
            return False
        self._record_matching_gpu_request(
            list(intent.gpu_ids) if intent.gpu_ids is not None else None
        )
        return True

    def _hand_added_projector_since_launch(self, extra_args: list) -> bool:
        """A text-only -hf launch whose repo publishes no projector, with one now
        dropped beside the cached weight (#9286). Without this the next Apply dedupes
        onto the projector-less server until Studio restarts."""
        last = getattr(self, "_last_load_intent", None)
        if (
            self._is_vision
            or self._disable_vision
            or not self._hf_repo
            or not self._gguf_path
            or last is None
            or last.is_vision
            or extra_args_disable_mmproj(extra_args)
        ):
            return False
        try:
            from utils.models.model_config import _hf_cached_local_mmproj
            return _hf_cached_local_mmproj(self._gguf_path) is not None
        except Exception as exc:
            logger.debug("Hand-added mmproj probe failed: %s", exc)
            return False

    def matches_load_source(self, intent: GgufLoadIntent) -> bool:
        """Whether the resident model has the intent's identity and weights."""
        # A retired child stays recorded until respawned: an identical load must replace it.
        _retired = getattr(self, "_retired_process", None)
        if _retired is not None and getattr(self, "_process", None) is _retired:
            return False
        if not self.is_loaded or (self._model_identifier or "").lower() != (
            intent.model_identifier.lower()
        ):
            return False
        custom_mmproj = _extra_args_device(intent.extra_args, {"--mmproj", "-mm"})
        source_path = intent.gguf_path
        if source_path is None and custom_mmproj is not None:
            if (self._hf_variant or "").lower() != (intent.hf_variant or "").lower():
                return False
            source_path = self._gguf_path
        if source_path is not None and self._gguf_path:
            resident_identity = getattr(self, "_gguf_load_identity", None)
            if resident_identity is not None:
                candidate_identity = LlamaCppBackend._gguf_load_source_identity(
                    source_path,
                    custom_mmproj or intent.mmproj_path,
                )
                return candidate_identity == resident_identity
            try:
                return Path(self._gguf_path).resolve() == Path(source_path).resolve()
            except OSError:
                return False
        return (self._hf_variant or "").lower() == (intent.hf_variant or "").lower()

    def components_match_intent(self, intent: GgufLoadIntent) -> bool:
        """``_runtime_matches_intent`` minus capacity and placement: same weights and components."""
        if intent.force_reload or not self.matches_load_source(intent):
            return False
        requested = self.requested_extra_args
        # Inherited extras would carry the resident's adapters and drafters unnamed by the caller.
        if intent.extra_args_inherited and requested:
            return False
        extras = requested if intent.extra_args_inherited else intent.extra_args
        if tuple(extras or ()) != tuple(requested or ()):
            return False
        if (self._chat_template_override or None) != (intent.chat_template_override or None):
            return False
        if not self._is_diffusion and (
            bool(self._disable_vision) != bool(intent.disable_vision)
            or self._requested_reasoning_budget
            != resolve_reasoning_budget(extras, intent.reasoning_budget)
            or self._requested_reasoning_budget_message
            != resolve_reasoning_budget_message(extras, intent.reasoning_budget_message)
        ):
            return False
        if _extra_args_set_spec_type(extras):
            return self._requested_spec_mode is None
        return (_canonicalize_spec_mode(intent.speculative_type) or "auto") == (
            self._requested_spec_mode or "auto"
        )

    def _classify_gpu_offload(
        self, expected_gpu: bool, detected_gpus: list[tuple[int, int]]
    ) -> Optional[bool]:
        """True if on GPU, False if only CPU buffers despite GPU intent, None with no signal; delegates to the shared classifier that tracks current llama.cpp logs."""
        if not detected_gpus or not expected_gpu:
            return None
        return classify_gpu_offload_lines(self._stdout_lines)

    @staticmethod
    def _cmd_has_gpu_companion(cmd: list, env: Optional[Mapping[str, str]] = None) -> bool:
        """A --mmproj or GPU drafter offloads regardless of --gpu-layers; CPU-forced drafters do not
        count."""
        _args = [str(a) for a in cmd]
        if any(a.startswith("--mmproj") or _flag_name(a) == "-mm" for a in _args):
            # --no-mmproj-offload disables clip.cpp's GPU backend, so the projector holds no VRAM.
            _off = [a for a in _args if a in ("--mmproj-offload", "--no-mmproj-offload")]
            if not _off or _off[-1] != "--no-mmproj-offload":
                return True
        if _extra_args_mtp_draft_path(cmd, env) is None:
            return False
        return not _extra_args_draft_offloaded_to_cpu(cmd, env)

    @staticmethod
    def _zero_offload_keeps_gpu_visible(cmd: list, env: Optional[Mapping[str, str]] = None) -> bool:
        """Shared with child masking: a device pin, tensor mode, mmproj or GPU drafter keeps GPUs
        visible."""
        return (
            LlamaCppBackend._cmd_has_gpu_device_pin(cmd, env)
            or _effective_tensor_parallel(cmd, False, env)
            or LlamaCppBackend._cmd_has_gpu_companion(cmd, env)
        )

    @staticmethod
    def _cmd_has_gpu_device_pin(cmd: list, env: Optional[Mapping[str, str]] = None) -> bool:
        """True when the effective main or draft ``--device`` pin names a GPU."""
        main_flags = {"--device", "-dev"}
        draft_flags = {"--spec-draft-device", "-devd", "--device-draft"}
        last_main: Optional[str] = None
        last_draft: Optional[str] = None
        args = [str(arg) for arg in cmd]
        for index, raw in enumerate(args):
            flag = _flag_name(raw)
            _, equals, inline = raw.partition("=")
            if flag not in main_flags and flag not in draft_flags:
                continue
            value = inline if equals else (args[index + 1] if index + 1 < len(args) else "")
            if flag in main_flags:
                last_main = value
            else:
                last_draft = value
        if last_main is None:
            last_main = (env or {}).get("LLAMA_ARG_DEVICE")

        def _names_gpu(value: Optional[str]) -> bool:
            if value is None:
                return False
            devices = [item.strip().lower() for item in value.split(",") if item.strip()]
            return not devices or any(item not in ("cpu", "none") for item in devices)

        return _names_gpu(last_main) or _names_gpu(last_draft)

    @staticmethod
    def _zero_offload_gpu_flag(
        spawn_cmd: list,
        detected_gpus: list,
        env: Optional[Mapping[str, str]] = None,
    ) -> Optional[bool]:
        """Residency flag for a manual zero-offload load: counted classifiers miss pins, mmproj and drafter VRAM, so reuse the launch predicate; None means no signal."""
        if LlamaCppBackend._is_vulkan_backend():
            # Unlike -ngl 0, the recovery's last-wins --device none removes Vulkan.
            main_device = (_extra_args_main_device(spawn_cmd) or "").strip().lower()
            if main_device in ("cpu", "none"):
                return LlamaCppBackend._zero_offload_keeps_gpu_visible(spawn_cmd, env)
            if detected_gpus:
                return True
        if not detected_gpus:
            return None
        return LlamaCppBackend._zero_offload_keeps_gpu_visible(spawn_cmd, env)
