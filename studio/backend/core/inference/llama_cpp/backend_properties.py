# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: properties."""

import httpx
import math

from dataclasses import replace
from pathlib import Path
from typing import (
    List,
    Optional,
)
from core.inference.llama_server_args import (
    _effective_tensor_parallel,
    _tensor_parallel_matches_loaded,
    resolve_reasoning_budget,
    resolve_reasoning_budget_message,
)

from .core_types import (
    GgufLoadIntent,
    _EMPTY_THINK_PROBE_LOCK,
    _SPEC_KIND_CAPABILITY,
)
from .draft_args import _normalized_load_mode
from .gguf_cache import (
    _SHARD_RE,
    _mtp_drafter_loads_standalone,
)
from .launch_flags import (
    _diffusion_manual_ngl,
    _swa_full_from_args_or_env,
)
from .launch_support import _local_ssl_context
from .log import logger
from .memory_policy import (
    _extra_args_set_spec_type,
    _planned_main_cache_types,
)
from .placement_args import _metal_device_is_paravirtual
from .reasoning import (
    _REASONING_EFFORT_CLAMPS_LOGGED,
    _clamp_reasoning_effort_to_levels,
    _coerce_reasoning_effort,
)
from .server_log import _GPU_MODEL_BUFFER_RE
from .spec_args import _canonicalize_spec_mode
from core.inference.context_window import estimate_message_tokens_with_priced_images
from core.inference.context_window import messages_with_priced_images
from typing import Iterable
from typing import Mapping
import os
import sys
from .batch_args import (
    _extra_args_n_parallel,
    _extra_args_split_mode,
)
from .draft_args import (
    _extra_args_device,
    _extra_args_main_device,
    _sidecar_adapter_paths,
)
from .gguf_cache import (
    _gguf_set_bytes,
)
from .launch_flags import (
    _args_place_tensors_on_cpu,
    _env_places_tensors_on_cpu,
    _kv_offload_from_args,
)
from .launch_support import (
    _extra_args_have_tensor_split,
)
from .reasoning import (
    _fit_reasoning_effort_to_template,
    _rendered_chat_template,
)
from .server_log import (
    _extra_weight_file_bytes,
)
from .tool_loop_text import (
    _image_token_price,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _PropertiesMixin:
    # ── Properties ────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._process is not None and self._healthy

    @property
    def is_active(self) -> bool:
        """True if a llama-server process exists (loading or loaded)."""
        proc = self._process
        if proc is None:
            return False
        return not (
            not self._healthy
            and proc is getattr(self, "_retired_process", None)
            and getattr(self, "_metal_oom_terminal_key", None) is not None
        )

    def resident_memory_mib(self) -> Optional[int]:
        """Best-effort MiB the loaded model holds: its GPU model buffers when known, else the
        child's RSS (CPU, and Apple unified memory where RSS covers the Metal buffers). Never raises."""
        process = self._process
        if process is None or process.poll() is not None:
            return None
        try:
            gpu = self.reclaimable_gpu_memory_gb()
            if gpu:
                return int(round(sum(gpu.values()) * 1024))
            import psutil

            return int(psutil.Process(process.pid).memory_info().rss // (1024 * 1024)) or None
        except Exception:  # noqa: BLE001 - a size is decoration, never a failed status read
            return None

    def reclaimable_gpu_memory_gb(self) -> Optional[dict[int, float]]:
        """A conservative lower bound from the startup log's model buffers; a live memory sample
        could race."""
        process = self._process
        if process is None or process.poll() is not None:
            return None
        physical_ids = list(getattr(self, "_child_gpu_physical_ids", None) or ())
        if not physical_ids:
            return None
        # The buffer lines are startup output, so once healthy the answer is fixed per process; the status poll
        # would otherwise rescan up to _STDOUT_TRIM_AT lines on the event loop every few seconds.
        ids = tuple(physical_ids)
        cached = getattr(self, "_reclaimable_gpu_cache", None)
        if cached is not None and cached[0] is process and cached[1] == ids:
            return dict(cached[2]) if cached[2] else None
        healthy = bool(getattr(self, "_healthy", False))
        reported: Optional[dict[int, float]] = {}
        for line in list(self._stdout_lines):
            match = _GPU_MODEL_BUFFER_RE.search(line)
            if match is None:
                continue
            ordinal = int(match.group(1))
            if ordinal >= len(physical_ids):
                reported = None
                break
            physical = int(physical_ids[ordinal])
            reported[physical] = reported.get(physical, 0.0) + float(match.group(2)) / 1024.0
        if self._process is not process or process.poll() is not None:
            return None
        if healthy:
            self._reclaimable_gpu_cache = (process, ids, dict(reported) if reported else None)
        return reported or None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def _auth_headers(self) -> "Optional[dict[str, str]]":
        """Bearer header for the child's per-launch --api-key, else None (key disabled)."""
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None

    @property
    def model_identifier(self) -> Optional[str]:
        return self._model_identifier

    def _estimate_countable_message(self, message: dict) -> int:
        return estimate_message_tokens_with_priced_images(message, self._price_image)

    def _countable_messages(self, messages: list[dict]) -> list[dict]:
        """The counting copy every fit and size probe uses: images charged, audio and video dropped."""
        from core.inference.mcp_images import prepare_image_turn_boundaries

        # Before pricing: a stand-in hides an image turn from Ministral's alternation fix-up.
        messages = prepare_image_turn_boundaries(
            messages,
            getattr(self, "_chat_template_override", None) or getattr(self, "_chat_template", None),
        )
        return messages_with_priced_images(messages, self._price_image)

    def _price_image(self, part: dict) -> int:
        return _image_token_price(self, part)

    @property
    def is_vision(self) -> bool:
        """Whether this model takes image input; ``_is_vision`` only records that a
        projector was attached, which happens for audio input too."""
        return self._is_vision and self._mmproj_accepts_image

    @property
    def is_diffusion(self) -> bool:
        """True when the loaded GGUF is a block-diffusion model (DiffusionGemma)."""
        return self._is_diffusion

    @property
    def swa_full(self) -> bool:
        """Whether the active llama-server received full-size SWA mode."""
        return self._swa_full

    @property
    def hf_variant(self) -> Optional[str]:
        return self._hf_variant

    @property
    def gguf_path(self) -> Optional[str]:
        return self._gguf_path

    @property
    def hf_repo(self) -> Optional[str]:
        """HF repo of the loaded model, or None for local/native file loads."""
        return self._hf_repo

    @property
    def mtp_draft_path(self) -> Optional[str]:
        return self._mtp_draft_path

    @property
    def mtp_draft_suppressed_path(self) -> Optional[str]:
        """Drafter the last load resolved and then dropped on purpose, else None.
        Duplicate-load checks accept it alongside mtp_draft_path so a drafter still on
        disk does not read as a settings change."""
        return self._mtp_draft_suppressed_path

    @property
    def spec_fallback_reason(self) -> Optional[str]:
        """Why MTP was disabled on the last MTP-requesting load, else None."""
        return self._spec_fallback_reason

    @property
    def mmproj_fallback_reason(self) -> Optional[str]:
        """How the active model recovered from an mmproj startup failure, else None."""
        return self._mmproj_fallback_reason

    def _binary_changed_since_launch(self) -> bool:
        """Whether a different llama-server is installed than the live one was launched
        from. False when either side is unreadable, so an install still in flight is not
        read as a finished one; the next Apply asks again."""
        # Same path space as the launch (macOS resolves managed entrypoints), else every Apply reloads.
        return self._binary_changed_since_revision(self._launch_binary_revision)

    def _binary_changed_since_revision(self, revision: tuple) -> bool:
        """No captured revision means no baseline, and discovery cannot recover one, so answer False."""
        if not revision:
            return False
        current = self._binary_revision(
            self._exec_path_for_launch(self._find_llama_server_binary())
        )
        if not current:
            return False
        return current != revision

    def spec_binary_fallback_can_retry(self) -> bool:
        """Checks the drafter's capability, not the stand-down reason; binary_outdated compares the file."""
        if self._spec_fallback_reason == "binary_outdated":
            return self._binary_changed_since_launch()
        if self._spec_fallback_reason != "binary_no_mtp":
            return False
        if self._launch_binary_revision and not self._binary_changed_since_launch():
            # Skip the probe for an untouched file: it advertises nothing new, and this runs on every load check.
            return False
        capability = _SPEC_KIND_CAPABILITY.get(self._spec_drafter_kind or "")
        if capability is None:
            return False
        try:
            return bool(self.probe_server_capabilities().get(capability))
        except Exception as exc:
            logger.debug("Could not recheck the speculative capability: %s", exc)
            return False

    @property
    def extra_args(self) -> Optional[List[str]]:
        """Extra llama-server flags from the last load (a copy). None =
        never set, [] = explicitly cleared. Used by the route for
        inheritance."""
        return list(self._extra_args) if self._extra_args is not None else None

    @property
    def requested_extra_args(self) -> Optional[List[str]]:
        """Extras as invoked, before launch-time rewrites, so a rewrite does not look like a
        settings change."""
        stored = (
            self._requested_extra_args
            if self._requested_extra_args is not None
            else self._extra_args
        )
        return list(stored) if stored is not None else None

    @property
    def requested_n_ctx(self) -> int:
        """n_ctx the last load was invoked with (not the effective cap).
        0 means Auto. Used by the route to detect Auto-vs-explicit flips."""
        return self._requested_n_ctx

    @property
    def reasoning_budget(self) -> int:
        return self._reasoning_budget

    @property
    def reasoning_budget_message(self) -> str:
        return self._reasoning_budget_message

    @property
    def extra_args_source(self) -> Optional[tuple[str, Optional[str]]]:
        """(model_identifier, hf_variant) the stored extra_args came from.
        ``None`` if no extras have ever been recorded. Used by the route
        to refuse cross-model inheritance (#5401)."""
        return self._extra_args_source

    @property
    def context_length(self) -> Optional[int]:
        """Return the effective context length the server is running at."""
        return self._effective_context_length or self._context_length

    @property
    def effective_parallel_slots(self) -> int:
        """Return the serving-slot count the active llama-server actually uses."""
        try:
            slots = int(getattr(self, "_effective_parallel_slots", 1))
        except (TypeError, ValueError):
            slots = 1
        return max(1, slots)

    @property
    def offloaded_layers(self) -> Optional[int]:
        """Layers llama.cpp actually put on a GPU, when its log said so."""
        return self._gpu_offload_layers[0] if self._gpu_offload_layers else None

    @property
    def offload_total_layers(self) -> Optional[int]:
        return self._gpu_offload_layers[1] if self._gpu_offload_layers else None

    @property
    def expert_cache_mib(self) -> Optional[float]:
        return self._placement_value("expert_cache_mib")

    @property
    def overflow_layers(self) -> Optional[int]:
        """Under --moe-cache-mib auto every routed expert is in RAM, so this is every MoE layer."""
        if self._placement_value("all_experts_in_ram"):
            return self.n_moe_layers or None
        return self._placement_value("overflow_layers")

    @property
    def host_weights_mib(self) -> Optional[float]:
        return self._placement_value("host_mib")

    @property
    def gpu_weights_mib(self) -> Optional[float]:
        return self._placement_value("gpu_mib")

    def _placement_value(self, key: str):
        placement = getattr(self, "_weight_placement", None)
        return placement.get(key) if placement else None

    @property
    def gpu_backend_unavailable(self) -> bool:
        """Studio found a GPU for this load but llama.cpp reported none."""
        return self._gpu_backend_unavailable

    @property
    def offload_overridden(self) -> bool:
        """Whether the user's own extras (Auto mode keeps -ngl) chose this split."""
        return self._offload_overridden

    @property
    def requested_parallel_slots(self) -> int:
        """--parallel the last load asked for, before any fit-time reduction.
        The reload dedupe compares requested-vs-requested (like requested_n_ctx);
        the effective count would reload forever after a fitter reduction."""
        try:
            slots = int(getattr(self, "_requested_n_parallel", 1))
        except (TypeError, ValueError):
            slots = 1
        return max(1, slots)

    @property
    def requested_n_batch(self) -> Optional[int]:
        """--batch-size the last load asked for; None means llama.cpp default."""
        return self._requested_n_batch

    @property
    def requested_n_ubatch(self) -> Optional[int]:
        """--ubatch-size the last load asked for; None means llama.cpp default."""
        return self._requested_n_ubatch

    @property
    def requested_load_mode(self) -> Optional[str]:
        """The caller's --load-mode intent, not any Model Memory override, so dedupe compares like
        with like."""
        return self._requested_load_mode

    @property
    def requested_spec_draft_cache_type(self) -> Optional[str]:
        """Draft KV cache dtype the last load asked for; None means f16."""
        return self._requested_spec_draft_cache_type

    @property
    def requested_ctx_checkpoints(self) -> Optional[int]:
        """--ctx-checkpoints the last load asked for; None means the default 32."""
        return self._requested_ctx_checkpoints

    @property
    def requested_cache_ram(self) -> Optional[int]:
        """--cache-ram (MiB) the last load asked for; None means the default 8192."""
        return self._requested_cache_ram

    @property
    def idle_slot_clearing_active(self) -> bool:
        """Idle KV cells come back only if True; False means nothing may be handed to another caller."""
        return self._idle_slot_clearing_active

    @property
    def max_context_length(self) -> Optional[int]:
        """UI safe-zone threshold: the VRAM-fit cap, or 4096 when weights exceed 90% of every subset."""
        return self._max_context_length or self._context_length

    @property
    def vram_fit_context_length(self) -> Optional[int]:
        """Largest context the load-time fit priced inside GPU / Metal memory, else None."""
        # A child wholly on CPU (--device none, Vulkan crash fallback, zero layers) holds no VRAM.
        if (
            self._gpu_offload_active is False
            or self._cpu_fallback_reason
            or self._arch_gate_forced_cpu
            or (self._gpu_memory_mode == "manual" and self._gpu_layers == 0)
        ):
            return None
        return self._vram_fit_context_length

    @property
    def native_context_length(self) -> Optional[int]:
        """Return the model's native context length from GGUF metadata."""
        return self._context_length

    def _commit_effective_parallel_slots(self, n_parallel: int) -> None:
        try:
            slots = int(n_parallel)
        except (TypeError, ValueError):
            slots = 1
        self._effective_parallel_slots = max(1, slots)

    def _reset_effective_parallel_slots(self) -> None:
        self._effective_parallel_slots = 1
        # Cleared with the effective count so a stale value can't skew the dedupe.
        self._requested_n_parallel = 1
        self._slots_cut_for_drafter = False
        self._requested_n_batch = None
        self._requested_n_ubatch = None
        self._requested_load_mode = None
        self._requested_spec_draft_cache_type = None
        self._requested_ctx_checkpoints = None
        self._requested_cache_ram = None
        self._idle_slot_clearing_active = False

    @staticmethod
    def _read_rss_bytes(pid: int) -> Optional[int]:
        """``RssFile`` of ``pid`` (Linux), ``VmRSS`` before kernel 4.5; 0 when neither, None without /proc.
        Not ``RssAnon``: CUDA pins the host weight buffer (``CUDA_Host``) before a byte of it is read."""
        vm_rss = None
        try:
            with open(f"/proc/{pid}/status", "r", encoding = "utf-8") as f:
                for line in f:
                    # IndexError guards a line with no value column.
                    if line.startswith("RssFile:"):
                        return int(line.split()[1]) * 1024  # kB -> bytes
                    if line.startswith("VmRSS:"):
                        vm_rss = int(line.split()[1]) * 1024
        except (FileNotFoundError, PermissionError, ValueError, IndexError, OSError):
            return None
        return 0 if vm_rss is None else vm_rss

    @staticmethod
    def _darwin_rusage(pid: int) -> Optional[tuple]:
        """``(ri_diskio_bytesread, ri_phys_footprint)`` of ``pid`` from proc_pid_rusage, or None."""
        import ctypes

        class _RusageInfoV2(ctypes.Structure):
            _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
                (name, ctypes.c_uint64)
                for name in (
                    "ri_user_time",
                    "ri_system_time",
                    "ri_pkg_idle_wkups",
                    "ri_interrupt_wkups",
                    "ri_pageins",
                    "ri_wired_size",
                    "ri_resident_size",
                    "ri_phys_footprint",
                    "ri_proc_start_abstime",
                    "ri_proc_exit_abstime",
                    "ri_child_user_time",
                    "ri_child_system_time",
                    "ri_child_pkg_idle_wkups",
                    "ri_child_interrupt_wkups",
                    "ri_child_pageins",
                    "ri_child_elapsed_abstime",
                    "ri_diskio_bytesread",
                    "ri_diskio_byteswritten",
                )
            ]

        try:
            info = _RusageInfoV2()
            # RUSAGE_INFO_V2 = 2
            if (
                ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(pid, 2, ctypes.byref(info))
                != 0
            ):
                return None
            return int(info.ri_diskio_bytesread), int(info.ri_phys_footprint)
        except (OSError, AttributeError, ValueError):
            return None

    @staticmethod
    def _read_load_bytes(pid: int) -> Optional[int]:
        """Bytes ``pid`` has taken in, for a load that streams weights to VRAM and keeps little RSS.

        Linux: ``rchar`` (page cache hits included). Windows: psutil read_bytes. macOS: the larger of
        disk bytes read (a cold mmap Metal load) and physical footprint (a copy into Metal or CPU
        buffers)."""
        if os.path.isdir("/proc"):
            try:
                with open(f"/proc/{pid}/io", "r", encoding = "utf-8") as f:
                    for line in f:
                        if line.startswith("rchar:"):
                            return int(line.split()[1])
            except (FileNotFoundError, PermissionError, ValueError, IndexError, OSError):
                return None
            return None
        if sys.platform == "darwin":
            usage = LlamaCppBackend._darwin_rusage(pid)
            return max(usage) if usage else None
        if sys.platform != "win32":
            return None
        try:
            import psutil
            return int(psutil.Process(pid).io_counters().read_bytes)
        except Exception:
            return None

    def load_progress(self) -> Optional[dict]:
        """Return live model-load progress, or None if not loading.

        ``bytes_loaded`` is the larger of the bytes read and Linux file-backed RSS; never a working set,
        which counts buffers allocated before any read. The total includes projector, drafter and adapters.

        Shape::

            {
                "phase": "mmap" | "ready",
                "bytes_loaded": int,   # as above, never decreasing
                "bytes_total":  int,   # weight files the server loads
                "fraction": float,     # bytes_loaded / bytes_total, 0..1
            }
        """
        proc = self._process
        if proc is None:
            return None
        pid = proc.pid
        if pid is None:
            return None

        gguf_path = self._gguf_path
        bytes_total = _gguf_set_bytes(Path(gguf_path)) if gguf_path else 0
        launched = getattr(self, "_process_env", None)
        bytes_total += _extra_weight_file_bytes(
            getattr(proc, "args", None),
            launched[1] if launched is not None and launched[0] is proc else None,
        )

        rss = LlamaCppBackend._read_rss_bytes(pid)
        read = LlamaCppBackend._read_load_bytes(pid)
        if rss is None and read is None:
            return None
        bytes_loaded = max(rss or 0, read or 0)

        # RSS drops once -ngl offloads to VRAM; keep a high-water mark so progress never regresses.
        hwm_pid, hwm = getattr(self, "_load_rss_hwm", (None, 0))
        hwm = bytes_loaded if hwm_pid != pid else max(hwm, bytes_loaded)
        self._load_rss_hwm = (pid, hwm)
        bytes_loaded = hwm

        phase = "ready" if self._healthy else "mmap"
        fraction = 0.0
        if bytes_total > 0:
            # Reads past the weights (headers, a second pass) must not push the figure past the total.
            bytes_loaded = min(bytes_loaded, bytes_total)
            fraction = min(1.0, bytes_loaded / bytes_total)
        # Healthy means loaded: offloaded mmap pages leave RSS far below the shard total.
        if self._healthy:
            if bytes_total > 0:
                bytes_loaded = bytes_total
            fraction = 1.0
        return {
            "phase": phase,
            "bytes_loaded": bytes_loaded,
            "bytes_total": bytes_total,
            "fraction": round(fraction, 4),
        }

    @property
    def chat_template(self) -> Optional[str]:
        return self._chat_template

    @property
    def markup_profile(self):
        """None if template and vocab are unreadable; curated patterns then apply so the model stays swept, not unprotected (getattr: test doubles skip __init__)."""
        cached = getattr(self, "_markup_profile", None)
        if cached is None:
            from core.inference.chat_template_helpers import model_markup
            cached = (
                model_markup(
                    getattr(self, "_chat_template_override", None)
                    or getattr(self, "_chat_template", None),
                    getattr(self, "_markup_tokens", None),
                ),
            )
            try:
                self._markup_profile = cached
            except AttributeError:
                pass
        return cached[0]

    @property
    def chat_template_override(self) -> Optional[str]:
        return self._chat_template_override

    def _effective_chat_template(self, chat_template_override: Optional[str]) -> Optional[str]:
        """Caller override, else a repaired GGUF template when llama-server Jinja cannot parse it; repair stays out of _chat_template_override or every Apply reloads."""
        if chat_template_override:
            return chat_template_override
        from core.inference.chat_template_helpers import repair_numeric_member_access

        repaired = repair_numeric_member_access(self._chat_template)
        if repaired:
            logger.warning(
                "The GGUF's chat template indexes with numeric member access, which "
                "llama-server's Jinja rejects; launching with a repaired copy"
            )
        return repaired

    @property
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    @property
    def reasoning_always_on(self) -> bool:
        return self._reasoning_always_on

    @property
    def reasoning_style(self) -> str:
        return self._reasoning_style

    @property
    def reasoning_effort_levels(self) -> list:
        """Discrete reasoning_effort levels the template offers; empty for the plain low/medium/high ladder."""
        return self._reasoning_effort_levels

    @property
    def supports_preserve_thinking(self) -> bool:
        return self._supports_preserve_thinking

    @property
    def preserve_thinking_default(self) -> bool:
        return self._preserve_thinking_default

    @property
    def reasoning_default(self) -> bool:
        return self._reasoning_default

    def _reasoning_kwargs(
        self,
        enable_thinking: bool,
        effort: Optional[str] = None,
    ) -> dict:
        if self._reasoning_style == "enable_thinking_effort":
            # GLM-5.2 style: enable_thinking is the on/off gate; when on, keep the template's default effort.
            return {"enable_thinking": enable_thinking}
        if self._reasoning_style == "reasoning_effort":
            effort = _fit_reasoning_effort_to_template(
                _rendered_chat_template(self),
                effort or ("high" if enable_thinking else "low"),
                thinking_off = not effort and not enable_thinking,
            )
            return _coerce_reasoning_effort(
                getattr(self, "_architecture", None), {"reasoning_effort": effort}
            )
        return {"enable_thinking": enable_thinking}

    def _replay_preserve_thinking(
        self,
        preserve_thinking: Optional[bool],
        messages,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Optional[bool]:
        """True for a thinking-off follow-up with no reasoning in its history, where the template proves
        that only preserve_thinking replays each answer's empty think block as generated."""
        # Omitted reads as off; the probe proves what replays.
        if preserve_thinking is True or not getattr(self, "_supports_preserve_thinking", False):
            return preserve_thinking
        mode = self._request_reasoning_kwargs(enable_thinking, reasoning_effort, None) or {}
        thinking = mode.get("enable_thinking")
        if thinking is None and enable_thinking is None and reasoning_effort is None:
            # Agents rarely send either: llama-server then renders the load-time default.
            thinking = getattr(self, "_reasoning_default", None)
        if thinking is not False:
            return preserve_thinking
        answered = False
        for message in messages or ():
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            answered = True
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                return preserve_thinking
            # Inline reasoning counts too.
            content = message.get("content")
            parts = content if isinstance(content, list) else [content]
            if any(
                "</think>" in (part.get("text") if isinstance(part, dict) else part)
                for part in parts
                if isinstance(part, str)
                or (isinstance(part, dict) and isinstance(part.get("text"), str))
            ):
                return preserve_thinking
        if not answered:
            return preserve_thinking
        return True if self._empty_think_replays_with_preserve() else preserve_thinking

    def _empty_think_probe_key(self) -> tuple:
        return (
            getattr(self, "base_url", None),
            id(getattr(self, "_chat_template", None)),
            id(getattr(self, "_chat_template_override", None)),
        )

    def _empty_think_replays_with_preserve(self) -> bool:
        """Probed once per server: generation prompt + answer prefixes the next render only when kept."""
        key = self._empty_think_probe_key()
        cached = getattr(self, "_empty_think_preserve_probe", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        with _EMPTY_THINK_PROBE_LOCK:
            cached = getattr(self, "_empty_think_preserve_probe", None)
            if cached is not None and cached[0] == key:
                return cached[1]
            return self._probe_empty_think_replay(key)

    def _probe_empty_think_replay(self, key: tuple) -> bool:
        result = False
        try:
            off = dict(self._request_reasoning_kwargs(False, None, None) or {})
            answer = "Studio replay probe answer."
            first = [{"role": "user", "content": "Studio replay probe question."}]
            history = first + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "Studio replay probe follow-up."},
            ]
            with httpx.Client(
                timeout = 10,
                headers = self._auth_headers,
                trust_env = False,
                verify = _local_ssl_context(),
            ) as client:
                refused = []

                def _render(msgs, preserve):
                    r = client.post(
                        f"{self.base_url}/apply-template",
                        json = {
                            "messages": msgs,
                            "chat_template_kwargs": {**off, "preserve_thinking": preserve},
                        },
                    )
                    # A 4xx is the template's answer, cached; 408 / 429 mean busy.
                    if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                        refused.append(r.status_code)
                    prompt = r.json().get("prompt") if r.status_code == 200 else None
                    return prompt if isinstance(prompt, str) else None

                generated = _render(first, False)
                kept = _render(history, True)
                dropped = _render(history, False)
            if generated is None or kept is None or dropped is None:
                if refused:
                    self._empty_think_preserve_probe = (key, False)
                return False
            result = kept.startswith(generated + answer) and not dropped.startswith(
                generated + answer
            )
        except Exception:  # noqa: BLE001 -- an unanswerable probe keeps the caller's setting
            return False
        self._empty_think_preserve_probe = (key, result)
        return result

    def _request_reasoning_kwargs(
        self,
        enable_thinking: Optional[bool],
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
    ) -> Optional[dict]:
        """preserve_thinking is merged only when the template supports it, independent of reasoning
        style."""
        kwargs: dict = {}
        # Always-on models hardcode <think> tags and ignore enable_thinking / reasoning_effort; skip them.
        if self._supports_reasoning and not self._reasoning_always_on:
            if self._reasoning_style == "enable_thinking_effort":
                # GLM-5.2 style: effort only matters while thinking is on; never coerce off into 'low'.
                thinking_off = enable_thinking is False or reasoning_effort == "none"
                # Clamp unsupported levels to avoid template defaults or errors.
                effort = (
                    None
                    if thinking_off
                    else _clamp_reasoning_effort_to_levels(
                        reasoning_effort, self._reasoning_effort_levels
                    )
                )
                if effort is not None and effort != reasoning_effort:
                    key = (tuple(self._reasoning_effort_levels), reasoning_effort)
                    if key not in _REASONING_EFFORT_CLAMPS_LOGGED:
                        _REASONING_EFFORT_CLAMPS_LOGGED.add(key)
                        logger.info(
                            f"reasoning_effort '{reasoning_effort}' is not offered by this "
                            f"template {list(self._reasoning_effort_levels)}; using '{effort}'"
                        )
                # A named effort implies thinking on, else the template defaults it off.
                if enable_thinking is not None or reasoning_effort == "none" or effort is not None:
                    kwargs["enable_thinking"] = not thinking_off
                if effort is not None:
                    kwargs["reasoning_effort"] = effort
            elif self._reasoning_style == "reasoning_effort":
                _levels = getattr(self, "_reasoning_effort_levels", None) or ()
                _template = _rendered_chat_template(self)
                reasoning_effort = _fit_reasoning_effort_to_template(_template, reasoning_effort)
                if reasoning_effort in ("none", "low", "medium", "high") or (
                    reasoning_effort in _levels
                ):
                    kwargs["reasoning_effort"] = reasoning_effort
                elif reasoning_effort == "minimal":
                    kwargs["reasoning_effort"] = "low"
                elif enable_thinking is not None:
                    kwargs["reasoning_effort"] = _fit_reasoning_effort_to_template(
                        _template,
                        "high" if enable_thinking else "low",
                        thinking_off = not enable_thinking,
                    )
            else:
                if enable_thinking is not None:
                    kwargs["enable_thinking"] = enable_thinking
        if self._supports_preserve_thinking and preserve_thinking is not None:
            kwargs["preserve_thinking"] = preserve_thinking
        _coerce_reasoning_effort(getattr(self, "_architecture", None), kwargs)
        return kwargs or None

    @property
    def supports_tools(self) -> bool:
        # Visual runner frames are dropped by the tool loop, so never route through tools.
        if self._is_diffusion:
            return False
        return self._supports_tools

    @property
    def supports_tool_passthrough(self) -> bool:
        # Passthrough skips the tool loop, so diffusion keeps the real _supports_tools.
        return self._supports_tools

    @property
    def cache_type_kv(self) -> Optional[str]:
        return self._cache_type_kv

    @property
    def tensor_parallel(self) -> bool:
        """Whether --split-mode tensor is active on the loaded server."""
        return self._tensor_parallel

    @property
    def gpu_memory_mode(self) -> str:
        """Active GPU memory strategy: 'auto' or 'manual' (gpu_layers < 0 = Auto/--fit, >= 0 = pinned)."""
        return self._gpu_memory_mode

    @property
    def gpu_layers(self) -> int:
        """Requested --gpu-layers for manual mode (-1 when not manual)."""
        return self._gpu_layers

    @property
    def holds_no_vram(self) -> bool:
        """True for manual zero-offload loads and arch-gated launches masked to CPU by the -1 sentinel, even automatic ones; else a VRAM-less server holds CHAT."""
        if self._arch_gate_forced_cpu:
            return True
        # A diffusion runner reports offload False only when its GPU token was masked: a
        # manual zero split, or an Auto fit that found no room on the card.
        if self._is_diffusion and self._gpu_offload_active is False:
            return True
        return (
            self._gpu_memory_mode == "manual"
            and self._gpu_layers == 0
            and self._gpu_offload_active is False
        )

    @property
    def diffusion_requested_ngl(self) -> Optional[int]:
        """The ask, not what runs, so repeats still dedupe; differs from gpu_layers only if the shim
        refused."""
        return getattr(self, "_diffusion_requested_ngl", None)

    @property
    def n_cpu_moe(self) -> int:
        """MoE expert layers manual mode kept on CPU (--n-cpu-moe); 0 = none."""
        return self._n_cpu_moe

    @property
    def tensor_split(self) -> Optional[List[float]]:
        """Manual returns the requested ratio; auto returns the normalized ratio actually emitted,
        or None."""
        if self._tensor_split is not None:
            return self._tensor_split
        if self._auto_tensor_split_emitted is not None:
            return list(self._auto_tensor_split_emitted)
        return None

    @property
    def gpu_ids(self) -> Optional[List[int]]:
        """User-picked physical GPU indices, or None for automatic selection."""
        return self._gpu_ids

    @property
    def requested_gpu_ids(self) -> Optional[List[int]]:
        """RAW requested GPU pin (before the fit narrowed it), or None for auto.
        gpu_ids echoes the EFFECTIVE pin for /status."""
        return self._requested_gpu_ids

    def matches_gpu_ids(self, gpu_ids: Optional[List[int]]) -> bool:
        """Whether a requested pin is already satisfied by the active runner.

        A regular GGUF load may narrow the requested placement pool to the
        smallest fitting subset. Accept both the original request and the
        effective status-echoed subset so either can round-trip without a
        needless reload. Diffusion records the lowest device of a pick that fits it,
        else the whole pick it split over, so either form matches.
        """
        if self._is_diffusion:
            full = sorted(int(x) for x in gpu_ids) if gpu_ids else None
            # Raw pick accepted too: a forced-CPU runner records no effective pin, so
            # without it re-sending the same selection would reload every time.
            held = ((self._gpu_ids or None), (self._requested_gpu_ids or None))
            return (full[:1] if full else None) in held or full in held

        requested = [int(x) for x in gpu_ids] if gpu_ids else None
        raw = self._requested_gpu_ids or None
        effective = self._gpu_ids or None
        return requested == raw or requested == effective

    def _adopt_widened_pin(self, pin_ids: List[int]) -> None:
        """Adopt the companion-widened pin as effective; else /status and dedupe keep the pre-widen pin and a reload drops the companion card; forced CPU stays None."""
        if self._gpu_ids is not None:
            self._gpu_ids = [int(i) for i in pin_ids]

    def _record_matching_gpu_request(self, gpu_ids: Optional[List[int]]) -> None:
        """Recorded even on a match so a reload does not restore GPUs the user just removed."""
        if self._is_diffusion:
            full = sorted(int(x) for x in gpu_ids) if gpu_ids else None
            # Keep the whole pick when the fit moved off its lowest GPU (a split, or one
            # higher card), or a later pick of that lowest GPU would match and not reload.
            moved = bool(self._gpu_ids) and bool(full) and self._gpu_ids != full[:1]
            self._requested_gpu_ids = full if moved else (full[:1] if full else None)
        else:
            self._requested_gpu_ids = [int(x) for x in gpu_ids] if gpu_ids else None
        if self._last_load_intent is not None:
            self._last_load_intent = replace(
                self._last_load_intent,
                gpu_ids = tuple(self._requested_gpu_ids) if self._requested_gpu_ids else None,
            )

    @property
    def last_load_intent(self) -> Optional[GgufLoadIntent]:
        return self._last_load_intent

    def _runtime_matches_intent(
        self, intent: GgufLoadIntent, effective_extra_args: Optional[list[str]]
    ) -> bool:
        """Whether active runtime settings satisfy one resolved caller intent."""

        if intent.force_reload:
            return False
        if self._requested_n_ctx != int(intent.n_ctx):
            return False
        if not self._is_diffusion and self._requested_n_parallel != max(1, int(intent.n_parallel)):
            return False
        # The same count chosen rather than defaulted is one Auto may not cut.
        if (
            not self._is_diffusion
            and self._slots_cut_for_drafter
            and not intent.n_parallel_from_default
        ):
            return False
        if not self._is_diffusion and (
            self._requested_n_batch != intent.n_batch or self._requested_n_ubatch != intent.n_ubatch
        ):
            return False
        # Requested against requested; a blank value is the llama.cpp default.
        if not self._is_diffusion and (
            _normalized_load_mode(self._requested_load_mode)
            != _normalized_load_mode(intent.load_mode)
            or self._requested_spec_draft_cache_type != intent.spec_draft_cache_type
            or self._requested_ctx_checkpoints != intent.ctx_checkpoints
            or self._requested_cache_ram != intent.cache_ram
        ):
            return False

        # Vision toggles change which files the child opens, so a server launched the other way cannot match; not on diffusion, which records False.
        if not self._is_diffusion and bool(self._disable_vision) != bool(intent.disable_vision):
            return False

        extra_args = list(effective_extra_args) if effective_extra_args is not None else None
        # Compare requests: an inherited LLAMA_ARG_THINK_BUDGET* would never converge.
        if not self._is_diffusion and (
            self._requested_reasoning_budget
            != resolve_reasoning_budget(extra_args, intent.reasoning_budget)
            or self._requested_reasoning_budget_message
            != resolve_reasoning_budget_message(extra_args, intent.reasoning_budget_message)
        ):
            return False
        # Omitted extras inherit the LAUNCHED list; judge on the flag, not value equality, since a deliberate stripped list must not read as inheriting.
        _invoked_extras = self.requested_extra_args if intent.extra_args_inherited else extra_args
        # Requested against requested per axis; the scalar _cache_type_kv misses extras/env.
        if self._requested_cache_types != _planned_main_cache_types(
            intent.cache_type_kv, _invoked_extras
        ):
            return False
        # Virtualised Metal never launches a tensor split, so judge the normalized request, else an extras -sm tensor the pin leaves in place would never match.
        if _metal_device_is_paravirtual():
            if self._tensor_parallel:
                return False
        elif not _tensor_parallel_matches_loaded(
            extra_args, intent.tensor_parallel, self._tensor_parallel
        ):
            # Only a RECORDED drop excuses a live layer server against a tensor request.
            if not (self._arch_gate_dropped_tensor_parallel and not self._tensor_parallel):
                return False
        if (
            self._layer_preserves_tensor_intent
            and not _effective_tensor_parallel(extra_args, intent.tensor_parallel)
            and not intent.preserve_multi_gpu_on_layer
        ):
            return False

        if self._is_diffusion:
            requested_ngl = _diffusion_manual_ngl(intent.gpu_memory_mode, intent.gpu_layers)
            if requested_ngl != self.diffusion_requested_ngl:
                return False
            # An older shim's dropped split matches until it gains --ngl; then the same request applies it.
            if (
                requested_ngl is not None
                and self._gpu_layers != requested_ngl
                and self.diffusion_split_supported()
            ):
                return False
        else:
            if self._swa_full != _swa_full_from_args_or_env(extra_args):
                return False
            if self._gpu_memory_mode != intent.gpu_memory_mode:
                return False
            if intent.gpu_memory_mode == "manual" and (
                self._gpu_layers != intent.gpu_layers
                or (
                    intent.gpu_layers >= 0
                    and (
                        self._n_cpu_moe != intent.n_cpu_moe
                        or (
                            (tuple(self._tensor_split) if self._tensor_split else None)
                            != (intent.tensor_split or None)
                            # A recorded drop (never None == None) means this request, so only other ratios reload.
                            and not (
                                self._arch_gate_dropped_tensor_split is not None
                                and self._arch_gate_dropped_tensor_split
                                == (intent.tensor_split or None)
                            )
                        )
                    )
                )
            ):
                return False
            # Auto mode can carry a user ratio the planner does not describe, so reuse needs an identical ratio.
            if (
                intent.gpu_memory_mode != "manual"
                and self._tensor_parallel
                and self._auto_tensor_split != self._auto_split_fingerprint(intent.tensor_split)
            ):
                return False

        if not self.matches_gpu_ids(list(intent.gpu_ids) if intent.gpu_ids else None):
            return False

        spec_owned_by_extra_args = _extra_args_set_spec_type(extra_args)
        speculative_type = (
            None
            if spec_owned_by_extra_args
            else _canonicalize_spec_mode(intent.speculative_type) or "auto"
        )
        backend_spec = self._requested_spec_mode
        if spec_owned_by_extra_args:
            if backend_spec is not None:
                return False
        elif speculative_type != (backend_spec or "auto"):
            return False
        # Retry only when the repo publishes a drafter, else it relaunches forever.
        if (
            intent.gguf_path is None
            and self._spec_fallback_reason == "drafter_not_found"
            # The fetch did find it and the load dropped it; refetching finds it again.
            and self._mtp_draft_suppressed_path is None
            and speculative_type in ("auto", "mtp", "mtp+ngram", "dspark", "dflash")
            and not (self._spec_drafter_kind == "dspark" and self._dspark_sidecar_absent)
            # DFlash uses _dflash_retry_needed; this arm would relaunch forever on permanent errors.
            and self._spec_drafter_kind != "dflash"
            and not spec_owned_by_extra_args
        ):
            return False
        # Stand-down asks the user to update llama.cpp; the identical request must reload once repaired.
        if (
            speculative_type in ("auto", "mtp", "mtp+ngram", "dspark", "dflash", "ngram")
            and self.spec_binary_fallback_can_retry()
        ):
            return False
        # Under Auto a failed fetch records no fallback reason, so the branch above cannot see it.
        if self._dflash_retry_needed and speculative_type in ("auto", "dflash"):
            return False
        # Reload once after a degraded probe starts answering, else Apply dedupes forever.
        if (
            self._capability_probe_inconclusive
            and not self._is_diffusion
            and not self.probe_server_capabilities().get("mtp_probe_inconclusive")
        ):
            return False

        compared_draft_n_max = self._spec_draft_n_max
        if self._spec_fallback_reason == "runtime_error" and self._last_load_intent is not None:
            # MTP-free recovery keeps the user's prior value in intent; only a changed value should retry MTP.
            compared_draft_n_max = self._last_load_intent.spec_draft_n_max
        draft_depth_matters = (
            self._speculative_type in ("draft-mtp", "draft-dspark", "draft-dflash")
            # Auto's Hybrid Mamba partial-offload stand-down engaged nothing, so the
            # types above cannot see it. The depth priced the rollback copies that
            # made the placement partial, so a change can re-enable MTP.
            or self._spec_fallback_reason in ("runtime_error", "mtp_partial_offload")
        )
        if draft_depth_matters:
            requested_draft_n_max = intent.spec_draft_n_max
            if (requested_draft_n_max is None) != (compared_draft_n_max is None):
                return False
            if requested_draft_n_max is not None and requested_draft_n_max != compared_draft_n_max:
                return False
        if (self._chat_template_override or None) != (intent.chat_template_override or None):
            return False

        if tuple(_invoked_extras or ()) != tuple(self.requested_extra_args or ()):
            return False
        if (intent.gguf_path is not None or intent.compare_mtp_draft) and speculative_type in (
            "auto",
            "mtp",
            "mtp+ngram",
            "dspark",
            "dflash",
        ):
            try:
                # Auto dspark or dflash counts as such; key off _spec_drafter_kind (survives fallback), not _speculative_type, or every Apply reloads a healthy server.
                _compare_dspark = speculative_type == "dspark" or (
                    speculative_type == "auto"
                    and (
                        self._speculative_type == "draft-dspark"
                        or self._spec_drafter_kind == "dspark"
                    )
                )
                _compare_dflash = speculative_type == "dflash" or (
                    speculative_type == "auto"
                    and (
                        self._speculative_type == "draft-dflash"
                        or self._spec_drafter_kind == "dflash"
                    )
                )
                if _compare_dspark:
                    intent_draft = intent.dspark_draft_path
                elif _compare_dflash:
                    intent_draft = intent.dflash_draft_path
                else:
                    intent_draft = intent.mtp_draft_path
                requested_draft = Path(intent_draft).resolve() if intent_draft else None
                # A drafter dropped on purpose counts as launched, else this reloads forever.
                _loaded = self._mtp_draft_path or self._mtp_draft_suppressed_path
                loaded_draft = Path(_loaded).resolve() if _loaded else None
            except OSError:
                return False
            if requested_draft != loaded_draft:
                return False
            # Only the unloadable drop is re-asked, since a paravirtual drop repeats for the same bytes; cache on (path, mtime, size) so only repaired files reparse.
            if (
                self._mtp_draft_suppressed_reason == "unloadable"
                and self._mtp_draft_suppressed_path
                and _mtp_drafter_loads_standalone(self._mtp_draft_suppressed_path)
            ):
                return False
        return True

    @property
    def n_layers(self) -> Optional[int]:
        """Model layer count (GGUF block_count), or None if unknown."""
        return self._n_layers

    @property
    def n_moe_layers(self) -> int:
        """Excludes leading dense layers, since --n-cpu-moe counts from layer 0 and those carry no
        experts."""
        if not self._n_experts or not self._n_layers:
            return 0
        return max(0, self._n_layers - (self._leading_dense_block_count or 0))

    _EMBEDDING_POOLING_TYPES = frozenset({1, 2, 3})  # llama_pooling_type MEAN, CLS, LAST

    @property
    def is_embedding_gguf(self) -> bool:
        """Only MEAN, CLS, LAST pool to one vector; NONE and RANK do not. Without pooling_type, fall back to arch and name hints so --embedding still launches."""
        pooling = getattr(self, "_pooling_type", None)
        if pooling is not None:
            return pooling in self._EMBEDDING_POOLING_TYPES
        path = getattr(self, "_gguf_path", None)
        if not path:
            return False
        from utils.models.gguf_metadata import is_gguf_embedding_model

        return is_gguf_embedding_model(
            path,
            model_identifier = self._model_identifier,
            architecture = self._architecture,
        )

    @staticmethod
    def _resolve_cpu_moe_flag(
        n_cpu_moe: int, n_moe_layers: int, leading_dense: int
    ) -> Optional[int]:
        """Offset past leading dense layers (--n-cpu-moe counts from layer 0); None when nothing to
        offload."""
        if n_cpu_moe <= 0 or n_moe_layers <= 0:
            return None
        return leading_dense + min(n_cpu_moe, n_moe_layers)

    @staticmethod
    def _sanitize_tensor_split(tensor_split: Optional[List[float]]) -> List[float]:
        """Negative and non-finite shares become 0, since inf would pass a > 0 gate; unreadable input returns [] and the call site drops the split."""
        try:
            return [
                x if math.isfinite(x) and x > 0.0 else 0.0 for x in (float(v) for v in tensor_split)
            ]
        except (TypeError, ValueError, OverflowError):
            return []

    @staticmethod
    def _format_tensor_split(values: List[float]) -> str:
        """Fixed-point, not %g (which prints 1000000,1 as 1e+06,1 in a pasted bug-report argv); a ratio that rounds to 0,0 is rescaled by its largest weight."""

        def _render(items):
            return [f"{x:.6f}".rstrip("0").rstrip(".") or "0" for x in items]

        rendered = _render(values)
        if values and all(part == "0" for part in rendered):
            peak = max(values)
            if peak > 0:
                rendered = _render(v / peak for v in values)
        return ",".join(rendered)

    def _mixed_pin_split(
        self,
        gpu_indices: List[int],
        spill_inputs: Optional[dict],
        shared_gpu_ids: Iterable[int],
        extra_args: Optional[Iterable[str]],
        env: Mapping[str, str],
        *,
        layer_min_gpus: int,
        tensor_parallel: bool,
        pin_owns_devices: bool = False,
    ) -> Optional[List[float]]:
        """The discrete-first split for a full-offload mixed Vulkan pin, or None to
        leave llama.cpp's own split (the pre-existing behaviour) wherever its inputs
        cannot be priced or someone else owns the placement."""
        if (
            not shared_gpu_ids
            or spill_inputs is None
            or tensor_parallel
            # A downgraded tensor request keeps every device in use.
            or layer_min_gpus > 1
            # A GPU-resident separate drafter spreads over every device.
            or spill_inputs["separate_draft_on_gpu"]
            or spill_inputs.get("env_mmproj_unsized")
            # A host-resident cache or projector draws on the pool the iGPU reports.
            or spill_inputs.get("host_mmproj_bytes")
            or not _kv_offload_from_args(extra_args, env)
            or _extra_args_have_tensor_split(extra_args, env)
            # argv only: a non-layer inherited LLAMA_ARG_SPLIT_MODE is popped before launch.
            or _extra_args_split_mode(extra_args, {}) not in (None, "layer")
            # Extras follow the generated --fit off; the fitter rejects a multi-device split.
            or any(
                str(a) in ("-fit", "--fit") or str(a).startswith(("-fit=", "--fit="))
                for a in (extra_args or ())
            )
            # RPC devices go ahead of the local ones, so the shares would land on them.
            or (_extra_args_device(extra_args, {"--rpc"}) or "").strip()
            or str(env.get("LLAMA_ARG_RPC", "")).strip()
            # Tensor overrides move bytes after the split is priced.
            or _args_place_tensors_on_cpu(extra_args)
            or _env_places_tensors_on_cpu(env)
            # A projector pinned to another device is charged to device 0 below.
            or (_extra_args_device(extra_args, {"--mmproj-device", "-mmdev"}) or "").strip()
            or str(env.get("MTMD_BACKEND_DEVICE", "")).strip()
            # A surviving device list owns the share order; a gpu_ids pin strips it.
            or (
                not pin_owns_devices
                and (
                    _extra_args_main_device(extra_args) is not None
                    or str(env.get("LLAMA_ARG_DEVICE", "")).strip()
                )
            )
        ):
            return None
        # Adapters may target only some layers, which a total cannot place.
        if _sidecar_adapter_paths(extra_args):
            return None
        # A larger pass-through --parallel (last-wins) grows the cache priced here.
        _np_override = _extra_args_n_parallel(extra_args, env)
        if _np_override is not None and _np_override > int(spill_inputs.get("n_parallel") or 1):
            return None
        layout = self._tensor_spill_layout(spill_inputs.get("model_path"))
        if layout is None or not layout.complete or not layout.blocks:
            return None
        n_blocks = len(layout.blocks)
        kv_total = int(spill_inputs["kv_cache_bytes"] or 0)
        kv_weights = list(spill_inputs.get("kv_layer_weights") or ())
        if len(kv_weights) != n_blocks or sum(kv_weights) <= 0:
            # MLA, hybrid recurrent and an unpriced sliding window are uneven (_kv_layer_weights).
            if kv_total and (
                self._kv_lora_rank is not None
                or self._ssm_inner_size is not None
                or self._sliding_window_pattern
                or self._effective_sliding_window()
            ):
                return None
            kv_weights = [1] * n_blocks
        kv_scale = kv_total / sum(kv_weights)
        layer_bytes = [
            blk.spillable_bytes + blk.resident_bytes + w * kv_scale
            for blk, w in zip(layout.blocks, kv_weights)
        ]
        # Trailing nextn rows count in llama.cpp's n_layer_all + 1; resident only with a draft.
        n_mtp = int(self._nextn_predict_layers or 0) if layout.has_excluded_blocks else 0
        if layout.has_excluded_blocks and n_mtp <= 0:
            return None
        mtp_row = (
            layout.excluded_block_bytes / n_mtp
            if n_mtp and spill_inputs.get("mtp_will_engage")
            else 0.0
        ) + max(kv_weights) * kv_scale  # archs walking n_layer_all cache these rows too
        layer_bytes.extend([mtp_row] * n_mtp)
        # The output layer rides the last row (the last device's run).
        layer_bytes.append(layout.lm_head_bytes + layout.other_resident_bytes)
        reserve = (
            spill_inputs["compute_buffer_flat"]
            + spill_inputs["extra_gpu_bytes"]
            + int(spill_inputs.get("env_mmproj_bytes") or 0)
        )
        # As _planned_tensor_spill; a hybrid run may hold one attention layer over its share.
        per_device = spill_inputs["ctx_compute_per_device"] + spill_inputs["soft_overhead"]
        if 0 < layout.n_attention_layers < n_blocks and len(kv_weights) == n_blocks:
            per_device += kv_total / layout.n_attention_layers
        mib = 1024 * 1024
        return self._discrete_first_split(
            list(gpu_indices),
            spill_inputs["gpu_usable_mib"],
            shared_gpu_ids,
            [x / mib for x in layer_bytes],
            per_device / mib,
            reserve / mib,
            self._PIPELINE_PER_DEVICE_OVERHEAD_MIB,
        )

    @staticmethod
    def _discrete_first_split(
        gpu_indices: List[int],
        usable_mib: dict[int, float],
        shared_gpu_ids: Iterable[int],
        layer_mib: List[float],
        per_device_mib: float = 0.0,
        main_reserve_mib: float = 0.0,
        pipeline_mib: float = 0.0,
    ) -> Optional[List[float]]:
        """Layer-count ``--tensor-split`` shares over ``gpu_indices``: discrete cards first,
        iGPUs (whose "free" is the host pool) take the overflow. llama.cpp cuts contiguous runs
        by COUNT (get_layer_buft_list). None unless the pin mixes both kinds and every run fits."""
        shared = set(shared_gpu_ids)
        discrete = [i for i in gpu_indices if i not in shared]
        igpus = [i for i in gpu_indices if i in shared]
        n = len(layer_mib)
        if not discrete or not igpus or n == 0:
            return None
        room = {
            i: usable_mib.get(i, 0.0)
            - per_device_mib
            - (main_reserve_mib if k == 0 else pipeline_mib)
            for k, i in enumerate(gpu_indices)
        }
        prefix = [0.0]
        for mib in layer_mib:
            prefix.append(prefix[-1] + mib)

        def most_layers(cap: float) -> int:
            k = 0
            while k < n and max(prefix[j + k + 1] - prefix[j] for j in range(n - k)) <= cap:
                k += 1
            return k

        counts: dict[int, float] = {}
        left = n
        for i in sorted(discrete, key = lambda d: room[d], reverse = True):
            counts[i] = min(most_layers(room[i]), left)
            left -= counts[i]
        if left == n:
            return None
        igpu_room = {i: max(0.0, room[i]) for i in igpus}
        igpu_total = sum(igpu_room.values())
        exact = {
            i: (left * igpu_room[i] / igpu_total if igpu_total > 0 else left / len(igpus))
            for i in igpus
        }
        for i in igpus:
            counts[i] = int(exact[i])
        for i in sorted(igpus, key = lambda d: exact[d] - counts[d], reverse = True)[
            : left - sum(counts[i] for i in igpus)
        ]:
            counts[i] += 1
        # The runs llama.cpp will really cut, in device order: each must fit its room.
        start = 0
        for i in gpu_indices:
            end = start + int(counts[i])
            if prefix[end] - prefix[start] > max(0.0, room[i]):
                return None
            start = end
        shares = [float(counts[i]) for i in gpu_indices]
        # Boundaries half a layer early: float rounding in upper_bound cannot add a layer.
        first = next(k for k, v in enumerate(shares) if v > 0)
        if first != len(shares) - 1:
            shares[first] -= 0.5
            shares[-1] += 0.5
        return shares

    @staticmethod
    def _auto_split_fingerprint(tensor_split: Optional[List[float]]) -> Optional[tuple[float, ...]]:
        """Requested vs requested, normalized; auto emits MiB budgets, not the caller ratio, or every repeat reloads the model; an empty ratio is None."""
        if not tensor_split:
            return None
        values = LlamaCppBackend._sanitize_tensor_split(tensor_split)
        total = sum(values)
        if not values or total <= 0:
            return None
        return tuple(round(v / total, 6) for v in values)

    @property
    def layer_preserves_tensor_intent(self) -> bool:
        """True when a downgraded tensor request kept this layer load multi-GPU."""
        return self._layer_preserves_tensor_intent

    @property
    def speculative_type(self) -> Optional[str]:
        return self._speculative_type

    @property
    def spec_drafter_kind(self) -> Optional[str]:
        """Resolved drafter kind (mtp, dspark, dflash), even when Auto chose it; unlike requested_spec_mode (stays auto) or speculative_type (default after fallback)."""
        return self._spec_drafter_kind

    @property
    def requested_spec_mode(self) -> Optional[str]:
        """Canonical UI-facing mode the user requested (see field doc)."""
        return self._requested_spec_mode

    @property
    def spec_draft_n_max(self) -> Optional[int]:
        """User --spec-draft-n-max override active on the load, or None when
        the platform default (6 GPU / 3 CPU) is in effect."""
        return self._spec_draft_n_max
