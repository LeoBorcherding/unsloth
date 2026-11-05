# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The composed LlamaCppBackend class."""

import atexit
import subprocess
import threading
import time

from pathlib import Path
from typing import (
    List,
    Optional,
)

from .backend_binary import _BinaryDiscoveryMixin
from .backend_capabilities import _CapabilityProbeMixin
from .backend_diffusion import _DiffusionRunnerMixin
from .backend_download import _DownloadMixin
from .backend_generation import _GenerationMixin
from .backend_gpu_devices import _GpuDevicesMixin
from .backend_gpu_fit import _GpuFitMixin
from .backend_gpu_memory import _GpuMemoryMixin
from .backend_health import _HealthMixin
from .backend_kv_cache import _KvCacheEstimateMixin
from .backend_launch_retries import _LaunchRetryMixin
from .backend_load import _LoadMixin
from .backend_preflight import _PreflightMixin
from .backend_process_io import _ProcessIoMixin
from .backend_properties import _PropertiesMixin
from .backend_shutdown import _ShutdownMixin
from .backend_slots import _SlotsMixin
from .backend_spill import _TensorSpillMixin
from .backend_start_failures import _StartFailureMixin
from .backend_tokens import _TokenCountingMixin
from .backend_tool_loop import _ToolLoopMixin
from .core_types import (
    GgufLoadIntent,
    _CpuFallbackRuntime,
)


class LlamaCppBackend(
    _PropertiesMixin,
    _BinaryDiscoveryMixin,
    _CapabilityProbeMixin,
    _GpuDevicesMixin,
    _GpuMemoryMixin,
    _GpuFitMixin,
    _KvCacheEstimateMixin,
    _DownloadMixin,
    _ProcessIoMixin,
    _DiffusionRunnerMixin,
    _PreflightMixin,
    _StartFailureMixin,
    _LaunchRetryMixin,
    _LoadMixin,
    _ShutdownMixin,
    _TensorSpillMixin,
    _SlotsMixin,
    _HealthMixin,
    _GenerationMixin,
    _ToolLoopMixin,
    _TokenCountingMixin,
):
    """Manages a llama-server subprocess for GGUF model inference.

    Lifecycle:
        1. load_model(): start llama-server with the GGUF file
        2. generate_chat_completion(): proxy to /v1/chat/completions, stream back
        3. unload_model(): terminate the subprocess
    """

    _spawn_lock = threading.Lock()

    # Held across a whole teardown so a lifecycle cannot reopen mid-kill; taken before _spawn_lock.
    _teardown_lock = threading.Lock()

    def __init__(self, *, manages_processes: bool = True):
        """manages_processes=False is an inert probe: no /proc orphan reaping, no atexit handler held."""
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        # Load-in-flight memory notice returned on LoadResponse; _begin_load_warnings clears it per load.
        self._last_load_warning: Optional[str] = None
        # Same, for a quant fallback: the download fills the pair, the launch publishes it.
        self._variant_fallback_warning: Optional[str] = None
        # The route's Hub notice for the running model, so /status matches the load response.
        self.hub_access_warning: Optional[str] = None
        self._pending_variant_fallback: Optional[tuple[str, str]] = None
        # Set per launch by _record_carveout_advice; None on nearly every load.
        self._last_carveout_advice: Optional[dict] = None
        self._model_identifier: Optional[str] = None
        self._gguf_path: Optional[str] = None
        # A local link can be retargeted while _gguf_path stays the same.
        self._gguf_load_identity: Optional[tuple] = None
        self._hf_repo: Optional[str] = None
        # Separate MTP drafter for this model; a drafter appearing beside the weights forces a reload.
        self._mtp_draft_path: Optional[str] = None
        # Without this record every repeat Apply tears down a healthy server.
        self._mtp_draft_suppressed_path: Optional[str] = None
        # Only 'unloadable' judges the contents, so only it reloads on repair.
        self._mtp_draft_suppressed_reason: Optional[str] = None
        # binary_no_mtp / binary_outdated: a newer prebuilt helps; runtime_error: maybe not.
        self._spec_fallback_reason: Optional[str] = None

        # Projector recovery at startup: cpu_offload keeps vision; others are a deliberate text-only retry.
        self._mmproj_fallback_reason: Optional[str] = None
        # An inconclusive probe reports every capability absent; tracked so a reload can retry.
        self._capability_probe_inconclusive: bool = False
        self._spec_drafter_kind: Optional[str] = None
        self._dspark_sidecar_absent: bool = False
        self._dflash_sidecar_absent: bool = False
        # Sidecar fetch failed transiently, so one more Apply is worth it.
        self._dflash_retry_needed: bool = False
        # Binary this load ran; studio update replaces it in place, so a stand-down can spot a real update.
        self._launch_binary_revision: tuple = ()
        # Lets the settings route see a pending reload before is_active is true.
        self._binary_revision_pending: Optional[tuple] = None
        # Set after an auto-Vulkan crash recovers with all devices disabled.
        self._cpu_fallback_reason: Optional[str] = None
        self._cpu_fallback_runtime: Optional[_CpuFallbackRuntime] = None
        self._pending_cpu_fallback_cleanups: list = []
        self._hf_variant: Optional[str] = None
        self._is_vision: bool = False
        # Tracked so a reload flipping the toggle is not satisfied by the live server.
        self._disable_vision: bool = False
        # Only a model WITH a projector can have had vision switched off by the user.
        self._vision_disabled_by_user: bool = False
        # Block-diffusion model, served by the diffusion runner; set from the GGUF architecture at load.
        self._architecture: Optional[str] = None
        self._gguf_type: Optional[str] = None
        self._is_diffusion: bool = False
        # Whole header walked? Separates "declares no architecture" from "unreadable".
        self._gguf_header_parsed: bool = False
        self._gguf_split_index: Optional[int] = None
        self._diffusion_visual_bin: Optional[str] = None
        self._healthy = False
        self._load_rss_hwm = (None, 0)  # (pid, peak VmRSS) for load_progress
        self._stats_logger = None  # vLLM-style engine-stats poller, set on load
        # Set by _classify_gpu_offload after _wait_for_health.
        self._gpu_offload_active: Optional[bool] = None
        # (offloaded, total) layers llama.cpp reported for this load, when it said.
        self._gpu_offload_layers: Optional[tuple[int, int]] = None
        # parse_weight_placement's result for this load.
        self._weight_placement: Optional[dict] = None
        # Whether the user's own extras pinned the layer split for this load.
        self._offload_overridden: bool = False
        # Studio saw a GPU for this load but llama.cpp's device table did not.
        self._gpu_backend_unavailable: bool = False
        # Diffusion only: the split the running child was asked for.
        self._diffusion_requested_ngl: Optional[int] = None
        self._context_length: Optional[int] = None
        self._effective_context_length: Optional[int] = None
        self._max_context_length: Optional[int] = None
        # Ceiling a fit priced inside the GPU / Metal budget; None for an anchor or floor.
        self._vram_fit_context_length: Optional[int] = None
        self._effective_parallel_slots: int = 1
        # --parallel the last load asked for, before any fit-time reduction.
        self._requested_n_parallel: int = 1
        self._slots_cut_for_drafter: bool = False
        # Budget the running child was sized against; None before any load, so the
        # settings route reports a stale budget instead of nagging on every save.
        self._vram_fraction_launched: Optional[float] = None
        # Budget a load has committed but not published, from planning to Popen, as _memory_pending_launch.
        self._vram_fraction_pending: Optional[float] = None
        # --batch-size / --ubatch-size the last load asked for; none = defaults or extras / env
        self._requested_n_batch: Optional[int] = None
        self._requested_n_ubatch: Optional[int] = None
        # The expert-spill micro-batch raise: the pair before it, and the argv tokens.
        self._moe_spill_batch_restore: Optional[tuple[Optional[int], Optional[int]]] = None
        self._moe_spill_batch_tokens: Optional[tuple[list[str], list[str]]] = None
        self._moe_cache_flags: List[str] = []
        # The tuning group the last load asked for; none = defaults, or left to
        # extras / env. What was REQUESTED, not what ran: Model Memory can replace
        # the load mode, and Windows full-offload tuning owns the two cache knobs.
        self._requested_load_mode: Optional[str] = None
        self._requested_spec_draft_cache_type: Optional[str] = None
        self._requested_ctx_checkpoints: Optional[int] = None
        self._requested_cache_ram: Optional[int] = None
        self._idle_slot_clearing_active: bool = False
        self._chat_template: Optional[str] = None
        self._markup_tokens: list = []
        self._markup_profile = None
        self._chat_template_override: Optional[str] = None
        self._supports_reasoning: bool = False
        self._reasoning_always_on: bool = False
        self._reasoning_budget: int = -1
        self._reasoning_budget_message: str = ""
        # What the load asked for, before the environment. See _runtime_matches_intent.
        self._requested_reasoning_budget: int = -1
        self._requested_reasoning_budget_message: str = ""
        self._reasoning_style: str = "enable_thinking"
        self._reasoning_effort_levels: list = []
        self._supports_preserve_thinking: bool = False
        self._preserve_thinking_default: bool = False
        self._supports_tools: bool = False
        self._cache_type_kv: Optional[str] = None
        # Whether --split-mode tensor was applied on the active load.
        self._tensor_parallel: bool = False
        # GPU memory strategy applied on the active load ("auto"/"manual").
        self._gpu_memory_mode: str = "auto"
        # Manual-mode load options (echoed back so the UI round-trips them).
        self._gpu_layers: int = -1
        # MoE expert layers to keep on CPU (--n-cpu-moe); 0 = none.
        self._n_cpu_moe: int = 0
        # Relative share per GPU for --tensor-split, in GPU order; None lets llama.cpp split by free VRAM.
        self._tensor_split: Optional[List[float]] = None
        # Recorded so _runtime_matches_intent reloads on a changed ratio.
        self._auto_tensor_split: Optional[tuple[float, ...]] = None
        # Split actually emitted to llama-server in auto mode, for /status; manual mode uses _tensor_split.
        self._auto_tensor_split_emitted: Optional[tuple[float, ...]] = None
        # Tensor-spill tokens added to argv, stripped by _without_subsequence on retry; empty if abstained.
        self._spill_plan_flags: List[str] = []
        # Ratio the arch gate dropped from argv; the live server already runs this request normalized.
        self._arch_gate_dropped_tensor_split: Optional[tuple[float, ...]] = None
        # The gate drops --split-mode tensor on forced CPU or one survivor; the request still asks for it.
        self._arch_gate_dropped_tensor_parallel: bool = False
        # Gate masked every device, so AUTOMATIC became zero-VRAM; holds_no_vram cannot demand manual mode.
        self._arch_gate_forced_cpu: bool = False
        # User-picked physical GPU indices (None = automatic selection).
        self._gpu_ids: Optional[List[int]] = None
        # Automatic placement leaves _gpu_ids unset, so do not infer this from status state.
        self._child_gpu_physical_ids: Optional[tuple[int, ...]] = None
        # RAW pin, so a narrowed [0, 1] re-sent as [0, 1] still dedupes.
        self._requested_gpu_ids: Optional[List[int]] = None
        # Layer load kept multi-GPU only to honor a downgraded tensor request, so a later tensor-off reloads.
        self._layer_preserves_tensor_intent: bool = False
        self._reasoning_default: bool = True
        self._speculative_type: Optional[str] = None
        # UI mode as requested, not the resolved flag set (auto may resolve to draft-mtp).
        self._requested_spec_mode: Optional[str] = None
        # User --spec-draft-n-max override (None = platform default).
        self._spec_draft_n_max: Optional[int] = None
        # KV-cache estimation fields (populated by _read_gguf_metadata)
        self._n_layers: Optional[int] = None
        # Leading dense-layer count offsets --n-cpu-moe, which counts from layer 0.
        self._n_experts: Optional[int] = None
        self._leading_dense_block_count: Optional[int] = None
        self._n_kv_heads: Optional[int] = None
        self._n_kv_heads_by_layer: Optional[list[int]] = None
        self._n_heads: Optional[int] = None
        self._embedding_length: Optional[int] = None
        # llama_pooling_type from the GGUF; present and non-zero marks an embedding model.
        self._pooling_type: Optional[int] = None
        # For the compute-graph buffer estimate; feed_forward_length is the widest layer if stored per layer.
        self._feed_forward_length: Optional[int] = None
        self._feed_forward_length_by_layer: Optional[list[int]] = None
        self._expert_used_count: Optional[int] = None
        self._expert_feed_forward_length: Optional[int] = None
        self._expert_shared_feed_forward_length: Optional[int] = None
        self._expert_shared_count: Optional[int] = None
        # Nemotron 3.5: routed experts run in this latent width, not n_embd.
        self._moe_latent_size: Optional[int] = None
        # Gemma 3n / Gemma 4 per-layer input embedding width.
        self._embedding_length_per_layer_input: Optional[int] = None
        self._vocab_size: Optional[int] = None
        # Set from reserves_micro_batch_outputs() by the callers that know the binary.
        self._reserves_micro_batch_outputs: bool = False
        self._resumes_thoughts: bool = True
        # Architecture-aware KV fields for 5-path estimation
        self._kv_key_length: Optional[int] = None
        self._kv_value_length: Optional[int] = None
        self._sliding_window: Optional[int] = None
        self._sliding_window_pattern: Optional[list[bool]] = None
        self._full_attention_interval: Optional[int] = None
        self._kv_lora_rank: Optional[int] = None
        self._key_length_mla: Optional[int] = None
        self._kv_key_length_swa: Optional[int] = None
        self._kv_value_length_swa: Optional[int] = None
        self._ssm_inner_size: Optional[int] = None
        self._ssm_state_size: Optional[int] = None
        self._ssm_group_count: Optional[int] = None
        self._ssm_conv_kernel: Optional[int] = None
        self._kda_head_dim: Optional[int] = None
        # Last N layers reuse earlier KV (Gemma 3n / Gemma 4 shared_kv_layers).
        self._shared_kv_layers: Optional[int] = None
        # MTP head count (llama.cpp #22673); >0 enables --spec-type draft-mtp.
        self._nextn_predict_layers: Optional[int] = None
        self._lock = threading.Lock()
        # Serialises loads so two llama-server processes never coexist; RLock for MTP recovery.
        self._serial_load_lock = threading.RLock()
        # Serialises respawns so many generations on a killed server trigger one reload (_respawn_if_dead).
        self._respawn_lock = threading.Lock()
        # load_model clears _cancel_event, so a racing respawn needs this to survive the clear.
        self._unload_epoch = 0
        # Set during binary swaps; load_model() rejects fast so no server starts from a half-swapped binary.
        self._llama_update_in_progress = False
        # Preserved across unload so /unload+/load Apply can inherit them.
        self._extra_args: Optional[List[str]] = None
        # Extras as REQUESTED, before launch-time rewrites, for duplicate-load checks.
        self._requested_extra_args: Optional[List[str]] = None
        self._extra_args_source: Optional[tuple[str, Optional[str]]] = None
        self._requested_n_ctx: int = 0
        # Last healthy caller intent for crash recovery. Memory-only and never logged.
        self._last_load_intent: Optional[GgufLoadIntent] = None
        self._mtp_runtime_fallback_lock = threading.Lock()
        self._mtp_runtime_fallback_in_progress = False
        # Background watchdog so an MTP crash recovers even when no request
        # observes it (direct proxy endpoints, or nothing in flight).
        self._mtp_watchdog_thread: Optional[threading.Thread] = None
        self._mtp_watchdog_stop = threading.Event()
        # True when the launch actually runs MTP (Unsloth- or user/env-driven);
        # gates the watchdog and recovery so pass-through MTP is covered.
        self._mtp_runtime_fallback_active = False
        # (load key, count) of Metal OOM retirements, and the key that stopped respawning.
        self._metal_oom_strikes = None
        self._metal_oom_terminal_key = None
        # (load key, count) of mid-session GPU OOM deaths, and the child that logged one.
        self._gpu_oom_strikes = None
        self._gpu_oom_process = None
        self._stdout_lines: list[str] = []
        self._stdout_thread: Optional[threading.Thread] = None
        # Wakes the health probe on subprocess readiness/exit, not left to the 500 ms fallback poll.
        self._health_probe_event = threading.Event()
        # llama-server tee log (see _drain_stdout / _kill_process).
        self._llama_log_fh = None
        self._llama_log_path: Optional[Path] = None
        self._cancel_event = threading.Event()
        self._api_key: Optional[str] = None
        self._api_key_file: Optional[Path] = None
        self._slot_save_dir: Optional[str] = None
        self._slot_save_binary: Optional[tuple[str, int]] = None
        # (gguf_identity, launch_fingerprint) at load lets a slot save detect swapped model files.
        self._slot_loaded_identity: Optional[tuple] = None
        self._prompt_cache_disabled: bool = False
        self._swa_full: bool = False
        self._kv_cache_unified: bool = False
        # (mlock, reserves_ram) the process was launched with; None when this policy does not govern it.
        self._memory_state: Optional[tuple[bool, bool]] = None
        # mmap and dio share the pair above, so without this they are indistinguishable.
        self._memory_direct_io: Optional[bool] = None
        # Same chain one toggle apart; against _memory_direct_io it asks if a reload would change the loader.
        self._memory_dio_applicable: bool = False
        # The managed tokens, so a rung that gives up the offload can take them back out.
        self._memory_dio_flags: list[str] = []
        # Compared against by the strip, so it can never remove a hand-typed pair.
        self._memory_dio_user_tokens: list[str] = []
        self._memory_policy_active: bool = False
        # Still marks a launch with managed flags removed, so a withdrawing rung can spot an unmanaged child.
        self._memory_policy_extras_touched: bool = False
        # Fully offloaded discrete-GPU launches skip mlock on purpose.
        self._memory_mlock_applicable: bool = True
        # --load-mode none tokens FIT emitted, removed by paths replacing placement; empty if user chose one.
        self._fit_load_mode_flags: list[str] = []
        self._lazy_mode_flags: list[str] = []
        self._apple_lazy_flags: list[str] = []
        # Mixed-pin --tensor-split tokens; the --fit on retry strips them (fit.cpp aborts on a set split).
        self._mixed_split_flags: list[str] = []
        # The pair a launch in flight is committed to, None when none is. ONE attribute:
        # a separate marker and snapshot read out of step answered a save with
        # reload_required=false about a child already committed to the pre-save flags.
        self._memory_pending_launch: Optional[tuple[bool, bool]] = None
        # Not on GgufLoadIntent: a provenance field there would force duplicate reloads.
        self._loaded_by_user_action: bool = False
        self._n_ubatch: int = self._DEFAULT_N_UBATCH
        self._flash_attn_enabled: bool = True
        self._effective_cache_types: tuple[str, str] = ("f16", "f16")
        # The pair this load ASKED for, before any launch-time rewrite.
        self._requested_cache_types: tuple[str, str] = ("f16", "f16")
        # Total KV context over all slots; _effective_context_length becomes the per-slot limit after /props.
        self._kv_cache_context_total: Optional[int] = None
        # True once a probe has completed; cleared on transient failure.
        self._is_audio: bool = False
        self._audio_type: Optional[str] = None
        self._audio_probed: bool = False
        self._codec_failure = threading.local()
        # Audio INPUT capability (distinct from _is_audio, which is TTS output).
        self._has_audio_input: bool = False
        # Needs mmproj, build and ffmpeg to align; the GGUF alone cannot tell.
        self._has_video_input: bool = False
        self._video_needs_ffmpeg: bool = False
        self._video_ffmpeg_dir: Optional[str] = None
        self._mmproj_has_audio: bool = False  # clip.has_audio_encoder, set at load
        # clip.projector_type set at load; the family whose image-token ceiling KV admission sizes images against.
        self._mmproj_projector_type: Optional[str] = None
        # clip.has_vision_encoder, set at load; True keeps an undeclared projector capable.
        self._mmproj_accepts_image: bool = True
        # Set in _kill_process (monotonic); load_model reads it to decide whether to wait for VRAM reclaim.
        self._last_kill_monotonic: float = 0.0
        self._planned_vram_mib: dict[int, int] = {}
        self._pending_plan_mib: dict[int, int] = {}

        if manages_processes:
            _reaped = self._kill_orphaned_servers()
            if _reaped:
                # Reaped VRAM frees lazily, so the first load waits before ranking GPUs.
                self._last_kill_monotonic = time.monotonic()
            atexit.register(self._cleanup)
