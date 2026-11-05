# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parsing llama-server stdout: log levels, GPU offload and device lines."""

import re

from typing import (
    Iterable,
    Optional,
)

from .memory_policy import _extra_args_set_any_flag
from core.inference.llama_server_args import _flag_name
from pathlib import Path
from typing import Mapping
import sys
from .draft_args import (
    _extra_args_mtp_draft_source,
    _sidecar_adapter_paths,
)
from .gguf_cache import (
    _gguf_set_bytes,
)

_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR = (
    "This Mac's Metal device is virtualised, where offloaded layers can return "
    "corrupt output, and the installed unsloth_zoo diffusion shim has no --ngl, "
    "so the zero-layer CPU pin cannot be applied and the runner would use Metal "
    "anyway. Run 'unsloth studio update' to get a shim with --ngl, or set "
    "UNSLOTH_ALLOW_PARAVIRTUAL_METAL=1 to keep the GPU on a VM you have verified."
)

_VULKAN_DIFFUSION_GPU_IDS_ERROR = (
    "GPU selection (gpu_ids) is not supported for a DiffusionGemma "
    "GGUF on a Vulkan llama.cpp build: the diffusion runner selects "
    "its device by CUDA physical index, which has no defined mapping "
    "to ggml Vulkan device ordinals. Omit gpu_ids to use the default "
    "device."
)


# llama-server may serve HTTP 200 on CPU when a GPU backend fails to init; classify the log.
# Priority: offload counts, then GPU model buffer lines, then device table (disconfirm only).
_GPU_OFFLOAD_MARKERS = (
    "CUDA",
    "ROCm",
    "ROCM",
    "HIP",
    "Metal",
    "Vulkan",
    "OpenCL",
    "SYCL",
    "MUSA",
    "CANN",
)
_OFFLOADED_LAYERS_RE = re.compile(
    r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to\s+gpu", re.IGNORECASE
)
# llama.cpp #23021 moved libllama INFO (offload counts, device table, buffer sizes) to trace (4).
_LLAMA_TRACE_VERBOSITY = 4
_TRACE_VERBOSITY_HELP_RE = re.compile(r"\b4:\s*trace\b", re.IGNORECASE)
_LOG_VERBOSITY_FLAGS = frozenset(
    {"-lv", "--verbosity", "--log-verbosity", "-v", "--verbose", "--log-verbose", "--log-disable"}
)
# Past this, the active llama-server log keeps only warnings and errors.
_LLAMA_LOG_FULL_BYTES = 64 * 1024 * 1024
# llama.cpp's prefix (stamp optional, colours allowed); a mid-line "x W " is record text.
_ANSI = r"(?:\x1b\[[0-9;]*m)*"
_LOG_LEVEL_TOKEN_RE = re.compile(rf"^{_ANSI}(?:\d+\.\d+\.\d+\.\d+{_ANSI} )?{_ANSI}([A-Z]) ")
# The startup head is never trimmed: its load lines are read for the server's whole life.
_STDOUT_TRIM_AT = 20000
_STDOUT_TAIL_KEEP = 5000
_STDOUT_HEAD_MAX = 5000


# Left in startup output when the ownership gate refuses, so it is not reported as a bad GGUF.
_PORT_OWNER_REFUSED_MARKER = "llama-server port ownership check refused the listener"


def _socket_owner_is_always_named() -> bool:
    """Windows' connection tables carry every socket's owning pid, so an unnamed one there is a
    psutil quirk rather than another user's listener (#15022)."""
    return sys.platform == "win32"


# Bind addresses whose listener accepts the health probe's 127.0.0.1 connection.
_LOOPBACK_REACHABLE_BIND_IPS = frozenset(
    {"127.0.0.1", "0.0.0.0", "::", "::ffff:127.0.0.1", "::ffff:0.0.0.0"}
)


def _llama_line_is_warning_or_error(line: str, line_lower: str) -> bool:
    level = _LOG_LEVEL_TOKEN_RE.match(line)
    if level is not None:
        return level.group(1) in "WE"
    return "error" in line_lower or "fail" in line_lower or "warn" in line_lower


def _trace_verbosity_args(caps: dict, extra_args: Optional[Iterable[str]], environ) -> list[str]:
    """``--verbosity 4`` when the build hides load lines at its default and the user chose none."""
    if not caps.get("supports_trace_verbosity"):
        return []
    if _extra_args_set_any_flag(extra_args, _LOG_VERBOSITY_FLAGS):
        return []
    if "LLAMA_ARG_LOG_VERBOSITY" in environ:
        return []
    return ["--verbosity", str(_LLAMA_TRACE_VERBOSITY)]


_GPU_MODEL_BUFFER_RE = re.compile(
    r"\b(?:CUDA|ROCm|ROCM|HIP|SYCL)(\d+)\s+model buffer size\s*=\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s+MiB\b",
    re.IGNORECASE,
)
_DEVICE_ROW_RE = re.compile(
    # MTL: ggml-metal names its devices MTL0, MTL1 (GGML_METAL_NAME), never "Metal".
    r"-\s*(CUDA|ROCm|ROCM|HIP|Metal|MTL|Vulkan|SYCL|OpenCL|MUSA|CANN|CPU)\w*\s*:",
    re.IGNORECASE,
)
_GPU_DEVICE_PREFIXES = (
    "cuda",
    "rocm",
    "hip",
    "metal",
    "mtl",
    "vulkan",
    "sycl",
    "opencl",
    "musa",
    "cann",
)


def _extra_weight_file_bytes(argv, env: Optional[Mapping[str, str]] = None) -> int:
    """GGUF sets of the projector, local drafter and adapters a llama-server launch loads. arg.cpp reads
    the ``LLAMA_ARG_*`` env first and lets argv override it, so ``env`` is the one the child got."""
    tokens = [str(token) for token in argv] if isinstance(argv, (list, tuple)) else []
    env_map = env or {}
    projector: Optional[str] = None
    for index, token in enumerate(tokens):
        if _flag_name(token) not in ("--mmproj", "-mm"):
            continue
        _, eq, value = token.partition("=")
        if eq:
            projector = value
        elif index + 1 < len(tokens):
            projector = tokens[index + 1]
    if projector is None:
        projector = (env_map.get("LLAMA_ARG_MMPROJ") or "").strip() or None
    drafter, remote = _extra_args_mtp_draft_source(tokens, env_map)
    paths = [path for path in (projector, None if remote else drafter) if path]
    paths += _sidecar_adapter_paths(tokens)
    return sum(_gguf_set_bytes(Path(path)) for path in dict.fromkeys(paths))


def parse_weight_placement(
    lines: "list[str]", model_bytes: Optional[int] = None
) -> "Optional[dict]":
    """Where llama.cpp put the main model's weights, from its startup log.

    ``gpu_mib`` / ``host_mib`` sum the main model's buffer block, the first allocated one (dry runs log 0.00 MiB).
    "offloaded 31/31" hides an expert spill, so ``overflow_layers`` comes from --fit's final per-device
    lines; the last device's count includes the output layer, which spills nothing (a 3090 gemma-4-26B
    load printing "8 overflowing" holds exactly seven layers' worth in CUDA_Host). None without --fit.
    A ``*_Mapped`` buffer logs its mmap span, first to last host tensor, which covers GPU tensors in
    between when experts spill; ``model_bytes`` (the main GGUF set) caps host at the file minus the GPU side."""
    blocks: list[list] = []
    current: Optional[list] = None
    # A build without GPU offload logs no "offloaded N/M layers" line, only the buffers: each run of them is a block.
    unmarked = False
    moe = False
    full_phase = False
    # device -> (layers, overflow or None for a dense-only line), in the order of the final verdict
    verdict: dict[str, tuple[int, Optional[int]]] = {}
    dense_only_order: list[str] = []
    extra_layer_on: set[int] = set()
    expert_cache_mib: Optional[float] = None
    for line in lines:
        if _FIT_DENSE_ONLY_MARK in line:
            moe = True
            continue
        if _FIT_FULL_LAYERS_MARK in line:
            full_phase = True
            verdict = {}
            continue
        cache = _FIT_MOE_CACHE_RE.search(line)
        if cache is not None:
            expert_cache_mib = float(cache.group("mib"))
            continue
        extra = _FIT_EXTRA_LAYER_RE.search(line)
        if extra is not None and full_phase:
            extra_layer_on.add(int(extra.group("index")))
            continue
        fit = _FIT_DEVICE_RE.search(line)
        if fit is not None:
            device = fit.group("device")
            overflow = fit.group("overflow")
            verdict[device] = (
                int(fit.group("layers")),
                int(overflow) if overflow is not None else None,
            )
            if not full_phase and device not in dense_only_order:
                dense_only_order.append(device)
            continue
        counts = _OFFLOADED_LAYERS_RE.search(line)
        if counts is not None:
            current, unmarked = [int(counts.group(2)), 0.0, 0.0, False], False
            blocks.append(current)
            continue
        buffer = _MODEL_BUFFER_RE.search(line)
        if buffer is None:
            if unmarked:
                current, unmarked = None, False
            continue
        if current is None:
            current, unmarked = [0, 0.0, 0.0, False], True
            blocks.append(current)
        mib = float(buffer.group(2))
        if _host_buffer(buffer.group(1)):
            current[2] += mib
            current[3] = current[3] or (mib > 0 and buffer.group(1).upper().endswith("_MAPPED"))
        else:
            current[1] += mib
    # A no-alloc dry run logs the same block with every buffer at 0.00 MiB.
    loaded = [block for block in blocks if block[1] + block[2] > 0]
    if not loaded:
        return None
    # llama-server loads the main model before any drafter, which can have MORE layers (Gemma 4 E2B: 35 vs 26B A4B: 30).
    main = loaded[0]
    host_mib = main[2]
    if main[3] and main[1] > 0 and model_bytes:
        host_mib = min(host_mib, max(0.0, model_bytes / (1024 * 1024) - main[1]))

    overflow: Optional[int] = None
    if verdict:
        devices = list(verdict)
        overflow = 0
        for index, device in enumerate(devices):
            layers, counted = verdict[device]
            if counted is None:
                # Dense-only fill (filled back to front, so the first device it printed is the last one). A dense
                # model's layers are full. The last device keeps one layer full and its output slot spills nothing.
                if not moe:
                    continue
                last = bool(dense_only_order) and device == dense_only_order[0]
                counted = max(0, layers - 2) if last else layers
            elif index < len(devices) - 1:
                if index in extra_layer_on and counted > 0:
                    counted -= 1
            elif counted > 0:
                counted -= 1
            overflow += counted
    return {
        "gpu_mib": round(main[1], 1),
        "host_mib": round(host_mib, 1),
        "overflow_layers": overflow,
        "all_experts_in_ram": expert_cache_mib is not None,
        "expert_cache_mib": expert_cache_mib,
    }


def _host_buffer(name: str) -> bool:
    """CPU* buffers, AMX and a GPU backend's pinned host buffer (CUDA_Host) live in system memory."""
    upper = name.upper()
    return "CPU" in upper or upper.endswith("_HOST") or upper == "AMX"


# --moe-cache-mib auto: every routed expert stays in host RAM and a cache of the busiest ones goes on the GPU. --fit
# returns before any per-device line on this path.
_FIT_MOE_CACHE_RE = re.compile(
    r"moe cache auto: all routed experts in host RAM, cache (?P<mib>\d+) MiB"
)


# One extra partial layer squeezed onto device i; on any device but the last it overflows to the NEXT GPU, not RAM.
_FIT_EXTRA_LAYER_RE = re.compile(
    r"set ngl_per_device\[(?P<index>\d+)\]\.\(n_layer, n_part, overflow_type\)=\(\d+, \d+, (?:UP|GATE|ATTN)\)"
)


# From here the per-device lines carry an explicit "(N overflowing)" count.
_FIT_FULL_LAYERS_MARK = "converting dense-only layers to full layers"


# The MoE-only fill: printed for a model with experts, whose GPU layers then hold only their dense weights.
_FIT_DENSE_ONLY_MARK = "filling dense-only layers back-to-front"


# --fit's verdict per device, e.g. "- CUDA0 (NVIDIA GeForce RTX 3090): 31 layers ( 8 overflowing), 22815 MiB used".
# An overflowing layer keeps its MoE experts (or part of a layer) out of that device's memory.
_FIT_DEVICE_RE = re.compile(
    r"common_params_fit_impl:\s+-\s+(?P<device>[A-Za-z]+\d+)\s+\(.*?\):\s+(?P<layers>\d+)\s+layers?"
    r"\s*(?:\(\s*(?P<overflow>\d+)\s+overflowing\s*\))?\s*,\s*\d+\s+MiB used"
)


# Any backend's "model buffer size" line (CUDA0, CPU_Mapped, RPC0[host:port], ...).
_MODEL_BUFFER_RE = re.compile(
    r"\b([A-Za-z][^\s=]*)\s+model buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s+MiB\b"
)


def parse_gpu_offload_counts(lines: "list[str]") -> "Optional[tuple[int, int]]":
    """``(offloaded, total)`` for the largest-M line, or None; ties keep the first
    line since llama.cpp logs the main model before its drafter."""
    max_total = -1
    offloaded_at_max = 0
    for line in lines:
        match = _OFFLOADED_LAYERS_RE.search(line)
        if not match:
            continue
        offloaded, total = int(match.group(1)), int(match.group(2))
        if total > max_total:
            max_total, offloaded_at_max = total, offloaded
    if max_total <= 0:
        return None
    return offloaded_at_max, max_total


def llama_saw_gpu_device(lines: "list[str]") -> Optional[bool]:
    """Whether llama.cpp's device table (rows after the header) lists a GPU, or None
    with no table. All-CPU means the GPU backend failed to load."""
    after_header = False
    saw_device_row = False
    saw_gpu_device = False
    for line in lines:
        if "device_info:" in line:
            after_header = True
            continue
        if not after_header:
            continue
        match = _DEVICE_ROW_RE.search(line)
        if not match:
            continue
        saw_device_row = True
        if match.group(1).lower().startswith(_GPU_DEVICE_PREFIXES):
            saw_gpu_device = True
    if not saw_device_row:
        return None
    return saw_gpu_device


def classify_gpu_offload_lines(lines: "list[str]") -> Optional[bool]:
    """True if the model landed on a GPU, False if it stayed on CPU despite GPU
    intent, None when the log has no usable signal."""
    # Decide on the largest-M offload line: a drafter on GPU must not mask a CPU main model.
    max_total = -1
    offloaded_at_max = 0
    for line in lines:
        match = _OFFLOADED_LAYERS_RE.search(line)
        if not match:
            continue
        offloaded, total = int(match.group(1)), int(match.group(2))
        if total > max_total or (total == max_total and offloaded > offloaded_at_max):
            max_total, offloaded_at_max = total, offloaded
    if max_total >= 0:
        return offloaded_at_max > 0

    # _Host buffers are CPU-pinned; buffer lines are authoritative over the device table.
    saw_model_buffer = False
    for line in lines:
        if "model buffer size" not in line:
            continue
        saw_model_buffer = True
        if "_Host" not in line and any(m in line for m in _GPU_OFFLOAD_MARKERS):
            return True
    if saw_model_buffer:
        return False

    # device_info lists available devices, not where the model loaded: disconfirm only.
    if llama_saw_gpu_device(lines) is False:
        return False
    return None
