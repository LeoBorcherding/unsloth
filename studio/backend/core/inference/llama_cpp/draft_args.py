# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Draft model arguments and device selection."""

import os
import re

from typing import (
    Collection,
    Iterable,
    List,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _flag_name,
    _GPU_LAYER_FLAGS,
    parse_split_mode_override,
    split_policy_starves_devices,
)

from .launch_flags import (
    _accumulated_spec_types,
    _args_place_tensors_on_cpu,
    _env_fixes_gpu_layers,
    _env_places_tensors_on_cpu,
)
from .memory_policy import _extra_args_set_any_flag
from core.inference.llama_server_args import _last_flag_value
from core.inference.llama_server_args import fit_is_effectively_on
from core.inference.llama_server_args import parse_ctx_override
from typing import Union
import math
from .launch_flags import (
    _CUDA_LIKE_DEVICE_RE,
    _LLAMA_ARG_TRUE_VALUES,
    _distinct_device_count,
    _is_positive_int,
)
from .memory_policy import (
    _LLAMA_FIT_MIN_CTX,
)


def _extra_args_requests_separate_draft(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Non-MTP draft modes load a separate model; accumulated, as an inherited one cannot be cleared."""
    return bool(
        _accumulated_spec_types(extra_args, env)
        & {"draft-simple", "draft-eagle3", "draft-dspark", "draft-dflash"}
    )


def _extra_args_spec_draft_n_max(extra_args: Optional[Iterable[str]]) -> Optional[int]:
    """Draft depth from extras (``--spec-draft-n-max`` or legacy ``--draft-max``), else None."""
    if not extra_args:
        return None
    args = [str(a) for a in extra_args]
    found: Optional[int] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag not in ("--spec-draft-n-max", "--draft-max"):
            continue
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        try:
            found = int(value)
        except (TypeError, ValueError):
            continue
    return found


def _extra_args_mtp_draft_path(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """An HF repo draft is not a local file, so it cannot be sized and falls back to the flat reserve."""
    return _extra_args_mtp_draft_source(extra_args, env)[0]


def _drop_sidecar_margins(
    cmd: List[str],
    integrity_flags: List[str],
    rebuilt_integrity: List[str],
    override_pairs: Iterable[List[str]],
    *,
    user_args: Iterable[str] = (),
) -> List[str]:
    """``cmd`` without Studio's sidecar margins; a user's own pairs survive, so it is idempotent."""
    out = list(cmd)
    user = [str(a) for a in user_args]

    def _count(tokens: List[str], pair: List[str]) -> int:
        return sum(tokens[i : i + len(pair)] == pair for i in range(len(tokens) - len(pair) + 1))

    def _studio_owns(pair: List[str]) -> bool:
        return bool(pair) and _count(out, pair) > _count(user, pair)

    if "--fit-target" in integrity_flags:
        old_at = integrity_flags.index("--fit-target")
        old_margin = integrity_flags[old_at : old_at + 2]
        new_margin: List[str] = []
        if "--fit-target" in rebuilt_integrity:
            new_at = rebuilt_integrity.index("--fit-target")
            new_margin = rebuilt_integrity[new_at : new_at + 2]
        at = _subsequence_index(out, old_margin, 0)
        if out[at : at + 2] == old_margin and _studio_owns(old_margin):
            out[at : at + 2] = new_margin
    for pair in reversed(list(override_pairs)):
        if not _studio_owns(pair):
            continue
        for i in range(len(out) - len(pair), -1, -1):
            if out[i : i + len(pair)] == pair:
                del out[i : i + len(pair)]
                break
    return out


def _sidecar_reserve_margin_flags(
    child_args: Iterable[str],
    user_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]],
    base_margin_mib: float,
    reserve_mib: int,
    *,
    per_device: Optional[list[int]] = None,
    base_by_device: Optional[list[float]] = None,
) -> list[str]:
    """``--fit-target`` keeping an unsized sidecar's reserve free on top of the user's margin, read
    off the child's argv + env."""
    args = [str(a) for a in child_args]
    if not reserve_mib or not _extra_args_mtp_draft_path(args, env = env):
        return []
    if not fit_is_effectively_on(args, env):
        return []
    # A scalar --fit-target is broadcast to every device, so a multi-device launch needs a
    # per-device list.
    devices = _extra_args_main_device(args) or (os.environ if env is None else env).get(
        "LLAMA_ARG_DEVICE"
    )
    if (
        not per_device
        and devices
        and _distinct_device_count(devices) > 1
        and not _split_mode_confines_to_one_device(args, env)
    ):
        return []
    if per_device:
        bases = (
            base_by_device
            if base_by_device and len(base_by_device) == len(per_device)
            else [base_margin_mib] * len(per_device)
        )
        margins = _fit_margins_by_device(user_args, env, 0.0, len(per_device))
        want_list = ",".join(
            str(math.ceil(max(b, m) + r)) for b, m, r in zip(bases, margins, per_device)
        )
        if _last_flag_value(args, {"-fitt", "--fit-target"}) == want_list:
            return []
        return ["--fit-target", want_list]
    # Studio's per-card margin, not the host-wide scalar a lowered budget sizes by the largest card.
    if base_by_device and len(set(base_by_device)) == 1:
        base_margin_mib = base_by_device[0]
    want = math.ceil(
        max(base_margin_mib, _single_device_fit_margin(user_args or (), env) or 0.0) + reserve_mib
    )
    have = _single_device_fit_margin(args, None)
    if have is not None and have >= want:
        return []
    return ["--fit-target", str(want)]


def _fit_margins_by_device(
    user_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]],
    base_margin_mib: float,
    count: int,
) -> list[float]:
    """Each card's user --fit-target margin as common/arg.cpp applies it; unset cards keep
    ``base_margin_mib``."""
    margins = [float(base_margin_mib)] * count
    for raw in _fit_target_values(user_args or (), env):
        entries = [e.strip() for e in str(raw).replace("/", ",").split(",") if e.strip()]
        try:
            values = [float(e) for e in entries]
        except ValueError:
            continue
        if len(values) == 1:
            margins = values * count
        else:
            margins[: len(values)] = values[:count]
    return margins


def _fit_target_values(user_args: Iterable[str], env: Optional[Mapping[str, str]]) -> list[str]:
    """Every --fit-target value llama.cpp applies, env first then argv order."""
    found = []
    if env and env.get("LLAMA_ARG_FIT_TARGET"):
        found.append(str(env["LLAMA_ARG_FIT_TARGET"]))
    tokens = [str(a) for a in user_args]
    for i, raw in enumerate(tokens):
        _, eq, inline = raw.partition("=")
        if _flag_name(raw) in ("-fitt", "--fit-target"):
            value = inline if eq else (tokens[i + 1] if i + 1 < len(tokens) else None)
            if value is not None:
                found.append(value)
    return found


def _single_device_fit_margin(
    args: Iterable[str], env: Optional[Mapping[str, str]]
) -> Optional[float]:
    """The margin the one launched device keeps: the FIRST entry of a ``--fit-target`` list.

    ``fit_target_margin_in`` takes the largest entry because it prices every device it
    credits; a single-device launch only ever uses device 0's entry (common/arg.cpp), so a
    larger later entry must not be added under the sidecar reserve."""
    raw = _last_flag_value(args, {"-fitt", "--fit-target"})
    if raw is None and env:
        raw = env.get("LLAMA_ARG_FIT_TARGET")
    if raw is None:
        return None
    first = str(raw).replace("/", ",").split(",")[0].strip()
    try:
        return float(first)
    except ValueError:
        return None


_HF_DRAFT_FLAGS = frozenset({"--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft"})
_LOCAL_DRAFT_FLAGS = frozenset({"--model-draft", "--spec-draft-model", "-md"})


def _draft_backend_sampling_off(
    spec_flags: list[str], caps: Mapping[str, object], vocab_size: Optional[int], vulkan: bool
) -> list[str]:
    if not vulkan or "--spec-type" not in spec_flags:
        return []
    i = spec_flags.index("--spec-type")
    spec_types = str(spec_flags[i + 1]).split(",") if i + 1 < len(spec_flags) else []
    if all(t == "none" or t.startswith("ngram") for t in spec_types):
        return []
    if vocab_size and vocab_size % _VULKAN_ALIGNED_VOCAB_MULTIPLE == 0:
        return []
    if _NO_DRAFT_BACKEND_SAMPLING_FLAG not in (caps.get("flags") or {}):
        return []
    return [_NO_DRAFT_BACKEND_SAMPLING_FLAG]


_NO_DRAFT_BACKEND_SAMPLING_FLAG = "--no-spec-draft-backend-sampling"


# Vulkan TOP_K cannot bind a misaligned view, and backend draft sampling reads logits row r at
# r * n_vocab * 4 bytes (#14987). n_vocab % 64 == 0 keeps rows 256-byte aligned, the spec maximum.
_VULKAN_ALIGNED_VOCAB_MULTIPLE = 64


def _extra_args_mtp_draft_source(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[Optional[str], bool]:
    """Last-wins across remote and local flags, so only the winning flag says whether it is a repo id."""
    args = [str(a) for a in extra_args] if extra_args else []
    found: Optional[str] = None
    found_remote = False
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag not in _HF_DRAFT_FLAGS and flag not in _LOCAL_DRAFT_FLAGS:
            continue
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        if value and not value.startswith("-"):
            found = value
            found_remote = flag in _HF_DRAFT_FLAGS
    if found is not None:
        return found, found_remote
    e = os.environ if env is None else env
    # The last two are pre-b8955 spellings.
    for var, remote in (
        ("LLAMA_ARG_SPEC_DRAFT_MODEL", False),
        ("LLAMA_ARG_SPEC_DRAFT_HF_REPO", True),
        ("LLAMA_ARG_MODEL_DRAFT", False),
        ("LLAMA_ARG_HFD_REPO", True),
    ):
        value = e.get(var)
        if value:
            return value, remote
    return None, False


def _normalized_load_mode(value: Optional[str]) -> Optional[str]:
    """auto is the default, so it maps to None; else the dedupe reloads a server already running it."""
    if value is None:
        return None
    mode = str(value).strip().lower()
    return None if mode in {"", "auto"} else mode


_BREAKDOWN_DEVICE_ROW_RE = re.compile(
    r"\|\s*-\s*(?P<name>[^|(]+?)\s*\([^|]*\|\s*\d+\s*=\s*\d+\s*\+\s*\(\s*\d+\s*=\s*(?P<model>\d+)\s*\+"
)
_BREAKDOWN_HOST_ROW_RE = re.compile(
    r"\|\s*-\s*(?P<name>[^|]+?)\s*\|\s*\d+\s*=\s*(?P<model>\d+)\s*\+"
)


_METAL_FIT_PROBE_TIMEOUT_S = 120


def _parse_metal_memory_breakdown(output: str) -> Optional[tuple[int, int, int]]:
    """Requires one Metal device row; host model MiB sums the model column of every host row."""
    rows: list[str] = []
    for line in output.splitlines():
        if "common_memory_breakdown_print" not in line:
            if rows:
                break
            continue
        if "memory breakdown" in line:
            if rows:
                break
            continue
        rows.append(line)
    metal: list[int] = []
    host = 0
    host_rows = 0
    for row in rows:
        device = _BREAKDOWN_DEVICE_ROW_RE.search(row)
        if device:
            if not device.group("name").strip().startswith("MTL"):
                return None
            metal.append(int(device.group("model")))
            continue
        hosted = _BREAKDOWN_HOST_ROW_RE.search(row)
        if hosted:
            host += int(hosted.group("model"))
            host_rows += 1
            continue
        return None
    if len(metal) != 1 or host_rows == 0:
        return None
    return metal[0], host, len(rows)


# KV cache types llama.cpp can map, module level so the draft-cache block and MTP reserve share it.
_VALID_KV_CACHE_TYPES: frozenset[str] = frozenset(
    {"f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl", "f32"}
)


def _extra_args_draft_cache_types(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[Optional[str], Optional[str]]:
    """Draft KV cache types (k_type, v_type), each from extras else the
    LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K/_V env, else None (f16). K and V are
    independent: a one-sided override must not apply to both."""
    args = [str(a) for a in extra_args] if extra_args else []
    k_flags = {"--cache-type-k-draft", "--spec-draft-type-k", "-ctkd"}
    v_flags = {"--cache-type-v-draft", "--spec-draft-type-v", "-ctvd"}
    k_type: Optional[str] = None
    v_type: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag not in k_flags and flag not in v_flags:
            continue
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        if not value or value.startswith("-"):
            continue
        if flag in k_flags:
            k_type = value
        else:
            v_type = value
    e = os.environ if env is None else env
    if k_type is None:
        k_type = e.get("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K") or None
    if v_type is None:
        v_type = e.get("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V") or None
    return k_type, v_type


_DRAFT_GPU_LAYER_FLAGS = frozenset(
    {"--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}
)


def _extra_args_draft_gpu_layers(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Last-wins over argv, falling back to LLAMA_ARG_N_GPU_LAYERS_DRAFT; None when neither is set."""
    args = [str(a) for a in extra_args] if extra_args else []
    last_ngl: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        if flag in _DRAFT_GPU_LAYER_FLAGS:
            last_ngl = value
    if last_ngl is None:
        last_ngl = (os.environ if env is None else env).get("LLAMA_ARG_N_GPU_LAYERS_DRAFT")
    return last_ngl


def _draft_is_split_across_host(
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    *,
    n_draft_layers: Optional[int] = None,
) -> bool:
    """A positive -ngld splits the drafter unless above its block count; equal leaves a block on host."""
    raw = _extra_args_draft_gpu_layers(extra_args, env)
    if raw is None:
        return False
    try:
        count = int(str(raw).strip())
    except (TypeError, ValueError):
        # std::stoi would throw and the child never starts: nothing to misprice.
        return False
    if count <= 0:
        return False
    return not (n_draft_layers and count > n_draft_layers)


def _draft_n_ubatch(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[int]:
    """A positive --spec-draft-n-ubatch (env twin overridden by argv), else None."""
    found = (os.environ if env is None else env).get("LLAMA_ARG_SPEC_DRAFT_N_UBATCH")
    tokens = [str(a) for a in args] if args else []
    for i, raw in enumerate(tokens):
        _, eq, inline = raw.partition("=")
        if _flag_name(raw) == "--spec-draft-n-ubatch":
            found = inline if eq else (tokens[i + 1] if i + 1 < len(tokens) else None)
    return int(str(found).strip()) if _is_positive_int(found) else None


def _sidecar_child_ctx(
    args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]],
    native_ctx: Optional[int],
    n_parallel: Optional[int],
    kv_unified: bool,
) -> int:
    """The context the child allocates: last positive -c, else LLAMA_ARG_CTX_SIZE, else the native
    length."""
    ctx = parse_ctx_override(args)
    if ctx is None:
        try:
            ctx = int(
                str((os.environ if env is None else env).get("LLAMA_ARG_CTX_SIZE", "")).strip()
            )
        except (TypeError, ValueError):
            ctx = None
    if ctx and ctx > 0:
        return ctx
    # Without -c, --kv-unified-per-slot N sizes the pool to n_parallel * N
    # (tools/server/server.cpp).
    source_env = os.environ if env is None else env
    tokens = [str(a) for a in args] if args else []
    full_window = str(source_env.get("LLAMA_ARG_CTX_SIZE", "")).strip() == "0"
    if str(source_env.get("LLAMA_ARG_FIT_CTX", "")).strip():
        full_window = False
    per_slot = source_env.get("LLAMA_ARG_KV_UNIFIED_PER_SLOT")
    for i, raw in enumerate(tokens):
        flag = _flag_name(raw)
        if flag not in ("-c", "--ctx-size", "-fitc", "--fit-ctx", "--kv-unified-per-slot"):
            continue
        _, eq, inline = raw.partition("=")
        value = inline if eq else (tokens[i + 1] if i + 1 < len(tokens) else "")
        if flag == "--kv-unified-per-slot":
            per_slot = value
        elif flag in ("-fitc", "--fit-ctx"):
            full_window = False
        elif str(value).strip() == "0":
            full_window = True
    if _is_positive_int(per_slot) and not full_window:
        return int(str(per_slot).strip()) * max(1, int(n_parallel or 1))
    bound = int(native_ctx or 0) or _LLAMA_FIT_MIN_CTX
    if not kv_unified:
        bound *= max(1, int(n_parallel or 1))
    return bound


def _draft_card_share_unknown(
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    *,
    n_draft_layers: Optional[int] = None,
) -> bool:
    """True when the drafter is not wholly on the target's card, so its whole file is not
    the share to reserve there: a positive -ngld below its depth (split with the host), or
    a draft tensor override, which llama.cpp applies per weight while picking buffers (the
    same flags the fit-voiding list treats as placement changes), or the draft CPU-MoE
    switches, which llama.cpp turns into draft tensor overrides."""
    if _extra_args_set_any_flag(extra_args, _DRAFT_OVERRIDE_TENSOR_FLAGS | _DRAFT_CPU_MOE_FLAGS):
        return True
    source_env = os.environ if env is None else env
    if source_env.get("LLAMA_ARG_SPEC_DRAFT_CPU_MOE") in _LLAMA_ARG_TRUE_VALUES:
        return True
    # Each positive count appends draft CPU overrides, so any positive one is a placement.
    if _is_positive_int(source_env.get("LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE")):
        return True
    args = [str(a) for a in extra_args] if extra_args else []
    for i, raw in enumerate(args):
        _, eq, inline = raw.partition("=")
        if _flag_name(raw) in _DRAFT_N_CPU_MOE_FLAGS and _is_positive_int(
            inline if eq else (args[i + 1] if i + 1 < len(args) else None)
        ):
            return True
    return _draft_is_split_across_host(extra_args, env, n_draft_layers = n_draft_layers)


_DRAFT_N_CPU_MOE_FLAGS = frozenset(
    {"--spec-draft-n-cpu-moe", "--spec-draft-ncmoe", "-ncmoed", "--n-cpu-moe-draft"}
)


_DRAFT_CPU_MOE_FLAGS = frozenset({"--spec-draft-cpu-moe", "-cmoed", "--cpu-moe-draft"})


_DRAFT_OVERRIDE_TENSOR_FLAGS = frozenset(
    {"-otd", "--override-tensor-draft", "--spec-draft-override-tensor"}
)


_SIDECAR_ADAPTER_FLAGS = (
    "--lora",
    "--lora-scaled",
    "--control-vector",
    "--control-vector-scaled",
)


def _sidecar_adapter_paths(extra_args: Optional[Iterable[str]]) -> list[str]:
    """Resident LoRA and control-vector files; legacy two-token -scaled read, malformed entries skipped."""
    args = [str(a) for a in extra_args] if extra_args else []
    paths: list[str] = []
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        if flag not in _SIDECAR_ADAPTER_FLAGS:
            continue
        _, eq, inline = raw.partition("=")
        operand = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        priced = 0
        for entry in (piece for piece in operand.split(",") if piece):
            if not flag.endswith("-scaled"):
                paths.append(entry)
                priced += 1
                continue
            # Upstream demands exactly 2 parts, so a bare or drive-lettered path throws.
            head, colon, tail = entry.partition(":")
            if not (colon and head) or ":" in tail:
                continue
            try:
                float(tail)
            except ValueError:
                continue
            paths.append(head)
            priced += 1
        if priced or not operand or eq or not flag.endswith("-scaled"):
            continue
        # Legacy "FNAME SCALE": the scale is the token AFTER the operand.
        scale = args[i + 2] if i + 2 < len(args) else ""
        try:
            float(scale.strip())
        except ValueError:
            continue
        paths.append(operand)
    return paths


def _sidecar_adapter_bytes(extra_args: Optional[Iterable[str]]) -> Optional[int]:
    """Adapters are resident whatever --load-mode says, so file size is charged whole as a ceiling."""
    total = 0
    for path in _sidecar_adapter_paths(extra_args):
        try:
            total += max(0, os.stat(path).st_size)
        except OSError:
            return None
    return total


def _extra_args_draft_offloaded_to_cpu(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Embedded MTP heads follow the main -ngl; only a zero count or cpu/none device moves the drafter."""
    last_ngl = _extra_args_draft_gpu_layers(extra_args, env)
    if last_ngl is not None:
        try:
            if int(last_ngl) == 0:
                return True
        except (TypeError, ValueError):
            pass
    last_dev = _extra_args_draft_device(extra_args)
    if last_dev is not None:
        devs = [d.strip().lower() for d in last_dev.split(",") if d.strip()]
        if devs and all(d in ("cpu", "none") for d in devs):
            return True
    return False


def _margin_mib_for(margin: Union[float, Mapping[int, float], None], idx: int) -> float:
    """The ``--fit-target`` margin one card keeps: a single broadcast value, or that
    card's entry in a per-device mapping (the largest entry for a card it omits)."""
    if isinstance(margin, Mapping):
        if not margin:
            return 0.0
        return max(0.0, float(margin.get(int(idx), max(margin.values()))))
    return max(0.0, float(margin or 0.0))


def _extra_args_device(
    extra_args: Optional[Iterable[str]], flags: Collection[str]
) -> Optional[str]:
    """Return the last value for any device flag in ``flags``."""
    args = [str(a) for a in extra_args] if extra_args else []
    value: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag in flags:
            value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
    return value


def _extra_args_main_device(extra_args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last explicit main-device value, if any."""
    return _extra_args_device(extra_args, {"--device", "-dev"})


def _without_subsequence(tokens: List[str], run: List[str]) -> List[str]:
    """Removes only the first occurrence, so an identical user flag elsewhere in the command survives."""
    if not run:
        return list(tokens)
    for i in range(len(tokens) - len(run) + 1):
        if tokens[i : i + len(run)] == run:
            return [*tokens[:i], *tokens[i + len(run) :]]
    return list(tokens)


def _count_subsequence(tokens: List[str], run: List[str]) -> int:
    """How many non-overlapping contiguous occurrences of ``run`` are in ``tokens``."""
    if not run:
        return 0
    total = 0
    i = 0
    while i <= len(tokens) - len(run):
        if tokens[i : i + len(run)] == run:
            total += 1
            i += len(run)
        else:
            i += 1
    return total


def _subsequence_index(tokens: List[str], run: List[str], hint: int) -> int:
    """Searches back from the hint first: narrowing argv earlier can only shift the block left."""
    span = len(tokens) - len(run)
    if not run or span < 0:
        return max(0, min(hint, len(tokens)))
    for i in range(min(hint, span), -1, -1):
        if tokens[i : i + len(run)] == run:
            return i
    for i in range(min(hint, span) + 1, span + 1):
        if tokens[i : i + len(run)] == run:
            return i
    return max(0, min(hint, len(tokens)))


_CPU_DEVICE_VALUES = frozenset({"cpu", "none"})


def _pass_through_overrides_placement(
    extra_args: Optional[Iterable[str]],
    source_env: Mapping[str, str],
    credited: int,
    *,
    is_vulkan_backend: bool = False,
    gpu_indices: Optional[Iterable[int]] = None,
) -> bool:
    """Checks the child's real extras and env against what the fit priced; see _fit_derived_load_mode."""
    # A --device naming fewer cards than the fit charged leaves credit for cards never opened.
    _dev_value = _extra_args_main_device(extra_args)
    if _dev_value is None:
        _dev_value = source_env.get("LLAMA_ARG_DEVICE")
    _dev_names = [d.strip() for d in str(_dev_value or "").split(",") if d.strip()]
    _credited = credited
    gpu_indices = list(gpu_indices) if gpu_indices is not None else None
    # DISTINCT names: llama.cpp appends every entry without deduplicating and
    # keeps the list verbatim, so "CUDA0,CUDA0" is ONE physical pool listed twice,
    # and counting it as two leaves the credit paying for a card that is not there.
    _device_narrows = bool(_dev_names) and len(set(_dev_names)) < _credited
    # Vulkan names are comparable, so a surviving --device must match exactly, not just by count.
    _vulkan_pin_replaced = bool(
        is_vulkan_backend
        and gpu_indices is not None
        and _dev_names
        and {name.strip().lower() for name in _dev_names} != {f"vulkan{idx}" for idx in gpu_indices}
    )
    return bool(
        _extra_args_set_any_flag(extra_args, _GPU_LAYER_FLAGS)
        # Env twin: the fitter never lowers an inherited LLAMA_ARG_N_GPU_LAYERS, so a prefix loads from RAM.
        or _env_fixes_gpu_layers(source_env)
        # With the env twin: only an explicit gpu_ids pin clears LLAMA_ARG_DEVICE,
        # so an inherited "none" puts the whole load in host RAM.
        or _device_selection_is_cpu(extra_args, source_env)
        or _device_narrows
        or _vulkan_pin_replaced
        # Pass-through and last-wins: --split-mode none, or a --tensor-split that
        # zeroes a device, leaves the pooled credit paying for cards held empty.
        or split_policy_starves_devices(extra_args, _credited, source_env)
        or _args_place_tensors_on_cpu(extra_args)
        or _env_places_tensors_on_cpu(source_env)
    )


def _device_selection_is_cpu(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """argv beats the env twin; an unreadable value answers True, like every other unknown in the gate."""
    value = _extra_args_main_device(extra_args)
    if value is None and env:
        value = env.get("LLAMA_ARG_DEVICE")
    if value is None:
        return False
    devices = [d.strip().lower() for d in str(value).split(",") if d.strip()]
    return not devices or all(d in _CPU_DEVICE_VALUES for d in devices)


def _split_mode_confines_to_one_device(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether the effective split mode confines all weights to one device."""
    try:
        mode = parse_split_mode_override(args)
    except ValueError:
        return False
    if mode is None and env:
        mode = env.get("LLAMA_ARG_SPLIT_MODE")
    return str(mode or "").strip().lower() == "none"


def _extra_args_draft_device(extra_args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last explicit draft-device value, if any."""
    return _extra_args_device(extra_args, {"--spec-draft-device", "-devd", "--device-draft"})


_GPU_DEVICE_TOKEN_RE = re.compile(r"(CUDA|ROCm)(\d+)$", re.IGNORECASE)
# One group per companion: llama.cpp is last-wins within a group, so only its last flag counts.
_COMPANION_DEVICE_FLAG_GROUPS = (
    frozenset({"--mmproj-device", "-mmdev"}),
    frozenset({"--spec-draft-device", "-devd", "--device-draft"}),
)


def _widen_pin_ids_for_companion_devices(
    cmd: List[str],
    pin_ids: list[int],
    inherited_ids: Optional[list[int]],
    *,
    allowed_ids: Optional[Iterable[int]] = None,
) -> tuple[list[int], str]:
    """Companion CUDA<n>/ROCm<n> tokens use unpinned numbering; a main --device keeps -ngl -1 local."""
    if not pin_ids:
        return list(pin_ids), ""
    main_ids = [int(i) for i in pin_ids]
    allowed = None if allowed_ids is None else {int(i) for i in allowed_ids}
    # A main --device past the pinned positions would resolve to a widened companion card.
    for token in str(_extra_args_main_device(cmd) or "").split(","):
        match = _GPU_DEVICE_TOKEN_RE.match(token.strip())
        if match and int(match.group(2)) >= len(main_ids):
            return list(main_ids), ""
    # (value index, prefix before "=" or None, [(token, physical id or None)])
    sites: list[tuple[int, Optional[str], list[tuple[str, Optional[int]]]]] = []
    prefix_word = None
    last_of_group: dict[int, int] = {}
    for i, raw in enumerate(cmd):
        for group, flags in enumerate(_COMPANION_DEVICE_FLAG_GROUPS):
            if _flag_name(str(raw)) in flags:
                last_of_group[group] = i
    for i in sorted(last_of_group.values()):
        raw = str(cmd[i])
        head, eq, inline = raw.partition("=")
        if eq:
            at, value, lead = i, inline, head + "="
        elif i + 1 < len(cmd):
            at, value, lead = i + 1, str(cmd[i + 1]), None
        else:
            continue
        tokens: list[tuple[str, Optional[int]]] = []
        for token in value.split(","):
            match = _GPU_DEVICE_TOKEN_RE.match(token.strip())
            physical = None
            if match:
                prefix_word = prefix_word or match.group(1)
                n = int(match.group(2))
                if inherited_ids is None:
                    physical = n
                elif n < len(inherited_ids):
                    physical = int(inherited_ids[n])
                if allowed is not None and physical not in main_ids and physical not in allowed:
                    physical = None
            tokens.append((token, physical))
        if any(physical is not None for _, physical in tokens):
            sites.append((at, lead, tokens))
    if not sites:
        return list(main_ids), ""
    widened = list(main_ids)
    for _, _, tokens in sites:
        for _, physical in tokens:
            if physical is not None and physical not in widened:
                widened.append(physical)
    changed = False
    for at, lead, tokens in sites:
        rewritten = []
        for token, physical in tokens:
            if physical is None:
                rewritten.append(token)
                continue
            match = _GPU_DEVICE_TOKEN_RE.match(token.strip())
            rewritten.append(f"{match.group(1)}{widened.index(physical)}")
        value = ",".join(rewritten)
        new = f"{lead}{value}" if lead is not None else value
        if new != cmd[at]:
            cmd[at] = new
            changed = True
    if len(widened) > len(main_ids) and _extra_args_main_device(cmd) is None:
        # Pinned cards lead the mask, so the main model's devices are the first positions.
        cmd.extend(["--device", ",".join(f"{prefix_word}{k}" for k in range(len(main_ids)))])
        changed = True
    if not changed and widened == main_ids:
        return list(main_ids), ""
    return widened, f"{main_ids} -> {widened}"


def _extra_args_draft_device_pin(extra_args: Optional[Iterable[str]]) -> Optional[str]:
    """Return a GPU draft-device override; cpu/none do not conflict with a pin."""
    last_dev = _extra_args_draft_device(extra_args)
    if last_dev is None:
        return None
    devs = [d.strip() for d in last_dev.split(",") if d.strip()]
    if not devs or all(d.lower() in ("cpu", "none") for d in devs):
        return None
    return last_dev


def _spawn_draft_on_target(
    run_cmd: List[str],
    spawn_devices: Optional[str],
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether the spawned argv's CUDA / ROCm draft pin names the target's device."""
    pin = _extra_args_draft_device_pin(run_cmd)
    if pin is not None:
        distinct = {d.strip() for d in pin.split(",") if d.strip()}
        pin = distinct.pop() if len(distinct) == 1 else pin
    if pin is None or not _CUDA_LIKE_DEVICE_RE.match(pin.strip()):
        return True
    if spawn_devices:
        named = [d.strip().lower() for d in spawn_devices.split(",") if d.strip()]
        if _split_mode_confines_to_one_device(run_cmd, env):
            # Only devices[main_gpu] is kept (main_gpu 0 by default).
            main = LlamaCppBackend._effective_main_gpu(run_cmd, env)
            main = 0 if main is None else main
            return 0 <= main < len(named) and pin.strip().lower() == named[main]
        return set(named) == {d.strip().lower() for d in pin.split(",") if d.strip()}
    return int(re.sub(r"\D", "", pin)) == (LlamaCppBackend._effective_main_gpu(run_cmd, env) or 0)
