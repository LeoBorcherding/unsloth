# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batch, ubatch, parallel slot and split-mode arguments."""

import math
import os
import re
import struct

from typing import (
    Iterable,
    Mapping,
    Optional,
    Sequence,
)
from core.inference.llama_server_args import (
    _flag_name,
    _SPLIT_MODE_FLAGS,
    extra_args_disable_mmproj,
    extra_args_image_max_tokens,
    extra_args_mmproj_auto,
)

from .draft_args import _extra_args_device
from .launch_flags import (
    _DEFAULT_LLAMA_N_BATCH,
    _DEFAULT_LLAMA_N_UBATCH,
    _GEMMA4V_CAUSAL_TEXT_N_EMBD,
    _MMPROJ_NON_CAUSAL_IMAGE_TOKENS,
    _MMPROJ_UNKNOWN_UBATCH,
    _args_place_tensors_on_cpu,
    _env_places_tensors_on_cpu,
)
from .log import logger
from .memory_policy import _TENSOR_SPLIT_FLAGS
from core.inference.llama_server_args import parse_ctx_override
from .draft_args import (
    _draft_n_ubatch,
)


def _emitted_n_batch(n_batch: Optional[int], n_parallel: int) -> Optional[int]:
    """Budgets must use this raised value; llama-server aborts below max(slots, 2)."""
    if n_batch is None:
        return None
    return max(int(n_batch), max(2, int(n_parallel or 1)))


def _with_parallel_slots(
    cmd: Sequence[str], slots: int, *, n_batch: Optional[int], supports_kv_unified: bool
) -> list[str]:
    """``cmd`` relaunched at ``slots``: Unsloth's own --parallel, its batch floor, and the
    --kv-unified a multi-slot launch adds. Each is emitted before the extras, so the
    first occurrence is the one to rewrite and a hand-typed flag still last-wins."""
    out = [str(a) for a in cmd]
    at = out.index("--parallel")
    out[at + 1] = str(slots)
    if n_batch is not None and "--batch-size" in out:
        b = out.index("--batch-size")
        out[b + 1] = str(_emitted_n_batch(n_batch, slots))
    if slots > 1 and supports_kv_unified and "--kv-unified" not in out and "-kvu" not in out:
        out[at + 2 : at + 2] = ["--kv-unified"]
    return out


def _extra_args_split_mode(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Last-wins ``-sm`` / ``--split-mode`` value, lowercased, or None."""
    source_env = os.environ if env is None else env
    found: Optional[str] = None
    raw = source_env.get("LLAMA_ARG_SPLIT_MODE")
    if raw and str(raw).strip():
        found = str(raw).strip().lower()
    args = [str(a) for a in extra_args] if extra_args else []
    for i, arg in enumerate(args):
        _, _, inline = arg.partition("=")
        if _flag_name(arg) not in _SPLIT_MODE_FLAGS:
            continue
        value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
        if value.strip():
            found = value.strip().lower()
    return found


def _extra_args_tensor_split(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[list[float]]:
    """Only llama.cpp's [,/]+ delimit; std::stof stops at the first bad character, so 3;1 is [3, 0]."""
    source_env = os.environ if env is None else env
    raw: Optional[str] = None
    env_value = source_env.get("LLAMA_ARG_TENSOR_SPLIT")
    if env_value and str(env_value).strip():
        raw = str(env_value).strip()
    args = [str(a) for a in extra_args] if extra_args else []
    for i, arg in enumerate(args):
        if _flag_name(arg) not in _TENSOR_SPLIT_FLAGS:
            continue
        _, _, inline = arg.partition("=")
        value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
        if value.strip():
            raw = value.strip()
    if raw is None:
        return None
    try:
        parts = [
            struct.unpack("=f", struct.pack("=f", float(p)))[0]
            for p in re.split(r"[,/]+", raw)
            if p.strip()
        ]
    except (OverflowError, ValueError):
        return None
    # isfinite: NaN passes the other tests and later int(round(...)) would raise.
    if not parts or any(not math.isfinite(p) or p < 0 for p in parts) or sum(parts) <= 0:
        return None
    try:
        running = 0.0
        for part in parts:
            running = struct.unpack("=f", struct.pack("=f", running + part))[0]
    except OverflowError:
        return None
    if not math.isfinite(running):
        return None
    return parts


def _pass_through_reprices_cache(extra_args: Optional[Iterable[str]], n_parallel) -> bool:
    """Whether an argv -c 0 or a larger --parallel changes the cache Studio priced."""
    try:
        if parse_ctx_override(extra_args) == 0:
            return True
    except ValueError:
        return True
    override = _extra_args_n_parallel(extra_args, {})
    return override is not None and override > int(n_parallel or 1)


def _extra_args_n_parallel(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[int]:
    """Extras beat both the emitted --parallel and LLAMA_ARG_N_PARALLEL, since llama.cpp is last-wins."""
    source_env = os.environ if env is None else env
    found: Optional[int] = None
    raw = source_env.get("LLAMA_ARG_N_PARALLEL")
    if raw:
        try:
            found = int(str(raw).strip())
        except (TypeError, ValueError):
            pass
    args = [str(a) for a in extra_args] if extra_args else []
    for i, arg in enumerate(args):
        name, _, inline = arg.partition("=")
        if name not in ("-np", "--parallel"):
            continue
        value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
        try:
            found = int(value.strip())
        except (TypeError, ValueError):
            continue
    return found


def _named_batch_sizes(
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    n_batch: Optional[int] = None,
    n_ubatch: Optional[int] = None,
) -> tuple[int, int, bool, bool]:
    """Precedence is environment, first-class fields, then extra arguments."""
    values = {
        "batch": _DEFAULT_LLAMA_N_BATCH,
        "ubatch": _DEFAULT_LLAMA_N_UBATCH,
    }
    named = {"batch": False, "ubatch": False}
    source_env = os.environ if env is None else env
    for key, env_name in (
        ("batch", "LLAMA_ARG_BATCH"),
        ("ubatch", "LLAMA_ARG_UBATCH"),
    ):
        raw = source_env.get(env_name)
        if raw:
            try:
                values[key] = int(raw)
                named[key] = True
            except (TypeError, ValueError):
                pass

    if n_batch is not None:
        values["batch"] = int(n_batch)
        named["batch"] = True
    if n_ubatch is not None:
        values["ubatch"] = int(n_ubatch)
        named["ubatch"] = True

    args = [str(a) for a in extra_args] if extra_args else []
    flags = {
        "-b": "batch",
        "--batch-size": "batch",
        "-ub": "ubatch",
        "--ubatch-size": "ubatch",
    }
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        key = flags.get(flag)
        if key is None:
            continue
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        try:
            values[key] = int(value)
            named[key] = True
        except (TypeError, ValueError):
            continue
    return values["batch"], values["ubatch"], named["batch"], named["ubatch"]


def _extra_args_n_ubatch(
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    n_ctx: Optional[int] = None,
    *,
    n_batch: Optional[int] = None,
    n_ubatch: Optional[int] = None,
) -> Optional[int]:
    """Effective ubatch after llama.cpp normalizes it, or None at defaults."""
    _batch, _ubatch, _batch_named, _ubatch_named = _named_batch_sizes(
        extra_args, env, n_batch, n_ubatch
    )
    values = {"batch": _batch, "ubatch": _ubatch}
    if not (_batch_named or _ubatch_named):
        return None

    # llama.cpp converts to uint32_t; a zero ubatch means use batch.
    batch = values["batch"] & 0xFFFFFFFF
    raw_ubatch = values["ubatch"]
    ubatch = batch if raw_ubatch == 0 else raw_ubatch & 0xFFFFFFFF
    effective = min(batch, ubatch)
    if n_ctx is not None and n_ctx > 0:
        effective = min(effective, n_ctx)
    return effective


def _read_gguf_embedding_length(path: Optional[str]) -> Optional[int]:
    """``{arch}.embedding_length``, cached, or None. Thin import-local wrapper."""
    if not path:
        return None
    try:
        from utils.models.gguf_metadata import read_gguf_embedding_length
        return read_gguf_embedding_length(str(path))
    except Exception as e:
        logger.debug(f"embedding length read failed: {e}")
        return None


def _unknown_projector_ubatch(
    extra_args: Optional[Iterable[str]] = None, env: Optional[Mapping[str, str]] = None
) -> int:
    """Return conservative headroom for an unclassified projector."""
    return max(_MMPROJ_UNKNOWN_UBATCH, extra_args_image_max_tokens(extra_args, env) or 0)


def _mmproj_required_ubatch(
    mmproj_path: Optional[str],
    n_embd_text: Optional[int] = None,
    extra_args: Optional[Iterable[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Audio-only projectors can still be marked as vision, so the image tower is inspected."""
    if not mmproj_path:
        return 0
    try:
        from utils.models.gguf_metadata import (
            mmproj_accepts_image,
            read_mmproj_vision_projector_type,
        )
        if not mmproj_accepts_image(mmproj_path):
            return 0
        family = (read_mmproj_vision_projector_type(mmproj_path) or "").strip().lower()
    except Exception as e:
        logger.debug(f"mmproj capability read failed: {e}")
        family = ""
    custom = extra_args_image_max_tokens(extra_args, env) or 0
    if not family:
        return _unknown_projector_ubatch(extra_args, env)
    if family not in _MMPROJ_NON_CAUSAL_IMAGE_TOKENS:
        return 0
    if family == "gemma4v" and n_embd_text in _GEMMA4V_CAUSAL_TEXT_N_EMBD:
        return 0
    # A custom image limit can exceed the family's built-in ceiling.
    ceiling = max(_MMPROJ_NON_CAUSAL_IMAGE_TOKENS[family], custom)
    return ceiling if ceiling > _DEFAULT_LLAMA_N_UBATCH else 0


def _launch_required_ubatch(
    mmproj_path: Optional[str],
    n_embd_text: Optional[int] = None,
    extra_args: Optional[Iterable[str]] = None,
    *,
    is_vision: bool = True,
    vision_off: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Takes the maximum over candidate projectors instead of copying llama.cpp's precedence rules."""
    required = 0

    # Pass-through arguments override managed projector flags.
    override = _extra_args_device(extra_args, {"--mmproj", "-mm"})
    if override and not vision_off and not extra_args_disable_mmproj(extra_args):
        required = max(
            required, _mmproj_required_ubatch(str(override), n_embd_text, extra_args, env)
        )

    if not vision_off:
        # Inherited projectors survive --no-mmproj but not the vision switch.
        source_env = os.environ if env is None else env
        if (source_env.get("LLAMA_ARG_MMPROJ_URL") or "").strip():
            required = max(required, _unknown_projector_ubatch(extra_args, env))
        inherited = (source_env.get("LLAMA_ARG_MMPROJ") or "").strip()
        if inherited:
            required = max(
                required, _mmproj_required_ubatch(inherited, n_embd_text, extra_args, env)
            )

    if is_vision and not vision_off and not extra_args_disable_mmproj(extra_args):
        if mmproj_path:
            required = max(
                required, _mmproj_required_ubatch(str(mmproj_path), n_embd_text, extra_args, env)
            )

    if not vision_off and extra_args_mmproj_auto(extra_args, env):
        # llama-server's adjacent-projector search can disagree with ours.
        required = max(
            required,
            _mmproj_required_ubatch(str(mmproj_path), n_embd_text, extra_args, env)
            if mmproj_path
            else _unknown_projector_ubatch(extra_args, env),
        )

    return required


def _batch_ubatch_for_mmproj(
    required_ubatch: int,
    n_batch: Optional[int],
    n_ubatch: Optional[int],
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
) -> tuple[Optional[int], Optional[int]]:
    """Raise an unset micro-batch, capped by the effective batch size."""
    if required_ubatch <= 0:
        return n_batch, n_ubatch
    batch, ubatch, batch_named, ubatch_named = _named_batch_sizes(
        extra_args, env, n_batch, n_ubatch
    )
    if ubatch_named:
        return n_batch, n_ubatch
    # Match llama_context_params, which converts the signed batch to uint32_t.
    batch &= 0xFFFFFFFF
    target = min(required_ubatch, batch)
    if target <= ubatch:
        return n_batch, n_ubatch
    return n_batch, target


def _embedding_batch_ubatch(
    n_ctx: int,
    n_batch: Optional[int],
    n_ubatch: Optional[int],
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
) -> tuple[Optional[int], Optional[int]]:
    """Size an unset batch pair to the context for pooling that needs one micro-batch."""
    _, _, batch_named, ubatch_named = _named_batch_sizes(extra_args, env, n_batch, n_ubatch)
    if (batch_named and ubatch_named) or n_ctx <= _DEFAULT_LLAMA_N_UBATCH:
        return n_batch, n_ubatch
    # llama.cpp caps the micro-batch at the batch, so the unset side must grow too.
    return (n_batch if batch_named else n_ctx), (n_ubatch if ubatch_named else n_ctx)


# With MoE experts in host RAM, a larger micro-batch cuts PCIe expert copies; 4096 was slower.
_MOE_SPILL_N_UBATCH = 2048


def _moe_spill_batch_ubatch(
    n_batch: Optional[int],
    n_ubatch: Optional[int],
    *,
    n_moe_layers: int,
    experts_on_host: bool,
    discrete_gpu: bool,
    user_named_batch: bool,
) -> tuple[Optional[int], Optional[int]]:
    """A user-set batch always wins; read user_named_batch before projector raises set n_ubatch."""
    if user_named_batch or not experts_on_host or not discrete_gpu or n_moe_layers <= 0:
        return n_batch, n_ubatch
    current = _DEFAULT_LLAMA_N_UBATCH if n_ubatch is None else int(n_ubatch)
    if current >= _MOE_SPILL_N_UBATCH:
        return n_batch, n_ubatch
    # llama.cpp caps the micro-batch at the batch, so keep batch >= micro-batch.
    batch = _DEFAULT_LLAMA_N_BATCH if n_batch is None else int(n_batch)
    if batch < _MOE_SPILL_N_UBATCH:
        n_batch = _MOE_SPILL_N_UBATCH
    return n_batch, _MOE_SPILL_N_UBATCH


def _override_targets_host(value: str) -> bool:
    """Whether an ``--override-tensor`` value sends any tensor to host memory
    (``CPU``, ``CPU_REPACK``, ``CUDA_Host``, ...), not only to GPUs."""
    targets = [
        part.rsplit("=", 1)[-1].strip().lower() for part in str(value).split(",") if "=" in part
    ]
    return any(t.startswith("cpu") or t.endswith("_host") for t in targets)


def _expert_spill_places_tensors_on_cpu(
    extra_args: Optional[Iterable[str]] = None, env: Optional[Mapping[str, str]] = None
) -> bool:
    """An -ot counts only when it targets host memory; -ot exps=CUDA0 keeps weights on a GPU."""
    args = [str(a) for a in extra_args or ()]
    kept: list[str] = []
    i = 0
    while i < len(args):
        if _flag_name(args[i]) in {"-ot", "--override-tensor"}:
            _, eq, inline = args[i].partition("=")
            step = 1 if eq else 2
            if _override_targets_host(inline if eq else (args[i + 1] if i + 1 < len(args) else "")):
                kept.extend(args[i : i + step])
            i += step
            continue
        kept.append(args[i])
        i += 1
    source_env = dict(os.environ if env is None else env)
    if not _override_targets_host(source_env.get("LLAMA_ARG_OVERRIDE_TENSOR") or ""):
        source_env.pop("LLAMA_ARG_OVERRIDE_TENSOR", None)
    return _args_place_tensors_on_cpu(kept) or _env_places_tensors_on_cpu(source_env)


def _capped_draft_n_ubatch(args, env, ctx, *batch_defaults) -> Optional[int]:
    """--spec-draft-n-ubatch capped as llama_context caps it: at --batch-size, then a positive
    context. ``batch_defaults`` are _named_batch_sizes' fallbacks."""
    ub = _draft_n_ubatch(args, env)
    if ub:
        ub = min(ub, _named_batch_sizes(args, env, *batch_defaults)[0] & 0xFFFFFFFF)
        if ctx and ctx > 0:
            ub = min(ub, int(ctx))
    return ub
