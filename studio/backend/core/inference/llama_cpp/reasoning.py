# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reasoning flags and effort levels."""

import os
import re
import threading

from typing import Optional

from .log import logger

# Ordered weakest -> strongest. Must stay in sync with REASONING_EFFORT_SCALE in
# studio/frontend/src/features/chat/provider-capabilities.ts.
_REASONING_EFFORT_SCALE = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# Matches Qwen3.8 but not future names such as Qwen3.80.
_QWEN38_MODEL_RE = re.compile(r"(?:^|[^a-z0-9])qwen3\.8(?:$|[-_/\\])", re.IGNORECASE)


def _extract_reasoning_effort_levels(chat_template: str) -> list:
    """Reports only the levels the template quotes, in canonical order, not the whole scale."""
    return [
        level
        for level in _REASONING_EFFORT_SCALE
        if f"'{level}'" in chat_template or f'"{level}"' in chat_template
    ]


def _reasoning_effort_allowlist(chat_template: str) -> list:
    match = _REASONING_EFFORT_ALLOWLIST_RE.search(chat_template)
    return _extract_reasoning_effort_levels(match.group(1)) if match else []


# Only a validation guard: a plain membership branch can group levels or reject the ones it names.
_REASONING_EFFORT_ALLOWLIST_RE = re.compile(
    r"\{%-?\s*if\s+reasoning_effort\s+not\s+in\s+\[([^\]]*)\]\s*-?%\}\s*\{\{-?\s*raise_exception\("
)


# Log each substitution once per (levels, requested) pair.
_REASONING_EFFORT_CLAMPS_LOGGED: set = set()


def _clamp_reasoning_effort_to_levels(effort: Optional[str], levels) -> Optional[str]:
    """Mirrors the UI clamp in provider-capabilities.ts: prefer the nearest lower level, then the weakest; an unavailable xhigh becomes max when offered."""
    if effort not in _REASONING_EFFORT_SCALE or not levels:
        return None
    if effort == "xhigh" and "xhigh" not in levels and "max" in levels:
        effort = "max"
    if effort in levels:
        return effort
    offered = [level for level in _REASONING_EFFORT_SCALE if level in levels and level != "none"]
    if not offered:
        return None
    rank = _REASONING_EFFORT_SCALE.index(effort)
    below = [level for level in offered if _REASONING_EFFORT_SCALE.index(level) < rank]
    return below[-1] if below else offered[0]


def _rendered_chat_template(backend) -> Optional[str]:
    # getattr: test doubles skip __init__ and borrow backend methods unbound.
    return getattr(backend, "_chat_template_override", None) or getattr(
        backend, "_chat_template", None
    )


def _fit_reasoning_effort_to_template(
    chat_template: Optional[str],
    effort,
    thinking_off: bool = False,
):
    """A level the template's raising allowlist rejects becomes one it accepts, so the render cannot fail."""
    allowed = _reasoning_effort_allowlist(chat_template or "")
    if not allowed or effort in allowed:
        return effort
    if thinking_off and "none" in allowed:
        return "none"
    return _clamp_reasoning_effort_to_levels(effort, allowed) or effort


# Keyed on identity, not name, so a rebuilt GGUF at the same path is logged again.
_GGUF_METADATA_LOGGED: "dict[tuple, bool]" = {}
_GGUF_METADATA_LOGGED_MAX = 32
_GGUF_METADATA_LOGGED_LOCK = threading.Lock()


def _note_gguf_metadata_read(gguf_path: str) -> bool:
    """True on the first read of this file or when its stat fails; False on repeat reads."""
    try:
        st = os.stat(gguf_path)
        key = (os.path.abspath(gguf_path), st.st_size, st.st_mtime_ns)
    except Exception:
        return True
    with _GGUF_METADATA_LOGGED_LOCK:
        if key in _GGUF_METADATA_LOGGED:
            return False
        _GGUF_METADATA_LOGGED[key] = True
        while len(_GGUF_METADATA_LOGGED) > _GGUF_METADATA_LOGGED_MAX:
            # Insertion-ordered, so this drops the oldest.
            del _GGUF_METADATA_LOGGED[next(iter(_GGUF_METADATA_LOGGED))]
        return True


def detect_reasoning_flags(
    chat_template: Optional[str],
    model_identifier: Optional[str] = None,
    *,
    log_source: Optional[str] = None,
    log_level: str = "info",
) -> dict:
    """Shared by the llama-server and safetensors paths so both agree on the flags the frontend sees."""
    flags = {
        "supports_reasoning": False,
        "reasoning_style": "enable_thinking",
        "reasoning_always_on": False,
        "reasoning_effort_levels": [],
        "supports_preserve_thinking": False,
        "preserve_thinking_default": False,
        "supports_tools": False,
    }
    if not chat_template:
        return flags
    tpl = chat_template
    prefix = f"{log_source}: " if log_source else ""
    # The GGUF sniffer re-derives these per request, so it logs at debug after the first read of a file.
    _log = logger.debug if log_level == "debug" else logger.info

    # 'none' is this style's disable sentinel; left in, a clamp would silently disable thinking.
    effort_levels = (
        [lv for lv in _extract_reasoning_effort_levels(tpl) if lv != "none"]
        if ("reasoning_effort" in tpl and "enable_thinking" in tpl)
        else []
    )
    if effort_levels:
        # DeepSeek-V4 takes high but its template only branches on max, so add it; match whole repo-name segments so deepseek-v40 does not false-match.
        from core.inference.model_ids import hf_cache_repo_id

        # A snapshot path's last component is the revision, so read it as its repo id first.
        repo_name = hf_cache_repo_id(model_identifier) or model_identifier or ""
        segments = re.split(r"[-_.]", repo_name.lower().split("/")[-1])
        is_dsv4 = "deepseek4" in segments or any(
            a == "deepseek" and b == "v4" for a, b in zip(segments, segments[1:])
        )
        if is_dsv4 and "high" not in effort_levels:
            effort_levels = sorted(set(effort_levels) | {"high"}, key = _REASONING_EFFORT_SCALE.index)
        # GLM-5.2 style: enable_thinking gate plus a discrete reasoning_effort level set.
        flags["supports_reasoning"] = True
        flags["reasoning_style"] = "enable_thinking_effort"
        flags["reasoning_effort_levels"] = effort_levels
        _log(
            f"{prefix}model supports reasoning "
            f"(enable_thinking + reasoning_effort: {effort_levels})"
        )
    elif "enable_thinking" in tpl:
        flags["supports_reasoning"] = True
        flags["reasoning_style"] = "enable_thinking"
        _log(f"{prefix}model supports reasoning (enable_thinking)")
    elif "reasoning_effort" in tpl:
        # gpt-oss / Harmony use reasoning_effort levels; trust the scan only if it has 'low' and 'high'.
        flags["supports_reasoning"] = True
        flags["reasoning_style"] = "reasoning_effort"
        scanned = _extract_reasoning_effort_levels(tpl)
        allowed = _reasoning_effort_allowlist(tpl)
        if any(level != "none" for level in allowed):
            flags["reasoning_effort_levels"] = allowed
        elif "low" in scanned and "high" in scanned:
            flags["reasoning_effort_levels"] = scanned
        _log(
            f"{prefix}model supports reasoning "
            f"(reasoning_effort: {flags['reasoning_effort_levels'] or 'default levels'})"
        )
    elif "thinking" in tpl:
        # DeepSeek uses 'thinking', not 'enable_thinking'
        normalized_id = (model_identifier or "").lower()
        if "deepseek" in normalized_id:
            flags["supports_reasoning"] = True
            _log(f"{prefix}model supports reasoning (DeepSeek thinking)")

    # Hardcoded <think> or reasoning_content means thinking is always on.
    if not flags["supports_reasoning"]:
        if ("<think>" in tpl and "</think>" in tpl) or "reasoning_content" in tpl:
            flags["supports_reasoning"] = True
            flags["reasoning_always_on"] = True
            _log(f"{prefix}model always reasons (<think> tags in template)")

    # preserve_thinking is an independent kwarg on some Qwen templates that keeps prior <think> blocks.
    if "preserve_thinking" in tpl:
        flags["supports_preserve_thinking"] = True
        # Qwen3.8 deliberately defaults preserve_thinking on; keep it family-scoped.
        flags["preserve_thinking_default"] = bool(_QWEN38_MODEL_RE.search(model_identifier or ""))
        _log(f"{prefix}model supports preserve_thinking")

    from core.inference.template_capabilities import template_supports_tools

    if isinstance(tpl, str) and template_supports_tools(tpl):
        flags["supports_tools"] = True
        _log(f"{prefix}model supports tool calling")

    return flags


# Inkling's float() coercion turns unknown names into 0 (no thinking), so map them.
_INKLING_REASONING_EFFORT = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
    "max": 0.99,
}


# Muse Spark's template accepts only low/medium/high/xhigh and raise_exception()s on
# anything else, so a "none" or "max" request would fail the render. It cannot turn
# reasoning off: "none" and "minimal" take the smallest budget, "max" the largest.
_MUSE_SPARK_REASONING_EFFORT = {
    "none": "low",
    "minimal": "low",
    "max": "xhigh",
}


def _coerce_reasoning_effort(architecture, kwargs: dict) -> dict:
    if architecture == "inkling":
        effort = kwargs.get("reasoning_effort")
        if isinstance(effort, str):
            mapped = _INKLING_REASONING_EFFORT.get(effort.strip().lower())
            if mapped is not None:
                kwargs["reasoning_effort"] = mapped
    elif architecture == "muse-spark":
        effort = kwargs.get("reasoning_effort")
        if isinstance(effort, str):
            mapped = _MUSE_SPARK_REASONING_EFFORT.get(effort.strip().lower())
            if mapped is not None:
                kwargs["reasoning_effort"] = mapped
    return kwargs
