# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text heuristics and constants for the tool-calling loop."""

import re

from typing import Optional
from core.inference.tool_call_parser import (
    blocked_markerless_prefix_end,
    held_bare_gemma_tail_len,
    is_reprompt_restatement as _is_reprompt_restatement,
    is_short_intent_without_action as _is_short_intent_without_action,
    promotable_gemma_call_pos,
    REPROMPT_MAX_CHARS as _REPROMPT_MAX_CHARS,
)
from core.tool_healing import _markerless_promotable
from core.inference.context_window import image_dimensions
from core.inference.llama_server_args import extra_args_image_max_tokens
from core.inference.llama_server_args import extra_args_image_min_tokens
import math

# Transport errors can beat the child to reapable; a request cannot wait the 5s MTP reload spends.
_RESPAWN_REAP_GRACE_S = 1.0


_GPU_OOM_HEADROOM_MAX_STRIKES = 3


# Extra --fit-target per mid-session GPU OOM of one load: an identical respawn hits the same
# allocation once the conversation is replayed (#16177).
_GPU_OOM_HEADROOM_STEP_MIB = 1024.0


def _finalize_reasoning_only_cumulative(
    cumulative: str,
    reasoning_text: str,
    finish_reason: Optional[str],
    promote_reasoning_only: bool,
    resumed_reasoning: str = "",
) -> str:
    """Snapshot must stay append-only or suffix consumers drop the fallback. Promote only on clean stop, never on length; include resumed_reasoning."""
    visible_fallback = (
        resumed_reasoning + reasoning_text
        if promote_reasoning_only and finish_reason != "length"
        else ""
    )
    return cumulative + "</think>" + visible_fallback


def _ends_inside_quote(text: str) -> bool:
    """Heuristic: text ends on a quote opener left unmatched on its line, or a lone quote."""
    stripped = (text or "").rstrip()
    if not stripped or stripped[-1] not in "`\"'":
        return False
    if len(stripped) > 1 and not (stripped[-2].isspace() or stripped[-2] in "*_([{"):
        return False
    last = stripped[-1]
    line = stripped.rsplit("\n", 1)[-1]
    # An apostrophe between letters (Qwen2's, don't) is not a quote mark.
    marks = sum(
        1
        for i, ch in enumerate(line)
        if ch == last
        and not (
            ch == "'" and 0 < i < len(line) - 1 and line[i - 1].isalnum() and line[i + 1].isalnum()
        )
    )
    return marks % 2 == 1


def _quote_cut_event(
    reasoning_text: str,
    answer_text: str,
    finish_reason: Optional[str],
    promote_reasoning_only: bool,
) -> Optional[dict]:
    """Flag a mid-quote `stop` in the shown answer; off for Anthropic, which cannot relay it."""
    if not promote_reasoning_only or finish_reason != "stop":
        return None
    shown = answer_text if answer_text.strip() else reasoning_text
    return {"type": "quote_cut"} if _ends_inside_quote(shown) else None


# Only large streamed tool args get an early provisional card; render_html is exempt.
_PROVISIONAL_ARGS_MIN_CHARS = 256
_DEFAULT_STREAM_STALL_TIMEOUT_S = 120.0  # 2 min
# Textual-fallback tool calls are not grammar-bounded, so cap them per turn.
_MAX_TOOL_CALLS_PER_TURN = 8
# The first retry already runs with thinking off; more attempts rarely help.
_MAX_LENGTH_CONTINUATIONS = 2
_CONTINUE_AFTER_LENGTH_STATUS = "Continuing after a long thought..."
_CONTINUE_TRUNCATED_ANSWER_STATUS = "Continuing the answer..."
# Below this a tool result carries only the cut notice; the floor catches a collapse to zero.
_MIN_USEFUL_RESULT_TOKENS = 64
# llama.cpp's default per-image embedding cap plus mtmd's delimiters, when the admission figure is out of reach.
_FALLBACK_IMAGE_TOKENS = 4096 + 128


def _mcp_image_tokens(backend) -> int:
    """Per-image KV allowance for the loaded projector, the one admission charges."""
    try:
        from routes.inference import _openai_llama_admission_image_tokens  # noqa: PLC0415
        return int(_openai_llama_admission_image_tokens(backend))
    except Exception:  # noqa: BLE001 -- sizing must never break a tool call
        return _FALLBACK_IMAGE_TOKENS


def _image_token_price(backend, part: dict) -> int:
    """One image's prompt cost for the fit and tool-result sizing (#14968): the admission ceiling,
    unless a known projector gives a closer upper bound (the ceiling evicts history that fits)."""
    ceiling = _mcp_image_tokens(backend)
    if ceiling <= 0:
        return 0
    projector = str(getattr(backend, "_mmproj_projector_type", None) or "").strip().lower()
    fixed = _FIXED_IMAGE_TOKENS.get(projector)
    if fixed:
        return min(ceiling, fixed + _IMAGE_WRAPPER_TOKENS)
    merged = _MERGED_PATCH_PX.get(projector)
    dims = image_dimensions(part) if merged else None
    if dims is None:
        return ceiling
    width, height = dims
    # Ceil, not round: mtmd may round either side up, and this must stay an upper bound.
    tokens = math.ceil(width / merged) * math.ceil(height / merged)
    extra_args = getattr(backend, "_extra_args", None)
    try:
        cap = extra_args_image_max_tokens(extra_args) or 0
        # mtmd resizes against --image-min-tokens on its own rounded grid, which this does not model.
        if extra_args_image_min_tokens(extra_args):
            return ceiling
    except Exception:  # noqa: BLE001 -- pricing must never break a count
        cap = 0
    cap = cap or ceiling - _IMAGE_WRAPPER_TOKENS
    return min(ceiling, min(tokens, cap) + _IMAGE_WRAPPER_TOKENS)


# Fixed per image: Gemma 3 pools 896 px to 16 x 16 (258 measured on Gemma 3 4B, ceiling 4224).
_FIXED_IMAGE_TOKENS = {"gemma3": 256}


# Qwen-VL merges a 2x2 patch square per embedding: 14 px patches on Qwen2/2.5-VL, 16 on Qwen3-VL.
_MERGED_PATCH_PX = {"qwen2vl_merger": 28, "qwen2.5vl_merger": 28, "qwen3vl_merger": 32}


# mtmd's delimiters around an image's embeddings, as admission allows for them.
_IMAGE_WRAPPER_TOKENS = 128


# How many times one tool may return a byte-identical result before the model is told the
# call is not going to change. Follows OpenClaw's tool-loop detection, which keys on the
# RESULT rather than the arguments and never intervenes while results are still changing:
# the arguments varied on every one of the 18 calls observed here, so an argument-keyed
# guard would have watched the whole thing happen.
_MAX_IDENTICAL_TOOL_RESULTS = 3

# Unset (no kwargs passed) differs from None, a real value meaning send no chat_template_kwargs.
_UNSET = object()


def _with_template_extras(reasoning_kw: Optional[dict], extras: Optional[dict]) -> Optional[dict]:
    """The caller's other template tags (e.g. verification) under Studio's resolved reasoning keys."""
    if not extras:
        return reasoning_kw
    return {**extras, **(reasoning_kw or {})}


# A result with this tail is the window's answer, not the tool's.
_WINDOW_NOTICE_MARKER = "chars for the model;"
# Zero-budget results use _zero_room_stub, which lacks the marker above; missing it lets a model re-read slices forever, so catch it too.
_ZERO_ROOM_MARKER = "no context room left"


def _is_window_notice(result: str) -> bool:
    return _WINDOW_NOTICE_MARKER in result or _ZERO_ROOM_MARKER in result


def _note_before_envelopes(result: str, tool_name: str, add_note) -> str:
    """Text after the __MCP_IMAGES__ array breaks its parse and ships the base64 as text (#11358)."""
    if not isinstance(result, str):
        return add_note(result)
    from core.inference.tools import _split_frontend_suffix

    body, suffix = _split_frontend_suffix(result, tool_name)
    return add_note(body) + suffix


# How far back the classifier looks for llama.cpp's argument-parsing error. It
# prints one and exits, so it is always at the end; generous enough to survive a
# usage hint printed after it, small enough that a 10 MB unterminated line (which
# _drain_stdout keeps whole) stays cheap to scan.
_FAILURE_SCAN_TAIL_CHARS = 64 * 1024
# Caps the child output quoted in errors: enough for a real flag or reason, not megabytes of noise.
_FAILURE_QUOTE_CHARS = 200
# Sentence-anchored: mid-sentence the same words are prose naming a tool and suppressing them loses answers. Keep invoke and query out of the verb list.
_FORCED_PLAN_INTENT = re.compile(
    r"(?:^|[.!?]\s+)\s*"
    r"(?:i\s+(?:(?:need|have|ought)\s+to|should|must)|need\s+to|going\s+to|must|should)"
    r"\s+(?:\w+\s+){0,2}?(?:call|use|run|search|fetch|render)\b",
    re.I | re.M,
)
# Negated forms announce a missing answer, so they are excluded.
_FINAL_ANSWER_SIGNAL = re.compile(
    r"\b(?:final\s+answer|answer\s*:|here\s+is|here's|in\s+summary|result\s*:"
    r"|(?:the\s+)?answer\s+is(?!\s+(?:not|unavailable|unknown|unclear|missing)\b))\b",
    re.I,
)
# A plan that pivots to an answer must survive; the pivot needs its own text. When in doubt keep output: a leaked plan is cosmetic, a dropped answer is not.
_ANSWER_PIVOT = re.compile(
    r"\b(?:but|however|although|though|that\s+said|in\s+the\s+meantime|meanwhile)\b"
    r"[\W_]*(?:\w+[\W_]+){1,}\w",
    re.I,
)


def _gguf_active_tool_names(active_tools: list[dict]) -> list[str]:
    names = [
        (tool.get("function") or {}).get("name")
        for tool in (active_tools or [])
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
    return [name for name in names if name]


# Lookbehind excludes the Mistral [CALL_ID]...[ARGS] shape.
_GGUF_REHEARSAL_ARGS_RE = re.compile(r"(?<!\[CALL_ID\])\b([\w-]+)\[ARGS\]")


def _gguf_rehearsal_signal_pos(text: str, active_tools: list[dict]) -> int:
    """Index of the first ``NAME[ARGS]`` whose NAME is markerless-promotable, else -1. An
    inactive-name ``foo[ARGS]`` in prose is not a call, nor is a bare execution-class
    ``terminal[ARGS]``; mirrors the safetensors ``_earliest_tool_signal`` gating."""
    active = set(_gguf_active_tool_names(active_tools))
    if not active:
        return -1
    for m in _GGUF_REHEARSAL_ARGS_RE.finditer(text):
        if _markerless_promotable(m.group(1), active):
            return m.start()
    return -1


def _gguf_has_genuine_tool_signal(text: str, signals, active_tools: list[dict]) -> bool:
    """Unambiguous markers count on a plain substring hit; an [ARGS] hit is genuine only with an active tool name before it, so inactive-name prose is not parsed."""
    for sig in signals:
        if sig == "[ARGS]":
            if _gguf_rehearsal_signal_pos(text, active_tools) >= 0:
                return True
            continue
        if sig in text:
            return True
    # The parser promotes bare Gemma anywhere, so a mid-prose one is a boundary too.
    return promotable_gemma_call_pos(text, lambda: set(_gguf_active_tool_names(active_tools))) >= 0


_TEXT_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([\w.\-]+)"')
_TEXT_TOOL_GEMMA_RE = re.compile(r"\s*call:([\w.\-]+)")
_TEXT_TOOL_REHEARSAL_RE = re.compile(r"\s*([\w.\-]+)\s*\[ARGS\]")


def _sniff_text_tool_name(text: str, enabled_names: set) -> str:
    """Only enabled names are returned, so prose cannot open a tool card; the final parse still decides."""
    # A blocked leading call will not run, so skip every markerless format before naming.
    text = text[blocked_markerless_prefix_end(text, 0, enabled_names) :]
    m = _TEXT_TOOL_NAME_RE.search(text[:4096])
    if m and m.group(1) in enabled_names:
        return m.group(1)
    m = _TEXT_TOOL_GEMMA_RE.match(text[:256])
    if m and _markerless_promotable(m.group(1), enabled_names):
        return m.group(1)
    m = _TEXT_TOOL_REHEARSAL_RE.match(text[:256])
    if m and _markerless_promotable(m.group(1), enabled_names):
        return m.group(1)
    return ""


def _is_rehearsal_prefix(
    stripped: str,
    active_tools: list[dict],
    *,
    _names: Optional[list] = None,
) -> bool:
    """Execution-class names stream rather than being held; otherwise it mirrors the safetensors loop."""
    if not stripped or any(ch.isspace() for ch in stripped):
        return False
    for name in _gguf_active_tool_names(active_tools) if _names is None else _names:
        # Use the shared gate, which also refuses mcp__* names.
        if (stripped == name or f"{name}[ARGS]".startswith(stripped)) and _markerless_promotable(
            name, None
        ):
            return True
    return False


def _held_rehearsal_tail_len(text: str, active_tools: list[dict]) -> int:
    """Holds a trailing bare tool name back while its [ARGS] may still arrive, including a bare-Gemma call:NAME{ prefix that the signal scan misses until its brace."""
    names = _gguf_active_tool_names(active_tools)
    # A tail longer than every NAME[ARGS] cannot be a prefix, so the scan stops there (CJK has no spaces).
    limit = max(
        (len(name) for name in names if isinstance(name, str)),
        default = 0,
    ) + len("[ARGS]")
    i = len(text)
    stop = max(0, i - limit - 1)
    while i > stop and not text[i - 1].isspace():
        i -= 1
    tail = text[i:]
    held = (
        len(tail)
        if tail and len(tail) <= limit and _is_rehearsal_prefix(tail, active_tools, _names = names)
        else 0
    )
    return max(held, held_bare_gemma_tail_len(text, lambda: set(names)))


def _should_suppress_forced_no_tool_output(text: str, previous: str = "") -> bool:
    """Suppresses only a repeat of the stall text that triggered the nudge, never a final answer."""
    stripped = text.strip()
    if not stripped or len(stripped) >= _REPROMPT_MAX_CHARS:
        return False
    if _FINAL_ANSWER_SIGNAL.search(stripped):
        return False
    plan = _FORCED_PLAN_INTENT.search(stripped)
    if plan is not None:
        # Only the plan itself is safe to drop; anything the turn pivots to after it is the answer.
        return _ANSWER_PIVOT.search(stripped[plan.end() :]) is None
    if not _is_short_intent_without_action(stripped):
        return False
    # INTENT_SIGNAL also fires on answer lead-ins; a stall only if the retry adds nothing.
    return not previous or _is_reprompt_restatement(stripped, previous)
