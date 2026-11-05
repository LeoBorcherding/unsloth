# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detecting unclosed code fences, markup and answer artifacts in model output."""

import re

from typing import Optional

# Plan-without-action re-prompt state now lives in tool_call_parser (imported above).

# Artifact = content no plan has, else a plan with code gets wiped. Numbered lists excluded: they cost errors both ways. Keep runs bounded against backtracking.
_CLOSED_CODE_FENCE = re.compile(
    r"(?<!`)(?P<bf>`{3,})(?!`)[^\r\n]{0,600}\r?\n[\s\S]{1,4000}?\r?\n[ \t>]*(?P=bf)`*[ \t]*(?:\r?\n|\Z)"
    r"|(?<!~)(?P<tf>~{3,})(?!~)[^\r\n]{0,600}\r?\n[\s\S]{1,4000}?\r?\n[ \t>]*(?P=tf)~*[ \t]*(?:\r?\n|\Z)",
    re.IGNORECASE,
)
_CLOSED_MARKUP_ARTIFACT = re.compile(
    r"(?:<!doctype\b[\s\S]{0,200}?)?<html\b[^>]{0,600}>[^<]{0,400}<[a-zA-Z!/][\s\S]{0,4000}?</html\s*>"
    r"|<svg\b[^>]{0,600}>[^<]{0,400}<[a-zA-Z!/][\s\S]{0,4000}?</svg\s*>",
    re.IGNORECASE,
)
_HAS_ANSWER_ARTIFACT = re.compile(
    # Tilde fences too, since models emit tildes when the body holds backticks. CommonMark: 3+ to open, as many to close, on a cleanly ended line.
    r"(?<!`)(?P<bf>`{3,})(?!`)[^\r\n]{0,600}\r?\n[\s\S]{1,4000}?\r?\n[ \t>]*(?P=bf)`*[ \t]*(?:\r?\n|\Z)"
    r"|(?<!~)(?P<tf>~{3,})(?!~)[^\r\n]{0,600}\r?\n[\s\S]{1,4000}?\r?\n[ \t>]*(?P=tf)~*[ \t]*(?:\r?\n|\Z)"
    r"|(?:<!doctype\b[\s\S]{0,200}?)?<html\b[^>]{0,600}>[^<]{0,400}<[a-zA-Z!/][\s\S]{0,4000}?</html\s*>"
    r"|<svg\b[^>]{0,600}>[^<]{0,400}<[a-zA-Z!/][\s\S]{0,4000}?</svg\s*>",
    re.IGNORECASE,
)

_FENCE_RUN_RE = re.compile(r"(?<!`)(?P<backticks>`{3,})(?!`)|(?<!~)(?P<tildes>~{3,})(?!~)")
# One list marker may precede the quote: "- > ```py" is a quote in a list item.
_BLOCKQUOTE_PREFIX = re.compile(r"^[ \t]*(?:[-*+]|\d{1,9}[.)])?[ \t]*(?:>[ \t]?)+")
# Language token that makes an inline run an opener; never ends in a dot so "here." is prose.
_FENCE_INFO_STRING_RE = re.compile(r"[A-Za-z][\w+#-]*(?:\.[\w+#-]+)*")
# A fence on a list-marker line is block level, so the prose rules below do not apply.
_LIST_MARKER_ONLY = re.compile(r"^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+$")
_ONE_QUOTE_MARKER = re.compile(r"[ \t]*>[ \t]?")


def _has_unclosed_code_fence(text: str) -> bool:
    """Column-0 fences always count; inline ones only if they cannot read as prose, since models open fences mid-line. Quote pairing is done here, not in the regex."""
    active_char: Optional[str] = None
    active_len = 0
    active_quote = 0
    active_base = 0
    closed_any = False
    for raw_line in text.splitlines():
        # A quote marker four columns in is indented code, not a container. List markers stay alone: deeper, they may continue a list not visible in one line.
        lead = raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]
        quote = _BLOCKQUOTE_PREFIX.match(raw_line)
        indented_quote = quote is not None and len(lead.expandtabs(4)) > 3
        if indented_quote:
            quote = None
        quote_depth = quote.group(0).count(">") if quote else 0
        line = raw_line[quote.end() :] if quote else raw_line
        runs = list(_FENCE_RUN_RE.finditer(line))
        for index, m in enumerate(runs):
            fence = m.group("backticks") or m.group("tildes")
            trailing = line[m.end() :].strip()
            ch = fence[0]
            prefix = line[: m.start()]
            indent = len(prefix.expandtabs(4))
            blank_prefix = not prefix.strip()
            if active_char is not None:
                # A closer is alone on its line, within 3 columns of the container.
                if (
                    blank_prefix
                    and active_base <= indent <= active_base + 3
                    and ch == active_char
                    and len(fence) >= active_len
                    and quote_depth == active_quote
                    and not trailing
                ):
                    active_char, active_len, active_quote, active_base = None, 0, 0, 0
                    closed_any = True
                continue
            # A later backtick run closes an inline span at any column; other delimiters are info-string text, even tildes inside a tilde info string.
            if ch == "`" and any(
                (later.group("backticks") or later.group("tildes"))[0] == ch
                for later in runs[index + 1 :]
            ):
                continue
            if blank_prefix:
                # A fence's baseline is 0 (its own indent); past three columns it is indented code and opens nothing.
                if indent <= 3:
                    active_char, active_len = ch, len(fence)
                    active_quote, active_base = quote_depth, 0
                continue
            # Models open mid-prose fences: a bare info string marks one, and only before any close. A column-0 fence is a real block and is unaffected.
            in_list = _LIST_MARKER_ONLY.match(prefix) is not None
            if not in_list:
                if indented_quote:
                    continue
                if trailing and not _FENCE_INFO_STRING_RE.fullmatch(trailing):
                    continue
                if closed_any:
                    continue
            active_char, active_len, active_quote = ch, len(fence), quote_depth
            # A list marker is the container, so its width is the baseline; mid-sentence the baseline is 0.
            active_base = indent if in_list else 0
    return active_char is not None


def _has_unclosed_markup_block(text: str) -> bool:
    """Counted rather than matched, so a closed block followed by an open one still counts as unclosed."""
    opens_html = len(re.findall(r"<html\b", text, re.IGNORECASE))
    closes_html = len(re.findall(r"</html\s*>", text, re.IGNORECASE))
    if opens_html > closes_html:
        return True
    opens_svg = len(re.findall(r"<svg\b", text, re.IGNORECASE))
    closes_svg = len(re.findall(r"</svg\s*>", text, re.IGNORECASE))
    return opens_svg > closes_svg


# "First, I'll create an <html></html> skeleton" is a plan, not a page.
_EMPTY_MARKUP_SKELETON = re.compile(
    r"<(html|svg)\b[^>]*>\s*</\1\s*>",
    re.IGNORECASE,
)
_DOCTYPE_PREFIX = re.compile(
    r"^<!doctype\b[\s\S]{0,200}?>",
    re.IGNORECASE,
)


def _is_empty_markup_skeleton(matched: str) -> bool:
    """True if ``matched`` is an empty <html></html> / <svg></svg>, doctype
    prefix and surrounding whitespace allowed."""
    candidate = _DOCTYPE_PREFIX.sub("", matched.strip(), count = 1).strip()
    return _EMPTY_MARKUP_SKELETON.fullmatch(candidate) is not None


def _is_blank_fence(matched: str, depth: int = 0) -> bool:
    """Only whitespace between the delimiter lines counts as blank, so a lone space is not an answer."""
    if matched[:1] not in ("`", "~"):
        return False
    lines = matched.splitlines()
    # Depth is the quote depth of the opener, read from the match start line. A deeper last line is a nested delimiter in the body, so it is content, not blank.
    closing = _BLOCKQUOTE_PREFIX.match(lines[-1]) if lines else None
    if (closing.group(0).count(">") if closing else 0) != depth:
        return False
    for line in lines[1:-1]:
        for _ in range(depth):
            marker = _ONE_QUOTE_MARKER.match(line)
            if marker is None:
                break
            line = line[marker.end() :]
        if line.strip():
            return False
    return True


def _first_real_artifact(text: str):
    """First _HAS_ANSWER_ARTIFACT match with something in it.

    Every match is inspected, so a skeleton followed by a real page counts."""
    for m in _HAS_ANSWER_ARTIFACT.finditer(text):
        if _is_empty_markup_skeleton(m.group(0)):
            continue
        # The container the fence sits in is whatever quotes the line it opens on.
        line_start = text.rfind("\n", 0, m.start()) + 1
        quote = _BLOCKQUOTE_PREFIX.match(text[line_start : m.start()])
        depth = quote.group(0).count(">") if quote else 0
        if _is_blank_fence(m.group(0), depth):
            continue
        return m
    return None


def _strip_markup_outside_fences(text: str) -> str:
    """Blocks that begin inside a code fence are kept, so an example's <html> cannot pair with prose."""
    fences = [m.span() for m in _CLOSED_CODE_FENCE.finditer(text)]

    def _keep_examples(m):
        # Either endpoint inside a fence makes the match that example's content.
        return m.group(0) if any(s <= m.start() < e or s < m.end() <= e for s, e in fences) else ""

    return _CLOSED_MARKUP_ARTIFACT.sub(_keep_examples, text)


# A reasoning opener ends at a tag boundary: "<think-card>" is an element of the
# answer, not a thought.
_THINK_OPEN_RE = re.compile(r"<think[\s>]")
_THINKING_OPEN_RE = re.compile(r"<thinking[\s>]")
_BRACKET_THINK_OPEN_RE = re.compile(r"\[THINK\]")


def _artifact_spans(text: str) -> "list[tuple[int, int]]":
    """Spans of the complete fences and complete markup in ``text``."""
    return [m.span() for m in _CLOSED_CODE_FENCE.finditer(text)] + [
        m.span() for m in _CLOSED_MARKUP_ARTIFACT.finditer(text)
    ]


def _find_outside_artifacts(
    text: str,
    needle: str,
    spans = None,
) -> int:
    """Ignores needle matches inside complete code fences or markup, since an example is not reasoning."""
    at = text.find(needle)
    if at < 0:
        return -1
    if spans is None:
        spans = _artifact_spans(text)
    while at >= 0:
        if not any(start <= at < end for start, end in spans):
            return at
        at = text.find(needle, at + 1)
    return -1


def _text_outside_think(text: str) -> str:
    """Drops only a leading reasoning block; a later think tag is quoted, not reasoning. A prefilled template may send only the closer."""
    stripped = text.lstrip()
    pairs = (
        (_THINKING_OPEN_RE, "<thinking", "</thinking>"),
        (_THINK_OPEN_RE, "<think", "</think>"),
        (_BRACKET_THINK_OPEN_RE, "[THINK]", "[/THINK]"),
    )
    # Settle the leading opener first: its block may name another marker.
    for lead, _opener, closer in pairs:
        if lead.match(stripped):
            # First closer, plainly: guessing which quoted closer is an example let spans swallow the real boundary. Matches _split_think_segments in routes/inference.py.
            close = text.find(closer)
            # With no closer, a thought cut off by the window runs to the end, and none of it was shown.
            return text[close + len(closer) :] if close >= 0 else ""
    # No leading opener: a bare closer comes from a template that prefilled the opener.
    spans = None
    for lead, _opener, closer in pairs:
        if closer in text and spans is None:
            spans = _artifact_spans(text)
        close = _find_outside_artifacts(text, closer, spans)
        # Boundary-aware so <think-card> in the trace is not an opener, which leaves the prefill unstripped.
        if close >= 0 and not lead.search(text[:close]):
            return text[close + len(closer) :]
    return text


def _has_answer_artifact(text: str) -> bool:
    """A closed fence or complete page counts; unfinished fences and empty skeletons do not."""
    # Strip closed artifacts first: tags or backticks inside finished code are content.
    if _has_unclosed_code_fence(_strip_markup_outside_fences(text)):
        return False
    # Nested pages are deliberately not inspected; that needs an HTML parser.
    return _first_real_artifact(text) is not None
