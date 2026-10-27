# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Page-aware recursive-separator chunking with token overlap. Each chunk records
its ``[page_char_start, page_char_end)`` span and ``source_page_index``, used by
the locator pass to highlight it on the PDF page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .parsers import Page

TokenCounter = Callable[[str], int]
SEPARATORS = ("\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ")


@dataclass(frozen = True)
class Chunk:
    text: str
    token_count: int
    page_number: int | None
    source_page_index: int
    chunk_index: int
    page_char_start: int
    page_char_end: int
    whole_document_token_count: int | None = None


@dataclass(frozen = True, slots = True)
class _Piece:
    text: str
    token_count: int


def _split(
    text: str,
    seps: tuple[str, ...],
    max_tokens: int,
    count: TokenCounter,
    token_count: int | None = None,
    window: int | None = None,
) -> list[_Piece]:
    """Recursively split into pieces each <= max_tokens (best effort). Pieces
    rejoin to ``text`` exactly, so offsets are a running length. A run with no
    separator is cut into ``window``-token pieces (default ``max_tokens``)."""
    if token_count is None:
        token_count = count(text)
    if token_count <= max_tokens:
        return [_Piece(text, token_count)]
    for i, sep in enumerate(seps):
        parts = text.split(sep)
        if len(parts) <= 1:
            continue
        if sep:
            parts = [p + sep for p in parts[:-1]] + parts[-1:]
        out: list[_Piece] = []
        for p in parts:
            if not p:
                continue
            tokens = count(p)
            out.extend(
                [_Piece(p, tokens)]
                if tokens <= max_tokens
                else _split(p, seps[i + 1 :], max_tokens, count, tokens, window)
            )
        return out
    # No separator left (minified code, base64, long URLs): each window is the largest
    # prefix within the budget (never under one character, so it ends). The search starts
    # at the previous window's length and steps out by doubling before it bisects, so a
    # uniform run costs about two counts per window, each about the window's own size.
    budget = window or max_tokens
    out = []
    start = 0
    guess = max(1, budget * 4)
    while start < len(text):
        rest = len(text) - start
        lo, lo_tokens, bad, step = 0, None, rest + 1, 1
        probe = min(rest, guess)
        tokens = count(text[start : start + probe])
        if tokens <= budget:
            lo, lo_tokens = probe, tokens
            while lo < rest:
                probe = min(rest, lo + step)
                tokens = count(text[start : start + probe])
                if tokens > budget:
                    bad = probe
                    break
                lo, lo_tokens, step = probe, tokens, step * 2
        else:
            bad = probe
            while bad > 1:
                probe = max(1, bad - step)
                tokens = count(text[start : start + probe])
                if tokens <= budget:
                    lo, lo_tokens = probe, tokens
                    break
                bad, step = probe, step * 2
        while bad - lo > 1:
            mid = (lo + bad) // 2
            tokens = count(text[start : start + mid])
            if tokens <= budget:
                lo, lo_tokens = mid, tokens
            else:
                bad = mid
        if lo == 0:
            lo = 1
        part = text[start : start + lo]
        out.append(_Piece(part, count(part) if lo_tokens is None else lo_tokens))
        start += lo
        guess = lo
    return out


def _merge(
    pieces: list[_Piece], starts: list[int], max_tokens: int, overlap: int
) -> list[tuple[str, int, int]]:
    """Greedy-merge pieces into <= max_tokens chunks with token overlap.
    ``starts[i]`` is ``pieces[i]``'s page char offset; returns
    ``(chunk_text, char_start, char_end)`` spans."""
    chunks: list[tuple[str, int, int]] = []
    buf: list[_Piece] = []
    buf_starts: list[int] = []
    buf_tok = 0

    def _flush() -> None:
        raw = "".join(piece.text for piece in buf)
        stripped = raw.strip()
        if not stripped:
            return
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        start = buf_starts[0] + lead
        end = buf_starts[0] + len(raw) - trail
        chunks.append((stripped, start, end))

    for piece, start in zip(pieces, starts):
        # Reuse counts after the GGUF tokenizer's cache evicts earlier pieces.
        pt = piece.token_count
        if buf and buf_tok + pt > max_tokens:
            _flush()
            # Bound carry so carry + piece fits max_tokens, or the embedder overflows.
            carry_budget = min(overlap, max(0, max_tokens - pt))
            carry, carry_starts, run = [], [], 0
            for prev, prev_start in zip(reversed(buf), reversed(buf_starts)):
                if run + prev.token_count > carry_budget:
                    break
                carry.insert(0, prev)
                carry_starts.insert(0, prev_start)
                run += prev.token_count
            buf, buf_starts, buf_tok = carry, carry_starts, run
        buf.append(piece)
        buf_starts.append(start)
        buf_tok += pt
    if buf:
        _flush()
    return chunks


def chunk_pages(
    pages: list[Page], *, max_tokens: int, overlap: int, count: TokenCounter
) -> list[Chunk]:
    """Split each page into overlapping chunks, tracking per-page char offsets."""
    out: list[Chunk] = []
    for page_index, page in enumerate(pages):
        # Small windows in a separator-free run, so _merge can carry whole ones as the overlap
        # and still fit the next one beside them.
        window = min(overlap, max_tokens - overlap) if 0 < overlap < max_tokens else None
        pieces = _split(page.text, SEPARATORS, max_tokens, count, window = window)
        # _split preserves offsets, so a running cursor gives exact ones.
        starts: list[int] = []
        cursor = 0
        for piece in pieces:
            starts.append(cursor)
            cursor += len(piece.text)
        previous_end = 0
        for text, char_start, char_end in _merge(pieces, starts, max_tokens, overlap):
            token_count = count(text)
            overlap_chars = min(len(text), max(0, previous_end - char_start))
            unique_text = text[overlap_chars:].lstrip("\r\n")
            out.append(
                Chunk(
                    text = text,
                    token_count = token_count,
                    page_number = page.page_number,
                    source_page_index = page_index,
                    chunk_index = len(out),
                    page_char_start = char_start,
                    page_char_end = char_end,
                    whole_document_token_count = (
                        token_count if unique_text == text else count(unique_text)
                    ),
                )
            )
            previous_end = max(previous_end, char_end)
    return out
