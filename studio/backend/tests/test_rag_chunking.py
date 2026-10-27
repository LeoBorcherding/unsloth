# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chunking unit tests (no DB, no model)."""

from core.rag.chunking import chunk_pages
from core.rag.parsers import Page, parse_text

from collections import Counter
from functools import lru_cache

WORDS = lambda t: len(t.split())  # noqa: E731


def _page(text: str, page_number = None) -> Page:
    return Page(text = text, page_number = page_number, char_count = len(text))


def test_chunk_token_bounds_and_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_pages([_page(text)], max_tokens = 128, overlap = 24, count = WORDS)
    assert len(chunks) >= 3
    assert all(c.token_count <= 128 for c in chunks)
    a, b = chunks[0].text.split(), chunks[1].text.split()
    shared = next((n for n in range(60, 0, -1) if a[-n:] == b[:n]), 0)
    assert shared == 24


def test_chunk_never_exceeds_max_with_overlap_carry():
    """Overlap carry is trimmed so no chunk exceeds max_tokens (else the embedder overflows)."""
    s1 = " ".join("a" for _ in range(10))
    s2 = " ".join("b" for _ in range(95))
    chunks = chunk_pages([_page(f"{s1}. {s2}")], max_tokens = 100, overlap = 24, count = WORDS)
    assert all(c.token_count <= 100 for c in chunks), [c.token_count for c in chunks]


def test_chunk_indices_are_sequential():
    chunks = chunk_pages([_page("alpha. " * 200)], max_tokens = 32, overlap = 0, count = WORDS)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_tracks_source_page_index():
    pages = [_page("alpha bravo " * 80, 1), _page("charlie delta " * 80, 2)]
    chunks = chunk_pages(pages, max_tokens = 32, overlap = 0, count = WORDS)
    page0 = [c for c in chunks if c.source_page_index == 0]
    page1 = [c for c in chunks if c.source_page_index == 1]
    assert page0 and page1
    assert all(c.page_number == 1 for c in page0)
    assert all(c.page_number == 2 for c in page1)


def test_chunk_char_offsets_locate_text_in_page():
    page_text = "alpha bravo charlie delta echo foxtrot golf hotel " * 30
    pages = [_page(page_text, 1)]
    chunks = chunk_pages(pages, max_tokens = 16, overlap = 0, count = WORDS)
    assert len(chunks) > 1
    for c in chunks:
        assert 0 <= c.page_char_start < c.page_char_end <= len(page_text)
        sliced = page_text[c.page_char_start : c.page_char_end]
        assert c.text in sliced or sliced.strip() == c.text


def test_empty_page_yields_no_chunks():
    chunks = chunk_pages([_page("   \n  ")], max_tokens = 32, overlap = 0, count = WORDS)
    assert chunks == []


def test_parse_text_single_page():
    pages = parse_text("hello world")
    assert len(pages) == 1
    assert pages[0].char_count == len("hello world")


def test_large_document_does_not_retokenize_every_piece_after_cache_eviction():
    requests = Counter()

    @lru_cache(maxsize = 4096)
    def count(text):
        requests[text] += 1
        return len(text.split())

    text = " ".join(f"word{i}" for i in range(10_000))
    chunks = chunk_pages([_page(text)], max_tokens = 500, overlap = 64, count = count)
    assert max(requests.values()) == 1
    assert len(chunks) > 20
    assert all(c.token_count <= 500 for c in chunks)
    assert chunks[-1].page_char_end == len(text)


B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def _unbreakable(n):
    return "".join(B64[i % 64] for i in range(n))


def test_unbreakable_run_fills_the_token_budget():
    # Base64 or minified code has no separator; windows must be sized in tokens, not characters.
    calls = Counter()

    def count(text):
        calls["n"] += 1
        return max(1, (len(text) + 3) // 4)

    text = _unbreakable(20_000)
    chunks = chunk_pages([_page(text)], max_tokens = 512, overlap = 0, count = count)
    assert len(chunks) == 10
    assert all(c.token_count <= 512 for c in chunks)
    assert min(c.token_count for c in chunks[:-1]) == 512
    assert calls["n"] < 500
    assert "".join(c.text for c in chunks) == text


def test_unbreakable_run_stays_within_budget_on_dense_text():
    text = _unbreakable(3_000)
    chunks = chunk_pages([_page(text)], max_tokens = 100, overlap = 0, count = len)
    assert all(c.token_count <= 100 for c in chunks)
    assert "".join(c.text for c in chunks) == text


def test_unbreakable_run_tokenizes_about_its_own_length():
    # Each window must not re-tokenize the rest of the run: that is quadratic on a long blob.
    seen = Counter()

    def count(text):
        seen["chars"] += len(text)
        return max(1, (len(text) + 3) // 4)

    text = _unbreakable(300_000)
    chunks = chunk_pages([_page(text)], max_tokens = 512, overlap = 0, count = count)
    assert "".join(c.text for c in chunks) == text
    assert seen["chars"] < 40 * len(text)


def test_unbreakable_run_keeps_the_overlap():
    def count(text):
        return max(1, (len(text) + 3) // 4)

    text = _unbreakable(20_000)
    chunks = chunk_pages([_page(text)], max_tokens = 512, overlap = 64, count = count)
    assert all(c.token_count <= 512 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.page_char_end - nxt.page_char_start == 64 * 4
    assert chunks[0].page_char_start == 0 and chunks[-1].page_char_end == len(text)


def test_unbreakable_run_counts_about_twice_per_window():
    # llama-server counts over HTTP, one /tokenize per call: a uniform run must not pay a
    # fresh search for every overlap-sized window.
    calls = Counter()

    def count(text):
        calls["n"] += 1
        return max(1, (len(text) + 3) // 4)

    text = _unbreakable(300_000)
    chunk_pages([_page(text)], max_tokens = 512, overlap = 64, count = count)
    assert calls["n"] < 4_000


def test_unbreakable_run_keeps_a_high_overlap():
    def count(text):
        return max(1, (len(text) + 3) // 4)

    text = _unbreakable(2_000)
    chunks = chunk_pages([_page(text)], max_tokens = 100, overlap = 99, count = count)
    assert all(c.token_count <= 100 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.page_char_end - nxt.page_char_start == 99 * 4
