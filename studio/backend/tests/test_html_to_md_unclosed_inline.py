# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unclosed inline tags under an open ``<p>`` / ``<li>`` keep web-fetch conversion linear.

Optional-end-tag recovery scanned the whole open-tag stack on every start tag
while a closable element was open, in the renderer and again in the title-button
scan, so n unclosed ``<i>`` / ``<span>`` cost O(n^2): a 58 KB page took ~25 s.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference._html_to_md import html_to_markdown


@pytest.mark.parametrize(
    "build",
    [
        lambda n: "<p>" + "a<i>2 " * n + "</p>",
        lambda n: "<p>" + "<b>x" * n,
        lambda n: "<p>" + "<span>x " * n,
        lambda n: "<p>" + "x<sup>2 " * n,
        # an open <li> that the <div>s don't close: every start tag used to walk the whole stack
        lambda n: "<ul><li>" + "<i>x" * n + "<div>y" * n,
    ],
    ids = ["i", "b", "span", "sup", "li-div"],
)
def test_unclosed_inline_tags_stay_linear(build):
    timings = []
    for count in (1000, 8000):
        start = time.perf_counter()
        html_to_markdown(build(count))
        timings.append(time.perf_counter() - start)
    # eight times the input stays well below quadratic (64x) growth
    assert timings[1] < 3 and timings[1] < timings[0] * 24, timings


@pytest.mark.parametrize(
    "html, expected",
    [
        ("<p>one<i>two<p>three", "one*two\n\nthree"),
        ("<ul><li>a<span>b<li>c</ul>", "* ab\n* c"),
        ("<ul><li>a<ul><li>b<li>c</ul><li>d</ul>", "* a\n\n  * b\n  * c\n\n* d"),
        ("<table><tr><td>a<i>b<td>c<tr><td>d</table>", "| a*b | c |\n| --- | --- |\n| d |"),
    ],
)
def test_recovery_still_closes_the_right_element(html, expected):
    assert html_to_markdown(html) == expected
