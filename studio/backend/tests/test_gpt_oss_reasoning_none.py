# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``reasoning_effort: "none"`` on gpt-oss reads as thinking off, not as a level.

llama-server only handles "none" on its top-level field; through
``chat_template_kwargs`` the stock gpt-oss template prints it verbatim as
``Reasoning: none``, a level the model was never trained on. A template that
branches on "none" itself (Inkling) still gets it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


# The reasoning line of the stock gpt-oss template.
GPT_OSS_TEMPLATE = """
{%- if reasoning_effort is not defined %}
    {%- set reasoning_effort = "medium" %}
{%- endif %}
{{- "Reasoning: " + reasoning_effort + "\\n\\n" }}
"""


def _shim(levels = (), architecture = "gpt-oss"):
    from core.inference.llama_cpp import LlamaCppBackend

    backend = object.__new__(LlamaCppBackend)
    backend._supports_reasoning = True
    backend._reasoning_always_on = False
    backend._reasoning_style = "reasoning_effort"
    backend._reasoning_effort_levels = list(levels)
    backend._supports_preserve_thinking = False
    backend._architecture = architecture
    return backend


@pytest.mark.parametrize("enable_thinking", [None, False])
def test_none_matches_the_think_toggle_off(enable_thinking):
    backend = _shim()
    assert backend._request_reasoning_kwargs(enable_thinking, "none", None) == {
        "reasoning_effort": "low"
    }
    assert backend._reasoning_kwargs(False) == {"reasoning_effort": "low"}


def test_gpt_oss_template_never_renders_none():
    jinja2 = pytest.importorskip("jinja2")
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(GPT_OSS_TEMPLATE, "unsloth/gpt-oss-20b-GGUF")
    assert flags["reasoning_style"] == "reasoning_effort"
    kwargs = _shim(flags["reasoning_effort_levels"])._request_reasoning_kwargs(None, "none", None)
    rendered = jinja2.Environment().from_string(GPT_OSS_TEMPLATE).render(**kwargs)
    assert rendered.strip() == "Reasoning: low"


def test_a_template_offering_none_still_gets_it():
    levels = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    assert _shim(levels, "inkling")._request_reasoning_kwargs(None, "none", None) == {
        "reasoning_effort": 0.0
    }
    assert _shim(levels, None)._request_reasoning_kwargs(None, "none", None) == {
        "reasoning_effort": "none"
    }


# Quotes "none" without both "low" and "high", so the scanned levels stay empty.
NONE_BRANCH_TEMPLATE = (
    '{%- if reasoning_effort == "none" %}Reasoning: off'
    "{%- else %}Reasoning: {{ reasoning_effort }}{%- endif %}"
)


def test_a_custom_template_whose_levels_were_not_scanned_keeps_none():
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(NONE_BRANCH_TEMPLATE, "someone/custom-model-GGUF")
    assert flags["reasoning_effort_levels"] == []
    assert _shim([], None)._request_reasoning_kwargs(None, "none", None) == {
        "reasoning_effort": "none"
    }


def test_a_gpt_oss_template_that_branches_on_none_keeps_it():
    override = _shim()
    override._chat_template = GPT_OSS_TEMPLATE
    override._chat_template_override = NONE_BRANCH_TEMPLATE
    assert override._request_reasoning_kwargs(None, "none", None) == {"reasoning_effort": "none"}
    stock = _shim()
    stock._chat_template = GPT_OSS_TEMPLATE
    assert stock._request_reasoning_kwargs(None, "none", None) == {"reasoning_effort": "low"}
