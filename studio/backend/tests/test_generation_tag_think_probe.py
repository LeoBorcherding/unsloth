# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A template using {% generation %} (LFM3-Omni) must still render the think-prefill probe."""

from routes.inference import _generation_prompt_opens_think, _sf_reasoning_prefill_mode

_TPL = (
    "{%- set enable_thinking = enable_thinking | default(false) -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'assistant' -%}{% generation %}{{ m.content }}<|im_end|>{% endgeneration %}"
    "{%- else -%}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|im_start|>assistant\n"
    "{%- if enable_thinking -%}<think>{%- else -%}<think></think>{%- endif -%}{%- endif -%}"
)
_MSGS = [{"role": "user", "content": "What is 2+2?"}]
_FEATURES = {"reasoning_style": "enable_thinking", "supports_reasoning": True}


def test_thinking_off_by_default_is_not_prefilled():
    assert _generation_prompt_opens_think(_TPL, None, None, _MSGS) is False
    assert _sf_reasoning_prefill_mode(_FEATURES, None, _TPL, messages = _MSGS) is False


def test_thinking_on_is_prefilled():
    assert _generation_prompt_opens_think(_TPL, True, None, _MSGS) is True
    assert _sf_reasoning_prefill_mode(_FEATURES, True, _TPL, messages = _MSGS) is True
