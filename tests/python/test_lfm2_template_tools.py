# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""lfm-2 / lfm-2.5 must render tools, tool calls and tool results the way LiquidAI's own templates do."""

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template


TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
TOOL_CHAT = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
    },
    {"role": "tool", "content": '{"temp": 20}'},
    {"role": "assistant", "content": "It is 20C in Paris."},
]
PLAIN = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Bye"},
]

_TOOL_SCHEMA = (
    '[{"type": "function", "function": {"name": "get_weather", "description": "Current weather.", '
    '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]'
)
_TURNS = (
    "<|im_start|>user\nWeather in Paris?<|im_end|>\n"
    "<|im_start|>assistant\n<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|><|im_end|>\n"
    "<|im_start|>tool\n{response}<|im_end|>\n"
    "<|im_start|>assistant\nIt is 20C in Paris.<|im_end|>\n"
)
_PLAIN = (
    "<|startoftext|><|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
    "<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n"
)

# Rendered offline by LiquidAI/LFM2-1.2B@40f3da0 and LiquidAI/LFM2.5-1.2B-Instruct@df58c17.
EXPECTED = {
    "lfm-2": {
        "tools": "<|startoftext|><|im_start|>system\nList of tools: <|tool_list_start|>" + _TOOL_SCHEMA
        + "<|tool_list_end|><|im_end|>\n"
        + _TURNS.format(response = '<|tool_response_start|>{"temp": 20}<|tool_response_end|>'),
        "tools_system": "<|startoftext|><|im_start|>system\nBe brief.\nList of tools: <|tool_list_start|>"
        + _TOOL_SCHEMA + "<|tool_list_end|><|im_end|>\n"
        + _TURNS.format(response = '<|tool_response_start|>{"temp": 20}<|tool_response_end|>'),
        "plain": _PLAIN,
    },
    "lfm-2.5": {
        "tools": "<|startoftext|><|im_start|>system\nList of tools: " + _TOOL_SCHEMA + "<|im_end|>\n"
        + _TURNS.format(response = '{"temp": 20}'),
        "tools_system": "<|startoftext|><|im_start|>system\nBe brief.\nList of tools: " + _TOOL_SCHEMA
        + "<|im_end|>\n" + _TURNS.format(response = '{"temp": 20}'),
        "plain": _PLAIN,
    },
}


@pytest.fixture(autouse = True)
def _scratch_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _tokenizer(name):
    vocab = {"<unk>": 0, "<|startoftext|>": 1, "<|im_end|>": 2, "<|pad|>": 3}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
        bos_token = "<|startoftext|>",
        eos_token = "<|im_end|>",
        unk_token = "<unk>",
        pad_token = "<|pad|>",
    )
    return get_chat_template(tokenizer, name)


@pytest.mark.parametrize("name", ["lfm-2", "lfm-2.5"])
def test_tool_calling_conversation_renders_like_liquidai(name):
    tokenizer = _tokenizer(name)
    rendered = tokenizer.apply_chat_template(TOOL_CHAT, tools = TOOLS, tokenize = False)
    assert rendered == EXPECTED[name]["tools"]


@pytest.mark.parametrize("name", ["lfm-2", "lfm-2.5"])
def test_tools_join_the_system_prompt(name):
    tokenizer = _tokenizer(name)
    messages = [{"role": "system", "content": "Be brief."}] + TOOL_CHAT
    rendered = tokenizer.apply_chat_template(messages, tools = TOOLS, tokenize = False)
    assert rendered == EXPECTED[name]["tools_system"]


@pytest.mark.parametrize("name", ["lfm-2", "lfm-2.5"])
def test_plain_chat_is_unchanged(name):
    tokenizer = _tokenizer(name)
    rendered = tokenizer.apply_chat_template(PLAIN, tokenize = False, add_generation_prompt = True)
    assert rendered == EXPECTED[name]["plain"]
