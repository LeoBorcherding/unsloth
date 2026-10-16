# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Expected strings: transformers apply_chat_template renders from unsloth/mistral-7b-instruct-v0.3's
own template, hard coded for offline runs."""

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template


@pytest.fixture(autouse = True)
def _scratch_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _render(messages):
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    tokenizer = get_chat_template(
        PreTrainedTokenizerFast(
            tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
            bos_token = "<s>",
            eos_token = "</s>",
            unk_token = "<unk>",
        ),
        "mistral",
    )
    return tokenizer.apply_chat_template(messages, tokenize = False)


SYSTEM = {"role": "system", "content": "You are a terse bot."}
USER = {"role": "user", "content": "Hi there"}
REPLY = {"role": "assistant", "content": "Hello! How can I help?"}
USER2 = {"role": "user", "content": "Bye"}
REPLY2 = {"role": "assistant", "content": "Goodbye."}


@pytest.mark.parametrize(
    "messages, official",
    [
        ([USER, REPLY], "<s>[INST] Hi there[/INST] Hello! How can I help?</s>"),
        (
            [USER, REPLY, USER2, REPLY2],
            "<s>[INST] Hi there[/INST] Hello! How can I help?</s>[INST] Bye[/INST] Goodbye.</s>",
        ),
        (
            [SYSTEM, USER, REPLY, USER2],
            "<s>[INST] Hi there[/INST] Hello! How can I help?</s>"
            "[INST] You are a terse bot.\n\nBye[/INST]",
        ),
    ],
)
def test_renders_like_mistral_v03(messages, official):
    assert _render(messages) == official


def test_training_keeps_the_system_prompt_on_the_last_user_turn():
    # Mistral's template only shows the system prompt when the last message is a user turn;
    # dropping it from training data would lose it silently.
    assert _render([SYSTEM, USER, REPLY, USER2, REPLY2]) == (
        "<s>[INST] Hi there[/INST] Hello! How can I help?</s>"
        "[INST] You are a terse bot.\n\nBye[/INST] Goodbye.</s>"
    )
