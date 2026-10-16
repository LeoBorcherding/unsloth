# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A mapper row replaces the checkpoint's own chat template at training and inference. These
models use a different format from the template they were mapped to (Hermes-3 is ChatML, not
Llama-3.1; Phi-4-mini uses <|user|>/<|end|>, tokens phi-4's template doesn't have), so their own
template has to survive. A tiny local tokenizer; nothing is downloaded."""

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils.datasets.model_mappings import MODEL_TO_TEMPLATE_MAPPER  # noqa: E402

NATIVE_FORMAT_MODELS = [
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "unsloth/Hermes-3-Llama-3.1-70B",
    "allenai/Llama-3.1-Tulu-3-8B",
    "microsoft/Phi-4-mini-instruct",
    "unsloth/phi-4-mini-reasoning",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "unsloth/Mistral-Small-3.1-24B-Instruct-2503",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
]

PHI4_MINI_STYLE = (
    "{% for m in messages %}{{ '<|' + m.role + '|>' + m.content + '<|end|>' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "<|endoftext|>": 1, "<|user|>": 2, "<|end|>": 3, "<|assistant|>": 4}
    backend = Tokenizer(models.WordLevel(vocab, unk_token = "[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = backend, unk_token = "[UNK]", eos_token = "<|end|>", pad_token = "<|endoftext|>"
    )
    tokenizer.chat_template = PHI4_MINI_STYLE
    return tokenizer


@pytest.mark.parametrize("name", NATIVE_FORMAT_MODELS)
def test_a_model_with_its_own_format_is_not_mapped(name):
    assert name.lower() not in MODEL_TO_TEMPLATE_MAPPER
    assert name not in MODEL_TO_TEMPLATE_MAPPER


def test_training_keeps_phi4_minis_own_template():
    try:
        from utils.datasets.chat_templates import get_tokenizer_chat_template
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"dataset helpers unavailable ({type(exc).__name__}: {exc})")
    tokenizer = get_tokenizer_chat_template(_tokenizer(), "microsoft/Phi-4-mini-instruct")
    assert tokenizer.chat_template == PHI4_MINI_STYLE


def test_models_whose_format_matches_keep_their_rows():
    assert MODEL_TO_TEMPLATE_MAPPER["meta-llama/llama-3.1-8b-instruct"] == "llama-3.1"
    assert MODEL_TO_TEMPLATE_MAPPER["microsoft/phi-4"] == "phi-4"
    assert MODEL_TO_TEMPLATE_MAPPER["qwen/qwen3-4b-instruct-2507"] == "qwen3-instruct"
    assert MODEL_TO_TEMPLATE_MAPPER["mistralai/mistral-7b-instruct-v0.3"] == "mistral"
