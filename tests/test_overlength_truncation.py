# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An input longer than max_seq_length is truncated, so the run must match the pre-truncated one.

The warning used to be followed by a shape mismatch: seq_length, position_ids and labels kept
the full length while input_ids were cut.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Unsloth's fast forward needs an accelerator", allow_module_level = True)

from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

from unsloth.models.llama import CausalLM_fast_forward, FastLlamaModel
from unsloth.models.mistral import MistralForCausalLM_fast_forward

MAX_SEQ_LENGTH = 8
FULL_LENGTH = 16
VOCAB_SIZE = 128


def _tiny():
    FastLlamaModel.pre_patch()
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 1,
            num_attention_heads = 4,
            num_key_value_heads = 4,
            vocab_size = VOCAB_SIZE,
            max_position_embeddings = 64,
        )
    )
    model = model.to("cuda", torch.float16)
    model.max_seq_length = MAX_SEQ_LENGTH
    model.model.max_seq_length = MAX_SEQ_LENGTH
    return model


def _batch():
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, FULL_LENGTH), generator = generator).cuda()
    # Offset from the default 0..n so a wrong slice of them shows up in the output.
    position_ids = torch.arange(FULL_LENGTH, device = "cuda").repeat(2, 1) + 5
    return input_ids, position_ids


@pytest.mark.parametrize("with_position_ids", [False, True], ids = ["no_pos", "pos"])
def test_overlength_training_batch_matches_the_truncated_one(with_position_ids):
    model = _tiny().train()
    input_ids, position_ids = _batch()
    extra = {"position_ids": position_ids} if with_position_ids else {}
    cut = {k: v[:, :MAX_SEQ_LENGTH] for k, v in extra.items()}

    with torch.no_grad():
        full = model(input_ids = input_ids, labels = input_ids, **extra)
        short = model(
            input_ids = input_ids[:, :MAX_SEQ_LENGTH],
            labels = input_ids[:, :MAX_SEQ_LENGTH],
            **cut,
        )
    assert torch.isfinite(full.loss)
    assert full.loss.item() == pytest.approx(short.loss.item(), rel = 1e-3)


def test_overlength_eval_with_mask_matches_the_truncated_one():
    model = _tiny().eval()
    input_ids, position_ids = _batch()
    # Left padding inside the kept window, so the 4D SDPA mask is actually built.
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :2] = 0

    with torch.no_grad():
        full = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
        )
        short = model(
            input_ids = input_ids[:, :MAX_SEQ_LENGTH],
            attention_mask = attention_mask[:, :MAX_SEQ_LENGTH],
            position_ids = position_ids[:, :MAX_SEQ_LENGTH],
        )
    assert full.logits.shape[1] == MAX_SEQ_LENGTH
    torch.testing.assert_close(full.logits, short.logits, atol = 1e-3, rtol = 1e-2)


class _TruncatingCausalLM:
    """The decoder stack already cut the batch to max_seq_length; only the head and loss run."""

    training = True

    def __init__(self):
        self.config = MistralConfig(
            hidden_size = 4,
            intermediate_size = 8,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            vocab_size = VOCAB_SIZE,
        )
        generator = torch.Generator().manual_seed(0)
        hidden_states = torch.randn(2, MAX_SEQ_LENGTH, 4, generator = generator).cuda()
        self.model = lambda **kwargs: BaseModelOutputWithPast(last_hidden_state = hidden_states)
        self.lm_head = torch.nn.Linear(4, VOCAB_SIZE, bias = False).cuda()


@pytest.mark.parametrize(
    "forward",
    [CausalLM_fast_forward(None), MistralForCausalLM_fast_forward],
    ids = ["llama", "mistral"],
)
def test_labels_follow_the_truncated_hidden_states(forward, monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)
    input_ids, _ = _batch()
    model = _TruncatingCausalLM()
    full = forward(model, input_ids = input_ids, labels = input_ids, return_dict = True)
    short = forward(
        model,
        input_ids = input_ids[:, :MAX_SEQ_LENGTH],
        labels = input_ids[:, :MAX_SEQ_LENGTH],
        return_dict = True,
    )
    assert full.loss.item() == pytest.approx(short.loss.item(), rel = 1e-6)
