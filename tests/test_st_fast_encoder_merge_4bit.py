# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fast-encoder save_pretrained_merged writes dense weights and leaves the live model's LoRA in place."""

import copy
import json
import os

import pytest
import torch

peft = pytest.importorskip("peft")
from transformers import BertConfig, BertModel


def _lora(model):
    config = peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["query", "value", "dense"])
    model = peft.get_peft_model(model, config)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.05)
    return model


def test_dense_copy_matches_peft_merge_and_keeps_the_adapters():
    from unsloth.models.sentence_transformer import _merged_dense_copy

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size = 64,
        hidden_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size = 64,
    )
    model = _lora(BertModel(config))
    reference = copy.deepcopy(model).merge_and_unload().state_dict()

    merged = _merged_dense_copy(model, torch.float32)

    assert type(merged) is BertModel
    assert merged.state_dict().keys() == reference.keys()
    for key, value in reference.items():
        torch.testing.assert_close(merged.state_dict()[key], value, msg = key)
    assert any("lora_" in name for name, _ in model.named_parameters())


def _cached(repo):
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_files_only = True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU for bitsandbytes 4-bit")
def test_4bit_fast_encoder_saves_a_dense_merge(tmp_path):
    pytest.importorskip("bitsandbytes")
    repo = "sentence-transformers/all-MiniLM-L6-v2"
    if not _cached(repo):
        pytest.skip(f"{repo} is not in the local Hugging Face cache")
    from sentence_transformers import SentenceTransformer
    from unsloth import FastSentenceTransformer

    model = FastSentenceTransformer.from_pretrained(
        repo, load_in_4bit = True, load_in_16bit = False, max_seq_length = 128
    )
    model = FastSentenceTransformer.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        target_modules = ["query", "key", "value", "dense"],
        use_gradient_checkpointing = False,
    )
    torch.manual_seed(0)
    with torch.no_grad():
        for name, param in model[0].auto_model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.05)
    model.eval()
    sentences = ["the cat sat on the mat", "unsloth makes finetuning faster"]
    live = model.encode(sentences, convert_to_tensor = True).float().cpu()

    out = str(tmp_path / "merged")
    model.save_pretrained_merged(out, model.tokenizer)

    with open(os.path.join(out, "config.json"), encoding = "utf-8") as f:
        assert "quantization_config" not in json.load(f)
    assert any("lora_" in name for name, _ in model[0].auto_model.named_parameters())
    after = model.encode(sentences, convert_to_tensor = True).float().cpu()
    torch.testing.assert_close(after, live)

    reloaded = SentenceTransformer(out, device = "cuda")
    assert not any(type(p).__name__ == "Params4bit" for p in reloaded[0].auto_model.parameters())
    embeddings = reloaded.encode(sentences, convert_to_tensor = True).float().cpu()
    torch.testing.assert_close(embeddings, live, atol = 1e-4, rtol = 1e-4)
