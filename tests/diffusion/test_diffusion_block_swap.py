# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""get_peft_model(offload_layers = ...) on text-diffusion models reaches the shared Block Swap install, and a
swapped model samples without CUDA Graphs. The swap itself needs CUDA; these check the wiring."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type = "toy", to_dict = lambda: {})
        self.layers = nn.ModuleList(nn.ModuleDict({"q_proj": nn.Linear(8, 8)}) for _ in range(3))

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = True


@pytest.fixture
def installs(monkeypatch):
    from unsloth.models import diffusion

    calls = []

    def fake_install(
        model,
        offload_layers = 0,
        prefetch_depth = 2,
        use_gradient_checkpointing = True,
    ):
        calls.append(
            dict(
                model = model,
                offload_layers = offload_layers,
                prefetch_depth = prefetch_depth,
                use_gradient_checkpointing = use_gradient_checkpointing,
            )
        )
        return "swapper"

    warnings = []
    monkeypatch.setattr(diffusion, "install_block_swap", fake_install)
    monkeypatch.setattr(diffusion.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
    return SimpleNamespace(calls = calls, warnings = warnings)


def _peft(model = None, **kwargs):
    from unsloth.models.diffusion import FastDiffusionModel
    kwargs.setdefault("use_gradient_checkpointing", False)
    return FastDiffusionModel.get_peft_model(
        model if model is not None else _Toy(), r = 4, target_modules = ["q_proj"], **kwargs
    )


@pytest.mark.parametrize("value", [2, "auto"])
def test_offload_layers_reaches_install_block_swap(installs, value):
    model = _peft(offload_layers = value, prefetch_depth = 3, use_gradient_checkpointing = True)
    (call,) = installs.calls
    assert call["model"] is model
    assert call["offload_layers"] == value
    assert call["prefetch_depth"] == 3
    assert call["use_gradient_checkpointing"] is True
    assert not any("offload_layers" in w for w in installs.warnings)
    # Installed after PEFT, so the LoRA weights exist and stay on the card.
    assert any("lora_A" in n for n, _ in call["model"].named_parameters())


def test_auto_keeps_the_embedding_on_the_card(installs):
    from unsloth.models.diffusion_profiles import unwrap_diffusion_model
    model = _peft(offload_layers = "auto", use_gradient_checkpointing = True)
    assert unwrap_diffusion_model(model)._unsloth_offload_embedding_mode is False


def test_legacy_name_and_off(installs):
    _peft(block_swap_layers = 1, use_gradient_checkpointing = True)
    assert installs.calls[-1]["offload_layers"] == 1
    installs.calls.clear()
    for off in (None, 0, False):
        _peft(offload_layers = off)
    assert installs.calls == []


@pytest.mark.parametrize("bad", [-1, 1.5, "all", True])
def test_bad_offload_layers_refused(installs, bad):
    with pytest.raises(ValueError, match = "offload_layers must be"):
        _peft(offload_layers = bad)
    assert installs.calls == []


def test_other_unsupported_options_still_warn(installs):
    _peft(offload_layers = 2, finetune_last_n_layers = 2, use_gradient_checkpointing = True)
    (warning,) = installs.warnings
    assert "finetune_last_n_layers" in warning and "offload_layers" not in warning


def test_existing_adapter_also_swaps(installs):
    model = _peft()
    again = _peft(model, offload_layers = 2, use_gradient_checkpointing = True)
    assert again is model and installs.calls[-1]["model"] is model


def test_without_cuda_the_shared_install_refuses_a_count_and_skips_auto(monkeypatch):
    # The real install_block_swap (no mock): "auto" swaps nothing off CUDA, a count says why it cannot.
    if torch.cuda.is_available():
        pytest.skip("checks the no-CUDA refusal")
    with pytest.raises(ValueError, match = "needs use_gradient_checkpointing"):
        _peft(offload_layers = 2, use_gradient_checkpointing = False)
    assert (
        getattr(
            _peft(offload_layers = "auto", use_gradient_checkpointing = True),
            "_unsloth_block_swap",
            None,
        )
        is None
    )
    with pytest.raises((ValueError, ImportError), match = "offload_layers needs"):
        _peft(offload_layers = 2, use_gradient_checkpointing = True)


def test_from_pretrained_offload_layers_warns_instead_of_dropping(monkeypatch, tmp_path):
    from unsloth.models import diffusion

    warnings = []
    monkeypatch.setattr(diffusion.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop

    monkeypatch.setattr(diffusion, "_load_diffusion_config", stop)
    with pytest.raises(Stop):
        diffusion.FastDiffusionModel.from_pretrained(
            str(tmp_path), offload_layers = 4, local_files_only = True
        )
    assert any("get_peft_model" in w for w in warnings)
    warnings.clear()
    with pytest.raises(Stop):
        diffusion.FastDiffusionModel.from_pretrained(str(tmp_path), local_files_only = True)
    assert warnings == []


def test_block_swapped_model_samples_eagerly(monkeypatch):
    from unsloth.models import diffusion_samplers

    class _Tok:
        eos_token_id = 2
        mask_token_id = 5

        def __call__(
            self,
            text,
            return_tensors = None,
            add_special_tokens = True,
        ):
            return {
                "input_ids": torch.tensor([[1, 7]]),
                "attention_mask": torch.ones(1, 2, dtype = torch.long),
            }

        def decode(
            self,
            tokens,
            skip_special_tokens = True,
        ):
            return " ".join(map(str, tokens))

    seen = []

    def fake(
        base,
        tokenizer,
        input_ids,
        attention_mask,
        compile_mode = None,
        **kwargs,
    ):
        seen.append(compile_mode)
        return torch.tensor([[3, 2]])

    monkeypatch.setitem(diffusion_samplers._SAMPLERS, "fake", fake)
    # As on a CUDA host, where "auto" picks CUDA Graph replay.
    monkeypatch.setattr(diffusion_samplers, "resolve_compile", lambda compile, device: "graphs")
    model = _Toy()
    model._unsloth_diffusion_profile = "fake"
    diffusion_samplers.generate_text(model, _Tok(), prompt = "hi", max_new_tokens = 2)
    model.layers._unsloth_block_swap = object()
    diffusion_samplers.generate_text(model, _Tok(), prompt = "hi", max_new_tokens = 2)
    assert seen == ["graphs", None]
