# Unsloth - 2x faster, 70% less memory LLM finetuning Tests for the `finetune_last_n_layers` parity knob (translation
# helper only, no CUDA / real checkpoint);
# mirrors unsloth-zoo's MLX layers_to_transform path.
from __future__ import annotations

import pytest


def test_get_total_transformer_layers_reads_num_hidden_layers():
    from unsloth.models.vision import _get_total_transformer_layers

    class FakeConfig:
        num_hidden_layers = 18

    class FakeModel:
        config = FakeConfig()

    assert _get_total_transformer_layers(FakeModel()) == 18


def test_get_total_transformer_layers_reads_text_config():
    from unsloth.models.vision import _get_total_transformer_layers

    class TextConfig:
        num_hidden_layers = 24

    class FakeConfig:
        text_config = TextConfig()

    class FakeModel:
        config = FakeConfig()

    assert _get_total_transformer_layers(FakeModel()) == 24


def test_get_total_transformer_layers_handles_alternative_attr_names():
    from unsloth.models.vision import _get_total_transformer_layers
    for attr in ("n_layer", "n_layers", "num_layers"):
        cfg = type("Cfg", (), {attr: 12})()
        model = type("M", (), {"config": cfg})()
        assert _get_total_transformer_layers(model) == 12


def test_get_total_transformer_layers_returns_none_when_unknown():
    from unsloth.models.vision import _get_total_transformer_layers

    class FakeConfig:
        pass

    class FakeModel:
        config = FakeConfig()

    assert _get_total_transformer_layers(FakeModel()) is None


def test_get_total_transformer_layers_returns_none_for_missing_config():
    from unsloth.models.vision import _get_total_transformer_layers
    class FakeModel:
        pass

    assert _get_total_transformer_layers(FakeModel()) is None


def test_finetune_last_n_layers_signature_present_on_llama_and_vision():
    """Both entry points expose finetune_last_n_layers with default None."""
    import inspect
    from unsloth.models.llama import FastLlamaModel
    from unsloth.models.vision import FastBaseModel

    for cls in (FastLlamaModel, FastBaseModel):
        sig = inspect.signature(cls.get_peft_model)
        assert (
            "finetune_last_n_layers" in sig.parameters
        ), f"{cls.__name__}.get_peft_model missing finetune_last_n_layers"
        assert sig.parameters["finetune_last_n_layers"].default is None, (
            f"{cls.__name__}.get_peft_model: finetune_last_n_layers default "
            f"must be None to preserve historical behavior"
        )


def _tiny_vlm(layers = 4):
    """Text decoder and vision tower with the same block count, like the tiny test VLMs."""
    import torch.nn as nn

    def stack():
        return nn.ModuleList(
            nn.ModuleDict({"self_attn": nn.ModuleDict({"q_proj": nn.Linear(4, 4)})})
            for _ in range(layers)
        )

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = stack()

    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = stack()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Language()
            self.visual = Visual()

    class VLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def get_decoder(self):
            return self.model.language_model

    return VLM()


def _lora_layers(target_modules):
    from peft import LoraConfig, inject_adapter_in_model

    config = LoraConfig(r = 2, target_modules = target_modules)
    model = inject_adapter_in_model(config, _tiny_vlm())
    return sorted(n.rsplit(".lora_A", 1)[0] for n, _ in model.named_modules() if n.endswith("lora_A"))


@pytest.mark.parametrize(
    "target_modules",
    [r"(?:.*?(?:visual|language).*?(?:self_attn).*?(?:q_proj).*?)", ["q_proj"]],
)
def test_last_n_layers_folds_into_target_modules(target_modules):
    """PEFT rejects layers_to_transform with a regex; folded in, only the text decoder's last layers train."""
    from unsloth.models.vision import _decoder_layers_path, _restrict_targets_to_layers

    path = _decoder_layers_path(_tiny_vlm(), 4)
    assert path == "model.language_model.layers"
    regex = _restrict_targets_to_layers(target_modules, path, [2, 3])
    assert _lora_layers(regex) == [
        "model.language_model.layers.2.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.q_proj",
    ]


def test_decoder_layers_path_needs_a_matching_layer_count():
    from unsloth.models.vision import _decoder_layers_path

    assert _decoder_layers_path(_tiny_vlm(), 5) is None
