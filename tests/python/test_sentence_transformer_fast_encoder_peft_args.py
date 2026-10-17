"""FastSentenceTransformer.get_peft_model on the fast-encoder route forwards the LoRA options
the non-fast route forwards (rsLoRA, modules_to_save, layer selection, random_state)."""

import torch
from sentence_transformers import SentenceTransformer
from transformers import BertConfig, BertModel

import unsloth  # noqa: F401
from unsloth.models.sentence_transformer import FastSentenceTransformer


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig(
            vocab_size = 64,
            hidden_size = 16,
            intermediate_size = 32,
            num_hidden_layers = 2,
            num_attention_heads = 2,
        )
        self.auto_model = BertModel(config)

    def save(self, output_path, *args, **kwargs):
        pass


def _fast_encoder():
    model = SentenceTransformer(modules = [_Encoder()], device = "cpu")
    model._unsloth_fast_encoder = True
    model._compile_mode = None
    return model


def _peft(**kwargs):
    model = FastSentenceTransformer.get_peft_model(
        _fast_encoder(), r = 4, target_modules = ["query", "value"], **kwargs
    )
    return model[0].auto_model


def test_lora_options_reach_the_config():
    peft_model = _peft(
        use_rslora = True,
        modules_to_save = ["pooler"],
        layers_to_transform = [1],
        layers_pattern = "layer",
        lora_dropout = 0.0,
    )
    cfg = peft_model.peft_config["default"]
    assert cfg.use_rslora
    assert "pooler" in cfg.modules_to_save
    assert cfg.layers_to_transform == [1]
    adapted = {n for n, _ in peft_model.named_modules() if n.endswith("lora_A.default")}
    assert adapted and all(".layer.1." in n for n in adapted)


def test_random_state_makes_the_init_reproducible():
    torch.manual_seed(0)
    a = _peft(random_state = 1234)
    torch.manual_seed(1)
    b = _peft(random_state = 1234)
    a_weights = {n: p for n, p in a.named_parameters() if "lora_A" in n}
    b_weights = dict((n, p) for n, p in b.named_parameters() if "lora_A" in n)
    assert a_weights
    for name, weight in a_weights.items():
        torch.testing.assert_close(weight, b_weights[name])
