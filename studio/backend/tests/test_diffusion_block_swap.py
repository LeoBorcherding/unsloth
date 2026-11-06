# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for Block Swap (offload_layers) on the DiT LoRA trainer: block discovery, the swap plan, placement,
compile skipping, config plumbing and the refusals. The swap itself needs CUDA copy streams, so it
is not run here."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from core.training import diffusion_dit_trainer as dit
from core.training import diffusion_train_common as dtc
from core.training.diffusion_train_common import (
    DiffusionLoraConfig,
    _config_from_dict,
    offload_layers_preflight_error,
)

FLUX = "black-forest-labs/FLUX.1-dev"


@pytest.fixture(autouse = True)
def _not_rocm(monkeypatch):
    monkeypatch.setattr(dit, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(dtc, "torch_is_rocm", lambda: False)


def _tiny_flux():
    from diffusers import FluxTransformer2DModel

    return FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 4,
        num_layers = 2,
        num_single_layers = 3,
        attention_head_dim = 8,
        num_attention_heads = 2,
        joint_attention_dim = 16,
        pooled_projection_dim = 8,
        axes_dims_rope = (2, 2, 4),
    )


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))


class _StandIn(nn.Module):
    _repeated_blocks = ["_Block"]

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([_Block() for _ in range(4)])
        self.misc = nn.ModuleList([nn.Linear(4, 4)])


def test_block_lists_on_a_tiny_flux_are_dual_then_single():
    t = _tiny_flux()
    lists = dit._dit_block_lists(t)
    assert [len(lst) for lst in lists] == [2, 3]
    assert lists[0] is t.transformer_blocks
    assert lists[1] is t.single_transformer_blocks


def test_block_lists_skip_lists_of_other_modules():
    t = _StandIn()
    assert dit._dit_block_lists(t) == [t.blocks]
    t._repeated_blocks = []
    assert dit._dit_block_lists(t) == []


def test_plan_spreads_over_the_whole_stack():
    t = _tiny_flux()
    lists = dit._dit_block_lists(t)
    plan = dit._plan_block_swap(lists, 2)
    # swap_indices(5, 2, "spread") = [1, 4]: dual block 1 and single block 2.
    assert [(lst is t.transformer_blocks, idx) for lst, idx in plan] == [(True, [1]), (False, [2])]
    plan = dit._plan_block_swap(lists, 5)
    assert [idx for _, idx in plan] == [[0, 1], [0, 1, 2]]
    # A list that gets no block is left out, so no empty BlockSwap is built.
    assert len(dit._plan_block_swap(lists, 1)) == 1


def test_plan_refuses_more_blocks_than_the_model_has():
    lists = dit._dit_block_lists(_tiny_flux())
    with pytest.raises(ValueError, match = "5 blocks"):
        dit._plan_block_swap(lists, 6)


def _second_device():
    # set_data refuses meta, so this needs a real second device: CUDA, or MPS on a Mac.
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("needs a CUDA or MPS device to move parameters to")


def test_place_unswapped_leaves_only_swapped_frozen_weights_on_host():
    dev = _second_device()
    t = _StandIn()
    t.requires_grad_(False)
    t.blocks[1].lin.bias.requires_grad_(True)  # stands in for a LoRA param inside a swapped block
    weight = t.embed.weight
    dit._place_unswapped(t, [t.blocks[1]], dev)
    # Same Parameter object, so an optimizer built earlier still holds it.
    assert t.embed.weight is weight
    assert t.embed.weight.device.type == dev.type
    assert t.blocks[0].lin.weight.device.type == dev.type
    assert t.blocks[1].lin.weight.device.type == "cpu"
    assert t.blocks[1].lin.bias.device.type == dev.type
    # Buffers are not swapped by BlockSwap, so they always go to the device.
    assert t.blocks[1].scale.device.type == dev.type


def test_compile_skips_swapped_blocks(monkeypatch):
    t = _StandIn()
    seen = []
    monkeypatch.setattr(_Block, "compile", lambda self, **kw: seen.append((self, kw)))
    assert dit._compile_blocks_except(t, [t.blocks[1], t.blocks[3]], dynamic = True)
    assert [m for m, _ in seen] == [t.blocks[0], t.blocks[2]]
    assert seen[0][1] == {"dynamic": True}
    seen.clear()
    assert not dit._compile_blocks_except(t, list(t.blocks))
    assert seen == []


def test_maybe_compile_routes_swapped_runs_through_the_skip(monkeypatch):
    t = _StandIn()
    calls = []
    t.compile_repeated_blocks = lambda **kw: calls.append("all")
    monkeypatch.setattr(dit, "_should_compile", lambda *a, **k: True)
    monkeypatch.setattr(
        dit,
        "_compile_blocks_except",
        lambda tr, skip, **kw: calls.append(("skip", len(skip))) or True,
    )
    cfg = DiffusionLoraConfig(base_model = FLUX, data_dir = "d", output_dir = "o")
    assert dit._maybe_compile_transformer(
        t, cfg, False, "cuda", None, "bf16", skip_blocks = [t.blocks[0]]
    )
    assert dit._maybe_compile_transformer(t, cfg, False, "cuda", None, "bf16")
    assert calls == [("skip", 1), "all"]


def _cfg(**kw):
    cfg = DiffusionLoraConfig(base_model = FLUX, data_dir = "d", output_dir = "o", **kw)
    return cfg.normalized()


def test_offload_layers_defaults_off_and_round_trips():
    assert _cfg().offload_layers == 0
    assert _cfg(offload_layers = 8).offload_layers == 8
    assert _cfg(offload_layers = 8, base_precision = "bf16").offload_layers == 8
    # The generic Unsloth config path can deliver numbers as strings.
    cfg = _config_from_dict(
        {"base_model": FLUX, "data_dir": "d", "output_dir": "o", "offload_layers": "6"}
    )
    assert cfg.normalized().offload_layers == 6


@pytest.mark.parametrize(
    "value, match",
    [
        (-1, ">= 0"),
        ("auto", "block count"),
        (True, "block count"),
        ("two", "whole number"),
        (2.5, "whole number"),
    ],
)
def test_offload_layers_bad_values_are_refused(value, match):
    with pytest.raises(ValueError, match = match):
        _cfg(offload_layers = value)


@pytest.mark.parametrize("precision", ["int8", "fp8", "mxfp8"])
def test_offload_layers_refuses_bases_without_plain_frozen_weights(precision):
    with pytest.raises(ValueError, match = "nf4 and bf16"):
        _cfg(offload_layers = 4, base_precision = precision)


def test_offload_layers_needs_gradient_checkpointing():
    with pytest.raises(ValueError, match = "gradient_checkpointing"):
        _cfg(offload_layers = 4, gradient_checkpointing = False)
    # Off stays valid without it.
    assert _cfg(offload_layers = 0, gradient_checkpointing = False).offload_layers == 0


@pytest.mark.parametrize(
    "base", ["stabilityai/stable-diffusion-xl-base-1.0", "MiniMaxAI/MiniMax-H3"]
)
def test_offload_layers_refused_outside_the_dit_trainer(base):
    with pytest.raises(ValueError, match = "not supported for"):
        DiffusionLoraConfig(
            base_model = base, data_dir = "d", output_dir = "o", offload_layers = 2
        ).normalized()


def test_preflight_is_silent_when_off():
    assert offload_layers_preflight_error(_cfg()) is None


def test_preflight_refuses_a_host_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert "CUDA or ROCm" in offload_layers_preflight_error(_cfg(offload_layers = 2))


def test_preflight_refuses_unified_memory(monkeypatch):
    class _Props:
        is_integrated = 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: _Props())
    assert "unified-memory" in offload_layers_preflight_error(_cfg(offload_layers = 2))


def test_preflight_passes_a_discrete_gpu(monkeypatch):
    class _Props:
        is_integrated = 0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: _Props())
    pytest.importorskip("unsloth_zoo.block_swap")
    assert offload_layers_preflight_error(_cfg(offload_layers = 2)) is None


def test_auto_precision_never_lands_on_a_torchao_mode_when_offloading(monkeypatch):
    monkeypatch.setattr(dit, "_pick_auto_precision", lambda *a, **k: "int8")
    spec = dit._SPECS["flux.1"]
    swapped = _cfg(base_precision = "auto", offload_layers = 4)
    assert dit._resolve_base_precision(swapped, spec, "cuda") == "bf16"
    assert dit._resolve_base_precision(_cfg(base_precision = "auto"), spec, "cuda") == "int8"
    monkeypatch.setattr(dit, "_pick_auto_precision", lambda *a, **k: "nf4")
    assert dit._resolve_base_precision(swapped, spec, "cuda") == "nf4"


def test_bf16_base_stays_on_host_when_offloading():
    class _Model(nn.Module):
        def to(self, *a, **k):
            raise AssertionError("moved to device")

    class _Cls:
        @staticmethod
        def from_pretrained(*a, **k):
            return _Model()

    cfg = _cfg(base_precision = "bf16", offload_layers = 4)
    on_host = dit._load_dit_transformer(_Cls, cfg, "cuda", "bf16")
    assert isinstance(on_host, _Model)
    with pytest.raises(AssertionError, match = "moved"):
        dit._load_dit_transformer(_Cls, _cfg(base_precision = "bf16"), "cuda", "bf16")


def test_request_model_carries_offload_layers():
    from pydantic import ValidationError

    from models.training import DiffusionTrainingStartRequest

    paths = {"base_model": FLUX, "data_dir": "d", "output_dir": "o"}
    req = DiffusionTrainingStartRequest(**paths, offload_layers = 12)
    assert req.model_dump()["offload_layers"] == 12
    assert DiffusionTrainingStartRequest(**paths).offload_layers == 0
    with pytest.raises(ValidationError):
        DiffusionTrainingStartRequest(**paths, offload_layers = -1)
