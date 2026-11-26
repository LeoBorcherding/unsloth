# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL >= 1.15's SFTTrainer refuses a LoRA lm_head; Unsloth's SFT loss trains one, so only dft keeps the refusal.
ast-loaded, CPU-only."""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import types
from pathlib import Path

import pytest

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"

# TRL 1.15.0 to 1.19.0 SFTTrainer.__init__, verbatim around the guard.
TRL_LM_HEAD_GUARD = """\
    def __init__(self, model, args):
        if is_peft_model(model) and isinstance(model.get_output_embeddings(), BaseTunerLayer):
            # The log-probabilities are computed by multiplying the hidden states by `lm_head.weight` directly, so an
            # adapter on the LM head would be ignored and never trained.
            raise ValueError(
                "Applying a PEFT adapter to `lm_head` is not supported: the log-probabilities are computed from "
                "`lm_head.weight` directly, so the adapter would never be trained. Remove `'lm_head'` from your "
                "`target_modules`."
            )
"""


def _patched(function):
    # The rl_replacements edit alone: its siblings read module globals this ast load does not bring in.
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "sft_trainer_allow_lm_head_adapter"
    ]
    if not nodes:
        return function
    assert 'RL_FUNCTIONS["sft_trainer"].append(sft_trainer_allow_lm_head_adapter)' in ast.unparse(
        tree
    ).replace("'", '"')
    namespace = {}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace["sft_trainer_allow_lm_head_adapter"]("__init__", function)


class BaseTunerLayer:
    pass


class _Model:
    def get_output_embeddings(self):
        return BaseTunerLayer()


def _init(source):
    namespace = {"is_peft_model": lambda m: True, "BaseTunerLayer": BaseTunerLayer}
    exec(textwrap.dedent(source), namespace)
    return lambda loss_type: namespace["__init__"](
        None, _Model(), types.SimpleNamespace(loss_type = loss_type)
    )


@pytest.fixture
def zoo_trains_lm_head_adapters(monkeypatch):
    module = types.ModuleType("unsloth_zoo.fused_losses.cross_entropy_loss")
    module.lm_head_needs_module_forward = lambda head: True
    monkeypatch.setitem(sys.modules, "unsloth_zoo.fused_losses.cross_entropy_loss", module)


def test_nll_trains_a_lora_lm_head(zoo_trains_lm_head_adapters):
    # target_modules = r".*(q_proj|v_proj|lm_head)" on TRL 1.15.0 to 1.19.0 raised in UnslothSFTTrainer.__init__.
    _init(_patched(TRL_LM_HEAD_GUARD))("nll")


def test_dft_keeps_trls_refusal(zoo_trains_lm_head_adapters):
    # dft runs on TRL's fused head, which reads lm_head.weight and would never train the adapter.
    with pytest.raises(ValueError, match = "lm_head"):
        _init(_patched(TRL_LM_HEAD_GUARD))("dft")


def test_older_zoo_keeps_trls_refusal(monkeypatch):
    # A zoo whose fused CE projects with lm_head.weight would leave the adapter untrained.
    monkeypatch.setitem(
        sys.modules,
        "unsloth_zoo.fused_losses.cross_entropy_loss",
        types.ModuleType("unsloth_zoo.fused_losses.cross_entropy_loss"),
    )
    assert _patched(TRL_LM_HEAD_GUARD) == TRL_LM_HEAD_GUARD


def test_guard_text_matches_installed_trl():
    # Read the file: an imported Unsloth may already have swapped trl's SFTTrainer.
    import importlib.util

    # find_spec on a dotted name imports the parent, and raises without it.
    spec = importlib.util.find_spec("trl") and importlib.util.find_spec("trl.trainer.sft_trainer")
    if spec is None or spec.origin is None:
        pytest.skip(reason = "trl is not installed")
    source = Path(spec.origin).read_text(encoding = "utf-8")
    if "BaseTunerLayer)" not in source:
        pytest.skip(reason = "this TRL has no lm_head adapter guard")
    assert (
        "if is_peft_model(model) and isinstance(model.get_output_embeddings(), BaseTunerLayer):"
        in source
    )


def test_only_dft_reaches_trls_fused_head():
    # The narrowed guard relies on this: nll and label smoothing go through super().compute_loss, so
    # transformers' Trainer runs the model forward and zoo's fused CE, never TRL's fused_lm_head.
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    outer = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "sft_trainer_compute_loss"
    )
    compute_loss = next(
        n for n in ast.walk(outer) if isinstance(n, ast.FunctionDef) and n.name == "compute_loss"
    )
    calls = []

    def visit(node, under_dft):
        if isinstance(node, ast.If):
            under_dft = under_dft or "dft" in ast.unparse(node.test)
        if isinstance(node, ast.Call) and any(k.arg == "fused_lm_head" for k in node.keywords):
            calls.append(under_dft)
        for child in ast.iter_child_nodes(node):
            visit(child, under_dft)

    visit(compute_loss, False)
    assert calls and all(calls)
