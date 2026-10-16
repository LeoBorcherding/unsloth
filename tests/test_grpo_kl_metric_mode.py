# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO's kl / completion_length bucket must follow model.training, not control.should_evaluate."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"


def _metric_block():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"), filename = str(SOURCE_PATH))
    outer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "grpo_trainer_compute_loss"
    )
    compute_loss = next(
        node
        for node in ast.walk(outer)
        if isinstance(node, ast.FunctionDef) and node.name == "compute_loss"
    )
    found = [
        node
        for node in ast.walk(compute_loss)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Constant)
        and node.test.left.value == "train"
    ]
    assert len(found) == 1, f"expected one `if \"train\" in self._metrics` block, found {len(found)}"
    return compile(ast.Module(body = found, type_ignores = []), str(SOURCE_PATH), "exec")


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def _run(training, should_evaluate):
    trainer = SimpleNamespace(
        model = SimpleNamespace(training = training),
        control = SimpleNamespace(should_evaluate = should_evaluate),
        _metrics = {"train": defaultdict(list), "eval": defaultdict(list)},
    )
    exec(
        _metric_block(),
        {"self": trainer, "completion_length": _Scalar(7.0), "mean_kl": _Scalar(0.25)},
    )
    return trainer._metrics


@pytest.mark.parametrize(
    ("training", "should_evaluate", "bucket"),
    [
        (True, False, "train"),
        # Scheduled eval.
        (False, True, "eval"),
        # eval_on_start and a manual trainer.evaluate(): should_evaluate stays False.
        (False, False, "eval"),
    ],
)
def test_kl_and_completion_length_land_in_the_running_mode(training, should_evaluate, bucket):
    metrics = _run(training, should_evaluate)
    other = "eval" if bucket == "train" else "train"
    assert metrics[bucket]["kl"] == [0.25]
    assert metrics[bucket]["completion_length"] == [7.0]
    assert "kl" not in metrics[other] and "completion_length" not in metrics[other]
