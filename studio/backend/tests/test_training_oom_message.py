# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GPU OOM reaches the user as the guided message wherever it lands, not as raw torch text."""

import inspect

import pytest

from core.training import worker

BUDGET = {"offload_vram_gb": 12}


@pytest.mark.parametrize(
    "text",
    [
        "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.99 GiB total capacity)",
        "HIP out of memory. Tried to allocate 1.50 GiB",
        "torch.OutOfMemoryError: allocation failed",
    ],
)
def test_an_oom_error_text_gets_the_guided_message(text):
    message = worker._oom_error(BUDGET, text)
    assert message.startswith("GPU ran out of VRAM during training.")
    assert "reduce max_seq_length" in message
    assert "capped at a 12 GiB VRAM budget" in message


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "Failed to load dataset",
        "CUDA error: device-side assert",
        "out of memory",  # str(MemoryError("out of memory")) from a host-RAM failure
        "DefaultCPUAllocator: can't allocate memory: out of memory",
    ],
)
def test_other_errors_are_left_alone(text):
    assert worker._oom_error(BUDGET, text) is None


def test_every_error_text_the_worker_relays_goes_through_the_oom_check():
    # _train_worker stores str(e) in progress.error and load_model returns False, so the
    # except-branch check never sees those; each relay site has to ask itself.
    source = inspect.getsource(worker.run_training_process)
    assert '"error": _oom_error(config, progress.error) or progress.error' in source
    assert "_oom_error(config, load_error)" in source
    # LoRA prepare, and dataset load (audio preprocessing puts its codec on the GPU).
    assert source.count("_oom_error(config, trainer.training_progress.error)") == 2
