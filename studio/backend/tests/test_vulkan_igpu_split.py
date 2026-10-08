# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Vulkan pin that mixes a discrete card with a shared-memory iGPU.

The iGPU reports its whole shared pool as free, so it outranked the discrete card in
``_select_gpus`` and, with no ``--tensor-split``, took the larger share of llama.cpp's
free-memory layer split. Reported on an RX 7700 XT (12 GB) + Ryzen iGPU, Windows:
Qwen3.8-27B UD-IQ4_XS loaded 4.8 GB on the card and 8.3 GB on the iGPU, 1.05 t/s,
against 4.4 t/s with the iGPU deselected.
"""

from __future__ import annotations

import inspect
import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import importlib as _importlib  # noqa: E402


def _maybe_stub(name: str, builder):
    try:
        _importlib.import_module(name)
    except ImportError:
        sys.modules[name] = builder()


def _build_loggers_stub():
    m = _types.ModuleType("loggers")
    m.get_logger = lambda name: __import__("logging").getLogger(name)
    return m


_maybe_stub("loggers", _build_loggers_stub)
_maybe_stub("structlog", lambda: _types.ModuleType("structlog"))

from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    _inherited_layer_tensor_split,
)

MIB = 1024 * 1024

# The reporter's probe: VK0 the RX 7700 XT, VK1 the iGPU after its host reserve.
DGPU, IGPU = 0, 1
GPUS = [(DGPU, 11313), (IGPU, 14352)]
SHARED = {IGPU}


def test_a_model_either_device_holds_lands_on_the_discrete_card():
    picked, use_fit = LlamaCppBackend._select_gpus(
        6000 * MIB, GPUS, usable_fraction = 0.9, shared_gpu_ids = SHARED
    )
    assert (picked, use_fit) == ([DGPU], False)


def test_without_shared_ids_the_larger_shared_pool_still_wins():
    # The old ranking, kept for every caller that has no shared set (CUDA, ROCm).
    picked, _ = LlamaCppBackend._select_gpus(6000 * MIB, GPUS, usable_fraction = 0.9)
    assert picked == [IGPU]


def test_a_model_that_needs_both_still_pins_both():
    picked, use_fit = LlamaCppBackend._select_gpus(
        15000 * MIB, GPUS, usable_fraction = 0.9, shared_gpu_ids = SHARED
    )
    assert (picked, use_fit) == ([DGPU, IGPU], False)


def test_split_aware_passes_the_shared_set_through():
    picked, _ = LlamaCppBackend._select_gpus_split_aware(
        6000 * MIB,
        GPUS,
        usable_fraction = 0.9,
        split_extra_bytes = 256 * MIB,
        shared_gpu_ids = SHARED,
    )
    assert picked == [DGPU]


def test_the_split_fills_the_discrete_card_first():
    # 13.26 GiB of weights + ~0.6 GiB of KV at 32K on iq4_nl.
    shares = LlamaCppBackend._discrete_first_split(
        [DGPU, IGPU],
        {DGPU: 10180.0, IGPU: 12917.0},
        SHARED,
        layered_mib = 14200.0,
        per_device_mib = 300.0,
    )
    assert shares == [9880.0, 4320.0]
    assert shares[0] > shares[1]


def test_nothing_left_over_gives_the_igpu_nothing():
    shares = LlamaCppBackend._discrete_first_split(
        [DGPU, IGPU], {DGPU: 10000.0, IGPU: 12000.0}, SHARED, layered_mib = 8000.0
    )
    assert shares == [8000.0, 0.0]


def test_the_split_is_positional_over_the_pin_order():
    shares = LlamaCppBackend._discrete_first_split(
        [IGPU, DGPU], {DGPU: 10000.0, IGPU: 12000.0}, SHARED, layered_mib = 12000.0
    )
    assert shares == [2000.0, 10000.0]


def test_overflow_is_shared_across_igpus_by_room():
    shares = LlamaCppBackend._discrete_first_split(
        [0, 1, 2], {0: 4000.0, 1: 3000.0, 2: 1000.0}, {1, 2}, layered_mib = 6000.0
    )
    assert shares == [4000.0, 1500.0, 500.0]


def test_only_a_mixed_pin_gets_a_split():
    usable = {0: 10000.0, 1: 12000.0}
    assert LlamaCppBackend._discrete_first_split([0, 1], usable, set(), 15000.0) is None
    assert LlamaCppBackend._discrete_first_split([0, 1], usable, {0, 1}, 15000.0) is None
    assert LlamaCppBackend._discrete_first_split([0, 1], usable, {1}, 0.0) is None


def test_the_launch_emits_it_only_where_nothing_else_owns_the_split():
    src = inspect.getsource(LlamaCppBackend.load_model)
    arm = src[src.index("_mixed_split = (") : src.index("if _mixed_split is not None:")]
    assert "not tensor_parallel" in arm
    assert "_TENSOR_SPLIT_FLAGS" in arm
    assert "_spill_inputs is not None" in arm
    assert "_inherited_layer_tensor_split(os.environ)" in arm


def test_an_inherited_layer_split_keeps_its_env_share():
    # The launch only scrubs LLAMA_ARG_TENSOR_SPLIT with a non-layer split mode, so in
    # layer mode (or unset) it reaches the child, and a CLI --tensor-split would win.
    assert _inherited_layer_tensor_split({"LLAMA_ARG_TENSOR_SPLIT": "3,1"})
    assert _inherited_layer_tensor_split(
        {"LLAMA_ARG_SPLIT_MODE": " Layer ", "LLAMA_ARG_TENSOR_SPLIT": "3,1"}
    )
    assert not _inherited_layer_tensor_split(
        {"LLAMA_ARG_SPLIT_MODE": "row", "LLAMA_ARG_TENSOR_SPLIT": "3,1"}
    )
    assert not _inherited_layer_tensor_split({"LLAMA_ARG_SPLIT_MODE": "layer"})
    assert not _inherited_layer_tensor_split({"LLAMA_ARG_TENSOR_SPLIT": "  "})


def test_auto_context_ranks_the_igpu_after_the_discrete_card():
    src = inspect.getsource(LlamaCppBackend.load_model)
    auto = src[src.index("# Auto context: prefer fewer GPUs") :]
    ranked = auto[auto.index("ranked = sorted(") : auto.index("_pipeline_overhead_mib")]
    assert "not in _shared_gpu_ids" in ranked


def test_a_second_discrete_card_keeps_the_layer_split_overhead():
    # _select_gpus charges the per-device pipeline overhead for every card after the
    # first, so the share must leave it free on the second discrete card.
    shares = LlamaCppBackend._discrete_first_split(
        [0, 1, 2],
        {0: 10000.0, 1: 8000.0, 2: 12000.0},
        {2},
        layered_mib = 20000.0,
        per_device_mib = 300.0,
        pipeline_mib = 1024.0,
    )
    assert shares == [9700.0, 6676.0, 3624.0]


def test_igpu_overflow_is_apportioned_after_their_reserves():
    # Each iGPU is an extra device: weighting by raw usable left the small one 540 MiB
    # short of the 1324 MiB the selector charged it.
    shares = LlamaCppBackend._discrete_first_split(
        [0, 1, 2],
        {0: 10000.0, 1: 8000.0, 2: 2000.0},
        {1, 2},
        layered_mib = 14000.0,
        per_device_mib = 300.0,
        pipeline_mib = 1024.0,
    )
    assert shares[0] == 9700.0
    assert shares[2] <= 2000.0 - 1324.0
    assert abs(sum(shares) - 14000.0) < 1e-6


def test_the_mtp_preflight_ranks_the_igpu_last_too():
    src = inspect.getsource(LlamaCppBackend.load_model)
    probe = src[src.index("def _probe_rank(") : src.index("_probe_overhead_mib =")]
    assert "not in _shared_gpu_ids" in probe


def test_the_first_discrete_card_keeps_the_flat_compute_buffer():
    # model_size_fit counts the flat compute buffer (5 GiB when dims are unknown) once;
    # filling the card to usable - per-device left none of it free under --fit off.
    shares = LlamaCppBackend._discrete_first_split(
        [DGPU, IGPU],
        {DGPU: 10180.0, IGPU: 12917.0},
        SHARED,
        layered_mib = 14200.0,
        per_device_mib = 0.0,
        pipeline_mib = 1024.0,
        first_mib = 5120.0,
    )
    assert shares == [5060.0, 9140.0]


def test_the_launch_passes_the_flat_buffer_and_extras():
    src = inspect.getsource(LlamaCppBackend.load_model)
    arm = src[src.index("_mixed_split = (") : src.index("if _mixed_split is not None:")]
    for term in ("compute_buffer_flat", "soft_overhead", "extra_gpu_bytes"):
        assert term in arm, term
    # layered_mib comes from model_size, which already holds a GPU-resident projector.
    assert "- (mmproj_size or 0)" in arm
    assert "(_shared_pool_mmproj or 0) / (1024 * 1024)," in arm


def test_a_shared_pool_projector_comes_off_the_igpu_room_once():
    # A CPU-pinned projector sits once in the host pool the iGPUs allocate from.
    shares = LlamaCppBackend._discrete_first_split(
        [0, 1, 2],
        {0: 4000.0, 1: 3000.0, 2: 3000.0},
        {1, 2},
        layered_mib = 6000.0,
        shared_pool_mib = 1000.0,
    )
    assert shares == [4000.0, 1000.0, 1000.0]


def test_a_split_that_overflows_the_igpu_rooms_is_declined():
    # 1976 MiB of iGPU room after reserves and the projector, 3000 MiB left to place.
    assert (
        LlamaCppBackend._discrete_first_split(
            [0, 1, 2],
            {0: 4000.0, 1: 3000.0, 2: 3000.0},
            {1, 2},
            layered_mib = 7000.0,
            pipeline_mib = 1024.0,
            shared_pool_mib = 1000.0,
        )
        is None
    )


def test_an_igpu_that_holds_it_alone_is_used_when_the_card_cannot_join():
    # 500 + 10000 MiB is short of 9500 + the 1024 MiB split overhead; the iGPU alone fits.
    picked, use_fit = LlamaCppBackend._select_gpus(
        9500 * MIB,
        [(DGPU, 500), (IGPU, 10000)],
        usable_fraction = 1.0,
        per_device_overhead_bytes = 1024 * MIB,
        shared_gpu_ids = SHARED,
    )
    assert (picked, use_fit) == ([IGPU], False)


def test_the_flat_buffer_is_reserved_on_the_pins_first_device():
    # llama.cpp books the flat buffer on device 0, here the iGPU the pin lists first.
    shares = LlamaCppBackend._discrete_first_split(
        [IGPU, DGPU],
        {DGPU: 4000.0, IGPU: 4000.0},
        SHARED,
        layered_mib = 7000.0,
        first_mib = 1000.0,
    )
    assert shares == [3000.0, 4000.0]


def test_two_igpus_reporting_one_shared_heap_are_credited_once():
    # A 4 GiB card plus two iGPUs that each report the same 8 GiB pool: 12 GiB, not 20.
    gpus = [(0, 4096), (1, 8192), (2, 8192)]
    picked, use_fit = LlamaCppBackend._select_gpus(
        16000 * MIB, gpus, usable_fraction = 1.0, shared_gpu_ids = {1, 2}
    )
    assert (picked, use_fit) == (None, True)
    assert (
        LlamaCppBackend._discrete_first_split(
            [0, 1, 2], {0: 4096.0, 1: 8192.0, 2: 8192.0}, {1, 2}, layered_mib = 16000.0
        )
        is None
    )


def test_a_dgpu_after_a_leading_igpu_keeps_its_pipeline_reserve():
    # Pipeline scratch lands on every device after the pin's first, here the dGPU.
    shares = LlamaCppBackend._discrete_first_split(
        [IGPU, DGPU],
        {DGPU: 4000.0, IGPU: 4000.0},
        SHARED,
        layered_mib = 6000.0,
        pipeline_mib = 1024.0,
    )
    assert shares == [3024.0, 2976.0]


def test_an_inherited_projector_is_reserved_on_the_first_device():
    # LLAMA_ARG_MMPROJ loads a projector the spill planner books on device 0.
    src = inspect.getsource(LlamaCppBackend.load_model)
    arm = src[src.index("_mixed_split = (") : src.index("if _mixed_split is not None:")]
    assert '_spill_inputs.get("env_mmproj_bytes")' in arm
    assert 'not _spill_inputs.get("env_mmproj_unsized")' in arm
