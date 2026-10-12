# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automatic GPU selection must size Offload layers runs by what stays on the card."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from core.training.training import TrainingBackend
from utils.hardware import DeviceType, auto_select_gpu_ids, estimate_required_model_memory_gb
import utils.hardware.hardware as _hw_module

_HW = "utils.hardware.hardware"

# unsloth/Qwen3-14B-unsloth-bnb-4bit's text config.
_QWEN3_14B = SimpleNamespace(
    hidden_size = 5120,
    num_hidden_layers = 40,
    num_attention_heads = 40,
    num_key_value_heads = 8,
    head_dim = 128,
    intermediate_size = 17408,
    vocab_size = 151936,
    tie_word_embeddings = False,
)

# RX 5700 XT 8 GB + RX 6500 XT 4 GB, free VRAM as torch reports it on an idle box.
_RX5700XT_RX6500XT = [
    {"index": 0, "vram_total_gb": 7.984, "vram_used_gb": 0.137},
    {"index": 1, "vram_total_gb": 3.984, "vram_used_gb": 0.137},
]


def _hardware_patches(stack, devices, estimate = None):
    stack.enter_context(patch(f"{_HW}.get_device", return_value = DeviceType.CUDA))
    stack.enter_context(patch(f"{_HW}.IS_ROCM", False))
    stack.enter_context(patch(f"{_HW}.rocm_gpu_ids_without_torch_kernels", return_value = set()))
    ids = [d["index"] for d in devices]
    stack.enter_context(
        patch(
            f"{_HW}._get_parent_visible_gpu_spec",
            return_value = {
                "raw": ",".join(map(str, ids)),
                "numeric_ids": ids,
                "supports_explicit_gpu_ids": True,
            },
        )
    )
    stack.enter_context(patch(f"{_HW}.get_parent_visible_gpu_ids", return_value = ids))
    stack.enter_context(patch(f"{_HW}.get_visible_gpu_count", return_value = len(ids)))
    stack.enter_context(
        patch(f"{_HW}.get_visible_gpu_utilization", return_value = {"devices": devices})
    )
    if estimate is not None:
        stack.enter_context(
            patch(f"{_HW}.estimate_required_model_memory_gb", return_value = estimate)
        )
    else:
        stack.enter_context(
            patch(
                f"{_HW}.estimate_fp16_model_size_bytes",
                return_value = (int(14.77e9 * 2), "config"),
            )
        )
        stack.enter_context(
            patch(f"{_HW}._resolve_model_identifier_for_gpu_estimate", side_effect = lambda m, **k: m)
        )
        stack.enter_context(
            patch(f"{_HW}._load_config_for_gpu_estimate", return_value = _QWEN3_14B)
        )
        stack.enter_context(
            patch(f"{_HW}._determine_attention_impl_for_gpu_estimate", return_value = "sdpa")
        )


def _mocked_estimate(required_gb, layer_gb, layers = 40):
    return (
        required_gb,
        {
            "required_gb": required_gb,
            "model_size_source": "config",
            "decoder_layers": layers,
            "decoder_layer_weights_gb": layer_gb,
        },
    )


class _ResetCaches:
    def tearDown(self):
        _hw_module._physical_gpu_count = None
        _hw_module._visible_gpu_count = None


class TestOffloadAwareAutoSelection(_ResetCaches, unittest.TestCase):
    def _select(self, devices, estimate = None, **kwargs):
        with ExitStack() as stack:
            _hardware_patches(stack, devices, estimate)
            return auto_select_gpu_ids(
                "unsloth/Qwen3-14B-unsloth-bnb-4bit",
                training_type = "LoRA/QLoRA",
                **kwargs,
            )

    def test_estimate_reports_decoder_layer_weights(self):
        with ExitStack() as stack:
            _hardware_patches(stack, _RX5700XT_RX6500XT)
            _, metadata = estimate_required_model_memory_gb(
                "unsloth/Qwen3-14B-unsloth-bnb-4bit", training_type = "LoRA/QLoRA"
            )
        self.assertEqual(metadata["decoder_layers"], 40)
        layers_gb = metadata["decoder_layers"] * metadata["decoder_layer_weights_gb"]
        self.assertLess(layers_gb, metadata["vram_breakdown"]["model_weights_gb"])
        self.assertGreater(layers_gb, 0.5 * metadata["vram_breakdown"]["model_weights_gb"])

    def test_offload_off_is_unchanged(self):
        selected, metadata = self._select(_RX5700XT_RX6500XT)
        self.assertEqual(selected, [0, 1])
        self.assertEqual(metadata["selection_mode"], "fallback_all")
        self.assertNotIn("offload_required_gb", metadata)

    def test_offload_count_that_fits_one_card_picks_one_card(self):
        selected, metadata = self._select(_RX5700XT_RX6500XT, offload_layers = 36)
        self.assertEqual(selected, [0])
        self.assertEqual(metadata["selection_mode"], "auto")
        self.assertLess(metadata["offload_required_gb"], metadata["required_gb"])

    def test_offload_auto_picks_one_card_when_no_split_holds_the_whole_model(self):
        selected, metadata = self._select(_RX5700XT_RX6500XT, offload_layers = "auto")
        self.assertEqual(selected, [0])
        self.assertEqual(metadata["selection_mode"], "auto")

    def test_offload_count_that_does_not_fit_one_card_still_splits(self):
        devices = [
            {"index": 0, "vram_total_gb": 12.0, "vram_used_gb": 0.0},
            {"index": 1, "vram_total_gb": 10.0, "vram_used_gb": 0.0},
        ]
        # 4 swapped minus a 3-slot pool frees one 0.2 GB layer: 13.8 > 12.
        selected, metadata = self._select(
            devices, _mocked_estimate(14.0, 0.2), offload_layers = 4
        )
        self.assertEqual(selected, [0, 1])
        self.assertEqual(metadata["selection_mode"], "auto")
        self.assertAlmostEqual(metadata["offload_required_gb"], 13.8, places = 3)

    def test_offload_auto_keeps_a_split_that_needs_no_offload(self):
        devices = [
            {"index": 0, "vram_total_gb": 12.0, "vram_used_gb": 0.0},
            {"index": 1, "vram_total_gb": 10.0, "vram_used_gb": 0.0},
        ]
        selected, _ = self._select(devices, _mocked_estimate(14.0, 0.2), offload_layers = "auto")
        self.assertEqual(selected, [0, 1])

    def test_offload_auto_that_does_not_fit_one_card_falls_back(self):
        devices = [
            {"index": 0, "vram_total_gb": 4.0, "vram_used_gb": 0.0},
            {"index": 1, "vram_total_gb": 4.0, "vram_used_gb": 0.0},
        ]
        # Floor: 14 - (39 - 3) * 0.2 = 6.8 > 4.
        selected, metadata = self._select(
            devices, _mocked_estimate(14.0, 0.2), offload_layers = "auto"
        )
        self.assertEqual(selected, [0, 1])
        self.assertEqual(metadata["selection_mode"], "fallback_all")
        self.assertAlmostEqual(metadata["offload_required_gb"], 6.8, places = 3)

    def test_offload_without_a_layer_estimate_is_unchanged(self):
        selected, metadata = self._select(
            _RX5700XT_RX6500XT,
            (14.0, {"required_gb": 14.0, "model_size_source": "config"}),
            offload_layers = 36,
        )
        self.assertEqual(selected, [0, 1])
        self.assertEqual(metadata["selection_mode"], "fallback_all")


class TestTrainingStartSizesOffload(_ResetCaches, unittest.TestCase):
    def _start(self, **kwargs):
        backend = TrainingBackend()

        class DummyProcess:
            pid = 12345

            def start(self):
                return None

        class DummyThread:
            def start(self):
                return None

        dummy_queue = object()
        with ExitStack() as stack:
            _hardware_patches(stack, _RX5700XT_RX6500XT)
            stack.enter_context(
                patch("core.training.training._CTX.Queue", side_effect = [dummy_queue, dummy_queue])
            )
            process = stack.enter_context(
                patch("core.training.training._CTX.Process", return_value = DummyProcess())
            )
            stack.enter_context(
                patch("core.training.training.threading.Thread", return_value = DummyThread())
            )
            backend.start_training(
                job_id = "test-job-offload",
                model_name = "unsloth/Qwen3-14B-unsloth-bnb-4bit",
                gpu_ids = None,
                **kwargs,
            )
        return process.call_args.kwargs["kwargs"]["config"]

    def test_offload_count_trains_on_one_card(self):
        config = self._start(training_type = "LoRA/QLoRA", offload_layers = 36)
        self.assertEqual(config["resolved_gpu_ids"], [0])

    def test_full_finetuning_ignores_offload(self):
        config = self._start(
            training_type = "Full Finetuning", load_in_4bit = False, offload_layers = 36
        )
        self.assertEqual(config["gpu_selection"]["selection_mode"], "fallback_all")
        self.assertNotIn("offload_required_gb", config["gpu_selection"])


if __name__ == "__main__":
    unittest.main()
