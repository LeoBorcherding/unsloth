# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from utils.hardware import gpu_breakdown as gb

GIB = 1024**3


def test_parse_process_counter_keys_by_pid_and_luid():
    lines = [
        "pid_100_luid_0x00000000_0x0000abcd_phys_0|2147483648",
        "pid_100_luid_0x00000000_0x0000abcd_phys_0|1073741824",
        "pid_200_luid_0x00000000_0x0000ef01_phys_0|0",
        "garbage line",
        "pid_300_luid_0x00000000_0x0000ef01_phys_0|nope",
    ]
    out = gb.parse_process_counter(lines)
    assert out == {(100, 0xABCD): 3 * GIB}


def test_parse_nvidia_compute_apps_skips_wddm_na():
    gpus = "0, GPU-aaa\n1, GPU-bbb\n"
    apps = "10, GPU-bbb, 2048\n11, GPU-aaa, [N/A]\n"
    assert gb.parse_nvidia_compute_apps(apps, gpus) == {(10, 1): 2048 * 1024**2}


DEVICES = [
    {"index": 0, "visible_ordinal": 0, "vram_used_gb": 3.45},
    {"index": 1, "visible_ordinal": 1, "vram_used_gb": 26.3},
]


def test_build_breakdown_splits_unsloth_and_other_apps():
    process_bytes = {
        (1, 0): 0.3 * GIB,  # backend context on the idle card
        (1, 1): 0.5 * GIB,
        (2, 1): 24.0 * GIB,  # llama-server
        (999, 1): 5.0 * GIB,  # a game, not ours
    }
    out = gb.build_breakdown(DEVICES, process_bytes, {1, 2}, {2: ("chat", "gemma")}, [])
    gpu0, gpu1 = out
    assert gpu0["unsloth_gb"] == 0.3
    assert gpu0["other_gb"] == 3.15
    assert gpu0["items"] == [{"kind": "runtime", "model": None, "part": None, "gb": 0.3}]
    assert gpu1["unsloth_gb"] == 24.5
    assert gpu1["other_gb"] == 1.8
    assert gpu1["items"][0] == {"kind": "chat", "model": "gemma", "part": None, "gb": 24.0}


def test_build_breakdown_names_components_and_never_exceeds_measured():
    process_bytes = {(1, 1): 10.0 * GIB}
    comps = [
        (1, 1, "image", "Qwen-Image", "transformer", 8.0 * GIB),
        (1, 1, "image", "Qwen-Image", "vae", 4.0 * GIB),  # larger than what is left
    ]
    items = gb.build_breakdown(DEVICES, process_bytes, {1}, {}, comps)[1]["items"]
    assert sum(i["gb"] for i in items) == 10.0
    assert items[0]["part"] == "transformer"
    assert {"kind": "image", "model": "Qwen-Image", "part": "vae", "gb": 2.0} in items
    assert not any(i["kind"] == "runtime" for i in items)


def test_build_breakdown_unknown_without_process_reading():
    out = gb.build_breakdown(DEVICES, None, {1}, {}, [])
    assert all(d["unsloth_gb"] is None and d["items"] == [] for d in out)


def test_short_model_name_handles_windows_paths_and_repo_ids():
    assert gb.short_model_name(r"L:\hub\snapshots\abc\gemma-4-12B-it-qat-UD-Q4_K_XL.gguf") == "gemma-4-12B-it-qat-UD-Q4_K_XL"
    assert gb.short_model_name("/models/q.GGUF") == "q"
    assert gb.short_model_name("unsloth/Qwen-Image-2.1-GGUF") == "Qwen-Image-2.1-GGUF"
