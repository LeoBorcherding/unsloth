"""Colab job: merges of newer model families (Qwen3, SmolLM3) through the PR's backend, then
check each merge against its parents (finite logits, perplexity, 30-token greedy completion).
Run from studio/backend with the Unsloth Studio python. Writes one JSON per merge to OUT."""

import json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, "tests/manual")
import mergekit_merge_check as mc
from models import MergeModelsRequest

OUT = Path("/content/sbx/mk_results_new")
OUT.mkdir(parents = True, exist_ok = True)

SPECS = {
    "qwen3_4b_slerp": dict(
        method = "slerp",
        base_model = "Qwen/Qwen3-4B-Instruct-2507",
        models = [
            {"model": "Qwen/Qwen3-4B-Instruct-2507", "layer_range": [0, 36]},
            {"model": "Qwen/Qwen3-4B-Thinking-2507", "layer_range": [0, 36]},
        ],
        t = [
            {"filter": "self_attn", "value": [0, 0.5, 0.3, 0.7, 1]},
            {"filter": "mlp", "value": [1, 0.5, 0.7, 0.3, 0]},
            {"value": 0.5},
        ],
        tokenizer_source = "base",
        dtype = "bfloat16",
    ),
    "qwen3_4b_ties": dict(
        method = "ties",
        base_model = "Qwen/Qwen3-4B-Base",
        models = [
            {"model": "Qwen/Qwen3-4B-Instruct-2507", "density": 0.5, "weight": 0.5},
            {"model": "Qwen/Qwen3-4B-Thinking-2507", "density": 0.5, "weight": 0.5},
        ],
        normalize = True,
        int8_mask = True,
        tokenizer_source = "base",
        dtype = "bfloat16",
    ),
    "qwen3_4b_dare_ties": dict(
        method = "dare_ties",
        base_model = "Qwen/Qwen3-4B-Base",
        models = [
            {"model": "Qwen/Qwen3-4B-Instruct-2507", "density": 0.53, "weight": 0.4},
            {"model": "Qwen/Qwen3-4B-Thinking-2507", "density": 0.53, "weight": 0.3},
            {"model": "Qwen/Qwen3-4B", "density": 0.53, "weight": 0.3},
        ],
        int8_mask = True,
        random_seed = 0,
        tokenizer_source = "base",
        dtype = "bfloat16",
    ),
    "smollm3_3b_ties": dict(
        method = "ties",
        base_model = "HuggingFaceTB/SmolLM3-3B-Base",
        models = [
            {"model": "HuggingFaceTB/SmolLM3-3B", "density": 0.5, "weight": 1.0},
        ],
        tokenizer_source = "base",
        dtype = "bfloat16",
    ),
}


def _gpu_used_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text = True
        )
        return int(out.split()[0])
    except Exception:
        return 0


def _host_used_mb() -> int:
    with open("/proc/meminfo") as f:
        info = {line.split(":")[0]: int(line.split()[1]) for line in f}
    return (info["MemTotal"] - info["MemAvailable"]) // 1024


def _sampler(stop, peaks):
    base_host, base_gpu = _host_used_mb(), _gpu_used_mb()
    while not stop.is_set():
        peaks["host_mb"] = max(peaks["host_mb"], _host_used_mb() - base_host)
        peaks["gpu_mb"] = max(peaks["gpu_mb"], _gpu_used_mb() - base_gpu)
        time.sleep(1)


def _dir_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9


def run(name: str) -> dict:
    fields = SPECS[name]
    request = MergeModelsRequest(**fields, output_name = f"{name}-{int(time.time())}", cuda = True)
    spec = request.model_dump(exclude = {"hf_token", "install_missing_dependencies", "op_id"})
    peaks, stop = {"host_mb": 0, "gpu_mb": 0}, threading.Event()
    sampler = threading.Thread(target = _sampler, args = (stop, peaks), daemon = True)
    sampler.start()
    started = time.monotonic()
    path = mc._merge(spec)
    merge_s = time.monotonic() - started
    stop.set()
    sampler.join()
    parents = [fields["base_model"]] + [m["model"] for m in fields["models"]]
    parents = list(dict.fromkeys(parents))
    rows = [mc._evaluate(str(path), "bfloat16")] + [mc._evaluate(p, "bfloat16") for p in parents]
    result = {
        "name": name,
        "method": fields["method"],
        "merge_seconds": round(merge_s, 1),
        "peak_host_mb": peaks["host_mb"],
        "peak_gpu_mb": peaks["gpu_mb"],
        "output_gb": round(_dir_gb(path), 2),
        "rows": rows,
    }
    (OUT / f"{name}.json").write_text(json.dumps(result, indent = 2))
    print(json.dumps(result, indent = 2), flush = True)
    shutil.rmtree(path, ignore_errors = True)
    return result


if __name__ == "__main__":
    failed = 0
    for name in sys.argv[1:]:
        try:
            run(name)
        except SystemExit:
            failed += 1
            (OUT / f"{name}.json").write_text(json.dumps({"name": name, "failed": True}))
    sys.exit(1 if failed else 0)
