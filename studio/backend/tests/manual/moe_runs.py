"""Colab job: MoE merges (Beyonder-4x7B-v2, a Qwen3 2x4B) through the PR's backend, then
check each merge against its parents (finite logits, perplexity, 30-token greedy completion).
Run from studio/backend with the Unsloth Studio python. Writes one JSON per merge to OUT."""

import json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, "tests/manual")
import mergekit_merge_check as mc
from models import MergeModelsRequest

OUT = Path("/content/sbx/mk_results_moe")
OUT.mkdir(parents = True, exist_ok = True)

SPECS = {
    # mlabonne/Beyonder-4x7B-v2, the published mergekit-moe config
    "beyonder_4x7b_moe": dict(
        method = "moe",
        base_model = "mlabonne/Marcoro14-7B-slerp",
        models = [
            {"model": "openchat/openchat-3.5-1210", "positive_prompts": ["chat", "assistant", "tell me", "explain"]},
            {"model": "beowolx/CodeNinja-1.0-OpenChat-7B", "positive_prompts": ["code", "python", "javascript", "programming", "algorithm"]},
            {"model": "maywell/PiVoT-0.1-Starling-LM-RP", "positive_prompts": ["storywriting", "write", "scene", "story", "character"]},
            {"model": "WizardLM/WizardMath-7B-V1.1", "positive_prompts": ["reason", "math", "mathematics", "solve", "count"]},
        ],
        dtype = "bfloat16",
    ),
    "qwen3_2x4b_moe": dict(
        method = "moe",
        base_model = "Qwen/Qwen3-4B-Instruct-2507",
        models = [
            {"model": "Qwen/Qwen3-4B-Instruct-2507", "positive_prompts": ["chat", "explain", "tell me", "write"]},
            {"model": "Qwen/Qwen3-4B-Thinking-2507", "positive_prompts": ["reason", "math", "solve", "prove", "step by step"]},
        ],
        architecture = "qwen3",
        dtype = "bfloat16",
    ),
}
# Too big for 24 GB in bf16 (~48 GB): evaluated in 4-bit nf4, and so is its base.
FOUR_BIT = {"beyonder_4x7b_moe"}


def _evaluate_4bit(name: str) -> dict:
    import gc, math, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    q = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_quant_type = "nf4", bnb_4bit_compute_dtype = torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config = q, device_map = "cuda:0")
    model.eval()
    with torch.no_grad():
        ids = tokenizer(mc.TEXT, return_tensors = "pt").input_ids.to("cuda")
        out = model(ids, labels = ids)
        finite = bool(torch.isfinite(out.logits).all())
        ppl = math.exp(out.loss.float().item()) if finite else float("nan")
        prompt = tokenizer(mc.PROMPT, return_tensors = "pt").to("cuda")
        gen = model.generate(**prompt, max_new_tokens = 30, do_sample = False,
                             pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id)
    text = tokenizer.decode(gen[0][prompt.input_ids.shape[1]:], skip_special_tokens = True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"name": name, "ppl": ppl, "finite": finite, "text": text, "eval": "nf4"}


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
    if name in FOUR_BIT:
        evaluate = _evaluate_4bit
    else:
        evaluate = lambda n: mc._evaluate(n, "bfloat16")
    rows = [evaluate(str(path))] + [evaluate(p) for p in parents]
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
