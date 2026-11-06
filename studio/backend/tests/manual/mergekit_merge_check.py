# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end MergeKit check for a GPU box: installs MergeKit's own venv through the backend
installer, merges through the export orchestrator (the code the /api/export/merge route calls),
then loads the result and every parent with transformers and prints a greedy completion and a
perplexity for each. Exits 1 when the merge fails, its logits are NaN/inf, or its perplexity is
over --max-ratio times the worse parent's (for MoE, the base's).

Run from ``studio/backend`` with the Unsloth Studio venv's python, e.g.::

    python tests/manual/mergekit_merge_check.py --method ties \\
        --base mistralai/Mistral-7B-v0.1 \\
        --models OpenPipe/mistral-ft-optimized-1218 mlabonne/NeuralHermes-2.5-Mistral-7B \\
        --weights 0.5 0.3 --density 0.5 --cuda

    python tests/manual/mergekit_merge_check.py --method moe \
        --base Qwen/Qwen3-4B-Instruct-2507 \
        --models Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Thinking-2507 \
        --prompts "chat, explain, tell me, write" "reason, math, solve, prove, step by step" --cuda

Set UNSLOTH_STUDIO_HOME to keep the venv and the merged model out of the default Studio home.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROMPT = "The three primary colors are"
TEXT = (
    "The sun rises in the east and sets in the west. Water boils at one hundred degrees "
    "Celsius at sea level. Paris is the capital of France, and Berlin is the capital of Germany."
)


def _spec(args) -> dict:
    from models import MergeModelsRequest

    if args.method == "moe":
        return _moe_spec(args)
    weights = args.weights or ([None] * len(args.models))
    densities = args.density or []
    if len(densities) == 1:
        densities = densities * len(args.models)
    models = []
    for i, name in enumerate(args.models):
        entry = {"model": name}
        if args.method != "slerp":
            entry["weight"] = weights[i] if weights[i] is not None else 1.0 / len(args.models)
        if densities:
            entry["density"] = densities[i]
        models.append(entry)
    request = MergeModelsRequest(
        method = args.method,
        models = models,
        base_model = args.base,
        t = args.t if args.method == "slerp" else None,
        dtype = args.dtype,
        random_seed = args.seed,
        output_name = args.output_name or f"mergekit-check-{args.method}-{int(time.time())}",
        cuda = args.cuda,
    )
    return request.model_dump(exclude = {"hf_token", "install_missing_dependencies", "op_id"})


def _moe_spec(args) -> dict:
    from models import MergeModelsRequest

    prompts = args.prompts or []
    if prompts and len(prompts) != len(args.models):
        sys.exit("--prompts takes one comma-separated list per expert")
    experts = [
        {
            "model": name,
            "positive_prompts": prompts[i].split(",") if prompts else None,
        }
        for i, name in enumerate(args.models)
    ]
    request = MergeModelsRequest(
        method = "moe",
        models = experts,
        base_model = args.base,
        gate_mode = args.gate_mode,
        experts_per_token = args.experts_per_token,
        architecture = args.architecture,
        dtype = args.dtype,
        random_seed = args.seed,
        output_name = args.output_name or f"mergekit-check-moe-{int(time.time())}",
        cuda = args.cuda,
    )
    return request.model_dump(exclude = {"hf_token", "install_missing_dependencies", "op_id"})


def _merge(spec: dict) -> Path:
    from core.export.orchestrator import get_export_backend

    backend = get_export_backend()
    done = threading.Event()

    def tail():
        cursor = 0
        while not done.is_set():
            entries, cursor = backend.get_logs_since(cursor)
            for entry in entries:
                print(f"[{entry['stream']}] {entry['line']}", flush = True)
            time.sleep(0.5)

    printer = threading.Thread(target = tail, daemon = True)
    printer.start()
    started = time.monotonic()
    ok, message, path = backend.merge_models(spec, install_missing_dependencies = True)
    time.sleep(1)
    done.set()
    printer.join()
    print(f"merge: {'ok' if ok else 'FAILED'} in {time.monotonic() - started:.0f}s: {message}")
    if not ok:
        sys.exit(1)
    return Path(path)


def _evaluate(name: str, dtype: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype = getattr(torch, dtype)).to(device)
    model.eval()
    with torch.no_grad():
        ids = tokenizer(TEXT, return_tensors = "pt").input_ids.to(device)
        out = model(ids, labels = ids)
        finite = bool(torch.isfinite(out.logits).all())
        ppl = math.exp(out.loss.float().item()) if finite else float("nan")
        prompt = tokenizer(PROMPT, return_tensors = "pt").to(device)
        generated = model.generate(
            **prompt,
            max_new_tokens = 30,
            do_sample = False,
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated[0][prompt.input_ids.shape[1] :], skip_special_tokens = True)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"name": name, "ppl": ppl, "finite": finite, "text": text}


def main(argv = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument(
        "--method",
        required = True,
        choices = ("linear", "slerp", "ties", "dare_ties", "task_arithmetic", "moe"),
    )
    parser.add_argument(
        "--models",
        nargs = "+",
        required = True,
        help = "SLERP: both models, base included. MoE: the experts",
    )
    parser.add_argument("--base", default = None)
    parser.add_argument("--weights", nargs = "+", type = float)
    parser.add_argument("--density", nargs = "+", type = float)
    parser.add_argument("--t", type = float, default = 0.5)
    parser.add_argument(
        "--prompts", nargs = "+", help = "MoE: one comma-separated prompt list per expert"
    )
    parser.add_argument(
        "--gate-mode", choices = ("hidden", "cheap_embed", "random", "uniform_random")
    )
    parser.add_argument("--experts-per-token", type = int)
    parser.add_argument("--architecture", choices = ("mixtral", "qwen3", "deepseek"))
    parser.add_argument("--dtype", default = "bfloat16", choices = ("bfloat16", "float16", "float32"))
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--cuda", action = "store_true", help = "Merge on the GPU (mergekit --cuda)")
    parser.add_argument("--output-name", default = None)
    parser.add_argument("--max-ratio", type = float, default = 3.0)
    args = parser.parse_args(argv)
    # A random-weight model's completion is arbitrary tokens; a cp1252 console cannot print them.
    sys.stdout.reconfigure(encoding = "utf-8", errors = "replace")

    merged = _merge(_spec(args))
    parents = list(dict.fromkeys([*([args.base] if args.base else []), *args.models]))
    rows = [_evaluate(str(merged), args.dtype), *(_evaluate(p, args.dtype) for p in parents)]
    for row in rows:
        print(
            f"\n== {row['name']}\nppl {row['ppl']:.3f}  finite logits {row['finite']}\n{PROMPT}{row['text']}"
        )
    result, parent_ppl = rows[0], [row["ppl"] for row in rows[1:]]
    # A MoE is the base with routed MLPs, so the base is the yardstick; parents[0] is the base.
    worst = parent_ppl[0] if args.method == "moe" else max(parent_ppl)
    bad = (
        not result["finite"]
        or not math.isfinite(result["ppl"])
        or result["ppl"] > args.max_ratio * worst
    )
    print(
        f"\nmerged ppl {result['ppl']:.3f}, parents {', '.join(f'{p:.3f}' for p in parent_ppl)}: "
        + ("GARBAGE" if bad else "ok")
    )
    print(f"merged model: {merged}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
