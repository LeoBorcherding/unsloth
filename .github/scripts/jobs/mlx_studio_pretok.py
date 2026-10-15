"""Studio's MLX training path on pre-tokenized uploads (unslothai/unsloth#13816), Apple Silicon only.

Mirrors core/training/worker.py:_run_mlx_training for the dataset side: Studio's
format_and_template_dataset (format_type "auto"), FastMLXModel + get_peft_model, MLXTrainer with the
worker's config, the worker's response-masking gate, then 3 optimizer steps per case:
  labels      input_ids + attention_mask + labels, one row longer than max_seq_length
  ids_only    chat-rendered input_ids only: trains on full sequences (the worker warns; zoo's MLX masker needs text)
  chat        a normal messages dataset (control: the PR must not move it)
Exit 0 all cases pass, 1 a case failed, 3 not Apple Silicon. Last stdout line: JOB_RESULT {...}.
Run from a checkout root: python jobs/mlx_studio_pretok.py [--model M] [--out F]."""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import traceback

MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
MAX_SEQ = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = MODEL)
    ap.add_argument("--out", default = "outputs/jobs/mlx_studio_pretok.json")
    ap.add_argument("--tiny", action = "store_true")
    args = ap.parse_args()
    result = {
        "job": "mlx_studio_pretok",
        "model": args.model,
        "cases": {},
        "passed": False,
        "skipped": False,
    }

    def finish(code):
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok = True)
        with open(args.out, "w", encoding = "utf-8") as f:
            json.dump(result, f, indent = 1)
        print("JOB_RESULT " + json.dumps(result))
        sys.exit(code)

    if sys.platform != "darwin" or platform.machine() != "arm64":
        result["skipped"] = True
        finish(3)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "structlog"], check = False)
    sys.path.insert(0, os.path.join(os.getcwd(), "studio", "backend"))
    from datasets import Dataset
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig, train_on_responses_only
    from utils.datasets.completion_masking import apply_completion_masking
    from utils.datasets.dataset_utils import format_and_template_dataset
    from utils.datasets.pretokenized import has_custom_labels

    model, tokenizer = FastMLXModel.from_pretrained(args.model, load_in_4bit = True, text_only = True)
    model = FastMLXModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0.0,
        random_state = 3407,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    tok = getattr(tokenizer, "tokenizer", tokenizer)

    def chat_ids(i):
        msgs = [
            {"role": "user", "content": f"What is {i}+{i}?"},
            {"role": "assistant", "content": f"It is {2 * i}."},
        ]
        out = tok.apply_chat_template(msgs, tokenize = True)
        return list(out["input_ids"] if isinstance(out, dict) else out)

    cases = {}
    rows = []
    for i in range(16):
        ids = chat_ids(i)
        if i == 0:
            ids = (ids * (MAX_SEQ // len(ids) + 2))[: MAX_SEQ + 40]
        cut = len(ids) // 2
        rows.append(
            {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * cut + ids[cut:]}
        )
    cases["labels"] = Dataset.from_list(rows)
    cases["ids_only"] = Dataset.from_list([{"input_ids": chat_ids(i)} for i in range(16)])
    cases["chat"] = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": f"What is {i}+{i}?"},
                    {"role": "assistant", "content": f"It is {2 * i}."},
                ]
            }
            for i in range(16)
        ]
    )

    ok = True
    for name, ds in cases.items():
        rec = {}
        try:
            info = format_and_template_dataset(
                ds, model_name = args.model, tokenizer = tokenizer, format_type = "auto"
            )
            rec["final_format"] = final = str(info.get("final_format", "")).lower()
            assert info.get("success", True), info.get("errors")
            data = info["dataset"]
            expect = "pretokenized" if name != "chat" else None
            if expect:
                assert final == expect, f"routed as {final}"
            cfg = MLXTrainingConfig(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 1,
                max_steps = 3,
                learning_rate = 2e-4,
                warmup_steps = 0,
                logging_steps = 1,
                max_seq_length = MAX_SEQ,
                seed = 3407,
                output_dir = tempfile.mkdtemp(prefix = f"mlx_pretok_{name}_"),
                packing = False,
            )
            trainer = MLXTrainer(model = model, tokenizer = tokenizer, train_dataset = data, args = cfg)
            # The worker's gate (core/training/worker.py, _run_mlx_training).
            mask = final not in ("raw_text", "pretokenized")
            rec["labels_column"] = has_custom_labels(data) if final == "pretokenized" else None
            rec["masking"] = False
            if mask:
                trainer, applied = apply_completion_masking(
                    trainer,
                    args.model,
                    train_on_responses_only,
                    notify = lambda level, message: print("notify", level, message),
                    dataset_template = None,
                )
                rec["masking"] = bool(applied)
            elif name == "labels":
                rec["labels_kept"] = True
            out = trainer.train()
            hist = [
                h
                for h in getattr(getattr(trainer, "state", None), "log_history", []) or []
                if "loss" in h
            ]
            losses = [float(h["loss"]) for h in hist]
            if not losses and isinstance(out, dict):
                losses = [float(x) for x in out.get("losses", [])]
            rec["losses"] = [round(x, 4) for x in losses]
            assert losses and all(math.isfinite(x) for x in losses), f"losses {losses}"
            rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"[:500]
            traceback.print_exc()
            ok = False
        result["cases"][name] = rec
        print(name, json.dumps(rec))
    result["passed"] = ok
    finish(0 if ok else 1)


if __name__ == "__main__":
    main()
