# Base vs GRPO adapter on gsm8k test[:N], greedy, scored with the same library rewards training used.
import glob, json, os, re, sys, time
sys.path.insert(0, "/content/unsloth/studio/backend")
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.training.rewards import get_reward, score_rule
MODEL, KEY, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
SYS = "Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
run = max(glob.glob(f"/root/.unsloth/studio/outputs/{MODEL.replace('/', '_')}_*"), key=os.path.getmtime)
print("adapter", run, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
rows = load_dataset("openai/gsm8k", "main", split="test").select(range(N))
R = {n: get_reward(n)["rule"] for n in ("exact-answer", "soft-xml-format", "strict-xml-format")}
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
def number_ok(t, ref):
    m = re.search(r"<answer>(.*?)</answer>", t, re.S)
    nums = NUM.findall(m.group(1)) if m else []
    want = ref.split("####")[-1].strip().replace(",", "")
    return bool(nums) and nums[-1].replace(",", "").rstrip(".") == want
def run_model(m, label):
    out, t0 = [], time.time()
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": r["question"]}], tokenize=False, add_generation_prompt=True) for r in rows]
    for i in range(0, N, 50):
        ids = tok(prompts[i:i+50], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = m.generate(**ids, max_new_tokens=768, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    recs = []
    for r, t in zip(rows, out):
        recs.append({"question": r["question"], "reference": r["answer"].split("####")[-1].strip(), "completion": t,
                     "exact": score_rule(R["exact-answer"], t, r["answer"]) > 0, "soft_format": score_rule(R["soft-xml-format"], t) > 0,
                     "strict_format": score_rule(R["strict-xml-format"], t) > 0, "number_correct": number_ok(t, r["answer"])})
    with open(f"/content/sbx/rfl/{KEY}-eval-{label}.jsonl", "w") as f:
        for x in recs: f.write(json.dumps(x) + "\n")
    s = {k: sum(x[k] for x in recs) for k in ("exact", "soft_format", "strict_format", "number_correct")}
    s.update(n=len(recs), seconds=int(time.time() - t0))
    print(label, json.dumps(s), flush=True)
    return s
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
res = {"model": MODEL, "adapter": run, "n": N, "base": run_model(base, "base")}
res["grpo"] = run_model(PeftModel.from_pretrained(base, run), "grpo")
json.dump(res, open(f"/content/sbx/rfl/{KEY}-eval.json", "w"), indent=1)
print("EVALDONE", flush=True)
