# Long GRPO run through the Flows GRPO Fine-tune node, then a Benchmark node on 200 gsm8k test rows.
# Usage: python long.py <model> <steps> <prefix>. Login is read on the VM and never printed.
import csv, json, sys, time, urllib.request, urllib.error, uuid
exec(open("/content/sbx/rfl/pw_lib.py").read())
MODEL, STEPS, KEY = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SYS = "Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
RUB = ("PASS only if the final answer in the <answer> block is the same number as the number after '####' "
       "at the end of the reference answer. Ignore units, dollar signs and wording. Otherwise FAIL.")
OUTD = "/content/sbx/rfl"

def api(method, path, body=None, tok=None):
    req = urllib.request.Request(BASE + path, method=method, headers={"Content-Type": "application/json"})
    if tok: req.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(req, json.dumps(body).encode() if body is not None else None, timeout=120) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        if e.code == 401 and tok and path != "/api/auth/login":
            return api(method, path, body, relogin())
        return e.code, e.read().decode()[:1000]

def relogin():
    global TOK
    u, p = open("/content/sbx/login").read().split()[:2]
    TOK = api("POST", "/api/auth/login", {"username": u, "password": p})[1]["access_token"]; del u, p
    return TOK

u, p = open("/content/sbx/login").read().split()[:2]
s, t = api("POST", "/api/auth/login", {"username": u, "password": p}); del u, p
assert s == 200, s
TOK = t["access_token"]

s, cat = api("GET", "/api/flows/nodes", None, TOK)
spec = next(n for n in cat["nodes"] if n["type"] == "train.finetune")
tp = {x["id"]: x["default"] for x in spec["params"]}
tp.update(base_model=MODEL, objective="grpo", max_steps=STEPS, lora_r=64, lora_alpha=64,
          warmup_steps=max(1, STEPS // 10), weight_decay=0.1, num_generations=8, max_completion_length=768,
          batch_size=8, gradient_accumulation_steps=1, learning_rate=float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5)
bspec = next(n for n in cat["nodes"] if n["type"] == "train.benchmark")
bp = {x["id"]: x["default"] for x in bspec["params"]}
bp.update(task="Recipe held-out", scorer="LLM judge", max_rows=int(os.environ.get('BENCH_ROWS', 100)), compare_base=True, system_prompt=SYS,
          rubric=RUB, prompt_column="question", reference_column="answer")
doc = {"id": uuid.uuid4().hex, "name": f"GSM8K GRPO {MODEL.split('/')[-1]}", "nodes": [
    {"id": "data", "type": "data.hf_dataset", "v": 1, "pos": [0, 0], "params": {"repo": "openai/gsm8k", "subset": "main", "split": "train"}},
    {"id": "tune", "type": "train.finetune", "v": 1, "pos": [440, 0], "params": tp},
    {"id": "test", "type": "data.hf_dataset", "v": 1, "pos": [440, 420], "params": {"repo": "openai/gsm8k", "subset": "main", "split": "test"}},
    {"id": "bench", "type": "train.benchmark", "v": 1, "pos": [900, 160], "params": bp}],
    "edges": [{"id": "e1", "source": "data", "source_port": "dataset", "target": "tune", "target_port": "dataset"},
              {"id": "e2", "source": "tune", "source_port": "checkpoint", "target": "bench", "target_port": "checkpoint"},
              {"id": "e3", "source": "test", "source_port": "dataset", "target": "bench", "target_port": "dataset"}]}
s, b = api("POST", "/api/flows", {"doc": doc}, TOK)
assert s in (200, 201), (s, b)
FLOWS[KEY] = b["id"]
json.dump({"flow": b["id"], "tune": tp, "bench": bp}, open(f"{OUTD}/{KEY}-config.json", "w"), indent=1)
print("flow", b["id"], flush=True)

hist = {}
def poll():
    s, st = api("GET", "/api/train/status", None, TOK)
    if s != 200 or not isinstance(st, dict): return None
    mh = st.get("metric_history") or {}
    for e in mh.get("rl") or []:
        hist.setdefault(e["step"], {}).update(e)
    steps, losses = mh.get("steps") or [], mh.get("loss") or []
    for i, sp in enumerate(steps):
        if i < len(losses): hist.setdefault(sp, {})["loss"] = losses[i]
    rows = [dict(step=k, **{x: y for x, y in v.items() if x != "step"}) for k, v in sorted(hist.items())]
    if rows:
        keys = ["step"] + sorted({k for r in rows for k in r} - {"step"})
        with open(f"{OUTD}/{KEY}-metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    return st

with sync_playwright() as pw:
    br, ctx, page, logs = start(pw)
    login(page)
    open_flow(page, KEY)
    shot(page, f"{KEY}-01-graph-start")
    el(node(page), f"{KEY}-02-node-start")
    inspect(page); shot(page, f"{KEY}-03-inspector")
    open_flow(page, KEY)
    run_flow(page)
    t0 = time.time(); i = 0; last = None; mid = False; lastpoll = 0
    while time.time() - t0 < 9 * 3600:
        s_, d = state(page), detail(page)
        if d != last:
            print(f"{int(time.time()-t0):6d}s {s_}: {d}", flush=True); last = d
        if time.time() - lastpoll > 30:
            poll(); lastpoll = time.time()
        if s_ == "training":
            el(node(page), f"{KEY}-frame-{i:04d}"); i += 1
            if not mid and hist and max(hist) >= STEPS // 2:
                el(node(page), f"{KEY}-04-node-mid"); shot(page, f"{KEY}-04b-graph-mid"); mid = True
        if s_ in ("adapter_ready", "failed", "stopped"):
            break
        page.wait_for_timeout(20000)
    poll()
    print("train done", s_, detail(page), int(time.time() - t0), "s", flush=True)
    el(node(page), f"{KEY}-05-node-done")
    bench = page.locator('[data-node-id="bench"]').first
    t1 = time.time()
    while s_ == "adapter_ready" and time.time() - t1 < 3 * 3600:
        lines = [l.strip() for l in bench.inner_text().split("\n")]
        if "Done" in lines or "Failed" in lines or "Cancelled" in lines:
            break
        page.wait_for_timeout(30000)
    print("bench done", int(time.time() - t1), "s:", bench.inner_text().replace("\n", " | ")[:600], flush=True)
    fit(page); shot(page, f"{KEY}-06-graph-done"); el(bench, f"{KEY}-07-bench-node")
    s, runs = api("GET", f"/api/flows/{FLOWS[KEY]}/runs", None, TOK)
    json.dump(runs, open(f"{OUTD}/{KEY}-flow-runs.json", "w"), indent=1, default=str)
    page.goto(BASE + "/benchmarks"); page.wait_for_load_state("networkidle"); page.wait_for_timeout(3000)
    tab = page.get_by_role("tab", name="Evals")
    if tab.count(): tab.first.click(); page.wait_for_timeout(3000)
    shot(page, f"{KEY}-08-evals-tab")
    print("\n".join(logs[-20:]))
    br.close()
print("ALLDONE", flush=True)
