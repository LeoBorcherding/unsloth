# Longer DPO through the Flows Fine-tune node: python dpo.py <model> <steps> <key>
import csv, json, sys, time, uuid, urllib.request
exec(open("/content/sbx/rfl/media_lib.py").read())
MODEL, STEPS, KEY = sys.argv[1], int(sys.argv[2]), sys.argv[3]
def api(method, path, body=None, tok=None):
    req = urllib.request.Request(BASE + path, method=method, headers={"Content-Type": "application/json"})
    if tok: req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, json.dumps(body).encode() if body is not None else None, timeout=120) as r:
        return json.loads(r.read() or b"null")
u, p = open("/content/sbx/login").read().split()[:2]
TOK = api("POST", "/api/auth/login", {"username": u, "password": p})["access_token"]; del u, p
cat = api("GET", "/api/flows/nodes", None, TOK)["nodes"]
tp = {x["id"]: x["default"] for x in next(n for n in cat if n["type"] == "train.finetune")["params"]}
tp.update(base_model=MODEL, objective="dpo", max_steps=STEPS, learning_rate=5e-6, beta=0.1, batch_size=2,
          gradient_accumulation_steps=4, warmup_steps=STEPS // 10, prompt_column="question",
          chosen_column="chosen", rejected_column="rejected", holdout_percent=0)
doc = {"id": uuid.uuid4().hex, "name": f"Orca DPO {MODEL.split('/')[-1]}", "nodes": [
    {"id": "data", "type": "data.hf_dataset", "v": 1, "pos": [0, 0], "params": {"repo": "Intel/orca_dpo_pairs", "split": "train"}},
    {"id": "tune", "type": "train.finetune", "v": 1, "pos": [460, 0], "params": tp}],
    "edges": [{"id": "e1", "source": "data", "source_port": "dataset", "target": "tune", "target_port": "dataset"}]}
fid = api("POST", "/api/flows", {"doc": doc}, TOK)["id"]
json.dump({"flow": fid, "tune": tp}, open(f"/content/sbx/rfl/{KEY}-config.json", "w"), indent=1)
hist = {}
def poll():
    st = api("GET", "/api/train/status", None, TOK)
    mh = st.get("metric_history") or {}
    for e in mh.get("rl") or []:
        hist.setdefault(e["step"], {}).update(e)
    for i, sp in enumerate(mh.get("steps") or []):
        if i < len(mh.get("loss") or []): hist.setdefault(sp, {})["loss"] = mh["loss"][i]
    rows = [dict(step=k, **{x: y for x, y in v.items() if x != "step"}) for k, v in sorted(hist.items())]
    if rows:
        keys = ["step"] + sorted({k for r in rows for k in r} - {"step"})
        with open(f"/content/sbx/rfl/{KEY}-metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
with sync_playwright() as pw:
    system_theme(); b = pw.chromium.launch(args=["--no-sandbox"])
    pages = {t: themed(b, t) for t in ("light", "dark")}
    for t, pg in pages.items():
        login(pg); open_flow(pg, fid)
        clip(pg, [node(pg, "data"), node(pg, "tune")], f"{KEY}-{t}-flow-ready")
    pages["light"].get_by_title("Run (Ctrl+Enter)").first.click()
    t0, last, mid, seen = time.time(), None, False, False
    while time.time() - t0 < 5 * 3600:
        st, d = train_state(pages["light"])
        if d != last: print(f"{int(time.time()-t0):6d}s {st}: {d}", flush=True); last = d
        if st == "training":
            seen = True
        if seen:
            poll()
        if st == "training" and not mid and hist and max(hist) >= STEPS // 2:
            for t, pg in pages.items(): clip(pg, [node(pg, "tune")], f"{KEY}-{t}-node-mid")
            mid = True
        if st in ("adapter_ready", "failed", "stopped"): break
        pages["light"].wait_for_timeout(20000)
    poll(); print("train done", st, d, int(time.time() - t0), "s", flush=True)
    for t, pg in pages.items():
        open_flow(pg, fid, 4000); zoom(pg, -2); pg.wait_for_timeout(800)
        clip(pg, [node(pg, "data"), node(pg, "tune")], f"{KEY}-{t}-flow-done")
        clip(pg, [node(pg, "tune")], f"{KEY}-{t}-node-done")
    b.close()
print("DPODONE", flush=True)
