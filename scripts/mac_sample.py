"""Sample the branch /api/system against macOS's own counters. Usage: mac_sample.py PHASE [N] [PORT]"""
import csv, json, re, subprocess, sys, time, urllib.request
from pathlib import Path

phase = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
port = int(sys.argv[3]) if len(sys.argv) > 3 else 8893
token = Path("/tmp/lmr-mac-token").read_text().strip()
out = Path(__file__).parent / "mac"
GiB = 1024**3


def get(path):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers={"Authorization": f"Bearer {token}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        body = json.load(r)
    return body, (time.perf_counter() - t0) * 1000


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def vm():
    text = sh("vm_stat")
    page = int(re.search(r"page size of (\d+)", text).group(1))
    v = {k.strip(): int(x) * page for k, x in re.findall(r"^(.+?):\s+(\d+)\.", text, re.M)}
    total = int(sh("sysctl", "-n", "hw.memsize"))
    return total, v


def server_pid():
    return int(sh("lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN").split()[0])


def tree(pid):
    pids, todo = [pid], [pid]
    while todo:
        kids = [int(p) for p in sh("pgrep", "-P", str(todo.pop())).split()]
        pids += kids
        todo += kids
    return pids


def rss_gb(pids):
    rows = sh("ps", "-o", "rss=", "-p", ",".join(map(str, pids))).split()
    return sum(int(r) for r in rows) * 1024 / GiB


def gpu_in_use_gb():
    m = re.search(r'"In use system memory"=(\d+)', sh("ioreg", "-r", "-d", "1", "-c", "IOAccelerator"))
    return int(m.group(1)) / GiB if m else None


rows = []
pid = server_pid()
for _ in range(n):
    t0 = time.time()
    sysinfo, ms = get("/api/system?gpu_max_age=1")
    inf, _ = get("/api/inference/status")
    total, v = vm()
    avail = v["Pages free"] + v["Pages inactive"] + v["Pages speculative"]
    am_used = (v["Anonymous pages"] - v["Pages purgeable"]) + v["Pages wired down"] + v["Pages occupied by compressor"]
    dev = (sysinfo.get("gpu", {}).get("devices") or [{}])[0]
    mem = sysinfo["memory"]
    rows.append({
        "phase": phase,
        "t": time.strftime("%H:%M:%S"),
        "api_ms": round(ms, 1),
        "ram_used_monitor": round(mem["total_gb"] - mem["available_gb"], 3),
        "ram_used_truth_vmstat": round((total - avail) / GiB, 3),
        "ram_used_activity_monitor_style": round(am_used / GiB, 3),
        "unsloth_monitor": round(mem.get("process_tree_used_mb", 0) / 1024, 3),
        "unsloth_truth_ps_rss": round(rss_gb(tree(pid)), 3),
        "gpu_in_use_monitor": dev.get("vram_used_gb"),
        "gpu_in_use_truth_ioreg": round(gpu_in_use_gb() or 0, 3),
        "model_resident_monitor": round((inf.get("resident_mib") or 0) / 1024, 3),
    })
    time.sleep(max(0, 1 - (time.time() - t0)))

with open(out / f"{phase}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(out / f"{phase}-system.json").write_text(json.dumps(sysinfo, indent=2))
(out / f"{phase}-inference-status.json").write_text(json.dumps(inf, indent=2))
print(phase, len(rows), rows[-1])
