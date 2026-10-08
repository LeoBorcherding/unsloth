"""Differential repro for unslothai/unsloth#11623 on a real OS (no mocks).

Part A (the user-visible symptom, no Unsloth code): another PROCESS listens on 0.0.0.0:P and answers
"OTHER". We then bind 127.0.0.1:P the way Studio's uvicorn does and listen, answering "STUDIO". If the
bind succeeds and a localhost client now reaches STUDIO, the other app was silently taken over.

Part B (the probe): `_is_port_free` extracted from each variant's run.py (main = before, PR heads =
after), called against real listeners in other processes:
  wild4      other process on 0.0.0.0           -> correct answer False  (the issue)
  loop4      other process on 127.0.0.1         -> False (control, must hold on every variant)
  free       nothing listening                  -> True  (control, must hold on every variant)
  wild6only  other process on [::] IPV6_V6ONLY  -> informational (IPv4 bind may legally coexist)
Each call is timed. Exit 0 = harness ran and controls held; the verdict is the printed data.
"""

import argparse
import ast
import errno
import importlib.util
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

CHILD = textwrap.dedent(r"""
    import socket, sys
    host, family, reuse, v6only = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4] == "1"
    s = socket.socket(family, socket.SOCK_STREAM)
    if reuse:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1 if v6only else 0)
    s.bind((host, 0 if sys.argv[5] == "0" else int(sys.argv[5])))
    s.listen(64)
    print("READY", s.getsockname()[1], flush=True)
    while True:
        c, _ = s.accept()
        try:
            c.sendall(b"OTHER\n")
        finally:
            c.close()
""")

NAMES = (
    "_is_port_free",
    "_listener_collides",
    "_CONNECT_REFUSED",
    "_is_loopback_address",
    "_addresses_collide",
    "_bind_addresses",
)


def load_variant(run_py: Path, host_policy):
    src = run_py.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], "id", None)
        if name in NAMES:
            keep.append(ast.get_source_segment(src, node))
    ns = {
        "socket": socket,
        "sys": sys,
        "errno": errno,
        "os": os,
        "is_wildcard_host": host_policy.is_wildcard_host,
    }
    exec(compile("\n\n".join(keep), str(run_py), "exec"), ns)
    return ns["_is_port_free"], sorted(n for n in NAMES if n in ns)


class Other:
    """A listener in a separate process (the 'other app')."""

    def __init__(
        self,
        host,
        family,
        port = 0,
        reuse = False,
        v6only = True,
    ):
        self.p = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                CHILD,
                host,
                str(int(family)),
                "1" if reuse else "0",
                "1" if v6only else "0",
                str(port),
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
        )
        line = self.p.stdout.readline().split()
        if not line or line[0] != "READY":
            err = self.p.stderr.read()
            self.p.kill()
            raise RuntimeError(f"child listener failed on {host}: {err.strip()[-300:]}")
        self.port = int(line[1])

    def close(self):
        self.p.kill()
        self.p.wait()


def who_answers(
    host,
    port,
    timeout = 2.0,
):
    try:
        with socket.create_connection((host, port), timeout = timeout) as c:
            c.settimeout(timeout)
            return c.recv(16).decode().strip() or "EMPTY"
    except OSError as e:
        return f"ERR:{type(e).__name__}:{getattr(e, 'errno', '')}"


def studio_like_bind(port, mode):
    """Bind 127.0.0.1:port like Studio. mode 'uvicorn' = uvicorn.Config.bind_socket() (what run.py's
    uvicorn.Config(host=, port=) ends up doing); 'plain' = bare bind, no socket options."""
    if mode == "uvicorn":
        import uvicorn
        cfg = uvicorn.Config("x:app", host = "127.0.0.1", port = port, log_level = "critical")
        sock = cfg.bind_socket()  # SystemExit on failure
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", port))
    sock.listen(64)
    stop = threading.Event()

    def serve():
        sock.settimeout(0.2)
        while not stop.is_set():
            try:
                c, _ = sock.accept()
            except OSError:
                continue
            try:
                c.sendall(b"STUDIO\n")
            finally:
                c.close()

    t = threading.Thread(target = serve, daemon = True)
    t.start()
    return sock, stop, t


def part_a(results):
    for reuse in (False, True):
        for mode in ("uvicorn", "plain"):
            other = Other("0.0.0.0", socket.AF_INET, reuse = reuse)
            port = other.port
            row = {
                "other_so_reuseaddr": reuse,
                "studio_bind": mode,
                "port": port,
                "before": {h: who_answers(h, port) for h in ("127.0.0.1", "localhost")},
            }
            try:
                sock, stop, t = studio_like_bind(port, mode)
                row["studio_bind_result"] = "BOUND"
                row["after"] = {h: who_answers(h, port) for h in ("127.0.0.1", "localhost")}
                stop.set()
                t.join(1)
                sock.close()
            except (OSError, SystemExit) as e:
                row["studio_bind_result"] = f"REFUSED ({type(e).__name__}: {e})"[:160]
                row["after"] = None
            row["hijacked"] = bool(row["after"]) and row["after"].get("127.0.0.1") == "STUDIO"
            other.close()
            results.append(row)
            print(
                f"[A] other 0.0.0.0 reuse={reuse!s:5} studio={mode:7} -> bind {row['studio_bind_result'][:40]:40}"
                f" before={row['before']} after={row['after']} HIJACKED={row['hijacked']}",
                flush = True,
            )


def timed(
    fn,
    host,
    port,
    n = 3,
):
    vals, times = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        vals.append(fn(host, port))
        times.append(time.perf_counter() - t0)
    return vals, statistics.median(times), (min(times), max(times))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def part_b(variants, results):
    has_v6 = socket.has_ipv6
    scenarios = [
        ("wild4", "0.0.0.0", socket.AF_INET, False),
        ("loop4", "127.0.0.1", socket.AF_INET, False),
        ("free", None, None, True),
    ]
    if has_v6:
        scenarios.append(("wild6only", "::", socket.AF_INET6, None))
    for scen, host, fam, expect in scenarios:
        for reuse in (False, True) if host else (False,):
            other = Other(host, fam, reuse = reuse) if host else None
            port = other.port if other else free_port()
            for vname, fn in variants.items():
                for probe_host in ("127.0.0.1", "localhost"):
                    vals, med, spread = timed(fn, probe_host, port)
                    row = {
                        "scenario": scen,
                        "other_so_reuseaddr": reuse,
                        "variant": vname,
                        "probe_host": probe_host,
                        "port": port,
                        "answers": vals,
                        "median_s": round(med, 3),
                        "spread_s": [round(x, 3) for x in spread],
                        "expected_free": expect,
                    }
                    if expect is None:
                        row["ok"] = None
                    else:
                        row["ok"] = all(v is expect for v in vals)
                    results.append(row)
                    print(
                        f"[B] {scen:9} reuse={reuse!s:5} {vname:12} {probe_host:9} -> free={vals} "
                        f"median={med:.3f}s expected_free={expect} ok={row['ok']}",
                        flush = True,
                    )
            if other:
                other.close()


def part_c(variants, results):
    """Full-backlog listener (same shape as the PR's test): raw connect code, psutil's view, answers."""
    for backlog in (0, 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("0.0.0.0", 0))
            listener.listen(backlog)
            port = listener.getsockname()[1]
            fillers = []
            for _ in range(4):
                f = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                f.settimeout(2)
                try:
                    f.connect(("127.0.0.1", port))
                    fillers.append(("ok", f))
                except OSError as e:
                    fillers.append((f"err {getattr(e, 'errno', e)}", f))
            row = {"backlog": backlog, "port": port, "fillers": [x for x, _ in fillers]}
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.25)
                t0 = time.perf_counter()
                row["probe_connect_ex"] = s.connect_ex(("127.0.0.1", port))
                row["probe_connect_s"] = round(time.perf_counter() - t0, 3)
            try:
                import psutil
                row["psutil"] = [
                    (str(c.family), c.laddr[0], c.laddr[1], c.status, c.pid)
                    for c in psutil.net_connections(kind = "tcp")
                    if c.laddr and c.laddr[1] == port
                ]
            except Exception as e:
                row["psutil"] = f"EXC {type(e).__name__}: {e}"
            row["answers"] = {k: fn("127.0.0.1", port) for k, fn in variants.items()}
            for _, f in fillers:
                f.close()
            results.append(row)
            print(f"[C] {row}", flush = True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required = True, help = "dir of <name>.py run.py copies")
    ap.add_argument("--host-policy", required = True)
    ap.add_argument("--out", required = True)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("host_policy", a.host_policy)
    hp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hp)

    variants, loaded = {}, {}
    for f in sorted(Path(a.variants).glob("*.py")):
        variants[f.stem], loaded[f.stem] = load_variant(f, hp)
    order = ["main"] + [k for k in variants if k != "main"]
    variants = {k: variants[k] for k in order}
    print("platform:", platform.platform(), "python", sys.version.split()[0], flush = True)
    print("variants:", loaded, flush = True)

    out = {
        "platform": platform.platform(),
        "python": sys.version,
        "variants": loaded,
        "part_a": [],
        "part_b": [],
        "part_c": [],
    }
    part_a(out["part_a"])
    part_b(variants, out["part_b"])
    part_c(variants, out["part_c"])

    # Harness integrity: controls must hold on every variant, else the setup (not a variant) is broken.
    bad = [r for r in out["part_b"] if r["scenario"] in ("loop4", "free") and r["ok"] is False]
    out["controls_ok"] = not bad
    Path(a.out).write_text(json.dumps(out, indent = 1))
    print("\n=== SUMMARY ===")
    for r in out["part_a"]:
        print(
            f"A other0.0.0.0 reuse={r['other_so_reuseaddr']} studio={r['studio_bind']}: "
            f"bind={r['studio_bind_result'][:20]} HIJACKED={r['hijacked']}"
        )
    for r in out["part_b"]:
        if r["scenario"] in ("wild4", "wild6only") or r["ok"] is False:
            print(
                f"B {r['scenario']} reuse={r['other_so_reuseaddr']} {r['variant']} {r['probe_host']}: "
                f"free={r['answers'][0]} ok={r['ok']} median={r['median_s']}s"
            )
    print("controls_ok:", out["controls_ok"])
    if bad:
        for r in bad:
            print("CONTROL FAILED:", r)
    sys.exit(0)


if __name__ == "__main__":
    main()
