"""Base-vs-head probe for unslothai/unsloth#13776 (MPS image budget sized against the Metal working set).

Three modes:
  arm     (--tree DIR --label L [--inject RAM,AVAIL,WS,INUSE]): import that tree's core.inference.diffusion_memory,
          print one JSON line: raw readings, snapshot, reserve kind, budgets, per-model refusal decisions.
  driver  (default): fetch the merge base and the PR head from GitHub (or --base-tree/--head-tree), run arms
          base,head,base,head (A/A) in subprocesses, compare, write --out. Apple Silicon: real Metal readings,
          then allocate+touch the head budget on MPS to show the bytes base refused are usable.
          Elsewhere: assert base == head on every non-Metal reading (cpu, mps-without-reading, cuda shapes).
Exit 0 = every assertion held; 1 = an assertion failed; 2 = harness fault (no MPS on an Apple Silicon leg, fetch
failure, import failure). A step that cannot fail proves nothing, so missing MPS on macOS arm64 is a FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

UPSTREAM = "https://github.com/unslothai/unsloth"
BASE_SHA = "97ed9415f7cc70ee6c8c71d4088bf44bf7058b33"
HEAD_SHA = "2adff795a5d6c98d5a6ab2af6a04627792bf6cb0"
PR = 13776

# Same sizes as the PR's test (Hub sizes of the unsloth GGUFs, bf16 Qwen3-4B encoder, VAEs).
TEXT_ENCODER_MIB = 7_672
MODELS = {
    "z-image Q2_K": (3_471, 159),
    "z-image Q3_K_S": (3_768, 159),
    "z-image Q4_K_M": (4_785, 159),
    "klein Q4_K_M": (2_483, 160),
    "klein Q6_K": (3_251, 160),
    "klein Q8_0": (4_101, 160),
}


# ----------------------------------------------------------------------------------------------- arm
def _arm(args) -> int:
    sys.path.insert(0, str(Path(args.tree).resolve()))
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # as _mps_or_cpu_target
    import core.inference.diffusion_memory as dm

    device = args.device
    raw = {}
    if args.inject:
        ram, avail, ws, inuse = (int(x) if x not in ("", "none") else None for x in args.inject.split(","))
        dm._system_memory_mib = lambda: (ram, avail)
        if hasattr(dm, "_metal_working_set_mib"):
            dm._metal_working_set_mib = lambda: ((ws, inuse or 0) if ws else (None, 0))
        raw = {"ram_mib": ram, "available_mib": avail, "working_set_mib": ws, "in_use_mib": inuse, "injected": True}
    else:
        ram, avail = dm._system_memory_mib()
        raw = {"ram_mib": ram, "available_mib": avail, "injected": False}
        try:
            import torch
            raw["torch"] = torch.__version__
            raw["mps_available"] = bool(torch.backends.mps.is_available())
            if raw["mps_available"]:
                raw["working_set_mib"] = int(torch.mps.recommended_max_memory()) // (1024 * 1024)
                raw["working_set_bytes"] = int(torch.mps.recommended_max_memory())
                raw["in_use_mib"] = int(torch.mps.driver_allocated_memory()) // (1024 * 1024)
        except Exception as exc:  # noqa: BLE001
            raw["torch_error"] = repr(exc)
    raw["has_metal_helper"] = hasattr(dm, "_metal_working_set_mib")

    calls = {"metal_helper": 0}
    if hasattr(dm, "_metal_working_set_mib"):
        inner = dm._metal_working_set_mib

        def counted():
            calls["metal_helper"] += 1
            return inner()

        dm._metal_working_set_mib = counted

    target = types.SimpleNamespace(device = device, backend = device, supports_model_cpu_offload = device == "cuda")
    if device == "cuda":
        free, total = (int(x) for x in args.cuda.split(","))
        dm_mem = dm.DeviceMemory("cuda", "cuda", args.cuda_kind, free, total)
        snap = dm_mem
    else:
        snap = dm.snapshot_device_memory(target)
    out = {
        "label": args.label,
        "device": device,
        "raw": raw,
        "snapshot": {
            "memory_kind": snap.memory_kind,
            "free_mib": snap.free_mib,
            "total_mib": snap.total_mib,
            "metal_working_set_mib": getattr(snap, "metal_working_set_mib", None),
        },
        "reserve_kind": dm._budget_reserve_kind(snap),
        "safe_budget_mib": dm._safe_device_budget_mib(snap),
        "fast_budget_mib": dm._fast_device_budget_mib(snap),
        "capacity_budget_mib": dm.total_capacity_budget_mib(snap),
        "metal_helper_calls": calls["metal_helper"],
    }
    decisions = {}
    sizes = dict(MODELS)
    for extra in args.extra_model or []:
        name, mib = extra.rsplit("=", 1)
        sizes[name] = (int(mib), 0, 0)
    for name, spec in sizes.items():
        if len(spec) == 3:  # synthetic: weights only, no companions
            weights, companions, te = spec[0], None, None
        else:
            gguf, vae = spec
            weights = dm.estimate_gguf_resident_mib(gguf) + TEXT_ENCODER_MIB + vae
            companions, te = TEXT_ENCODER_MIB + vae, TEXT_ENCODER_MIB
        plan = dm.plan_diffusion_memory(
            target = target,
            device_memory = snap,
            model_dense_mib = weights,
            companion_dense_mib = companions,
            text_encoder_dense_mib = te,
            runtime_headroom_mib = dm.estimate_image_runtime_mib(width = None, height = None),
        )
        msg = dm.unified_memory_shortfall_message(plan, family = name.split()[0])
        decisions[name] = {
            "required_mib": int(weights) + int(plan.estimates.get("base_overhead_mib") or 0),
            "policy": plan.offload_policy,
            "refused": msg is not None,
            "message": msg,
        }
    out["decisions"] = decisions
    # per-call cost of the snapshot (the PR adds two torch.mps calls to it)
    if args.time_snapshot and device == "mps":
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            dm.snapshot_device_memory(target)
        out["snapshot_us_per_call"] = round((time.perf_counter() - t0) / n * 1e6, 2)
    print("ARM_JSON " + json.dumps(out))
    return 0


# -------------------------------------------------------------------------------------------- driver
def _sh(cmd, cwd = None, check = True):
    print("+ " + " ".join(cmd), flush = True)
    r = subprocess.run(cmd, cwd = cwd, capture_output = True, text = True)
    if check and r.returncode != 0:
        print(r.stdout + r.stderr, flush = True)
        raise SystemExit(2)
    return r


def _fetch_tree(root: Path, sha: str, ref: str | None, dest: Path) -> Path:
    if (dest / "core" / "inference" / "diffusion_memory.py").is_file():
        return dest
    repo = root / "_probe_git"
    if not (repo / ".git").exists():
        repo.mkdir(parents = True, exist_ok = True)
        _sh(["git", "init", "-q"], cwd = repo)
    for spec in ([sha] + ([ref] if ref else [])):
        r = _sh(["git", "fetch", "-q", "--depth=1", UPSTREAM, spec], cwd = repo, check = False)
        if r.returncode == 0:
            got = _sh(["git", "rev-parse", "FETCH_HEAD"], cwd = repo).stdout.strip()
            if got == sha:
                break
    else:
        print(f"could not fetch {sha}", flush = True)
        raise SystemExit(2)
    dest.mkdir(parents = True, exist_ok = True)
    arch = subprocess.run(["git", "archive", sha, "studio/backend"], cwd = repo, capture_output = True, check = True)
    subprocess.run(["tar", "-x", "-C", str(dest), "--strip-components=2"], input = arch.stdout, check = True)
    return dest


def _run_arm(tree: Path, label: str, extra: list[str]) -> dict:
    cmd = [sys.executable, __file__, "arm", "--tree", str(tree), "--label", label, *extra]
    r = subprocess.run(cmd, capture_output = True, text = True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("ARM_JSON ")), None)
    if r.returncode != 0 or line is None:
        print(r.stdout[-4000:] + r.stderr[-4000:], flush = True)
        raise SystemExit(2)
    return json.loads(line[len("ARM_JSON "):])


def _mps_alloc_proof(mib: int) -> dict:
    """Allocate ``mib`` MiB on MPS in 256 MiB blocks, touch every block, run a matmul over each, sync."""
    if os.environ.get("PROBE_FAKE_APPLE") == "1":
        return {"requested_mib": mib, "ok": True, "fake": True}
    import torch
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    t0 = time.perf_counter()
    blocks, done = [], 0
    try:
        while done < mib:
            take = min(256, mib - done)
            n = take * 1024 * 1024 // 2
            b = torch.empty(n, dtype = torch.float16, device = "mps")
            b.fill_(1.0)
            blocks.append(b)
            done += take
        torch.mps.synchronize()
        alloc_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        acc = torch.zeros((), device = "mps", dtype = torch.float32)
        for b in blocks:
            side = int((b.numel()) ** 0.5)
            m = b[: side * side].view(side, side)[:1024, :1024]
            acc = acc + (m @ m).float().sum()
        torch.mps.synchronize()
        compute_s = time.perf_counter() - t1
        driver = int(torch.mps.driver_allocated_memory()) // (1024 * 1024)
        ok = True
        err = None
    except Exception as exc:  # noqa: BLE001
        ok, err, alloc_s, compute_s, driver = False, repr(exc), time.perf_counter() - t0, None, None
    del blocks
    torch.mps.empty_cache()
    return {"requested_mib": mib, "allocated_mib": done, "ok": ok, "error": err,
            "alloc_s": round(alloc_s, 3), "compute_s": None if compute_s is None else round(compute_s, 3),
            "driver_allocated_mib_at_peak": driver}


def _driver(args) -> int:
    root = Path(args.workdir).resolve()
    root.mkdir(parents = True, exist_ok = True)
    base = Path(args.base_tree) if args.base_tree else _fetch_tree(root, BASE_SHA, None, root / "base")
    head = Path(args.head_tree) if args.head_tree else _fetch_tree(root, HEAD_SHA, f"refs/pull/{PR}/head", root / "head")
    extra_trees = []
    for spec in args.extra_tree or []:
        lab, p = spec.split("=", 1)
        extra_trees.append((lab, Path(p)))
    fix = None
    if args.fix_arm:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mps_budget_probe_fixhunk import NEW, OLD
        fix = root / "fix"
        if fix.exists():
            shutil.rmtree(fix)
        shutil.copytree(head, fix)
        p = fix / "core" / "inference" / "diffusion_memory.py"
        src = p.read_text()
        if src.count(OLD) != 1:
            print("fix hunk does not apply exactly once to the head tree", flush = True)
            return 2
        p.write_text(src.replace(OLD, NEW))
    is_apple = (sys.platform == "darwin" and platform.machine() == "arm64") or os.environ.get("PROBE_FAKE_APPLE") == "1"
    report = {"host": {"platform": sys.platform, "machine": platform.machine(), "python": sys.version.split()[0],
                       "runner": os.environ.get("RUNNER_NAME"), "image": os.environ.get("ImageOS"),
                       "image_version": os.environ.get("ImageVersion")},
              "base_sha": BASE_SHA, "head_sha": HEAD_SHA, "checks": []}
    failures = []

    def check(name, ok, detail):
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + " :: " + json.dumps(detail), flush = True)
        if not ok:
            failures.append(name)

    if is_apple:
        import torch
        if not torch.backends.mps.is_available():
            print("FAIL: Apple Silicon leg without MPS: nothing here can be measured", flush = True)
            return 2
        arms = []
        for lab, tree in [("base", base), ("head", head), ("base", base), ("head", head)] + \
                         [(l, t) for l, t in extra_trees]:
            arms.append(_run_arm(tree, lab, ["--device", "mps", "--time-snapshot"]))
        report["real_arms"] = arms
        b1, h1, b2, h2 = arms[:4]
        ws = h1["raw"].get("working_set_mib")
        check("real working set read from torch.mps.recommended_max_memory", bool(ws), h1["raw"])
        check("head snapshot carries the working set (new path ran)",
              h1["snapshot"]["metal_working_set_mib"] == ws and h1["reserve_kind"] == "metal_working_set"
              and h1["metal_helper_calls"] >= 1, h1["snapshot"] | {"reserve_kind": h1["reserve_kind"]})
        check("base snapshot is the RAM reading", b1["snapshot"]["metal_working_set_mib"] is None
              and b1["reserve_kind"] == "unified_memory", b1["snapshot"])
        noise = abs(b1["safe_budget_mib"] - b2["safe_budget_mib"])
        noise_h = abs(h1["safe_budget_mib"] - h2["safe_budget_mib"])
        report["aa_noise_mib"] = {"base": noise, "head": noise_h}
        # head bound: min(avail, ws - in_use) - 15% ws ; computed from the arm's own readings
        exp_free = min(h1["raw"]["available_mib"], ws - h1["raw"]["in_use_mib"])
        check("head free = min(available RAM, working set - in use)",
              abs(h1["snapshot"]["free_mib"] - exp_free) <= max(64, noise_h + 64),
              {"head_free": h1["snapshot"]["free_mib"], "expected": exp_free})
        delta = h1["safe_budget_mib"] - b1["safe_budget_mib"]
        report["budget_delta_mib"] = delta
        print(f"REAL budgets: base {b1['safe_budget_mib']} / {b2['safe_budget_mib']} MiB, head "
              f"{h1['safe_budget_mib']} / {h2['safe_budget_mib']} MiB, delta {delta} (A/A noise {noise}/{noise_h})",
              flush = True)
        # a model RAM-minus-20% refuses on THIS machine's real readings but the working set holds:
        # weights sized so required = midpoint of the two budgets, then decided by both trees, then allocated.
        if delta > 2 * max(noise, noise_h, 64):
            overhead = 2048
            mid_required = (b1["safe_budget_mib"] + h1["safe_budget_mib"]) // 2
            weights = mid_required - overhead
            if weights > 0:
                syn = [f"synthetic-midpoint={weights}"]
                bm = _run_arm(base, "base", ["--device", "mps", *sum([["--extra-model", s] for s in syn], [])])
                hm = _run_arm(head, "head", ["--device", "mps", *sum([["--extra-model", s] for s in syn], [])])
                bd, hd = bm["decisions"]["synthetic-midpoint"], hm["decisions"]["synthetic-midpoint"]
                report["midpoint"] = {"weights_mib": weights, "required_mib": hd["required_mib"],
                                      "base_budget": bm["safe_budget_mib"], "head_budget": hm["safe_budget_mib"],
                                      "base_refused": bd["refused"], "head_refused": hd["refused"],
                                      "base_message": bd["message"]}
                check("decision flips on real readings: base refuses, head admits",
                      bd["refused"] and not hd["refused"], report["midpoint"])
                proof = _mps_alloc_proof(hd["required_mib"])
                report["alloc_proof"] = proof
                check("the bytes base refused are allocatable + usable on real Metal", proof["ok"], proof)
            else:
                check("midpoint model has positive weights", False, {"mid_required": mid_required})
        else:
            check("real budgets differ beyond A/A noise", False, {"delta": delta, "noise": [noise, noise_h]})
        # Reporter readings (24 GiB MacBook Air) through both trees, same as the PR table.
        rep = "24576,17203,16384,0"
        rb = _run_arm(base, "base", ["--device", "mps", "--inject", rep])
        rh = _run_arm(head, "head", ["--device", "mps", "--inject", rep])
        report["reporter"] = {"base": rb, "head": rh}
        flips = {n: (rb["decisions"][n]["refused"], rh["decisions"][n]["refused"]) for n in MODELS}
        check("reporter readings: base refuses all 6, head admits Z-Image Q2_K/Q3_K_S, Klein Q4_K_M/Q6_K",
              all(v[0] for v in flips.values())
              and [n for n, v in flips.items() if not v[1]] == ["z-image Q2_K", "z-image Q3_K_S", "klein Q4_K_M",
                                                                "klein Q6_K"],
              {n: {"base_refused": a, "head_refused": b} for n, (a, b) in flips.items()})
        if fix is not None:
            # real readings: fix arm between two base arms, so its budget is compared to a bracketing base
            fb1 = _run_arm(base, "base", ["--device", "mps"])
            fx = _run_arm(fix, "fix", ["--device", "mps", "--time-snapshot"])
            fb2 = _run_arm(base, "base", ["--device", "mps"])
            fh = _run_arm(head, "head", ["--device", "mps"])
            report["fix_real"] = {"base_before": fb1, "fix": fx, "base_after": fb2, "head": fh}
            lo_base = min(fb1["safe_budget_mib"], fb2["safe_budget_mib"])
            band = abs(fb1["safe_budget_mib"] - fb2["safe_budget_mib"]) + 64
            check("fix (real readings): budget never below base (bracketed A/A)",
                  fx["safe_budget_mib"] >= lo_base - band,
                  {"fix": fx["safe_budget_mib"], "base": [fb1["safe_budget_mib"], fb2["safe_budget_mib"]],
                   "head": fh["safe_budget_mib"], "fix_reading": fx["snapshot"], "fix_reserve": fx["reserve_kind"]})
            check("fix (real readings): metal helper ran", fx["metal_helper_calls"] >= 1, fx["metal_helper_calls"])
            rf = _run_arm(fix, "fix", ["--device", "mps", "--inject", rep])
            report["reporter"]["fix"] = rf
            check("fix (reporter readings): same decisions as head",
                  {n: rf["decisions"][n]["refused"] for n in MODELS} == {n: rh["decisions"][n]["refused"] for n in MODELS},
                  {n: rf["decisions"][n]["refused"] for n in MODELS})
            # idle 64 GB Mac (available above the working set): head narrows, fix keeps base
            idle = "65536,57671,49152,0"
            ib = _run_arm(base, "base", ["--device", "mps", "--inject", idle, "--extra-model", "gap=42000"])
            ih = _run_arm(head, "head", ["--device", "mps", "--inject", idle, "--extra-model", "gap=42000"])
            ifx = _run_arm(fix, "fix", ["--device", "mps", "--inject", idle, "--extra-model", "gap=42000"])
            got = {k: (v["safe_budget_mib"], v["decisions"]["gap"]["refused"]) for k, v in
                   (("base", ib), ("head", ih), ("fix", ifx))}
            report["idle64"] = got
            check("idle 64 GB Mac, 44,048 MiB load: base admits, head refuses, fix admits",
                  got["base"][1] is False and got["head"][1] is True and got["fix"][1] is False, got)
    else:
        # Not Apple Silicon: the PR must be byte-identical on every reading it can take here.
        cells = [
            ("cpu", ["--device", "cpu"]),
            ("mps-no-reading", ["--device", "mps"]),  # torch here has no MPS reading: fail-open to RAM
            ("cuda-discrete", ["--device", "cuda", "--cuda", "20000,24564", "--cuda-kind", "discrete_vram"]),
            ("cuda-unified", ["--device", "cuda", "--cuda", "60000,122880", "--cuda-kind", "unified_memory"]),
        ]
        for (name, extra), (hlab, htree) in [(c, t) for c in cells
                                               for t in [("head", head)] + ([("fix", fix)] if fix else [])]:
            name = f"{name} [{hlab}]"
            b = _run_arm(base, "base", extra)
            h = _run_arm(htree, hlab, extra)
            keys = ("snapshot", "reserve_kind", "safe_budget_mib", "fast_budget_mib", "capacity_budget_mib")
            bb = {k: b[k] for k in keys}
            hh = {k: h[k] for k in keys}
            bb["decisions"] = {n: d["refused"] for n, d in b["decisions"].items()}
            hh["decisions"] = {n: d["refused"] for n, d in h["decisions"].items()}
            # RAM moves between the two subprocesses: compare the base arm against a head arm fed the SAME reading
            if name.split()[0] in ("cpu", "mps-no-reading"):
                inj = f"{b['raw']['ram_mib']},{b['raw']['available_mib']},none,0"
                b = _run_arm(base, "base", extra + ["--inject", inj])
                h = _run_arm(htree, hlab, extra + ["--inject", inj])
                bb = {k: b[k] for k in keys} | {"decisions": {n: (d["refused"], d["message"]) for n, d in b["decisions"].items()}}
                hh = {k: h[k] for k in keys} | {"decisions": {n: (d["refused"], d["message"]) for n, d in h["decisions"].items()}}
            # messages on a non-Metal unified device keep base's wording except the documented decimal/floor text
            bcmp = dict(bb); hcmp = dict(hh)
            bcmp["snapshot"] = {k: v for k, v in bb["snapshot"].items() if k != "metal_working_set_mib"}
            hcmp["snapshot"] = {k: v for k, v in hh["snapshot"].items() if k != "metal_working_set_mib"}
            if "decisions" in bcmp and bcmp["decisions"] and isinstance(next(iter(bcmp["decisions"].values())), tuple):
                bcmp["decisions"] = {n: v[0] for n, v in bcmp["decisions"].items()}
                hcmp["decisions"] = {n: v[0] for n, v in hcmp["decisions"].items()}
            check(f"{name}: head budget/decisions == base", bcmp == hcmp, {"base": bcmp, "head": hcmp,
                                                                          "head_helper_calls": h["metal_helper_calls"]})
    report["failures"] = failures
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent = 1))
    print(f"SUMMARY {len(report['checks']) - len(failures)}/{len(report['checks'])} checks passed", flush = True)
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs = "?", default = "driver", choices = ["driver", "arm"])
    p.add_argument("--tree")
    p.add_argument("--label", default = "?")
    p.add_argument("--device", default = "mps")
    p.add_argument("--inject")
    p.add_argument("--cuda", default = "20000,24564")
    p.add_argument("--cuda-kind", default = "discrete_vram")
    p.add_argument("--extra-model", action = "append")
    p.add_argument("--time-snapshot", action = "store_true")
    p.add_argument("--base-tree")
    p.add_argument("--head-tree")
    p.add_argument("--extra-tree", action = "append", help = "LABEL=DIR, extra arm on Apple Silicon")
    p.add_argument("--workdir", default = "probe_13776")
    p.add_argument("--out")
    p.add_argument("--fix-arm", action = "store_true", help = "also run head + mps_budget_probe_fixhunk")
    p.add_argument("--tiny", action = "store_true")  # staging forces --tiny on non-Apple legs
    args = p.parse_args()
    return _arm(args) if args.mode == "arm" else _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
