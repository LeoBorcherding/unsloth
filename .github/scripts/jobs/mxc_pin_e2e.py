"""Windows MXC dual-pin end-to-end check (staging CI job, Windows legs only).

Runs the branch's own studio/install_mxc_prebuilt.py into a fresh Studio home, then checks:
  1. which pin Studio selected for this Windows (1.1.0 with the built-in container, else 1.0.1) and why
  2. the installer's result, the installed pin, `wxc-exec --probe` tier and any probe demotion
  3. Authenticode signatures of the installed binaries
  4. on a host without the built-in container: --prepare-host, then the opt-in fallback sandbox
  5. real sandboxed tool calls through tools.execute_tool in required mode (Python and Terminal): backend,
     effective mode, output and warm median latency
  6. optional: the branch's MXC tests

Usage: python .github/scripts/jobs/mxc_pin_e2e.py --tiny --out job_mxc_pin.json [--tests a.py,b.py]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path.cwd()
BACKEND = REPO / "studio" / "backend"
INSTALLER = REPO / "studio" / "install_mxc_prebuilt.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mxc_install_check import _ensure_deps, _pytest, _run, _signature  # noqa: E402

_CHILD = r"""
import json, os, statistics, sys, time
sys.path.insert(0, os.environ["TOOLS_BACKEND"])
import logging
logging.disable(logging.WARNING)
from core.inference import mxc_runtime, tools
out = {"pin": mxc_runtime.selected_pin().version, "release": mxc_runtime._windows_release()}
cases = {
    "python": ("python", {"code": "import sys\nprint('PY-OK', sys.version_info[:2])"}),
    "terminal": ("terminal", {"command": "echo TERM-OK"}),
}
for name, (tool, args) in cases.items():
    times, results = [], []
    for i in range(int(os.environ.get("TOOLS_N", "6"))):
        call = dict(args)
        if tool == "python":
            call["code"] = call["code"] + f"\n# {i}"
        started = time.perf_counter()
        text = tools.execute_tool(tool, call, session_id = None, tool_execution_mode = "required")
        times.append((time.perf_counter() - started) * 1000)
        results.append(text)
    record = getattr(tools, "_last_tool_execution_record", None)
    out[name] = {
        "backend": getattr(record, "backend", None),
        "effective_mode": getattr(record, "effective_mode", None),
        "first_ms": round(times[0], 1),
        "warm_median_ms": round(statistics.median(times[1:] or times), 1),
        "ok": all(("PY-OK" if tool == "python" else "TERM-OK") in r for r in results),
        "sample": results[-1][-400:],
    }
try:
    from core.inference import os_sandbox
    cap = os_sandbox.os_isolation_capability("python") if hasattr(os_sandbox, "os_isolation_capability") else None
    out["capability"] = None if cap is None else {"backend": cap.backend, "available": cap.available, "reason": cap.reason}
except Exception as exc:  # noqa: BLE001
    out["capability_error"] = f"{type(exc).__name__}: {exc}"
print("E2E " + json.dumps(out, default = str))
"""


def _expected_pin(release):
    if release is None:
        return None
    product_type, build, ubr = release
    floors = {26100: 9278, 26200: 9278, 26300: 9550, 28000: 2804}
    if product_type != 1:
        return "1.0.1"
    if build in floors:
        return "1.1.0" if ubr >= floors[build] else "1.0.1"
    return "1.1.0" if build > max(floors) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action = "store_true")
    p.add_argument("--out", default = "job_mxc_pin.json")
    p.add_argument("--tests", default = "")
    args = p.parse_args()
    out_path = Path(args.out).resolve()
    result = {"job": "mxc_pin_e2e", "passed": False, "checks": {}}
    result["host"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    checks = result["checks"]
    try:
        if sys.platform != "win32":
            result["skipped"] = "Windows only"
            result["passed"] = True
            return
        checks["deps"] = _ensure_deps()
        home = Path(tempfile.mkdtemp(prefix = "mxc-pin-home-"))
        env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home)}
        env.pop("UNSLOTH_MXC_PIN", None)
        sys.path.insert(0, str(BACKEND))
        os.environ["UNSLOTH_STUDIO_HOME"] = str(home)
        from core.inference import mxc_runtime

        release = mxc_runtime._windows_release()
        checks["windows_release"] = release
        checks["selected_before_install"] = mxc_runtime.selected_pin().version
        checks["expected_pin"] = _expected_pin(release)
        install_dir = mxc_runtime._installed_package_root()
        install = _run([sys.executable, str(INSTALLER), "--install-dir", str(install_dir)], env = env)
        checks["install"] = install
        checks["selected_after_install"] = mxc_runtime.selected_pin().version
        checks["demoted_by_probe"] = mxc_runtime.demoted_by_probe()
        installed = None
        for pin in mxc_runtime.PINS:
            try:
                mxc_runtime._validate_runtime(install_dir, pin)
                installed = pin.version
            except Exception:  # noqa: BLE001
                pass
        checks["installed_pin"] = installed
        checks["installed_files"] = sorted(x.name for x in install_dir.glob("*.exe"))
        checks["signatures"] = {
            name: _signature(install_dir / name) for name in checks["installed_files"]
        }
        probe = _run([str(install_dir / "wxc-exec.exe"), "--probe"], timeout = 120)
        tier = None
        try:
            tier = json.loads(probe["out"][probe["out"].index("{") :]).get("tier")
        except Exception:  # noqa: BLE001
            pass
        checks["probe_tier"] = tier
        child_env = {**env, "TOOLS_BACKEND": str(BACKEND), "TOOLS_N": "4" if args.tiny else "8"}
        if tier != "base-container":
            prep = _run(
                [
                    sys.executable,
                    str(INSTALLER),
                    "--prepare-host",
                    "--install-dir",
                    str(install_dir),
                ],
                env = env,
                timeout = 1800,
            )
            checks["prepare_host"] = prep
            child_env["UNSLOTH_MXC_ALLOW_DACL_FALLBACK"] = "1"
            child_env["UNSLOTH_MXC_PERSISTENT_READ_GRANTS"] = "1"
        work = Path(tempfile.mkdtemp(prefix = "mxc-pin-work-"))
        child = subprocess.run(
            [sys.executable, "-c", _CHILD],
            capture_output = True,
            text = True,
            env = child_env,
            cwd = work,
            timeout = 1800,
        )
        line = next((l for l in child.stdout.splitlines() if l.startswith("E2E ")), None)
        checks["e2e"] = (
            json.loads(line[4:])
            if line
            else {"rc": child.returncode, "out": (child.stdout + child.stderr)[-4000:]}
        )
        if args.tests:
            checks["pytest"] = _pytest([t for t in args.tests.split(",") if t])
        e2e = checks["e2e"]
        sigs = checks["signatures"]
        result["passed"] = bool(
            install["rc"] == 0
            and installed == checks["selected_after_install"]
            and (checks["expected_pin"] is None or installed == checks["expected_pin"])
            and sigs
            and all(
                s.get("status") == "Valid" and "Microsoft Corporation" in (s.get("signer") or "")
                for s in sigs.values()
            )
            and (installed != "1.1.0" or tier == "base-container")
            and all(
                e2e.get(name, {}).get("ok")
                and e2e.get(name, {}).get("backend") == "mxc-processcontainer"
                for name in ("python", "terminal")
            )
            and (not args.tests or (checks.get("pytest") or {}).get("rc") == 0)
        )
    except Exception as exc:  # noqa: BLE001
        import traceback
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
    finally:
        out_path.write_text(json.dumps(result, indent = 1, default = str))
        print(json.dumps(result, indent = 1, default = str)[:30000], flush = True)
        print(
            "JOB_RESULT " + json.dumps({"passed": result["passed"], "host": result["host"]}),
            flush = True,
        )
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
