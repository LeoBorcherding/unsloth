# SPDX-License-Identifier: AGPL-3.0-only
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

BASE = "c62288858b3ca6f298b87bf1ccfeb362f7ec77ee"
HEAD = "f1931d2da3e5aef403a222b436c8d1c38cd79cdf"
FILES = ["tests/test_pp_fsdp2_gloo.py", "tests/test_pp_native_scaler_fused_gloo.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required = True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    result = {"base": BASE, "head": HEAD, "arms": {}, "passed": False}
    root = Path(tempfile.mkdtemp(prefix = "pr16417-", dir = os.environ.get("RUNNER_TEMP")))
    try:
        from huggingface_hub import hf_hub_download

        for repo in (
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "trl-internal-testing/tiny-Qwen3ForCausalLM",
            "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
        ):
            hf_hub_download(repo, "config.json")
        env = dict(
            os.environ,
            HF_HUB_OFFLINE = "1",
            TRANSFORMERS_OFFLINE = "1",
            CUDA_VISIBLE_DEVICES = "",
            GLOO_SOCKET_IFNAME = "lo0",
            PYTHONUNBUFFERED = "1",
            OMP_NUM_THREADS = "1",
            MKL_NUM_THREADS = "1",
        )
        env.pop("UNSLOTH_COMPILE_DISABLE", None)
        for name, sha in (("head", HEAD), ("base", BASE)):
            subprocess.run(
                ["git", "fetch", "https://github.com/unslothai/unsloth.git", sha], check = True
            )
            tree = root / name
            subprocess.run(["git", "worktree", "add", "--detach", str(tree), sha], check = True)
        for file in FILES:
            shutil.copy2(root / "head" / file, root / "base" / file)
        for name in ("base", "head"):
            tree = root / name
            env["PYTHONPATH"] = str(tree)
            xml = root / f"{name}.xml"
            command = [
                sys.executable,
                "-m",
                "pytest",
                *FILES,
                "-o",
                "addopts=",
                "-vv",
                "-s",
                "--tb=short",
                "--durations=20",
                "--timeout=600",
                "--timeout-method=thread",
                f"--junitxml={xml}",
            ]
            print(f"ARM {name} {result[name]} imports from {tree}", flush = True)
            start = time.monotonic()
            log = root / f"{name}.log"
            with log.open("w", encoding = "utf-8") as stream:
                proc = subprocess.Popen(
                    command,
                    cwd = tree,
                    env = env,
                    stdout = subprocess.PIPE,
                    stderr = subprocess.STDOUT,
                    text = True,
                )
                for line in proc.stdout:
                    print(f"{name}| {line}", end = "", flush = True)
                    stream.write(line)
                code = proc.wait()
            cases = {}
            for case in ET.parse(xml).getroot().iter("testcase"):
                key = case.get("name")
                failed = list(case.findall("failure")) + list(case.findall("error"))
                cases[key] = {
                    "status": "failed"
                    if failed
                    else "skipped"
                    if case.find("skipped") is not None
                    else "passed",
                    "failure": "\n".join(item.text or "" for item in failed),
                }
            result["arms"][name] = {
                "returncode": code,
                "seconds": time.monotonic() - start,
                "cases": cases,
                "log": log.read_text(encoding = "utf-8"),
            }
            out.write_text(json.dumps(result, indent = 2), encoding = "utf-8")
        base, head = (result["arms"][n]["cases"] for n in ("base", "head"))
        result["new_failures"] = [
            key
            for key, value in head.items()
            if value["status"] != "passed" and base.get(key, {}).get("status") == "passed"
        ]
        result["passed"] = (
            len(head) >= 15
            and set(base) == set(head)
            and not result["new_failures"]
            and all(
                head[key]["status"] == "passed"
                for key in (
                    "test_optimizer_skip_and_scale_agreement",
                    "test_explicit_pp_unscale_agrees_once",
                    "test_update_removes_hooks_and_releases_pipeline",
                )
            )
            and base["test_explicit_pp_unscale_agrees_once"]["status"] == "failed"
        )
    finally:
        out.write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
