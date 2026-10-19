"""Apple Silicon: a same-model MLX reload with a different KV width, from the real MLX backend into
the Studio frontend store, base vs head (unsloth#8039 / #14210).

Loads a tiny MLX model twice with different `kv_quant` values through Studio's MLXInferenceBackend,
maps each resident record onto the /api/inference/status fields exactly as routes/inference.py
does, and feeds the two statuses through applyActiveModelStatusToStore twice: once with the
checkout's file (head) and once with the file at --base-sha. Run from the repo root.

    python jobs/mac_mlx_same_model_reload.py --base-sha <merge base> --out result.json

Checks: the backend reports both widths; base keeps the first width in the control (the defect);
head moves control, baseline and verdict to the second.
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
import urllib.request
from pathlib import Path

TINY = "mlx-community/SmolLM-135M-Instruct-4bit"
STORE_FILE = "studio/frontend/src/features/chat/lib/apply-inference-status-to-store.ts"
PROBE = "studio/frontend/tests/zz_mac_reload_probe.ts"
PROBE_SRC = r"""
import { register } from "node:module";
import { installLocalStorageFake } from "./helpers/kit.ts";
installLocalStorageFake();
register("./store-settings-resolver.mjs", import.meta.url);
const { useChatRuntimeStore } = await import("../src/features/chat/stores/chat-runtime-store.ts");
const { applyActiveModelStatusToStore } = await import("../src/features/chat/lib/apply-inference-status-to-store.ts");
const [first, second] = JSON.parse(process.env.PROBE_STATUSES ?? "[]");
useChatRuntimeStore.setState({
  modelLoading: false,
  activeGgufVariant: null,
  params: { ...useChatRuntimeStore.getState().params, checkpoint: "" },
});
applyActiveModelStatusToStore(first, { previousCheckpoint: "" });
useChatRuntimeStore.setState({
  params: { ...useChatRuntimeStore.getState().params, checkpoint: first.model_identifier },
});
const pick = () => {
  const s = useChatRuntimeStore.getState();
  return {
    mlxKvQuant: s.mlxKvQuant,
    loadedMlxKvQuantRequested: s.loadedMlxKvQuantRequested,
    mlxKvQuantReason: s.mlxKvQuantReason,
  };
};
const afterFirst = pick();
applyActiveModelStatusToStore(second, { previousCheckpoint: first.model_identifier });
console.log("PROBE_RESULT " + JSON.stringify({ afterFirst, afterSecond: pick() }));
"""


def status_from(info, name):
    # Same keys routes/inference.py fills from the resident record.
    return {
        "active_model": name,
        "model_identifier": name,
        "is_gguf": False,
        "is_mlx": True,
        "gguf_variant": None,
        "mlx_kv_quant_requested": info.get("mlx_kv_quant_requested"),
        "mlx_kv_quant_reason": info.get("mlx_kv_quant_reason"),
        "mlx_kv_quant_note": info.get("mlx_kv_quant_note"),
        "mlx_int8_prefill_requested": info.get("mlx_int8_prefill_requested"),
        "chat_template_override": info.get("chat_template_override_requested"),
        "chat_template_override_reason": info.get("chat_template_override_reason"),
    }


def load_status(kv_quant):
    from core.inference.mlx_inference import MLXInferenceBackend
    from utils.models.model_config import ModelConfig

    backend = MLXInferenceBackend()
    assert backend.load_model(
        ModelConfig.from_identifier(TINY), max_seq_length = 512, kv_quant = kv_quant
    )
    name = backend.active_model_name
    status = status_from(backend.models[name], name)
    backend.unload_model(name)
    return status


def probe(statuses):
    env = dict(os.environ, PROBE_STATUSES = json.dumps(statuses))
    out = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", PROBE.split("studio/frontend/")[1]],
        cwd = "studio/frontend",
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    line = next((l for l in out.stdout.splitlines() if l.startswith("PROBE_RESULT ")), None)
    if line is None:
        raise RuntimeError(
            f"probe failed rc={out.returncode}\n{out.stdout[-2000:]}\n{out.stderr[-4000:]}"
        )
    return json.loads(line[len("PROBE_RESULT ") :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-sha", required = True)
    ap.add_argument("--out", default = "mac_mlx_same_model_reload.json")
    ap.add_argument("--first", default = "8")
    ap.add_argument("--second", default = "4")
    a = ap.parse_args()

    result = {
        "job": "mac_mlx_same_model_reload",
        "model": TINY,
        "base_sha": a.base_sha,
        "host": f"{platform.system()} {platform.machine()}",
        "checks": {},
        "passed": False,
        "error": None,
        "skipped": False,
    }

    def check(name, ok, detail):
        result["checks"][name] = {"ok": bool(ok), "detail": detail}

    def write():
        Path(a.out).write_text(json.dumps(result, indent = 2))

    if sys.platform != "darwin" or platform.machine() != "arm64":
        result["skipped"] = True
        write()
        print("JOB_RESULT " + json.dumps(result))
        return 3
    t0 = time.time()
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-r",
                "studio/backend/requirements/studio.txt",
            ],
            check = True,
        )
        sys.path.insert(0, str(Path("studio/backend").resolve()))
        first, second = load_status(a.first), load_status(a.second)
        result["statuses"] = [first, second]
        check(
            "backend_reports_both_widths",
            first["mlx_kv_quant_requested"] == a.first
            and second["mlx_kv_quant_requested"] == a.second,
            [first["mlx_kv_quant_requested"], second["mlx_kv_quant_requested"]],
        )

        subprocess.run(
            ["npm", "ci", "--no-audit", "--no-fund", "--ignore-scripts"],
            cwd = "studio/frontend",
            check = True,
        )
        Path(PROBE).write_text(PROBE_SRC)
        result["head"] = probe([first, second])
        head_file = Path(STORE_FILE)
        saved = head_file.read_text()
        url = f"https://raw.githubusercontent.com/unslothai/unsloth/{a.base_sha}/{STORE_FILE}"
        with urllib.request.urlopen(url, timeout = 60) as r:
            head_file.write_text(r.read().decode())
        try:
            result["base"] = probe([first, second])
        finally:
            head_file.write_text(saved)
            Path(PROBE).unlink(missing_ok = True)

        b, h = result["base"]["afterSecond"], result["head"]["afterSecond"]
        check(
            "both_seed_first_width",
            result["base"]["afterFirst"]["mlxKvQuant"] == a.first
            and result["head"]["afterFirst"]["mlxKvQuant"] == a.first,
            [result["base"]["afterFirst"], result["head"]["afterFirst"]],
        )
        check("base_keeps_stale_width", b["mlxKvQuant"] == a.first, b)
        check(
            "head_follows_reload",
            h["mlxKvQuant"] == a.second
            and h["loadedMlxKvQuantRequested"] == a.second
            and h["mlxKvQuantReason"] == second["mlx_kv_quant_reason"],
            h,
        )
        result["passed"] = all(c["ok"] for c in result["checks"].values())
    except Exception as exc:  # recorded, then the job fails
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["seconds"] = round(time.time() - t0, 1)
    write()
    print("JOB_RESULT " + json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
