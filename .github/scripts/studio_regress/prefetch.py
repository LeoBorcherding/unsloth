"""Download the fixture models a selection needs, once, into the shared HF cache.

    python studio_regress.py prefetch --only compaction,inference_settings
    python studio_regress.py prefetch --pr 11606
    python studio_regress.py prefetch --all

GGUF fixtures fetch only their variant (`*<variant>*.gguf` + configs) into the suite HF cache
(engine.SUITE_HF_HOME), which Studio reads with HF_HUB_OFFLINE=1. `local_snapshot = true` specs are
tiny diffusers pipelines Studio only loads from a path: they land in $WORKSPACE/hf_tiny/<name>
(journeys/_diffusion.ensure_tiny). `fetch_script` specs run a script from the scripts dir instead
(diffusion_bench/fetch_tiny.py fills the same hf_tiny root for the external diffusion_bench targets).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from studio_regress import selection


def fixtures_for(names, data):
    keys = []
    for t in data["target"]:
        if t["name"] in names:
            keys += [k for k in t.get("models") or [] if k not in keys]
    return {k: data["models"][k] for k in keys}


SCRIPTS = Path(__file__).resolve().parent.parent


def fetch(spec):
    if spec.get("fetch_script"):
        argv = [sys.executable, str(SCRIPTS / spec["fetch_script"]), *spec.get("args", [])]
        subprocess.run(argv, cwd=str(SCRIPTS), check=True)
        return " ".join(argv[1:])
    if spec.get("local_snapshot"):
        from studio_regress.journeys import _diffusion
        return _diffusion.ensure_tiny(spec["repo"])
    from huggingface_hub import snapshot_download
    allow = None
    if spec.get("variant"):
        allow = [f"*{spec['variant']}*.gguf", "*.json", "*.md"]
    from studio_regress.engine import SUITE_HF_HOME
    return snapshot_download(spec["repo"], allow_patterns=allow, cache_dir=str(SUITE_HF_HOME / "hub"))


def main(argv=None):
    p = argparse.ArgumentParser(description="Prefetch fixture models")
    p.add_argument("--pr")
    p.add_argument("--repo", default="unslothai/unsloth")
    p.add_argument("--only")
    p.add_argument("--all", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    data = selection.load()
    sel = {"selected": []}
    if a.pr and not a.only and not a.all:
        files, labels = selection.pr_files_and_labels(a.pr, a.repo)
        sel = selection.select(files, labels, data)
    names = selection.apply_overrides(sel, only=a.only.split(",") if a.only else None,
                                      all_targets=a.all, data=data)
    specs = fixtures_for(names, data)
    rc = 0
    for k, spec in specs.items():
        if a.dry_run:
            print(f"would fetch {k}: {spec}")
            continue
        try:
            print(f"{k}: {fetch(spec)}")
        except Exception as e:
            print(f"{k}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
