"""Journey 9: unsloth's own Playwright suites, run from EACH side's source checkout.

Base passes and head fails -> FAIL_HEAD (functional regression) through the normal diff.
Each script gets its own fresh-state Studio in bootstrap state (they drive first-run
change-password themselves) on a private port, with the tiny CI GGUF. Their screenshots are
kept as `_evidence_png` (reviewable, NOT pixel-diffed: the suites take full-page shots with
live content, which would only add noise; the pass / fail and the last reached step are
the signal).
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

from studio_regress import engine
from studio_regress.contract import Journey, Step, StepFailed, StepUnreachable

SCRIPTS = {"s01_chat_ui": "playwright_chat_ui.py", "s02_extra_ui": "playwright_extra_ui.py"}
TIMEOUT_S = 1200
_STEP_RX = re.compile(r"^\[ui\]\s*(?:==>|STEP|step)?\s*(.+)$")


def _last_step(text):
    last = None
    for line in text.splitlines():
        if line.startswith("[ui]") or line.startswith("STEP") or line.startswith("==>"):
            last = line.strip()[:160]
    return last


def _suite_step(step_id, script):
    async def act(ctx):
        src = ctx.state.get("src")
        if not src or not (Path(src) / "tests" / "studio" / script).exists():
            raise StepUnreachable(f"{script} not in this side's source ({src})")
        from pr_ui_scenes._common import pick_free_ports
        root = Path(ctx.state["root"])
        home = engine.state_home(ctx.state["install_home"],
                                 Path(os.environ.get("WORKSPACE", ".")) / "temp" / "studio_regress" /
                                 "state" / f"{root.name}_upstream")
        port = pick_free_ports(1, seed=f"{root.name}_{step_id}")[0]
        log_dir = Path(ctx.out_dir) / "upstream_ci"
        log_dir.mkdir(parents=True, exist_ok=True)
        inst = await asyncio.to_thread(engine.launch, home, port, log_dir / f"{step_id}_studio.log")
        port = inst.port   # launch moves to a fresh port if this one was taken
        art = log_dir / step_id
        art.mkdir(exist_ok=True)
        m = ctx.models.get("gguf_270m", {"repo": "unsloth/gemma-3-270m-it-GGUF", "variant": "UD-Q4_K_XL"})
        env = {**os.environ, "BASE_URL": f"http://127.0.0.1:{port}",
               "STUDIO_OLD_PW": inst.bootstrap_password or "",
               "STUDIO_NEW_PW": "UpstreamNew-2026!x", "GGUF_REPO": m["repo"],
               "GGUF_VARIANT": m.get("variant", "UD-Q4_K_XL"), "PW_ART_DIR": str(art),
               "STUDIO_UI_STRICT": "1", "PYTHONUNBUFFERED": "1"}
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(Path(src) / "tests" / "studio" / script),
                cwd=str(Path(src) / "tests" / "studio"), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), TIMEOUT_S)
                rc = proc.returncode
            except asyncio.TimeoutError:
                proc.kill()
                out, rc = b"", -9
        finally:
            await asyncio.to_thread(engine.stop, port)
        text = out.decode(errors="replace")
        (log_dir / f"{step_id}.log").write_text(text)
        root_side = Path(ctx.out_dir)
        pngs = sorted(str(p.relative_to(root_side)) for p in art.glob("*.png"))
        facts = {"script": script, "exit": rc, "passed": rc == 0, "_evidence_png": pngs,
                 "_last_step": _last_step(text)}
        if rc != 0:
            tail = "\n".join(text.splitlines()[-8:])
            raise StepFailed(f"{script} exit {rc}; last: {facts['_last_step']}\n{tail}"[:700])
        return facts
    act.__name__ = step_id
    return act


JOURNEY = Journey(
    name="upstream_ci", tier="model", needs=("gguf_270m",), routes=("/chat", "/settings"), independent=True,
    steps=tuple(Step(sid, _suite_step(sid, s), shot=False, timeout_s=TIMEOUT_S + 120)
                for sid, s in SCRIPTS.items()),
)
