"""Side execution: fresh-state homes, Studio launch, auth, and the per-step loop.

Public helpers (forks B/C journeys and harnesses use these):

  state_home(install_home, dest)      overlay home: heavy install dirs symlinked from the
                                      (shared, read-only) install, mutable state (auth/, *.db,
                                      assets, logs, settings) fresh. Same install on both sides
                                      is then a true A/A control, and N sessions share 1 install.
  launch(home, port, log, env)        `unsloth studio -p PORT` in its own session / process group
  api_login(base_url, user, pw)       -> access_token (raises on failure)
  rotate_bootstrap(base_url, home, new_pw)  bootstrap login + change-password over the API
  authed_context(browser, base_url, token)  fixture context with the SPA's auth keys seeded
  run_journey(journey, ctx, out_dir)  frozen step loop: action, capture, facts (with _status)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from studio_regress import fixture, plat  # noqa: E402
from studio_regress.contract import Ctx, StepFailed, StepUnreachable  # noqa: E402

# Install entries shared by symlink; everything else in a state home is created fresh.
SHARED_ENTRIES = ("unsloth_studio", "bin", "llama.cpp", "share", ".unsloth-studio-owned",
                  ".uidiff_sha", ".llama.cpp.install.lock")
SHARED_PREFIXES = (".venv_t5_", ".venv")
# <home>/cache holds hub-state (persisted Hub UI state) and uv / triton scratch: fresh per side.
# Models come from SUITE_HF_HOME, not from here.
SHARED_CACHE = ("cache",)

WS = Path(os.environ.get("WORKSPACE") or SCRIPTS.parent.parent.parent)
# Playwright's browsers for this suite live in the workspace (run.py's isolated path already
# points there). Without this default, a shared side looks under ~/.cache/ms-playwright, which
# holds another playwright version's build, and every side fails at chromium.launch.
if not os.environ.get("PLAYWRIGHT_BROWSERS_PATH") and (WS / "temp" / "pw_browsers").is_dir():
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(WS / "temp" / "pw_browsers")
# One suite-owned HF cache holding only prefetched fixtures: the Hub "On device" list and every
# model load are then identical on both sides and across runs (the host HF cache changes under
# us as other sessions download). Studio runs offline by default; `--online` / a journey's
# STUDIO_ENV can lift it.
SUITE_HF_HOME = WS / "temp" / "studio_regress" / "hf_home"

# Studio also scans ~/.cache/huggingface/hub and LM Studio dirs under $HOME for "On device"
# models; a suite HOME keeps the host's downloads (which change under us) out of the UI.
SUITE_HOME = WS / "temp" / "studio_regress" / "home"

STUDIO_ENV = {
    "HOME": str(SUITE_HOME),
    "HF_HOME": str(SUITE_HF_HOME),
    "HF_HUB_CACHE": str(SUITE_HF_HOME / "hub"),
    "HF_XET_CACHE": str(SUITE_HF_HOME / "xet"),
    # Legacy / sibling cache vars the host sets: Studio scans them for "On device" models too.
    "HUGGINGFACE_HUB_CACHE": str(SUITE_HF_HOME / "hub"),
    "TRANSFORMERS_CACHE": str(SUITE_HF_HOME / "hub"),
    "HF_DATASETS_CACHE": str(SUITE_HF_HOME / "datasets"),
    "HF_ASSETS_CACHE": str(SUITE_HF_HOME / "assets"),
    "HF_MODULES_CACHE": str(SUITE_HF_HOME / "modules"),
    "XDG_CACHE_HOME": str(SUITE_HOME / ".cache"),
    "HF_HUB_OFFLINE": "1",
    "UNSLOTH_DISABLE_UPDATE_CHECK": "1",
    "UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK": "1",
    "UNSLOTH_HELPER_MODEL_DISABLE": "1",
    "UNSLOTH_STUDIO_DISABLE_TORCH_WARM": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def state_home(install_home: Path, dest: Path, share_cache: bool = False, venv_shim: bool = False) -> Path:
    """venv_shim: make <dest>/unsloth_studio a real venv dir (own bin + pyvenv.cfg, shared lib) so
    sys.prefix equals $UNSLOTH_STUDIO_HOME/unsloth_studio. The `unsloth run` / `unsloth studio`
    CLI compares the two unresolved and re-execs forever through a plain symlink."""
    install_home, dest = Path(install_home).resolve(), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for entry in install_home.iterdir():
        name = entry.name
        if venv_shim and name == "unsloth_studio":
            _venv_shim(entry, dest / name)
        elif name in SHARED_ENTRIES or name.startswith(SHARED_PREFIXES) or (
                share_cache and name in SHARED_CACHE):
            plat.link(dest / name, entry)
    return dest


def _venv_shim(venv: Path, dest: Path):
    dest.mkdir()
    (dest / "bin").mkdir()
    for entry in venv.iterdir():
        if entry.name != "bin":
            plat.link(dest / entry.name, entry)
    old, new = str(venv / "bin"), str(dest / "bin")
    for f in (venv / "bin").iterdir():
        out = dest / "bin" / f.name
        if f.is_symlink():                      # python -> /usr/bin/python3.13
            out.symlink_to(os.readlink(f))
            continue
        head = f.read_bytes()[:512]
        if old.encode() in head and not f.name.startswith("activate"):
            out.write_bytes(f.read_bytes().replace(old.encode(), new.encode(), 2))
            out.chmod(f.stat().st_mode)
        else:
            plat.link(out, f)


# A Studio that lost the bind: uvicorn's errno 98 (Linux) / 48 (macOS) text, WinError 10048, and
# Studio's own "Port N is already in use ... will use port M instead" when it moves on.
_LOST_BIND = ("address already in use", "only one usage of each socket address")
_MOVED_PORT = re.compile(r"\bport \d+ is already in use\b")


def _lost_bind(log_path) -> bool:
    try:
        text = Path(log_path).read_text(errors="replace").lower()
    except OSError:
        return False
    return any(s in text for s in _LOST_BIND) or bool(_MOVED_PORT.search(text))


def studio_bin(home) -> str:
    """The `unsloth` CLI of an install / state home. Windows installs put unsloth.exe in bin/
    (a .cmd when the exe is blocked) and in the venv's Scripts/; POSIX ones a bin/unsloth shim."""
    home = Path(home)
    if plat._is_windows():
        cands = (home / "bin" / "unsloth.exe", home / "unsloth_studio" / "Scripts" / "unsloth.exe",
                 home / "bin" / "unsloth.cmd")
    else:
        cands = (home / "bin" / "unsloth", home / ".venv_t5_550" / "bin" / "unsloth",
                 home / ".venv_t5_530" / "bin" / "unsloth", home / "unsloth_studio" / "bin" / "unsloth")
    for c in cands:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"`unsloth` CLI not found under {home}")


def _pgrep(port) -> list:
    """Pids whose command line is `... studio -p PORT` (candidates only: ownership is _ours)."""
    import shutil
    import subprocess
    if not shutil.which("pgrep"):
        return []
    out = subprocess.run(["pgrep", "-f", f"studio -p {port}"], capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if p.isdigit() and int(p) != os.getpid()]


def _owns_port(port: int, home, log_path: Path, pid=None) -> bool:
    """The Studio we started is the one answering: the process listening on `port` is the one we
    spawned (`pid`) or one of its descendants (on Windows unsloth.exe runs the backend as a child),
    or, for a Studio this process did not spawn, runs with OUR home (Linux /proc). pick_free_ports
    is check-then-use and this host runs other Studios, so the health check can pass against a
    stranger's server while ours died with errno 98 or moved to the next port."""
    if _lost_bind(log_path):
        return False
    tree = plat.descendants(pid) if pid else set()
    if pid and not tree:
        return False   # ours already exited: whoever answered is a stranger
    cands = plat.listening_pids(port) or set(_pgrep(port))
    if not cands:      # no lsof / pgrep / Get-NetTCPConnection: our live tree is the best proof left
        return bool(tree)
    return any(p in tree or _ours(p, home) for p in cands)


def _spawn(home: Path, port: int, log_path: Path, env: dict, healthz_timeout_s: int, password_timeout_s: int):
    """`unsloth studio -p PORT` in its own session / process group, output to log_path. Returns
    (StudioInstall, Popen); the Popen is recorded in _LAUNCHED before any wait, so stop() can
    always reach it. Raises on a health timeout or an early exit."""
    import subprocess
    from studio_test_kit.lifecycle import StudioInstall, _read_bootstrap_password, health_body_ok
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    inst = StudioInstall(home=Path(home), repo=Path(home), branch="")
    with open(log_path, "w") as fh:
        proc = subprocess.Popen([studio_bin(home), "studio", "-p", str(port)],
                                env={**os.environ, "UNSLOTH_STUDIO_HOME": str(home), **env},
                                stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                **plat.group_kwargs())
    _LAUNCHED[port] = {"home": str(Path(home).resolve()), "proc": proc}
    inst.port, inst.pid = port, proc.pid
    inst.bootstrap_password = _read_bootstrap_password(Path(home), log_path, time.time() + password_timeout_s)
    urls = [f"http://127.0.0.1:{port}/api/health", f"http://127.0.0.1:{port}/healthz"]
    deadline = time.time() + healthz_timeout_s
    while time.time() < deadline:
        if any(health_body_ok(u) for u in urls):
            return inst, proc
        if proc.poll() is not None and not plat.descendants(proc.pid):
            raise RuntimeError(f"Studio exited {proc.returncode} before answering on :{port} (see {log_path})")
        time.sleep(1)
    raise TimeoutError(f"Studio on :{port} did not answer /api/health within {healthz_timeout_s}s")


def launch(home: Path, port: int, log_path: Path, extra_env: dict | None = None,
           healthz_timeout_s: int = 240, attempts: int = 3):
    """Start Studio on `port`; on a lost bind, retry on a fresh port. The returned install's
    `.port` is the port actually serving (callers must use it, not the one they passed)."""
    from pr_ui_scenes._common import pick_free_ports
    env = {**STUDIO_ENV, **(extra_env or {})}
    for i in range(attempts):
        try:
            inst, proc = _spawn(home, port, log_path, env, healthz_timeout_s, 60)
        except Exception:
            if i == attempts - 1 or not _lost_bind(log_path):
                # A Studio that started but never answered (health / password timeout) would
                # otherwise outlive the run: callers only stop what launch() returned.
                stop(port, home=home)
                raise
        else:
            if _owns_port(port, home, log_path, proc.pid):
                return inst
        stop(port, home=home)
        with open(log_path, "a") as fh:
            fh.write(f"\n[studio_regress] port {port} was taken by another process; retrying\n")
        port = pick_free_ports(1)[0]
    raise RuntimeError(f"could not start Studio on a free port after {attempts} attempts (see {log_path})")


# port -> {"home": state home, "proc": Popen} of the Studio this process launched there.
_LAUNCHED: dict = {}


def _home_matches(val, home) -> bool:
    val = str(Path(val).resolve())
    if home:
        return val == str(Path(home).resolve())
    return val.startswith(str((WS / "temp" / "studio_regress").resolve()))


def _ours(pid: int, home: str | None) -> bool:
    """Only this suite's Studios: `pid` belongs to the process tree of a Studio this process
    spawned with that home, or (Linux, where another process's environment is readable) runs with
    UNSLOTH_STUDIO_HOME set to the home we launched (when unknown, a state home under this
    workspace's temp/studio_regress). The same unix user runs other workspaces' Studios on this
    host, possibly on the same port; never a command-line match."""
    for rec in list(_LAUNCHED.values()):
        if (home is None or _home_matches(rec["home"], home)) and int(pid) in plat.descendants(rec["proc"].pid):
            return True
    env = plat.process_env(pid)
    val = (env or {}).get("UNSLOTH_STUDIO_HOME")
    return bool(val) and _home_matches(val, home)


def stop(port: int, home=None):
    """Stop this suite's Studio on `port`: the tree we spawned (own session / process group, so
    llama-server children go too) plus, on Linux, any `studio -p PORT` running with our home."""
    rec = _LAUNCHED.pop(port, None)
    home = home or (rec or {}).get("home")
    proc = (rec or {}).get("proc")
    extra = [p for p in _pgrep(port) if _ours(p, home)]
    if proc is None and not extra:
        return
    plat.stop_tree(proc.pid if proc is not None else None, grace_s=20, extra=extra, proc=proc)


def _post(url, payload, token=None, timeout=60):
    import httpx
    h = {"Authorization": f"Bearer {token}"} if token else {}
    r = httpx.post(url, json=payload, headers=h, timeout=timeout)
    return r


def api_login(base_url, password, username="unsloth"):
    r = _post(f"{base_url}/api/auth/login", {"username": username, "password": password})
    if r.status_code != 200:
        raise RuntimeError(f"login failed {r.status_code}: {r.text[:200]}")
    return r.json()


def rotate_bootstrap(base_url, home, new_password):
    """Bootstrap login + change-password; idempotent when already rotated to new_password."""
    boot = Path(home) / "auth" / ".bootstrap_password"
    if not boot.exists():
        return api_login(base_url, new_password)
    old = boot.read_text().strip()
    tok = api_login(base_url, old)
    r = _post(f"{base_url}/api/auth/change-password",
              {"current_password": old, "new_password": new_password}, token=tok["access_token"])
    if r.status_code != 200:
        raise RuntimeError(f"change-password failed {r.status_code}: {r.text[:200]}")
    return r.json()


_WARMED: set = set()


def warm(base_url, token):
    """First /api/system (and /api/system/hardware) on a fresh Studio probes every GPU (~10 s on an 8-GPU host); later calls
    are cached. Pages that render before it answers show "Checking for GPUs..." / "No GPU
    detected", the ones after show the devices, so which one a shot catches is timing. Pay it
    once per Studio before any page opens."""
    key = (base_url, token)   # both sides of a run share a port; the token is per Studio
    if key in _WARMED:
        return
    import httpx
    # /api/system/hardware (Export's GPU check, the About tab's hardware list) probes cold too.
    for path in ("/api/system", "/api/system/hardware?include_details=true"):
        try:
            httpx.get(f"{base_url}{path}", headers={"Authorization": f"Bearer {token}"}, timeout=180)
        except Exception:
            pass
    _WARMED.add(key)


async def authed_context(browser, base_url, tokens: dict, **kw):
    import asyncio
    await asyncio.to_thread(warm, base_url, tokens["access_token"])
    ctx = await fixture.new_context(browser, **kw)
    seed = {"unsloth_auth_token": tokens["access_token"],
            "unsloth_auth_refresh_token": tokens.get("refresh_token", "")}
    await ctx.add_init_script(
        "(() => { try { const s = " + json.dumps(seed) +
        "; for (const k in s) if (!localStorage.getItem(k)) localStorage.setItem(k, s[k]); } catch (e) {} })();")
    return ctx


class ApiClient:
    """Tiny async client bound to one Studio + token (ctx.api)."""

    def __init__(self, base_url, token):
        import httpx
        self.base_url = base_url
        self.c = httpx.AsyncClient(base_url=base_url, timeout=120,
                                   headers={"Authorization": f"Bearer {token}"})

    async def get(self, path, **kw):
        r = await self.c.get(path, **kw)
        r.raise_for_status()
        return r.json()

    async def post(self, path, json=None, **kw):
        r = await self.c.post(path, json=json, **kw)
        r.raise_for_status()
        return r.json() if r.content else {}

    async def raw(self, method, path, **kw):
        return await self.c.request(method, path, **kw)

    # The model / GPU journeys (journeys/_b_common.py, parallel.py) speak httpx directly:
    # request() returns the response without raising, stream() is httpx's streaming context.
    async def request(self, method, path, **kw):
        return await self.c.request(method, path, **kw)

    def stream(self, method, path, **kw):
        return self.c.stream(method, path, **kw)

    async def aclose(self):
        await self.c.aclose()


async def run_journey(journey, ctx: Ctx, out_dir: Path) -> dict:
    """Run the frozen step list; one facts file per step whatever happens.

    After the first failed / unreachable step the remaining steps are recorded as
    `skipped_after_failure` (both sides see the same outcome shape, and the diff reports
    FAIL_HEAD / DIVERGED on the step that actually broke)."""
    jdir = Path(out_dir) / journey.name
    jdir.mkdir(parents=True, exist_ok=True)
    timings, broken = {}, None
    for step in journey.steps:
        facts = {"_step": step.id}
        t0 = time.perf_counter()
        if broken and not journey.independent:
            facts.update(_status="skipped_after_failure", _after=broken)
            (jdir / f"{step.id}.facts.json").write_text(json.dumps(facts, indent=1, default=str))
            continue
        try:
            got = await asyncio.wait_for(step.action(ctx), timeout=step.timeout_s)
            facts.update(got or {})
            facts["_status"] = "ok"
        except StepUnreachable as e:
            facts.update(_status="unreachable", _error=str(e)[:500])
            broken = step.id
        except (StepFailed, asyncio.TimeoutError, Exception) as e:  # noqa: B014
            facts.update(_status="failed", _error=f"{type(e).__name__}: {e}"[:800],
                         _trace=traceback.format_exc()[-1500:])
            broken = step.id
        if ctx.page is not None:
            try:
                rects, dom = await fixture.capture(ctx.page, jdir, step.id, masks=step.masks,
                                                   full_page=step.full_page, shot=step.shot)
                facts["_masks"] = rects
                facts["_clicked"] = await fixture.drain_clicks(ctx.page)
                if dom.get("_mask_overrun"):
                    facts["_mask_overrun"] = dom["_mask_overrun"]
            except Exception as e:
                facts.setdefault("_capture_error", str(e)[:300])
        facts["_s"] = round(time.perf_counter() - t0, 2)
        timings[step.id] = facts["_s"]
        (jdir / f"{step.id}.facts.json").write_text(json.dumps(facts, indent=1, default=str))
    if journey.teardown is not None:
        try:
            await journey.teardown(ctx)
        except Exception as e:  # teardown must never mask the step results
            print(f"[studio_regress] {journey.name} teardown: {e}", file=sys.stderr)
    return {"journey": journey.name, "broken": broken, "timings": timings}
