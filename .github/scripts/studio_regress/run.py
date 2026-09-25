"""Run selected journeys on the PR merge base (before) and head (after), then diff.

    python studio_regress.py run --pr 11606                      # switchboard selection
    python studio_regress.py run --pr 11606 --only auth,crawl --tier fast
    python studio_regress.py run --home-before H --home-after H --only auth,settings,crawl
                                                                 # A/A null control, no PR
    python studio_regress.py run --base-url http://127.0.0.1:8888 --password P --only settings
                                                                 # dev: one side, running Studio

Installs: --home-before/--home-after reuse existing install homes; with --pr and no homes,
each side's SHA is installed once into temp/studio_regress/installs/<sha> (flock-guarded,
`.uidiff_sha` stamp; reused by every later run / tmux session). Each side then gets a FRESH
state home (engine.state_home) so auth / db / settings start identical on both sides.

Order per side: auth (if selected; it needs the bootstrap state) -> other journeys in
switchboard dependency order -> crawl last. One fresh browser context per journey.

Output: <root>/{before,after}/<journey>/<step>.{png,dom.json,facts.json}, manifest.json,
<side>/crawl_manifest/<route>.json, report.json (contract schema), summary.md.

Core targets (kind job / regression, core.py) need no Studio install: they compare the PR merge
base with the head in the Core interpreter (--core-python) and add "<target>/run" steps. When
only Core targets are selected, no Studio is installed. They need --pr and --side both.

    python studio_regress.py run --pr 1373 --gh-repo unslothai/unsloth-zoo   # Core only
    python studio_regress.py run --pr 5123 --only jobs/sft.py,regression_smoke

Exit: 0 no regression and no UI change, 3 UI changed (VISUAL/DOM/DIVERGED) without a
functional regression, 1 functional regression (FAIL_HEAD), 2 VOID (install/launch failed,
selected journeys produced no steps, or a Core target proved nothing).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))

from studio_regress import core, coverage, diff, engine, gc, plat, selection  # noqa: E402
from studio_regress.contract import SUITE_VERSION, Ctx  # noqa: E402

WS = Path(os.environ.get("WORKSPACE") or SCRIPTS.parent.parent.parent)
INSTALLS = WS / "temp" / "studio_regress" / "installs"
EXIT = {"clean": 0, "regression": 1, "void": 2, "ui_changed": 3}


def _log(msg):
    print(f"[studio_regress] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ journey registry
def load_journeys():
    """name -> (Journey, module) for every journeys/*.py exposing JOURNEY, plus crawl."""
    out = {}
    for f in sorted((HERE / "journeys").glob("*.py")):
        if f.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"studio_regress.journeys.{f.stem}")
        except Exception as e:  # a broken journey must not take the whole suite down
            _log(f"journey module {f.stem} failed to import: {e}")
            continue
        j = getattr(mod, "JOURNEY", None)
        if j is not None:
            out[j.name] = (j, mod)
    cj = coverage.crawl_journey()
    out[cj.name] = (cj, coverage)
    return out


def order(names, journeys, data):
    """auth first, deps before dependents, crawl last; unknown names dropped (logged)."""
    deps = {t["name"]: t.get("deps") or [] for t in data["target"]}
    names = [n for n in names if n in journeys]
    out, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for d in deps.get(n, []):
            if d in journeys:
                visit(d)
        out.append(n)
    for n in names:
        visit(n)
    out.sort(key=lambda n: (0 if n == "auth" else 2 if n == "crawl" else 1))
    return out


# ------------------------------------------------------------------ installs
@contextlib.contextmanager
def _install_lock(sha):
    INSTALLS.mkdir(parents=True, exist_ok=True)
    with plat.locked(INSTALLS / f"{sha}.lock"):
        yield


def ensure_install(repo: Path, sha: str, log_dir: Path) -> Path:
    """Install `sha` once into INSTALLS/<sha> (shared by every session)."""
    from pr_ui_diff import Side, install_side, make_worktree
    home = INSTALLS / sha
    with _install_lock(sha):
        stamp = home / ".uidiff_sha"
        if stamp.exists() and stamp.read_text().strip() == sha:
            gc.touch(home)   # `studio_regress.py gc` reclaims installs unused for days
            return home
        wt = make_worktree(repo, sha, WS / "temp" / "studio_regress" / "src" / sha)
        if stamp.exists():
            stamp.unlink()
        install_side(Side(label=sha[:9], sha=sha, worktree=wt, home=home), log_dir)
        stamp.write_text(sha)
        gc.touch(home)
    return home


def source_dir(install_home):
    """The unsloth source checkout an install home was built from (for upstream tests)."""
    stamp = Path(install_home) / ".uidiff_sha"
    if stamp.exists():
        d = WS / "temp" / "studio_regress" / "src" / stamp.read_text().strip()
        if d.is_dir():
            return d
    return None


# ------------------------------------------------------------------ one side
def _models(data, keys, extra=None):
    """ctx.models for one journey: switchboard [models] specs by key ({"repo", "variant"} for
    GGUF / training fixtures), a local snapshot dir for `local_snapshot = true` fixtures (tiny
    diffusers pipelines load from a path), plus run-provided `_`-keys (`_studio_home`,
    `_studio_log`)."""
    out = dict(extra or {})
    for k in keys:
        spec = (data.get("models") or {}).get(k)
        if spec is None:
            continue
        if spec.get("local_snapshot"):
            from studio_regress.journeys import _diffusion
            try:
                out[k] = _diffusion.ensure_tiny(spec["repo"])
            except Exception as e:   # the journey falls back to its own ensure_* and says why
                _log(f"fixture {k}: local snapshot of {spec['repo']} failed: {e}")
        else:
            out[k] = dict(spec)
    return out


def isolated_names(names, journeys):
    """Journeys whose module sets ISOLATED = True run in a nested run.py inside
    studio_regress.isolation (Studio, tool children, Chromium and driver in one network
    namespace), never in the side's shared Studio. Inside that nested run there is nothing left
    to isolate."""
    if os.environ.get("STUDIO_REGRESS_ISOLATION"):
        return []
    return [n for n in names if getattr(journeys[n][1], "ISOLATED", False)]


def run_isolated(side, home, root, names, online=False):
    """One side of the ISOLATED journeys: `run.py --only N --side S` inside the isolation
    wrapper, writing into the same <root>/<side>/ layout. Returns the wrapper's exit code."""
    tr = WS / "temp" / "studio_regress"
    browsers = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or WS / "temp" / "pw_browsers")
    cmd = [sys.executable, str(HERE / "run.py"), "--only", ",".join(names), "--side", side,
           f"--home-{side}", str(home), "--root", str(root), "--no-prefetch"] + (["--online"] if online else [])
    rw = [root, tr / "state", tr / "home", tr / "hf_home"]
    ro = [Path(home), SCRIPTS, tr / "src", Path(sys.prefix), browsers]
    for d in rw:
        d.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "studio_regress.isolation",
            "--mode", os.environ.get("STUDIO_REGRESS_ISOLATION_MODE", "auto")]
    argv += [f"--rw={d}" for d in rw] + [f"--ro={d}" for d in ro if d.exists()]
    env = {**os.environ, "PLAYWRIGHT_BROWSERS_PATH": str(browsers)}
    with open(root / "logs" / f"isolated_{side}.log", "a") as log:
        return subprocess.call(argv + ["--"] + cmd, cwd=SCRIPTS, env=env, stdout=log, stderr=subprocess.STDOUT)


async def run_side(side, install_home, root, names, journeys, data, password, base_url=None,
                   extra_env=None):
    """Launch a fresh-state Studio for `side` (unless base_url), run journeys, stop."""
    from playwright.async_api import async_playwright
    from pr_ui_scenes._common import pick_free_ports

    out = root / side
    out.mkdir(parents=True, exist_ok=True)
    timings, port, home = {}, None, None
    env = dict(extra_env or {})
    studio_log = root / "logs" / (f"studio_{side}_isolated.log" if os.environ.get("STUDIO_REGRESS_ISOLATION")
                                  else f"studio_{side}.log")
    for n in names:
        env.update(getattr(journeys[n][1], "STUDIO_ENV", {}) or {})
    if base_url is None:
        home = engine.state_home(install_home, WS / "temp" / "studio_regress" / "state" /
                                 root.name)  # same path both sides: path text shows in the UI
        port = pick_free_ports(1, seed=root.name)[0]   # same port both sides: it shows in the UI
        t0 = time.time()
        inst = engine.launch(home, port, studio_log, env)
        port = inst.port   # launch moves to a fresh port if this one was taken
        base_url = f"http://127.0.0.1:{port}"
        timings["_launch"] = round(time.time() - t0, 1)
        bootstrap = inst.bootstrap_password
    else:
        bootstrap = None
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            state_common = {"home": str(home) if home else None, "install_home": str(install_home or ""),
                            "src": str(source_dir(install_home)) if install_home else None,
                            "password": password, "bootstrap_password": bootstrap, "side": side,
                            "root": str(root)}
            tokens = None
            run_models = {"_studio_home": str(home) if home else None,
                          "_studio_log": str(studio_log) if home else None}
            for n in names:
                j, _mod = journeys[n]
                t0 = time.time()
                if n == "auth":
                    if bootstrap is None:
                        _log(f"{side}: auth skipped (no bootstrap state on a reused Studio)")
                        continue
                    bctx = await engine.fixture.new_context(browser)
                    page = await engine.fixture.prepare_page(await bctx.new_page())
                    ctx = Ctx(page=page, base_url=base_url, api=None, side=side, out_dir=str(out),
                              models=_models(data, []), state=dict(state_common))
                    await engine.run_journey(j, ctx, out)
                    await bctx.close()
                elif n == "crawl":
                    if tokens is None:
                        tokens = (engine.rotate_bootstrap(base_url, home, password) if home
                                  else engine.api_login(base_url, password))
                    await _run_crawl(browser, base_url, tokens, j, side, out, state_common)
                else:
                    if tokens is None:
                        tokens = (engine.rotate_bootstrap(base_url, home, password) if home
                                  else engine.api_login(base_url, password))
                    tgt = next((t for t in data["target"] if t["name"] == n), {})
                    bctx = await engine.authed_context(browser, base_url, tokens)
                    page = await engine.fixture.prepare_page(await bctx.new_page())
                    api = engine.ApiClient(base_url, tokens["access_token"])
                    ctx = Ctx(page=page, base_url=base_url, api=api, side=side, out_dir=str(out),
                              models=_models(data, tgt.get("models") or [], run_models),
                              state=dict(state_common))
                    await engine.run_journey(j, ctx, out)
                    await api.aclose()
                    await bctx.close()
                timings[n] = round(time.time() - t0, 1)
                _log(f"{side}: {n} {timings[n]}s")
            await browser.close()
    finally:
        if port:
            engine.stop(port)
    return timings


CRAWL_WORKERS = 3


async def _run_crawl(browser, base_url, tokens, j, side, out, state_common):
    """Crawl routes in CRAWL_WORKERS concurrent browser contexts (partitioned by step index,
    so the partition is identical on both sides); each writes its own step files."""
    from studio_regress.contract import Journey

    async def part(steps):
        bctx = await engine.authed_context(browser, base_url, tokens)
        page = await engine.fixture.prepare_page(await bctx.new_page())
        ctx = Ctx(page=page, base_url=base_url, api=None, side=side, out_dir=str(out),
                  state=dict(state_common))
        await engine.run_journey(Journey(name=j.name, tier=j.tier, steps=steps), ctx, out)
        await bctx.close()
    parts = [tuple(j.steps[i::CRAWL_WORKERS]) for i in range(CRAWL_WORKERS)]
    await asyncio.gather(*(part(p) for p in parts if p))


# ------------------------------------------------------------------ report
def write_report(root, meta, steps, cov, timings):
    func = diff.functional_verdict(steps)
    counts = diff.summarize(steps)
    changed = [s for s in steps if s["verdict"] in ("VISUAL_DIFF", "DOM_ONLY_DIFF", "DIVERGED")]
    report = {**meta, "suite_version": SUITE_VERSION, "coverage": cov, "steps": steps,
              "functional": func, "counts": counts, "timings": timings}
    (root / "report.json").write_text(json.dumps(report, indent=1, default=str))
    lines = [f"# studio_regress {meta.get('pr') or 'A/A'}", "",
             f"functional: **{func}**  counts: {json.dumps(counts)}"]
    if cov:   # Core-only runs drive no browser
        lines.append(f"coverage: {cov.get('overall')}% overall ({cov.get('exercised')}/{cov.get('inventory')}), "
                     f"safe {cov.get('overall_safe')}%")
    lines.append("")
    listed = changed + [s for s in steps if s["verdict"] in ("FAIL_HEAD", "FAIL_BOTH", "FLAKY", "VOID")]
    listed += [s for s in steps if s.get("mask_overrun") and s not in listed]
    for s in listed:
        lines.append(f"- {s['verdict']} `{s['key']}` px={s['pixels_changed']} {s.get('note', '')}")
    lines += ["", "timings (s): " + json.dumps(timings)]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    if func == "REGRESSION":
        return report, EXIT["regression"]
    if not steps or func == "VOID" or any(s["verdict"] == "VOID" for s in steps):
        return report, EXIT["void"]
    return report, EXIT["ui_changed"] if changed else EXIT["clean"]


def main(argv=None):
    p = argparse.ArgumentParser(description="Studio before/after regression run")
    p.add_argument("--pr", type=int)
    p.add_argument("--gh-repo", default="unslothai/unsloth")
    p.add_argument("--repo", default=str(WS / "unsloth"), help="local unsloth clone")
    p.add_argument("--root", help="output dir (default outputs/studio_regress/pr<N> or aa_<ts>)")
    p.add_argument("--home-before")
    p.add_argument("--home-after")
    p.add_argument("--base-url", help="dev: drive an already-running Studio (single side)")
    p.add_argument("--password", help="with --base-url: its password")
    p.add_argument("--side", choices=("before", "after", "both"), default="both")
    p.add_argument("--only", help="comma-separated targets / globs")
    p.add_argument("--exclude", help="comma-separated globs")
    p.add_argument("--tier", action="append", choices=("fast", "model", "gpu"))
    p.add_argument("--all", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--no-prefetch", action="store_true", help="skip fixture model prefetch")
    p.add_argument("--online", action="store_true", help="let Studio reach the network (default offline)")
    p.add_argument("--record-flaky", action="store_true", help="A/A: record non-SAME keys as FLAKY")
    p.add_argument("--core-python", default=None,
                   help="interpreter for job / regression targets (default $STUDIO_REGRESS_CORE_PYTHON, "
                        "else temp/venv_core, else $VIRTUAL_ENV); needs torch + unsloth deps")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    data = selection.load()
    journeys = load_journeys()
    if a.list:
        for n, (j, _m) in journeys.items():
            print(f"{n:24} {j.tier:6} {len(j.steps):3} steps")
        for t in data["target"]:
            if t["kind"] == "external":
                print(f"{t['name']:24} {t.get('tier', '?'):6} external: {' '.join(t['cmd'])}")
            elif t["kind"] in selection.CORE_KINDS:
                print(f"{t['name']:24} {t.get('tier', '?'):6} {t['kind']}: {', '.join(selection.target_repos(t))}")
        missing = [t["name"] for t in data["target"] if t["kind"] == "journey" and t["name"] not in journeys]
        if missing:
            print("registered but not implemented:", ", ".join(missing))
        return 0

    meta = {"pr": a.pr, "repo": a.gh_repo}
    if a.pr and not a.only and not a.all:
        files, labels = selection.pr_files_and_labels(a.pr, a.gh_repo)
        sel = selection.select(files, labels, data, tiers=a.tier, repo=a.gh_repo)
        meta["selection"] = sel
    else:
        sel = {"selected": []}
    names = selection.apply_overrides(sel, only=a.only.split(",") if a.only else None,
                                      exclude=a.exclude.split(",") if a.exclude else None,
                                      all_targets=a.all, data=data)
    local_skip = {t["name"] for t in data["target"] if t.get("staging_only")}
    if local_skip & set(names):
        _log(f"staging only, skipped locally: {sorted(local_skip & set(names))}")
    by_name = {t["name"]: t for t in data["target"]}
    for_repo = {n for n in names if n in by_name and a.gh_repo in selection.target_repos(by_name[n])}
    if set(names) - for_repo:
        _log(f"not for {a.gh_repo}, skipped: {sorted(set(names) - for_repo)}")
    externals = [by_name[n] for n in names if by_name.get(n, {}).get("kind") == "external"
                 and n in for_repo and (not a.tier or by_name[n].get("tier") in a.tier)]
    cores = [by_name[n] for n in names if by_name.get(n, {}).get("kind") in selection.CORE_KINDS
             and n in for_repo and (not a.tier or by_name[n].get("tier") in a.tier)]
    if cores and not (a.pr and a.side == "both" and not a.base_url):
        _log(f"Core targets need --pr and both sides, skipped: {[t['name'] for t in cores]}")
        cores = []
    names = [n for n in names if n in journeys and n not in local_skip and n in for_repo]
    if a.tier:
        names = [n for n in names if journeys[n][0].tier in a.tier]
    names = order(names, journeys, data)
    if not names and not externals and not cores:
        _log("nothing selected")
        return EXIT["void"]
    # absolute: external targets and the nested isolated run.py use another cwd
    root = Path(a.root or (WS / "outputs" / "studio_regress" /
                           (f"pr{a.pr}" if a.pr else f"aa_{time.strftime('%Y%m%d_%H%M%S')}"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    _log(f"journeys: {names} -> {root}")
    if cores:
        _log(f"core: {[t['name'] for t in cores]}")
    if not names and not externals:   # Core only: no Studio install, no browser
        steps, ctimes = core.run_all(cores, a.pr, a.gh_repo, root, python=a.core_python, log=_log)
        for st in steps:   # the SHAs the Core targets actually compared
            meta.setdefault("base_sha", (st.get("before") or {}).get("sha"))
            meta.setdefault("head_sha", (st.get("after") or {}).get("sha"))
        report, code = write_report(root, meta, steps, {}, {"core": ctimes})
        print((root / "summary.md").read_text())
        if a.json:
            print(json.dumps({"exit": code, "root": str(root), "counts": report["counts"]}, indent=1))
        return code

    # installs
    homes = {"before": a.home_before, "after": a.home_after}
    if a.base_url:
        sides = [a.side if a.side != "both" else "after"]
    else:
        sides = ["before", "after"] if a.side == "both" else [a.side]
        # Resolve and install only the sides this run drives: a single-side run with its own
        # --home-<side> (the staging legs) needs neither gh nor the other side's install.
        if a.pr and any(not homes[s] for s in sides):
            from pr_ui_diff import resolve_shas
            mb, head_sha, head_ref = resolve_shas(Path(a.repo), a.pr, a.gh_repo)
            meta.update(base_sha=mb, head_sha=head_sha, merge_base=mb, head_ref=head_ref)
            try:
                for s, sha in (("before", mb), ("after", head_sha)):
                    if s in sides and not homes[s]:
                        homes[s] = str(ensure_install(Path(a.repo), sha, root / "logs"))
            except Exception as e:
                _log(f"install failed: {e}")
                return EXIT["void"]
        for s in sides:
            if not homes[s]:
                p.error(f"--home-{s} (or --pr) required")
    meta["homes"] = homes
    keys = [k for n in names for k in journeys[n][0].keys()] + [f"{t['name']}/run" for t in externals]
    mpath = root / "manifest.json"
    if mpath.exists() and os.environ.get("STUDIO_REGRESS_ISOLATION"):   # nested isolated run: merge
        old = json.loads(mpath.read_text())
        names_m = old.get("journeys", []) + [n for n in names if n not in old.get("journeys", [])]
        keys = old.get("keys", []) + [k for k in keys if k not in old.get("keys", [])]
    else:
        names_m = names
    mpath.write_text(json.dumps({"journeys": names_m, "keys": keys}, indent=1))

    if not a.no_prefetch:
        from studio_regress import prefetch
        for k, spec in prefetch.fixtures_for(names + [t["name"] for t in externals], data).items():
            try:
                prefetch.fetch(spec)
            except Exception as e:
                _log(f"prefetch {k} failed: {e}")
    password = a.password or ("Regress-" + secrets.token_urlsafe(10).replace("-", "x"))
    timings = {}
    iso = isolated_names(names, journeys) if not a.base_url else []
    shared = [n for n in names if n not in iso]
    for s in sides if names else ():
        try:
            if shared:
                timings[s] = asyncio.run(run_side(s, homes.get(s), root, shared, journeys, data,
                                                  password if a.base_url else password + s[:1],
                                                  base_url=a.base_url,
                                                  extra_env={"HF_HUB_OFFLINE": "0"} if a.online else None))
        except Exception as e:
            _log(f"{s}: side failed: {type(e).__name__}: {e}")
            return EXIT["void"]
        if iso:
            t0 = time.time()
            rc = run_isolated(s, homes[s], root, iso, online=a.online)
            timings.setdefault(s, {})["_isolated"] = round(time.time() - t0, 1)
            _log(f"{s}: isolated {iso} exit {rc} ({root / 'logs' / f'isolated_{s}.log'})")
    if len(sides) < 2:
        if externals:
            _log(f"external targets need both sides, skipped: {[t['name'] for t in externals]}")
        _log(f"single side done: {root / sides[0]}")
        return 0
    ext_keys = {f"{t['name']}/run" for t in externals}
    steps = [x for x in diff.diff_all(root, flaky=set() if a.record_flaky else None) if x["key"] not in ext_keys]
    for t in externals:   # head first, base only when head fails (external.py)
        from studio_regress import external
        t0 = time.time()
        steps.append(external.run_target(t, homes, root, env=engine.STUDIO_ENV))
        timings.setdefault("external", {})[t["name"]] = round(time.time() - t0, 1)
        _log(f"external {t['name']}: {steps[-1]['verdict']} {timings['external'][t['name']]}s")
    if cores:
        csteps, ctimes = core.run_all(cores, a.pr, a.gh_repo, root, python=a.core_python, log=_log)
        steps += csteps
        timings["core"] = ctimes
    if a.record_flaky:
        bad = [x for x in steps if x["verdict"] in diff.FLAKY_SOURCES]
        diff.record_flaky((), steps=bad)
        _log(f"recorded {len(bad)} flaky keys")
    cov = coverage.compute(root / "after")
    report, code = write_report(root, meta, steps, cov, timings)
    print((root / "summary.md").read_text())
    if a.json:
        print(json.dumps({"exit": code, "root": str(root), "counts": report["counts"],
                          "coverage": cov}, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
