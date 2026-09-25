"""Reclaim the per-SHA caches studio_regress leaves in temp/: Studio installs (multi-GB each),
their source worktrees, and the Core A/B trees.

    python studio_regress.py gc [--days 3] [--dry-run]

An entry goes once it has not been used for --days (the `.last_used` marker ensure_install and
core.tree touch on every reuse; the directory mtime when there is none) AND nothing holds it:
its creation lock is free and no live process has the path in its cwd, executable, command line
or environment (a running Studio's python lives inside its install; a regression run carries its
trees on PYTHONPATH). Git worktrees are removed through `git worktree remove` so the parent repo
keeps no dangling entry. outputs/ (evidence) is never touched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

WS = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])
ROOT = WS / "temp" / "studio_regress"
MARKER = ".last_used"


def touch(d):
    """Record a use of a cached install / tree (ignored when the dir is gone)."""
    try:
        (Path(d) / MARKER).touch()
    except OSError:
        pass


def last_used(d):
    m = Path(d) / MARKER
    try:
        return m.stat().st_mtime if m.exists() else Path(d).stat().st_mtime
    except OSError:
        return 0.0


def _proc_refs():
    """Every cwd / exe / cmdline / environ string of this user's live processes. Without /proc
    (macOS, Windows) only command lines are visible: `ps` / Win32_Process. None when nothing can
    be listed, which collect() treats as "everything in use"."""
    from studio_regress import plat
    if plat._is_windows() or not (plat.PROC / "self").exists():
        return _cmdline_refs()
    refs = []
    for p in plat.PROC.iterdir():
        if not p.name.isdigit() or p.name == str(os.getpid()):
            continue
        for link in ("cwd", "exe"):
            try:
                refs.append(os.readlink(p / link))
            except OSError:
                pass
        for f in ("cmdline", "environ"):
            try:
                refs.append((p / f).read_bytes().replace(b"\0", b" ").decode(errors="replace"))
            except OSError:
                pass
    return refs


def _cmdline_refs():
    from studio_regress import plat
    if plat._is_windows():
        ps = shutil.which("powershell") or shutil.which("pwsh")
        argv = [ps, "-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance Win32_Process | ForEach-Object { $_.ExecutablePath; $_.CommandLine }"] if ps else None
    else:
        argv = ["ps", "-A", "-ww", "-o", "command="]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=60) if argv else None
    except (OSError, subprocess.SubprocessError):
        r = None
    if r is None or r.returncode != 0:
        return None
    return [line for line in r.stdout.splitlines() if line.strip()]


def in_use(path, refs):
    if refs is None:   # could not list processes: never reclaim blind
        return True
    s = str(Path(path).resolve())
    return any(s in r for r in refs)


def _lock_free(lock):
    from studio_regress import plat
    if not lock.exists():
        return True
    try:
        with open(lock, "a+") as fh:
            if not plat.lock(fh, blocking=False):
                return False
            plat.unlock(fh)
            return True
    except OSError:
        return False


def _remove(d):
    """git worktree remove when `d` is a linked worktree, then make sure the dir is gone."""
    if (d / ".git").is_file():
        common = subprocess.run(["git", "-C", str(d), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True).stdout.strip()
        if common:
            main = Path(common).parent
            subprocess.run(["git", "-C", str(main), "worktree", "remove", "--force", str(d)],
                           capture_output=True, text=True)
            shutil.rmtree(d, ignore_errors=True)
            subprocess.run(["git", "-C", str(main), "worktree", "prune"], capture_output=True, text=True)
            return
    shutil.rmtree(d, ignore_errors=True)


def candidates(root=ROOT):
    """(dir, its creation lock, linked source worktree or None) for every cached entry."""
    out = []
    inst = root / "installs"
    for d in sorted(inst.iterdir()) if inst.is_dir() else ():
        if d.is_dir():
            out.append((d, inst / f"{d.name}.lock", root / "src" / d.name))
    trees = root / "trees"
    for d in sorted(trees.iterdir()) if trees.is_dir() else ():
        if d.is_dir():
            out.append((d, trees / f"{d.name}.lock", None))
    return out


def collect(days=3.0, dry_run=False, root=ROOT, refs=None, now=None, log=print):
    now = time.time() if now is None else now
    refs = _proc_refs() if refs is None else refs
    removed, kept = [], []
    for d, lock, src in candidates(root):
        age_d = (now - last_used(d)) / 86400
        why = ("recent" if age_d < days else
               "in use" if in_use(d, refs) or (src is not None and src.exists() and in_use(src, refs)) else
               "locked" if not _lock_free(lock) else None)
        if why:
            kept.append((d, why))
            continue
        removed.append(d)
        log(f"{'would remove' if dry_run else 'remove'} {d} (unused {age_d:.1f} d)")
        if not dry_run:
            _remove(d)
            if src is not None and src.exists():
                _remove(src)
            lock.unlink(missing_ok=True)
    return removed, kept


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Reclaim unused studio_regress installs and trees")
    p.add_argument("--days", type=float, default=3.0, help="keep anything used within this many days")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    removed, kept = collect(a.days, a.dry_run)
    busy = [f"{d.name} ({w})" for d, w in kept if w != "recent"]
    print(f"{'would remove' if a.dry_run else 'removed'} {len(removed)}, kept {len(kept)}"
          + (f"; held: {', '.join(busy)}" if busy else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
