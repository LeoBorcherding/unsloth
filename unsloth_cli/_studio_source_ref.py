# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run Unsloth Studio from a branch, tag or commit of unslothai/unsloth instead of the release.

`unsloth studio update --ref <ref>` downloads that commit's source and installs it the way
`--local` installs a checkout. The pin file records what is installed, so the desktop's launch
preflight leaves it alone; the request file is how the backend asks the desktop to switch.
Stdlib only: the backend imports this too.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

REPO = "unslothai/unsloth"
PIN_FILE = "source-ref.json"
REQUEST_FILE = "source-ref-request.json"
SOURCES_DIR = "sources"
ALLOW_API_ENV = "UNSLOTH_ALLOW_SOURCE_SWITCH"
# git's own ref rules, narrowed: no leading dash (option injection), no `..`, no `@{`, no trailing `.lock`.
_REF = re.compile(r"^(?!-)(?!.*\.\.)(?!.*@\{)(?!.*\.lock$)[A-Za-z0-9._/-]{1,200}$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_TIMEOUT = 60


class SourceRefError(RuntimeError):
    pass


def valid_ref(ref: str) -> bool:
    return bool(_REF.match(ref)) and not ref.endswith("/")


def resolve_ref(ref: str) -> str:
    """The full commit SHA a branch, tag or SHA names on unslothai/unsloth."""
    if not valid_ref(ref):
        raise SourceRefError(f"not a valid branch, tag or commit: {ref!r}")
    url = f"https://api.github.com/repos/{REPO}/commits/{urllib.parse.quote(ref, safe = '/')}"
    req = urllib.request.Request(url, headers = {"Accept": "application/vnd.github.sha"})
    try:
        with urllib.request.urlopen(req, timeout = _TIMEOUT) as resp:
            sha = resp.read(64).decode("ascii", "replace").strip()
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 422):
            raise SourceRefError(f"{ref!r} is not a branch, tag or commit on {REPO}") from exc
        raise SourceRefError(f"GitHub answered {exc.code} looking up {ref!r}") from exc
    except OSError as exc:
        raise SourceRefError(f"could not reach GitHub to look up {ref!r}: {exc}") from exc
    if not _SHA.match(sha):
        raise SourceRefError(f"GitHub returned no commit for {ref!r}")
    return sha


def fetch_source(sha: str, studio_home: Path) -> Path:
    """Download and unpack one commit under <studio_home>/sources. Reuses an earlier download."""
    if not _SHA.match(sha):
        raise SourceRefError(f"not a full commit SHA: {sha!r}")
    root = studio_home / SOURCES_DIR
    dest = root / f"unsloth-{sha[:12]}"
    if (dest / "pyproject.toml").is_file():
        return dest
    root.mkdir(parents = True, exist_ok = True)
    url = f"https://codeload.github.com/{REPO}/tar.gz/{sha}"
    with tempfile.TemporaryDirectory(dir = root, prefix = ".download-") as tmp:
        archive = Path(tmp) / "src.tar.gz"
        try:
            with urllib.request.urlopen(url, timeout = _TIMEOUT) as resp, archive.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
        except OSError as exc:
            raise SourceRefError(f"could not download {sha[:12]}: {exc}") from exc
        out = Path(tmp) / "out"
        with tarfile.open(archive) as tar:
            _safe_extract(tar, out)
        tops = [p for p in out.iterdir() if p.is_dir()]
        if len(tops) != 1 or not (tops[0] / "pyproject.toml").is_file():
            raise SourceRefError(f"the {sha[:12]} archive is not an Unsloth source tree")
        shutil.rmtree(dest, ignore_errors = True)
        os.replace(tops[0], dest)
    return dest


def _safe_extract(tar: tarfile.TarFile, out: Path) -> None:
    if hasattr(tarfile, "data_filter"):
        tar.extractall(out, filter = "data")
        return
    base = out.resolve()
    for member in tar.getmembers():
        target = (out / member.name).resolve()
        if base not in target.parents and target != base:
            raise SourceRefError(f"archive member escapes the target: {member.name}")
        if member.issym() or member.islnk():
            raise SourceRefError(f"archive holds a link: {member.name}")
    tar.extractall(out)


def prune_sources(studio_home: Path, keep: Optional[Path]) -> None:
    root = studio_home / SOURCES_DIR
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.name.startswith("unsloth-") and child != keep:
            shutil.rmtree(child, ignore_errors = True)


def _read(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys = True), encoding = "utf-8")
    os.replace(tmp, path)


def read_pin(studio_home: Path) -> Optional[dict]:
    return _read(studio_home / PIN_FILE)


def write_pin(studio_home: Path, ref: str, sha: str, path: Path) -> None:
    _write(studio_home / PIN_FILE, {"ref": ref, "sha": sha, "path": str(path), "installed_at": time.time()})


def clear_pin(studio_home: Path) -> None:
    (studio_home / PIN_FILE).unlink(missing_ok = True)


def read_request(studio_home: Path) -> Optional[dict]:
    return _read(studio_home / REQUEST_FILE)


def write_request(studio_home: Path, ref: Optional[str], sha: Optional[str]) -> dict:
    """ref None means go back to the release."""
    data = {"ref": ref, "sha": sha, "requested_at": time.time()}
    _write(studio_home / REQUEST_FILE, data)
    return data


def clear_request(studio_home: Path) -> None:
    (studio_home / REQUEST_FILE).unlink(missing_ok = True)


def api_switch_allowed() -> bool:
    return os.environ.get(ALLOW_API_ENV, "").strip() == "1"
