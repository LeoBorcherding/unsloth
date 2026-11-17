# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The optional second venv with the other GPU vendor's torch (#12248).

install.sh builds it with UNSLOTH_SECOND_VENDOR_TORCH=1 on a mixed NVIDIA+AMD host and records
it in ``<studio root>/second_torch_venv.json``. Nothing runs jobs in it yet.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

RECORD_NAME = "second_torch_venv.json"
_VENDORS = ("cuda", "rocm")


@dataclass(frozen = True)
class SecondTorchVenv:
    vendor: str
    python: Path
    venv: Path
    torch_version: str


def second_torch_venv(
    root: Path | None = None, *, main_vendor: str | None = None
) -> SecondTorchVenv | None:
    """The recorded second venv, or None when there is none usable.

    ``main_vendor`` ("cuda"/"rocm") is the main venv's torch; a record for the same vendor is
    stale (the main venv was reinstalled for the other card) and is ignored.
    """
    if root is None:
        from utils.paths.storage_roots import studio_root
        root = studio_root()
    try:
        data = json.loads((Path(root) / RECORD_NAME).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != 1:
        return None
    vendor, python, venv = data.get("vendor"), data.get("python"), data.get("venv")
    if vendor not in _VENDORS or not isinstance(python, str) or not isinstance(venv, str):
        return None
    if main_vendor is not None and vendor == main_vendor:
        return None
    python_path = Path(python)
    if not (python_path.is_file() and os.access(python_path, os.X_OK)):
        return None
    return SecondTorchVenv(
        vendor = vendor,
        python = python_path,
        venv = Path(venv),
        torch_version = str(data.get("torch_version") or ""),
    )
