# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pick the ``libbitsandbytes_rocm<NN>.dll`` suffix for BNB_ROCM_VERSION. Stdlib only, never imports torch.

Mirrored by ``unsloth/import_fixes.py``; tests/test_bnb_rocm_dll_rank.py keeps the two in step.
"""

import importlib.util
import os
import re
import sys
from typing import Iterable, Optional

_TORCH_HIP_RE = re.compile(r"^hip\s*(?::[^=]*)?=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)


def torch_hip_version() -> Optional[str]:
    """torch.version.hip, read from torch/version.py when torch is not imported yet."""
    torch = sys.modules.get("torch")
    if torch is not None:
        return getattr(getattr(torch, "version", None), "hip", None) or None
    try:
        spec = importlib.util.find_spec("torch")
        if spec is None or not spec.submodule_search_locations:
            return None
        for pkg_dir in spec.submodule_search_locations:
            path = os.path.join(pkg_dir, "version.py")
            if os.path.isfile(path):
                with open(path, encoding = "utf-8") as f:
                    match = _TORCH_HIP_RE.search(f.read())
                return match.group(1) if match else None
    except Exception:
        pass
    return None


def pick_bnb_rocm_dll_suffix(
    suffixes: Iterable[str], hip_version: Optional[str] = None
) -> Optional[str]:
    """Highest suffix by (major, minor), restricted to torch's HIP major when any DLL shares it.

    Suffixes split like bitsandbytes' own ``_split_cuda_version`` (one-digit major: "713" is 7.13).
    ROCm 10.x torch still reports HIP 7.x (10.1 is 7.16), so its DLLs rank as 7.x. With no
    same-major DLL the highest overall wins, as before.
    """
    suffixes = [s for s in suffixes if s and s.isdigit()]
    if not suffixes:
        return None
    major = None
    try:
        major = int(str(hip_version).split(".")[0]) if hip_version else None
    except ValueError:
        pass
    ranked = [((int(s[:1]), int(s[1:] or 0)), s) for s in suffixes]
    same_major = [r for r in ranked if r[0][0] == major]
    return max(same_major or ranked)[1]
