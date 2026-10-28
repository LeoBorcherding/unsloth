# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm unified-memory classifier; a leaf so ``--no-torch`` installs can import it."""

from __future__ import annotations

import re
from typing import Any

# Integrated-only Radeon names for the other APU targets, as whole tokens (a substring would read "RX 7600M" as "600M"):
# 820M Krackan Point 2 (gfx1153), 8040S Strix Halo (gfx1151), 780M / 760M / 740M Phoenix / Hawk Point (gfx1103),
# 680M / 660M Rembrandt (gfx1035), 610M Raphael / Dragon Range / Mendocino (gfx1036 / gfx1037).
_ROCM_APU_NAME_RE = re.compile(r"(?<![\w-])(?:820m|8040s|780m|760m|740m|680m|660m|610m)(?!\w)")


def rocm_classify_unified_memory(props: Any) -> tuple[str, bool]:
    """Classify a ROCm device as unified-memory (APU) or discrete.

    Returns ``(gcn_arch, is_unified)``:
    - ``gcn_arch``: canonical arch string (e.g. ``"gfx1151"``) when a known
      attribute is present, else ``""``.
    - ``is_unified``: ``True`` for AMD APUs with a shared GPU/system-RAM pool
      (gfx1150 Strix Point, gfx1151 Strix Halo, gfx1152 Krackan Point) — these
      need a lower ``set_per_process_memory_fraction`` cap to leave OS headroom.

    Priority: truthy ``props.is_integrated`` (only ever upgrades), then
    ``gcnArchName`` variants, then device-name markers (890M/880M, 8065S/8060S/8050S,
    860M/840M, and the whole-token names in ``_ROCM_APU_NAME_RE``) when no arch attribute
    is populated.
    """
    gcn_arch = ""
    for _attr in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
        _v = (getattr(props, _attr, "") or "").split(":")[0].strip()
        if _v:
            gcn_arch = _v
            break

    # Additive only: a wheel omitting is_integrated cannot downgrade the known APU set.
    if getattr(props, "is_integrated", 0):
        return gcn_arch, True

    if gcn_arch:
        # gfx1152 (Krackan Point) shares system RAM like gfx1150/gfx1151.
        return gcn_arch, gcn_arch.lower() in {"gfx1150", "gfx1151", "gfx1152"}

    # Only reached under IS_ROCM, so NVIDIA GeForce 840M cannot collide.
    dev_lower = (getattr(props, "name", "") or "").lower()
    is_unified = (
        "890m" in dev_lower
        or "880m" in dev_lower
        or "8065s" in dev_lower
        or "8060s" in dev_lower
        or "8050s" in dev_lower
        or "860m" in dev_lower
        or "840m" in dev_lower
        or _ROCM_APU_NAME_RE.search(dev_lower) is not None
    )
    return gcn_arch, is_unified
