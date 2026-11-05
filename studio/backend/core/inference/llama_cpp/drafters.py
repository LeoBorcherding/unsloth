# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MTP, DSpark and DFlash drafter detection."""

import re

from pathlib import Path
from typing import Optional
from utils import gguf_names as _gguf_names

# Gemma 4 ships MTP as a separate drafter; Gemma 3n ships none.
_GEMMA_MTP_FAMILY_RE = re.compile(r"gemma[-_]?4[-_]", re.IGNORECASE)


def _is_gemma_mtp_family(name: Optional[str]) -> bool:
    """Match Gemma 4 by name."""
    return bool(name) and bool(_GEMMA_MTP_FAMILY_RE.search(name))


def _is_gemma_mtp_name(model_identifier: Optional[str], gguf_path: Optional[str] = None) -> bool:
    """Match Gemma 4 by id or GGUF filename."""
    return _is_gemma_mtp_family(model_identifier) or _is_gemma_mtp_family(
        Path(gguf_path).name if gguf_path else None
    )


def _is_mtp_model_name(model_identifier: Optional[str], gguf_path: Optional[str] = None) -> bool:
    """Name-based MTP detector. Fallback for the metadata signal."""
    for cand in (model_identifier, Path(gguf_path).name if gguf_path else None):
        if cand and "-mtp" in cand.lower():
            return True
        # Gemma 4 too: a failed drafter download surfaces a fallback reason instead of a silent default.
        if cand and _is_gemma_mtp_family(cand):
            return True
    return False


_DRAFTER_KINDS = _gguf_names.DRAFTER_KINDS
_DRAFTER_DIR_KINDS = _gguf_names.DRAFTER_DIR_KINDS

# Human label per resolved spec mode, for launch logging and the UI notice.
_DRAFTER_DISPLAY_LABELS = {"dspark": "DSpark", "dflash": "DFlash", "eagle3": "EAGLE-3"}


def _drafter_path_kind(path: str) -> Optional[str]:
    return _gguf_names.drafter_path_kind(path)


_IMATRIX_TOKEN_RE = _gguf_names.IMATRIX_TOKEN_RE


def _is_companion_gguf_path(path: str) -> bool:
    """Exclude companions from main-model selection; use _is_mtp_only_drafter_path for launch."""
    p = path.lower()
    if not p.endswith(".gguf"):
        return False
    if "mmproj" in p:
        return True
    if _gguf_names.is_imatrix_filename(p):
        return True
    return _gguf_names.is_mtp_drafter_path(path)


def _is_mtp_only_drafter_path(path: str) -> bool:
    """MTP only, not the other companion GGUFs. Each drafter kind needs its own
    ``--spec-type``, so discovery narrows to one kind rather than reusing the
    broad companion predicate, which exists to EXCLUDE companions from menus."""
    return _drafter_path_kind(path) == "mtp"


def _is_dspark_drafter_path(path: str) -> bool:
    """Broader than detect_dspark_file (no name check): only for repo listings and snapshots, since a local folder may hold anything."""
    return _drafter_path_kind(path) == "dspark"


def _is_dflash_drafter_path(path: str) -> bool:
    """Prefix only, since dflash is also a family name; a dflash/ folder does not mark a sidecar. Only a root-level dflash- prefixed GGUF names one."""
    return _drafter_path_kind(path) == "dflash"


def _is_root_dflash_drafter_path(path: str) -> bool:
    """Prefix AND repo root: a nested dflash- weight would otherwise download before rejection."""
    normalized = path.replace("\\", "/")
    return "/" not in normalized and _is_dflash_drafter_path(normalized)


_BIG_ENDIAN_GGUF_FILENAME_RE = re.compile(r"(^|[-_])be(?:[._-]|$)", re.IGNORECASE)
_GGUF_KNOWN_QUANT_RE = _gguf_names.GGUF_QUANT_RE
_DSPARK_BEFORE_QUANT_RE = _gguf_names.DSPARK_BEFORE_QUANT_RE


def _is_big_endian_gguf_path(path: str, variant_key: str = "") -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0].lower()
    variant_key = variant_key.strip().lower()
    variant_index = stem.find(variant_key) if variant_key else -1
    parent = normalized.rsplit("/", 1)[0].lower() if "/" in normalized else ""
    variant_in_parent_only = (
        bool(parent)
        and variant_index < 0
        and (
            (variant_key and variant_key in parent)
            or (not variant_key and _GGUF_KNOWN_QUANT_RE.search(parent) is not None)
        )
    )
    for match in _BIG_ENDIAN_GGUF_FILENAME_RE.finditer(stem):
        if variant_index >= 0 and variant_index < match.start():
            return True
        tail = stem[match.end() :].lstrip("._-")
        if not tail or _GGUF_KNOWN_QUANT_RE.search(tail) is None:
            return not variant_in_parent_only
    return False
