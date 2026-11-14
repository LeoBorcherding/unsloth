# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hugging Face cache inspection: GGUF snapshots, variants, shards and companions."""

import os
import re

from pathlib import Path
from typing import (
    Callable,
    Generator,
    Iterable,
    Mapping,
    Optional,
)
from hub.utils.hf_tokens import call_hub_with_anonymous_retry
from utils.paths.path_utils import is_appledouble_metadata

from .drafters import (
    _is_big_endian_gguf_path,
    _is_companion_gguf_path,
    _is_dspark_drafter_path,
)
from .log import logger
import functools

# ── Pre-compiled patterns for GGUF shard detection ───────────
_SHARD_FULL_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_SHARD_RE = re.compile(r"^(.*)-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


def _gguf_set_bytes(path: Path) -> int:
    """Size of a GGUF plus, when it is a split one, its sibling shards: llama-server opens the whole set."""
    total = 0
    try:
        if path.is_file():
            total += path.stat().st_size
        m = _SHARD_FULL_RE.match(path.name)
        if m is None or not path.parent.is_dir():
            return total
        # llama.cpp names a shard "<prefix>-<i>-of-<count>.gguf" from the first one's, so both must match: a
        # drafter split as "<main prefix>-MTP-..." or a stale set with another count is not this set.
        prefix, count = m.group(1).lower(), int(m.group(3))
        for sibling in path.parent.iterdir():
            if sibling.name == path.name:
                continue
            sibling_match = _SHARD_FULL_RE.match(sibling.name)
            if (
                sibling_match is None
                or sibling_match.group(1).lower() != prefix
                or int(sibling_match.group(3)) != count
                or not 1 <= int(sibling_match.group(2)) <= count
            ):
                continue
            try:
                if sibling.is_file():
                    total += sibling.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def _gguf_snapshot_files(snapshot: Path) -> list[str]:
    return [
        p.relative_to(snapshot).as_posix()
        for p in snapshot.rglob("*")
        if p.is_file() and p.name.lower().endswith(".gguf") and not is_appledouble_metadata(p)
    ]


def _cached_hf_snapshot_file(
    repo_id: str,
    filename: str,
    *,
    expected_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    """Return a cached snapshot file even when HF's current-ref probe misses it."""
    if not filename:
        return None
    parts = [part for part in filename.replace("\\", "/").split("/") if part]
    if not parts or any(part in (".", "..") for part in parts):
        return None
    try:
        if cache_dir is None:
            from utils.models.model_config import _iter_hf_cache_snapshots
            snapshots = _iter_hf_cache_snapshots(repo_id)
        else:
            from hub.utils.hf_cache_state import iter_active_repo_cache_dirs
            snapshots = (
                snapshot
                for repo_dir in iter_active_repo_cache_dirs(
                    "model",
                    repo_id,
                    root = Path(cache_dir),
                )
                for snapshot in (repo_dir / "snapshots").glob("*")
                if snapshot.is_dir()
            )
        for snap in snapshots:
            candidate = snap.joinpath(*parts)
            if not candidate.is_file():
                continue
            if expected_size:
                try:
                    if candidate.stat().st_size < expected_size:
                        continue
                except OSError:
                    continue
            return str(candidate)
    except Exception as e:
        logger.debug("Snapshot cache lookup failed for %s/%s: %s", repo_id, filename, e)
    return None


def _snapshot_has_all_shards(
    main_path: str, main_filename: str, shards: Iterable[str], expected_sizes: dict[str, int]
) -> bool:
    """llama.cpp finds split shards beside the main file, so the whole set must share a snapshot."""
    root = Path(main_path)
    for _ in [part for part in main_filename.replace("\\", "/").split("/") if part]:
        root = root.parent
    for shard in shards:
        parts = [part for part in shard.replace("\\", "/").split("/") if part]
        if not parts or any(part in (".", "..") for part in parts):
            return False
        sibling = root.joinpath(*parts)
        try:
            if not sibling.is_file():
                return False
            expected = expected_sizes.get(shard)
            if expected and sibling.stat().st_size < expected:
                return False
        except OSError:
            return False
    return True


def _resolve_repo_id_casing(hf_repo: str) -> str:
    """Maps a case-variant id to the cached casing so main and companion GGUFs share one cache entry; returns hf_repo unchanged on failure."""
    try:
        from utils.paths import resolve_cached_repo_id_case
        return resolve_cached_repo_id_case(hf_repo)
    except Exception:
        return hf_repo


def _cached_gguf_snapshots(repo_id: str):
    """Retain the existing active-cache lookup; add remembered chat copies once."""
    from utils.models.model_config import _iter_hf_cache_snapshots
    from hub.utils.gguf_sources import gguf_cache_snapshots

    seen = set()
    for snapshots in (_iter_hf_cache_snapshots(repo_id), gguf_cache_snapshots(repo_id)):
        for snapshot in snapshots:
            key = str(snapshot.resolve())
            if key not in seen:
                seen.add(key)
                yield snapshot


def _cached_colocated_split_main(
    repo_id: str, main_filename: str, shards: Iterable[str], expected_sizes: dict[str, int]
) -> Optional[str]:
    """Prefers a snapshot with the full split set; a newer partial main would force a refetch."""
    main_parts = [part for part in main_filename.replace("\\", "/").split("/") if part]
    if not main_parts or any(part in (".", "..") for part in main_parts):
        return None
    try:
        for snap in _cached_gguf_snapshots(repo_id):
            main_path = snap.joinpath(*main_parts)
            if not main_path.is_file():
                continue
            expected_main = expected_sizes.get(main_filename)
            try:
                if expected_main and main_path.stat().st_size < expected_main:
                    continue
            except OSError:
                continue
            if _snapshot_has_all_shards(str(main_path), main_filename, shards, expected_sizes):
                return str(main_path)
    except Exception as e:
        logger.debug("Co-located split snapshot lookup failed for %s: %s", repo_id, e)
    return None


def _cached_variant_resolution(repo_id: str, hf_variant: str) -> tuple[Optional[str], list[str]]:
    """Find a cached main GGUF and its shards for a variant."""
    candidate = next(_cached_variant_candidates(repo_id, hf_variant), None)
    if candidate is None:
        return None, []
    _, main, shards, _ = candidate
    return main, shards


def _cached_variant_candidates(
    repo_id: str,
    hf_variant: str,
    *,
    require_mmproj: bool = False,
    strict: bool = False,
) -> Generator[tuple[str, str, list[str], Path], None, None]:
    """Incomplete or cancelled copies come last, and strict drops them so only a Hub-free copy qualifies."""
    try:
        from hub.utils.gguf_sources import (
            cached_gguf_manifest_complete,
            cached_gguf_source_partial,
        )

        pending_downloads = []
        for snap in _cached_gguf_snapshots(repo_id):
            cached_files = _gguf_snapshot_files(snap)
            matches = _gguf_files_for_variant(cached_files, hf_variant)
            if not matches:
                continue
            main = matches[0]
            shards = _gguf_extra_shards(matches, main)
            split = _SHARD_FULL_RE.match(main)
            if split:
                numbers = {
                    int(match.group(2))
                    for path in [main, *shards]
                    if (match := _SHARD_FULL_RE.match(path))
                }
                if numbers != set(range(1, int(split.group(3)) + 1)):
                    continue
            main_path = snap.joinpath(*main.replace("\\", "/").split("/"))
            if not main_path.is_file() or not _snapshot_has_all_shards(
                str(main_path), main, shards, {}
            ):
                continue
            if require_mmproj and not _pick_mmproj(cached_files):
                continue
            candidate = (str(main_path), main, shards, snap)
            if not cached_gguf_manifest_complete(
                repo_id, hf_variant, snap
            ) or cached_gguf_source_partial(repo_id, hf_variant, snap):
                pending_downloads.append(candidate)
                continue
            yield candidate
        # Keep existing reuse semantics when no completed duplicate is available.
        if not strict:
            yield from pending_downloads
    except Exception as e:
        logger.debug(f"Cache lookup for variant failed: {e}")


def _cached_variant_sizes(
    repo_id: str, hf_variant: str, main_path: str
) -> Optional[tuple[tuple[str, int], ...]]:
    """Sorted (filename, size) pairs of the complete cached copy at main_path, or None; compared later to catch truncated or dropped shards offline."""
    target = os.path.abspath(main_path)
    for cached_main, main, shards, snap in _cached_variant_candidates(repo_id, hf_variant):
        if os.path.abspath(cached_main) != target:
            continue
        try:
            return tuple(
                sorted(
                    (name, os.path.getsize(snap.joinpath(*name.replace("\\", "/").split("/"))))
                    for name in (main, *shards)
                )
            )
        except OSError:
            return None
    return None


def _cached_candidate_matches_revision_size(
    repo_id: str, candidate: tuple[str, str, list[str], Path], hf_token: Optional[str]
) -> bool:
    """Compares sizes with the Hub revision when reachable, so a manually truncated cache is not reused."""
    main_path, main, shards, snap = candidate
    paths = [main, *shards]
    try:
        from huggingface_hub import get_paths_info
        infos = list(
            call_hub_with_anonymous_retry(
                get_paths_info,
                hf_token,
                repo_id,
                paths,
                revision = snap.name,
            )
        )
    except Exception as e:
        logger.debug(
            "Could not size-check cached GGUF %s at revision %s: %s",
            repo_id,
            snap.name,
            e,
        )
        return True

    if not infos:
        # The Hub answers an unknown revision with an empty result, not an error; treat it like unreachable metadata.
        return True
    expected_sizes = {info.path: info.size for info in infos if info.size is not None}
    if any(path not in expected_sizes for path in paths):
        return False
    try:
        if os.path.getsize(main_path) < expected_sizes[main]:
            return False
    except OSError:
        return False
    return _snapshot_has_all_shards(main_path, main, shards, expected_sizes)


def _cached_complete_candidate(
    repo_id: str, gguf_filename: Optional[str], shards: list[str]
) -> Optional[tuple[str, str, list[str], Path]]:
    """Return one complete exact-filename cache candidate with snapshot context."""
    if not gguf_filename:
        return None
    if shards:
        main_path = _cached_colocated_split_main(repo_id, gguf_filename, shards, {})
    else:
        m = _SHARD_FULL_RE.match(gguf_filename)
        if m and int(m.group(3)) > 1:
            return None
        main_path = _cached_hf_snapshot_file(repo_id, gguf_filename)
    if main_path is None:
        return None
    snap = _snapshot_dir_of(main_path)
    if snap is None:
        return None
    return main_path, gguf_filename, shards, snap


def cached_gguf_for_load(
    hf_repo: str,
    hf_variant: Optional[str],
    *,
    require_mmproj: bool = False,
    verify_sizes: bool = False,
    hf_token: Optional[str] = None,
    strict: bool = False,
) -> Optional[str]:
    """Return a cached GGUF that can be loaded without downloading (``strict``: and whose
    recorded download is complete, see _cached_variant_candidates)."""
    if not hf_variant:
        return None
    hf_repo = _resolve_repo_id_casing(hf_repo)
    for candidate in _cached_variant_candidates(
        hf_repo,
        hf_variant,
        require_mmproj = require_mmproj,
        strict = strict,
    ):
        if verify_sizes and not _cached_candidate_matches_revision_size(
            hf_repo, candidate, hf_token
        ):
            continue
        return candidate[0]
    return None


def _snapshot_dir_of(path: str) -> Optional[Path]:
    """Return the HF cache snapshot containing path, if any."""
    try:
        p = Path(os.path.abspath(path))
    except OSError:
        return None
    for ancestor in p.parents:
        if ancestor.parent.name == "snapshots":
            return ancestor
    return None


def _hub_cache_dir_for_snapshot_path(path: Optional[str]) -> Optional[str]:
    """Return the HF Hub cache root that owns a snapshot-contained path."""
    if not path:
        return None
    snapshot = _snapshot_dir_of(path)
    if snapshot is None or snapshot.parent.name != "snapshots":
        return None
    return str(snapshot.parent.parent.parent)


def _companion_snapshot_sibling(
    near_path: str, pick: Callable[[list[str]], Optional[str]]
) -> Optional[str]:
    """Find a companion in the same snapshot as near_path."""
    snap = _snapshot_dir_of(near_path)
    if snap is None:
        return None
    try:
        sibling = pick(_gguf_snapshot_files(snap))
    except Exception:
        return None
    if not sibling:
        return None
    candidate = snap / sibling
    if not candidate.is_file():
        return None
    # Split companion is usable only as a whole set: llama-server resolves siblings from its directory, so shard 1 alone is not a reuse.
    from utils.models.model_config import _drafter_split_is_complete

    return str(candidate) if _drafter_split_is_complete(candidate) else None


def _pick_dspark(candidates: list[str]) -> Optional[str]:
    """The DSpark drafter a listing offers, or None. Module level for the same reason
    ``_pick_mmproj`` is: both are handed a live repo listing as well as a snapshot."""
    from hub.utils.gguf import drop_shadowed_appledouble_names
    from utils.models.model_config import dspark_preference_key

    # Every GGUF under dspark/ qualifies, so a sidecar ranks equal to its sibling and sorts first.
    files = sorted(
        (
            name
            for name in drop_shadowed_appledouble_names(list(candidates))
            if _is_dspark_drafter_path(name)
        ),
        key = dspark_preference_key,
    )
    return files[0] if files else None


_MTP_SHARD_SUFFIX_RE = re.compile(r"-[0-9]{5}-of-[0-9]{5}$")


def _is_published_mtp_drafter_name(path: str) -> bool:
    """Basename only, since any MTP/ file is too broad to launch. Names mtp-<model> or <model>-MTP, with any shard suffix stripped first."""
    lower = Path(path).name.lower()
    if not lower.endswith(".gguf"):
        return False
    stem = _MTP_SHARD_SUFFIX_RE.sub("", Path(lower).stem)
    return lower.startswith("mtp-") or stem.endswith("-mtp")


def _mtp_head_borrows(path: str) -> bool:
    """Is this the published ``-shared-`` form, which loads only under its target?"""
    return "-shared-" in Path(path).name.lower()


def _pick_mtp(candidates: list[str], *, allow_nested: bool = True) -> Optional[str]:
    """The MTP drafter a listing offers, or None; module level so /kv-cache-estimate prices the drafter the launch opens. allow_nested=False: root only."""
    from hub.utils.gguf import drop_shadowed_appledouble_names
    from utils.models.drafters import split_listing_is_complete
    from utils.models.drafters.preference import mtp_preference_key

    names = drop_shadowed_appledouble_names(list(candidates))

    def _launchable(name: str) -> bool:
        # Half a split set is unusable, so settle completeness before ranking, or an incomplete head shadows a complete lower-ranked one.
        return split_listing_is_complete(names, name)

    # Root first, so a root mirror (Gemma 4) beats a subdir copy; the mtp- prefix already excludes AppleDouble ._mtp shadows, so no filter is needed.
    mtp_files = sorted(
        f
        for f in names
        if f.lower().endswith(".gguf")
        and "/" not in f
        and Path(f).name.lower().startswith("mtp-")
        and _launchable(f)
    )
    if mtp_files:
        return mtp_files[0]

    # Fall back to MTP/, where Qwen3.8-Flash-Next publishes heads (matches the offline cache).
    if not allow_nested:
        return None
    nested = sorted(
        (
            name
            for name in names
            if "/" in name and _is_published_mtp_drafter_name(name) and _launchable(name)
        ),
        key = mtp_preference_key,
    )
    return nested[0] if nested else None


def _pick_mtp_root_only(candidates: list[str]) -> Optional[str]:
    """Root mirror only: a passed sidecar displaces the embedded head, so a nested copy buys nothing."""
    return _pick_mtp(candidates, allow_nested = False)


def _mtp_drafter_loads_standalone(path: str) -> bool:
    """A --model-draft file must have the embeddings it needs to load; the rule lives in gguf_metadata so the memory estimate shares its cache."""
    from utils.models.gguf_metadata import mtp_drafter_loads_standalone
    return mtp_drafter_loads_standalone(path)


def _pick_mmproj(
    candidates: list[str],
    *,
    order: Optional[tuple] = None,
    weight: Optional[str] = None,
) -> Optional[str]:
    from hub.utils.gguf import drop_shadowed_appledouble_names

    # "._mmproj-F16.gguf" satisfies the F16 preference and sorts ahead of the real adapter.
    candidates = drop_shadowed_appledouble_names(list(candidates))
    mmproj_files = sorted(
        f for f in candidates if f.lower().endswith(".gguf") and "mmproj" in Path(f).name.lower()
    )
    if not mmproj_files:
        return None
    if order:
        from utils.models.model_config import mmproj_order_rank, mmproj_weight_match

        # Pair with the weight first, so a repo of several checkpoints never hands over another one's projector.
        return max(
            mmproj_files,
            key = lambda f: (mmproj_weight_match(weight, f), mmproj_order_rank(f, order)),
        )
    return next((f for f in mmproj_files if f.lower().endswith("-f16.gguf")), mmproj_files[0])


def _mmproj_picker(
    model: Optional[str], weight_path: Optional[str] = None
) -> Callable[[list[str]], Optional[str]]:
    from utils.models.model_config import mmproj_order

    order = mmproj_order(model, weight_path)
    if not order:
        return _pick_mmproj
    weight = Path(weight_path).name if weight_path else None
    return functools.partial(_pick_mmproj, order = order, weight = weight)


def _hub_download_in_flight(hf_repo: str) -> bool:
    try:
        from hub.utils.download_registry import get_models_registry
        return bool(get_models_registry().active_job_refs(hf_repo))
    except Exception:
        return False


def _hub_download_blocks_gguf_load(
    hf_repo: str,
    hf_variant: Optional[str],
    *,
    require_mmproj: bool = False,
    hf_token: Optional[str] = None,
) -> bool:
    """A same-variant Hub job always blocks (it can reclaim the snapshot this load reuses); other jobs block only with no complete cached copy."""
    try:
        from hub.utils.download_registry import get_models_registry

        registry = get_models_registry()
        if not registry.active_job_refs(hf_repo):
            return False
        if registry.has_active_variant(hf_repo, hf_variant):
            return True
    except Exception:
        return False
    return (
        cached_gguf_for_load(
            hf_repo,
            hf_variant,
            require_mmproj = require_mmproj,
            verify_sizes = True,
            hf_token = hf_token,
        )
        is None
    )


def _drafter_set_bytes(sizes: Mapping[str, int], name: str) -> int:
    """Total listed bytes of ``name``'s whole shard set (its own size when single)."""
    return (sizes.get(name, 0) or 0) + sum(
        sizes.get(shard, 0) or 0 for shard in _gguf_extra_shards(sizes, name)
    )


def _gguf_extra_shards(files: Iterable[str], first_shard: str) -> list[str]:
    m = _SHARD_FULL_RE.match(first_shard)
    if not m:
        return []
    prefix = m.group(1)
    total = m.group(3)
    sibling_pat = re.compile(
        r"^" + re.escape(prefix) + r"-\d{5}-of-" + re.escape(total) + r"\.gguf$",
        re.IGNORECASE,
    )
    return sorted(f for f in files if f != first_shard and sibling_pat.match(f))


def _quant_label_for_endian(path: str) -> Optional[str]:
    """With no label the caller keeps its previous argument, so the endian test is never dropped."""
    try:
        from hub.utils.gguf import extract_quant_token
        return extract_quant_token(path)
    except Exception:  # noqa: BLE001 -- an unreadable label must not sink the resolution
        return None


def _gguf_files_for_variant(files: Iterable[str], variant: str) -> list[str]:
    """Exact quant-label matches win over substrings, so stories260K does not match stories260K-be.gguf."""
    from hub.utils.gguf import drop_shadowed_appledouble_names

    variant_key = variant.strip().lower()
    # A repo listing has no bytes to read; a local one arrives already filtered on its headers.
    main_files = [
        f
        for f in drop_shadowed_appledouble_names(list(files))
        if f.lower().endswith(".gguf")
        and not _is_companion_gguf_path(f)
        # Pass the file's own quant label: a path-qualified key misreads -be.gguf as big-endian.
        and not _is_big_endian_gguf_path(f, _quant_label_for_endian(f) or variant_key)
    ]
    if not variant_key:
        return sorted(main_files)

    try:
        from utils.models.model_config import _extract_quant_label, _gguf_variant_key
    except Exception:
        _extract_quant_label = None
        _gguf_variant_key = None

    if _gguf_variant_key is not None:
        try:
            # Variant's own files first: mixing checkpoints makes llama-server read the wrong shards.
            owned = sorted(f for f in main_files if _gguf_variant_key(f).lower() == variant_key)
            if owned:
                return owned
        except Exception as e:
            logger.warning("Failed to derive GGUF variant keys: %s", e)

    if _extract_quant_label is not None:
        try:
            exact = sorted(f for f in main_files if _extract_quant_label(f).lower() == variant_key)
            if exact:
                return exact
        except Exception as e:
            logger.warning("Failed to extract GGUF quant labels: %s", e)

    boundary = re.compile(r"(?<![a-zA-Z0-9])" + re.escape(variant_key) + r"(?![a-zA-Z0-9])")
    return sorted(f for f in main_files if boundary.search(f.lower()))


def _resolve_variant_gguf_files(
    hf_repo: str,
    hf_variant: Optional[str],
    hf_token: Optional[str] = None,
) -> tuple[Optional[str], list[str]]:
    """Module level so the header probe and the download resolve the same file; two copies would drift."""
    gguf_filename: Optional[str] = None
    gguf_extra_shards: list[str] = []
    if not hf_variant:
        return None, []
    try:
        from huggingface_hub import list_repo_files

        files = call_hub_with_anonymous_retry(list_repo_files, hf_token, hf_repo)
        gguf_files = _gguf_files_for_variant(files, hf_variant)
        if gguf_files:
            gguf_filename = gguf_files[0]
            gguf_extra_shards = _gguf_extra_shards(gguf_files, gguf_filename)
    except Exception as e:
        logger.warning(f"Could not list repo files: {e}")

    # Fall back to the local cache when the repo listing is unavailable.
    if not gguf_filename:
        cached_name, cached_shards = _cached_variant_resolution(hf_repo, hf_variant)
        if cached_name:
            gguf_filename = cached_name
            gguf_extra_shards = cached_shards
            logger.info(
                "Resolved variant %s -> %s from local HF cache",
                hf_variant,
                gguf_filename,
            )

    if not gguf_filename:
        repo_name = hf_repo.split("/")[-1].replace("-GGUF", "")
        gguf_filename = f"{repo_name}-{hf_variant}.gguf"
    return gguf_filename, gguf_extra_shards
