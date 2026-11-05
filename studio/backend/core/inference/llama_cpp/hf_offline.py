# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hugging Face reachability and offline-mode helpers."""

import contextlib
import os

from typing import Optional

from .log import logger


def _hf_endpoint_host() -> str:
    """Host of the configured hub endpoint (HF_ENDPOINT aware)."""
    from utils.utils import hf_endpoint_host
    return hf_endpoint_host()


def _probe_dns_dead(host: Optional[str] = None, timeout: float = 2.0) -> bool:
    """Quick DNS check on a daemon thread, so concurrent sockets aren't
    affected by socket.setdefaulttimeout. Defaults to the configured endpoint's host."""
    from utils.utils import dns_host_dead, hf_endpoint_host
    return dns_host_dead(host or hf_endpoint_host(), timeout)


def _hf_env_offline() -> bool:
    """Accepts any truthy HF_HUB_OFFLINE (1, true, yes, on) to match _env_offline, not only 1."""
    try:
        from utils.models.model_config import _env_offline
        return _env_offline()
    except Exception:
        return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _hf_unreachable() -> bool:
    """DNS alone misses captive portals and WAN outages; the DNS shortcut is skipped when a proxy is set."""
    try:
        from utils.utils import hf_dns_dead, hf_reachability_memo, hf_unreachable
    except Exception:
        return False
    # Reuse a fresh verdict (a slow resolver costs a timeout per guard). Peek only, never record: recording a dead lookup would make recovery sticky for the TTL.
    memo = hf_reachability_memo()
    if memo is not None:
        return memo
    if hf_dns_dead():
        return True
    return hf_unreachable()


# Separates "nothing remembered" from a remembered verdict of None (not a media GGUF).
_MISS = object()


# GGUF unsigned value-type id -> byte width (uint8, uint16, uint32, uint64).
_GGUF_UINT_WIDTHS = {0: 1, 2: 2, 4: 4, 10: 8}


_GGUF_MARKUP_FIRST_BYTES = frozenset(b"<[")


@contextlib.contextmanager
def _hf_offline_if_unreachable():
    """Refcounted so one guard exit cannot re-enable the Hub mid-request. Forces offline only when unreachable; no-op if the user set the env var."""
    try:
        from utils.utils import force_hf_offline, force_hf_offline_state, hf_env_offline
    except Exception:
        yield False
        return

    ours, hf_hub_env_present = force_hf_offline_state()
    # Membership keeps HF_HUB_OFFLINE=0 as a stay-online override. Read it under the refcount so a concurrent guard cannot look user-owned.
    if hf_hub_env_present and not ours:
        yield False
        return
    # The hub ignores TRANSFORMERS_OFFLINE, so honour it here without probing.
    if not ours and hf_env_offline():
        with force_hf_offline():
            yield True
        return
    # Already forced by an in-flight guard: reuse that verdict and hold a reference.
    if not ours and not _hf_unreachable():
        yield False
        return

    if not ours:
        logger.warning("Hugging Face endpoint unreachable; using local HF cache for this load.")
    with force_hf_offline():
        yield True


def _hf_offline_if_unreachable_for(model_name):
    """Guard, but only for remote repo ids: a local path is served from disk and never
    reaches the hub, so probing would add seconds per request and prevent no retry."""
    try:
        from utils.paths import is_local_path
        if isinstance(model_name, str) and is_local_path(model_name):
            return contextlib.nullcontext()
    except Exception:
        pass
    return _hf_offline_if_unreachable()


def _hf_repo_from_url(url: Optional[str]) -> Optional[str]:
    """Strips a huggingface.co or mirrored HF_ENDPOINT URL to owner/name; huggingface.co still accepted."""
    if not url:
        return None
    from utils.hf_endpoint import get_hf_endpoint

    marker = "huggingface.co/"
    if marker not in url:
        mirror = get_hf_endpoint().split("://", 1)[-1].rstrip("/") + "/"
        if mirror in url:
            marker = mirror
        else:
            return None
    tail = url.split(marker, 1)[1].rstrip("/")
    parts = tail.split("/")
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


# Lazy import to avoid pulling transformers in at module level.
def _extract_model_size_b(model_id: str):
    from utils.models import extract_model_size_b
    return extract_model_size_b(model_id)
