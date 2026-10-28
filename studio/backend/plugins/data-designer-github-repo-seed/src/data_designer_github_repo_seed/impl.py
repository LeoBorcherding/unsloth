# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Optional

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.resources.seed_reader import SeedReader
from utils.paths import github_seed_cache_root
from utils.paths.lazy import LazyPath

from .config import GitHubRepoSeedSource
from .scraper import ScrapeConfig, materialize_to_jsonl


# Each job runs in its own spawned worker, so the scrape is cached on disk, in the account's tmp root, for
# the preview and run that follow. The key holds a short SHA-256 of the token, never the token.
_CACHE_DIR = LazyPath(github_seed_cache_root)
_SCRAPE_TTL_SECONDS = 60 * 60
# Deleted only well past the TTL: a run that started on a scrape keeps reading it for hours.
_SCRAPE_KEEP_SECONDS = 24 * 60 * 60
_SCRAPE_CACHE_LOCK = threading.Lock()
# This job's scrape per key: the engine asks for the URI again when it cycles the seed mid-run.
_JOB_SCRAPES: dict[str, str] = {}


def _scrape_cache_key(cfg: ScrapeConfig) -> str:
    token_digest = hashlib.sha256(
        (cfg.token or "").encode("utf-8"),
    ).hexdigest()[:16]
    key = (
        list(cfg.repos),
        list(cfg.item_types),
        cfg.limit,
        bool(cfg.include_comments),
        cfg.max_comments_per_item,
        token_digest,
    )
    return hashlib.sha256(json.dumps(key).encode("utf-8")).hexdigest()[:24]


def _fresh_cached_scrape(cache_dir: Path, key: str) -> Optional[Path]:
    """The newest scrape for `key` younger than the TTL; scrapes of any key past the keep window are removed."""
    now = time.time()
    fresh: Optional[Path] = None
    fresh_mtime = 0.0
    for path in cache_dir.glob("github_*.jsonl"):
        try:
            mtime = path.stat().st_mtime
            if now - mtime > _SCRAPE_KEEP_SECONDS:
                path.unlink()
                continue
        except OSError:
            # Gone already, or a job still has it open (Windows).
            continue
        if (
            now - mtime <= _SCRAPE_TTL_SECONDS
            and path.name.startswith(f"github_{key}_")
            and mtime > fresh_mtime
        ):
            fresh, fresh_mtime = path, mtime
    return fresh


class GitHubRepoSeedReader(SeedReader[GitHubRepoSeedSource]):
    def create_duckdb_connection(self):
        return lazy.duckdb.connect()

    def get_dataset_uri(self) -> str:
        cfg = ScrapeConfig(
            repos = list(self.source.repos),
            token = self.source.token,
            item_types = list(self.source.item_types),
            limit = self.source.limit,
            include_comments = self.source.include_comments,
            max_comments_per_item = self.source.max_comments_per_item,
        )
        cache_dir = Path(_CACHE_DIR)
        key = _scrape_cache_key(cfg)
        with _SCRAPE_CACHE_LOCK:
            path = _JOB_SCRAPES.get(key)
            if path and Path(path).exists():
                return path
            cache_dir.mkdir(parents = True, exist_ok = True)
            cached = _fresh_cached_scrape(cache_dir, key)
            if cached is None:
                cached = materialize_to_jsonl(cfg, cache_dir, stem = f"github_{key}")
            _JOB_SCRAPES[key] = str(cached)
            return str(cached)
