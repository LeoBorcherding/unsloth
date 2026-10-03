# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which unslothai/unsloth source the desktop runs, and a way to change it over the API.

GET  /api/studio/source  -> the pinned ref (null on the release) and any pending switch
POST /api/studio/source  -> {"ref": "<branch|tag|sha>"} or {"ref": null} for the release
DELETE /api/studio/source -> drop a pending switch

The backend cannot reinstall the venv it runs from, so POST only records the request; the
desktop picks it up, stops the backend, runs `unsloth studio update --ref` and starts it again.
Off unless UNSLOTH_ALLOW_SOURCE_SWITCH=1, since it lets an API key install arbitrary branches.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import policy
from auth.authentication import get_current_subject
from unsloth_cli import _studio_source_ref as source_ref
from utils.paths.storage_roots import studio_root

router = APIRouter()

DESKTOP_MANAGED_ENV = "UNSLOTH_DESKTOP_MANAGED"


class SourceSwitchRequest(BaseModel):
    ref: Optional[str] = Field(None, max_length = 200)


def _desktop_managed() -> bool:
    return os.environ.get(DESKTOP_MANAGED_ENV) == "1"


def _status() -> dict:
    home = studio_root()
    return {
        "pinned": source_ref.read_pin(home),
        "pending": source_ref.read_request(home),
        "switch_allowed": source_ref.api_switch_allowed(),
        "desktop_managed": _desktop_managed(),
    }


@router.get("/source")
def studio_source(current_subject: str = Depends(get_current_subject)):
    return _status()


@router.post(
    "/source",
    status_code = 202,
    dependencies = [Depends(get_current_subject), Depends(policy.require_owner)],
)
def studio_source_switch(body: SourceSwitchRequest, current_subject: str = Depends(get_current_subject)):
    # Sync def: resolve_ref calls GitHub, so this runs in the threadpool.
    if not source_ref.api_switch_allowed():
        raise HTTPException(
            status_code = 403,
            detail = f"Switching the Unsloth source over the API is off. Set {source_ref.ALLOW_API_ENV}=1 to allow it.",
        )
    ref = (body.ref or "").strip() or None
    if not _desktop_managed():
        # Without the desktop nothing restarts the backend after the update.
        command = f"unsloth studio update --ref {ref}" if ref else "unsloth studio update"
        raise HTTPException(
            status_code = 409,
            detail = f"Only Unsloth Desktop switches in place. Stop Unsloth Studio and run: {command}",
        )
    sha = None
    if ref is not None:
        try:
            sha = source_ref.resolve_ref(ref)
        except source_ref.SourceRefError as exc:
            raise HTTPException(status_code = 422, detail = str(exc)) from exc
    source_ref.write_request(studio_root(), ref, sha)
    return _status()


@router.delete(
    "/source",
    dependencies = [Depends(get_current_subject), Depends(policy.require_owner)],
)
def studio_source_cancel(current_subject: str = Depends(get_current_subject)):
    """Drop a pending switch, e.g. when the user declines it mid-training."""
    source_ref.clear_request(studio_root())
    return _status()
