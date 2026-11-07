# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video timeline: trim and join clips into one MP4 (multipart ``clips`` plus a JSON ``spec``). Runs no model."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from starlette.datastructures import UploadFile

from auth.authentication import get_current_subject
from core.inference import video_timeline
from core.inference.video_timeline import TimelineError
from utils.upload_limits import VIDEO_TIMELINE_UPLOAD_MAX_BYTES

router = APIRouter()


@router.post("/video/timeline/compose")
async def compose_timeline(request: Request, current_subject: str = Depends(get_current_subject)):
    """``spec`` is ``{"segments": [{"clip", "start", "end", "skip_head", "skip_tail"}], "width", "height", "fps"}``;
    ``clips`` are the files the segments index, in order. Answers the joined MP4; ``X-Segment-Frames`` lists the
    frames each segment contributed."""
    try:
        form = await request.form(max_files = video_timeline.MAX_SEGMENTS)
    except Exception:  # noqa: BLE001
        raise HTTPException(
            status_code = 400, detail = "Send the clips and a spec as multipart/form-data."
        )
    uploads = [
        value
        for name, value in form.multi_items()
        if name == "clips" and isinstance(value, UploadFile)
    ]
    if sum(u.size or 0 for u in uploads) > VIDEO_TIMELINE_UPLOAD_MAX_BYTES:
        raise HTTPException(status_code = 413, detail = "The upload is too large.")
    raw_spec = form.get("spec")
    try:
        spec = json.loads(raw_spec) if isinstance(raw_spec, str) else None
    except ValueError:
        spec = None
    if not isinstance(spec, dict):
        raise HTTPException(
            status_code = 400, detail = "spec must be a JSON object with a segments list."
        )
    try:
        segments = video_timeline.parse_segments(spec.get("segments"), len(uploads))
        size = {}
        for key, low, high in (("width", 16, 4096), ("height", 16, 4096), ("fps", 1, 60)):
            value = spec.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise TimelineError(f"{key} must be a whole number from {low} to {high}.")
            if key != "fps" and value % 2:
                raise TimelineError(f"{key} must be even for H.264.")
            size[key] = value
        clips = [await u.read() for u in uploads]
        mp4, counts = await asyncio.to_thread(video_timeline.compose, clips, segments, **size)
    except TimelineError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    return Response(mp4, media_type = "video/mp4", headers = {"X-Segment-Frames": json.dumps(counts)})
