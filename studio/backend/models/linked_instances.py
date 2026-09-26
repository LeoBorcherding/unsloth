# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic schemas for the linked Unsloth Studio instances API."""

from typing import Optional

from pydantic import BaseModel, Field


class LinkedInstanceCreate(BaseModel):
    name: str = Field(..., max_length = 32, description = "Short name; models appear as @<name>/<model>")
    base_url: str = Field(..., max_length = 2048, description = "The remote's URL, e.g. its trycloudflare.com address")
    api_key: str = Field(..., min_length = 1, max_length = 512, description = "An API key created on the remote")


class LinkedInstanceUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length = 32)
    base_url: Optional[str] = Field(None, max_length = 2048)
    api_key: Optional[str] = Field(None, max_length = 512)


class LinkedInstance(BaseModel):
    id: str
    name: str
    base_url: str
    created_at: str
    updated_at: str


class LinkedInstanceStatus(BaseModel):
    id: str
    online: bool
    error: Optional[str] = None
    models: list[str] = Field(default_factory = list, description = "Model ids as this server exposes them")
