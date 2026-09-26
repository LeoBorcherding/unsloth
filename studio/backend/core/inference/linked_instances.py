# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Serve models from linked Unsloth Studio instances through this server's /v1 API.

A linked model is addressed as ``@<instance>/<remote model id>``. HF org names cannot start
with ``@``, so the prefix never shadows a local id. Requests are forwarded verbatim with the
model id unwrapped, and the response (JSON or SSE) is passed back untouched.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from storage import linked_instances_db

MODEL_PREFIX = "@"
# Set on forwarded requests so a remote never forwards again (A links B links A).
HOP_HEADER = "X-Unsloth-Linked-Hop"
_FORWARDED_HEADERS = ("anthropic-version", "anthropic-beta")
_CATALOG_TTL_S = 10.0
_PROBE_TIMEOUT = httpx.Timeout(8.0, connect = 5.0)

_catalog_cache: dict[str, tuple[float, list[dict]]] = {}
_http_client: Optional[httpx.AsyncClient] = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        from core.inference.external_provider import _create_shared_http_client

        _http_client = _create_shared_http_client()
    return _http_client


def normalize_base_url(base_url: str) -> str:
    """Validated origin without a trailing ``/`` or ``/v1``, so either form can be pasted."""
    from core.inference.providers import validate_provider_base_url

    url = validate_provider_base_url(base_url)
    if url.lower().endswith("/v1"):
        url = url[:-3].rstrip("/")
    return url


def split_model(model: object) -> Optional[tuple[str, str]]:
    if not isinstance(model, str) or not model.startswith(MODEL_PREFIX):
        return None
    name, _, remote = model[len(MODEL_PREFIX):].partition("/")
    if not name or not remote:
        return None
    return name.lower(), remote


def _auth_headers(instance: dict) -> dict[str, str]:
    key = linked_instances_db.get_api_key(instance["id"])
    headers = {HOP_HEADER: "1"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _may_use_linked(request: Request) -> bool:
    from auth.authentication import request_admitted_without_credential
    from hub.services.models import account_access

    if request.headers.get(HOP_HEADER):
        return False
    if account_access.managed_account():
        return False
    return not request_admitted_without_credential(request)


async def resolve(request: Request, model: object) -> Optional[tuple[dict, str]]:
    """``(instance, remote model id)`` when ``model`` names a linked instance, else ``None``."""
    parts = split_model(model)
    if parts is None:
        return None
    name, remote_model = parts
    if not _may_use_linked(request):
        raise HTTPException(
            status_code = 403,
            detail = "Linked instances can only be used by the installation owner with an API key or UI session.",
        )
    instance = await asyncio.to_thread(linked_instances_db.get_instance_by_name, name)
    if instance is None:
        raise HTTPException(status_code = 404, detail = f"No linked instance named '{name}'.")
    return instance, remote_model


async def forward(request: Request, path: str, target: tuple[dict, str]) -> Response:
    """POST the caller's JSON body to ``<instance>/v1/<path>`` with the model id unwrapped."""
    instance, remote_model = target
    body = await request.json()
    body["model"] = remote_model
    stream = bool(body.get("stream"))
    headers = await asyncio.to_thread(_auth_headers, instance)
    for name in _FORWARDED_HEADERS:
        if value := request.headers.get(name):
            headers[name] = value

    client = _client()
    upstream_request = client.build_request(
        "POST",
        f"{instance['base_url']}/v1/{path}",
        json = body,
        headers = headers,
        # A generation can sit silent for minutes before its first token.
        timeout = httpx.Timeout(None if stream else 900.0, connect = 10.0),
    )
    try:
        upstream = await client.send(upstream_request, stream = True)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code = 502,
            detail = f"Linked instance '{instance['name']}' is unreachable ({type(exc).__name__}).",
        ) from exc

    media_type = upstream.headers.get("content-type", "application/json")
    if not stream or upstream.status_code >= 400:
        try:
            content = await upstream.aread()
        finally:
            await upstream.aclose()
        return Response(content, status_code = upstream.status_code, media_type = media_type)

    async def relay():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        relay(),
        status_code = upstream.status_code,
        media_type = media_type,
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def fetch_models(instance: dict) -> list[dict]:
    headers = await asyncio.to_thread(_auth_headers, instance)
    response = await _client().get(
        f"{instance['base_url']}/v1/models", headers = headers, timeout = _PROBE_TIMEOUT
    )
    response.raise_for_status()
    data = response.json().get("data")
    if not isinstance(data, list):
        raise ValueError("Unexpected /v1/models response")
    return [m for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)]


async def probe(instance: dict) -> dict:
    try:
        models = await fetch_models(instance)
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        error = "The API key was rejected." if code in (401, 403) else f"HTTP {code}"
        return {"online": False, "error": error, "models": []}
    except (httpx.HTTPError, ValueError) as exc:
        return {"online": False, "error": type(exc).__name__, "models": []}
    return {
        "online": True,
        "error": None,
        # A remote's own linked models are left out: no chains, no loops.
        "models": [m for m in models if not m["id"].startswith(MODEL_PREFIX)],
    }


async def _instance_catalog(instance: dict) -> list[dict]:
    cached = _catalog_cache.get(instance["id"])
    if cached and time.monotonic() - cached[0] < _CATALOG_TTL_S:
        return cached[1]
    result = await probe(instance)
    objects = [
        {
            **model,
            "id": f"{MODEL_PREFIX}{instance['name']}/{model['id']}",
            "owned_by": instance["name"],
            "linked_instance": instance["name"],
        }
        for model in result["models"]
    ]
    _catalog_cache[instance["id"]] = (time.monotonic(), objects)
    return objects


async def catalog_objects(request: Optional[Request]) -> list[dict]:
    """Every linked instance's models for /v1/models; an offline instance contributes none."""
    if request is None or not _may_use_linked(request):
        return []
    instances = await asyncio.to_thread(linked_instances_db.list_instances)
    results = await asyncio.gather(*(_instance_catalog(i) for i in instances))
    return [model for models in results for model in models]


def forget(instance_id: str) -> None:
    _catalog_cache.pop(instance_id, None)
