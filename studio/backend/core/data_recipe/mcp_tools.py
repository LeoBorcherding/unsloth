# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool listing for data recipe MCP providers (Load tools and Check).

The engine's own list_tools pools a session per provider config until shutdown and caches the list
with no expiry, so a stdio child outlived every Load tools click and a changed server kept its old
tools. Each listing here opens its own session and closes it, which also stops a stdio child on timeout."""

from __future__ import annotations

import asyncio
from typing import Any, Callable
from urllib.parse import urlsplit

CHECK_TIMEOUT_SEC = 15.0


class ToolListError(Exception):
    """A listing failure whose message is safe to show."""


def _is_stdio(provider: Any) -> bool:
    return getattr(provider, "command", None) is not None


def _transport(provider: Any):
    from mcp import StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    if _is_stdio(provider):
        params = StdioServerParameters(
            command = provider.command, args = provider.args, env = provider.env
        )
        return stdio_client(params)
    headers = {"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else None
    if provider.provider_type == "streamable_http":
        return streamablehttp_client(provider.endpoint, headers = headers)
    return sse_client(provider.endpoint, headers = headers)


async def _list(provider: Any) -> list[str]:
    from mcp import ClientSession

    async with _transport(provider) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            result = await session.list_tools()
    return sorted({tool.name for tool in result.tools if getattr(tool, "name", "")})


def list_tool_names(provider: Any, timeout_sec: float | None) -> list[str]:
    try:
        return asyncio.run(asyncio.wait_for(_list(provider), timeout_sec))
    except asyncio.TimeoutError as exc:
        raise ToolListError(
            f"Timed out after {timeout_sec:g}s while listing tools on {provider.name!r}."
        ) from exc


def root_cause(exc: BaseException) -> BaseException:
    """First real error under the task-group exception groups and cancellations the MCP clients raise."""
    while True:
        group = getattr(exc, "exceptions", None)
        if isinstance(group, (list, tuple)) and group:
            leaves = [e for e in group if not isinstance(e, asyncio.CancelledError)]
            exc = (leaves or list(group))[0]
            continue
        if isinstance(exc, asyncio.CancelledError) and exc.__context__ is not None:
            exc = exc.__context__
            continue
        return exc


def describe_error(exc: BaseException, provider: Any) -> str | None:
    """User-facing reason for a known listing failure, else None."""
    import httpx
    from mcp.shared.exceptions import McpError

    cause = root_cause(exc)
    if isinstance(cause, ToolListError):
        return str(cause)
    if _is_stdio(provider):
        if isinstance(cause, FileNotFoundError):
            return f"Command not found: {provider.command}"
        if isinstance(cause, McpError):
            return "The server exited before it listed its tools."
        return None
    if isinstance(cause, (httpx.ConnectError, ConnectionRefusedError)):
        address = urlsplit(str(provider.endpoint)).netloc or str(provider.endpoint)
        return f"Nothing is listening at {address}."
    # streamable_http turns a 404 on initialize into this JSON-RPC error.
    if isinstance(cause, McpError) and getattr(cause.error, "code", None) == 32600:
        return "No MCP endpoint at that path. Try /mcp or /sse."
    if isinstance(cause, httpx.HTTPStatusError):
        if cause.response.status_code == 404:
            return "No MCP endpoint at that path. Try /mcp or /sse."
        return f"The server answered HTTP {cause.response.status_code}."
    return None


def tool_wiring_errors(
    recipe: dict[str, Any], build_providers: Callable[[dict[str, Any]], list[Any]]
) -> list[tuple[str | None, str]]:
    """(column, message) for the tool wiring the run's MCP health check would fail on."""
    configs = [tc for tc in recipe.get("tool_configs") or [] if isinstance(tc, dict)]
    by_alias = {str(tc.get("tool_alias", "")): tc for tc in configs}
    errors: list[tuple[str | None, str]] = []

    used: list[str] = []
    for column in recipe.get("columns") or []:
        alias = column.get("tool_alias") if isinstance(column, dict) else None
        if not alias:
            continue
        if alias not in by_alias:
            errors.append((column.get("name"), f"Tool profile {alias!r} doesn't exist."))
        elif alias not in used:
            used.append(alias)
    if not configs:
        return errors

    by_name = {provider.name: provider for provider in build_providers(recipe)}

    for alias, tc in by_alias.items():
        for name in tc.get("providers") or []:
            if name not in by_name:
                errors.append(
                    (None, f"Tool profile {alias!r} uses tool server {name!r}, which doesn't exist.")
                )

    listed: dict[str, list[str] | str] = {}
    for alias in used:
        tc = by_alias[alias]
        names: set[str] = set()
        reachable = True
        for name in tc.get("providers") or []:
            provider = by_name.get(name)
            if provider is None:
                reachable = False
                continue
            if name not in listed:
                try:
                    listed[name] = list_tool_names(provider, CHECK_TIMEOUT_SEC)
                except Exception as exc:
                    listed[name] = describe_error(exc, provider) or "Failed to load tools."
            result = listed[name]
            if isinstance(result, str):
                errors.append((None, f"Tool server {name!r}: {result}"))
                reachable = False
            else:
                names.update(result)
        missing = [tool for tool in tc.get("allow_tools") or [] if tool not in names]
        if reachable and missing:
            errors.append(
                (None, f"Tool profile {alias!r} allows {missing}, which its servers don't have.")
            )
    return errors
