# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load tools and Check on data recipe tool servers, against small local stdio stubs."""

from __future__ import annotations

import http.server
import json
import os
import socket
import sys
import textwrap
import threading

import pytest

pytest.importorskip("data_designer.config.mcp")
pytest.importorskip("mcp.server.fastmcp")
psutil = pytest.importorskip("psutil")

from models.data_recipe import McpToolsListRequest, RecipePayload  # noqa: E402
from routes.data_recipe.mcp import list_mcp_tools  # noqa: E402
from routes.data_recipe.validate import validate  # noqa: E402
from utils import host_policy  # noqa: E402

STUB = textwrap.dedent(
    """
    import json, os, sys, time
    with open(os.environ["STUB_PIDFILE"], "a") as f:
        f.write(f"{os.getpid()}\\n")
    if os.environ.get("STUB_EXIT"):
        sys.exit(int(os.environ["STUB_EXIT"]))
    if os.environ.get("STUB_HANG"):
        time.sleep(600)
    from mcp.server.fastmcp import FastMCP
    server = FastMCP("stub")
    for name in json.load(open(os.environ["STUB_TOOLS"])):
        def tool(a: int = 0) -> int:
            return a
        server.add_tool(tool, name = name)
    server.run("stdio")
    """
)


@pytest.fixture(autouse = True)
def stdio_on(monkeypatch, tmp_path):
    saved = os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP")
    host_policy._reset_loopback_default_state()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    os.environ["UNSLOTH_STUDIO_ALLOW_STDIO_MCP"] = "1"
    yield
    host_policy._reset_loopback_default_state()
    if saved is None:
        os.environ.pop("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", None)
    else:
        os.environ["UNSLOTH_STUDIO_ALLOW_STDIO_MCP"] = saved


class Stub:
    def __init__(self, root):
        self.script = root / "stub_stdio.py"
        self.script.write_text(STUB)
        self.pidfile = root / "stub.pids"
        self.tools = root / "tools.json"
        self.set_tools(["add", "lookup"])

    def set_tools(self, names):
        self.tools.write_text(json.dumps(names))

    def provider(self, name = "calc", **env):
        env = {"STUB_PIDFILE": str(self.pidfile), "STUB_TOOLS": str(self.tools), **env}
        return {
            "provider_type": "stdio",
            "name": name,
            "command": sys.executable,
            "args": ["-I", str(self.script)],
            "env": env,
        }

    def pids(self):
        if not self.pidfile.exists():
            return []
        return [int(line) for line in self.pidfile.read_text().split()]

    def alive(self):
        procs = [psutil.Process(pid) for pid in self.pids() if psutil.pid_exists(pid)]
        _, alive = psutil.wait_procs(procs, timeout = 5)
        return [proc.pid for proc in alive]


@pytest.fixture
def stub(tmp_path):
    stub = Stub(tmp_path)
    yield stub
    for pid in stub.pids():
        try:
            psutil.Process(pid).kill()
        except psutil.Error:
            pass


def _load(*providers, timeout_sec = 20):
    payload = McpToolsListRequest(mcp_providers = list(providers), timeout_sec = timeout_sec)
    return list_mcp_tools(payload).providers


def test_load_tools_leaves_no_stdio_server_running(stub):
    for value in ("a", "b", "c"):
        (result,) = _load(stub.provider(EDIT = value))
        assert result.tools == ["add", "lookup"], result
    assert len(stub.pids()) == 3
    assert stub.alive() == []


def test_load_tools_stops_a_hung_stdio_server(stub):
    (result,) = _load(stub.provider(STUB_HANG = "1"), timeout_sec = 2)
    assert result.error == "Timed out after 2s while listing tools on 'calc'."
    assert len(stub.pids()) == 1
    assert stub.alive() == []


def test_load_tools_sees_a_changed_tool_list(stub):
    stub.set_tools(["add"])
    assert _load(stub.provider())[0].tools == ["add"]
    stub.set_tools(["add", "lookup", "search"])
    assert _load(stub.provider())[0].tools == ["add", "lookup", "search"]


@pytest.fixture
def not_found_server():
    class NotFound(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(404)
            self.end_headers()

        do_POST = do_GET

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), NotFound)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_load_tools_says_why_it_failed(stub, not_found_server):
    port = _free_port()
    missing = os.path.join(str(stub.script.parent), "does-not-exist.exe")
    results = _load(
        {"provider_type": "stdio", "name": "bad-command", "command": missing, "args": [], "env": {}},
        stub.provider("exits", STUB_EXIT = "3"),
        {
            "provider_type": "streamable_http",
            "name": "refused",
            "endpoint": f"http://127.0.0.1:{port}/mcp",
        },
        {
            "provider_type": "streamable_http",
            "name": "wrong-path",
            "endpoint": f"http://127.0.0.1:{not_found_server}/nope",
        },
        timeout_sec = 10,
    )
    assert {r.name: r.error for r in results} == {
        "bad-command": f"Command not found: {missing}",
        "exits": "The server exited before it listed its tools.",
        "refused": f"Nothing is listening at 127.0.0.1:{port}.",
        "wrong-path": "No MCP endpoint at that path. Try /mcp or /sse.",
    }


def _recipe(stub, **overrides):
    recipe = {
        "model_providers": [{"name": "p", "endpoint": "http://127.0.0.1:9/v1", "api_key": "x"}],
        "model_configs": [{"alias": "m", "model": "x", "provider": "p"}],
        "mcp_providers": [stub.provider()],
        "tool_configs": [{"tool_alias": "calc-tools", "providers": ["calc"], "allow_tools": ["add"]}],
        "columns": [
            {
                "name": "answer",
                "column_type": "llm-text",
                "prompt": "Add 2 and 3.",
                "model_alias": "m",
                "tool_alias": "calc-tools",
            }
        ],
    }
    recipe.update(overrides)
    return recipe


def _check(recipe):
    out = validate(RecipePayload(recipe = recipe))
    return out.valid, [error.message for error in out.errors]


def test_check_passes_good_tool_wiring(stub):
    assert _check(_recipe(stub)) == (True, [])
    assert stub.alive() == []


def test_check_refuses_a_stdio_server_while_stdio_is_off(stub, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "0")
    assert _check(_recipe(stub)) == (
        False,
        ["Local (stdio) MCP servers are disabled on this host."],
    )
    assert stub.pids() == []


@pytest.mark.parametrize(
    "case,expected",
    [
        ("unknown alias", "Tool profile 'ghost' doesn't exist."),
        ("unknown provider", "Tool profile 'calc-tools' uses tool server 'nope', which doesn't exist."),
        ("missing allowed tool", "Tool profile 'calc-tools' allows ['divide'], which its servers don't have."),
        ("bad command", "Tool server 'calc': Command not found: "),
        ("refused", "Tool server 'calc': Nothing is listening at 127.0.0.1:"),
    ],
)
def test_check_reports_broken_tool_wiring(stub, case, expected):
    recipe = _recipe(stub)
    if case == "unknown alias":
        recipe["columns"][0]["tool_alias"] = "ghost"
    elif case == "unknown provider":
        recipe["tool_configs"][0]["providers"] = ["nope"]
    elif case == "missing allowed tool":
        recipe["tool_configs"][0]["allow_tools"] = ["add", "divide"]
    elif case == "bad command":
        recipe["mcp_providers"][0]["command"] = str(stub.script.parent / "does-not-exist.exe")
    else:
        recipe["mcp_providers"] = [
            {
                "provider_type": "streamable_http",
                "name": "calc",
                "endpoint": f"http://127.0.0.1:{_free_port()}/mcp",
            }
        ]
    valid, messages = _check(recipe)
    assert valid is False
    assert len(messages) == 1 and messages[0].startswith(expected), messages


def _github_seeded(recipe):
    recipe["seed_config"] = {
        "source": {
            "seed_type": "github_repo",
            "repos": ["octo/repo"],
            "item_types": ["issues"],
            "limit": 5,
            "token": "",
        },
    }
    return recipe


def test_check_reports_broken_tool_wiring_on_a_github_seed(stub):
    pytest.importorskip("data_designer_github_repo_seed")
    recipe = _github_seeded(_recipe(stub))
    recipe["tool_configs"][0]["allow_tools"] = ["add", "divide"]
    assert _check(recipe) == (
        False,
        ["Tool profile 'calc-tools' allows ['divide'], which its servers don't have."],
    )
    assert stub.alive() == []


def test_check_refuses_stdio_on_a_github_seed_while_stdio_is_off(stub, monkeypatch):
    pytest.importorskip("data_designer_github_repo_seed")
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "0")
    assert _check(_github_seeded(_recipe(stub))) == (
        False,
        ["Local (stdio) MCP servers are disabled on this host."],
    )
    assert stub.pids() == []
