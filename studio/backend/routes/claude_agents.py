# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Claude Agent script routes.

Wraps the Claude Code CLI (``claude -p``) so Studio can run saved
prompt-script agents (prompt templates with ``{placeholder}`` variables bound
to file paths or text) and a live chat session. Claude Code's ``stream-json``
events are passed through to the client over SSE.

Agent definitions are plain JSON files on disk so they are easy to share and
version:

    <studio_root>/claude-agents/scripts/<id>.json             standalone agents
    <studio_root>/claude-agents/workflows/<group>/<id>.json   agents that chain together

Powered by Claude Code (Anthropic). Requires the ``claude`` CLI on PATH.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import shutil
import subprocess
import threading
import time
import uuid

from collections import deque
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from utils.paths import ensure_dir, studio_root

logger = logging.getLogger(__name__)

router = APIRouter()

PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
# Permission rule syntax, e.g. "Read,Glob,Bash(git diff *)". Conservative
# whitelist; these values end up on the claude CLI command line.
_SAFE_TOOLS_RE = re.compile(r"^[A-Za-z0-9_,()*:./ -]+$")
_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_PERMISSION_MODES = {"default", "plan", "acceptEdits", "dontAsk", "bypassPermissions"}

_STDERR_TAIL_LINES = 40


# ============ Storage ============


def claude_agents_root() -> Path:
    return studio_root() / "claude-agents"


def scripts_root() -> Path:
    return claude_agents_root() / "scripts"


def workflows_root() -> Path:
    return claude_agents_root() / "workflows"


class AgentScript(BaseModel):
    id: str = Field(max_length = 128)
    name: str = Field(max_length = 200)
    description: str = Field(default = "", max_length = 2000)
    promptTemplate: str = Field(max_length = 100_000)
    allowedTools: str = Field(default = "Read,Glob,Grep", max_length = 1000)
    permissionMode: str = Field(default = "default", max_length = 32)
    cwd: str = Field(default = "", max_length = 1000)
    # Optional grouping folder for agents designed to chain together.
    workflow: str | None = Field(default = None, max_length = 128)
    prebuilt: bool = False
    createdAt: int = 0
    updatedAt: int = 0


def _agent_dir(workflow: str | None) -> Path:
    if workflow:
        return workflows_root() / workflow
    return scripts_root()


def _agent_path(agent_id: str, workflow: str | None) -> Path:
    return _agent_dir(workflow) / f"{agent_id}.json"


def _validate_agent(agent: AgentScript) -> None:
    if not _SAFE_ID_RE.match(agent.id):
        raise HTTPException(status_code = 400, detail = "Invalid agent id")
    if agent.workflow is not None and not _SAFE_ID_RE.match(agent.workflow):
        raise HTTPException(status_code = 400, detail = "Invalid workflow name")
    if agent.permissionMode not in _PERMISSION_MODES:
        raise HTTPException(status_code = 400, detail = "Invalid permission mode")
    if agent.allowedTools and not _SAFE_TOOLS_RE.match(agent.allowedTools):
        raise HTTPException(status_code = 400, detail = "Invalid characters in allowed tools")


def _load_agents() -> list[AgentScript]:
    agents: list[AgentScript] = []
    roots = [scripts_root()]
    if workflows_root().is_dir():
        roots.extend(p for p in sorted(workflows_root().iterdir()) if p.is_dir())
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding = "utf-8"))
                agents.append(AgentScript(**data))
            except Exception:  # noqa: BLE001 - skip corrupt files, keep the rest
                logger.warning("Skipping unreadable agent file: %s", path)
    return agents


def _save_agent(agent: AgentScript) -> None:
    path = _agent_path(agent.id, agent.workflow)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(agent.model_dump(), indent = 2, ensure_ascii = False),
        encoding = "utf-8",
    )


def _find_agent(agent_id: str) -> AgentScript | None:
    for agent in _load_agents():
        if agent.id == agent_id:
            return agent
    return None


# ============ Prebuilt agents ============

# Mirrors the standalone claude-scripts pr-agent: review a PR with read-only
# tools and post findings as a deletable issue comment via the gh CLI.
_PREBUILT_AGENTS = [
    {
        "id": "review-pr",
        "name": "PR Review",
        "description": "Review a GitHub PR against a local checkout and post findings as a gh CLI comment.",
        "promptTemplate": (
            "You are a code reviewer. The local codebase for this project is at: {local_path}\n\n"
            "Please review the following GitHub PR: {pr_link}\n\n"
            "Steps:\n"
            "1. Use the gh CLI to fetch the PR diff and description.\n"
            "2. Read the relevant files from the local codebase at {local_path} to understand context.\n"
            "3. Post your findings as a regular issue comment using ONLY this exact gh CLI command: "
            "`gh pr comment <PR_URL> --body \"...\"` - do NOT use `gh pr review`, do NOT submit a PR review, "
            "do NOT use the reviews API. A regular comment can be deleted; a PR review cannot. "
            "Cover: correctness issues, edge cases, potential bugs, code style/consistency with the "
            "existing codebase, and any blocking concerns vs nits. Be specific - reference file paths "
            "and line numbers. Group findings by severity (blocking / nit). Keep the tone constructive."
        ),
        "allowedTools": "Bash,Read,Glob,Grep,WebFetch",
        "permissionMode": "default",
        "prebuilt": True,
    },
    {
        "id": "doc-for-file",
        "name": "Generate Doc for File",
        "description": "Write documentation for a single file that fits the codebase's existing doc style.",
        "promptTemplate": (
            "Generate documentation for this file: {file}\n\n"
            "First read the file, then look at how the surrounding codebase documents similar "
            "modules (READMEs, docstrings, doc folders) and match that style. Write the doc to a "
            "sensible location next to the existing docs and report the path you wrote it to."
        ),
        "allowedTools": "Read,Glob,Grep,Write,Edit",
        "permissionMode": "acceptEdits",
        "prebuilt": True,
    },
]


def _seed_prebuilt_agents() -> None:
    ensure_dir(scripts_root())
    now = int(time.time() * 1000)
    for spec in _PREBUILT_AGENTS:
        path = _agent_path(spec["id"], None)
        if path.exists():
            continue
        agent = AgentScript(**spec, createdAt = now, updatedAt = now)
        _save_agent(agent)


# ============ Claude CLI invocation ============


def _claude_available() -> str | None:
    """Return the resolved claude executable path, or None if not installed."""
    return shutil.which("claude")


def _quote_windows_arg(arg: str) -> str:
    # Args are pre-validated against whitelists that exclude double quotes,
    # so plain double-quoting is sufficient for cmd.exe.
    if re.match(r"^[A-Za-z0-9_,.:/=-]+$", arg):
        return arg
    return f'"{arg}"'


def _build_command(flags: list[str]) -> tuple[str | list[str], bool]:
    """Build the claude invocation. Returns (command, use_shell).

    The prompt is always passed on stdin (`-p -`) to avoid quoting issues.
    On Windows the npm `claude.cmd` shim needs a shell to resolve.
    """
    if platform.system() == "Windows":
        flag_str = " ".join(_quote_windows_arg(f) for f in flags)
        return f"claude -p - {flag_str}".strip(), True
    return ["claude", "-p", "-", *flags], False


# Registry of running processes so the client can cancel a run.
_RUNS_LOCK = threading.Lock()
_RUNNING: dict[str, subprocess.Popen] = {}


def _register_run(run_id: str, proc: subprocess.Popen) -> None:
    with _RUNS_LOCK:
        _RUNNING[run_id] = proc


def _unregister_run(run_id: str) -> None:
    with _RUNS_LOCK:
        _RUNNING.pop(run_id, None)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii = False)}\n\n"


def _stream_claude(run_id: str, prompt: str, flags: list[str], cwd: str | None):
    yield _sse({"type": "studio", "subtype": "run_started", "runId": run_id})

    command, use_shell = _build_command(flags)
    try:
        proc = subprocess.Popen(
            command,
            shell = use_shell,
            cwd = cwd or None,
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            encoding = "utf-8",
            errors = "replace",
        )
    except OSError as exc:
        yield _sse({"type": "studio", "subtype": "run_error", "runId": run_id, "error": str(exc)})
        return

    _register_run(run_id, proc)
    stderr_tail: deque[str] = deque(maxlen = _STDERR_TAIL_LINES)

    def _feed_stdin() -> None:
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except OSError:
            pass

    def _drain_stderr() -> None:
        try:
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))
        except (OSError, ValueError):
            pass

    threading.Thread(target = _feed_stdin, daemon = True).start()
    stderr_thread = threading.Thread(target = _drain_stderr, daemon = True)
    stderr_thread.start()

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            # stream-json lines are already JSON objects; pass them through.
            yield f"data: {line}\n\n"
        proc.wait()
        stderr_thread.join(timeout = 2)
        exit_code = proc.returncode
        payload = {"type": "studio", "subtype": "run_finished", "runId": run_id, "exitCode": exit_code}
        if exit_code not in (0, None) and stderr_tail:
            payload["stderr"] = "\n".join(stderr_tail)
        if exit_code == 127:
            payload["error"] = "'claude' not found. Install Claude Code and ensure it is on PATH."
        yield _sse(payload)
    except GeneratorExit:
        # Client disconnected; stop the CLI run.
        proc.kill()
        raise
    finally:
        _unregister_run(run_id)


# ============ Request models ============


class RunAgentRequest(BaseModel):
    # Run either a saved agent by id, or an ad-hoc prompt template.
    agentId: str | None = Field(default = None, max_length = 128)
    promptTemplate: str | None = Field(default = None, max_length = 100_000)
    # Placeholder bindings, e.g. {"file1": "C:/repo/src/main.py"}.
    variables: dict[str, str] = Field(default_factory = dict)
    allowedTools: str | None = Field(default = None, max_length = 1000)
    permissionMode: str | None = Field(default = None, max_length = 32)
    cwd: str | None = Field(default = None, max_length = 1000)
    # Live chat: resume a previous session and let Claude see the saved scripts.
    sessionId: str | None = Field(default = None, max_length = 64)
    includeAgentCatalog: bool = False


def _render_prompt(template: str, variables: dict[str, str]) -> str:
    def _sub(match: re.Match) -> str:
        return variables.get(match.group(1), match.group(0))

    return PLACEHOLDER_RE.sub(_sub, template)


def _agent_catalog_preamble() -> str:
    agents = _load_agents()
    if not agents:
        return ""
    lines = [
        "You are the Claude Agent chat inside Unsloth Studio. The user has these "
        "saved agent scripts (prompt templates with {placeholder} variables):",
        "",
    ]
    for agent in agents:
        lines.append(f"### {agent.name} (id: {agent.id})")
        if agent.description:
            lines.append(agent.description)
        lines.append("```")
        lines.append(agent.promptTemplate)
        lines.append("```")
        lines.append("")
    lines.append(
        "When the user asks you to run one of these scripts, substitute the "
        "{placeholder} variables with the values they give you and carry out the "
        "script's instructions yourself."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ============ Routes ============


@router.get("/status")
def claude_status(current_subject: str = Depends(get_current_subject)):
    exe = _claude_available()
    if not exe:
        return {"available": False, "version": None}
    try:
        command, use_shell = (
            ("claude --version", True)
            if platform.system() == "Windows"
            else (["claude", "--version"], False)
        )
        result = subprocess.run(
            command,
            shell = use_shell,
            capture_output = True,
            text = True,
            timeout = 15,
        )
        version = result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        version = None
    return {"available": True, "version": version}


@router.get("/agents")
def list_agents(current_subject: str = Depends(get_current_subject)):
    _seed_prebuilt_agents()
    return {"agents": [a.model_dump() for a in _load_agents()]}


@router.put("/agents/{agent_id}")
def put_agent(
    agent_id: str,
    agent: AgentScript,
    current_subject: str = Depends(get_current_subject),
):
    if agent.id != agent_id:
        raise HTTPException(status_code = 400, detail = "ID mismatch")
    _validate_agent(agent)
    # If the agent moved between folders, drop the old file.
    existing = _find_agent(agent_id)
    if existing is not None and existing.workflow != agent.workflow:
        _agent_path(agent_id, existing.workflow).unlink(missing_ok = True)
    _save_agent(agent)
    return agent.model_dump()


@router.delete("/agents/{agent_id}", status_code = 204)
def remove_agent(agent_id: str, current_subject: str = Depends(get_current_subject)):
    if not _SAFE_ID_RE.match(agent_id):
        raise HTTPException(status_code = 400, detail = "Invalid agent id")
    existing = _find_agent(agent_id)
    if existing is None:
        return
    _agent_path(agent_id, existing.workflow).unlink(missing_ok = True)


@router.post("/run")
def run_agent(
    req: RunAgentRequest,
    current_subject: str = Depends(get_current_subject),
):
    if not _claude_available():
        raise HTTPException(
            status_code = 503,
            detail = "'claude' CLI not found. Install Claude Code and ensure it is on PATH.",
        )

    agent: AgentScript | None = None
    if req.agentId:
        agent = _find_agent(req.agentId)
        if agent is None:
            raise HTTPException(status_code = 404, detail = "Agent not found")

    template = req.promptTemplate if req.promptTemplate is not None else (
        agent.promptTemplate if agent else None
    )
    if not template or not template.strip():
        raise HTTPException(status_code = 400, detail = "No prompt provided")

    prompt = _render_prompt(template, req.variables)
    if req.includeAgentCatalog and not req.sessionId:
        prompt = _agent_catalog_preamble() + prompt

    allowed_tools = req.allowedTools if req.allowedTools is not None else (
        agent.allowedTools if agent else ""
    )
    if allowed_tools and not _SAFE_TOOLS_RE.match(allowed_tools):
        raise HTTPException(status_code = 400, detail = "Invalid characters in allowed tools")

    permission_mode = req.permissionMode or (agent.permissionMode if agent else "default")
    if permission_mode not in _PERMISSION_MODES:
        raise HTTPException(status_code = 400, detail = "Invalid permission mode")

    if req.sessionId and not _SAFE_SESSION_RE.match(req.sessionId):
        raise HTTPException(status_code = 400, detail = "Invalid session id")

    cwd = req.cwd or (agent.cwd if agent else "") or None
    if cwd and not Path(cwd).is_dir():
        raise HTTPException(status_code = 400, detail = f"Working directory not found: {cwd}")

    flags = ["--output-format", "stream-json", "--verbose", "--include-partial-messages"]
    if permission_mode != "default":
        flags += ["--permission-mode", permission_mode]
    if allowed_tools:
        flags += ["--allowedTools", allowed_tools]
    if req.sessionId:
        flags += ["--resume", req.sessionId]

    run_id = uuid.uuid4().hex

    return StreamingResponse(
        _stream_claude(run_id, prompt, flags, cwd),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Run-Id": run_id,
        },
    )


@router.post("/runs/{run_id}/cancel", status_code = 204)
def cancel_run(run_id: str, current_subject: str = Depends(get_current_subject)):
    with _RUNS_LOCK:
        proc = _RUNNING.get(run_id)
    if proc is None:
        raise HTTPException(status_code = 404, detail = "Run not found")
    proc.kill()
