# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Custom nodes run the uploaded script's ``run(inputs, params)`` exactly as the Python block runs its code: the same
``execute_tool("python")`` call, per-flow sandbox session, timeout bounds and owner approval (by the script's hash,
checked by the engine before this handler is reached). See ``core/flows/custom_nodes.py`` for the format."""

from __future__ import annotations

import asyncio
import threading

from core.flows import custom_nodes
from core.flows.nodes.automation import _program, parse_python_output, python_session
from core.flows.registry import HANDLERS, FlowNodeError, NodeContext


async def run_custom(ctx: NodeContext) -> dict:
    from core.inference.tools import execute_tool

    source = ctx.params.get(custom_nodes.SCRIPT_PARAM)
    if not isinstance(source, str) or custom_nodes.node_type(source) != ctx.spec["type"]:
        raise FlowNodeError("This custom node's script does not match its type; add it again from the Custom section")
    value = {
        "inputs": {p["id"]: ctx.inputs.get(p["id"]) for p in ctx.spec["inputs"] if not p.get("param")},
        "params": {p["id"]: ctx.params.get(p["id"]) for p in ctx.spec["params"] if p["id"] != "timeout_s"},
    }
    program = _program(source, value, filename = "<custom node>", entry = True)
    timeout = max(1, min(int(ctx.params.get("timeout_s") or 60), 600))
    stop = threading.Event()

    async def cancel() -> None:
        stop.set()

    ctx.on_cancel(cancel)
    output = await asyncio.to_thread(
        execute_tool,
        "python",
        {"code": program},
        cancel_event = stop,
        timeout = timeout,
        session_id = python_session(ctx.flow_id),
    )
    result, log = parse_python_output(str(output))
    try:
        outputs = custom_nodes.check_outputs(ctx.spec, result)
    except ValueError as exc:
        raise FlowNodeError(str(exc) + (f"\n{log[-2000:]}" if log else "")) from None
    outputs["log"] = log[-20000:]
    return outputs


HANDLERS[custom_nodes.HANDLER] = run_custom
