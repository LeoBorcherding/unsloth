# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Custom nodes: a Python script with a ``NODE = {...}`` header and a ``def run(inputs, params)`` becomes a block.

The header is read with ``ast`` and ``ast.literal_eval`` only; nothing in the file runs on upload, parse or save. The
script runs only as a block, through the same sandbox call, approval (by the hash of the script) and limits as the
Python block (``core/flows/nodes/custom.py``). A flow embeds the script in the node (``_script``), so a shared flow
carries it; the node type is ``custom.n<first 16 hex of its sha256>`` and ``_script_id`` names the library entry it
came from, so the editor can tell when the library copy changed."""

from __future__ import annotations

import ast
import base64
import functools
import hashlib
import json
import math
import re
import sqlite3
import textwrap
import threading
import time
import uuid
import zlib
from typing import Any, Optional

from core.flows import schema
from core.flows.specs import CATEGORIES, SPECS_BY_TYPE, P, inp, out, spec

HANDLER = "custom.script"
TYPE_PREFIX = "custom.n"
SCRIPT_PARAM = "_script"
ID_PARAM = "_script_id"
HASH_PARAM = "_script_hash"
MAX_SCRIPT_BYTES = schema.MAX_STRING_PARAM
MAX_PORTS = 16
MAX_LIBRARY = 200
PORT_TYPES = ("text", "number", "bool", "json", "list", "any")
PARAM_TYPES = ("text", "number", "bool", "select")
CATEGORY_IDS = tuple(c["id"] for c in CATEGORIES)
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_TOP_KEYS = {"name", "description", "category", "inputs", "outputs", "params"}
_PORT_KEYS = {"name", "type", "label", "optional"}
_PARAM_KEYS = {"name", "type", "label", "default", "min", "max", "options"}
_RESERVED_OUTPUTS = {"log"}
_RESERVED_PARAMS = {"timeout_s"}


class CustomNodeError(ValueError):
    """A script that cannot be a node, with the line to look at when there is one."""

    def __init__(self, message: str, line: Optional[int] = None):
        super().__init__(f"Line {line}: {message}" if line else message)
        self.reason = message
        self.line = line


def _digest(source: str) -> str:
    # surrogatepass: a document can carry a lone surrogate; it still hashes, and parse_header refuses it.
    return hashlib.sha256(source.encode("utf-8", "surrogatepass")).hexdigest()


def script_hash(source: str) -> str:
    return "sha256:" + _digest(source)


def node_type(source: str) -> str:
    return TYPE_PREFIX + _digest(source)[:16]


def is_custom_type(type_: Any) -> bool:
    return isinstance(type_, str) and type_.startswith(TYPE_PREFIX) and type_ not in SPECS_BY_TYPE


def _literal(value: ast.AST, what: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        raise CustomNodeError(
            f"{what} must be plain data (strings, numbers, True/False, lists and dicts), not code",
            getattr(value, "lineno", None),
        ) from None


def _entries(value: Optional[ast.AST], field: str, line: int) -> list[tuple[dict, int]]:
    """Each entry of a list field with its own line, so an error points at the entry."""
    if value is None:
        return []
    if not isinstance(value, ast.List):
        raise CustomNodeError(f'"{field}" must be a list of {{...}} entries', line)
    if len(value.elts) > MAX_PORTS:
        raise CustomNodeError(f'"{field}" has more than {MAX_PORTS} entries', line)
    found = []
    for elt in value.elts:
        entry = _literal(elt, f'Each "{field}" entry')
        if not isinstance(entry, dict):
            raise CustomNodeError(f'Each "{field}" entry must be a {{...}} dict', elt.lineno)
        found.append((entry, elt.lineno))
    return found


def _check_keys(entry: dict, allowed: set, what: str, line: int) -> None:
    unknown = sorted(str(k) for k in entry if k not in allowed)
    if unknown:
        raise CustomNodeError(
            f"{what} has unknown key {unknown[0]!r}; allowed: {', '.join(sorted(allowed))}", line
        )


def _label(entry: dict, name: str, line: int) -> str:
    label = entry.get("label", name.replace("_", " ").capitalize())
    if not isinstance(label, str) or not label.strip() or len(label) > 60:
        raise CustomNodeError(f'"label" of {name!r} must be text of 1 to 60 characters', line)
    return label.strip()


def _name(entry: dict, field: str, seen: set, reserved: set, line: int) -> str:
    name = entry.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        raise CustomNodeError(
            f'Every "{field}" entry needs a "name": lowercase letters, digits and _, starting with a letter',
            line,
        )
    if name in seen:
        raise CustomNodeError(f'"{field}" has {name!r} twice', line)
    if name in reserved:
        raise CustomNodeError(f'{name!r} is reserved in "{field}"; pick another name', line)
    seen.add(name)
    return name


def _port(entry: dict, field: str, seen: set, reserved: set, line: int) -> dict:
    _check_keys(entry, _PORT_KEYS, f'A "{field}" entry', line)
    name = _name(entry, field, seen, reserved, line)
    kind = entry.get("type")
    if kind not in PORT_TYPES:
        raise CustomNodeError(
            f"{name!r} has type {kind!r}; custom nodes take {', '.join(PORT_TYPES)} "
            "(media and models stay outside the sandbox)",
            line,
        )
    optional = entry.get("optional", False)
    if not isinstance(optional, bool):
        raise CustomNodeError(f'"optional" of {name!r} must be True or False', line)
    label = _label(entry, name, line)
    if field == "inputs":
        return inp(name, kind, label, optional = optional)
    if "optional" in entry:
        raise CustomNodeError(f'Outputs have no "optional"; remove it from {name!r}', line)
    return out(name, kind, label)


def _is_number(value: Any) -> bool:
    # 1e309 literal_evals to inf, which no JSON response can carry; ints past 2**53 lose precision in the browser.
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) <= 2**53
    return isinstance(value, float) and math.isfinite(value)


def _param(entry: dict, seen: set, line: int) -> dict:
    _check_keys(entry, _PARAM_KEYS, 'A "params" entry', line)
    name = _name(entry, "params", seen, _RESERVED_PARAMS, line)
    kind = entry.get("type")
    if kind not in PARAM_TYPES:
        raise CustomNodeError(
            f"{name!r} has type {kind!r}; params are {', '.join(PARAM_TYPES)}", line
        )
    label = _label(entry, name, line)
    extra: dict[str, Any] = {}
    if kind != "number" and ("min" in entry or "max" in entry):
        raise CustomNodeError(f'Only number params take "min" and "max" ({name!r})', line)
    if kind != "select" and "options" in entry:
        raise CustomNodeError(f'Only select params take "options" ({name!r})', line)
    if kind == "text":
        default = entry.get("default", "")
        if not isinstance(default, str):
            raise CustomNodeError(f'The default of text param {name!r} must be a string', line)
    elif kind == "bool":
        default = entry.get("default", False)
        if not isinstance(default, bool):
            raise CustomNodeError(f'The default of bool param {name!r} must be True or False', line)
    elif kind == "number":
        default = entry.get("default", 0)
        for key in ("default", "min", "max"):
            if key in entry and not _is_number(entry[key]):
                raise CustomNodeError(f'"{key}" of number param {name!r} must be a finite number', line)
        for key in ("min", "max"):
            if key in entry:
                extra[key] = entry[key]
        if "min" in extra and "max" in extra and extra["min"] > extra["max"]:
            raise CustomNodeError(f'"min" of {name!r} is above its "max"', line)
        if ("min" in extra and default < extra["min"]) or ("max" in extra and default > extra["max"]):
            raise CustomNodeError(f"The default of {name!r} is outside its min and max", line)
    else:
        options = entry.get("options")
        if (
            not isinstance(options, list)
            or not options
            or len(options) > 50
            or not all(isinstance(o, str) and o for o in options)
            or len(set(options)) != len(options)
        ):
            raise CustomNodeError(
                f'Select param {name!r} needs "options": a list of 1 to 50 different strings', line
            )
        default = entry.get("default", options[0])
        if default not in options:
            raise CustomNodeError(f"The default of {name!r} is not one of its options", line)
        extra["options"] = options
    return P(name, kind, default, label, **extra)


def _header(tree: ast.Module) -> tuple[ast.Dict, int]:
    found = []
    for stmt in tree.body:
        targets = []
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets = [stmt.target]
        if any(isinstance(t, ast.Name) and t.id == "NODE" for t in targets):
            found.append(stmt)
    if not found:
        raise CustomNodeError("No NODE = {...} header at the top level of the script")
    if len(found) > 1:
        raise CustomNodeError("NODE is assigned more than once; keep one header", found[1].lineno)
    value = found[0].value
    if not isinstance(value, ast.Dict):
        raise CustomNodeError("NODE must be a {...} dict literal", found[0].lineno)
    return value, found[0].lineno


def _check_run(tree: ast.Module) -> None:
    runs = [
        s
        for s in tree.body
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == "run"
    ]
    if not runs:
        raise CustomNodeError("No top-level def run(inputs, params) function")
    fn = runs[-1]
    if isinstance(fn, ast.AsyncFunctionDef):
        raise CustomNodeError("run must be a plain def, not async def", fn.lineno)
    args = fn.args
    positional = [*args.posonlyargs, *args.args]
    if (
        len(positional) != 2
        or args.vararg
        or args.kwarg
        or args.kwonlyargs
        or args.defaults
    ):
        raise CustomNodeError("run must take exactly two arguments: def run(inputs, params)", fn.lineno)


def parse_header(source: Any) -> dict:
    """The validated header of ``source`` as {name, description, category, inputs, outputs, params} (spec shapes)."""
    if not isinstance(source, str) or not source.strip():
        raise CustomNodeError("The script is empty")
    try:
        size = len(source.encode("utf-8"))
    except UnicodeEncodeError:
        raise CustomNodeError("The script is not valid UTF-8 text") from None
    if size > MAX_SCRIPT_BYTES:
        raise CustomNodeError(f"The script is over {MAX_SCRIPT_BYTES // 1000} KB")
    if "\x00" in source:
        raise CustomNodeError("The script contains a NUL byte")
    try:
        tree = ast.parse(source, filename = "<custom node>")
    except SyntaxError as exc:
        raise CustomNodeError(f"Python syntax error: {exc.msg}", exc.lineno) from None
    except (ValueError, MemoryError, RecursionError):
        raise CustomNodeError("The script could not be parsed (too deeply nested)") from None
    value, line = _header(tree)
    fields: dict[str, tuple[ast.AST, int]] = {}
    for key, item in zip(value.keys, value.values):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise CustomNodeError("NODE keys must be strings", getattr(key, "lineno", line))
        if key.value not in _TOP_KEYS:
            raise CustomNodeError(
                f"NODE has unknown key {key.value!r}; allowed: {', '.join(sorted(_TOP_KEYS))}", key.lineno
            )
        fields[key.value] = (item, key.lineno)

    def text(field: str, limit: int, default: Optional[str] = None) -> str:
        if field not in fields:
            if default is None:
                raise CustomNodeError(f'NODE needs a "{field}"', line)
            return default
        node, at = fields[field]
        found = _literal(node, f'"{field}"')
        if not isinstance(found, str) or not found.strip() or len(found) > limit:
            raise CustomNodeError(f'"{field}" must be text of 1 to {limit} characters', at)
        return found.strip()

    name = text("name", 60)
    description = text("description", 300, "")
    category = text("category", 32, "custom")
    if category not in CATEGORY_IDS:
        raise CustomNodeError(
            f"category {category!r} is not a Flows category; use one of {', '.join(CATEGORY_IDS)}",
            fields["category"][1],
        )
    def entries(field: str) -> list[tuple[dict, int]]:
        return _entries(fields[field][0], field, fields[field][1]) if field in fields else []

    found_in, found_out, found_params = entries("inputs"), entries("outputs"), entries("params")
    if not found_out:
        raise CustomNodeError('NODE needs at least one entry in "outputs"', line)
    seen: set = set()
    inputs = [_port(e, "inputs", seen, set(), at) for e, at in found_in]
    seen = set()
    outputs = [_port(e, "outputs", seen, _RESERVED_OUTPUTS, at) for e, at in found_out]
    seen = set()
    params = [_param(e, seen, at) for e, at in found_params]
    _check_run(tree)
    return {
        "name": name,
        "description": description,
        "category": category,
        "inputs": inputs,
        "outputs": outputs,
        "params": params,
    }


@functools.lru_cache(maxsize = 256)
def _spec_for_source(source: str) -> str:
    header = parse_header(source)
    found = spec(
        node_type(source),
        header["name"],
        header["category"],
        header["description"] or "A custom node from your Python script.",
        inputs = header["inputs"],
        outputs = [*header["outputs"], out("log", "text")],
        params = [
            *header["params"],
            P("timeout_s", "int", 60, "Timeout (seconds)", min = 1, max = 600, advanced = True),
        ],
        side_effect = True,
        requires_approval = ["python"],
        icon = "braces",
    )
    found["handler"] = HANDLER
    found["code_param"] = SCRIPT_PARAM
    found["custom"] = {"hash": script_hash(source)}
    # Cached as JSON so no caller can mutate the shared copy.
    return json.dumps(found)


def build_spec(source: str) -> dict:
    """The node spec for ``source``; raises CustomNodeError."""
    return json.loads(_spec_for_source(source))


def spec_or_error(node: dict) -> tuple[Optional[dict], Optional[str]]:
    """(spec, None) for a built-in or a valid custom node; (None, why) otherwise."""
    type_ = node.get("type")
    builtin = SPECS_BY_TYPE.get(type_)
    if builtin is not None:
        return builtin, None
    if not is_custom_type(type_):
        return None, f"Unknown node type {type_}"
    source = (node.get("params") or {}).get(SCRIPT_PARAM)
    if not isinstance(source, str):
        return None, "This custom node has no script; add it again from the Custom section"
    if node_type(source) != type_:
        return None, "This custom node's script does not match its type; add it again from the Custom section"
    try:
        return build_spec(source), None
    except CustomNodeError as exc:
        return None, f"Custom node script: {exc}"


def spec_for_node(node: Any) -> Optional[dict]:
    if not isinstance(node, dict):
        return None
    return spec_or_error(node)[0]


def check_outputs(found_spec: dict, result: Any) -> dict:
    """``run()``'s dict mapped to the declared outputs, each checked against its port type."""
    if not isinstance(result, dict):
        raise ValueError(f"run() must return a dict of outputs, not {type(result).__name__}")
    outputs: dict[str, Any] = {}
    for port in found_spec["outputs"]:
        pid, kind = port["id"], port["type"]
        if pid == "log":
            continue
        if pid not in result:
            raise ValueError(f"run() returned no {pid!r} ({kind})")
        value = result[pid]
        ok = {
            "text": isinstance(value, str),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "bool": isinstance(value, bool),
            "list": isinstance(value, list),
        }.get(kind, True)
        if not ok:
            raise ValueError(f"run() returned {type(value).__name__} for {pid!r}, which is a {kind} output")
        outputs[pid] = value
    return outputs


# --- The library: per account, in the account's studio.db beside its flows. -------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS flow_custom_nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    hash TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
)
"""


def _connect() -> sqlite3.Connection:
    from storage.studio_db import get_connection

    conn = get_connection()
    conn.execute(_SCHEMA)
    return conn


def _entry(row: sqlite3.Row) -> dict:
    entry = {
        "id": row["id"],
        "name": row["name"],
        "source": row["source"],
        "hash": row["hash"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }
    try:
        entry["spec"], entry["error"] = build_spec(row["source"]), None
    except CustomNodeError as exc:
        entry["spec"], entry["error"] = None, str(exc)
    return entry


def list_nodes() -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM flow_custom_nodes ORDER BY name COLLATE NOCASE, id").fetchall()
        return [_entry(r) for r in rows]
    finally:
        conn.close()


def get_node(node_id: str) -> Optional[dict]:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM flow_custom_nodes WHERE id = ?", (node_id,)).fetchone()
        return _entry(row) if row else None
    finally:
        conn.close()


def save_node(source: str, node_id: Optional[str] = None) -> dict:
    """Add (no id) or replace a library entry; the script is parsed first and never run. KeyError: no such id."""
    header = parse_header(source)
    build_spec(source)
    now = int(time.time() * 1000)
    conn = _connect()
    try:
        if node_id is None:
            count = conn.execute("SELECT COUNT(*) FROM flow_custom_nodes").fetchone()[0]
            if count >= MAX_LIBRARY:
                raise CustomNodeError(f"The library holds {MAX_LIBRARY} custom nodes; delete one first")
            node_id = "cn_" + uuid.uuid4().hex[:12]
            conn.execute(
                "INSERT INTO flow_custom_nodes (id, name, source, hash, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (node_id, header["name"], source, script_hash(source), now, now),
            )
        else:
            changed = conn.execute(
                "UPDATE flow_custom_nodes SET name = ?, source = ?, hash = ?, updated_at = ? WHERE id = ?",
                (header["name"], source, script_hash(source), now, node_id),
            ).rowcount
            if not changed:
                raise KeyError(node_id)
        conn.commit()
    finally:
        conn.close()
    return get_node(node_id)


def delete_node(node_id: str) -> bool:
    conn = _connect()
    try:
        gone = conn.execute("DELETE FROM flow_custom_nodes WHERE id = ?", (node_id,)).rowcount
        conn.commit()
        return bool(gone)
    finally:
        conn.close()


# --- What the sandbox can import. ---------------------------------------------------------------------------------

_LIBS_MARKER = "<<FLOW_LIBRARIES>>"
# Compressed: the tool cuts output past its result limit (16k chars by default), and a Studio env has ~800 packages.
_LIBS_PROBE = (
    "import base64, importlib.metadata as m, json, sys, zlib\n"
    "found = {}\n"
    "for d in m.distributions():\n"
    "    name = (d.metadata.get('Name') or '').strip()\n"
    "    if name:\n"
    "        found[name.lower()] = (name, d.version)\n"
    "body = json.dumps({'python': sys.version.split()[0], 'packages': [list(v) for _, v in sorted(found.items())]})\n"
    f"print({_LIBS_MARKER!r} + base64.b64encode(zlib.compress(body.encode(), 9)).decode())\n"
)
_libs_lock = threading.Lock()
_libs_cache: Optional[dict] = None


def _parse_probe(output: str) -> Optional[dict]:
    _, marker, tail = str(output).partition(_LIBS_MARKER)
    if not marker:
        return None
    try:
        found = json.loads(zlib.decompress(base64.b64decode(tail.split("\n", 1)[0].strip())))
    except (ValueError, zlib.error):
        return None
    if not isinstance(found, dict) or not isinstance(found.get("packages"), list):
        return None
    packages = [
        {"name": str(p[0]), "version": str(p[1])}
        for p in found["packages"]
        if isinstance(p, list) and len(p) == 2
    ]
    return {"python": str(found.get("python") or ""), "packages": packages}


def sandbox_libraries(refresh: bool = False) -> dict:
    """The distributions importable inside the Python sandbox, asked of the sandbox itself once per process.

    Falls back to this process's interpreter (the one the sandbox launches) when the probe cannot run, and says so
    in ``source``."""
    global _libs_cache
    with _libs_lock:
        if _libs_cache is not None and not refresh:
            return _libs_cache
        found, source, note = None, "sandbox", ""
        try:
            from core.inference.tools import execute_tool

            output = execute_tool(
                "python", {"code": _LIBS_PROBE}, timeout = 60, session_id = "flow-custom-node-libraries"
            )
            found = _parse_probe(str(output))
            if found is None:
                note = str(output).strip()[:300]
        except Exception as exc:  # noqa: BLE001 - the list falls back, the dialog stays usable
            note = str(exc)[:300]
        if found is None:
            import importlib.metadata as metadata
            import sys

            names: dict[str, tuple[str, str]] = {}
            for dist in metadata.distributions():
                name = (dist.metadata.get("Name") or "").strip()
                if name:
                    names[name.lower()] = (name, dist.version)
            found = {
                "python": sys.version.split()[0],
                "packages": [{"name": n, "version": v} for _, (n, v) in sorted(names.items())],
            }
            source = "studio-python"
        result = {**found, "source": source, "note": note}
        if source == "sandbox":
            _libs_cache = result
        return result


# --- The starter script. -------------------------------------------------------------------------------------------

TEMPLATE = '''\
# A custom node for Unsloth Studio Flows.
#
# Upload this file in Flows > Nodes > Custom > Upload script. Studio reads the NODE header
# without running anything; the file only runs when the node runs, inside the same sandbox
# as the Python block, after you approve it for the flow.
#
# The sandbox runs Python {python} with the standard library plus these packages
# (read from the sandbox when this template was downloaded):
#
{libraries}

NODE = {{
    # Shown on the card and in the palette.
    "name": "Repeat text",
    "description": "Repeats the input text and reports how long the result is.",
    # Optional: which Flows category colours the card (text, logic, transform, data, ...).
    "category": "text",
    # Ports you can wire. Types: text, number, bool, json, list, any.
    "inputs": [
        {{"name": "text", "type": "text", "label": "Text"}},
    ],
    "outputs": [
        {{"name": "text", "type": "text"}},
        {{"name": "length", "type": "number"}},
    ],
    # Settings in the inspector. Types: text, number (min/max), bool, select (options).
    "params": [
        {{"name": "times", "type": "number", "default": 2, "min": 1, "max": 20}},
        {{"name": "separator", "type": "text", "default": " "}},
        {{"name": "shout", "type": "bool", "default": False}},
        {{"name": "style", "type": "select", "options": ["plain", "numbered"], "default": "plain"}},
    ],
}}


def run(inputs, params):
    """inputs: {{input name: value}}, params: {{param name: value}}. Return {{output name: value}}."""
    text = inputs["text"] or ""
    if params["shout"]:
        text = text.upper()
    parts = [text] * int(params["times"])
    if params["style"] == "numbered":
        parts = [f"{{i + 1}}. {{p}}" for i, p in enumerate(parts)]
    joined = params["separator"].join(parts)
    print("repeated", len(parts), "times")  # print() shows in the node's log output
    return {{"text": joined, "length": len(joined)}}
'''


def template_source(libraries: Optional[dict] = None) -> str:
    libraries = libraries or {"python": "3", "packages": []}
    names = ", ".join(f"{p['name']} {p['version']}" for p in libraries.get("packages") or [])
    wrapped = textwrap.wrap(names or "(the list was not available)", 96) or [""]
    return TEMPLATE.format(
        python = libraries.get("python") or "3",
        libraries = "\n".join(f"#   {line}" for line in wrapped),
    )
