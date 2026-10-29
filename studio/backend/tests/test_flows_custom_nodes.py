# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import os
import subprocess
import sys
import time

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.flows import capabilities, compiler, custom_nodes
from core.flows import engine as engine_mod
from core.flows.nodes import automation, custom, inputs
from core.flows.registry import FlowNodeError
from routes import flows as flows_routes
from utils.account_context import OWNER, bind_account, reset_account

from ._flows_helpers import doc, edge, node

TEMPLATE = custom_nodes.template_source({"python": "3.12", "packages": [{"name": "numpy", "version": "2.1"}]})

MINIMAL = '''NODE = {
    "name": "Double",
    "inputs": [{"name": "x", "type": "number"}],
    "outputs": [{"name": "y", "type": "number"}],
}


def run(inputs, params):
    return {"y": inputs["x"] * 2}
'''


def _custom(nid: str, source: str = TEMPLATE, **params) -> dict:
    return node(
        nid,
        custom_nodes.node_type(source),
        _script = source,
        _script_id = "cn_test",
        _script_hash = custom_nodes.script_hash(source),
        **params,
    )


def _subprocess_execute_tool(calls: list):
    """Stands in for the sandbox on machines without one: runs the exact program the handler built."""

    def fake(name, args, **kwargs):
        calls.append((name, args, kwargs))
        done = subprocess.run(
            [sys.executable, "-I", "-c", args["code"]], capture_output = True, text = True, timeout = 60
        )
        return done.stdout + done.stderr

    return fake


# --- The header -----------------------------------------------------------------------------------------------------


def test_the_template_is_a_valid_node_and_lists_the_libraries():
    assert "numpy 2.1" in TEMPLATE and "Python 3.12" in TEMPLATE
    spec = custom_nodes.build_spec(TEMPLATE)
    assert spec["type"] == custom_nodes.node_type(TEMPLATE)
    assert spec["type"].startswith("custom.n") and len(spec["type"]) == len("custom.n") + 16
    assert spec["title"] == "Repeat text" and spec["category"] == "text"
    assert [(p["id"], p["type"]) for p in spec["inputs"]] == [("text", "text")]
    assert [(p["id"], p["type"]) for p in spec["outputs"]] == [
        ("text", "text"),
        ("length", "number"),
        ("log", "text"),
    ]
    assert [(p["id"], p["kind"], p["default"]) for p in spec["params"]] == [
        ("times", "number", 2),
        ("separator", "text", " "),
        ("shout", "bool", False),
        ("style", "select", "plain"),
        ("timeout_s", "int", 60),
    ]
    assert spec["params"][0]["min"] == 1 and spec["params"][3]["options"] == ["plain", "numbered"]
    # Same approval, caching and handler rules as the Python block.
    assert spec["requires_approval"] == ["python"] and spec["side_effect"] and not spec["cacheable"]
    assert spec["handler"] == "custom.script" and spec["code_param"] == "_script"


def test_parsing_never_runs_the_script(tmp_path):
    marker = tmp_path / "ran"
    source = f"open({str(marker)!r}, 'w').write('x')\nraise SystemExit(3)\n" + MINIMAL
    custom_nodes.parse_header(source)
    custom_nodes.build_spec(source)
    assert not marker.exists()


@pytest.mark.parametrize(
    "source, line, words",
    [
        ("NODE = {\n  'name': 'x',,\n}\n", 2, "syntax"),
        ("def run(inputs, params):\n    return {}\n", None, "No NODE"),
        ("NODE = make_header()\ndef run(inputs, params): pass\n", 1, "dict literal"),
        (
            "NODE = {\n  'name': __import__('os').system('id'),\n  'outputs': []}\ndef run(i, p): pass\n",
            2,
            "plain data",
        ),
        ("NODE = {\n  'name': 'x',\n  'colour': 'red',\n}\n", 3, "unknown key 'colour'"),
        (
            "NODE = {'name': 'x',\n 'outputs': [\n   {'name': 'pic', 'type': 'image'}]}\ndef run(i, p): pass\n",
            3,
            "media",
        ),
        (
            "NODE = {'name': 'x',\n 'outputs': [{'name': 'y', 'type': 'txt'}]}\ndef run(i, p): pass\n",
            2,
            "'txt'",
        ),
        (
            "NODE = {'name': 'x', 'outputs': [\n {'name': 'y', 'type': 'text'},\n {'name': 'y', 'type': 'text'}]}\n"
            "def run(i, p): pass\n",
            3,
            "twice",
        ),
        ("NODE = {'name': 'x', 'outputs': [\n {'name': 'log', 'type': 'text'}]}\ndef run(i, p): pass\n", 2, "reserved"),
        ("NODE = {'name': 'x', 'outputs': [\n {'name': 'Bad-Name', 'type': 'text'}]}\ndef run(i, p): pass\n", 2, "lowercase"),
        ("NODE = {'name': 'x', 'outputs': []}\ndef run(i, p): pass\n", 1, "at least one"),
        ("NODE = {'name': 'x', 'outputs': [{'name': 'y', 'type': 'text'}]}\n", None, "def run"),
        ("NODE = {'name': 'x', 'outputs': [{'name': 'y', 'type': 'text'}]}\n\n\nasync def run(i, p): pass\n", 4, "async"),
        ("NODE = {'name': 'x', 'outputs': [{'name': 'y', 'type': 'text'}]}\ndef run(inputs): pass\n", 2, "two arguments"),
        (
            "NODE = {'name': 'x', 'outputs': [{'name': 'y', 'type': 'text'}],\n 'params': [\n"
            "  {'name': 'mode', 'type': 'select', 'options': ['a', 'b'], 'default': 'c'}]}\ndef run(i, p): pass\n",
            3,
            "not one of its options",
        ),
        (
            "NODE = {'name': 'x', 'outputs': [{'name': 'y', 'type': 'text'}],\n 'params': [\n"
            "  {'name': 'n', 'type': 'number', 'default': 50, 'max': 10}]}\ndef run(i, p): pass\n",
            3,
            "outside its min and max",
        ),
        (
            "NODE = {'name': 'x', 'outputs': [{'name': 'y', 'type': 'text'}],\n 'params': [\n"
            "  {'name': 'timeout_s', 'type': 'number'}]}\ndef run(i, p): pass\n",
            3,
            "reserved",
        ),
        (
            "NODE = {'name': 'x', 'outputs': [{'name': 'y', 'type': 'text'}],\n 'params': [\n"
            "  {'name': 'n', 'type': 'number', 'default': 1e309}]}\ndef run(i, p): pass\n",
            3,
            "finite number",
        ),
        (
            "NODE = {'name': 'x', 'outputs': [{'name': 'y', 'type': 'text'}],\n 'params': [\n"
            "  {'name': 'n', 'type': 'number', 'max': -1e400}]}\ndef run(i, p): pass\n",
            3,
            "finite number",
        ),
        (
            "NODE = {'name': 'x', 'outputs': [{'name': 'y', 'type': 'text'}],\n 'params': [\n"
            "  {'name': 'n', 'type': 'number', 'default': 0x" + "f" * 300 + "}]}\ndef run(i, p): pass\n",
            3,
            "finite number",
        ),
        ("NODE = {'name': 'x', 'category': 'misc', 'outputs': [{'name': 'y', 'type': 'text'}]}\ndef run(i, p): pass\n", 1, "not a Flows category"),
        (MINIMAL + "\nNODE = {}\n", 11, "more than once"),
        ("", None, "empty"),
    ],
)
def test_bad_scripts_are_refused_with_the_line(source, line, words):
    with pytest.raises(custom_nodes.CustomNodeError) as caught:
        custom_nodes.parse_header(source)
    assert words.lower() in str(caught.value).lower()
    assert caught.value.line == line
    if line:
        assert str(caught.value).startswith(f"Line {line}: ")


def test_a_script_over_the_size_limit_is_refused():
    with pytest.raises(custom_nodes.CustomNodeError, match = "KB"):
        custom_nodes.parse_header(MINIMAL + "#" * custom_nodes.MAX_SCRIPT_BYTES)


def test_outputs_are_checked_against_their_port_types():
    spec = custom_nodes.build_spec(TEMPLATE)
    assert custom_nodes.check_outputs(spec, {"text": "a", "length": 1, "extra": 1}) == {"text": "a", "length": 1}
    for result, words in [
        ([1], "must return a dict"),
        ({"text": "a"}, "no 'length'"),
        ({"text": 3, "length": 1}, "int for 'text'"),
        ({"text": "a", "length": True}, "bool for 'length'"),
        ({"text": "a", "length": "1"}, "str for 'length'"),
    ]:
        with pytest.raises(ValueError, match = words):
            custom_nodes.check_outputs(spec, result)


# --- In a flow ------------------------------------------------------------------------------------------------------


def test_a_flow_compiles_against_the_scripts_ports_and_needs_its_hash_approved():
    d = doc(
        [node("t", "input.text", value = "hi"), _custom("c")],
        [edge("t", "text", "c", "text")],
    )
    plan = compiler.compile_flow(compiler.parse_document(d)[0])
    assert plan.specs["c"]["title"] == "Repeat text"
    need = capabilities.doc_needs(d)
    assert need["python"] == [capabilities.code_hash(TEMPLATE)]
    # A port the script does not declare is refused like any other.
    bad = doc([node("t", "input.text"), _custom("c")], [edge("t", "text", "c", "nope")])
    with pytest.raises(compiler.CompileError, match = "no input nope"):
        compiler.compile_flow(compiler.parse_document(bad)[0])


def test_a_script_edited_inside_the_document_no_longer_matches_its_type():
    tampered = _custom("c")
    tampered["params"]["_script"] = TEMPLATE.replace("Repeat text", "Other")
    with pytest.raises(compiler.CompileError, match = "does not match its type"):
        compiler.compile_flow(compiler.parse_document(doc([tampered]))[0])


@pytest.fixture
def ran(monkeypatch):
    calls: list = []
    import core.inference.tools as tools

    monkeypatch.setattr(tools, "execute_tool", _subprocess_execute_tool(calls))
    return calls


class Ctx:
    flow_id = "f1"

    def __init__(self, source = TEMPLATE, params = None, inputs = None):
        self.spec = custom_nodes.build_spec(source)
        self.params = {"_script": source, **{p["id"]: p["default"] for p in self.spec["params"]}, **(params or {})}
        self.inputs = inputs or {}

    def on_cancel(self, fn):
        pass


def test_the_template_runs_through_the_python_blocks_sandbox_call(ran):
    ctx = Ctx(params = {"times": 3, "separator": "-", "shout": True, "style": "numbered"}, inputs = {"text": "hi"})
    outputs = asyncio.run(custom.run_custom(ctx))
    assert outputs == {"text": "1. HI-2. HI-3. HI", "length": 17, "log": "repeated 3 times"}
    [(name, args, kwargs)] = ran
    # Same tool, session and timeout as the Python block.
    assert name == "python" and kwargs["session_id"] == automation.python_session("f1")
    assert kwargs["timeout"] == 60 and "cancel_event" in kwargs
    assert "<custom node>" in args["code"]


def test_errors_and_wrong_outputs_show_on_the_node(ran):
    broken = MINIMAL.replace('inputs["x"] * 2', 'inputs["x"] / 0')
    with pytest.raises(FlowNodeError, match = r"ZeroDivisionError") as caught:
        asyncio.run(custom.run_custom(Ctx(broken, inputs = {"x": 1})))
    assert "<custom node>" in str(caught.value) and "line 9" in str(caught.value)
    wrong = MINIMAL.replace('{"y": inputs["x"] * 2}', '{"y": "two"}')
    with pytest.raises(FlowNodeError, match = "str for 'y'"):
        asyncio.run(custom.run_custom(Ctx(wrong, inputs = {"x": 1})))
    timeout = Ctx(MINIMAL, params = {"timeout_s": 5000}, inputs = {"x": 2})
    assert asyncio.run(custom.run_custom(timeout))["y"] == 4
    assert ran[-1][2]["timeout"] == 600


def test_a_handler_refuses_a_script_that_is_not_its_type():
    ctx = Ctx()
    ctx.params["_script"] = MINIMAL
    with pytest.raises(FlowNodeError, match = "does not match"):
        asyncio.run(custom.run_custom(ctx))


@pytest.mark.skipif(
    os.environ.get("UNSLOTH_FLOWS_REAL_SANDBOX") != "1",
    reason = "set UNSLOTH_FLOWS_REAL_SANDBOX=1 to run the template in this machine's real Python sandbox",
)
def test_the_template_in_the_real_sandbox():
    ctx = Ctx(inputs = {"text": "hi"})
    assert asyncio.run(custom.run_custom(ctx))["text"] == "hi hi"


# --- Routes and the engine --------------------------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, ran):
    eng = engine_mod.FlowEngine()
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None
    eng.handlers = {"input.text": inputs.text, "custom.script": custom.run_custom}
    monkeypatch.setattr(engine_mod, "ENGINE", eng)
    app = FastAPI()

    @app.middleware("http")
    async def as_owner(request: Request, call_next):
        token = bind_account(OWNER)
        try:
            return await call_next(request)
        finally:
            reset_account(token)

    app.include_router(flows_routes.router, prefix = "/api/flows")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    with TestClient(app) as test_client:
        yield test_client


def test_library_upload_replace_delete(client):
    bad = client.post("/api/flows/custom-nodes", json = {"source": "NODE = {\n 'name': 1}\n"})
    assert bad.status_code == 400 and bad.json()["detail"]["line"] == 2
    huge = MINIMAL.replace('"outputs"', '"params": [{"name": "n", "type": "number", "default": 1e309}],\n    "outputs"')
    refused = client.post("/api/flows/custom-nodes", json = {"source": huge})
    assert refused.status_code == 400 and "finite" in refused.json()["detail"]["message"]
    assert client.get("/api/flows/custom-nodes").json()["nodes"] == []
    made = client.post("/api/flows/custom-nodes", json = {"source": TEMPLATE}).json()
    assert made["name"] == "Repeat text" and made["spec"]["type"] == custom_nodes.node_type(TEMPLATE)
    assert made["hash"] == custom_nodes.script_hash(TEMPLATE)
    listed = client.get("/api/flows/custom-nodes").json()["nodes"]
    assert [n["id"] for n in listed] == [made["id"]]
    replaced = client.put(f"/api/flows/custom-nodes/{made['id']}", json = {"source": MINIMAL}).json()
    assert replaced["id"] == made["id"] and replaced["name"] == "Double"
    assert replaced["hash"] != made["hash"]
    assert client.put("/api/flows/custom-nodes/cn_none", json = {"source": MINIMAL}).status_code == 404
    assert client.delete(f"/api/flows/custom-nodes/{made['id']}").status_code == 204
    assert client.get("/api/flows/custom-nodes").json()["nodes"] == []
    parsed = client.post("/api/flows/custom-nodes/parse", json = {"source": MINIMAL}).json()
    assert parsed["spec"]["title"] == "Double"


def test_the_template_download_and_library_list_come_from_the_sandbox(client, monkeypatch):
    monkeypatch.setattr(custom_nodes, "_libs_cache", None)
    libs = client.get("/api/flows/custom-nodes/libraries").json()
    # The fake sandbox is this interpreter, so the probe answers from inside it.
    assert libs["source"] == "sandbox" and any(p["name"].lower() == "pytest" for p in libs["packages"])
    got = client.get("/api/flows/custom-nodes/template")
    assert "my_node_template.py" in got.headers["content-disposition"]
    assert custom_nodes.build_spec(got.text)["title"] == "Repeat text"
    assert "pytest" in got.text.lower()


def test_a_custom_node_runs_in_a_flow_once_its_script_is_approved(client):
    d = doc(
        [node("t", "input.text", value = "hey"), _custom("c", times = 2, separator = "+")],
        [edge("t", "text", "c", "text")],
        id = "cf1",
    )
    assert client.post("/api/flows", json = {"doc": d}).status_code == 200

    def run() -> dict:
        run_id = client.post("/api/flows/cf1/runs", json = {}).json()["run_id"]
        for _ in range(200):
            snap = client.get(f"/api/flows/runs/{run_id}").json()
            if snap["run"]["status"] in ("completed", "failed", "cancelled"):
                return snap
            time.sleep(0.05)
        raise AssertionError("run did not finish")

    refused = run()["nodes"]["c"]
    assert refused["status"] == "failed" and "not approved" in refused["error"]
    pending = client.get("/api/flows/cf1/capabilities").json()["pending"]
    assert pending["python"] == [capabilities.code_hash(TEMPLATE)]
    client.post("/api/flows/cf1/approve", json = pending)
    snap = run()["nodes"]["c"]
    assert snap["status"] == "done", snap
    assert snap["outputs"] == {"text": "hey+hey", "length": 7, "log": "repeated 2 times"}


def test_an_exported_flow_carries_the_script():
    d = doc([_custom("c")])
    from core.flows import package

    scrubbed = package.scrub(d)
    assert scrubbed["nodes"][0]["params"]["_script"] == TEMPLATE
    assert custom_nodes.spec_for_node(scrubbed["nodes"][0])["title"] == "Repeat text"


def test_the_library_probe_fits_a_large_env_under_the_tool_output_limit():
    import base64
    import json
    import zlib

    packages = [[f"package-{i}", f"{i}.0.{i % 7}"] for i in range(900)]
    body = json.dumps({"python": "3.13", "packages": packages})
    line = custom_nodes._LIBS_MARKER + base64.b64encode(zlib.compress(body.encode(), 9)).decode()
    assert len(line) < 16000
    found = custom_nodes._parse_probe(line + "\n")
    assert found["python"] == "3.13" and len(found["packages"]) == 900
    # Cut mid-line (the tool truncates long output): no list rather than a wrong one.
    assert custom_nodes._parse_probe(line[: len(line) // 2]) is None
