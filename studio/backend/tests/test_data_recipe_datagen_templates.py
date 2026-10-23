# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Tool Calls and Reasoning Traces templates run end to end, with a stub standing in for the local model."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.data_recipe.quality import compute_quality

TEMPLATES = (
    Path(__file__).resolve().parents[2]
    / "frontend/src/features/data-recipes/learning-recipes"
)


def _answer(body: str) -> str:
    if "Solve this problem" in body:
        n = body.count("a") % 97
        return "```json\n" + json.dumps({"reasoning": f"Step 1: {n} + 1.", "final_answer": str(n + 1)}) + "\n```"
    if "Return only the JSON object" in body:
        return json.dumps({"city": "Tokyo", "days": len(body) % 7})
    return f"Generated text {len(body)} {hash(body) % 100003}"


@pytest.fixture(name = "provider")
def fixture_provider():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"])).decode()
            reply = json.dumps(
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "x",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": _answer(body)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def _preview(tmp_path: Path, name: str, endpoint: str, rows: int = 4):
    pytest.importorskip("data_designer")
    from core.data_recipe.service import build_config_builder, create_data_designer

    recipe = json.loads((TEMPLATES / name).read_text(encoding = "utf-8"))["recipe"]
    for provider in recipe["model_providers"]:
        provider["endpoint"] = endpoint
        provider.pop("is_local", None)
    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    result = designer.preview(build_config_builder(recipe), num_records = rows)
    analysis = result.analysis.model_dump(mode = "json") if result.analysis else None
    return result.dataset.to_dict(orient = "records"), analysis


def test_tool_calls_template(tmp_path: Path, provider) -> None:
    rows, analysis = _preview(tmp_path, "tool-calls.json", provider)
    assert len(rows) == 4
    for row in rows:
        assert set(row) >= {"tool_name", "tools", "user_message", "arguments", "arguments_valid"}
        assert len(json.loads(row["tools"])) == 4
        assert json.loads(row["arguments"])["city"] == "Tokyo"
        assert row["arguments_valid"]["is_valid"] is True
        assert "city" not in row and "user_style" not in row
    quality = compute_quality(rows, analysis)
    assert quality["validators"][0]["pass_rate"] == 1.0
    assert quality["content_columns"] == ["user_message", "arguments"]


def test_reasoning_traces_template(tmp_path: Path, provider) -> None:
    rows, analysis = _preview(tmp_path, "reasoning-traces.json", provider)
    assert len(rows) == 4
    for row in rows:
        solution = row["solution"]
        assert row["assistant"] == (
            f"<think>\n{solution['reasoning']}\n</think>\n\n{solution['final_answer']}"
        )
    quality = compute_quality(rows, analysis)
    assert quality["columns"][1]["name"] == "solution"
    assert quality["columns"][1]["format_failures"] == 0
