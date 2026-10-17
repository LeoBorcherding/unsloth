# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A slow whisper.cpp transcription fails that request only; it must not turn the engine off.

whisper-server sends nothing until the whole clip is transcribed, so a long clip on CPU used to
hit the fixed 600 s socket timeout, latch note_runtime_inference_failure, and route every later
request to Transformers until a restart.
"""

from __future__ import annotations

import json
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from core.inference import stt_ggml_sidecar


@pytest.fixture(autouse = True)
def _clean_runtime_state():
    stt_ggml_sidecar.clear_runtime_inference_failure()
    yield
    stt_ggml_sidecar.clear_runtime_inference_failure()


@pytest.fixture
def slow_server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(1.5)
            body = json.dumps({"text": "hello"}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    yield types.SimpleNamespace(_port = server.server_address[1], _route = "")
    server.shutdown()
    server.server_close()


def _wav(seconds: float) -> bytes:
    return b"\0" * (44 + int(seconds * 2 * stt_ggml_sidecar._TARGET_SAMPLE_RATE))


def test_a_timeout_fails_the_request_without_disabling_the_engine(monkeypatch, slow_server):
    monkeypatch.setattr(stt_ggml_sidecar, "_TRANSCRIBE_TIMEOUT_SECONDS", 0.3)

    with pytest.raises(stt_ggml_sidecar.SttEngineUnavailableError) as raised:
        stt_ggml_sidecar.GgmlSttSidecar._post_inference(slow_server, _wav(0.01), None, False)

    assert stt_ggml_sidecar.runtime_inference_failure() is None
    assert "from now on" not in str(raised.value)


def test_the_timeout_scales_with_clip_length(monkeypatch):
    seen: list[float] = []

    class RefusingConnection:
        def __init__(self, host, port, timeout = None):
            seen.append(timeout)

        def request(self, *args, **kwargs):
            raise ConnectionRefusedError("refused")

        def close(self):
            pass

    monkeypatch.setattr(stt_ggml_sidecar.http.client, "HTTPConnection", RefusingConnection)
    fake = types.SimpleNamespace(_port = 1, _route = "")

    for seconds in (5, 30 * 60):
        with pytest.raises(stt_ggml_sidecar.SttEngineUnavailableError):
            stt_ggml_sidecar.GgmlSttSidecar._post_inference(fake, _wav(seconds), None, False)

    assert seen[0] == stt_ggml_sidecar._TRANSCRIBE_TIMEOUT_SECONDS
    # Thirty minutes of audio gets more than the fixed 600 s a CPU run of large-v3 needs.
    assert seen[1] >= 2 * 30 * 60
    # A refused connection is still a broken runtime.
    assert "ConnectionRefusedError" in (stt_ggml_sidecar.runtime_inference_failure() or "")
