# SPDX-License-Identifier: AGPL-3.0-only
"""The GPU worker loop's generate_audio_input branch honours an unload's drain.

An unload sets drain_event, then cancel_event. A speech-to-text command that clears
cancel_event without checking the drain erases that cancel and transcribes to the end,
stalling the model switch. The branch is lifted out of run_inference_process and run
as written, with the handler stubbed.
"""

import ast
import inspect
import queue
import textwrap
import threading

from core.inference import worker


def _gpu_audio_input_branch():
    tree = ast.parse(textwrap.dedent(inspect.getsource(worker.run_inference_process)))
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "generate_audio_input"
    ]
    # The MLX loop comes first in the source; the GPU loop's branch is the last one.
    branch = max(branches, key = lambda n: n.lineno)
    body = "\n".join(ast.unparse(stmt) for stmt in branch.body)
    src = (
        "def _branch(backend, cmd, resp_queue, cancel_event, drain_event):\n"
        "    for _ in (0,):\n" + textwrap.indent(body, " " * 8) + "\n"
    )
    calls = []
    namespace = dict(vars(worker))
    namespace["_handle_generate_audio_input"] = lambda *args: calls.append(args)
    exec(compile(src, "<generate_audio_input branch>", "exec"), namespace)
    return namespace["_branch"], calls


class _UnloadLandsAfterFirstCheck:
    """drain_event that reads unset once, then set: the unload arrives mid-branch."""

    def __init__(self, cancel_event):
        self._reads = 0
        self._cancel = cancel_event

    def is_set(self):
        self._reads += 1
        if self._reads == 1:
            self._cancel.set()
            return False
        return True


def test_a_queued_transcription_is_skipped_while_an_unload_drains():
    branch, calls = _gpu_audio_input_branch()
    drain, cancel, rq = threading.Event(), threading.Event(), queue.Queue()
    drain.set()
    cancel.set()

    branch(object(), {"type": "generate_audio_input", "request_id": "r1"}, rq, cancel, drain)

    assert calls == []
    assert cancel.is_set()
    resp = rq.get_nowait()
    assert (resp["type"], resp["request_id"], resp["cancelled"]) == ("gen_done", "r1", True)


def test_an_unload_landing_before_the_clear_still_skips_the_transcription():
    branch, calls = _gpu_audio_input_branch()
    cancel, rq = threading.Event(), queue.Queue()

    branch(
        object(),
        {"type": "generate_audio_input", "request_id": "r2"},
        rq,
        cancel,
        _UnloadLandsAfterFirstCheck(cancel),
    )

    assert calls == []
    assert rq.get_nowait()["cancelled"] is True


def test_a_transcription_runs_with_a_fresh_cancel_when_nothing_drains():
    branch, calls = _gpu_audio_input_branch()
    drain, cancel, rq = threading.Event(), threading.Event(), queue.Queue()
    cancel.set()

    branch(object(), {"type": "generate_audio_input", "request_id": "r3"}, rq, cancel, drain)

    assert len(calls) == 1
    assert not cancel.is_set()
    assert rq.empty()
