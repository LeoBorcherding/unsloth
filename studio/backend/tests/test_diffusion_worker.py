# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Torch image generation behind a process boundary: spawn, calls, cancel, crash, shutdown."""

import os
import sys
import textwrap
import threading
import time
import pytest

from core.inference.diffusion_conditioning import LocalizedEdit
from core.inference.diffusion_worker import (
    DiffusionWorkerBackend,
    DiffusionWorkerClient,
    DiffusionWorkerCrashed,
)
from core.inference.generation_ticket import GenerationTicket, generation_ticket
from utils.account_context import AccountContext, bind_account, reset_account

_FAKE = textwrap.dedent(
    """
    import contextlib
    import os
    import sys
    import threading
    import time

    from core.inference.generation_ticket import bind_generation_cancel
    from utils.account_context import current_account_id


    class Unpicklable(Exception):
        def __init__(self, a, b):
            super().__init__(f"{a}/{b}")


    class Fake:
        def __init__(self):
            self._state = threading.Lock()
            self._active_generate_account = None
            self._gen_cancel = None
            self.unloads = 0
            self.transition = False
            self.waiters = set()

        @property
        def is_loaded(self):
            return True

        def status(self):
            return {"loaded": True, "pid": os.getpid(), "executable": sys.executable}

        def echo(self, *args, **kwargs):
            return args, kwargs

        def boom(self):
            raise ValueError("bad request")

        def odd(self):
            raise Unpicklable(1, 2)

        def crash(self):
            sys.stderr.write("worker fell over\\n")
            sys.stderr.flush()
            os._exit(3)

        def swap_state(self):
            self._state = threading.Lock()

        def generate(self, *, steps=200, sampler=None, shift=None, **_):
            cancel = threading.Event()
            bind_generation_cancel(cancel)
            self._gen_cancel = cancel
            self._active_generate_account = current_account_id()
            try:
                for step in range(steps):
                    if cancel.wait(0.01):
                        raise RuntimeError("Generation cancelled.")
                return {"images": ["ok"], "account": current_account_id()}
            finally:
                self._active_generate_account = None
                self._gen_cancel = None

        def cancel_generate(self, expected_account=None):
            cancel = self._gen_cancel
            if cancel is not None:
                cancel.set()
                return True
            # Mirrors DiffusionBackend: during a load or unload, Stop signals the queued waiters.
            if not self.transition or not self.waiters:
                return False
            for waiter in list(self.waiters):
                waiter.set()
            return True

        @contextlib.contextmanager
        def generation_waiter(self, cancel):
            self.waiters.add(cancel)
            try:
                yield
            finally:
                self.waiters.discard(cancel)

        def set_transition(self, value):
            self.transition = value

        def waiter_count(self):
            return len(self.waiters)

        def unload(self, **kwargs):
            self.unloads += 1
            return {"unloads": self.unloads, **kwargs}


    def make():
        return Fake()
    """
)


@pytest.fixture
def fake_dir(tmp_path):
    (tmp_path / "fake_diffusion_target.py").write_text(_FAKE)
    return tmp_path


def _client(fake_dir, **kwargs):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(fake_dir)
    return DiffusionWorkerClient(target = "fake_diffusion_target:make", env = env, **kwargs)


@pytest.fixture
def backend(fake_dir):
    proxy = DiffusionWorkerBackend(_client(fake_dir))
    yield proxy
    proxy.shutdown()


def _wait_dead(pid, timeout = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.05)
    return False


def test_calls_run_in_a_separate_process(backend):
    status = backend.status()
    assert status["loaded"] is True
    assert status["pid"] != os.getpid()
    assert backend.is_loaded is True


def test_worker_uses_the_given_python(fake_dir, tmp_path):
    if sys.platform == "win32":
        pytest.skip("symlinked interpreter")
    link = tmp_path / "other-venv-python"
    link.symlink_to(sys.executable)
    proxy = DiffusionWorkerBackend(_client(fake_dir, python_executable = str(link)))
    try:
        assert proxy.status()["executable"] == str(link)
    finally:
        proxy.shutdown()


def test_arguments_and_results_cross_the_boundary(backend):
    from PIL import Image

    image = Image.new("RGB", (8, 4), (10, 20, 30))
    edit = LocalizedEdit(mode = "mask", image = "abc")
    args, kwargs = backend.echo(image, localized_edit = edit)
    assert args[0].size == (8, 4) and args[0].getpixel((0, 0)) == (10, 20, 30)
    assert kwargs["localized_edit"] == edit


class _OnlyHere:
    pass


def test_an_argument_the_worker_cannot_import_fails_only_that_call(backend):
    pid = backend.status()["pid"]
    with pytest.raises(Exception, match = "No module named|_OnlyHere"):
        backend.echo(_OnlyHere())
    with pytest.raises(Exception):
        backend.echo(lambda: None)
    assert backend.status()["pid"] == pid


def test_backend_errors_keep_their_type(backend):
    with pytest.raises(ValueError, match = "bad request"):
        backend.boom()
    # One the server cannot rebuild still surfaces with its name and message.
    with pytest.raises(RuntimeError, match = "Unpicklable: 1/2"):
        backend.odd()


def test_live_state_comes_back_as_a_stable_reference(backend):
    first = backend._state
    assert first and backend._state is first
    backend.swap_state()
    assert backend._state is not first


def test_forwarded_methods_keep_the_backend_signature(backend):
    import inspect

    # routes/inference.py offers sampler, shift and friends only when the engine's signature names them.
    params = inspect.signature(backend.generate).parameters
    assert {"steps", "sampler", "shift"} <= set(params)
    assert params["steps"].default == 200
    assert list(inspect.signature(backend.cancel_generate).parameters) == ["expected_account"]


def test_generation_runs_as_the_calling_account(backend):
    token = bind_account(AccountContext("acct-7", "seven"))
    try:
        result = backend.generate(steps = 1)
    finally:
        reset_account(token)
    assert result == {"images": ["ok"], "account": "acct-7"}


def _start_generate(backend, ticket = None):
    out = {}

    def run():
        try:
            if ticket is None:
                out["result"] = backend.generate(steps = 2000)
            else:
                with generation_ticket(ticket):
                    out["result"] = backend.generate(steps = 2000)
        except BaseException as exc:  # noqa: BLE001
            out["error"] = exc

    thread = threading.Thread(target = run)
    thread.start()
    deadline = time.monotonic() + 10
    while not backend.generation_in_flight() and time.monotonic() < deadline:
        time.sleep(0.01)
    while backend._active_generate_account is None and time.monotonic() < deadline:
        time.sleep(0.01)
    return thread, out


def test_stop_ticket_cancels_the_generation_in_the_worker(backend):
    ticket = GenerationTicket()
    thread, out = _start_generate(backend, ticket)
    assert backend.generation_holder() is not None
    ticket.stop()
    thread.join(10)
    assert not thread.is_alive()
    assert "cancelled" in str(out.get("error"))
    assert not backend.generation_in_flight()
    assert backend.generation_holder() is None


def test_cancel_generate_reaches_the_worker(backend):
    thread, out = _start_generate(backend)
    assert backend.cancel_generate() is True
    thread.join(10)
    assert "cancelled" in str(out.get("error"))


def test_stop_during_a_transition_reaches_a_queued_waiter(backend):
    stopped = threading.Event()
    with backend.generation_waiter(stopped):
        assert backend.waiter_count() == 1
        backend.set_transition(True)
        assert backend.cancel_generate() is True
        assert stopped.wait(5)
    deadline = time.monotonic() + 5
    while backend.waiter_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert backend.waiter_count() == 0


def test_leaving_a_waiter_unregisters_it_without_signalling(backend):
    stopped = threading.Event()
    with backend.generation_waiter(stopped):
        pass
    deadline = time.monotonic() + 5
    while backend.waiter_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert backend.waiter_count() == 0
    backend.set_transition(True)
    assert backend.cancel_generate() is False
    assert not stopped.is_set()


def test_crash_fails_the_call_and_the_next_call_respawns(backend):
    first_pid = backend.status()["pid"]
    thread, out = _start_generate(backend)
    with pytest.raises(DiffusionWorkerCrashed, match = "exit code 3") as info:
        backend.crash()
    assert "worker fell over" in str(info.value)
    thread.join(10)
    assert isinstance(out.get("error"), DiffusionWorkerCrashed)
    assert not backend.generation_in_flight()
    assert backend.status()["pid"] != first_pid


def test_shutdown_kills_the_worker_and_fails_waiting_calls(backend):
    pid = backend.status()["pid"]
    thread, out = _start_generate(backend)
    backend.shutdown()
    thread.join(10)
    assert isinstance(out.get("error"), DiffusionWorkerCrashed)
    assert _wait_dead(pid)


def test_worker_exits_when_the_server_goes_away(fake_dir):
    client = _client(fake_dir)
    pid = client.call("status")["pid"]
    # The server dying closes the pipe; the worker must not outlive it.
    client._session.proc.stdin.close()
    assert _wait_dead(pid)


def test_unload_guard_is_checked_before_crossing(backend):
    assert backend.unload(only_if = lambda: False) is None
    assert backend.unload(only_if = lambda: True, expected_account = "a")["unloads"] == 1


def test_get_diffusion_backend_returns_the_worker_proxy(monkeypatch):
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_worker as worker

    monkeypatch.setenv("UNSLOTH_DIFFUSION_WORKER", "1")
    monkeypatch.setattr(worker, "_worker_backend", None)
    proxy = diffusion.get_diffusion_backend()
    try:
        assert type(proxy).__name__ == "DiffusionWorkerBackend"
        assert diffusion.get_diffusion_backend() is proxy
        # The real backend answers from the worker, not this process.
        assert proxy.is_loaded is False
        assert proxy.status()["loaded"] is False
        import inspect

        assert {"sampler", "sd_controls", "shift", "live_preview", "mask_blur"} <= set(
            inspect.signature(proxy.generate).parameters
        )
        assert proxy._client.pid not in (None, os.getpid())
    finally:
        proxy.shutdown()
        monkeypatch.setattr(worker, "_worker_backend", None)


def test_default_stays_in_process(monkeypatch):
    import core.inference.diffusion as diffusion
    monkeypatch.delenv("UNSLOTH_DIFFUSION_WORKER", raising = False)
    assert type(diffusion.get_diffusion_backend()).__name__ == "DiffusionBackend"
