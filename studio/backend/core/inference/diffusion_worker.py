# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Torch image generation in a child process (UNSLOTH_DIFFUSION_WORKER=1).

The child runs the real ``DiffusionBackend``; the server holds a proxy that forwards every
attribute and call over length-prefixed pickle frames on the child's stdin/stdout. The child is a
plain ``python -m`` run, not multiprocessing, so its interpreter can come from another venv
(``python_executable`` / UNSLOTH_DIFFUSION_WORKER_PYTHON)."""

from __future__ import annotations

import atexit
import collections
import importlib
import inspect
import itertools
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

_BACKEND_PATH = str(Path(__file__).resolve().parent.parent.parent)
_ENABLE_TOKENS = frozenset({"1", "on", "true", "yes"})
_CHILD_ENV = "UNSLOTH_DIFFUSION_WORKER_CHILD"
_DEFAULT_TARGET = "core.inference.diffusion_worker:child_backend"
# Protocol 4 so a child on an older Python in another venv can still read the frames.
_PICKLE_PROTOCOL = 4
_HEADER = struct.Struct(">Q")
_READY_TIMEOUT_SECONDS = 120.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_STDERR_TAIL_LINES = 40
_CALLABLE = "__diffusion_worker_callable__"


class DiffusionWorkerError(RuntimeError):
    """The image worker failed in a way that is not the backend's own error."""


class DiffusionWorkerCrashed(DiffusionWorkerError):
    """The image worker exited while a call was in flight."""


def worker_enabled() -> bool:
    if os.environ.get(_CHILD_ENV):
        return False
    return os.environ.get("UNSLOTH_DIFFUSION_WORKER", "").strip().lower() in _ENABLE_TOKENS


def _write_frame(stream, lock: threading.Lock, message: Any) -> None:
    data = pickle.dumps(message, protocol = _PICKLE_PROTOCOL)
    with lock:
        stream.write(_HEADER.pack(len(data)) + data)
        stream.flush()


def _read_exact(stream, size: int) -> Optional[bytes]:
    chunks = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            return None
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _read_frame(stream) -> Any:
    header = _read_exact(stream, _HEADER.size)
    if header is None:
        raise EOFError
    body = _read_exact(stream, _HEADER.unpack(header)[0])
    if body is None:
        raise EOFError
    return pickle.loads(body)


class _RemoteRef:
    """Stands in for a child attribute that cannot cross (a load state holding a pipeline): truthy,
    and the same object for as long as the child holds the same one, so identity checks still work."""

    __slots__ = ("name", "ident")

    def __init__(self, name: str, ident: int) -> None:
        self.name = name
        self.ident = ident

    def __repr__(self) -> str:
        return f"<remote {self.name} {self.ident:#x}>"


class _Pending:
    __slots__ = ("event", "ok", "value")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.ok = False
        self.value: Any = None


class _Session:
    """One child process and the calls waiting on it; ``dead`` is set once, under ``lock``."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self.lock = threading.Lock()
        self.pending: dict[int, _Pending] = {}
        self.dead: Optional[str] = None
        self.ready = threading.Event()


class DiffusionWorkerClient:
    """Owns the image worker. A crash fails the calls in flight; the next call starts a new child."""

    def __init__(
        self,
        *,
        python_executable: Optional[str] = None,
        target: str = _DEFAULT_TARGET,
        env: Optional[dict[str, str]] = None,
        ready_timeout: float = _READY_TIMEOUT_SECONDS,
    ) -> None:
        self.python_executable = python_executable or sys.executable
        self.target = target
        self._env = env
        self._ready_timeout = ready_timeout
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._session: Optional[_Session] = None
        self._ids = itertools.count(1)
        self._stderr_tail: collections.deque = collections.deque(maxlen = _STDERR_TAIL_LINES)

    @property
    def pid(self) -> Optional[int]:
        session = self._session
        return session.proc.pid if session is not None and session.dead is None else None

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ if self._env is None else self._env)
        env[_CHILD_ENV] = "1"
        env.pop("UNSLOTH_DIFFUSION_WORKER", None)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (_BACKEND_PATH, env.get("PYTHONPATH", "")) if p
        )
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def _ensure_started(self) -> _Session:
        with self._lock:
            session = self._session
            if session is None or session.dead is not None:
                self._stderr_tail.clear()
                try:
                    proc = subprocess.Popen(
                        [
                            self.python_executable,
                            "-m",
                            "core.inference.diffusion_worker",
                            self.target,
                        ],
                        stdin = subprocess.PIPE,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                        cwd = _BACKEND_PATH,
                        env = self._child_env(),
                    )
                except OSError as exc:
                    raise DiffusionWorkerError(
                        f"Could not start the image worker with {self.python_executable}: {exc}"
                    ) from exc
                session = self._session = _Session(proc)
                threading.Thread(
                    target = self._drain_stderr,
                    args = (proc,),
                    daemon = True,
                    name = "diffusion-worker-stderr",
                ).start()
                threading.Thread(
                    target = self._read_loop,
                    args = (session,),
                    daemon = True,
                    name = "diffusion-worker-reader",
                ).start()
        if not session.ready.wait(self._ready_timeout) or session.dead is not None:
            self._kill(session.proc)
            self._mark_dead(session, "did not start")
            raise DiffusionWorkerError(session.dead)
        return session

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace")
            self._stderr_tail.append(line.rstrip("\n"))
            try:
                sys.stderr.write(line)
            except Exception:  # noqa: BLE001 - a closed server stderr must not stop the drain
                pass

    def _read_loop(self, session: _Session) -> None:
        proc = session.proc
        try:
            while True:
                message = _read_frame(proc.stdout)
                if message[0] == "ready":
                    session.ready.set()
                    continue
                with session.lock:
                    pending = session.pending.pop(message[1], None)
                if pending is None:
                    continue
                if message[0] == "result":
                    try:
                        pending.value, pending.ok = pickle.loads(message[2]), True
                    except Exception as exc:  # noqa: BLE001 - a result class this side cannot import
                        pending.value = DiffusionWorkerError(
                            f"Could not read the image worker's result: {exc}"
                        )
                else:
                    pending.value = self._rebuild_error(message)
                pending.event.set()
        except Exception:  # noqa: BLE001 - EOF or a torn frame both mean the child is gone
            pass
        try:
            proc.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            self._kill(proc)
        # Let the stderr drain catch up so the error carries the child's last words.
        time.sleep(0.1)
        self._mark_dead(session, "exited")

    def _mark_dead(self, session: _Session, what: str) -> None:
        with session.lock:
            if session.dead is None:
                code = session.proc.poll()
                tail = "\n".join(list(self._stderr_tail)[-10:])
                message = f"The image worker {what} (exit code {code})."
                session.dead = f"{message}\n{tail}" if tail else message
            pending, session.pending = session.pending, {}
        session.ready.set()
        for item in pending.values():
            item.value = DiffusionWorkerCrashed(session.dead)
            item.event.set()

    @staticmethod
    def _rebuild_error(message: tuple) -> BaseException:
        _, _, payload, type_name, text, child_tb = message
        if payload is not None:
            try:
                exc = pickle.loads(payload)
                if isinstance(exc, BaseException):
                    exc.__cause__ = DiffusionWorkerError(f"in the image worker:\n{child_tb}")
                    return exc
            except Exception:  # noqa: BLE001 - an error class the server cannot rebuild
                pass
        return DiffusionWorkerError(f"{type_name}: {text}")

    def submit(
        self,
        op: str,
        *payload: Any,
        account: Any = None,
    ) -> tuple[int, _Pending]:
        # Pickled here, so an argument that cannot cross fails this call alone.
        body = pickle.dumps((account, *payload), protocol = _PICKLE_PROTOCOL)
        session = self._ensure_started()
        call_id = next(self._ids)
        pending = _Pending()
        with session.lock:
            if session.dead is not None:
                raise DiffusionWorkerCrashed(session.dead)
            session.pending[call_id] = pending
        try:
            _write_frame(session.proc.stdin, self._send_lock, (op, call_id, body))
        except (OSError, ValueError) as exc:
            with session.lock:
                session.pending.pop(call_id, None)
            raise DiffusionWorkerCrashed("The image worker is unreachable.") from exc
        return call_id, pending

    def cancel(self, call_id: int) -> None:
        session = self._session
        if session is None or session.dead is not None:
            return
        try:
            _write_frame(session.proc.stdin, self._send_lock, ("cancel", call_id))
        except (OSError, ValueError):
            pass

    @staticmethod
    def result(pending: _Pending) -> Any:
        pending.event.wait()
        if pending.ok:
            return pending.value
        raise pending.value

    def call(
        self,
        method: str,
        *args: Any,
        account: Any = None,
        **kwargs: Any,
    ) -> Any:
        _, pending = self.submit("call", method, args, kwargs, account = account)
        return self.result(pending)

    def getattr(self, name: str) -> Any:
        _, pending = self.submit("getattr", name)
        return self.result(pending)

    def shutdown(self, timeout: float = _SHUTDOWN_TIMEOUT_SECONDS) -> None:
        with self._lock:
            session, self._session = self._session, None
        if session is None or session.dead is not None:
            return
        try:
            _write_frame(session.proc.stdin, self._send_lock, ("shutdown", 0))
            session.proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            session.proc.wait(timeout = timeout)
        except subprocess.TimeoutExpired:
            self._kill(session.proc)
        self._mark_dead(session, "was shut down")

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout = 5)


class DiffusionWorkerBackend:
    """Drop-in for ``DiffusionBackend`` whose pipeline lives in the worker. Anything not defined
    here is read from, or called on, the child's backend."""

    def __init__(self, client: Optional[DiffusionWorkerClient] = None) -> None:
        object.__setattr__(
            self,
            "_client",
            client
            or DiffusionWorkerClient(
                python_executable = os.environ.get("UNSLOTH_DIFFUSION_WORKER_PYTHON") or None
            ),
        )
        object.__setattr__(self, "_refs", {})
        object.__setattr__(self, "_signatures", {})
        object.__setattr__(self, "_local_lock", threading.Lock())
        object.__setattr__(self, "_in_flight", 0)
        object.__setattr__(self, "_holders", [])

    @staticmethod
    def _account() -> Any:
        from utils.account_context import current_account
        return current_account()

    def _wrap(self, name: str, value: Any) -> Any:
        if isinstance(value, tuple) and len(value) == 3 and value[0] == _CALLABLE:

            def forward(*args: Any, **kwargs: Any) -> Any:
                return self._client.call(name, *args, account = self._account(), **kwargs)

            forward.__name__ = name
            # The routes probe the engine's signature to decide which options it takes.
            forward.__signature__ = _rebuild_signature(value[2])
            return forward
        if isinstance(value, tuple) and len(value) == 3 and value[0] == _RemoteRef.__name__:
            key = (value[1], value[2])
            ref = self._refs.get(key)
            if ref is None:
                ref = self._refs[key] = _RemoteRef(value[1], value[2])
            return ref
        return value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return self._wrap(name, self._client.getattr(name))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{name} lives in the image worker and cannot be set from the server")

    def generation_in_flight(self) -> bool:
        return self._in_flight > 0

    def generation_holder(self) -> Optional[object]:
        with self._local_lock:
            return self._holders[0] if self._holders else None

    @contextmanager
    def generation_waiter(self, cancel: threading.Event):
        # Registered in the child's backend, so a Stop during a load or unload reaches this waiter there.
        call_id, pending = self._client.submit("waiter")

        def relay() -> None:
            pending.event.wait()
            if pending.ok and pending.value:
                cancel.set()

        threading.Thread(target = relay, daemon = True, name = "diffusion-worker-waiter").start()
        try:
            yield
        finally:
            self._client.cancel(call_id)

    @property
    def generate(self) -> Callable[..., dict[str, Any]]:
        signature = self._signatures.get("generate")
        if signature is None:
            remote = self._client.getattr("generate")
            signature = self._signatures["generate"] = _rebuild_signature(remote[2])

        def generate(**kwargs: Any) -> dict[str, Any]:
            return self._generate(**kwargs)

        generate.__signature__ = signature
        return generate

    def _generate(self, **kwargs: Any) -> dict[str, Any]:
        from core.inference.generation_ticket import bind_generation_cancel

        cancel = threading.Event()
        bind_generation_cancel(cancel)
        with self._local_lock:
            object.__setattr__(self, "_in_flight", self._in_flight + 1)
            self._holders.append(cancel)
        try:
            call_id, pending = self._client.submit(
                "call", "generate", (), kwargs, account = self._account()
            )
            sent = False
            # The parent's ticket (Stop, a superseding load) sets ``cancel``; the child stops only this call.
            while not pending.event.wait(0.05):
                if not sent and cancel.is_set():
                    self._client.cancel(call_id)
                    sent = True
            return self._client.result(pending)
        finally:
            with self._local_lock:
                object.__setattr__(self, "_in_flight", self._in_flight - 1)
                self._holders.remove(cancel)

    def unload(
        self,
        *,
        only_if: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ):
        # A callable cannot cross, so the guard is checked here, just before the call.
        if only_if is not None and not only_if():
            return None
        return self._client.call("unload", account = self._account(), **kwargs)

    def shutdown(self) -> None:
        self._client.shutdown()


_worker_backend: Optional[DiffusionWorkerBackend] = None
_worker_lock = threading.Lock()


def get_diffusion_worker_backend() -> DiffusionWorkerBackend:
    global _worker_backend
    with _worker_lock:
        if _worker_backend is None:
            _worker_backend = DiffusionWorkerBackend()
            atexit.register(_worker_backend.shutdown)
        return _worker_backend


def shutdown_diffusion_worker() -> None:
    backend = _worker_backend
    if backend is not None:
        backend.shutdown()


def generation_in_flight() -> bool:
    backend = _worker_backend
    return backend is not None and backend.generation_in_flight()


# Child side.


def child_backend() -> Any:
    from core.inference.diffusion import get_diffusion_backend
    return get_diffusion_backend()


def _load_target(spec: str) -> Any:
    module_name, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module_name), attr)()


def _encode_signature(fn: Any) -> Optional[list]:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    encoded = []
    for param in params:
        default = param.default
        try:
            pickle.dumps(default, protocol = _PICKLE_PROTOCOL)
        except Exception:  # noqa: BLE001 - the name and kind are what callers read
            default = None
        encoded.append((param.name, int(param.kind), param.default is param.empty, default))
    return encoded


def _rebuild_signature(encoded: Optional[list]) -> inspect.Signature:
    if encoded is None:
        return inspect.Signature(
            [
                inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
                inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
            ]
        )
    return inspect.Signature(
        [
            inspect.Parameter(
                name,
                inspect._ParameterKind(kind),
                default = inspect.Parameter.empty if no_default else default,
            )
            for name, kind, no_default, default in encoded
        ]
    )


def _encode_attr(name: str, value: Any) -> Any:
    if callable(value) and not isinstance(value, type):
        return (_CALLABLE, name, _encode_signature(value))
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    try:
        pickle.dumps(value, protocol = _PICKLE_PROTOCOL)
        return value
    except Exception:  # noqa: BLE001 - live objects (a pipeline, an Event) go back as a ref
        return (_RemoteRef.__name__, name, id(value))


def _error_frame(call_id: int, exc: BaseException) -> tuple:
    try:
        payload = pickle.dumps(exc, protocol = _PICKLE_PROTOCOL)
        pickle.loads(payload)
    except Exception:  # noqa: BLE001
        payload = None
    return ("error", call_id, payload, type(exc).__name__, str(exc), traceback.format_exc())


def serve(target: Any, stdin, stdout) -> None:
    from core.inference.generation_ticket import GenerationTicket, generation_ticket
    from utils.account_context import run_as

    send_lock = threading.Lock()
    tickets: dict[int, GenerationTicket] = {}
    tickets_lock = threading.Lock()
    waiter_cm: dict[int, Any] = {}

    def run(
        op: str,
        call_id: int,
        body: bytes,
        ticket: GenerationTicket,
        waiter: Any = None,
    ) -> None:
        try:
            account, *payload = pickle.loads(body)
            if op == "waiter":
                if isinstance(waiter, BaseException):
                    raise waiter
                waited, ended = waiter
                try:
                    while not (waited.wait(0.05) or ended.is_set()):
                        pass
                finally:
                    waiter_cm.pop(call_id).__exit__(None, None, None)
                value = waited.is_set()
            elif op == "getattr":
                value = _encode_attr(payload[0], getattr(target, payload[0]))
            else:
                method, args, kwargs = payload
                fn = getattr(target, method)

                def invoke() -> Any:
                    with generation_ticket(ticket):
                        return fn(*args, **kwargs)

                value = run_as(account, invoke) if account is not None else invoke()
            frame = ("result", call_id, pickle.dumps(value, protocol = _PICKLE_PROTOCOL))
        except BaseException as exc:  # noqa: BLE001 - every failure goes back to the caller
            frame = _error_frame(call_id, exc)
        finally:
            with tickets_lock:
                tickets.pop(call_id, None)
        _write_frame(stdout, send_lock, frame)

    _write_frame(stdout, send_lock, ("ready", os.getpid()))
    while True:
        try:
            message = _read_frame(stdin)
        except EOFError:
            return
        op = message[0]
        if op == "shutdown":
            return
        if op == "cancel":
            with tickets_lock:
                ticket = tickets.get(message[1])
            if ticket is not None:
                ticket.stop()
            continue
        # Registered before the thread starts, so a Stop sent right behind the call is not lost.
        ticket = GenerationTicket()
        with tickets_lock:
            tickets[message[1]] = ticket
        waiter = None
        if op == "waiter":
            # Entered here, in frame order, so a cancel_generate sent after it already sees this waiter.
            waited, ended = threading.Event(), threading.Event()
            ticket.bind(ended)
            try:
                manager = target.generation_waiter(waited)
                manager.__enter__()
            except Exception as exc:  # noqa: BLE001 - reported on the call, never kills the loop
                waiter = exc
            else:
                waiter_cm[message[1]] = manager
                waiter = (waited, ended)
        threading.Thread(
            target = run,
            args = (*message, ticket, waiter),
            daemon = True,
            name = f"diffusion-worker-{op}",
        ).start()


def main(argv: list[str]) -> None:
    if _BACKEND_PATH not in sys.path:
        sys.path.insert(0, _BACKEND_PATH)
    # The frames own fd 0 and 1; anything a library prints goes to stderr instead.
    stdin = os.fdopen(os.dup(0), "rb", buffering = 0)
    stdout = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    target = _load_target(argv[0] if argv else _DEFAULT_TARGET)
    try:
        serve(target, stdin, stdout)
    finally:
        # Exit without waiting on daemon threads still inside a render; the device is freed with the process.
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
