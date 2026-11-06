# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Token counting abort, API keys, TLS and process timing helpers."""

import contextlib
import functools
import os
import ssl
import sys
import time

from pathlib import Path
from typing import (
    Iterable,
    Mapping,
    Optional,
)
import json


class CountAborted(Exception):
    """A token count stood down mid-flight because its answer stopped mattering."""

    __slots__ = ()


def _count_timeout(messages, system_text, tools) -> float:
    try:
        chars = len(json.dumps([system_text, messages, tools], ensure_ascii = False, default = str))
    except Exception:  # noqa: BLE001 -- an unmeasurable prompt keeps the floor
        chars = 0
    return _COUNT_TIMEOUT_FLOOR_S + chars / _COUNT_TIMEOUT_CHARS_PER_S


_COUNT_TIMEOUT_CHARS_PER_S = 100_000


# Single-threaded render + tokenize ran ~600K chars/s on the #15800 reporter's CPU; a sixth of that still catches a hang.
_COUNT_TIMEOUT_FLOOR_S = 10.0


# Module level so tests can point the orphan sweep at a fixture tree.
_PROC_ROOT = "/proc"


def _metal_capable_host() -> bool:
    """Apple Silicon, where an empty CUDA/HIP probe still means GPU offload. Not
    ``darwin``: auto_detect_backend resolves an Intel Mac to CPU, which wants the line."""
    try:
        from utils.hardware import is_apple_silicon
        return bool(is_apple_silicon())
    except Exception:  # noqa: BLE001 -- a diagnostic must not break a load
        return sys.platform == "darwin"


# Path.expanduser() raises for a deleted user; this never does.
def _expanded_user_path(value) -> Path:
    try:
        from utils.llama_cpp_path_settings import expanded_user_path
        return expanded_user_path(value)
    except Exception:
        return Path(os.path.expanduser(str(value)))


def _llama_server_api_key_enabled() -> bool:
    """UNSLOTH_LLAMA_SERVER_API_KEY=0 opts out, unless UNSLOTH_DIRECT_STREAM=1 needs the key."""
    return (
        os.getenv("UNSLOTH_DIRECT_STREAM", "0") == "1"
        or os.getenv("UNSLOTH_LLAMA_SERVER_API_KEY", "1") != "0"
    )


def _write_direct_stream_key(key: str, previous: "Optional[Path]" = None) -> "Path":
    """0600 key file, one per backend so concurrent launches never swap keys; ``previous`` is rewritten on relaunch."""
    import secrets as _secrets

    from utils.paths.storage_roots import auth_root

    directory = auth_root()
    directory.mkdir(parents = True, exist_ok = True)
    if previous is None:
        # Left by a killed Studio; llama-server reads its key only at startup.
        stale_before = time.time() - 600
        for stale in directory.glob("llama_api_key_*"):
            with contextlib.suppress(OSError):
                if stale.stat().st_mtime < stale_before:
                    stale.unlink()
    path = (
        previous if previous is not None else directory / f"llama_api_key_{_secrets.token_hex(8)}"
    )
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding = "utf-8") as handle:
        handle.write(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def _llama_server_key_via_env() -> bool:
    """Windows with a non-ASCII profile path: llama-server cannot open --api-key-file, so pass it by env."""
    from utils.paths.storage_roots import auth_root
    return sys.platform == "win32" and not str(auth_root()).isascii()


def _llama_server_key_launch(
    key: str, previous: "Optional[Path]" = None
) -> "tuple[list[str], dict[str, str], Optional[Path]]":
    """argv and child env that hand ``key`` to llama-server, plus the key file when one was written."""
    if _llama_server_key_via_env():
        # Like the 0600 file, a process environment is readable only by this user, and it stays off the command line.
        return [], {"LLAMA_API_KEY": key}, None
    path = _write_direct_stream_key(key, previous)
    return ["--api-key-file", str(path)], {}, path


def _extra_args_have_tensor_split(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]]
) -> bool:
    """Presence counts even if unparseable; reads the child env, as the variable is scrubbed earlier."""
    for token in extra_args or ():
        if str(token).split("=", 1)[0] in ("--tensor-split", "-ts", "--tensor_split"):
            return True
    return bool(str((env or {}).get("LLAMA_ARG_TENSOR_SPLIT", "")).strip())


@functools.lru_cache(maxsize = 1)
def _local_ssl_context() -> ssl.SSLContext:
    # httpx loads CA certificates even for loopback HTTP. Share that setup, not streaming sockets.
    return ssl.create_default_context()


# Freeing a large model's GPU memory on a busy driver took over 5 s after SIGKILL; this only bounds a stuck process.
_SIGKILL_REAP_TIMEOUT_S = 30
_QUIT_SIGNAL_EXIT_GRACE_S = 2
