# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: binary discovery."""

import os
import shutil
import sys
import time

from pathlib import Path
from typing import Optional

from .launch_support import _expanded_user_path
from .log import logger

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _BinaryDiscoveryMixin:
    # ── Binary discovery ──────────────────────────────────────────

    @staticmethod
    def _resolved_studio_root_and_is_legacy() -> "tuple[Optional[Path], bool]":
        """On failure the root counts as legacy, and resolved_root is None, so read it only when not
        legacy."""
        try:
            from utils.paths.storage_roots import (  # noqa: WPS433
                studio_root as _sr,
                unsloth_home as _uh,
            )

            # llama.cpp sits beside studio/ under the master root that run.py and main.py export; not legacy.
            master = _uh()
            if master is not None:
                return master, False
            resolved = _sr()
            legacy_studio = Path.home() / ".unsloth" / "studio"
            try:
                is_legacy = resolved.resolve() == legacy_studio.resolve()
            except (OSError, ValueError):
                is_legacy = resolved == legacy_studio
            return (None if is_legacy else resolved), is_legacy
        except (ImportError, OSError, ValueError):
            return None, True

    @staticmethod
    def _find_llama_server_binary(*, include_denied: bool = False) -> Optional[str]:
        """Env LLAMA_SERVER_PATH, UNSLOTH_LLAMA_CPP_PATH, Unsloth folder setting, ~/.unsloth/llama.cpp, legacy ./llama.cpp, PATH, then legacy ./bin."""
        binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"

        def _file_status(p: Path) -> str:
            # is_file() raises PermissionError when locked (Windows AV/ACL); never treat as missing.
            for _ in range(5):
                try:
                    if not p.is_file():
                        return "absent"
                    if sys.platform != "win32" and not os.access(p, os.X_OK):
                        return "non_executable"
                    return "file"
                except PermissionError:
                    time.sleep(0.2)
                except OSError:
                    return "absent"
            return "denied"

        def _is_file(p: Path) -> bool:
            return _file_status(p) == "file"

        def _layout_candidates(d: Path) -> list:
            # Settings validation and discovery share one layout contract, so any folder the UI accepts launches.
            from utils.llama_cpp_path_settings import llama_server_candidates
            return list(llama_server_candidates(d))

        def _unavailable(p: object) -> None:
            # Report an access-denied pinned binary instead of silently downgrading.
            logger.warning(
                f"llama-server at {p} is not executable or is access-denied "
                "(permissions, antivirus, or an in-flight install); not falling "
                "back to another binary, retry once it is available"
            )
            return None

        def _scan_pinned(paths: list):
            # First executable wins; access-denied stops, non-executable keeps looking; include_denied returns the unavailable path (diffusion only needs its dir).
            non_executable = None
            from utils.llama_cpp_path_settings import prefer_gpu_capable

            statuses = {p: _file_status(p) for p in paths}
            first = next((p for p in paths if statuses[p] == "file"), None)
            # A CPU-only first hit yields to a GPU-capable sibling, unless a denied one precedes it.
            if first is not None and not any(
                statuses[p] == "denied" for p in paths[: paths.index(first)]
            ):
                paths = prefer_gpu_capable(paths, lambda p: statuses[p] == "file")
            for p in paths:
                st = statuses[p]
                if st == "file":
                    return str(p), None
                if st == "denied":
                    return (str(p), None) if include_denied else (None, p)
                if st == "non_executable" and non_executable is None:
                    # A source checkout may leave a stale root entrypoint beside a valid CMake build; keep looking in this pinned layout, never fall through to another runtime.
                    non_executable = p
            if non_executable is not None:
                # A missing execute bit needs repair, not retry, so it must not pass preflight.
                return (None, None) if include_denied else (None, non_executable)
            return None, None

        # 1. Env var: direct path to binary
        env_path = os.environ.get("LLAMA_SERVER_PATH")
        if env_path:
            hit, locked = _scan_pinned([Path(env_path)])
            if locked is not None:
                return _unavailable(locked)
            if hit:
                return hit

        # 1b. UNSLOTH_LLAMA_CPP_PATH: custom llama.cpp install dir
        custom_llama_cpp = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
        managed_path_marker = os.environ.get("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH") == "1"
        if custom_llama_cpp and not managed_path_marker:
            # Expand ~ like every other reader; not Path.expanduser, which raises on unknown users.
            hit, locked = _scan_pinned(_layout_candidates(_expanded_user_path(custom_llama_cpp)))
            if locked is not None:
                return _unavailable(locked)
            if hit:
                return hit

        # 2. Unsloth setting: a deliberate pin, so never silently fall through to the bundle.
        try:
            from utils.llama_cpp_path_settings import get_stored_custom_llama_cpp_path
            studio_custom_llama_cpp = get_stored_custom_llama_cpp_path()
        except Exception:
            studio_custom_llama_cpp = None
        if studio_custom_llama_cpp is not None:
            hit, locked = _scan_pinned(_layout_candidates(studio_custom_llama_cpp))
            if locked is not None:
                return _unavailable(locked)
            if hit:
                return hit
            logger.warning(
                "The custom llama.cpp folder selected in Unsloth no longer contains "
                "%s; not falling back to another runtime",
                binary_name,
            )
            return None

        # 3-5. Env mode uses $STUDIO_HOME/llama.cpp; default or HOME-redirect uses ~/.unsloth/llama.cpp, the sibling of studio.
        legacy_llama = Path.home() / ".unsloth" / "llama.cpp"
        _resolved_sr, _is_legacy = LlamaCppBackend._resolved_studio_root_and_is_legacy()
        if _is_legacy:
            search_roots = [legacy_llama]
        else:
            # Must match _kill_orphaned_servers (excludes legacy root in custom mode), so we never spawn a server we then refuse to clean up.
            search_roots = [_resolved_sr / "llama.cpp"]
        for unsloth_home in search_roots:
            hit, locked = _scan_pinned(_layout_candidates(unsloth_home))
            if locked is not None:
                return _unavailable(locked)
            if hit:
                return hit

        # 5-6. Legacy in-tree build; a fallback, so a denied candidate just continues.
        project_root = Path(__file__).resolve().parents[5]
        from utils.llama_cpp_path_settings import prefer_gpu_capable

        for p in prefer_gpu_capable(_layout_candidates(project_root / "llama.cpp"), _is_file):
            if _is_file(p):
                return str(p)

        # 7. System PATH
        system_path = shutil.which("llama-server")
        if system_path:
            return system_path

        # 8. Legacy: extracted to bin/
        bin_path = project_root / "bin" / binary_name
        if _is_file(bin_path):
            return str(bin_path)

        return None
