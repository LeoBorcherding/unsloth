# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model merging with MergeKit (LGPL-3.0), run as a subprocess from a venv of its own: it pins
transformers>=5,<6 and peft>=0.18, which Studio's env must not be upgraded to. The venv shares
Studio's torch the way opt-in engine envs do, so no second torch is downloaded."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

MERGEKIT_VERSION = "0.1.5"
# Bump the schema to force a rebuild when the venv layout changes.
_FINGERPRINT = f"mergekit=={MERGEKIT_VERSION}|schema=1"
_MARKER = ".unsloth_mergekit_fingerprint"
# uv drops a dependency overridden with a marker that never holds; Studio's torch is used instead.
_NO_TORCH_OVERRIDE = 'torch ; sys_platform == "never"\n'

# Methods that merge task vectors against a base model.
TASK_METHODS = ("ties", "dare_ties", "task_arithmetic")
METHODS = ("linear", "slerp", *TASK_METHODS)

Log = Callable[[str, str], None]


def venv_dir() -> Path:
    from utils.paths.storage_roots import studio_root
    return studio_root() / ".venv_mergekit"


def venv_python(root: Optional[Path] = None) -> Path:
    root = root or venv_dir()
    return root / "Scripts" / "python.exe" if os.name == "nt" else root / "bin" / "python"


def _venv_site(root: Path) -> Path:
    if os.name == "nt":
        return root / "Lib" / "site-packages"
    return root / "lib" / "python{}.{}".format(*sys.version_info[:2]) / "site-packages"


def is_installed() -> bool:
    root = venv_dir()
    try:
        marker = (root / _MARKER).read_text(encoding = "utf-8").strip()
    except OSError:
        return False
    return marker == _FINGERPRINT and venv_python(root).is_file()


def probe() -> dict:
    """Whether a merge can run now, or what installing MergeKit first would do."""
    from utils.transformers_version import _env_offline

    ready = is_installed()
    blocked = None
    if not ready and shutil.which("uv") is None:
        blocked = "uv is unavailable. Repair the Unsloth installation and retry."
    elif not ready and _env_offline():
        blocked = "MergeKit is not installed and Hugging Face offline mode is set."
    return {
        "ready": ready,
        "needs_consent": not ready and blocked is None,
        "install_summary": None
        if ready
        else (
            f"Unsloth will install MergeKit {MERGEKIT_VERSION} (LGPL-3.0) into its own environment "
            f"at {venv_dir()}. It uses your PyTorch; this Unsloth Studio environment is not changed."
        ),
        "blocked_reason": blocked,
        "venv_path": str(venv_dir()),
        "version": MERGEKIT_VERSION,
    }


def build_config(spec: dict) -> dict:
    """MergeKit's YAML config, as a dict, from a validated MergeModelsRequest dump."""
    method = spec["method"]
    base = spec.get("base_model")
    entries = spec["models"]
    config: dict = {}
    if any(entry.get("layer_range") for entry in entries):
        config["slices"] = [
            {
                "sources": [
                    {"model": entry["model"], "layer_range": list(entry["layer_range"])}
                    for entry in entries
                ]
            }
        ]
    else:
        models = []
        for entry in entries:
            parameters = {
                key: entry[key] for key in ("density", "weight") if entry.get(key) is not None
            }
            models.append(
                {"model": entry["model"], **({"parameters": parameters} if parameters else {})}
            )
        if method in TASK_METHODS:
            models.insert(0, {"model": base})
        config["models"] = models
    config["merge_method"] = method
    if base and method != "linear":
        config["base_model"] = base
    parameters: dict = {}
    t = spec.get("t")
    if isinstance(t, list):
        parameters["t"] = [
            {key: value for key, value in step.items() if value is not None} for step in t
        ]
    elif t is not None:
        parameters["t"] = t
    for key in ("normalize", "int8_mask"):
        if spec.get(key) is not None:
            parameters[key] = spec[key]
    if parameters:
        config["parameters"] = parameters
    if spec.get("tokenizer_source"):
        config["tokenizer_source"] = spec["tokenizer_source"]
    config["dtype"] = spec["dtype"]
    return config


def config_yaml(spec: dict) -> str:
    import yaml
    return yaml.safe_dump(build_config(spec), sort_keys = False)


def merge_command(
    config_path: Path,
    out_dir: Path,
    *,
    cuda: bool = False,
    random_seed: Optional[int] = None,
    python: Optional[Path] = None,
) -> list[str]:
    """mergekit-yaml, through the venv's interpreter so a cancel stops the merge itself."""
    argv = [
        str(python or venv_python()),
        "-I",
        "-m",
        "mergekit.scripts.run_yaml",
        str(config_path),
        str(out_dir),
        "--lazy-unpickle",
        "--out-shard-size",
        "5B",
        "--copy-tokenizer",
    ]
    if cuda:
        argv.append("--cuda")
    if random_seed is not None:
        argv += ["--random-seed", str(random_seed)]
    return argv


def child_env(hf_token = None) -> dict[str, str]:
    from hub.utils.hf_tokens import apply_token_to_child_env
    from utils.child_stdio import utf8_child_env
    from utils.hf_cache_settings import get_hf_cache_paths
    from utils.native_path_leases import child_env_without_native_path_secret

    env = utf8_child_env(get_hf_cache_paths().child_env(child_env_without_native_path_secret()))
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    apply_token_to_child_env(env, hf_token)
    # tqdm redraws become log lines; one a second is plenty.
    env["TQDM_MININTERVAL"] = "1"
    return env


def _stream(
    argv: list[str], env: dict[str, str], log: Log, on_spawn: Callable[[subprocess.Popen], None]
) -> int:
    from utils.process_lifetime import child_popen_kwargs, spawn_on_lifetime_thread
    from utils.subprocess_compat import windows_hidden_subprocess_kwargs

    # Dies with Studio; text mode turns tqdm's carriage returns into lines.
    process = spawn_on_lifetime_thread(
        lambda: subprocess.Popen(
            argv,
            env = env,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            **windows_hidden_subprocess_kwargs(),
            **child_popen_kwargs(),
        )
    )
    on_spawn(process)
    for line in process.stdout:
        if line.strip():
            log(line.rstrip(), "stdout")
    return process.wait()


def install(
    log: Log,
    on_spawn: Callable[[subprocess.Popen], None],
    cancelled: Callable[[], bool] = lambda: False,
) -> None:
    """Create the MergeKit venv; on any failure it is removed, so a retry starts clean."""
    from core.inference import engine_install

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is unavailable. Repair the Unsloth installation and retry.")
    root = venv_dir()
    if root.is_symlink():
        raise RuntimeError("The MergeKit directory must not be a symbolic link.")
    shutil.rmtree(root, ignore_errors = True)
    env = child_env()
    shared = "torch" in engine_install._studio_packages()

    def run(argv: list[str], failure: str) -> None:
        if _stream(argv, env, log, on_spawn) != 0 or cancelled():
            raise RuntimeError("MergeKit installation cancelled." if cancelled() else failure)

    try:
        log(f"Installing MergeKit {MERGEKIT_VERSION} into {root}", "status")
        run(
            [uv, "venv", "--python", sys.executable, str(root)],
            "Could not create MergeKit's environment.",
        )
        argv = [uv, "pip", "install", "--python", str(venv_python(root))]
        if shared:
            overrides = root / "unsloth-overrides.txt"
            overrides.write_text(_NO_TORCH_OVERRIDE, encoding = "utf-8")
            argv += ["--override", str(overrides)]
        run([*argv, f"mergekit=={MERGEKIT_VERSION}"], "MergeKit installation failed.")
        if shared:
            # Appended after the venv's own site-packages, so MergeKit's transformers wins.
            site = _venv_site(root)
            (site / f"{engine_install._BASE_MODULE}.py").write_text(
                engine_install._STUDIO_BASE_SOURCE.format(paths = engine_install._studio_site()),
                encoding = "utf-8",
            )
            (site / engine_install._BASE_PTH).write_text(
                f"import {engine_install._BASE_MODULE}\n", encoding = "utf-8"
            )
        run(
            [str(venv_python(root)), "-I", "-c", "import torch, mergekit.merge"],
            "MergeKit was installed but cannot import.",
        )
        (root / _MARKER).write_text(_FINGERPRINT, encoding = "utf-8")
        log("MergeKit is ready", "status")
    except BaseException:
        shutil.rmtree(root, ignore_errors = True)
        raise


def merge(
    spec: dict,
    out_dir: Path,
    *,
    log: Log,
    on_spawn: Callable[[subprocess.Popen], None],
    hf_token = None,
    install_missing: bool = False,
    cancelled: Callable[[], bool] = lambda: False,
) -> Path:
    """Run the merge into out_dir, which must not exist yet; it appears only once complete."""
    if not is_installed():
        if not install_missing:
            raise RuntimeError("MergeKit is not installed. Confirm its installation to merge.")
        install(log, on_spawn, cancelled)
    out_dir = Path(out_dir)
    if out_dir.exists() or out_dir.is_symlink():
        raise RuntimeError(f"{out_dir} already exists; choose another output name.")
    out_dir.parent.mkdir(parents = True, exist_ok = True)
    # Beside the outputs folder, not in it: the checkpoint scan would list a half-written merge.
    staging = Path(tempfile.mkdtemp(prefix = ".mergekit-", dir = out_dir.parent.parent))
    try:
        config_path = staging / "mergekit_config.yml"
        config_path.write_text(config_yaml(spec), encoding = "utf-8")
        log(f"Merging {len(spec['models'])} models with {spec['method']}", "status")
        argv = merge_command(
            config_path,
            staging / "model",
            cuda = bool(spec.get("cuda")),
            random_seed = spec.get("random_seed"),
        )
        code = _stream(argv, child_env(hf_token), log, on_spawn)
        if cancelled():
            raise RuntimeError("Merge cancelled.")
        if code != 0:
            raise RuntimeError(f"MergeKit exited with code {code}; see the log for details.")
        if not (staging / "model" / "config.json").is_file():
            raise RuntimeError("MergeKit finished without writing config.json.")
        os.replace(staging / "model", out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors = True)
    log(f"Merged model saved to {out_dir}", "status")
    return out_dir
