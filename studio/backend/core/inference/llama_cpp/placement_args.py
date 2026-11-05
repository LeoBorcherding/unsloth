# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Paravirtual Metal placement, context checkpoints and projector offload."""

import functools
import os
import subprocess
import sys

from typing import (
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
)
from core.inference.llama_server_args import (
    _flag_name,
    effective_ctx_checkpoints,
    LLAMA_CTX_CHECKPOINTS_DEFAULT,
    parse_split_mode_override,
    strip_shadowing_flags,
)

from .launch_flags import (
    _LLAMA_ARG_FALSE_VALUES,
    _LLAMA_ARG_TRUE_VALUES,
    _env_ctx_checkpoints_override,
)
from .log import logger
from .memory_policy import _extra_args_set_any_flag
from core.inference.llama_server_args import _last_flag_value
from core.inference.llama_video_input import video_tools
import shutil


@functools.lru_cache(maxsize = 1)
def _metal_device_is_paravirtual() -> bool:
    """Virtualised Apple GPU can corrupt offload; UNSLOTH_ALLOW_PARAVIRTUAL_METAL=1 overrides. Probe hw.model before SPDisplaysDataType (empty on headless VMs)."""
    if sys.platform != "darwin":
        return False
    if os.environ.get("UNSLOTH_ALLOW_PARAVIRTUAL_METAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        logger.info("UNSLOTH_ALLOW_PARAVIRTUAL_METAL set: keeping Metal offload.")
        return False
    name = ""
    try:
        import mlx.core as mx
        name = str(mx.device_info().get("device_name") or "")
    except Exception:
        name = ""
    if "paravirtual" not in name.lower():
        for _probe_cmd, _match in (
            # VMM-x86_64 is the Intel-guest identifier; it contains neither virtual nor paravirtual, so it needs its own match.
            (["sysctl", "-n", "hw.model"], ("virtual", "vmm-")),
            (["system_profiler", "SPDisplaysDataType"], ("paravirtual",)),
        ):
            try:
                probe = subprocess.run(
                    _probe_cmd,
                    capture_output = True,
                    text = True,
                    timeout = 30,
                    encoding = "utf-8",
                    errors = "replace",
                )
            except Exception:
                continue
            _out = (probe.stdout or "").lower()
            if any(m in _out for m in _match):
                name = f"{name} {probe.stdout}".strip()
                break
    _low = name.lower()
    if "paravirtual" in _low or "virtual" in _low or "vmm-" in _low:
        logger.warning(
            "Metal device looks virtualised (%s). Offload can return corrupt output on "
            "paravirtual Apple GPUs, so GGUF inference will run on CPU. MLX is "
            "unaffected. Set UNSLOTH_ALLOW_PARAVIRTUAL_METAL=1 to keep the GPU.",
            name.strip()[:120] or "unknown",
        )
        return True
    return False


def _paravirtual_probe_answered(server_caps: Mapping[str, object]) -> bool:
    """An unanswered --help probe reads as unknown, not absent: a malformed LLAMA_ARG_* can break it."""
    return not server_caps.get("mtp_probe_inconclusive")


def _paravirtual_draft_ngl_flag(server_caps: Mapping[str, object]) -> Optional[str]:
    """The draft-layer flag to pin with, or None only when the build truly lacks one."""
    flag = server_caps.get("spec_draft_ngl_flag")
    if flag:
        return str(flag)
    # With the probe unanswered, use the --gpu-layers-draft spelling every launchable build has had.
    if not _paravirtual_probe_answered(server_caps):
        return "--gpu-layers-draft"
    return None


def _argv_keeps_projector_on_gpu(
    args: Sequence[str], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether *args* still load a projector onto the device. arg.cpp applies
    ``LLAMA_ARG_MMPROJ(_URL)`` before argv, so the env alone still loads one."""
    tokens = [str(a) for a in args]
    env_map = env or {}

    def _flag(token: str) -> str:
        return token.split("=", 1)[0].replace("_", "-").lower()

    from_argv = any(_flag(a) in ("--mmproj", "-mm") for a in tokens)
    from_env = any(
        str(env_map.get(name) or "").strip()
        for name in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL")
    )
    if not from_argv and not from_env:
        return False
    return _resolved_mmproj_offload(tokens, env_map) is not False


def _paravirtual_mmproj_pinnable(server_caps: Mapping[str, object]) -> bool:
    """True when --no-mmproj-offload can be emitted, including on an unanswered probe."""
    return bool(
        server_caps.get("supports_no_mmproj_offload")
        or not _paravirtual_probe_answered(server_caps)
    )


def ctx_checkpoints_allocated(server_caps: Mapping[str, object]) -> bool:
    """Silence is not absence: false only when a parsed help listing confirms no checkpoint flag."""
    if server_caps.get("ctx_checkpoints_flag"):
        return True
    if not server_caps.get("help_probe_ok"):
        return True
    return not bool(server_caps.get("flags"))


def ctx_checkpoints_default_for_caps(server_caps: Mapping[str, object]) -> int:
    """The count this build keeps unflagged: what it advertised, else upstream's today."""
    advertised = server_caps.get("ctx_checkpoints_default")
    if isinstance(advertised, int) and advertised >= 0:
        return advertised
    return LLAMA_CTX_CHECKPOINTS_DEFAULT


def effective_ctx_checkpoints_for_caps(
    server_caps: Mapping[str, object],
    extra_args: Optional[Iterable[str]],
    requested: Optional[int],
    *,
    per_checkpoint_bytes: int = 0,
    n_parallel: int = 1,
    total_host_bytes: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Unsupported builds get zero; without a confirmed alias only pass-through extras can override."""
    if not ctx_checkpoints_allocated(server_caps):
        return 0
    emittable = bool(server_caps.get("ctx_checkpoints_flag"))
    describes_a_child = bool(server_caps.get("found"))
    return effective_ctx_checkpoints(
        extra_args,
        requested if (emittable or not describes_a_child) else None,
        supports_flag = True,
        per_checkpoint_bytes = per_checkpoint_bytes,
        n_parallel = n_parallel,
        total_host_bytes = total_host_bytes if emittable else None,
        upstream_default = ctx_checkpoints_default_for_caps(server_caps),
        inherited = _env_ctx_checkpoints_override(env),
    )


def _idle_slot_clearing_active(cmd: Sequence[str], *, supports_cache_ram: bool) -> bool:
    """Old binaries without --cache-ram never clear idle slots; an absent flag keeps the default-on clearing; --cache-ram 0 disables it; last-wins."""
    enabled = bool(supports_cache_ram)
    for i, arg in enumerate(cmd):
        if arg == "--cache-ram":
            # Missing or unreadable --cache-ram value reads as no clearing, not the default; the conservative answer.
            try:
                enabled = int(cmd[i + 1]) > 0
            except (IndexError, TypeError, ValueError):
                enabled = False
        elif arg == "--cache-idle-slots":
            enabled = True
        elif arg == "--no-cache-idle-slots":
            enabled = False
    return enabled


def _mmproj_env_is_audio_only(path: Optional[str]) -> bool:
    """Anything unknown counts as not audio-only and is cleared, the safer error of the two."""
    if not path:
        return False
    try:
        from utils.models.gguf_metadata import mmproj_capabilities
        has_audio, accepts_image = mmproj_capabilities(str(path))
    except Exception as e:
        logger.debug(f"inherited mmproj capability read failed: {e}")
        return False
    return has_audio and not accepts_image


# llama.cpp takes the last of either spelling; naming one means the user owns placement.
_MMPROJ_OFFLOAD_FLAGS: frozenset = frozenset({"--mmproj-offload", "--no-mmproj-offload"})


def _extra_args_set_mmproj_offload(extra_args: Optional[Sequence[str]]) -> bool:
    """Matched via _flag_name: arg.cpp folds underscores to dashes, so --mmproj_offload counts too."""
    return _extra_args_set_any_flag(extra_args, _MMPROJ_OFFLOAD_FLAGS)


def _extra_args_mmproj_offload_value(extra_args: Optional[Sequence[str]]) -> Optional[bool]:
    """Last occurrence wins across both spellings; --no-mmproj-offload keeps the projector in host RAM."""
    if not extra_args:
        return None
    value: Optional[bool] = None
    for raw in extra_args:
        flag = _flag_name(str(raw))
        if flag == "--mmproj-offload":
            value = True
        elif flag == "--no-mmproj-offload":
            value = False
    return value


# Env applies before argv; LLAMA_ARG_NO_ form wins on presence, even empty.
_MMPROJ_OFFLOAD_ENV_VAR = "LLAMA_ARG_MMPROJ_OFFLOAD"
_MMPROJ_OFFLOAD_NEG_ENV_VAR = "LLAMA_ARG_NO_MMPROJ_OFFLOAD"

# Gemma 4 expects video at up to 1 fps; llama-server defaults to 4.
_LLAMA_VIDEO_FPS = "1"
_VIDEO_FPS_ENV_VAR = "LLAMA_ARG_VIDEO_FPS"


_VIDEO_FFMPEG_DIR_ENV_VAR = "LLAMA_ARG_VIDEO_FFMPEG_DIR"


_VIDEO_TIMESTAMP_ENV_VAR = "LLAMA_ARG_VIDEO_TIMESTAMP_INTERVAL"


_LLAMA_SERVER_DEFAULT_VIDEO_FPS = 4.0


def _video_fps_flags(
    server_caps: Mapping[str, object],
    env: Optional[Mapping[str, str]] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> list[str]:
    """Return the default FPS and timestamp flags unless unsupported or set by the user.

    User extras are appended later and retain their last-wins override.
    """
    source = os.environ if env is None else env
    flags: list[str] = []
    if server_caps.get("supports_video_fps") and source.get(_VIDEO_FPS_ENV_VAR) is None:
        flags += ["--video-fps", _LLAMA_VIDEO_FPS]
    if (
        server_caps.get("supports_video_timestamp_interval")
        and source.get(_VIDEO_TIMESTAMP_ENV_VAR) is None
        and not _extra_args_set_any_flag(extra_args, {"--video-timestamp-interval"})
    ):
        # llama-server writes a timestamp only every 5000 ms by default, so the model snaps
        # every event to a multiple of 5 s. One label per 2 sampled frames: Qwen-VL merges
        # frame pairs and a label between them splits the pair (about 1.7x the prompt), and
        # Gemma 4 placed events best at this spacing. Floor, so drift never skips a label.
        fps = requested_video_fps(extra_args, source)
        if fps is None:
            fps = (
                float(_LLAMA_VIDEO_FPS)
                if server_caps.get("supports_video_fps")
                else _LLAMA_SERVER_DEFAULT_VIDEO_FPS
            )
        flags += ["--video-timestamp-interval", str(max(1, int(2000 / fps)))]
    return flags


def _video_ffmpeg_dir_flags(
    server_caps: Mapping[str, object],
    env: Optional[Mapping[str, str]] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> tuple[list[str], Optional[str]]:
    if not server_caps.get("supports_video_ffmpeg_dir"):
        return [], None
    source = os.environ if env is None else env
    try:
        chosen = _last_flag_value(extra_args, frozenset({"--video-ffmpeg-dir"}), preserve_raw = True)
    except ValueError:
        chosen = None
    chosen = chosen or source.get(_VIDEO_FFMPEG_DIR_ENV_VAR) or None
    if chosen:
        return [], chosen
    # CreateProcess searches the server's directory before PATH.
    if sys.platform == "win32":
        return [], None
    tools = video_tools()
    if tools is None:
        return [], None
    ffmpeg_dir = os.path.dirname(tools[0])
    if os.path.dirname(tools[1]) != ffmpeg_dir:
        return [], None
    return ["--video-ffmpeg-dir", ffmpeg_dir], ffmpeg_dir


def requested_video_fps(
    extra_args: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None
) -> Optional[float]:
    """Argv beats LLAMA_ARG_VIDEO_FPS and the last occurrence wins; underscore spellings such as --video_fps count too; preprocessing must see the rate."""
    args = [str(a) for a in (extra_args or ())]
    raw = None
    for i in range(len(args) - 1, -1, -1):
        if _flag_name(args[i]) != "--video-fps":
            continue
        raw = (
            args[i].split("=", 1)[1]
            if "=" in args[i]
            else (args[i + 1] if i + 1 < len(args) else None)
        )
        break
    if raw is None:
        source = os.environ if env is None else env
        raw = source.get(_VIDEO_FPS_ENV_VAR)
    try:
        rate = float(raw)
    except (TypeError, ValueError):
        return None
    return rate if rate > 0 else None


def _video_tools_reachable(
    cmd: Sequence[str],
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[str] = None,
    supports_ffmpeg_dir: bool = True,
) -> bool:
    """Whether the child can launch ffmpeg and ffprobe, which /props' video modality never checks (#16065)."""
    try:
        search = _video_tools_search(cmd, env, cwd, supports_ffmpeg_dir)
    except Exception:
        return True
    names = ("ffmpeg.exe", "ffprobe.exe") if sys.platform == "win32" else ("ffmpeg", "ffprobe")
    return all(shutil.which(name, path = search) for name in names)


def _video_tools_search(
    cmd: Sequence[str],
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[str] = None,
    supports_ffmpeg_dir: bool = True,
) -> str:
    source = os.environ if env is None else env
    bin_dir = _last_flag_value(cmd, frozenset({"--video-ffmpeg-dir"}), preserve_raw = True)
    search = bin_dir or (source.get(_VIDEO_FFMPEG_DIR_ENV_VAR) if supports_ffmpeg_dir else None)
    if not search:
        search = source.get("PATH", os.defpath)
        if sys.platform == "win32":
            launch_dir = os.path.dirname(str(cmd[0])) if cmd else ""
            search = os.pathsep.join(d for d in (launch_dir, cwd or os.getcwd(), search) if d)
    return search


def _env_sets_mmproj_offload(env: Optional[Mapping[str, str]] = None) -> bool:
    """Presence, not value, decides: LLAMA_ARG_NO_MMPROJ_OFFLOAD= counts, as llama.cpp tests with getenv."""
    source = os.environ if env is None else env
    return any(
        source.get(name) is not None
        for name in (_MMPROJ_OFFLOAD_ENV_VAR, _MMPROJ_OFFLOAD_NEG_ENV_VAR)
    )


def _env_mmproj_offload_value(env: Optional[Mapping[str, str]] = None) -> Optional[bool]:
    """Negative spelling wins on presence; an unrecognised positive value is None (arg.cpp raises)."""
    source = os.environ if env is None else env
    if source.get(_MMPROJ_OFFLOAD_NEG_ENV_VAR) is not None:
        return False
    raw = source.get(_MMPROJ_OFFLOAD_ENV_VAR)
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in _LLAMA_ARG_TRUE_VALUES:
        return True
    if value in _LLAMA_ARG_FALSE_VALUES:
        return False
    return None


def _resolved_mmproj_offload(
    extra_args: Optional[Sequence[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[bool]:
    """Argv overrides the inherited environment here, and only silence in both gives None."""
    from_args = _extra_args_mmproj_offload_value(extra_args)
    return from_args if from_args is not None else _env_mmproj_offload_value(env)


_OVERRIDE_TENSOR_FLAGS: frozenset = frozenset(
    {
        "-ot",
        "--override-tensor",
        "-otd",
        "--override-tensor-draft",
        "--spec-draft-override-tensor",
    }
)


def _paravirtual_strip_gpu_overrides(
    extra_args: Optional[Iterable[str]], *, log_dropped: bool = True
) -> Optional[List[str]]:
    """Drops non-CPU -ot/-otd entries, which place weights on the corrupt device despite --gpu-layers 0; log_dropped=False for comparators."""
    if not extra_args:
        return extra_args if extra_args is None else list(extra_args)
    args = [str(a) for a in extra_args]
    out: List[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        # llama.cpp folds _ to -, so --override_tensor is the same flag.
        base, _, inline = tok.partition("=")
        if _flag_name(base) in _OVERRIDE_TENSOR_FLAGS:
            value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
            targets = [
                part.rsplit("=", 1)[-1].strip().lower() for part in value.split(",") if "=" in part
            ]
            if targets and all(t == "cpu" for t in targets):
                out.append(tok)
                if not inline and i + 1 < len(args):
                    out.append(args[i + 1])
            elif log_dropped:
                logger.warning(
                    "Dropping %s %s: this Mac's Metal device is virtualised, so an "
                    "override that keeps weights on it would return corrupt output.",
                    base,
                    value,
                )
            i += 1 if inline else 2
            continue
        out.append(tok)
        i += 1
    return out


class ParavirtualPlacement(NamedTuple):
    """A load request's GPU placement after the virtualised-Metal CPU pin."""

    gpu_memory_mode: Literal["auto", "manual"]
    gpu_layers: int
    tensor_parallel: bool
    tensor_split: Optional[List[float]]
    n_cpu_moe: int
    extra_args: Optional[List[str]]


def paravirtual_normalized_request(
    *,
    gpu_memory_mode: Literal["auto", "manual"] = "auto",
    gpu_layers: int = -1,
    tensor_parallel: bool = False,
    tensor_split: Optional[List[float]] = None,
    n_cpu_moe: int = 0,
    extra_args: Optional[Iterable[str]] = None,
    log_dropped: bool = True,
) -> ParavirtualPlacement:
    """Shared by launch and both duplicate-load comparators, which must see the normalized request or they tear down a healthy CPU server on each Apply."""
    normalized = extra_args if extra_args is None else list(extra_args)
    if normalized:
        # An extra -ngl re-enables the corrupt offload; only manual mode strips extras, Auto keeps them.
        normalized = strip_shadowing_flags(
            normalized,
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = False,
            strip_offload = True,
        )
        # strip_offload misses --override-tensor aimed at Metal.
        normalized = _paravirtual_strip_gpu_overrides(normalized, log_dropped = log_dropped)
    return ParavirtualPlacement(
        gpu_memory_mode = "manual",
        gpu_layers = 0,
        # No GPU layers to split, and the launch drops these anyway; normalize so the recorded state matches.
        tensor_parallel = False,
        tensor_split = None,
        n_cpu_moe = 0,
        extra_args = normalized,
    )


def _paravirtual_split_mode_pin(extra_args: Optional[Iterable[str]]) -> List[str]:
    """Pins -sm to layer; -sm row or tensor aborts startup, and stripping would turn Apply into a swap."""
    mode = (parse_split_mode_override(extra_args) or "").strip().lower()
    if not mode or mode == "layer":
        return []
    logger.warning(
        "Overriding --split-mode %s with the default layer split: this Mac's "
        "Metal device is virtualised, so the model runs on CPU and a GPU split "
        "mode would only fail the load.",
        mode,
    )
    return ["--split-mode", "layer"]
