# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: llama-server capability probe."""

import re
import subprocess
import sys
import threading
import time

from pathlib import Path
from typing import (
    Iterable,
    Optional,
)
from core.inference.llama_server_args import (
    parse_reasoning_budget_message_override,
    parse_reasoning_budget_override,
    resolve_reasoning_budget,
    resolve_reasoning_budget_message,
)
from utils.code_integrity import code_integrity_block_reason
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

from .log import logger
from .server_log import _TRACE_VERBOSITY_HELP_RE
from .launch_flags import (
    _fit_measures_borrowed_drafts,
)


class _CapabilityProbeMixin:
    # ── llama-server capability probe ─────────────────────────────

    # ── llama-server capability probe ─────────────────────────────
    # b10259..b10268 advertise draft-dspark but abort on load; match the base build number, gate only managed prebuilts (source builds carry no marker).
    _BROKEN_DSPARK_BUILDS = range(10259, 10269)

    @classmethod
    def _dspark_release_is_broken(cls, release_tag: Optional[str]) -> bool:
        match = re.match(r"b(\d+)", str(release_tag or ""))
        return bool(match) and int(match.group(1)) in cls._BROKEN_DSPARK_BUILDS

    # Keyed on (mtime_ns, size): updates replace the binary in place within the same second.
    _CAPABILITY_PROBE_RETRY_SECONDS = 30.0
    _CAPABILITY_PROBE_RETRY_MAX_SECONDS = 600.0
    _capability_cache: dict[tuple[str, int, int], dict[str, object]] = {}
    # (bin_path, mtime_ns, size) -> build number from `llama-server --version`, or None.
    _build_number_cache: dict[tuple[str, int, int], Optional[int]] = {}
    _capability_retry_after: dict[tuple[str, int, int], float] = {}
    _capability_retry_backoff: dict[tuple[str, int, int], float] = {}
    _capability_cache_lock = threading.Lock()

    # Older builds take a bare boolean -fa and exit on a following "on".
    _FLASH_ATTN_ENUM_RE = re.compile(
        r"(?im)^[^\n]*--flash-attn[^\n]*\bon\b[ \t]*\|[ \t]*\boff\b[^\n]*$"
    )

    @classmethod
    def _flash_attn_takes_value(cls, help_text: str) -> bool:
        """Fails open: a help text without --flash-attn keeps the value form, which the pinned
        prebuilt uses."""
        if not help_text or "--flash-attn" not in help_text:
            return True
        return bool(cls._FLASH_ATTN_ENUM_RE.search(help_text))

    # llama.cpp#23861 capped reserved output rows; older builds reserve one per ubatch token.
    _OUTPUT_ROWS_CAPPED_BUILD = 9460

    @classmethod
    def probe_build_number(cls, binary: Optional[str] = None) -> Optional[int]:
        """None for a missing binary, a failed probe, or build 1, which is a source build without tags."""
        bin_path = cls._exec_path_for_launch(binary or cls._find_llama_server_binary())
        if not bin_path or not Path(bin_path).is_file():
            return None
        try:
            binary_stat = Path(bin_path).stat()
            cache_key = (bin_path, binary_stat.st_mtime_ns, binary_stat.st_size)
        except OSError:
            cache_key = (bin_path, 0, 0)
        with cls._capability_cache_lock:
            if cache_key in cls._build_number_cache:
                return cls._build_number_cache[cache_key]
        build: Optional[int] = None
        try:
            env = cls._llama_server_env_for_binary(bin_path)
            if sys.platform == "darwin":
                env["GGML_METAL_DEVICES"] = "0"  # see probe_server_capabilities
            result = subprocess.run(
                [bin_path, "--version"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                check = False,
                env = env,
                **_windows_hidden_subprocess_kwargs(),
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            match = re.search(r"version:[^\r\n]*\bbuild\s+(\d+)\b", output) or re.search(
                r"version:\s*(\d+)\b(?!\.)", output
            )
            if match and int(match.group(1)) > 1:
                build = int(match.group(1))
        except Exception as exc:
            logger.debug("llama-server --version probe failed: %s", exc)
        with cls._capability_cache_lock:
            cls._build_number_cache[cache_key] = build
        return build

    @classmethod
    def reserves_micro_batch_outputs(cls, binary: Optional[str] = None) -> bool:
        """Builds before b9460, and the b9415 macOS fallback, reserve rows; unknown builds count as
        current."""
        try:
            build = cls.probe_build_number(binary)
        except Exception:
            return False
        return build is not None and build < cls._OUTPUT_ROWS_CAPPED_BUILD

    # ggml-org/llama.cpp#23089 added continuation inside an unfinished reasoning block.
    _THOUGHT_CONTINUATION_BUILD = 9200

    @classmethod
    def resumes_thoughts(cls, binary: Optional[str] = None) -> bool:
        """Whether this llama-server continues a trailing turn that holds only reasoning.

        Unknown builds are treated as current, matching Studio's prebuilts."""
        try:
            build = cls.probe_build_number(binary)
        except Exception:
            return True
        return build is None or build >= cls._THOUGHT_CONTINUATION_BUILD

    _ADVERTISED_DEFAULT_RE = re.compile(r"\(default:\s*(-?\d+)")

    @classmethod
    def _advertised_int_default(cls, help_block: Optional[str]) -> Optional[int]:
        """None means the build did not say; help wraps across columns, so the caller passes the
        joined text."""
        if not help_block:
            return None
        match = cls._ADVERTISED_DEFAULT_RE.search(help_block)
        return int(match.group(1)) if match else None

    @classmethod
    def probe_server_capabilities(cls, binary: Optional[str] = None) -> dict[str, object]:
        """Parse llama-server --help for flags; ngram_mod_flavor needs descriptions (legacy names are removal stubs); use the real exec path (SIP strips DYLD)."""
        bin_path = cls._exec_path_for_launch(binary or cls._find_llama_server_binary())
        if not bin_path or not Path(bin_path).is_file():
            return {
                "found": False,
                "mtp_token": None,
                "supports_mtp": False,
                "supports_dspark": False,
                "supports_dflash": False,
                "supports_eagle3": False,
                "mtp_probe_inconclusive": True,
                "ngram_mod_flavor": None,
                "supports_ngram_mod": False,
                "spec_draft_n_max_flag": None,
                "spec_draft_n_max_default": None,
                "supports_kv_unified": False,
                # Fail OPEN: these are emitted on every launch; only a positive absence drops one.
                "supports_no_context_shift": True,
                "supports_jinja": True,
                "supports_api_key_file": True,
                "supports_flash_attn": True,
                "flash_attn_takes_value": True,
                "supports_fit_ctx": False,
                "supports_fit_target": False,
                "fit_measures_borrowed_drafts": False,
                "supports_cache_ram": False,
                "supports_ctx_checkpoints": False,
                "ctx_checkpoints_flag": None,
                "ctx_checkpoints_default": None,
                "supports_no_cache_prompt": False,
                "supports_metrics": False,
                "supports_slot_save": False,
                "supports_reasoning_budget": False,
                "supports_reasoning_budget_message": False,
                "reasoning_budget_probe_inconclusive": True,
                # Fails CLOSED, unlike --jinja: a build without --reasoning exits on it.
                "supports_reasoning_flag": False,
                "supports_no_mmproj_offload": False,
                "supports_trace_verbosity": False,
                "supports_video_fps": False,
                "supports_video_timestamp_interval": False,
                "supports_video_ffmpeg_dir": None,
                "supports_load_mode": False,
                "supports_moe_cache": False,
                "supports_moe_cache_auto": False,
                "supports_lazy_mode": False,
                "lazy_mode_flag": None,
                "spec_draft_ngl_flag": None,
                "spec_draft_cache_k_flag": None,
                "spec_draft_cache_v_flag": None,
                "flags": {},
                "help_probe_ok": False,
            }
        try:
            binary_stat = Path(bin_path).stat()
            mtime_ns = binary_stat.st_mtime_ns
            size = binary_stat.st_size
        except OSError:
            mtime_ns = 0
            size = 0
        cache_key = (bin_path, mtime_ns, size)
        hit = None
        with cls._capability_cache_lock:
            cached = cls._capability_cache.get(cache_key)
            if cached is not None:
                retry_after = cls._capability_retry_after.get(cache_key)
                if not cached.get("mtp_probe_inconclusive") or (
                    retry_after is not None and time.monotonic() < retry_after
                ):
                    hit = cached
                else:
                    cls._capability_cache.pop(cache_key, None)
                    cls._capability_retry_after.pop(cache_key, None)
        if hit is not None:
            # A copy: the cached dict is shared across threads.
            if hit.get("supports_fit_target"):
                hit = {
                    **hit,
                    "fit_measures_borrowed_drafts": _fit_measures_borrowed_drafts(bin_path),
                }
            return hit

        mtp_token: Optional[str] = None
        supports_dspark = False
        supports_dflash = False
        supports_eagle3 = False
        ngram_mod_flavor: Optional[str] = None
        spec_draft_n_max_flag: Optional[str] = None
        spec_draft_n_max_default: Optional[int] = None
        supports_kv_unified = False
        supports_trace_verbosity = False
        # See the fallback dict: these fail open.
        supports_no_context_shift = True
        supports_jinja = True
        supports_api_key_file = True
        supports_flash_attn = True
        flash_attn_takes_value = True
        supports_fit_ctx = False
        supports_fit_target = False
        supports_cache_ram = False
        supports_ctx_checkpoints = False
        ctx_checkpoints_flag = None
        ctx_checkpoints_default = None
        supports_no_cache_prompt = False
        supports_metrics = False
        supports_slot_save = False
        supports_reasoning_budget = False
        supports_reasoning_budget_message = False
        supports_reasoning_flag = False
        supports_no_mmproj_offload = False
        supports_video_fps = False
        supports_video_timestamp_interval = False
        supports_video_ffmpeg_dir = None
        supports_load_mode = False
        supports_moe_cache = False
        supports_moe_cache_auto = False
        supports_lazy_mode = False
        lazy_mode_flag = None
        spec_draft_ngl_flag = None
        spec_draft_cache_k_flag = None
        spec_draft_cache_v_flag = None
        saw_spec_type = False
        probe_ok = False
        help_text = ""
        # Pre-initialised: a failed probe must fall back to nothing known, not raise.
        blocks: dict[str, str] = {}
        # Out here: a timed-out probe skips the loop but the catalogue is still built.
        takes_value: dict[str, bool] = {}

        def _is_real(flag: str) -> bool:
            """True if the flag exists AND is not a removal stub."""
            desc = blocks.get(flag)
            if desc is None:
                return False
            return "argument has been removed" not in desc

        try:
            probe_env = cls._llama_server_env_for_binary(bin_path)
            # Describe the binary, not the user's LLAMA_ARG_* defaults.
            for name in tuple(probe_env):
                if name.startswith("LLAMA_ARG_"):
                    probe_env.pop(name, None)
            if sys.platform == "darwin":
                # --help would otherwise init Metal and compile shaders, exceeding the timeout.
                probe_env["GGML_METAL_DEVICES"] = "0"
            code_integrity_blocked: Optional[str] = None
            result = subprocess.run(
                [bin_path, "--help"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                check = False,
                env = probe_env,
                # Else the probe flashes a console window on every status poll.
                **_windows_hidden_subprocess_kwargs(),
            )
            probe_ok = result.returncode == 0
            help_text = (result.stdout or "") + "\n" + (result.stderr or "")
            if not probe_ok:
                # The loader kills the process: NTSTATUS arrives as the return code or in output.
                code_integrity_blocked = code_integrity_block_reason(
                    result.returncode
                ) or code_integrity_block_reason(help_text)
                if code_integrity_blocked is not None:
                    logger.warning(
                        "llama-server is blocked by Windows code integrity policy: "
                        f"{code_integrity_blocked}. Binary: {bin_path}"
                    )
            # One block per flag, so a removed-argument description stays with its flag.
            current_flags: list[str] = []
            current_desc: list[str] = []
            # Flag -> whether its declaration shows a value placeholder.
            current_takes_value = False
            for line in help_text.splitlines():
                stripped = line.strip()
                if stripped.startswith("-") and not line.startswith(" "):
                    # New flag line; flush previous.
                    if current_flags:
                        desc = " ".join(current_desc)
                        for f in current_flags:
                            blocks[f] = desc
                            takes_value[f] = current_takes_value
                    current_flags = []
                    current_desc = [stripped]
                    # A placeholder follows its flag by a single space; the description sits 2+ spaces away.
                    current_takes_value = False
                    _previous_end = None
                    for _match in re.finditer(r"\S+", stripped):
                        _token = _match.group()
                        _flag_shaped = _token.startswith("-") and re.match(
                            r"-{1,2}[A-Za-z][A-Za-z0-9_-]*,?$", _token
                        )
                        if _flag_shaped:
                            _previous_end = _match.end()
                            continue
                        # The first non-alias token: a value placeholder when one space away, otherwise the description.
                        current_takes_value = (
                            _previous_end is not None and _match.start() - _previous_end == 1
                        )
                        break
                    # Declaration prefix only, so flag references in descriptions are ignored.
                    for tok in re.split(r"[,\s]+", stripped):
                        if tok.startswith("--") and re.match(r"--[A-Za-z][A-Za-z0-9_-]*$", tok):
                            current_flags.append(tok)
                        elif tok.startswith("-") and re.match(r"-[A-Za-z][A-Za-z0-9_-]*$", tok):
                            # Short aliases kept: the editor validates typed flags like -t against them.
                            current_flags.append(tok)
                        elif tok.startswith("-") and len(tok) > 1:
                            # A value placeholder like -1; keep scanning aliases.
                            continue
                        else:
                            # First non-flag token marks end of decl.
                            break
                else:
                    current_desc.append(stripped)
            if current_flags:
                desc = " ".join(current_desc)
                for f in current_flags:
                    blocks[f] = desc
                    takes_value[f] = current_takes_value

            # Read the full --spec-type block, since some builds put the enum on the next line; prefer draft-mtp over mtp.
            spec_help = blocks.get("--spec-type") or ""
            if not spec_help:
                # Fallback: join --spec-type lines, avoiding incidental "mtp" in --help.
                spec_help = "\n".join(
                    line for line in help_text.splitlines() if "--spec-type" in line
                )
            mtp_token = cls._mtp_token_from_spec_help(spec_help)
            supports_dspark = bool(
                re.search(
                    r"(?<![a-z0-9_-])draft-dspark(?![a-z0-9_-])",
                    spec_help.lower(),
                )
            )
            supports_eagle3 = bool(
                re.search(r"(?<![a-z0-9_-])draft-eagle3(?![a-z0-9_-])", spec_help.lower())
            )
            # Same word-boundary match as DSpark: draft-dflash only exists on
            # builds that carry the arch, and emitting a --spec-type the binary
            # does not know aborts the launch instead of falling back.
            supports_dflash = bool(
                re.search(
                    r"(?<![a-z0-9_-])draft-dflash(?![a-z0-9_-])",
                    spec_help.lower(),
                )
            )
            # Only a resolved --spec-type block confirms missing MTP; otherwise fail open.
            saw_spec_type = bool(spec_help.strip()) and "--spec-type" in spec_help

            # Post-rename builds keep legacy args as stubs; pre-rename ones only have legacy.
            new_ngram_real = (
                _is_real("--spec-ngram-mod-n-match")
                and _is_real("--spec-ngram-mod-n-min")
                and _is_real("--spec-ngram-mod-n-max")
            )
            legacy_ngram_real = (
                _is_real("--spec-ngram-size-n")
                and _is_real("--draft-max")
                and _is_real("--draft-min")
            )
            if new_ngram_real:
                ngram_mod_flavor = "new"
            elif legacy_ngram_real:
                ngram_mod_flavor = "legacy"

            # n_max flag: prefer post-rename, fall back to legacy.
            if _is_real("--spec-draft-n-max"):
                spec_draft_n_max_flag = "--spec-draft-n-max"
            elif _is_real("--draft-max"):
                spec_draft_n_max_flag = "--draft-max"
            # The build's own default depth, needed when extras own --spec-type: the child runs on it and the Hybrid Mamba rollback reserve scales by it.
            if spec_draft_n_max_flag:
                _n_max_default = re.search(
                    r"default:\s*(\d+)", blocks.get(spec_draft_n_max_flag) or ""
                )
                if _n_max_default:
                    spec_draft_n_max_default = int(_n_max_default.group(1))

            supports_kv_unified = _is_real("--kv-unified")
            # Fails closed: an older build's level 4 may be per-token debug on the decode path.
            supports_trace_verbosity = bool(
                probe_ok
                and _is_real("--verbosity")
                and _TRACE_VERBOSITY_HELP_RE.search(blocks.get("--verbosity") or "")
            )
            # Empty or failed --help must never read as absent for a flag we always send.
            if probe_ok and blocks:
                supports_no_context_shift = _is_real("--no-context-shift")
                supports_jinja = _is_real("--jinja")
                supports_api_key_file = _is_real("--api-key-file")
                # Two answers, not one: whether the flag exists, and whether its
                # declaration takes a value. A build predating flash attention
                # has neither, and emitting the flag there is an immediate exit.
                supports_flash_attn = _is_real("--flash-attn")
                flash_attn_takes_value = cls._flash_attn_takes_value(help_text)
            supports_fit_ctx = _is_real("--fit-ctx")
            supports_fit_target = _is_real("--fit-target")
            supports_cache_ram = _is_real("--cache-ram")
            # --swa-checkpoints is the older alias; probing only the modern name drops the Checkpoints control on old builds, so record which name this build has.
            for _alias in ("--ctx-checkpoints", "--swa-checkpoints"):
                if _is_real(_alias):
                    ctx_checkpoints_flag = _alias
                    break
            supports_ctx_checkpoints = ctx_checkpoints_flag is not None
            # The default was 3 on older builds, so a hardcoded 32 would raise their count.
            if ctx_checkpoints_flag:
                ctx_checkpoints_default = cls._advertised_int_default(
                    blocks.get(ctx_checkpoints_flag)
                )
            supports_no_cache_prompt = _is_real("--no-cache-prompt")
            supports_metrics = _is_real("--metrics")
            supports_slot_save = _is_real("--slot-save-path")
            supports_reasoning_budget = _is_real("--reasoning-budget")
            supports_reasoning_budget_message = _is_real("--reasoning-budget-message")
            # Catalogue, not help text: --reasoning-format / -budget / -preserve contain it.
            supports_reasoning_flag = _is_real("--reasoning")
            supports_no_mmproj_offload = _is_real("--no-mmproj-offload")
            supports_video_fps = _is_real("--video-fps")
            supports_video_timestamp_interval = _is_real("--video-timestamp-interval")
            supports_video_ffmpeg_dir = (
                _is_real("--video-ffmpeg-dir") if probe_ok and blocks else None
            )
            # --load-mode supersedes --mlock / --no-mmap, which are deprecated.
            # Pre-initialised above: a failed probe must fall back, not raise.
            supports_load_mode = _is_real("--load-mode")
            # "auto" is a fork addition; upstream builds exit on it, so fail closed.
            supports_moe_cache = bool(probe_ok and _is_real("--moe-cache-mib"))
            supports_moe_cache_auto = bool(
                supports_moe_cache
                and re.search(
                    r"(?<![a-z0-9_-])auto(?![a-z0-9_-])",
                    (blocks.get("--moe-cache-mib") or "").lower(),
                )
            )
            if probe_ok:
                lazy_mode_flag = next(
                    (f for f in ("--lazy-mode", "--tensor-read-lazy") if _is_real(f)), None
                )
            supports_lazy_mode = lazy_mode_flag is not None
            # Record WHICH alias this build has: --spec-draft-ngl only landed in
            # b8955, and a build exposing only --gpu-layers-draft would refuse to
            # start on the newer name. Long forms only, since the block parser above
            # skips short aliases, so "-ngld" could never have matched.
            for _alias in ("--spec-draft-ngl", "--gpu-layers-draft", "--n-gpu-layers-draft"):
                if _is_real(_alias):
                    spec_draft_ngl_flag = _alias
                    break
            # Draft cache K and V are probed separately, since builds may carry only one flag
            # spelling.
            for _alias in ("--spec-draft-type-k", "--cache-type-k-draft"):
                if _is_real(_alias):
                    spec_draft_cache_k_flag = _alias
                    break
            for _alias in ("--spec-draft-type-v", "--cache-type-v-draft"):
                if _is_real(_alias):
                    spec_draft_cache_v_flag = _alias
                    break
        except (OSError, subprocess.SubprocessError) as exc:
            blocked = code_integrity_block_reason(exc)
            code_integrity_blocked = blocked
            if blocked is not None:
                # Warning, not debug: the only other symptom is a silent probe.
                logger.warning(
                    f"llama-server is blocked by Windows code integrity policy: {blocked}. "
                    f"Binary: {bin_path}"
                )
            else:
                logger.debug(f"llama-server --help probe failed: {exc}")
            saw_spec_type = False
            probe_ok = False
            help_text = ""

        help_nonempty = bool(help_text.strip())
        # Confirmed only by a successful --help with an MTP --spec-type block.
        if saw_spec_type and probe_ok:
            supports_mtp = mtp_token is not None
            mtp_probe_inconclusive = False
        elif help_nonempty and probe_ok:
            supports_mtp = False
            mtp_probe_inconclusive = False
        else:
            supports_mtp = False
            mtp_probe_inconclusive = True

        if supports_dspark:
            try:
                from utils.llama_cpp_freshness import read_install_marker

                marker = read_install_marker(bin_path) or {}
                release_tag = marker.get("release_tag") or marker.get("tag")
                if cls._dspark_release_is_broken(release_tag):
                    supports_dspark = False
                    logger.info(
                        "Disabling DSpark for known-broken llama.cpp prebuilt %s",
                        release_tag,
                    )
            except Exception as exc:
                logger.debug("Could not inspect llama.cpp prebuilt identity: %s", exc)

        info = {
            "found": True,
            "mtp_token": mtp_token,
            "supports_mtp": supports_mtp,
            "supports_dspark": bool(supports_dspark and saw_spec_type and probe_ok),
            "supports_dflash": bool(supports_dflash and saw_spec_type and probe_ok),
            "supports_eagle3": bool(supports_eagle3 and saw_spec_type and probe_ok),
            "mtp_probe_inconclusive": mtp_probe_inconclusive,
            "ngram_mod_flavor": ngram_mod_flavor,
            "supports_ngram_mod": ngram_mod_flavor is not None,
            "spec_draft_n_max_flag": spec_draft_n_max_flag,
            "spec_draft_n_max_default": spec_draft_n_max_default,
            "supports_kv_unified": supports_kv_unified,
            "supports_no_context_shift": supports_no_context_shift,
            "supports_jinja": supports_jinja,
            "supports_api_key_file": supports_api_key_file,
            "supports_flash_attn": supports_flash_attn,
            "flash_attn_takes_value": flash_attn_takes_value,
            "supports_fit_ctx": supports_fit_ctx,
            "supports_fit_target": supports_fit_target,
            "fit_measures_borrowed_drafts": supports_fit_target
            and _fit_measures_borrowed_drafts(bin_path),
            "supports_cache_ram": supports_cache_ram,
            "supports_ctx_checkpoints": supports_ctx_checkpoints,
            "ctx_checkpoints_flag": ctx_checkpoints_flag,
            "ctx_checkpoints_default": ctx_checkpoints_default,
            "supports_no_cache_prompt": supports_no_cache_prompt,
            "supports_metrics": supports_metrics,
            "supports_slot_save": supports_slot_save,
            "supports_reasoning_budget": supports_reasoning_budget,
            "supports_reasoning_budget_message": supports_reasoning_budget_message,
            "reasoning_budget_probe_inconclusive": not (probe_ok and help_nonempty),
            "supports_reasoning_flag": supports_reasoning_flag,
            "supports_no_mmproj_offload": supports_no_mmproj_offload,
            "supports_trace_verbosity": supports_trace_verbosity,
            "supports_video_fps": supports_video_fps,
            "supports_video_timestamp_interval": supports_video_timestamp_interval,
            "supports_video_ffmpeg_dir": supports_video_ffmpeg_dir,
            "supports_load_mode": supports_load_mode,
            "supports_moe_cache": supports_moe_cache,
            "supports_moe_cache_auto": supports_moe_cache_auto,
            "supports_lazy_mode": supports_lazy_mode,
            "lazy_mode_flag": lazy_mode_flag,
            "spec_draft_ngl_flag": spec_draft_ngl_flag,
            "spec_draft_cache_k_flag": spec_draft_cache_k_flag,
            "spec_draft_cache_v_flag": spec_draft_cache_v_flag,
            # Validate pass-through args against this build; removal stubs are excluded, since a stub reads as supported and the editor stays quiet on refused flags.
            "flags": {flag: desc for flag, desc in blocks.items() if _is_real(flag)},
            # Switch flags take no value, so the editor can reject '--verbose foo' that llama-server refuses.
            "switch_flags": sorted(
                flag for flag, takes in takes_value.items() if not takes and _is_real(flag)
            ),
            # A nonzero exit can leave a partial catalogue.
            "help_probe_ok": bool(probe_ok),
        }
        with cls._capability_cache_lock:
            published = cls._capability_cache.get(cache_key)
            if published is not None and (
                not published.get("mtp_probe_inconclusive") or mtp_probe_inconclusive
            ):
                # Callers may finish out of order; never replace a conclusive result with a weaker one.
                return published
            cls._capability_cache[cache_key] = info
            if mtp_probe_inconclusive:
                # Do not pin transient failures forever or repeat timeouts; double only confirmed blocks, else busy machines get stuck on stale caps.
                if code_integrity_blocked is not None:
                    delay = cls._capability_retry_backoff.get(
                        cache_key, cls._CAPABILITY_PROBE_RETRY_SECONDS
                    )
                    cls._capability_retry_backoff[cache_key] = min(
                        delay * 2.0, cls._CAPABILITY_PROBE_RETRY_MAX_SECONDS
                    )
                else:
                    delay = cls._CAPABILITY_PROBE_RETRY_SECONDS
                    cls._capability_retry_backoff.pop(cache_key, None)
                cls._capability_retry_after[cache_key] = time.monotonic() + delay
            else:
                cls._capability_retry_after.pop(cache_key, None)
                cls._capability_retry_backoff.pop(cache_key, None)
            return info

    @classmethod
    def validate_reasoning_budget_capabilities(
        cls,
        binary: Optional[str],
        *,
        extra_args: Optional[Iterable[str]],
        reasoning_budget: int,
        reasoning_budget_message: str,
    ) -> dict[str, object]:
        """Reject configured reasoning flags before a live swap; explicit config only, since LLAMA_ARG_THINK_BUDGET* is machine-wide and no Studio control clears it."""
        effective_budget = resolve_reasoning_budget(extra_args, reasoning_budget)
        effective_message = resolve_reasoning_budget_message(extra_args, reasoning_budget_message)
        needs_budget, needs_message = cls.reasoning_budget_settings_requested(
            extra_args = extra_args,
            reasoning_budget = effective_budget,
            reasoning_budget_message = effective_message,
        )
        if not needs_budget and not needs_message:
            return cls.probe_server_capabilities(binary) if binary else {}
        if not binary:
            raise ValueError(
                "Reasoning budget settings require a llama-server runtime. "
                "Install or update llama.cpp, then try again."
            )

        caps = cls.probe_server_capabilities(binary)
        missing = []
        if needs_budget and not caps.get("supports_reasoning_budget"):
            missing.append("--reasoning-budget")
        if needs_message and not caps.get("supports_reasoning_budget_message"):
            missing.append("--reasoning-budget-message")
        if missing:
            verdict = (
                "could not verify support for"
                if caps.get("reasoning_budget_probe_inconclusive")
                else "does not support"
            )
            raise ValueError(
                f"llama-server at {binary} {verdict} {', '.join(missing)}. "
                "Update llama.cpp or clear the Reasoning Budget settings."
            )
        if needs_budget and effective_budget > 0:
            value_key = f"supports_reasoning_budget_value:{effective_budget}"
            if value_key not in caps:
                probe_binary = cls._exec_path_for_launch(binary)
                probe_env = cls._llama_server_env_for_binary(probe_binary)
                # llama-server parses LLAMA_ARG_* before argv, so clear stale inherited values.
                for name in tuple(probe_env):
                    if name.startswith("LLAMA_ARG_"):
                        probe_env.pop(name, None)
                if sys.platform == "darwin":
                    probe_env["GGML_METAL_DEVICES"] = "0"
                try:
                    result = subprocess.run(
                        [probe_binary, "--reasoning-budget", str(effective_budget), "--help"],
                        capture_output = True,
                        text = True,
                        encoding = "utf-8",
                        errors = "replace",
                        timeout = 10,
                        check = False,
                        env = probe_env,
                    )
                    caps[value_key] = result.returncode == 0
                except (OSError, subprocess.SubprocessError):
                    caps[value_key] = False
            if not caps[value_key]:
                raise ValueError(
                    f"llama-server at {binary} does not accept positive reasoning budgets. "
                    "Update llama.cpp, use 0 or -1, or clear the Reasoning Budget setting."
                )
        return caps

    @staticmethod
    def reasoning_budget_settings_requested(
        *, extra_args: Optional[Iterable[str]], reasoning_budget: int, reasoning_budget_message: str
    ) -> tuple[bool, bool]:
        budget_override = parse_reasoning_budget_override(extra_args)
        message_override = parse_reasoning_budget_message_override(extra_args)
        return (
            reasoning_budget != -1 or budget_override is not None,
            bool(reasoning_budget_message) or message_override is not None,
        )

    @staticmethod
    def _mtp_token_from_spec_help(spec_help: str) -> Optional[str]:
        """Prefers draft-mtp over the later bare mtp rename; None when neither is a --spec-type enum
        value."""
        text = spec_help or ""
        if "draft-mtp" in text:
            return "draft-mtp"
        # Bare `mtp` enum token (`|mtp|`, `,mtp,`, ...), not a substring.
        if re.search(r"(?<![A-Za-z0-9_-])mtp(?![A-Za-z0-9_-])", text):
            return "mtp"
        return None
