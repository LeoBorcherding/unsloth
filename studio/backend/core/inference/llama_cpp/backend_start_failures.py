# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, start failure classification."""

import os
import re
import sys

from typing import (
    Optional,
    Sequence,
)
from utils.code_integrity import (
    code_integrity_block_reason,
    code_integrity_user_message,
)
from utils.gguf_archs import (
    is_audio_cpp_gguf_architecture,
    is_speech_gguf_architecture,
)

from .binary_runtime import (
    _is_installer_entrypoint,
    _llama_lib_dir,
    _resolve_llama_binary,
)
from .tool_loop_text import (
    _FAILURE_QUOTE_CHARS,
    _FAILURE_SCAN_TAIL_CHARS,
)
from utils.gguf_archs import is_foreign_asr_gguf_architecture
from .server_log import (
    _PORT_OWNER_REFUSED_MARKER,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _StartFailureMixin:
    @staticmethod
    def _is_bundled_llama_library(lib: str) -> bool:
        # The loader prints a soname when absent, a full path when rejected; compare basenames.
        return os.path.basename((lib or "").strip()).startswith(
            LlamaCppBackend._BUNDLED_LIB_PREFIXES
        )

    @staticmethod
    def _is_unsloth_managed_binary(binary: Optional[str]) -> bool:
        """A pinned LLAMA_SERVER_PATH or PATH llama-server is unmanaged, so update cannot repair it; an unknown binary (none passed) keeps the managed default."""
        if not binary:
            return True
        try:
            from utils.llama_cpp_path_settings import custom_llama_cpp_path_source
            from utils.llama_cpp_update import (
                _active_install_is_local_link,
                _llama_install_root,
            )

            # A Settings-selected checkout is user-owned even if it looks like our tree.
            if custom_llama_cpp_path_source() == "studio":
                return False
            # A --with-llama-cpp-dir symlinked tree is the user's; the updater refuses to write through it.
            if _active_install_is_local_link(binary):
                return False
            return _llama_install_root(binary) is not None
        except Exception:
            return True

    @staticmethod
    def _is_llama_install_tree(binary: Optional[str]) -> bool:
        """A --with-llama-cpp-dir checkout is the active install but not updater-replaceable."""
        if not binary:
            return True
        try:
            from utils.llama_cpp_update import _llama_install_root
            return _llama_install_root(binary) is not None
        except Exception:
            return True

    @staticmethod
    def _is_library_path(lib: str) -> bool:
        # glibc treats any slash as a pathname (no search); match both separators since the text
        # comes from another host.
        lib = (lib or "").strip()
        return "/" in lib or "\\" in lib

    @staticmethod
    def _missing_library_message(lib: str, binary: Optional[str] = None) -> str:
        """The loader could not FIND ``lib`` ("cannot open shared object file")."""
        if LlamaCppBackend._is_bundled_llama_library(lib):
            if LlamaCppBackend._is_unsloth_managed_binary(binary):
                return (
                    f"llama-server could not start: {lib} is part of Unsloth's own "
                    "llama.cpp runtime and is missing from the install. Run "
                    "`unsloth studio update` to reinstall it, then load the model "
                    "again."
                )
            return (
                f"llama-server could not start: {lib} is part of the llama.cpp "
                "runtime that the llama-server binary in use was built with. That "
                "binary is a custom install Unsloth does not manage, so reinstall "
                "or rebuild that llama.cpp, then load the model again."
            )
        # A pathname was opened directly, so no package will put that exact file there.
        if LlamaCppBackend._is_library_path(lib):
            _remedy = (
                "run `unsloth studio update`"
                if LlamaCppBackend._is_unsloth_managed_binary(binary)
                else "reinstall or rebuild that custom llama.cpp"
            )
            return (
                f"llama-server could not start: {lib} is missing from that exact "
                f"location. Restore it, or {_remedy}, then load the model again."
            )
        return (
            f"llama-server could not start: the system library "
            f"{lib or 'it needs'} is missing. Install it with your package "
            f"manager{LlamaCppBackend._SHARED_LIB_PACKAGES.get(lib, '')}, "
            "then load the model again."
        )

    @staticmethod
    def _unloadable_library_message(
        lib: str,
        reason: str,
        binary: Optional[str] = None,
    ) -> str:
        """The file exists but the loader refused it, so installing a package is the wrong advice."""
        detail = f" ({reason})" if reason else ""
        if LlamaCppBackend._is_bundled_llama_library(lib):
            if LlamaCppBackend._is_unsloth_managed_binary(binary):
                return (
                    f"llama-server could not start: {lib}, part of Unsloth's own "
                    f"llama.cpp runtime, could not be loaded{detail}. The install "
                    "is incomplete or mismatched. Run `unsloth studio update` to "
                    "reinstall it, then load the model again."
                )
            return (
                f"llama-server could not start: {lib}, part of the llama.cpp "
                "runtime that the llama-server binary in use was built with, "
                f"could not be loaded{detail}. That binary is a custom install "
                "Unsloth does not manage, so reinstall or rebuild that "
                "llama.cpp, then load the model again."
            )
        return (
            f"llama-server could not start: the library {lib or 'it needs'} "
            f"could not be loaded{detail}. Reinstall or update whatever "
            "provides it, then load the model again."
        )

    @staticmethod
    def _exec_path_for_launch(binary: Optional[str]) -> Optional[str]:
        """Resolves only an installer-template macOS wrapper past /bin/sh (SIP strips DYLD_*), by shape not provenance; any other wrapper is left alone."""
        if not binary or sys.platform != "darwin":
            return binary
        if not LlamaCppBackend._is_unsloth_managed_binary(binary):
            try:
                from utils.llama_cpp_path_settings import custom_llama_cpp_path_source
                if custom_llama_cpp_path_source() != "studio":
                    return binary
            except Exception:
                return binary
        if not _is_installer_entrypoint(binary):
            return binary
        # template_only: resolve only through links that are also ours.
        return str(_resolve_llama_binary(binary, template_only = True))

    @staticmethod
    def _runtime_remedy(binary: Optional[str]) -> str:
        """Pinned or PATH llama-server binaries get a rebuild instruction, since update cannot touch
        them."""
        return (
            "run `unsloth studio update` to reinstall the llama.cpp runtime"
            if LlamaCppBackend._is_unsloth_managed_binary(binary)
            else "reinstall or rebuild that custom llama.cpp"
        )

    # A tried: list can run to kilobytes; cap like the glibc branch.
    _MACOS_REASON_CHARS = 300

    # macOS PATH_MAX is 1024; anything past double is not from the loader.
    _DYLD_LIB_NAME_MAX_CHARS = 2048

    @staticmethod
    def _plain_dyld_lib_name(lib: str) -> str:
        """Reduces dyld @rpath, @loader_path or @executable_path names to basename: they are lookup tokens, so as paths they would name a missing directory."""
        if lib.startswith(("@rpath/", "@loader_path/", "@executable_path/")):
            lib = os.path.basename(lib)
        # Quoted from untrusted output into an HTTP error, so cap it.
        if len(lib) > LlamaCppBackend._DYLD_LIB_NAME_MAX_CHARS:
            lib = lib[: LlamaCppBackend._DYLD_LIB_NAME_MAX_CHARS] + "..."
        return lib

    # Ten paths of a long length each is generous; it only cuts text no real dyld reason produces.
    _DYLD_REASON_MAX_CHARS = 4000
    # How far past Library not loaded to scan for its Reason: four times the cap, so noise before a real reason is still found whole.
    _DYLD_REASON_SCAN_CHARS = 16000
    # Candidates come from one tried: list; dyld searches a bounded set, so a longer list is not dyld's.
    _DYLD_MAX_CANDIDATES = 128

    # dyld framing must start a line; phrases like 'Referenced from:' can come from wrappers or GGUF metadata.
    _DYLD_FRAMING_RE = re.compile(
        r"(?im)^[ \t]*(?:dyld(?:\[\d+\])?:|termination reason:.*\bdyld\b)"
    )

    @staticmethod
    def _dyld_reason_after(output: str, start: int) -> str:
        """Searched only after the dyld message, in a capped window, since the match can backtrack badly."""
        # Windowed: the reason sits right under the line, and matching all output could use huge memory.
        window = output[start : start + LlamaCppBackend._DYLD_REASON_SCAN_CHARS]
        match = re.compile(
            r"reason:[ \t]*([^\r\n]*(?:\r?\n[ \t]+[^\r\n]*)*)",
            re.IGNORECASE,
        ).search(window)
        if not match:
            return ""
        return " ".join(match.group(1)[: LlamaCppBackend._DYLD_REASON_MAX_CHARS].split())

    @staticmethod
    def _dyld_tried_verdict(reason: str, binary: Optional[str]) -> str:
        """Prefers the dyld candidate in our own lib dir, else the first; the bare reason if no list."""
        lib_dir = ""
        if binary:
            try:
                lib_dir = str(_llama_lib_dir(binary))
            except (OSError, ValueError):
                lib_dir = ""
        # Use finditer with a cap, not findall: a list built before the first useful match is wasted work.
        first: Optional[str] = None
        for index, match in enumerate(re.finditer(r"'([^']+)'\s*\(([^)]*)\)", reason)):
            if index >= LlamaCppBackend._DYLD_MAX_CANDIDATES:
                break
            path, verdict = match.group(1), match.group(2)
            if first is None:
                first = verdict
            if lib_dir and os.path.dirname(path) == lib_dir:
                return verdict
        return first if first is not None else reason

    @staticmethod
    def _classify_macos_loader_failure(output: str, binary: Optional[str]) -> Optional[str]:
        """Only dyld-framed output counts: GGUF metadata can echo 'Library not loaded' text."""
        if not output:
            return None
        if not LlamaCppBackend._DYLD_FRAMING_RE.search(output):
            return None
        lowered = output.lower()

        # Same markers as install_llama_prebuilt.looks_like_macos_incompatibility.
        if ("built for macos" in lowered and "newer than running os" in lowered) or (
            "symbol not found" in lowered and "mtlresidency" in lowered
        ):
            return (
                "llama-server could not start: the installed llama.cpp runtime "
                "was built for a newer version of macOS than this Mac is "
                f"running. Update macOS, or {LlamaCppBackend._runtime_remedy(binary)} "
                "to get a build that matches this OS."
            )

        # "Library not loaded: <path>", then the "Reason:" that belongs to it.
        not_loaded = re.search(
            r"library not loaded:[ \t]*([^\r\n]+)",
            output,
            re.IGNORECASE,
        )
        if not_loaded:
            lib = LlamaCppBackend._plain_dyld_lib_name(not_loaded.group(1).strip())
            reason = LlamaCppBackend._dyld_reason_after(output, not_loaded.end())
            # dyld4 gives a per-path verdict; judge our path, not a stale copy elsewhere.
            verdict = LlamaCppBackend._dyld_tried_verdict(reason, binary)
            verdict_l = verdict.lower()
            # Code-signature refusal: a hard SIGKILL on Apple Silicon, not a GGUF or memory problem.
            if "code signature" in verdict_l or "not valid for use in process" in verdict_l:
                return (
                    f"llama-server could not start: macOS rejected the code "
                    f"signature of {lib}. The file was modified or is "
                    f"incompletely signed, so {LlamaCppBackend._runtime_remedy(binary)}."
                )
            # dyld4 reports incompatible architecture; older dyld reported wrong architecture.
            if "incompatible architecture" in verdict_l or "wrong architecture" in verdict_l:
                return (
                    f"llama-server could not start: {lib} was built for a "
                    "different CPU architecture than this Mac. Its install is "
                    f"for the wrong platform, so {LlamaCppBackend._runtime_remedy(binary)}."
                )
            # Only 'no such file' / 'image not found' mean absent; otherwise dyld found and refused it.
            if not verdict_l or "no such file" in verdict_l or "image not found" in verdict_l:
                return LlamaCppBackend._missing_library_message(lib, binary)
            return LlamaCppBackend._unloadable_library_message(
                lib, verdict[: LlamaCppBackend._MACOS_REASON_CHARS], binary
            )

        # Symbol mismatch: a system framework means the build needs a newer macOS, ours means mixed
        # builds. Only with dyld framing, since 'Symbol not found' is ordinary English.
        if "symbol not found" in lowered and any(
            marker in lowered for marker in ("dyld[", "dyld:", "referenced from:", "expected in:")
        ):
            expected = re.search(r"expected in:\s*(?:<[^>]*>\s*)?([^\r\n]+)", output, re.IGNORECASE)
            # Newer dyld quotes the path; strip quotes or the /System prefix test fails.
            expected_path = expected.group(1).strip().strip("'\"") if expected else ""
            if expected_path.startswith(("/System/", "/usr/lib/")):
                return (
                    "llama-server could not start: it needs a symbol this "
                    "version of macOS does not provide, so the installed "
                    "llama.cpp runtime was built for a newer macOS. Update "
                    f"macOS, or {LlamaCppBackend._runtime_remedy(binary)} to get "
                    "a build that matches this OS."
                )
            return (
                "llama-server could not start: its llama.cpp libraries are from "
                "a different build than the llama-server binary (a symbol it "
                "needs is missing). The install is mismatched, so "
                f"{LlamaCppBackend._runtime_remedy(binary)}."
            )
        return None

    # The shell's EBADARCH message; never reaches dyld, so matched on its own.
    _BAD_CPU_TYPE_RE = re.compile(r"(?im)^.*:[ \t]*bad cpu type in executable[ \t]*$")

    @staticmethod
    def _classify_llama_start_failure(
        output: str,
        gguf_path: Optional[str],
        model_identifier: Optional[str],
        returncode: Optional[int] = None,
        binary: Optional[str] = None,
        log_path: "Optional[Path | str]" = None,
        secrets: Sequence[Optional[str]] = (),
        extra_args: Optional[Sequence[str]] = None,
    ) -> str:
        """Redacts every classified message, since each branch quotes untrusted child output."""
        return LlamaCppBackend._scrub_secret_values(
            LlamaCppBackend._classify_start_failure_text(
                output,
                gguf_path,
                model_identifier,
                returncode,
                binary,
                log_path,
                secrets,
                extra_args,
            ),
            secrets,
        )

    @staticmethod
    def _classify_start_failure_text(
        output: str,
        gguf_path: Optional[str],
        model_identifier: Optional[str],
        returncode: Optional[int] = None,
        binary: Optional[str] = None,
        log_path: "Optional[Path | str]" = None,
        secrets: Sequence[Optional[str]] = (),
        extra_args: Optional[Sequence[str]] = None,
    ) -> str:
        """Classifies the child's output, which is never trusted, even when it mimics Unsloth's own
        framing."""
        lowered = (output or "").lower()

        # A healthy server we stopped ourselves: its output holds no failure to classify.
        if _PORT_OWNER_REFUSED_MARKER.lower() in lowered:
            return (
                "llama-server started and answered on its local port, but Studio could not "
                "confirm that the listener on that port belongs to the server it launched, so it "
                "stopped the server. The model file and memory are not the problem; the backend "
                "log names the port and the process that held it."
            )

        # Then: every branch below advises reinstall, memory or administrator,
        # none of which lift a refusal to load a file that is present.
        blocked = code_integrity_block_reason(returncode) or code_integrity_block_reason(output)
        if blocked is not None:
            return code_integrity_user_message(binary or "the llama.cpp runtime", blocked)

        cuda_image_error = LlamaCppBackend._cuda_kernel_image_error(output)
        if cuda_image_error is not None:
            return LlamaCppBackend._cuda_kernel_image_message(cuda_image_error, binary, log_path)

        # Split the glibc loader message on ': ', not whitespace, since library paths may contain
        # spaces.
        missing_lib = re.search(
            r"error while loading shared libraries:[ \t]*([^\r\n]+?)"
            r"(?::[ \t]+([^\r\n]*))?[ \t]*\r?$",
            output or "",
            re.MULTILINE,
        )
        if missing_lib:
            _lib = missing_lib.group(1).strip()
            _reason = (missing_lib.group(2) or "").strip()
            # glibc omits the object name on its own allocation failures.
            if not (
                "/" in _lib or "\\" in _lib or any(_e in _lib for _e in (".so", ".dylib", ".dll"))
            ):
                return LlamaCppBackend._unloadable_library_message(
                    "", f"{_lib} {_reason}".strip(), binary
                )
            # Only ENOENT means absent; EACCES etc. mean it exists but cannot be opened, so no package fix; a truncated tail keeps the missing-library reading.
            _reason_l = _reason.lower()
            _errno_text = ""
            if "cannot open shared object file:" in _reason_l:
                _errno_text = _reason_l.split("cannot open shared object file:", 1)[1].strip()
            if not _reason_l or (
                "cannot open" in _reason_l and (not _errno_text or "no such file" in _errno_text)
            ):
                return LlamaCppBackend._missing_library_message(_lib, binary)
            return LlamaCppBackend._unloadable_library_message(_lib, _reason, binary)

        # macOS: dyld shares no wording with glibc.
        macos_detail = LlamaCppBackend._classify_macos_loader_failure(output or "", binary)
        if macos_detail:
            return macos_detail

        # Wrong-arch executable: the shell rejects it before dyld runs.
        if LlamaCppBackend._BAD_CPU_TYPE_RE.search(output or ""):
            return (
                "llama-server could not start: the executable was built for a "
                "different CPU architecture than this Mac, so "
                f"{LlamaCppBackend._runtime_remedy(binary)}."
            )

        # Argument parsing runs before the model loads: an unknown flag vs a rejected value. Bounded
        # to the tail, where llama.cpp prints argument errors.
        scan_tail = (output or "")[-_FAILURE_SCAN_TAIL_CHARS:]

        def _short(value: str) -> str:
            """Bounds the child-output fragment quoted into an error, since a wrapper could print 64 KiB."""
            value = value.strip()
            return (
                value
                if len(value) <= _FAILURE_QUOTE_CHARS
                else (value[:_FAILURE_QUOTE_CHARS] + "...")
            )

        unknown_arg = re.search(r"error:\s*invalid argument:\s*(\S+)", scan_tail, re.IGNORECASE)
        if unknown_arg:
            # Name both owners: Unsloth's own conditional flags can be rejected too.
            return (
                f"llama-server does not recognise the argument "
                f"'{_short(unknown_arg.group(1))}'. Check it in this model's extra "
                f"arguments, or reinstall llama.cpp if you did not set it."
            )

        bad_arg_value = re.search(
            r'error while handling argument "([^"]+)":\s*([^\r\n]*)',
            scan_tail,
            re.IGNORECASE,
        )
        if bad_arg_value:
            _arg = _short(bad_arg_value.group(1))
            _why = _short(bad_arg_value.group(2) or "")
            # llama.cpp surfaces a failed std::stoi as just 'stoi'.
            if _why.lower() in {"stoi", "stof", "stod", "stoul", "stoll"}:
                _why = "the value is not a number"
            _tail = f": {_why}" if _why else ""
            # Blame the extra arguments only when they really carry the flag; an alias spelling gets neutral wording rather than a guess.
            _owned_by_user = False
            if extra_args:
                from core.inference.llama_server_args import _flag_name
                _failed_flag = _flag_name(_arg)
                _owned_by_user = _failed_flag is not None and any(
                    _flag_name(str(token)) == _failed_flag for token in extra_args
                )
            if _owned_by_user:
                return (
                    f"llama-server rejected the value for '{_arg}'{_tail}. Fix it in "
                    f"the extra arguments for this model."
                )
            return (
                f"llama-server rejected the value for '{_arg}'{_tail}. Check it in "
                f"this model's extra arguments, or reinstall llama.cpp if you did "
                f"not set it."
            )

        # --split-mode tensor is arch-gated in llama.cpp; point at the toggle.
        if "split_mode_tensor not implemented" in lowered:
            return (
                "Tensor parallelism is not supported for this model's "
                "architecture. Turn off Tensor Parallelism in the model "
                "settings and reload."
            )

        if LlamaCppBackend._is_sched_reserve_abort(output or ""):
            return LlamaCppBackend._sched_reserve_abort_message()

        # An older build refusing quantized KV under tensor split: the remedy is an update.
        if LlamaCppBackend._is_tensor_quant_kv_unsupported(lowered):
            return (
                "This llama.cpp build cannot use a quantized KV cache together "
                "with Tensor Parallelism (support landed in b9455). Update "
                "llama.cpp, or set the KV cache type to f16, or turn off Tensor "
                "Parallelism in the model settings."
            )

        # llama.cpp prints the bad magic with %c, so a binary header shows as garbage.
        if "invalid magic characters" in lowered:
            # Not necessarily the main model: the projector and drafter report this too.
            base = os.path.basename(gguf_path) if gguf_path else ""
            named = f", loading {base}" if base else ""
            # A ._ path proves the volume keeps xattrs in AppleDouble companions; dot_clean fixes it.
            remedy = (
                'This volume has no native extended attributes, so macOS keeps them in "._" '
                'companions: run "dot_clean -m" on the folder to remove them, or keep models '
                "on an APFS disk."
                if base.startswith("._")
                else "Re-download the model, or pick a different file."
            )
            return LlamaCppBackend._with_startup_diagnostics(
                f"llama-server opened a file that is not a GGUF{named}: it does not start with "
                "the GGUF magic. It may be that file or a companion loaded with it, such as a "
                f"vision projector or a drafter, which llama-server does not always name. {remedy}",
                output,
                log_path,
                secrets,
            )

        # Projectors, drafters and shards fail this way too: never name the main GGUF.
        if "data is not within the file bounds" in lowered:
            return LlamaCppBackend._with_startup_diagnostics(
                "A model file appears incomplete or damaged. Check that the download finished.",
                output,
                log_path,
                secrets,
            )

        if "failed to open gguf file" in lowered:
            return LlamaCppBackend._with_startup_diagnostics(
                "Couldn't open a model file. Check it exists and Unsloth has permission to read it.",
                output,
                log_path,
                secrets,
            )

        # GPU evidence on the allocation line itself: a host allocation failure keeps the fallback.
        if LlamaCppBackend._is_gpu_memory_start_failure(scan_tail):
            return LlamaCppBackend._with_startup_diagnostics(
                "Couldn't allocate memory. Try lowering the context length or choosing a smaller model.",
                output,
                log_path,
                secrets,
            )

        # Detect Ollama first so the arch branch keeps the Ollama hint.
        gguf = gguf_path or ""
        is_ollama = (
            ".studio_links" in gguf
            or os.sep + "ollama_links" + os.sep in gguf
            or os.sep + ".cache" + os.sep + "ollama" + os.sep in gguf
            or (model_identifier or "").startswith("ollama/")
        )

        # Unknown-arch: diffusion to Images page, Ollama to Ollama hint, else unsupported; exact match only.
        arch_match = re.search(r"unknown model architecture:\s*'([^']+)'", lowered)
        if arch_match:
            arch = arch_match.group(1)
            if is_audio_cpp_gguf_architecture(arch):
                return (
                    "This is an audio GGUF (speech, music or transcription), which llama-server "
                    "cannot run as a chat/completion model. Open it from Unsloth's Audio page instead."
                )
            if is_foreign_asr_gguf_architecture(arch):
                return (
                    f"'{arch}' is a speech-recognition GGUF for parakeet.cpp / LocalAI, which "
                    "llama-server cannot run as a chat/completion model. Updating llama.cpp will "
                    "not add support for it. For Parakeet in Unsloth, open the Audio page and use "
                    "the curated audio.cpp model (Parakeet-TDT-0.6B-v3-GGUF)."
                )
            if is_speech_gguf_architecture(arch):
                # Name the build that runs it, not a page this file will not appear on.
                return (
                    f"'{arch}' is a text-to-speech GGUF, which llama-server cannot run as a "
                    "chat/completion model. Run the model's Transformers build (for example "
                    "unsloth/csm-1b) from Unsloth's Audio page instead."
                )
            if arch in LlamaCppBackend._UNRUNNABLE_MEDIA_ARCHES:
                return (
                    f"'{arch}' is an image / video generation GGUF, which llama-server "
                    "cannot run as a chat/completion model. Unsloth Studio cannot run "
                    "this architecture at all: neither the Images page nor the Video "
                    "page accepts it."
                )
            if arch in LlamaCppBackend._VIDEO_ARCHES:
                if LlamaCppBackend._video_arch_is_pickable(arch, gguf_path, model_identifier):
                    return (
                        f"'{arch}' is a text-to-video GGUF, which llama-server "
                        "cannot run as a chat/completion model. Open it from "
                        "Unsloth's Video page instead of a chat."
                    )
                return (
                    f"'{arch}' is a text-to-video GGUF, which llama-server cannot run as "
                    "a chat/completion model. Unsloth cannot assemble this particular "
                    "model from a GGUF, so the Video page does not list it either."
                )
            if (
                arch in LlamaCppBackend._IMAGE_ARCHES
                and LlamaCppBackend._image_arch_is_pickable(gguf_path, model_identifier, arch)
            ) or (
                arch in LlamaCppBackend._AMBIGUOUS_IMAGE_ARCHES
                and LlamaCppBackend._ambiguous_image_arch_is_pickable(gguf_path, model_identifier)
            ):
                return (
                    f"'{arch}' is a diffusion (image-generation) GGUF, which "
                    "llama-server cannot run as a chat/completion model. Use "
                    "Unsloth's Images page to generate with local diffusion "
                    "GGUFs such as FLUX and Qwen-Image."
                )
            if arch in LlamaCppBackend._IMAGE_ARCHES:
                return (
                    f"'{arch}' is a diffusion (image-generation) GGUF, which "
                    "llama-server cannot run as a chat/completion model. No image engine "
                    "installed here can build this model family, so Unsloth's Images page "
                    "does not list it either. Updating Unsloth may add support for it."
                )
            if arch in LlamaCppBackend._AMBIGUOUS_IMAGE_ARCHES:
                # Unresolved shared archs are hidden from the Images picker.
                return (
                    f"'{arch}' is a diffusion (image-generation) GGUF, which "
                    "llama-server cannot run as a chat/completion model. Unsloth Studio "
                    f"could not tell which model family this file is -- '{arch}' is "
                    "shared by several -- so the Images page cannot assemble it either."
                )
            if is_ollama:
                return (
                    "Some Ollama models do not work with llama.cpp. Try a "
                    "different model, or use this model directly through "
                    "Ollama instead."
                )
            # Includes an arch the installed build predates.
            return (
                f"The installed llama.cpp does not recognise this GGUF's model "
                f"architecture ('{arch}'). Check the model card for its required runtime. "
                "If the model is newer than this llama.cpp build, updating llama.cpp may "
                "add support; a model requiring a custom runtime may still be incompatible."
            )

        # Other Ollama compat failures, only when the output shows a GGUF compat issue.
        if is_ollama:
            gguf_compat_hints = (
                "key not found",
                "unknown model architecture",
                "failed to load model",
            )
            if any(h in lowered for h in gguf_compat_hints):
                return (
                    "Some Ollama models do not work with llama.cpp. Try a "
                    "different model, or use this model directly through "
                    "Ollama instead."
                )

        # GGUF converted for another llama.cpp format revision (#8513: pre-#24908 MiniMax-M3 lacks indexer keys).
        missing_key = re.search(r"key not found in model:\s*([\w.\-]+)", scan_tail, re.IGNORECASE)
        if missing_key:
            return LlamaCppBackend._with_startup_diagnostics(
                f"This GGUF is missing the metadata key '{_short(missing_key.group(1))}', "
                "which the installed llama.cpp requires to load it; this is not a memory problem. "
                "This can happen with an incompatible conversion, incomplete metadata, or "
                "an adapter selected as a full model. Check the model card for its required "
                "runtime and whether it needs a base model. Use a compatible full or updated "
                "GGUF, or update llama.cpp if the model requires a newer build.",
                output,
                log_path,
                secrets,
            )

        # A type this build lacks (prism-ml PQ2_0 / PTQ1_0 = 142 / 143) or packs differently (group-128 Q2_0, #11259).
        # A truncated download fails later instead, as "data is not within the file bounds".
        # Bundled IQ1_XS..IQ1_XXXS = 64-66, so ids 43-63 fail later as an empty block-size-0 slot (#13908).
        bad_type = re.search(
            r"tensor '([^']*)' (?:has invalid ggml type (\d+)|of type (\d+) .*?not a multiple of block size \(0\))",
            scan_tail,
        )
        bad_layout = re.search(r"tensor '[^']*' has offset \d+, expected \d+", scan_tail)
        if bad_type or bad_layout:
            detail = (
                f"tensor '{_short(bad_type.group(1))}' uses quantization type "
                f"{bad_type.group(2) or bad_type.group(3)}, which this build does not have"
                if bad_type
                else "its tensor layout does not match this build's quantization types"
            )
            return LlamaCppBackend._with_startup_diagnostics(
                "The installed llama.cpp cannot read this GGUF's weights "
                f"({detail}); this is not a memory problem. The file was likely made for "
                "a different llama.cpp build: a newer one, or a publisher's own fork. Pick "
                "another quant of this model, or use the llama.cpp build the "
                "model card names. If the model card says it runs on stock llama.cpp, "
                "the file may be damaged: download it again.",
                output,
                log_path,
                secrets,
            )

        # 127 without any loader line above is not specific to a library: it is
        # also what a shell wrapper entrypoint reports when its exec target is
        # gone (_llama_lib_dir supports those, as does a custom
        # LLAMA_SERVER_PATH), and what a "symbol lookup error" from a mismatched
        # runtime exits with. Name both causes instead of blaming a distro
        # package -- but still keep it off the GGUF and off memory.
        # A ROCm symbol lookup is also 127, but the generic text below would send
        # the user to reinstall a binary Vulkan and a shell launch both run fine.
        _rocm_miss = LlamaCppBackend._bundled_hip_symbol_miss(output or "")
        if _rocm_miss:
            _miss_obj, _miss_sym = _rocm_miss
            return (
                f"llama-server could not start: bundled {_short(os.path.basename(_miss_obj))} "
                f"looked up {_short(_miss_sym)} in a different ROCm than it was built "
                "against. That is a HIP/ROCR version mix, not a missing "
                "binary and not out of VRAM. Try the Vulkan backend, or "
                f"{LlamaCppBackend._runtime_remedy(binary)}."
            )
        if returncode == 127:
            # The updater cannot touch a pinned binary.
            _remedy = (
                "run `unsloth studio update` to reinstall the llama.cpp runtime"
                if LlamaCppBackend._is_unsloth_managed_binary(binary)
                else "reinstall or rebuild that custom llama.cpp"
            )
            return (
                "llama-server exited immediately (status 127): either the "
                "llama-server executable could not be found or run, or one of "
                f"the shared libraries it needs could not be loaded. Check the "
                f"llama-server log, and {_remedy}."
            )

        # SIGKILL with no output is usually the OOM killer.
        if returncode == -9:
            # macOS also SIGKILLs an invalid code signature before any output.
            if sys.platform == "darwin":
                # Log is often empty: name likely causes (incl. macOS SIGKILL); signal 9 may also be a supervisor.
                return LlamaCppBackend._with_startup_diagnostics(
                    "llama-server was stopped by macOS (signal 9) before it "
                    "started. This is most often out of memory, so try a "
                    "smaller or more quantized GGUF, or lower the context "
                    "length. macOS also sends signal 9 when it refuses a "
                    "binary's code signature, so if memory is not the problem, "
                    f"{LlamaCppBackend._runtime_remedy(binary)}.",
                    output,
                    log_path,
                    secrets,
                )
            return (
                "llama-server was stopped by the operating system (signal 9), "
                "most likely out of memory. Try a smaller or more quantized "
                "GGUF, lower the context length, or free memory (on WSL, raise "
                "the memory limit in .wslconfig)."
            )
        # SIGTERM is also how unload/cancel stops the server; report neutrally.
        if returncode == -15:
            return (
                "llama-server was terminated (signal 15) before it became "
                "healthy. If you cancelled or unloaded the model this is "
                "expected; otherwise check the llama-server log for the cause."
            )

        # A startup stall is not a bad GGUF; memory pressure or a proxy intercepting the loopback probe can also cause it.
        if "health check timed out" in lowered:
            return (
                "llama-server started but stopped making progress before it became "
                "healthy on its local /health endpoint. Try a smaller context length or a more "
                "quantized GGUF, and if you use a VPN or HTTP proxy make sure "
                "localhost bypasses it (NO_PROXY=127.0.0.1,localhost)."
            )

        # Fallback: carry the output tail and log path so the report is actionable.
        return LlamaCppBackend._with_startup_diagnostics(
            "llama-server failed to start. "
            "Check that the GGUF file is valid and you have enough memory.",
            output,
            log_path,
            secrets,
        )

    # Bounded so a chatty server cannot flood an API error.
    _STARTUP_TAIL_CHARS = 2000

    # Labels _with_startup_diagnostics writes, used to stay idempotent. Keep in sync with the writer.
    _DIAGNOSTICS_MARKERS = ("llama-server output:", "Full log: ")

    # Token shapes redacted on sight, whatever the variable name.
    _SECRET_VALUE_RES = (
        re.compile(r"hf_(?:oauth_[A-Za-z0-9._~+/=-]{20,}|[A-Za-z0-9]{20,})"),
        re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
        re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
        # HTTP credentials. Full RFC 9110 token68 class so no base64 tail survives.
        re.compile(r"(?i)(?:bearer|basic)\s+[A-Za-z0-9._~+/\-]+=*"),
        # Our own --api-key matches no shape and is in no env var.
        re.compile(r"(?<=--api-key )\S+"),
        re.compile(r"(?<=--api-key=)\S+"),
        # URL userinfo: keep the scheme, drop the credential.
        re.compile(r"(?<=://)[^/@\s]+(?=@)"),
    )

    # NAME = value in shell, JSON or bare form; the unterminated-quote arm must come after the quoted one; names may contain hyphen and dot.
    _SECRET_ASSIGNMENT_RE = re.compile(
        r"""(?P<q>["']?)(?P<name>[A-Za-z_][A-Za-z0-9_.\-]{0,63})(?P=q)
            (?P<sep>[ \t]*[=:][ \t]*)
            (?:(?P<vq>["'])(?P<qval>(?:\\.|(?!(?P=vq))[^\\])*)(?P=vq)
             |  (?P<uq>["'])(?P<uval>[^\r\n]*)
             |  (?P<val>[^\s,;]+))""",
        re.VERBOSE,
    )

    @staticmethod
    def _redact_secret_assignments(text: str) -> str:
        """Replaces whatever follows a secret-looking name, so escaping or quoting cannot hide the value."""
        try:
            from utils.prebuilt.child_env import is_secret_env_name
        except Exception:  # noqa: BLE001 - redaction must never break a load error
            return text

        def sub(m: "re.Match[str]") -> str:
            # Normalise hyphen and dot to underscore before asking the predicate: its markers are underscored, so api-key missed while api_key matched.
            name = m.group("name")
            if not is_secret_env_name(name.replace("-", "_").replace(".", "_")):
                return m.group(0)
            q, vq, uq = m.group("q"), m.group("vq"), m.group("uq")
            if vq:
                body = f"{vq}***{vq}"
            elif uq:
                # Unterminated: keep the opening delimiter, drop the rest; do not invent a closing quote.
                body = f"{uq}***"
            else:
                body = "***"
            return f"{q}{m.group('name')}{q}{m.group('sep')}{body}"

        try:
            return LlamaCppBackend._SECRET_ASSIGNMENT_RE.sub(sub, text)
        except Exception:  # noqa: BLE001
            return text

    @staticmethod
    def _scrub_secret_values(text: str, extra: Sequence[Optional[str]] = ()) -> str:
        """Redacts env values by value and by secret-looking name, since each alone misses something."""
        if not text:
            return text
        cleaned = LlamaCppBackend._redact_secret_assignments(text)
        # Minted secrets plus env values, replaced longest first since credentials can overlap.
        literals = {literal for literal in extra if literal and len(literal) >= 8}
        # Reuse the managed-server denylist by name so the two cannot drift; URL userinfo is withheld whatever its name.
        try:
            from utils.prebuilt.child_env import URL_USERINFO_RE, is_secret_env_name
            for name, value in os.environ.items():
                if not value:
                    continue
                secret_name = is_secret_env_name(name)
                if len(value) < 8:
                    # Too short to replace globally, so redact only next to its (optionally quoted) name.
                    if secret_name:
                        cleaned = re.sub(
                            rf"([\"']?{re.escape(name)}[\"']?\s*[=:]\s*)"
                            rf"([\"']?){re.escape(value)}\2",
                            r"\1\2***\2",
                            cleaned,
                        )
                    continue
                if secret_name or URL_USERINFO_RE.search(value):
                    literals.add(value)
        except Exception:  # noqa: BLE001 - redaction must never break a load error
            pass
        for literal in sorted(literals, key = len, reverse = True):
            cleaned = cleaned.replace(literal, "***")
        # By shape, for credentials not present in our environment to match on.
        for pattern in LlamaCppBackend._SECRET_VALUE_RES:
            cleaned = pattern.sub("***", cleaned)
        return cleaned

    @staticmethod
    def _has_startup_diagnostics(text: str) -> bool:
        """Matches our framing, not bare marker words, so a server echoing them still gets its tail."""
        return text.startswith(LlamaCppBackend._DIAGNOSTICS_MARKERS) or any(
            f"\n\n{marker}" in text for marker in LlamaCppBackend._DIAGNOSTICS_MARKERS
        )

    # Caps window widening so a pathological env value cannot force a full-output scan.
    _MAX_SECRET_SCAN_CHARS = 262144

    @staticmethod
    def _max_secret_len(extra: Sequence[Optional[str]] = ()) -> int:
        """Longest value _scrub_secret_values could be asked to match, capped."""
        longest = 0
        for literal in extra:
            if literal:
                longest = max(longest, len(literal))
        try:
            from utils.prebuilt.child_env import URL_USERINFO_RE, is_secret_env_name
            for name, value in os.environ.items():
                if not value or len(value) <= longest:
                    continue
                if is_secret_env_name(name) or URL_USERINFO_RE.search(value):
                    longest = len(value)
        except Exception:  # noqa: BLE001 - a narrower window still redacts most
            pass
        return min(longest, LlamaCppBackend._MAX_SECRET_SCAN_CHARS)

    @staticmethod
    def _with_startup_diagnostics(
        message: str,
        output: Optional[str],
        log_path: "Optional[Path | str]",
        extra_secrets: Sequence[Optional[str]] = (),
    ) -> str:
        """Appends the output tail and log path to message, returning it unchanged when neither exists; a message already carrying the block is returned as is."""
        if LlamaCppBackend._has_startup_diagnostics(message):
            return message
        parts = [message]
        # Slice before filtering (runaway lines), widened by the longest secret so a long
        # credential cannot straddle the cut and escape redaction.
        _window = LlamaCppBackend._STARTUP_TAIL_CHARS * 4 + LlamaCppBackend._max_secret_len(
            extra_secrets
        )
        raw = (output or "")[-_window:]
        # Strip control characters (progress bars, colour codes) from API error tails; keep newlines and tabs.
        tail = "".join(ch for ch in raw if ch in "\n\t" or ch.isprintable()).strip()
        if tail:
            tail = LlamaCppBackend._scrub_secret_values(tail, extra_secrets)
            tail = tail[-LlamaCppBackend._STARTUP_TAIL_CHARS :].lstrip()
            parts.append(f"llama-server output:\n{tail}")
        if log_path:
            parts.append(f"Full log: {log_path}")
        return "\n\n".join(parts)
