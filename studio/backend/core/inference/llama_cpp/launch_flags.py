# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Launch argument and environment inspection."""

import ast
import functools
import os
import re
import sys
import threading

from pathlib import Path
from typing import (
    Iterable,
    List,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _effective_tensor_parallel,
    _flag_name,
    _GPU_LAYER_FLAGS,
    parse_gpu_layers_override,
    parse_split_mode_override,
)

from .memory_policy import (
    _CPU_FFN_COUNT_FLAGS,
    _CPU_MOE_COUNT_FLAGS,
    _CPU_PLACEMENT_PRESENCE_FLAGS,
    _GENERATION_CPU_AFFINITY_FLAGS,
    _THREAD_OVERRIDE_FLAGS,
    _active_vram_fraction,
    _extra_args_set_any_flag,
    _extra_args_set_spec_type,
    _vram_usable_mib,
)
import time
from .binary_runtime import (
    _llama_lib_dir,
    _resolve_llama_binary,
)


def _linux_math_core_count(
    cpu_root: Path = Path("/sys/devices/system/cpu"),
    logical_cpus: Optional[int] = None,
    vendor_id: Optional[str] = None,
    event_source_root: Path = Path("/sys/bus/event_source/devices"),
    pinnable_cpus: Optional[set[int]] = None,
    probe_affinity: Optional[bool] = None,
) -> Optional[int]:
    """Physical cores as llama.cpp counts them: SMT siblings dropped; hybrid Intel keeps only P-cores; AMD capacity classes keep every physical core."""
    if logical_cpus is not None and logical_cpus < 1:
        return None

    def cpu_list(path: Path) -> Optional[set[int]]:
        try:
            raw = path.read_text(encoding = "utf-8").strip()
            if not raw:
                return set()
            result: set[int] = set()
            for part in raw.split(","):
                bounds = [int(value) for value in part.split("-")]
                if len(bounds) == 1 and bounds[0] >= 0:
                    result.add(bounds[0])
                elif len(bounds) == 2 and 0 <= bounds[0] <= bounds[1]:
                    result.update(range(bounds[0], bounds[1] + 1))
                else:
                    return None
            return result
        except (OSError, ValueError):
            return None

    if vendor_id is None:
        try:
            vendor_match = re.search(
                r"^vendor_id\s*:\s*(\S+)",
                Path("/proc/cpuinfo").read_text(encoding = "utf-8"),
                flags = re.MULTILINE,
            )
            vendor_id = vendor_match.group(1) if vendor_match else ""
        except OSError:
            vendor_id = ""
    siblings: list[str] = []
    capacities: list[Optional[int]] = []
    cpu = 0
    while logical_cpus is None or cpu < logical_cpus:
        cpu_path = cpu_root / f"cpu{cpu}"
        try:
            sibling_set = (
                (cpu_path / "topology" / "thread_siblings").read_text(encoding = "utf-8").strip()
            )
        except OSError:
            break
        if not sibling_set:
            break
        siblings.append(sibling_set)
        try:
            capacities.append(int((cpu_path / "cpu_capacity").read_text(encoding = "utf-8").strip()))
        except (OSError, ValueError):
            capacities.append(None)
        cpu += 1
    if not siblings:
        return None
    online_cpus = cpu_list(cpu_root / "online")
    active_siblings = {
        sibling_set
        for cpu, sibling_set in enumerate(siblings)
        if not online_cpus or cpu in online_cpus
    }
    if not active_siblings:
        return None
    physical_count = len(active_siblings)
    if vendor_id == "GenuineIntel":
        native_count = len(online_cpus) if online_cpus else len(siblings)

        def hybrid_math_count(performance_cpus: set[int]) -> int:
            should_probe = pinnable_cpus is None and (
                cpu_root == Path("/sys/devices/system/cpu")
                if probe_affinity is None
                else probe_affinity
            )

            def count(pin_cpus: bool) -> int:
                result = 0
                cpu = 0
                while cpu < native_count:
                    if online_cpus and cpu not in online_cpus:
                        return physical_count
                    if pinnable_cpus is not None and cpu not in pinnable_cpus:
                        return physical_count
                    if pin_cpus:
                        try:
                            os.sched_setaffinity(0, {cpu})
                        except OSError:
                            return physical_count
                    if cpu not in performance_cpus:
                        cpu += 1
                        continue
                    result += 1
                    cpu += 2
                return result or physical_count

            if not should_probe:
                return count(False)
            if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
                return physical_count

            answer = [physical_count]

            def probe() -> None:
                try:
                    saved_affinity = os.sched_getaffinity(0)
                except OSError:
                    return
                try:
                    answer[0] = count(True)
                finally:
                    try:
                        os.sched_setaffinity(0, saved_affinity)
                    except OSError:
                        pass

            worker = threading.Thread(target = probe)
            try:
                worker.start()
                worker.join()
            except RuntimeError:
                return physical_count
            return answer[0]

        core_mask = cpu_list(event_source_root / "cpu_core" / "cpus")
        atom_path = event_source_root / "cpu_atom"
        lowpower_path = event_source_root / "cpu_lowpower"
        hybrid_pmu = atom_path.exists() or lowpower_path.exists()
        if core_mask:
            return hybrid_math_count(core_mask)
        if hybrid_pmu:
            atom_mask = cpu_list(atom_path / "cpus") or cpu_list(lowpower_path / "cpus")
            if core_mask == set() and atom_mask:
                return physical_count
            return 1
        active_capacities = [
            capacity
            for cpu, capacity in enumerate(capacities)
            if not online_cpus or cpu in online_cpus
        ]
        if (
            all(capacity is not None for capacity in active_capacities)
            and len(set(active_capacities)) > 1
        ):
            fastest = max(capacity for capacity in active_capacities if capacity is not None)
            return hybrid_math_count(
                {
                    cpu
                    for cpu, capacity in enumerate(capacities)
                    if (not online_cpus or cpu in online_cpus) and capacity == fastest
                }
            )
    return physical_count


def _spilled_decode_threads(
    n_threads: Optional[int] = None, extra_args: Optional[Iterable[str]] = None
) -> Optional[int]:
    """Physical cores as llama.cpp counts them, not os.cpu_count; inherited LLAMA_ARG_THREADS is ignored; explicit affinity or SMT oversubscription returns None."""
    requested: Optional[int] = None
    if n_threads is not None and n_threads > 0:
        requested = int(n_threads)
    args = [str(arg) for arg in extra_args] if extra_args else []
    if any(arg.partition("=")[0] in _GENERATION_CPU_AFFINITY_FLAGS for arg in args):
        return None
    for i, arg in enumerate(args):
        name, _, inline = arg.partition("=")
        if name not in _THREAD_OVERRIDE_FLAGS:
            continue
        value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
        try:
            requested = int(value.strip())
        except (TypeError, ValueError):
            continue
    physical: Optional[int] = None
    logical: Optional[int] = None
    affinity_count: Optional[int] = None
    if sys.platform.startswith("linux"):
        if hasattr(os, "sched_getaffinity"):
            try:
                affinity_count = len(os.sched_getaffinity(0))
            except OSError:
                pass
        if os.uname().machine.lower() in ("x86_64", "amd64"):
            physical = _linux_math_core_count()
    try:
        import psutil

        answer = psutil.cpu_count(logical = True)
        if answer and answer > 0:
            logical = int(answer)
        if physical is None:
            answer = psutil.cpu_count(logical = False)
            if answer and answer > 0:
                physical = int(answer)
    except Exception:
        pass
    if logical is None:
        logical = os.cpu_count()
    if affinity_count is not None and logical is not None and affinity_count < logical:
        return None
    if physical is None:
        physical = logical if logical and logical <= 4 else (logical // 2 if logical else 4)
    if requested is None:
        return physical
    actual = requested if requested > 0 else (logical or os.cpu_count() or physical)
    if actual > physical:
        return None
    return max(1, actual)


def _strip_flag_pairs(args: Iterable[str], flags: frozenset[str]) -> list[str]:
    """Drop each ``flag value`` pair naming one of ``flags``."""
    out: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if str(arg) in flags:
            skip = True
            continue
        out.append(str(arg))
    return out


# common_params defaults in the bundled llama.cpp runtime.
_DEFAULT_LLAMA_N_BATCH = 2048
_DEFAULT_LLAMA_N_UBATCH = 512
# Non-causal image decoding needs one image chunk per micro-batch; limits from clip.cpp.
_MMPROJ_NON_CAUSAL_IMAGE_TOKENS = {
    "gemma4v": 1120,
    "gemma4uv": 1120,
    "gemma3": 256,
    "deepseek4v": 384,
}
# A projector whose family we cannot read gets headroom instead of a ceiling.
_MMPROJ_UNKNOWN_UBATCH = 2048
_GEMMA4V_CAUSAL_TEXT_N_EMBD = frozenset({1536, 2560})  # E2B and E4B
_LLAMA_ARG_TRUE_VALUES = frozenset({"on", "enabled", "true", "1"})
_LLAMA_ARG_FALSE_VALUES = frozenset({"off", "disabled", "false", "0"})
_LLAMA_ARG_AUTO_VALUES = frozenset({"auto", "-1"})
_LLAMA_ARG_TRUE_OR_AUTO_VALUES = _LLAMA_ARG_TRUE_VALUES | _LLAMA_ARG_AUTO_VALUES
_LLAMA_ARG_TRUE_FALSE_AUTO_VALUES = _LLAMA_ARG_TRUE_OR_AUTO_VALUES | _LLAMA_ARG_FALSE_VALUES


def _diffusion_manual_ngl(gpu_memory_mode: str, gpu_layers: int) -> Optional[int]:
    """The --ngl the diffusion runner launches with, set only by an explicit manual count, else None; shared with dedup guards so an inert pref cannot loop."""
    return int(gpu_layers) if (gpu_memory_mode == "manual" and int(gpu_layers) >= 0) else None


def _scale_diffusion_required_gb(required_gb: float, ngl: int, n_layers: Optional[int]) -> float:
    """Only ngl resident layers compete for VRAM, so sizing the whole file 409s a split that fits; an unknown layer count keeps the full estimate."""
    if not n_layers or n_layers <= 0 or ngl >= n_layers:
        return required_gb
    return required_gb * (ngl / n_layers)


# The runner's own CUDA context, which the free reading taken before spawn cannot see.
_DIFFUSION_RUNNER_OVERHEAD_MIB = 512.0
# Context the fitted split leaves room for when the runner auto-sizes (maxtok 0).
_DIFFUSION_FIT_CTX = 4096


def _diffusion_auto_ngl(
    file_bytes: int,
    n_layers: Optional[int],
    n_head: Optional[int],
    maxtok: int,
    free_mib: float,
    total_mib: float,
    *,
    sysmem_fallback: bool = False,
) -> Optional[int]:
    """GPU layers for an Auto-mode diffusion load, or None to keep every layer on the GPU.

    Only a GGUF that cannot fit moves: all layers die in cudaMalloc, or on Windows spill into
    shared system memory until RAM and the page file run out (#6456). The split leaves room
    for the compute buffer (measured 1.75x the [n_head, N, N] fp32 scores) and the 8% of VRAM
    the runner's context probe keeps free (diffusion-gemma-visual-server.cpp)."""
    if not file_bytes or not n_layers or n_layers <= 0 or (free_mib <= 0 and total_mib <= 0):
        return None
    usable = _vram_usable_mib(
        free_mib, total_mib, _active_vram_fraction(), sysmem_fallback = sysmem_fallback
    )
    usable -= _DIFFUSION_RUNNER_OVERHEAD_MIB
    file_mib = file_bytes / (1024 * 1024)
    if file_mib <= usable:
        return None
    # Mirrors unsloth_zoo's _canvas_maxtok: larger requests are sent as 0 (auto-size).
    n_ctx = maxtok if 0 < maxtok <= 8192 else _DIFFUSION_FIT_CTX
    scores_mib = max(1, n_head or 1) * n_ctx * n_ctx * 4 / (1024 * 1024)
    ctx_reserve = 1.75 * scores_mib + 0.08 * (total_mib if total_mib > 0 else free_mib)
    # The whole file over the block count also charges token_embd, which stays on the CPU.
    per_layer_mib = file_mib / n_layers
    return max(0, min(n_layers - 1, int((usable - ctx_reserve) // per_layer_mib)))


def _source_declares_option(source: str, option: str) -> bool:
    """Parsed with ast, not grepped (substring hits comments, misses quoted flags); falls back to a substring scan when the file will not parse."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return option in source
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == option:
                return True
    return False


def _shim_supports_ngl(shim_cmd: List[str]) -> bool:
    """Reads the source, not a spawned probe; an unsloth_zoo lacking --ngl makes argparse exit."""
    # Key on the command shape, not the suffix, and probe that exact override file.
    if len(shim_cmd) >= 2 and "-m" not in shim_cmd:
        candidates = [Path(shim_cmd[-1])]
    else:
        candidates = []
        try:
            import importlib.util
            spec = importlib.util.find_spec("unsloth_zoo")
            if spec is not None and spec.submodule_search_locations:
                pkg_dir = Path(list(spec.submodule_search_locations)[0])
                candidates.append(pkg_dir / "diffusion_studio" / "shim.py")
        except Exception:
            return False
    for path in candidates:
        try:
            source = path.read_text(encoding = "utf-8", errors = "ignore")
        except OSError:
            continue
        if _source_declares_option(source, "--ngl"):
            return True
    return False


def _visual_server_splits_gpus(visual_bin: Optional[str]) -> bool:
    """Whether the diffusion visual server survives a model split over several GPUs.

    Builds before the fix abort on the first request once split: with the prompt-KV store
    off (it is single-device) every [prompt|canvas] row asked for logits past the server's
    canvas-sized output cap. Fixed builds answer ``--capabilities`` with ``multi_gpu_split``;
    read that literal from the binary rather than spawning it, since an old build takes
    10-30 s to fail on the unknown argument.
    """
    if not visual_bin:
        return False
    try:
        st = os.stat(visual_bin)
    except OSError:
        return False
    return _binary_has_marker(visual_bin, st.st_mtime_ns, st.st_size, b"multi_gpu_split")


@functools.lru_cache(maxsize = 16)
def _binary_has_marker(path: str, _mtime_ns: int, _size: int, marker: bytes) -> bool:
    try:
        with open(path, "rb") as f:
            return marker in f.read()
    except OSError:
        return False


def _fit_measures_borrowed_drafts(bin_path: Optional[str]) -> bool:
    """Whether ``bin_path``'s fitter sizes a borrowed-tensor draft itself; fails closed."""
    if not bin_path:
        return False
    candidates = [Path(bin_path)]
    try:
        # A wrapper script execs the real server.
        server = Path(_resolve_llama_binary(bin_path))
        candidates.append(server)
        lib_dir = _llama_lib_dir(bin_path)
        for lib in _linked_llama_libs(server, lib_dir):
            candidates.append(lib)
            candidates.extend(l for l in _linked_llama_libs(lib, lib_dir) if l not in candidates)
    except Exception:  # noqa: BLE001 -- an unreadable lib dir is "not known"
        pass
    seen: set = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            st = candidate.stat()
        except OSError:
            continue
        if _binary_has_marker(
            str(candidate), st.st_mtime_ns, st.st_size, _FIT_MEASURES_BORROWED_DRAFT_MARKER
        ):
            return True
    return False


def _distinct_device_count(devices) -> int:
    """Distinct names in a ``--device`` list: a repeated card is one memory pool."""
    return len({d.strip().lower() for d in str(devices).split(",") if d.strip()})


def _draft_pin_device_index(
    pin: Optional[str], spawn_devices: Optional[str], device_count: int
) -> Optional[int]:
    """The child device index a one-card draft pin names, or None."""
    names = {d.strip().lower() for d in (pin or "").split(",") if d.strip()}
    if len(names) != 1:
        return None
    (name,) = names
    if spawn_devices:
        listed = [d.strip().lower() for d in spawn_devices.split(",") if d.strip()]
        if name not in listed or len(set(listed)) != len(listed):
            return None
        return listed.index(name)
    if not _CUDA_LIKE_DEVICE_RE.match(name):
        return None
    k = int(re.sub(r"\D", "", name))
    return k if 0 <= k < device_count else None


def _sidecar_borrows_target_tensors(path: Optional[str]) -> bool:
    """Whether the draft borrows token_embd or output, which the fitter cannot measure
    (src/llama-context.cpp)."""
    if not path:
        return True
    try:
        from utils.models.gguf_metadata import _gguf_shard_has_named_tensor
        from utils.models.model_config import colocated_split_shards

        shards, complete = colocated_split_shards(Path(path))
        if not complete or not shards:
            return True
        return any(
            not any(_gguf_shard_has_named_tensor(str(shard), name) is True for shard in shards)
            for name in ("token_embd.weight", "output.weight")
        )
    except Exception:
        return True


def _sidecar_borrows_output(path: Optional[str]) -> bool:
    """Whether the draft's logits use the target's output head (src/models/dflash.cpp); unknown
    counts as borrowed."""
    if not path:
        return True
    try:
        from utils.models.gguf_metadata import _gguf_shard_has_named_tensor
        from utils.models.model_config import colocated_split_shards

        shards, complete = colocated_split_shards(Path(path))
        return not (
            complete
            and any(
                _gguf_shard_has_named_tensor(str(shard), "output.weight") is True
                for shard in shards
            )
        )
    except Exception:
        return True


_CUDA_LIKE_DEVICE_RE = re.compile(r"(?i)(cuda|rocm|hip)\d+$")


def _linked_llama_libs(binary: Path, lib_dir: Path) -> list:
    """The server-impl / common libraries ``binary`` links by name (DT_NEEDED, LC_LOAD_DYLIB, PE
    imports), not stray siblings."""
    try:
        st = binary.stat()
        key = (str(binary), st.st_mtime_ns, st.st_size)
    except OSError:
        return []
    embedded = _LINKED_LIB_NAMES_CACHE.get(key)
    if embedded is None:
        try:
            data = binary.read_bytes()
        except OSError:
            return []
        embedded = tuple(
            tuple(n for n in names if n.encode() in data)
            for names in (_LLAMA_SERVER_IMPL_LIB_NAMES, _LLAMA_COMMON_LIB_NAMES)
        )
        _LINKED_LIB_NAMES_CACHE[key] = embedded
    linked = []
    for names in embedded:
        hit = next((n for n in names if (lib_dir / n).exists()), None)
        if hit is not None:
            linked.append(lib_dir / hit)
    return linked


# (path, mtime_ns, size) -> embedded library names; the server binary is ~100 MB.
_LINKED_LIB_NAMES_CACHE: dict = {}


# The server links libllama-server-impl, which links libllama-common.
_LLAMA_SERVER_IMPL_LIB_NAMES = (
    "libllama-server-impl.so",
    "libllama-server-impl.dylib",
    "llama-server-impl.dll",
)


_LLAMA_COMMON_LIB_NAMES = (
    "libllama-common.so.0",
    "libllama-common.so",
    "libllama-common.0.dylib",
    "libllama-common.dylib",
    "llama-common.dll",
)


# No --help flag or build number announces it: read the trace string the new fit path compiles in.
_FIT_MEASURES_BORROWED_DRAFT_MARKER = (
    b"measuring the extra model next to an allocation-free target context"
)


def _swa_full_from_args_or_env(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether llama.cpp receives the enable-only full-size SWA option."""
    if _extra_args_set_any_flag(extra_args, {"--swa-full"}):
        return True
    value = (os.environ if env is None else env).get("LLAMA_ARG_SWA_FULL")
    return value in _LLAMA_ARG_TRUE_VALUES


def _reasoning_from_env(env: Mapping[str, str]) -> Optional[bool]:
    value = env.get("LLAMA_ARG_REASONING")
    if value in _LLAMA_ARG_TRUE_VALUES:
        return True
    if value in _LLAMA_ARG_FALSE_VALUES:
        return False
    return None


def _template_preserves_thinking_by_default(template: str, context: Mapping) -> bool:
    """Whether ``template`` replays an earlier turn's reasoning with preserve_thinking
    omitted (Qwen3.8's own does, ``|default(false)`` does not); False if unrenderable."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def _raise(message):
        raise ValueError(message)

    marker = "unsloth-preserve-thinking-probe"
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": marker},
        {"role": "user", "content": "q2"},
    ]
    try:
        jinja = ImmutableSandboxedEnvironment(
            trim_blocks = True,
            lstrip_blocks = True,
            extensions = ["jinja2.ext.loopcontrols", "jinja2.ext.do"],
        )
        jinja.globals.update(raise_exception = _raise, strftime_now = lambda fmt: time.strftime(fmt))
        rendered = jinja.from_string(template).render(
            **{**context, "messages": messages, "add_generation_prompt": True}
        )
    except Exception:
        return False
    return marker in rendered


def _reasoning_preserve_from_args_or_env(
    extra_args: Optional[Iterable[str]], env: Mapping[str, str]
) -> Optional[bool]:
    """The user's --[no-]reasoning-preserve choice, else None: last flag wins, as in
    llama.cpp's parser, and the extras beat the inherited LLAMA_ARG_REASONING_PRESERVE."""
    choice: Optional[bool] = None
    for raw in extra_args or ():
        flag = _flag_name(str(raw))
        if flag == "--reasoning-preserve":
            choice = True
        elif flag == "--no-reasoning-preserve":
            choice = False
    if choice is not None:
        return choice
    value = env.get("LLAMA_ARG_REASONING_PRESERVE")
    if value in _LLAMA_ARG_TRUE_VALUES:
        return True
    if value in _LLAMA_ARG_FALSE_VALUES:
        return False
    return None


def _reasoning_effort_from_env(env: Mapping[str, str]) -> Optional[str]:
    value = env.get("LLAMA_ARG_REASONING_EFFORT")
    return None if value == "default" else value


def _env_asks_for_the_native_context(env: Optional[Mapping[str, str]] = None) -> bool:
    """LLAMA_ARG_CTX_SIZE=0 pins the native context, as -c 0 does; llama.cpp rejects unparseable ones."""
    value = (os.environ if env is None else env).get("LLAMA_ARG_CTX_SIZE")
    try:
        return int(str(value).strip()) == 0
    except (AttributeError, TypeError, ValueError):
        return False


def _env_ctx_checkpoints_override(env: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """Env count is explicit: an emitted --ctx-checkpoints would overrule it, so the auto cap yields."""
    value = (os.environ if env is None else env).get("LLAMA_ARG_CTX_CHECKPOINTS")
    try:
        count = int(str(value).strip())
    except (AttributeError, TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _kv_offload_from_args(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """LLAMA_ARG_NO_KV_OFFLOAD counts by presence, even when empty, and forces KV offload off; a CLI -kvo beats a false LLAMA_ARG_KV_OFFLOAD."""
    source_env = os.environ if env is None else env
    enabled = True
    if source_env.get("LLAMA_ARG_NO_KV_OFFLOAD") is not None:
        enabled = False
    else:
        value = source_env.get("LLAMA_ARG_KV_OFFLOAD")
        if value in _LLAMA_ARG_FALSE_VALUES:
            enabled = False
        elif value in _LLAMA_ARG_TRUE_VALUES:
            enabled = True
    for raw in extra_args or ():
        flag = _flag_name(str(raw))
        if flag in {"-kvo", "--kv-offload"}:
            enabled = True
        elif flag in {"-nkvo", "--no-kv-offload"}:
            enabled = False
    return enabled


def _is_positive_int(value: Optional[str]) -> bool:
    """Whether ``value`` parses as an int > 0 (llama.cpp's std::stoi)."""
    try:
        return int(str(value).strip()) > 0
    except (TypeError, ValueError):
        return False


def _args_place_tensors_on_cpu(extra_args: Optional[Iterable[str]] = None) -> bool:
    """True when extras keep weights on CPU whatever the layer count says; a CPU-MoE or CPU-FFN count counts only when positive (-ncmoe 0 places nothing)."""
    values = [str(raw) for raw in extra_args or ()]
    for i, raw in enumerate(values):
        flag = _flag_name(raw)
        # Any -ot counts, even if it matches nothing: non-empty is enough, and the pattern is not guessed at.
        if flag in _CPU_PLACEMENT_PRESENCE_FLAGS:
            return True
        if flag in _CPU_MOE_COUNT_FLAGS | _CPU_FFN_COUNT_FLAGS:
            _, eq, inline = raw.partition("=")
            if _is_positive_int(inline if eq else (values[i + 1] if i + 1 < len(values) else "")):
                return True
    return False


def _env_places_tensors_on_cpu(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when inherited env keeps weights on CPU whatever argv says; it survives argv stripping, and the host-residency gate must not miss it."""
    source_env = os.environ if env is None else env
    return bool(
        source_env.get("LLAMA_ARG_OVERRIDE_TENSOR")
        or source_env.get("LLAMA_ARG_CPU_MOE") in _LLAMA_ARG_TRUE_VALUES
        or _is_positive_int(source_env.get("LLAMA_ARG_N_CPU_MOE"))
        or _is_positive_int(source_env.get("LLAMA_ARG_N_CPU_FFN"))
    )


def _env_fixes_gpu_layers(env: Optional[Mapping[str, str]] = None) -> bool:
    """Any inherited count but auto/-1 overrides the fit; empty counts as unset (stoi would throw)."""
    raw = (os.environ if env is None else env).get("LLAMA_ARG_N_GPU_LAYERS")
    value = str(raw or "").strip().lower()
    return bool(value) and value not in _LLAMA_ARG_AUTO_VALUES


def _pipeline_parallel_disabled_by_args(
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    n_layers: Optional[int] = None,
) -> bool:
    """Pipelining needs -sm layer, KV offload on, no tensor overrides (-ot, -cmoe, -ncmoe, not -otd); ambiguous input stays pipelined, as under-reserving OOMs."""
    source_env = os.environ if env is None else env
    if _env_places_tensors_on_cpu(source_env):
        return True
    if not _kv_offload_from_args(extra_args, env = env):
        return True
    # -sm tensor reaches here only after a downgrade strips it, so only none/row disable it.
    split_mode = parse_split_mode_override(extra_args)
    if split_mode is not None and split_mode.strip().lower() not in {"layer", "tensor"}:
        return True
    # Pipelining needs n_gpu_layers > n_layer_all (block_count, not the MTP-less hparams n_layer); a user -ngl that cannot exceed it turns it off.
    if n_layers:
        try:
            gpu_layers = parse_gpu_layers_override(extra_args)
        except ValueError:
            gpu_layers = None  # malformed: unknown, so keep the step
        if gpu_layers is not None and 0 <= gpu_layers <= n_layers:
            return True
    # Same predicate as the Model Memory host-residency gate; -ncmoe 0 pushes no override, so pipelining stays on.
    return _args_place_tensors_on_cpu(extra_args)


def _kv_unified_from_args(
    extra_args: Optional[Iterable[str]],
    default: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Resolve llama.cpp's environment and last-wins unified KV flags."""
    enabled = False
    value = (os.environ if env is None else env).get("LLAMA_ARG_KV_UNIFIED")
    if value in _LLAMA_ARG_TRUE_VALUES:
        enabled = True
    elif value in _LLAMA_ARG_FALSE_VALUES:
        enabled = False
    if default:
        # Unsloth's managed --kv-unified is appended after environment parsing and before user extras.
        enabled = True
    for raw in extra_args or ():
        flag = _flag_name(str(raw))
        if flag in {"-kvu", "--kv-unified"}:
            enabled = True
        elif flag in {"-no-kvu", "--no-kv-unified"}:
            enabled = False
    return enabled


# llama.cpp forces flash attention off for Grok (KQ logit softcap).
_FLASH_ATTN_INCOMPATIBLE_ARCHITECTURES = frozenset({"grok"})


def _architecture_forces_flash_attn_off(architecture: Optional[str]) -> bool:
    """Whether llama.cpp will disable flash attention for this architecture on its own."""
    return str(architecture or "").strip().lower() in _FLASH_ATTN_INCOMPATIBLE_ARCHITECTURES


def _planned_flash_attn_state(
    extra_args: Optional[Iterable[str]] = None,
    *,
    planned_cache_types: Optional[tuple[str, str]] = None,
    supports_flash_attn: bool = True,
    tensor_parallel: bool = False,
    architecture: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """One answer for the estimate and for the argv: the estimator floors V at f16 and pads
    it model-wide when flash attention is off, so resolving the two separately prices a
    different load, not a conservative one (#9697, #10489)."""
    if not supports_flash_attn:
        return False
    if _architecture_forces_flash_attn_off(architecture):
        # Ahead of every rule below, as llama.cpp puts it ahead of both its own upgrades.
        return False
    v_type = (planned_cache_types[1] if planned_cache_types else "f16") or "f16"
    quantized_v = str(v_type).strip().lower() not in {"f16", "bf16", "f32"}
    if quantized_v:
        # Not a choice: llama.cpp turns it on itself rather than refusing the load.
        return True
    # Prepended: load_model appends a managed --flash-attn on that only user extras beat.
    effective_args = ["--flash-attn", "on", *(str(arg) for arg in extra_args or ())]
    if _asked_for_auto_flash_attn(effective_args, env = env):
        if _effective_tensor_parallel(extra_args, tensor_parallel, env):
            # Not undecided: llama.cpp upgrades AUTO to ENABLED under SPLIT_MODE_TENSOR.
            return True
        # ``auto`` is decided at load time and can come back no; size for the costlier answer.
        return False
    return _flash_attn_enabled_from_args(effective_args, default = True, env = env)


def _user_fit_disabled(
    extra_args: Optional[Iterable[str]] = None, *, env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether the USER has turned llama.cpp's fitter off, last-wins: Unsloth's own token
    comes first and a retry rewrites it, so only a user's later value survives a re-place."""
    values = [str(arg) for arg in extra_args] if extra_args else []
    asked: Optional[str] = None
    inherited = (os.environ if env is None else env).get("LLAMA_ARG_FIT")
    if inherited is not None and str(inherited).strip():
        asked = str(inherited).strip()
    for i, raw in enumerate(values):
        if _flag_name(raw) not in {"-fit", "--fit"}:
            continue
        _, eq, inline = raw.partition("=")
        value = inline if eq else "on"
        if not eq and i + 1 < len(values):
            value = values[i + 1]
        asked = value
    return str(asked).strip().lower() in {"off", "0", "false", "no", "disabled"}


def _placement_is_fitter_proof(
    extra_args: Optional[Iterable[str]] = None,
    *,
    gpu_layers: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether a layer count llama.cpp's fitter refuses to move owns this placement:
    ``common_params_fit_impl`` throws "n_gpu_layers already set by user" (common/fit.cpp:377)
    for any value but -1, so re-enabling the fitter for a respawn buys nothing."""
    if gpu_layers is not None and gpu_layers >= 0:
        return True
    values = [str(arg) for arg in extra_args] if extra_args else []
    for i, raw in enumerate(values):
        if _flag_name(raw) not in _GPU_LAYER_FLAGS:
            continue
        _, eq, inline = raw.partition("=")
        value = inline if eq else ""
        if not eq and i + 1 < len(values):
            value = values[i + 1]
        value = value.strip().lower()
        if value and value not in _LLAMA_ARG_AUTO_VALUES:
            return True
    return bool(_env_fixes_gpu_layers(env))


def _reserved_flash_attn_state(
    planned: bool,
    extra_args: Optional[Iterable[str]] = None,
    *,
    tensor_parallel: bool = False,
    gpu_layers: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """A flash-attn respawn pads V model-wide and may not be re-placed, so keep the planned state; tensor mode has no respawn, so skip this."""
    if not planned:
        return planned
    if _effective_tensor_parallel(extra_args, tensor_parallel, env):
        return planned
    if _user_fit_disabled(extra_args, env = env):
        return False
    if _placement_is_fitter_proof(extra_args, gpu_layers = gpu_layers, env = env):
        return False
    return planned


def _asked_for_auto_flash_attn(
    extra_args: Optional[Iterable[str]] = None, *, env: Optional[Mapping[str, str]] = None
) -> bool:
    """``_flash_attn_enabled_from_args`` folds auto into True because it answers what the
    argv will SAY; sizing needs to know nobody has decided yet."""
    asked: Optional[str] = None
    value = (os.environ if env is None else env).get("LLAMA_ARG_FLASH_ATTN")
    if value in _LLAMA_ARG_TRUE_FALSE_AUTO_VALUES:
        asked = value
    values = [str(arg) for arg in extra_args] if extra_args else []
    for i, raw in enumerate(values):
        if _flag_name(raw) not in {"-fa", "--flash-attn"}:
            continue
        _, eq, inline = raw.partition("=")
        value = inline if eq else "on"
        if not eq and i + 1 < len(values) and values[i + 1] in _LLAMA_ARG_TRUE_FALSE_AUTO_VALUES:
            value = values[i + 1]
        asked = value
    return asked in _LLAMA_ARG_AUTO_VALUES


def _flash_attn_enabled_from_args(
    args: Optional[Iterable[str]],
    default: bool = True,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Resolve llama.cpp's environment and last-wins flash-attention settings."""
    enabled = default
    # llama.cpp applies the env before argv, so the CLI still wins.
    value = (os.environ if env is None else env).get("LLAMA_ARG_FLASH_ATTN")
    if value in _LLAMA_ARG_FALSE_VALUES:
        enabled = False
    elif value in _LLAMA_ARG_TRUE_OR_AUTO_VALUES:
        enabled = True
    values = [str(arg) for arg in args] if args else []
    for i, raw in enumerate(values):
        if _flag_name(raw) not in {"-fa", "--flash-attn"}:
            continue
        _, eq, inline = raw.partition("=")
        value = inline if eq else "on"
        if not eq and i + 1 < len(values) and values[i + 1] in _LLAMA_ARG_TRUE_FALSE_AUTO_VALUES:
            value = values[i + 1]
        if value in _LLAMA_ARG_FALSE_VALUES:
            enabled = False
        elif value in _LLAMA_ARG_TRUE_OR_AUTO_VALUES:
            enabled = True
    return enabled


def _effective_spec_type(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Display label only: llama.cpp accumulates spec types, so ask _extra_args_requests_mtp for MTP."""
    args = [str(a) for a in extra_args] if extra_args else []
    cli_present = False
    cli_value: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag == "--spec-default":
            cli_present = True
            cli_value = "default"
            continue
        if flag != "--spec-type":
            continue
        cli_present = True
        cli_value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
    if cli_present:
        return cli_value
    return (os.environ if env is None else env).get("LLAMA_ARG_SPEC_TYPE")


def _child_spec_env(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Mapping[str, str]:
    """Spec env the child sees: empty unless extras name --spec-type (launch scrubs LLAMA_ARG_SPEC_*); predict from this, not os.environ."""
    if not _extra_args_set_spec_type(extra_args):
        return {}
    return os.environ if env is None else env


# Spec types giving the target context recurrent-state rollback snapshots; both MTP spellings are listed since builds advertise either name.
_TARGET_ROLLBACK_SPEC_TYPES = frozenset(
    {"mtp", "draft-mtp", "draft-eagle3", "draft-dflash", "draft-dspark"}
)


# Mirrors llama.cpp is_recr_impl for SSM headers with no full_attention_interval.
_SSM_EVERY_LAYER_ARCHS = frozenset({"mamba", "mamba2", "falcon-h1"})
# Nemotron-H also zeroes head_count_kv on its MLP/MoE layers; its SSM layers have no FFN.
_SSM_FFN_FREE_LAYER_ARCHS = frozenset({"nemotron_h", "nemotron_h_moe"})


# Archs whose target KV excludes embedded MTP/NextN blocks (per-arch llama_kv_cache filter).
# Fail closed: unlisted archs keep every block; under-reserving OOMs the load.
_TARGET_KV_EXCLUDES_NEXTN_ARCHS = frozenset(
    {
        # Hybrid attention + recurrent, llama-model.cpp:2289
        "qwen3next",
        "qwen35",
        "qwen35moe",
        "minimax-01",
        "nemotron_h",
        "nemotron_h_moe",
        # MLA/DSA trunk with a dense MTP head, llama-model.cpp:2129
        "glm-dsa",
        "deepseek32",
        # Hybrid KDA + DSA trunk; both GLM-5.3-Flash ports filter blk.45 out
        "glm5next",
        "glm5-next",
        # Plain attention trunk with an explicit nextn filter, llama-model.cpp:2356
        "step35",
        "hy_v3",
        "mimo2",
    }
)


def _accumulated_spec_types(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> set:
    """Types accumulate rather than replace, and even --spec-type none only appends, so read them all."""
    values = []
    args = [str(a) for a in extra_args] if extra_args else []
    for i, raw in enumerate(args):
        _, eq, inline = raw.partition("=")
        if _flag_name(raw) == "--spec-type":
            values.append(inline if eq else (args[i + 1] if i + 1 < len(args) else ""))
    env_value = (os.environ if env is None else env).get("LLAMA_ARG_SPEC_TYPE")
    if env_value:
        values.append(env_value)
    return {p.strip().lower() for v in values for p in v.split(",") if p.strip()}


def _extra_args_requests_mtp(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True if MTP lands in llama.cpp's spec-type vector, so the budget must reserve
    for it and the slots must clamp. Per _accumulated_spec_types, ANY source counts."""
    return bool(_accumulated_spec_types(extra_args, env) & {"mtp", "draft-mtp"})


def _extra_args_requests_dspark(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True if DSpark lands in llama.cpp's accumulated spec-type vector."""
    return "draft-dspark" in _accumulated_spec_types(extra_args, env)


def _dflash_drafter_breaks_on_images(
    model_path: Optional[str], dflash_path: Optional[str], projector: Optional[str]
) -> bool:
    """#13511: llama.cpp skips an M-RoPE image in the DFlash drafter (its rows share one
    position) and a NeoX drafter then rejects the jump, so the first image sent after any
    text fails the slot. Unreadable headers fail open."""
    if not (model_path and dflash_path and projector):
        return False
    try:
        from utils.models.gguf_metadata import gguf_has_mrope_sections, mmproj_accepts_image
        return bool(
            mmproj_accepts_image(projector)
            and gguf_has_mrope_sections(model_path) is True
            and gguf_has_mrope_sections(dflash_path) is False
        )
    except Exception:
        return False


def _extra_args_requests_dflash(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True if DFlash lands in llama.cpp's accumulated spec-type vector."""
    return "draft-dflash" in _accumulated_spec_types(extra_args, env)
