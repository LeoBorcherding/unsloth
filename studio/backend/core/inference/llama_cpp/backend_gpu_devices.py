# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: GPU allocation, device enumeration and visibility."""

import bisect
import ctypes
import math
import os
import re
import struct
import subprocess
import sys
import threading

from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _GPU_LAYER_FLAGS,
    fit_is_effectively_on,
    fit_is_enabled_in,
    parse_gpu_layers_override,
    resolve_effective_load_state,
)
from utils.child_stdio import utf8_child_env
from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

from .binary_runtime import (
    _DEVICE_VISIBILITY_ENV,
    _MISSING,
    _NEEDS_ROCM_CLASSIFICATION,
    _SELF_EVIDENTLY_DISCRETE,
    _apply_igpu_host_reserve_mib,
    _llama_lib_dir,
    _load_probe_memo_get,
    _load_probe_memo_put,
    _parse_listed_devices,
)
from .draft_args import (
    _CPU_DEVICE_VALUES,
    _count_subsequence,
    _device_selection_is_cpu,
    _extra_args_device,
    _extra_args_main_device,
    _split_mode_confines_to_one_device,
    _without_subsequence,
)
from .gguf_cache import _SHARD_FULL_RE
from .launch_flags import (
    _args_place_tensors_on_cpu,
    _env_places_tensors_on_cpu,
)
from .log import logger
from .memory_policy import _extra_args_set_any_flag
from .moe_cache import (
    _LAZY_READ_TENSORS,
    _NEXTN_SKIPPED_WITHOUT_MTP_ARCHS,
    _effective_lazy_mode,
    _lazy_read_bytes,
)
from core.inference.llama_server_args import _effective_tensor_parallel
from core.inference.lookup_tables import is_per_layer_embedding as _is_per_layer_embedding
from typing import Sequence
from .binary_runtime import (
    _HOST_RAM_HEADROOM_MIB,
    _LLAMA_SHARED_LIB_RE,
    _resolve_llama_binary,
)
from .draft_args import (
    _sidecar_adapter_paths,
)
from .launch_flags import (
    _env_fixes_gpu_layers,
)
from .launch_support import (
    _extra_args_have_tensor_split,
)
from .memory_policy import (
    _FIT_CONTROL_ENV_VARS,
    _FIT_CONTROL_FLAGS,
)
from .moe_cache import (
    _LAZY_MODE_SPELLINGS,
    _LAZY_RESIDENT_MARGIN_FRAC,
    _gpu_device_node_present,
    _host_forced_in_files,
    _is_host_input_or_head_tensor,
    _lazy_read_names,
    _override_places_tensor_on_device,
    _prompt_cache_host_bytes,
    _without_torch_gpu_probe,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _GpuDevicesMixin:
    # ── GPU allocation ────────────────────────────────────────────

    @staticmethod
    def _get_gguf_size_bytes(model_path: str) -> int:
        """Total GGUF size in bytes, including split shards."""
        main = Path(model_path)
        total = main.stat().st_size

        # Check for split shards (e.g. model-00001-of-00003.gguf)
        m = _SHARD_FULL_RE.match(main.name)
        if m:
            prefix, _, num_total = m.group(1), m.group(2), m.group(3)
            sibling_pat = re.compile(
                r"^" + re.escape(prefix) + r"-\d{5}-of-" + re.escape(num_total) + r"\.gguf$",
                re.IGNORECASE,
            )
            for sibling in main.parent.iterdir():
                if sibling != main and sibling_pat.match(sibling.name):
                    total += sibling.stat().st_size

        return total

    @staticmethod
    def _gguf_scan_tensor_bytes(
        path: str, also: Optional[Callable[[str], bool]] = None
    ) -> "tuple[Optional[str], dict[str, int], int]":
        """(architecture, lazy-read candidate bytes by name, routed expert bytes) from one
        GGUF header. Streamed: ``GGUFReader`` materialises the vocabulary under the load
        lock. A quant type gguf predates is sized from the next tensor's offset. Names
        ``also`` accepts are kept by name too, routed experts included."""
        from gguf.constants import GGML_QUANT_SIZES
        from core.inference.offload_layout import _BLOCK_RE, _MOE_EXPERT_RE

        wanted = {name.encode() for names in _LAZY_READ_TENSORS.values() for name in names}
        architecture: Optional[str] = None
        alignment = 32  # GGUF's default when general.alignment is absent
        named: dict[str, int] = {}
        experts = 0
        offsets: list[int] = []
        unsized: list[tuple[str, int]] = []
        with open(path, "rb") as f:
            if struct.unpack("<I", f.read(4))[0] != 0x46554747:  # b"GGUF"
                return None, {}, 0
            # Version 1 used 32-bit counts and lengths, which this layout misreads.
            if struct.unpack("<I", f.read(4))[0] < 2:
                return None, {}, 0
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            for _ in range(n_kv):
                key = f.read(struct.unpack("<Q", f.read(8))[0])
                value_type = struct.unpack("<I", f.read(4))[0]
                if key == b"general.architecture" and value_type == 8:
                    raw = f.read(struct.unpack("<Q", f.read(8))[0])
                    architecture = raw.decode("utf-8", "replace")
                    continue
                if key == b"general.alignment" and value_type == 4:
                    alignment = max(1, struct.unpack("<I", f.read(4))[0])
                    continue
                LlamaCppBackend._gguf_skip_value(f, value_type)
            for _ in range(n_tensors):
                raw_name = f.read(struct.unpack("<Q", f.read(8))[0])
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                ggml_type = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                offsets.append(offset)
                name = raw_name.decode("utf-8", "replace")
                block_match = _BLOCK_RE.match(name)
                is_expert = bool(block_match and _MOE_EXPERT_RE.match(block_match.group(2)))
                kept = also is not None and also(name)
                if not is_expert and raw_name not in wanted and not kept:
                    continue
                block = GGML_QUANT_SIZES.get(ggml_type)
                if block is None:
                    unsized.append((name, offset))
                    continue
                size = math.prod(dims) // block[0] * block[1]
                if is_expert:
                    experts += size
                if not is_expert or kept:
                    named[name] = size
            if unsized:
                data_start = -(-f.tell() // alignment) * alignment
                data_end = max(0, os.path.getsize(path) - data_start)
                ordered = sorted(offsets)
                for name, offset in unsized:
                    later = ordered[bisect.bisect_right(ordered, offset) :]
                    size = max(0, (later[0] if later else data_end) - offset)
                    block_match = _BLOCK_RE.match(name)
                    is_expert = bool(block_match and _MOE_EXPERT_RE.match(block_match.group(2)))
                    if is_expert:
                        experts += size
                    if not is_expert or (also is not None and also(name)):
                        named[name] = size
        return architecture, named, experts

    def _gguf_tensor_scan(
        self, model_path: Optional[str]
    ) -> "Optional[tuple[Optional[str], dict[str, int], int]]":
        """``_gguf_scan_tensor_bytes`` over every shard, cached per file identity; None
        when a shard is missing or unreadable."""
        from core.inference.offload_layout import split_shard_paths

        if not model_path:
            return None
        paths = split_shard_paths(model_path) or [model_path]
        try:
            key = tuple((p, st.st_size, st.st_mtime_ns) for p in paths for st in (os.stat(p),))
        except OSError:
            return None
        cached = getattr(self, "_gguf_tensor_scan_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            architecture: Optional[str] = None
            named: dict[str, int] = {}
            experts = 0
            for p in paths:
                arch, part, part_experts = self._gguf_scan_tensor_bytes(
                    p, also = _is_host_input_or_head_tensor
                )
                # Shard 1 carries the metadata.
                architecture = architecture or arch
                for name, size in part.items():
                    named[name] = named.get(name, 0) + size
                experts += part_experts
            result = (architecture, named, experts)
        except Exception as e:  # unreadable or truncated: discount nothing
            logger.debug("GGUF tensor scan failed for %s (%s)", model_path, e)
            result = None
        self._gguf_tensor_scan_cache = (key, result)
        return result

    def _unloaded_nextn_bytes(self, model_path: Optional[str]) -> int:
        """Bytes of the embedded NextN/MTP blocks llama.cpp leaves unloaded without
        ``--spec-type draft-mtp`` (nextn_flags marks them TENSOR_SKIP unless load_mtp).
        Only for architectures measured to skip them; 0 when unknown or unreadable."""
        from core.inference.offload_layout import _BLOCK_RE, split_shard_paths

        nextn = int(getattr(self, "_nextn_predict_layers", None) or 0)
        n_blocks = int(getattr(self, "_n_layers", None) or 0)
        if (
            not model_path
            or nextn <= 0
            or n_blocks <= nextn
            or getattr(self, "_architecture", None) not in _NEXTN_SKIPPED_WITHOUT_MTP_ARCHS
        ):
            return 0
        first = n_blocks - nextn

        def in_head(name: str) -> bool:
            match = _BLOCK_RE.match(name)
            return bool(match) and int(match.group(1)) >= first

        try:
            return sum(
                size
                for p in (split_shard_paths(model_path) or [model_path])
                for name, size in self._gguf_scan_tensor_bytes(p, also = in_head)[1].items()
                if in_head(name)
            )
        except Exception as e:  # missing shard or truncated header: discount nothing
            logger.debug("NextN block scan failed for %s (%s)", model_path, e)
            return 0

    def _lazy_read_host_bytes(
        self,
        model_path: Optional[str],
        *,
        caps: Optional[Mapping[str, object]],
        extra_args: Optional[Iterable[str]] = None,
        env: Optional[Mapping[str, str]] = None,
        auto_resolves_off: bool = False,
    ) -> int:
        """Bytes llama.cpp leaves mmap'd from disk under every load mode (``_LAZY_READ_TENSORS``);
        zero without lazy-read support, under ``off``, or on an unreadable header.
        ``auto_resolves_off``: llama.cpp turns ``auto`` into ``off`` for these devices."""
        if not model_path or not (caps or {}).get("supports_lazy_mode"):
            return 0
        mode = _effective_lazy_mode(
            extra_args, env, flag = str((caps or {}).get("lazy_mode_flag") or "--lazy-mode")
        )
        if mode == "off" or (mode == "auto" and auto_resolves_off):
            return 0
        scan = self._gguf_tensor_scan(model_path)
        if scan is None:
            return 0
        architecture, named, _experts = scan
        return _lazy_read_bytes(architecture, named, mode)

    def _lazy_auto_resolves_off(
        self,
        *,
        gpu_indices: Optional[Iterable[int]],
        detected_gpus: Optional[Iterable[tuple]],
        shared_gpu_ids: Optional[Iterable[int]],
        is_vulkan_backend: bool,
        probe_integrated_cuda: bool = True,
        extra_args: Optional[Iterable[str]] = None,
        env: Optional[Mapping[str, str]] = None,
        device_list: Optional[bool] = None,
        tensor_parallel: Optional[bool] = None,
        integrated_gpu_ids: Optional[Iterable[int]] = None,
        child_gpu_order: Optional[Sequence[int]] = None,
    ) -> bool:
        """Whether llama.cpp resolves ``--lazy-mode auto`` to ``off`` for this child.

        ``child_gpu_order``: the ids the child's visibility mask will hold, in order
        (``_planned_child_gpu_order``), for a preview whose ``env`` does not carry that
        mask yet; a compact ``CUDA<k>`` / ``ROCm<k>`` names entry k of it."""
        tensor = (
            _effective_tensor_parallel(extra_args, False, env)
            if tensor_parallel is None
            else bool(tensor_parallel)
        )
        selected = None
        override = self._effective_device_ids(extra_args, env)
        one_device = not tensor and _split_mode_confines_to_one_device(extra_args, env)
        if one_device:
            main = self._effective_main_gpu(extra_args, env)
            main = 0 if main is None else main
            if main < 0:
                return False
        if override is not None and one_device:
            override = [d for d in override if d.lower() not in _CPU_DEVICE_VALUES]
            override = override[main : main + 1]
        if override is not None:
            # An RPC device reports mmap support (ggml-rpc.cpp device_get_props), so it never turns
            # auto off.
            gpu_names = [
                d
                for d in override
                if d.lower() not in _CPU_DEVICE_VALUES and not d.strip().lower().startswith("rpc")
            ]
            ordinals = [self._device_ordinal(d) for d in gpu_names]
            if all(o is not None for o in ordinals):
                if child_gpu_order is not None and not is_vulkan_backend:
                    _order = [int(i) for i in child_gpu_order]
                    selected = [_order[o] for o in ordinals if 0 <= o < len(_order)] or _order
                else:
                    selected = list(self._physical_ordinals(ordinals, env, is_vulkan_backend) or ())
        elif tensor or (device_list if device_list is not None else is_vulkan_backend):
            try:
                selected = list(gpu_indices or ()) or None
            except TypeError:
                selected = None
            if selected is None and tensor:
                selected = [row[0] for row in (detected_gpus or ())] or None
        if selected is not None:
            if selected and override is None and one_device:
                main = self._effective_main_gpu(extra_args, env)
                main = 0 if main is None else main
                selected = selected[main : main + 1] if main >= 0 else []
            selected = set(selected)
            if not selected:
                return False
        if selected:
            if is_vulkan_backend:
                return bool(selected & (set(shared_gpu_ids or ()) | set(integrated_gpu_ids or ())))
            try:
                if selected & LlamaCppBackend._rocm_unified_memory_gpu_ids():
                    return True
            except Exception:
                pass
            if not probe_integrated_cuda and not self._integrated_cuda_probe_is_free():
                return False
            try:
                return bool(selected & LlamaCppBackend._integrated_cuda_gpu_ids())
            except Exception:
                return False
        rows = [(row[0], row[1]) for row in (detected_gpus or ())]
        if self._offload_target_shares_system_memory(
            is_vulkan_backend = is_vulkan_backend,
            shared_gpu_ids = shared_gpu_ids,
            detected_gpus = rows,
            gpu_indices = gpu_indices,
        ):
            return True
        if is_vulkan_backend:
            return False
        if not probe_integrated_cuda and not self._integrated_cuda_probe_is_free():
            return False
        return self._integrated_cuda_selection_is_all_shared(
            list(gpu_indices) if gpu_indices else None
        )

    def _apple_lookup_table_discount(
        self,
        probe_discount: int,
        model_path: Optional[str],
        *,
        caps: Optional[Mapping[str, object]],
        extra_args: Optional[Iterable[str]],
        model_size_fit: int,
        need_at_ctx: Optional[int],
        budget_mib: int,
        cache_ram: Optional[int],
        env: Optional[Mapping[str, str]] = None,
        binary: Optional[str] = None,
        cpu_forced: bool = False,
    ) -> int:
        """The Metal embedding discount with the lazily read lookup table settled."""
        source_env = os.environ if env is None else env
        self._apple_lazy_flags = []
        table = self._lazy_read_host_bytes(
            model_path, caps = caps, extra_args = extra_args, env = source_env
        )
        if table <= 0:
            auto_table = self._lazy_read_host_bytes(model_path, caps = caps, extra_args = (), env = {})
            if (
                auto_table > 0
                and _effective_lazy_mode(
                    extra_args,
                    source_env,
                    flag = str((caps or {}).get("lazy_mode_flag") or "--lazy-mode"),
                )
                == "off"
            ):
                return max(0, int(probe_discount) - auto_table)
            return probe_discount
        if _sidecar_adapter_paths(extra_args):
            return probe_discount
        scan = self._gguf_tensor_scan(model_path)
        streamed = (
            _lazy_read_names(
                scan[0],
                scan[1],
                _effective_lazy_mode(
                    extra_args,
                    source_env,
                    flag = str((caps or {}).get("lazy_mode_flag") or "--lazy-mode"),
                ),
            )
            if scan is not None
            else frozenset()
        )
        if (
            scan is not None
            and scan[0] not in self._host_forced_table_archs(binary)
            and any(
                _override_places_tensor_on_device(name, extra_args, source_env)
                for name in scan[1]
                if _is_per_layer_embedding(name) and name not in streamed
            )
        ):
            return probe_discount

        def _resident_fits(table_bytes: int) -> Optional[bool]:
            if self._child_runs_on_cpu(binary, extra_args, source_env, cpu_forced = cpu_forced):
                return None
            if any(
                str(source_env.get(name) or "").strip()
                for name in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL")
            ):
                return None
            avail = self._available_system_memory_mib()
            prompt = _prompt_cache_host_bytes(
                cache_ram,
                extra_args,
                source_env,
                supports_cache_ram = bool((caps or {}).get("supports_cache_ram")),
            )
            if (
                avail is None
                or prompt is None
                or need_at_ctx is None
                or budget_mib <= 0
                or not self._can_estimate_kv()
            ):
                return None
            need = (
                int(model_size_fit)
                + int(need_at_ctx)
                + int(prompt)
                + int(table_bytes * _LAZY_RESIDENT_MARGIN_FRAC)
            )
            ceiling_mib = min(int(budget_mib), int(avail) - _HOST_RAM_HEADROOM_MIB)
            return need <= max(0, ceiling_mib) * 1024 * 1024

        self._apple_lazy_flags = self._managed_lazy_mode_flags(
            model_path,
            caps = caps,
            extra_args = extra_args,
            env = source_env,
            resident_fits = _resident_fits,
            binary = binary,
        )
        if self._apple_lazy_flags:
            return max(0, int(probe_discount) - table)
        if table > probe_discount:
            logger.info(
                "Metal fit: %.2f GB of lookup tables are read lazily from disk, outside "
                "the Metal working set; not charged to unified memory.",
                table / (1024**3),
            )
        return max(int(probe_discount), table)

    def _managed_lazy_mode_flags(
        self,
        model_path: Optional[str],
        *,
        caps: Optional[Mapping[str, object]],
        extra_args: Optional[Iterable[str]] = None,
        env: Optional[Mapping[str, str]] = None,
        auto_resolves_off: bool = False,
        resident_fits: Callable[[int], Optional[bool]],
        binary: Optional[str] = None,
    ) -> list[str]:
        """``[<lazy flag>, "off"]`` to keep a lazily read lookup table resident, else ``[]``."""
        caps = caps or {}
        if not model_path or not caps.get("supports_lazy_mode"):
            return []
        flag = str(caps.get("lazy_mode_flag") or "--lazy-mode")
        source_env = os.environ if env is None else env
        flags, env_name = _LAZY_MODE_SPELLINGS.get(flag, (frozenset({flag}), ""))
        if (env_name and str(source_env.get(env_name) or "").strip()) or _extra_args_set_any_flag(
            extra_args, flags
        ):
            return []
        if auto_resolves_off:
            return []
        table = self._lazy_read_host_bytes(
            model_path, caps = caps, extra_args = extra_args, env = source_env
        )
        if table <= 0:
            return []
        scan = self._gguf_tensor_scan(model_path)
        if scan is None or (
            scan[0] not in self._host_forced_table_archs(binary)
            and any(
                _override_places_tensor_on_device(name, extra_args, source_env)
                for name in scan[1]
                if _is_per_layer_embedding(name)
            )
        ):
            return []
        try:
            fits = resident_fits(table)
        except Exception as e:  # noqa: BLE001 -- unpriced: keep llama.cpp's default
            logger.debug("Lookup tables: resident fit unpriced (%s)", e)
            fits = None
        key = (model_path, table, bool(fits))
        if getattr(self, "_lazy_mode_decision_logged", None) != key:
            self._lazy_mode_decision_logged = key
            if fits:
                logger.info(
                    "Lookup tables: keeping the %.1f GB lookup table in RAM (%s off); "
                    "free RAM covers it beside the weights and the KV cache.",
                    table / (1024**3),
                    flag,
                )
            else:
                logger.info(
                    "Lookup tables: streaming the %.1f GB lookup table from disk "
                    "(llama.cpp's lazy read); RAM cannot hold it beside the weights and "
                    "the KV cache. Prefill and decode can be slower and vary.",
                    table / (1024**3),
                )
        return [flag, "off"] if fits else []

    def lookup_tables_stream_by_default(
        self,
        extra_args: Optional[Iterable[str]] = None,
        env: Optional[Mapping[str, str]] = None,
        *,
        gpu_ids: Optional[Iterable[int]] = None,
        tensor_parallel: Optional[bool] = None,
    ) -> bool:
        """Whether a launch on this host leaves lazily read lookup tables on disk."""
        _env = dict(os.environ if env is None else env)
        try:
            binary = self._find_llama_server_binary()
            caps = self.probe_server_capabilities(binary)
            if not caps.get("supports_lazy_mode"):
                return False
            mode = _effective_lazy_mode(
                extra_args, _env, flag = str(caps.get("lazy_mode_flag") or "--lazy-mode")
            )
            if mode != "auto":
                return mode == "on"
            tensor = (
                _effective_tensor_parallel(extra_args, False, _env)
                if tensor_parallel is None
                else bool(tensor_parallel)
            )
            self._reconcile_split_mode_env(_env, tensor_parallel = tensor)
            cpu_child = self._child_runs_on_cpu(binary, extra_args, _env, tensor_parallel = tensor)
            vulkan = not cpu_child and self._is_vulkan_backend(binary)
            rows = []
            if not cpu_child:
                with _without_torch_gpu_probe():
                    rows = list(self._get_gpu_memory(binary, for_llama_server = True) or ())
                # An empty Vulkan inventory is a failed probe: it may be an iGPU, where auto is off.
                if not rows and (vulkan or _gpu_device_node_present()):
                    return False
            shared = {idx for idx, _free, total in rows if total <= 0} if vulkan else set()
            picked = [int(i) for i in gpu_ids] if gpu_ids else None
            return not self._lazy_auto_resolves_off(
                gpu_indices = picked,
                detected_gpus = [(idx, free) for idx, free, _total in rows],
                shared_gpu_ids = shared,
                is_vulkan_backend = vulkan,
                probe_integrated_cuda = False,
                extra_args = extra_args,
                env = _env,
                device_list = None if picked else False,
                tensor_parallel = tensor_parallel,
                integrated_gpu_ids = shared,
            )
        except Exception:
            return False

    @staticmethod
    def _host_forced_table_archs(binary: Optional[str] = None) -> frozenset[str]:
        """Archs whose table the build keeps on the host whatever -ot says; none if unreadable."""
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return frozenset()
        try:
            resolved = _resolve_llama_binary(binary)
            libs = sorted(
                p
                for p in resolved.parent.iterdir()
                if _LLAMA_SHARED_LIB_RE.match(p.name) and p.is_file()
            )
            key = tuple(
                (str(p), st.st_size, st.st_mtime_ns)
                for p in (libs or [resolved])
                for st in (p.stat(),)
            )
        except OSError:
            return frozenset()
        return _host_forced_in_files(key)

    @staticmethod
    def _metal_placement_unset(
        extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]], *, layers_fixed: bool
    ) -> bool:
        """Whether nothing but Studio places this Metal launch, so a measured discount may pin full
        offload."""
        source_env = os.environ if env is None else env
        return not (
            layers_fixed
            or _extra_args_set_any_flag(extra_args, _GPU_LAYER_FLAGS)
            or _env_fixes_gpu_layers(source_env)
            or _device_selection_is_cpu(extra_args, source_env)
            or _args_place_tensors_on_cpu(extra_args)
            or _env_places_tensors_on_cpu(source_env)
            or _extra_args_set_any_flag(extra_args, _FIT_CONTROL_FLAGS)
            or any(source_env.get(name) for name in _FIT_CONTROL_ENV_VARS)
            or fit_is_enabled_in(extra_args)
            # A pass-through adapter sits on the base tensor's buffer (llama-adapter.cpp:335) and
            # nothing here charges it: keep the discount unspent.
            or _sidecar_adapter_paths(extra_args)
        )

    def _child_runs_on_cpu(
        self,
        binary: Optional[str],
        extra_args: Optional[Iterable[str]],
        env: Optional[Mapping[str, str]],
        *,
        cpu_forced: bool = False,
        tensor_parallel: Optional[bool] = None,
    ) -> bool:
        """Whether the child gets no model device (CPU build, CPU --device or forced CPU launch)."""
        if (
            cpu_forced
            or self._binary_ships_no_gpu_backend(binary, env)
            or _device_selection_is_cpu(extra_args, env)
        ):
            return True
        tensor = (
            _effective_tensor_parallel(extra_args, False, env)
            if tensor_parallel is None
            else bool(tensor_parallel)
        )
        # Split mode none with a negative main GPU clears the model's devices (src/llama.cpp).
        if not tensor and _split_mode_confines_to_one_device(extra_args, env):
            main = self._effective_main_gpu(extra_args, env)
            return main is not None and main < 0
        return False

    @staticmethod
    def _installed_ggml_backends(binary: Optional[str] = None) -> frozenset[str]:
        """Backend libraries shipped beside llama-server."""
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return frozenset()
        try:
            files = tuple(path.name for path in _llama_lib_dir(binary).iterdir() if path.is_file())
        except OSError:
            return frozenset()
        backends = set()
        for backend in ("base", "cpu", "cuda", "hip", "vulkan"):
            stem = f"ggml-{backend}.dll" if sys.platform == "win32" else f"libggml-{backend}.so"
            if any(name == stem or name.startswith(stem + ".") for name in files):
                backends.add(backend)
        return frozenset(backends)

    def _record_memory_state(
        self,
        argv,
        env = None,
    ) -> None:
        """Record the ``(mlock, reserves_ram)`` a launch really runs with, and
        whether it streams, from ONE parse. A dozen paths record a state; a
        second parse at each of them would drift from this one."""
        mlock, reserves_ram, direct_io = resolve_effective_load_state(argv, env)
        self._memory_state = (mlock, reserves_ram)
        self._memory_direct_io = direct_io

    def _drop_managed_dio(
        self,
        argv,
        reason: str,
        clear_record: bool = True,
    ):
        """Strip Unsloth's own DirectIO pair unless full offload is confirmed (DirectIO reads CPU layers into buffers); never the user's --load-mode."""
        if not self._memory_dio_flags:
            self._memory_dio_applicable = False
            return argv
        # Strip only while this argv holds more pairs than the user wrote.
        if _count_subsequence(list(argv), self._memory_dio_flags) <= _count_subsequence(
            list(getattr(self, "_memory_dio_user_tokens", []) or []), self._memory_dio_flags
        ):
            if clear_record:
                self._memory_dio_flags = []
            self._memory_dio_applicable = False
            return argv
        stripped = _without_subsequence(argv, self._memory_dio_flags)
        if clear_record:
            self._memory_dio_flags = []
        self._memory_dio_applicable = False
        # A child equal to an unmanaged one must not be torn down when toggles go off.
        self._memory_policy_active = self._memory_policy_extras_touched
        logger.info("Model Memory: dropping the managed --load-mode dio; %s", reason)
        return stripped

    @staticmethod
    def _vulkan_offload_is_discrete(binary: Optional[str], gpu_indices = None) -> bool:
        """An unanswered probe declines, since an iGPU's VRAM is system RAM and DirectIO would be wrong."""
        try:
            rows = LlamaCppBackend._run_vulkan_probe(binary)
        except Exception:
            return False
        if not rows:
            return False
        wanted = set(gpu_indices) if gpu_indices else None
        selected = [r for r in rows if wanted is None or r["index"] in wanted]
        if not selected:
            return False
        # EVERY ordinal, not merely some: a partial intersection confirmed an unclassified one.
        if wanted is not None and wanted - {r["index"] for r in selected}:
            return False
        # The probe reports all-False when the type query fails, so require type_known.
        if not all(r.get("type_known") for r in selected):
            return False
        return not any(r["is_igpu"] for r in selected)

    @classmethod
    def _gpu_offload_confirmed(
        cls,
        binary: Optional[str],
        env: Optional[Mapping[str, str]],
        gpu_indices,
        host_resident: bool,
        dio_possible: bool,
        extra_args = None,
    ) -> bool:
        """Whether weights sit on a discrete GPU; unanswered decline (DirectIO over host RAM is worse than nothing); gate on dio_possible, never the toggle."""
        if not dio_possible or host_resident:
            return False
        devices = cls._enumerated_gpu_devices(binary, env)
        # A pass-through --device is appended last and decides placement; values are ggml ids.
        override = cls._effective_device_ids(extra_args, env)
        if override is not None:
            if any(d.lower() in _CPU_DEVICE_VALUES for d in override):
                return False
            listed = {d.lower() for d in (devices or [])}
            if not override or not {d.lower() for d in override} <= listed:
                return False
            gpu_indices = [
                cls._device_ordinal(d) for d in override if cls._device_ordinal(d) is not None
            ]
            devices = [d for d in (devices or []) if d.lower() in {o.lower() for o in override}]
        if not cls._offload_devices_are_live(devices, gpu_indices):
            return False
        # Check EVERY selected device; the first kind's verdict alone would pass a mixed set with an iGPU.
        selected = cls._selected_devices(devices, gpu_indices)
        backends = {cls._device_backend(d) for d in selected}
        if backends - {"vulkan"} - _SELF_EVIDENTLY_DISCRETE - _NEEDS_ROCM_CLASSIFICATION:
            return False
        # An AMD id only counts once something has actually looked at the device.
        if (backends & _NEEDS_ROCM_CLASSIFICATION) and not cls._rocm_classification_answered():
            return False
        if "vulkan" in backends:
            # Only the VULKAN ordinals: a CUDA target's 0 otherwise satisfies Vulkan row 0.
            vulkan_ordinals = [
                cls._device_ordinal(d) for d in selected if cls._device_backend(d) == "vulkan"
            ]
            if any(o is None for o in vulkan_ordinals):
                return False
            return cls._vulkan_offload_is_discrete(binary, vulkan_ordinals)
        return True

    @staticmethod
    def _device_ordinal(device: str) -> Optional[int]:
        """The trailing ordinal of a ggml device id, or None."""
        match = re.search(r"(\d+)$", device)
        return int(match.group(1)) if match else None

    @staticmethod
    def _compact_ordinals(
        gpu_indices,
        env: Optional[Mapping[str, str]],
        is_vulkan = False,
    ):
        """A mask reindexes survivors from 0, so a physical id's position in the mask is its compact
        ordinal."""
        if not gpu_indices or not env or is_vulkan:
            return gpu_indices
        # GGML_VK_VISIBLE_DEVICES omitted: ggml already compacts Vulkan ordinals.
        for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            raw = env.get(name)
            if not raw or not str(raw).strip():
                continue
            try:
                order = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
            except ValueError:
                return gpu_indices
            if not order:
                return gpu_indices
            try:
                mapped = [order.index(int(i)) for i in gpu_indices if int(i) in order]
            except (TypeError, ValueError):
                return gpu_indices
            return mapped or gpu_indices
        return gpu_indices

    @staticmethod
    def _physical_ordinals(
        compact,
        env: Optional[Mapping[str, str]],
        is_vulkan = False,
    ):
        """--device takes compact ids but _amd_apu_wants_unified_memory needs physical ones, so map back."""
        if not compact or not env or is_vulkan:
            return compact
        for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            raw = env.get(name)
            if not raw or not str(raw).strip():
                continue
            try:
                order = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
            except ValueError:
                return compact
            if not order:
                return compact
            try:
                mapped = [order[int(i)] for i in compact if 0 <= int(i) < len(order)]
            except (TypeError, ValueError):
                return compact
            return mapped or compact
        return compact

    @classmethod
    def _effective_main_gpu(cls, extra_args, env) -> Optional[int]:
        """--main-gpu may name a device outside the plan, so it replaces the selection, not narrowing it."""
        if not _split_mode_confines_to_one_device(extra_args, env):
            return None
        value = _extra_args_device(extra_args, {"--main-gpu", "-mg"})
        if value is None and env:
            value = env.get("LLAMA_ARG_MAIN_GPU")
        if value is None:
            return None
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    @classmethod
    def _effective_gpu_selection(
        cls,
        binary,
        env,
        gpu_indices,
        extra_args,
        dio_possible = True,
        is_vulkan = False,
    ):
        """Returns (physical, compact), resolved once for all consumers so verdict and confirmation match; the APU check needs physical ids, --list-devices compact."""
        if not dio_possible:
            return gpu_indices, gpu_indices
        override = cls._effective_device_ids(extra_args, env)
        if override and not any(d.lower() in _CPU_DEVICE_VALUES for d in override):
            listed = {d.lower() for d in (cls._enumerated_gpu_devices(binary, env) or [])}
            if {d.lower() for d in override} <= listed:
                # `--device` names compact ids, so physical is the derived form.
                ordinals = [cls._device_ordinal(d) for d in override]
                compact = [o for o in ordinals if o is not None]
                if not compact:
                    return gpu_indices, cls._compact_ordinals(gpu_indices, env, is_vulkan)
                return cls._physical_ordinals(compact, env, is_vulkan), compact
            return gpu_indices, cls._compact_ordinals(gpu_indices, env, is_vulkan)
        # Otherwise PHYSICAL, and `--main-gpu` under `-sm none` replaces the selection.
        main_gpu = cls._effective_main_gpu(extra_args, env)
        if main_gpu is not None:
            # `--main-gpu` indexes the FILTERED list, so it too arrives compact.
            return cls._physical_ordinals([main_gpu], env, is_vulkan), [main_gpu]
        return gpu_indices, cls._compact_ordinals(gpu_indices, env, is_vulkan)

    @classmethod
    def _effective_device_ids(cls, extra_args, env) -> Optional[list[str]]:
        """Last argv value wins, then LLAMA_ARG_DEVICE, as llama.cpp does; values are ggml ids."""
        value = _extra_args_main_device(extra_args)
        if value is None and env:
            value = env.get("LLAMA_ARG_DEVICE")
        if value is None:
            return None
        return [d.strip() for d in str(value).split(",") if d.strip()]

    @staticmethod
    def _selected_devices(devices: Optional[list[str]], gpu_indices) -> list[str]:
        """The enumerated GPU devices this launch will use. ggml ids are
        ``<Backend><ordinal>``; the CPU device is not a placement target, and with no
        pin every GPU device is in play."""
        if not devices:
            return []
        gpu = [d for d in devices if not d.upper().startswith("CPU")]
        if not gpu_indices:
            return gpu
        wanted = {int(i) for i in gpu_indices}
        picked = []
        for device in gpu:
            match = re.search(r"(\d+)$", device)
            if match and int(match.group(1)) in wanted:
                picked.append(device)
        return picked

    @classmethod
    def _offload_devices_are_live(cls, devices: Optional[list[str]], gpu_indices) -> bool:
        """Ordinal coverage, not device count: a multi-backend build can list one ordinal twice."""
        selected = cls._selected_devices(devices, gpu_indices)
        if not selected:
            return False
        if not gpu_indices:
            return True
        covered = {cls._device_ordinal(d) for d in selected}
        return {int(i) for i in gpu_indices} <= covered

    @staticmethod
    def _device_backend(device: str) -> str:
        """The ggml backend behind a device id, lowercased. What `_is_vulkan_backend`
        and the plugin-root scan approximated from filenames: the build reports what it
        actually loaded, so an external ``GGML_BACKEND_PATH`` plugin names itself too."""
        return re.sub(r"\d+$", "", device).strip().lower()

    @classmethod
    def _devices_are_vulkan(cls, devices: Optional[list[str]], gpu_indices) -> bool:
        """Whether any device in play is a Vulkan one, read off the build's own ids."""
        return any(
            cls._device_backend(d) == "vulkan" for d in cls._selected_devices(devices, gpu_indices)
        )

    @classmethod
    def _enumerated_gpu_devices(
        cls,
        binary: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> Optional[list[str]]:
        """None means an old build, timeout or crash; [] is a real answer of no devices, and both
        decline."""
        binary = binary or cls._find_llama_server_binary()
        if not binary:
            return None
        key = (binary, cls._device_visibility_key(env))
        memo = _load_probe_memo_get("devices", key)
        if memo is not None:
            return None if memo is _MISSING else memo
        devices = cls._run_list_devices(binary, env)
        _load_probe_memo_put("devices", key, _MISSING if devices is None else devices)
        return devices

    @staticmethod
    def _device_visibility_key(env: Optional[Mapping[str, str]]):
        """What this environment lets the child see, as a memo key. A rung that masks
        an adapter has a different answer coming, and without this a narrowing onto a
        usable discrete GPU kept the original set's failure."""
        source = os.environ if env is None else env
        return tuple((name, source.get(name)) for name in _DEVICE_VISIBILITY_ENV)

    @staticmethod
    def _run_list_devices(
        binary: str, env: Optional[Mapping[str, str]] = None
    ) -> Optional[list[str]]:
        """``--list-devices`` once, parsed. The flag prints and exits, so this leaves no
        server behind; it is on the argv denylist for that reason, which does not apply
        to Unsloth calling it deliberately."""
        try:
            probe_env = LlamaCppBackend._llama_server_env_for_binary(binary)
        except Exception:
            probe_env = child_env_without_native_path_secret()
        # Probe must see the child env: replay only the caller's removals, not its whole mapping, since on Windows the venv CUDA runtime is added to PATH.
        if env is not None:
            # BOTH directions: removals alone dropped a narrowing rung's new mask.
            for name in _DEVICE_VISIBILITY_ENV:
                if name in env:
                    probe_env[name] = env[name]
                else:
                    probe_env.pop(name, None)
            for name in list(probe_env):
                if name not in env and name in os.environ:
                    probe_env.pop(name, None)
        try:
            result = subprocess.run(
                [binary, "--list-devices"],
                capture_output = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 30,
                env = utf8_child_env(probe_env),
                **_windows_hidden_subprocess_kwargs(),
            )
        except Exception as e:
            logger.debug(f"llama-server --list-devices failed: {e}")
            return None
        if result.returncode != 0:
            logger.debug(f"llama-server --list-devices exited {result.returncode}")
            return None
        return _parse_listed_devices(result.stdout)

    @staticmethod
    def _is_vulkan_backend(binary: Optional[str] = None) -> bool:
        """Vulkan build is identified by its ggml lib; a CUDA or HIP lib beside it defers to that
        backend."""
        backends = LlamaCppBackend._installed_ggml_backends(binary)
        return "vulkan" in backends and not backends.intersection({"cuda", "hip"})

    _SYSMEM_FALLBACK_RISK: dict[Optional[str], bool] = {}

    @staticmethod
    def _sysmem_fallback_risk(binary: Optional[str] = None) -> bool:
        """True when over-budget allocations spill to host RAM; Windows CUDA only, no probe (integrated CUDA is Linux-only; a probe would pin a CUDA context)."""
        if sys.platform != "win32":
            return False
        if binary is None:
            try:
                binary = LlamaCppBackend._find_llama_server_binary()
            except Exception:  # noqa: BLE001 - classify with whatever the lib lookup resolves
                binary = None
        # Keyed by resolved path and build revision: runtimes switch, updates swap in place.
        key = LlamaCppBackend._binary_revision(binary) or binary
        cached = LlamaCppBackend._SYSMEM_FALLBACK_RISK.get(key)
        if cached is not None:
            return cached
        try:
            backends = LlamaCppBackend._installed_ggml_backends(binary)
        except Exception as e:  # noqa: BLE001 - an unreadable lib dir is not a load failure
            logger.debug("sysmem-fallback classification failed, keeping the base budget: %s", e)
            return False  # not cached: a transient read error must not pin False forever
        if not backends:
            return False  # an unreadable dir scans empty; same rule, not cached
        risk = "cuda" in backends
        LlamaCppBackend._SYSMEM_FALLBACK_RISK[key] = risk
        return risk

    @staticmethod
    def _active_gpu_visibility_mask() -> Optional[str]:
        """The raw visibility mask in force (HIP, then ROCR, then CUDA on ROCm; CUDA
        otherwise), or None when unset. Split out so a caller can tell "no mask" from
        "a mask I cannot map to physical ids"."""
        try:
            import torch

            # Same ROCm detection as _emit_child_gpu_visibility: if they differ an inherited ROCR mask reads as none and the child re-exposes hidden GPUs.
            is_rocm = LlamaCppBackend._torch_is_rocm(torch)
        except Exception:
            is_rocm = False
        if is_rocm:
            hip_v = os.environ.get("HIP_VISIBLE_DEVICES")
            # ROCR_VISIBLE_DEVICES is Linux-only: on Windows a stray ROCR var masks nothing and must not be read as the ordinal to physical mapping.
            rocr_v = None if sys.platform == "win32" else os.environ.get("ROCR_VISIBLE_DEVICES")
            cvd = (
                hip_v
                if hip_v is not None
                else rocr_v
                if rocr_v is not None
                else os.environ.get("CUDA_VISIBLE_DEVICES")
            )
        else:
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        return cvd

    @staticmethod
    def _resolve_visible_physical_ids() -> Optional[list[int]]:
        """Physical GPU ids behind the active visibility mask (HIP/ROCR/CUDA on
        ROCm, CUDA otherwise). None when unset, [] for an empty mask. Shared by the
        APU / datacenter / free-memory probes so they agree on the mapping."""
        cvd = LlamaCppBackend._active_gpu_visibility_mask()
        if cvd is None:
            return None
        try:
            return [int(x.strip()) for x in cvd.split(",") if x.strip()]
        except ValueError:
            pass
        # A resolvable CUDA UUID mask, else the GPU auto-pick escapes the mask onto hidden cards (#8873).
        try:
            from utils.hardware.hardware import _get_parent_visible_gpu_spec
            spec = _get_parent_visible_gpu_spec()
        except Exception:
            return None
        if spec.get("index_kind") == "cuda_device":
            # R570+ MIG ids are CUDA devices; the MIG paths pin through the snapshot.
            return None
        return spec["numeric_ids"] if spec.get("raw") == cvd.strip() else None

    @staticmethod
    def _visibility_mask_is_unmappable() -> bool:
        """True if a mask is set but unresolvable (UUID or MIG): pinning callers fail open, since numbers would expose hidden cards. False if unset or empty."""
        return (
            LlamaCppBackend._active_gpu_visibility_mask() is not None
            and LlamaCppBackend._resolve_visible_physical_ids() is None
        )

    @staticmethod
    def _emit_child_gpu_visibility(
        env: dict,
        pinned: str,
        *,
        prefer_rocr: bool = False,
    ) -> None:
        """Default masks HIP, clearing inherited ROCR (they stack); prefer_rocr masks ROCR to avoid HSA segfaults; CPU -1 keeps masks so none re-expose hidden GPUs."""
        env["CUDA_VISIBLE_DEVICES"] = pinned
        if pinned == "-1":
            # An inherited LLAMA_ARG_DEVICE or LLAMA_ARG_MAIN_GPU naming a device that no longer enumerates makes llama.cpp exit rather than use CPU.
            LlamaCppBackend._clear_device_placement_env(env)
        try:
            import torch as _torch

            # AMD SDK wheels may leave torch.version.hip unset; also check 'rocm' in __version__.
            if (
                getattr(_torch.version, "hip", None) is not None
                or "rocm" in getattr(_torch, "__version__", "").lower()
            ):
                if prefer_rocr and pinned != "-1" and sys.platform != "win32":
                    env["ROCR_VISIBLE_DEVICES"] = pinned
                    env.pop("HIP_VISIBLE_DEVICES", None)
                    # ROCR re-indexes from 0 and HIP reads CUDA_VISIBLE_DEVICES here, so pass post-ROCR ordinals 0..N-1, not physical ids: else HIP sees 0 devices.
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(
                        str(i) for i in range(len(pinned.split(",")))
                    )
                else:
                    env["HIP_VISIBLE_DEVICES"] = pinned
                    if pinned != "-1":
                        env.pop("ROCR_VISIBLE_DEVICES", None)
        except Exception as e:
            logger.debug("Failed to set ROCm visibility env vars for child: %s", e)

    @staticmethod
    def _unmasked_child_gpu_physical_ids() -> Optional[tuple[int, ...]]:
        from utils.hardware import get_parent_visible_gpu_ids

        parent_gpu_ids = get_parent_visible_gpu_ids()
        if len(parent_gpu_ids) == 1:
            return (int(parent_gpu_ids[0]),)
        return None

    @staticmethod
    def _inherited_child_gpu_order(pinned_ids: Iterable[int]) -> Optional[list[int]]:
        """Ids in parent mask order, or None if already that order or the mask is partial; numeric masks set device order, so re-emitting verbatim undoes it."""
        inherited = LlamaCppBackend._resolve_visible_physical_ids()
        if not inherited:
            # A uuid mask the ordinal resolvers leave open (FASTEST_FIRST) still orders
            # the cards the memory probe matched by uuid.
            inherited = LlamaCppBackend._smi_uuid_mask_order()
        if not inherited:
            return None
        wanted = [int(i) for i in pinned_ids]
        ordered = [i for i in inherited if i in set(wanted)]
        if sorted(ordered) != sorted(wanted) or ordered == wanted:
            return None
        return ordered

    def _planned_child_gpu_order(
        self,
        *,
        gpu_ids,
        gpu_indices,
        tp_tensor_split,
        extra_args,
        env,
        is_vulkan_backend: bool,
        cpu_forced: bool = False,
    ) -> "tuple[Optional[list[int]], Optional[list[int]]]":
        """``(device order, visibility mask order)`` the spawn will give the child; mask None when
        inherited."""
        order, split = self._picker_ordered_gpu_indices(gpu_ids, gpu_indices, tp_tensor_split)
        if order is None or is_vulkan_backend or cpu_forced:
            return (None if order is None else [int(i) for i in order]), None
        order = [int(i) for i in order]
        mask = list(order)
        if not gpu_ids:
            inherited = self._inherited_child_gpu_order(mask)
            if (
                inherited is not None
                and not _extra_args_have_tensor_split(extra_args, env)
                and (not split or len(split) == len(mask))
            ):
                mask = inherited
        try:
            if LlamaCppBackend._pinned_child_visibility(mask) is None:
                return order, None
        except Exception:
            return order, None
        return order, mask

    @staticmethod
    def _picker_ordered_gpu_indices(gpu_ids, gpu_indices, tp_tensor_split):
        """``(gpu_indices, tp_tensor_split)`` in the picker's order; a reorder the shares cannot
        follow is vetoed."""
        if not gpu_ids:
            return gpu_indices, tp_tensor_split
        picked = [int(i) for i in gpu_ids]
        if gpu_indices is None:
            return picked, tp_tensor_split
        before = [int(i) for i in gpu_indices]
        kept = set(before)
        reordered = [i for i in picked if i in kept] + [i for i in before if i not in set(picked)]
        if tp_tensor_split and len(tp_tensor_split) == len(before):
            share_by_id = dict(zip(before, tp_tensor_split))
            if all(i in share_by_id for i in reordered):
                return reordered, [share_by_id[i] for i in reordered]
        elif not tp_tensor_split:
            return reordered, tp_tensor_split
        return gpu_indices, tp_tensor_split

    @staticmethod
    def _repoint_emitted_tensor_split(
        cmd: list, old_order: list[int], new_order: list[int]
    ) -> bool:
        """Moves --tensor-split shares onto reordered devices, since shares are positional: else the roomier card gets the smaller share. False vetoes the reorder."""
        try:
            at = len(cmd) - 1 - cmd[::-1].index("--tensor-split")
        except ValueError:
            return True  # no shares emitted, nothing positional to carry
        if at + 1 >= len(cmd):
            return False
        shares = cmd[at + 1].split(",")
        if len(shares) != len(old_order):
            return False
        by_id = dict(zip(old_order, shares))
        cmd[at + 1] = ",".join(by_id[i] for i in new_order)
        return True

    @staticmethod
    def _pin_visible_gpu_order_for_split(env: dict) -> Optional[tuple[int, ...]]:
        """Pin the child's GPU enumeration to the picker's order for a manual
        ``--tensor-split`` across the whole visible set. CUDA's default
        FASTEST_FIRST enumeration applies the shares to the wrong cards on
        heterogeneous hosts (#5025), and CUDA_DEVICE_ORDER only fixes the
        numbering base: an inherited numeric visibility mask ALSO defines
        enumeration order, so a reordered parent mask (CUDA_VISIBLE_DEVICES=3,1)
        would still hand the shares to the wrong cards. The UI built the split
        positionally over get_backend_visible_gpu_info's device list (ascending
        physical via nvidia-smi, inherited mask order on the torch fallback), so
        re-emit the same set in that report order -- not an assumed ascending
        sort. The visible set itself never changes. No mask, an empty mask, or a
        UUID/MIG mask (which resolves to None) is left alone -- the multi-GPU
        controls are hidden for the latter."""
        mig_rows = LlamaCppBackend._mig_snapshot_rows()
        if mig_rows is not None:
            # R570+ MIG: pin UUIDs; numeric ids would resolve against the whole host.
            from utils.hardware import mig

            # Snapshot order = picker order; never sort (a numeric mask 2,0 is not ascending).
            order = [int(r["index"]) for r in mig_rows]
            if not order:
                return None
            env["CUDA_VISIBLE_DEVICES"] = mig.visibility_for(mig_rows, order)
            return tuple(order)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        inherited = LlamaCppBackend._resolve_visible_physical_ids()
        if not inherited:
            return None
        order = None
        try:
            from utils.hardware import get_backend_visible_gpu_info
            info = get_backend_visible_gpu_info()
            if info.get("available") and info.get("index_kind") == "physical":
                reported = [d["index"] for d in info.get("devices", [])]
                if sorted(reported) == sorted(inherited):
                    order = reported
        except Exception as e:
            logger.debug("Could not read reported GPU order for split pin: %s", e)
        if order is None:
            order = sorted(inherited)
        # Re-emit via ROCR when the parent masked there, else HSA sees hidden GPUs and may segfault.
        prefer_rocr = (
            sys.platform != "win32"
            and env.get("HIP_VISIBLE_DEVICES") is None
            and env.get("ROCR_VISIBLE_DEVICES") is not None
        )
        LlamaCppBackend._emit_child_gpu_visibility(
            env, ",".join(str(i) for i in order), prefer_rocr = prefer_rocr
        )
        return tuple(int(i) for i in order)

    @staticmethod
    def _vulkan_probe_answered(binary: Optional[str]) -> bool:
        """Whether the Vulkan probe returned any rows; the iGPU check maps no answer to not integrated."""
        try:
            return bool(LlamaCppBackend._run_vulkan_probe(binary))
        except Exception:
            return False

    @staticmethod
    def _vulkan_targets_are_igpus(binary: Optional[str], gpu_indices = None) -> bool:
        """True if any Vulkan device is an iGPU, whose VRAM is shared host RAM and worth page-locking."""
        try:
            rows = LlamaCppBackend._run_vulkan_probe(binary)
        except Exception:
            return False
        if not rows:
            return False
        wanted = set(gpu_indices) if gpu_indices else None
        selected = [r for r in rows if wanted is None or r["index"] in wanted]
        return any(r["is_igpu"] for r in selected)

    def _weights_in_host_memory(
        self,
        *,
        fully_gpu_offloaded: bool,
        gpu_memory_mode: Optional[str],
        gpu_layers: Optional[int],
        extra_args: Optional[Iterable[str]],
        gpu_indices = None,
        is_vulkan_backend: bool = False,
        binary: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        probe_vulkan: bool = True,
        fit_active: bool = False,
    ) -> bool:
        """Errs towards True: it only gates skipping page-lock, so an unknown placement must keep mlock."""
        from utils.hardware import is_apple_silicon

        # CPU placement in extras or inherited env keeps weights in RAM whatever the layer count: check on the auto path too, never short-circuit past it.
        if _args_place_tensors_on_cpu(extra_args):
            return True
        if _env_places_tensors_on_cpu(env):
            return True
        # A --device naming no GPU beats any layer count: nowhere to offload, so the model stays in host RAM.
        if _device_selection_is_cpu(extra_args, env):
            return True
        # User extras are appended last (last-wins), so -ngl or an enabled fit voids the prediction.
        if _extra_args_set_any_flag(extra_args, _GPU_LAYER_FLAGS) or fit_is_enabled_in(extra_args):
            fully_gpu_offloaded = False
        all_on_gpu = fully_gpu_offloaded or self._offloads_every_layer(
            gpu_memory_mode = gpu_memory_mode,
            gpu_layers = gpu_layers,
            extra_args = extra_args,
            fit_active = fit_active,
        )
        if not all_on_gpu:
            return True
        # Unified memory: the VRAM is host RAM, so weights stay pageable however fully they are offloaded.
        if is_apple_silicon():
            return True
        if is_vulkan_backend:
            # gpu_indices are Vulkan ordinals here; the ROCm helper would misread them.
            return not probe_vulkan or self._vulkan_targets_are_igpus(binary, gpu_indices)
        return self._amd_apu_wants_unified_memory(gpu_indices)

    def _offloads_every_layer(
        self,
        *,
        gpu_memory_mode: Optional[str],
        gpu_layers: Optional[int],
        extra_args: Optional[Iterable[str]],
        fit_active: bool = False,
    ) -> bool:
        """Full offload via a route fully_gpu_offloaded misses (manual max or user -ngl); unknowns (unreadable blocks, CPU experts, tensor overrides) answer False."""
        n_layers = self.n_layers
        if not n_layers:
            return False
        if self._n_cpu_moe:
            return False
        # Parsed predicate, not presence: --n-cpu-moe 0 places nothing.
        if _args_place_tensors_on_cpu(extra_args):
            return False
        requested: Optional[int] = None
        if gpu_memory_mode == "manual" and gpu_layers is not None and gpu_layers >= 0:
            # Manual owns offload and strips -ngl from extras, so the field rules; picker max is n_layers + 1.
            requested = gpu_layers
        else:
            try:
                requested = parse_gpu_layers_override(extra_args)
            except ValueError:
                return False
        if requested is None:
            return False
        # -1 is llama.cpp's default, so an active fitter may lower it; a concrete count stands.
        if requested == -1:
            return not fit_active
        return requested > n_layers

    def _partially_offloads_layers(
        self,
        args: Optional[Iterable[str]],
        env: Optional[Mapping[str, str]] = None,
        fit_implies_partial: bool = True,
    ) -> bool:
        """A concrete layer count owns placement; 0 is CPU-only, not partial, and keeps CPU MTP; --fit on implies partial only in Auto, never Manual."""
        n_layers = self.n_layers
        if not n_layers:
            return False
        try:
            requested = parse_gpu_layers_override(args)
        except ValueError:
            return False
        source_env = os.environ if env is None else env
        if _device_selection_is_cpu(args, source_env):
            return False
        if requested is None:
            raw = source_env.get("LLAMA_ARG_N_GPU_LAYERS")
            if raw is not None and raw.strip():
                try:
                    requested = int(raw)
                except ValueError:
                    requested = None
        if requested == 0:
            return False
        if requested is not None and requested > 0:
            # llama.cpp needs a count above block_count to include the output layer (_offloads_every_layer).
            return requested <= n_layers
        return fit_implies_partial and fit_is_effectively_on(args, source_env)

    @staticmethod
    def _torch_is_rocm(torch) -> bool:
        """Whether this torch is a ROCm build. AMD SDK wheels leave
        ``version.hip`` unset and only encode "rocm" in ``__version__``."""
        return (
            getattr(torch.version, "hip", None) is not None
            or "rocm" in getattr(torch, "__version__", "").lower()
        )

    @staticmethod
    def _host_torch_is_rocm() -> bool:
        """Whether torch here is a ROCm build, without the caller importing torch.
        False when torch is missing or unreadable. Cheap guard so a CUDA or
        CPU-only host does not pay for ROCm-only work (#7624)."""
        try:
            import torch
            return LlamaCppBackend._torch_is_rocm(torch)
        except Exception:
            return False

    @staticmethod
    def _rocm_classification_answered() -> bool:
        """Kept apart: a page-lock skip may treat no APU and could-not-tell alike; a loader choice
        may not."""
        try:
            import torch

            if not LlamaCppBackend._torch_is_rocm(torch):
                return False
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return False
            from core.training.worker import _rocm_classify_unified_memory

            count = torch.cuda.device_count()
            if count <= 0:
                return False
            # Classify every visible device, not just the selected ones: mapping ggml ordinals onto torch ones is not owned here; unreadable ones read discrete.
            for ordinal in range(count):
                _rocm_classify_unified_memory(torch.cuda.get_device_properties(ordinal))
            return True
        except Exception:
            return False

    @staticmethod
    def _rocm_unified_memory_gpu_ids() -> set[int]:
        """PHYSICAL ids of visible ROCm APUs, classified by the training worker so both paths agree."""
        try:
            import torch

            if not LlamaCppBackend._torch_is_rocm(torch):
                return set()
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return set()
            from core.training.worker import _rocm_classify_unified_memory

            # Map ordinal to physical id via the ROCm mask (HIP, then ROCR, then CUDA), as _get_gpu_memory does.
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            unified: set[int] = set()
            for ordinal in range(torch.cuda.device_count()):
                try:
                    _arch, is_unified = _rocm_classify_unified_memory(
                        torch.cuda.get_device_properties(ordinal)
                    )
                except Exception:
                    continue
                if not is_unified:
                    continue
                unified.add(
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
            return unified
        except Exception:
            return set()

    @staticmethod
    def _rocm_selected_pool_mib(gpu_indices = None) -> Optional[int]:
        """Combined carve-out MiB of selected APUs, or None if any device is not one (ggml's is global)."""
        try:
            import torch

            if not LlamaCppBackend._torch_is_rocm(torch):
                return None
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return None
            from core.training.worker import _rocm_classify_unified_memory

            wanted = None if gpu_indices is None else set(gpu_indices)
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            total_mib = 0
            seen: set[int] = set()
            for ordinal in range(torch.cuda.device_count()):
                pid = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                if wanted is not None and pid not in wanted:
                    continue
                try:
                    props = torch.cuda.get_device_properties(ordinal)
                    _arch, is_unified = _rocm_classify_unified_memory(props)
                except Exception:
                    return None
                if not is_unified:
                    return None
                total = int(getattr(props, "total_memory", 0) or 0)
                if total <= 0:
                    return None
                total_mib += total // (1024 * 1024)
                seen.add(pid)
            if not seen or (wanted is not None and seen != wanted):
                return None
            return total_mib
        except Exception:
            return None

    @staticmethod
    def _unified_memory_would_help(gpu_indices = None, need_bytes = None) -> bool:
        """Only if host RAM is the larger pool vs the carve-out total and weights overflow the carve-out, since managed pages fault in k_set_rows on gfx1151."""
        try:
            pool_mib = LlamaCppBackend._rocm_selected_pool_mib(gpu_indices)
            if not pool_mib:
                return False
            host_mib = LlamaCppBackend._available_system_memory_mib()
            if not host_mib:
                return False
            if int(host_mib) <= int(pool_mib):
                return False
            if need_bytes is None:
                return False
            return int(need_bytes) > int(pool_mib) * 1024 * 1024
        except Exception:
            return False

    @staticmethod
    def _rocm_arch_by_physical_id() -> dict[int, str]:
        """Lowercased base gfx arch per PHYSICAL id, honoring the ROCm mask; alternate SDK attr spellings are needed or the gate silently fails open."""
        arch_by_id: dict[int, str] = {}
        try:
            import torch

            if not LlamaCppBackend._torch_is_rocm(torch):
                return arch_by_id
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return arch_by_id
            from core.training.worker import _rocm_classify_unified_memory

            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            for ordinal in range(torch.cuda.device_count()):
                try:
                    _arch, _is_unified = _rocm_classify_unified_memory(
                        torch.cuda.get_device_properties(ordinal)
                    )
                except Exception:
                    continue
                _base = (_arch or "").split(":")[0].strip().lower()
                if not _base:
                    continue
                pid = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                arch_by_id[pid] = _base
        except Exception:
            return arch_by_id
        return arch_by_id

    @staticmethod
    def _offload_target_shares_system_memory(
        *, is_vulkan_backend: bool, shared_gpu_ids, detected_gpus, gpu_indices
    ) -> bool:
        """True only when every device shares host RAM as VRAM; mixed, unreadable or unpinned is False. Keep Vulkan ordinals and physical ids in separate sets."""
        devices = (
            list(gpu_indices) if gpu_indices else [idx for idx, _free in (detected_gpus or ())]
        )
        if not devices:
            return False
        if is_vulkan_backend:
            shared = set(shared_gpu_ids or ())
            return bool(shared) and all(idx in shared for idx in devices)
        # ROCm only: a CUDA get_device_properties() leaks a ~700 MiB primary context.
        try:
            unified = LlamaCppBackend._rocm_unified_memory_gpu_ids()
        except Exception:
            return False
        return bool(unified) and all(idx in unified for idx in devices)

    @staticmethod
    def _amd_apu_wants_unified_memory(gpu_indices = None) -> bool:
        """True only for AMD APUs, since GGML_CUDA_ENABLE_UNIFIED_MEMORY hurts discrete GPUs."""
        try:
            unified = LlamaCppBackend._rocm_unified_memory_gpu_ids()
            if gpu_indices is None:
                return bool(unified)
            return any(_i in unified for _i in gpu_indices)
        except Exception:
            return False

    # PHYSICAL integrated ids, keyed per visibility mask; integratedness cannot change under a fixed mask.
    _INTEGRATED_CUDA_IDS: dict[tuple, set[int]] = {}

    @staticmethod
    def _integrated_cuda_mask_key() -> tuple:
        return tuple(
            os.environ.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        )

    @staticmethod
    def _integrated_cuda_probe_is_free() -> bool:
        """True if the integrated-CUDA answer is cached for this mask; the preflight never probes, since the budget is already taken and both arms fill the cache."""
        return LlamaCppBackend._integrated_cuda_mask_key() in LlamaCppBackend._INTEGRATED_CUDA_IDS

    @staticmethod
    def _integrated_cuda_gpu_ids() -> set[int]:
        """PHYSICAL ids of visible integrated CUDA GPUs; ROCm is left to _rocm_unified_memory_gpu_ids."""
        cached = LlamaCppBackend._INTEGRATED_CUDA_IDS.get(
            LlamaCppBackend._integrated_cuda_mask_key()
        )
        if cached is not None:
            # Settled: a second pass over the devices could learn nothing.
            return set(cached)
        try:
            import torch

            if LlamaCppBackend._torch_is_rocm(torch):
                return set()
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return set()
            # Same ordinal-to-physical map as the ROCm twin, so masked CUDA_VISIBLE_DEVICES=2 hosts stay correct.
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            integrated: set[int] = set()
            # An incomplete pass is returned but not cached, so a failed query is retried.
            complete = True
            for ordinal in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(ordinal)
                except Exception:
                    complete = False
                    continue
                # Both attribute spellings: torch renamed `integrated`.
                if not (
                    getattr(props, "is_integrated", False) or getattr(props, "integrated", False)
                ):
                    continue
                integrated.add(
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
            # Cached for the preflight only if the pass reached every device; a torch that raised says nothing.
            if complete:
                LlamaCppBackend._INTEGRATED_CUDA_IDS[
                    LlamaCppBackend._integrated_cuda_mask_key()
                ] = integrated
            return integrated
        except Exception:
            return set()

    # {physical id: pool total MiB} per visibility mask, filled beside _INTEGRATED_CUDA_IDS.
    _INTEGRATED_CUDA_POOL_MIB: dict[tuple, dict[int, int]] = {}

    @staticmethod
    def _integrated_cuda_pool_total_mib() -> dict[int, int]:
        """Pool MiB per integrated CUDA device, from props.total_memory: what it can actually allocate."""
        key = LlamaCppBackend._integrated_cuda_mask_key()
        cached = LlamaCppBackend._INTEGRATED_CUDA_POOL_MIB.get(key)
        if cached is not None:
            return dict(cached)
        integrated = LlamaCppBackend._integrated_cuda_gpu_ids()
        if not integrated:
            return {}
        try:
            import torch

            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            totals: dict[int, int] = {}
            complete = True
            for ordinal in range(torch.cuda.device_count()):
                idx = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                if idx not in integrated:
                    continue
                try:
                    props = torch.cuda.get_device_properties(ordinal)
                    totals[idx] = int(props.total_memory) // (1024 * 1024)
                except Exception:
                    complete = False
            if complete:
                LlamaCppBackend._INTEGRATED_CUDA_POOL_MIB[key] = totals
            return totals
        except Exception:
            return {}

    @staticmethod
    def _widen_integrated_cuda_rows(gpus: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        """Only widens: a total is replaced by a larger pool, and free is floored at the CLI reading."""
        try:
            integrated = LlamaCppBackend._integrated_cuda_gpu_ids()
            if not integrated or not gpus:
                return gpus
            # nvidia-smi and CUDA order differ unless CUDA_DEVICE_ORDER=PCI_BUS_ID and the mask maps.
            order = (os.environ.get("CUDA_DEVICE_ORDER") or "").strip().upper()
            raw_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
            mask_unset = raw_mask is None or not raw_mask.strip()
            mappable = mask_unset or LlamaCppBackend._resolve_visible_physical_ids() is not None
            # No single-row exemption; main.py sets PCI_BUS_ID, so Spark/Jetson/N1X qualify.
            if not (mappable and order == "PCI_BUS_ID"):
                logger.debug(
                    "Not widening integrated CUDA rows: CUDA_DEVICE_ORDER is %r and "
                    "the mask %s, so a CUDA id cannot be joined to an nvidia-smi index "
                    "here.",
                    order or "unset (FASTEST_FIRST)",
                    "resolves to no physical ids" if not mappable else "is mappable",
                )
                return gpus
            totals = LlamaCppBackend._integrated_cuda_pool_total_mib()
            avail = LlamaCppBackend._available_system_memory_mib()
            cgroup_mib = LlamaCppBackend._cgroup_available_memory_mib()
            # Integrated devices share one pool, so divide by their count.
            shared_count = max(1, sum(1 for idx, _f, _t in gpus if idx in integrated))
            out: list[tuple[int, int, int]] = []
            for idx, free_mib, total_mib in gpus:
                if idx not in integrated:
                    out.append((idx, free_mib, total_mib))
                    continue
                cli_total_mib = total_mib
                pool_mib = totals.get(idx) or 0
                if pool_mib > total_mib:
                    total_mib = pool_mib
                raw_mib = free_mib
                # MemAvailable: both free readings undercount here. A zero total never caps.
                if avail is not None and total_mib > 0:
                    raw_mib = min(total_mib, max(raw_mib, avail))
                cgroup_bound = False
                if cgroup_mib is not None and cgroup_mib <= raw_mib:
                    # Charged to the cgroup, so the free reading is the real ceiling.
                    raw_mib = cgroup_mib
                    cgroup_bound = True
                if shared_count > 1:
                    raw_mib //= shared_count
                    total_mib //= shared_count
                capped = _apply_igpu_host_reserve_mib(raw_mib, True)
                # Floor at the driver's free reading, except under a cgroup (charged to memory.max).
                if not cgroup_bound:
                    capped = max(capped, free_mib // shared_count)
                # Log only on change: _get_gpu_memory is uncached and polled every 0.25 s.
                if capped != free_mib or total_mib != cli_total_mib:
                    logger.info(
                        f"CUDA device {idx} is a unified-memory SoC sharing system RAM; "
                        f"sizing it against the shared pool "
                        f"({free_mib}->{capped}MiB free, "
                        f"{pool_mib or total_mib}MiB pool)"
                    )
                out.append((idx, capped, 0 if cgroup_bound else total_mib))
            return sorted(out, key = lambda g: g[0])
        except Exception as e:
            logger.debug(f"integrated CUDA pool sizing failed: {e}")
            return gpus

    @staticmethod
    def _integrated_cuda_unified_memory(gpu_indices = None) -> bool:
        """True if any credited CUDA device is an integrated SoC, since its pool would be double counted."""
        try:
            integrated = LlamaCppBackend._integrated_cuda_gpu_ids()
            if gpu_indices is None:
                return bool(integrated)
            return any(_i in integrated for _i in gpu_indices)
        except Exception:
            return False

    @staticmethod
    def _integrated_cuda_selection_is_all_shared(gpu_indices = None) -> bool:
        """True only if EVERY credited CUDA device is integrated; any discrete card voids it, else an otherwise fitting load is called oversized."""
        try:
            integrated = LlamaCppBackend._integrated_cuda_gpu_ids()
            if not integrated:
                return False
            if gpu_indices is not None:
                return bool(gpu_indices) and all(_i in integrated for _i in gpu_indices)
            # Nothing pinned, so the credited set is every visible device.
            visible = LlamaCppBackend._resolve_visible_physical_ids()
            if visible is None:
                import torch
                visible = list(range(torch.cuda.device_count()))
            return bool(visible) and all(_i in integrated for _i in visible)
        except Exception:
            return False

    # "Off" spellings ggml itself ignores (it tests presence); we honour them.
    _UNIFIED_MEMORY_OFF = frozenset({"", "0", "false", "no", "off"})

    @staticmethod
    def _unified_memory_opted_out(env = None) -> bool:
        """True if GGML_CUDA_ENABLE_UNIFIED_MEMORY must be unset (ggml tests presence, so =0 is no opt-out); UNSLOTH_DISABLE_UNIFIED_MEMORY=1 also unsets it."""
        try:
            source = os.environ if env is None else env
            if str(source.get("UNSLOTH_DISABLE_UNIFIED_MEMORY", "")).strip() == "1":
                return True
            value = source.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY")
            if value is None:
                return False
            return str(value).strip().lower() in LlamaCppBackend._UNIFIED_MEMORY_OFF
        except Exception:
            return False

    @staticmethod
    def _unified_memory_opted_in(env = None) -> bool:
        """True when UNSLOTH_ENABLE_UNIFIED_MEMORY=1 asks for managed allocation even
        though the weights fit. Exact "1"; the opt-out wins over it."""
        try:
            source = os.environ if env is None else env
            return str(source.get("UNSLOTH_ENABLE_UNIFIED_MEMORY", "")).strip() == "1"
        except Exception:
            return False

    @staticmethod
    def _unified_memory_for_launch(
        gpu_indices,
        need_bytes,
        *,
        opted_in = False,
    ) -> bool:
        """The launch-time decision behind GGML_CUDA_ENABLE_UNIFIED_MEMORY. It is
        process-wide and hurts discrete cards, so a mixed selection is always False."""
        if opted_in and LlamaCppBackend._rocm_selected_pool_mib(gpu_indices) is not None:
            return True
        return LlamaCppBackend._unified_memory_would_help(gpu_indices, need_bytes = need_bytes)

    # Whole-word so "a100" does not match "RTX A1000".
    _DATACENTER_GPU_RE = re.compile(
        r"\b(?:a100|a30|h100|h200|h800|gh200|b200|b100|b300|gb200|gb300|"
        r"l40s?|l4|rtx pro 6000|rtx 6000 ada)\b"
    )

    # NVLink-capable subset; PCIe peer copies may be silently dropped by an IOMMU.
    # Necessary, not sufficient: _p2p_veto_reason also needs a positive NV# link.
    _NVLINK_FABRIC_GPU_RE = re.compile(
        r"\b(?:a100|a30|h100|h200|h800|gh200|b200|b100|b300|gb200|gb300)\b"
    )

    @staticmethod
    def _all_selected_gpus_match(pattern, gpu_indices = None) -> bool:
        """True iff every selected NVIDIA GPU name matches pattern (mixed, consumer, ROCm or CPU is False); maps physical ids to mask-relative ordinals for lookup."""
        try:
            import torch

            if getattr(torch.version, "hip", None) is not None:
                return False  # ROCm reuses torch.cuda.*; not a CUDA part
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return False
            count = torch.cuda.device_count()

            # Mirror _get_gpu_free_memory; unset or unparsable CUDA_VISIBLE_DEVICES keeps physical id == ordinal.
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()

            names_by_id: dict[int, str] = {}
            for ordinal in range(count):
                try:
                    name = (torch.cuda.get_device_properties(ordinal).name or "").lower()
                except Exception:
                    continue
                pid = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                names_by_id[pid] = name

            indices = list(gpu_indices) if gpu_indices else list(names_by_id)
            saw = False
            for _i in indices:
                name = names_by_id.get(_i)
                if name is None:
                    continue  # not visible -> skip (fail conservative)
                saw = True
                if not pattern.search(name):
                    return False
            return saw
        except Exception:
            return False

    @staticmethod
    def _is_datacenter_gpu(gpu_indices = None) -> bool:
        """True iff every selected GPU qualifies for the FP32-accum tuning. Mask
        handling: _all_selected_gpus_match. Peer-copy safety: _p2p_veto_reason."""
        return LlamaCppBackend._all_selected_gpus_match(
            LlamaCppBackend._DATACENTER_GPU_RE, gpu_indices
        )

    # NV# is a bonded NVLink set; SYS/NODE/PHB/PXB/PIX cross PCIe. Header is ANSI-underlined.
    _TOPO_GPU_LABEL_RE = re.compile(r"^GPU(\d+)$")
    _TOPO_NVLINK_RE = re.compile(r"^NV\d+$")
    _ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    # None = not probed; else a 1-tuple whose value may be None (unavailable).
    _NVLINK_TOPO_CACHE = None
    # Bumped on refresh so a slow prime cannot publish over a newer invalidation.
    _NVLINK_TOPO_GENERATION = 0
    _NVLINK_TOPO_LOCK = threading.Lock()

    # Its own vocabulary: a fabricated NV18 could be misread as a link count.
    _NVML_NVLINK_LABEL = "NVLINK"
    _NVML_NO_NVLINK_LABEL = "NO-NVLINK"

    # nvml.h. Only scalars and opaque handles cross, so no struct layout to get wrong.
    _NVML_SUCCESS = 0
    _NVML_P2P_CAPS_INDEX_NVLINK = 2
    _NVML_P2P_STATUS_OK = 0
    # 1..5 are the documented denials; UNKNOWN (6) and anything past the enum are not.
    _NVML_P2P_STATUS_DEFINITE_NEGATIVES = frozenset({1, 2, 3, 4, 5})
    _NVML_FEATURE_ENABLED = 1
    # Link enumeration stops at the first error; this only bounds a runaway loop.
    _NVML_MAX_LINKS = 64

    @classmethod
    def _matrix_is_nvml(cls, matrix: dict) -> bool:
        """Whether this matrix came from NVML, i.e. whether nvidia-smi is known to
        have answered."""
        return any(
            v in (cls._NVML_NVLINK_LABEL, cls._NVML_NO_NVLINK_LABEL) for v in matrix.values()
        )

    @classmethod
    def _label_is_nvlink(cls, label: str) -> bool:
        """The one place that decides whether a label means NVLink, so the
        selected-pair check and the whole-box check cannot drift apart."""
        return bool(cls._TOPO_NVLINK_RE.match(label)) or label == cls._NVML_NVLINK_LABEL

    @staticmethod
    def _nvml_library():
        """The driver's own NVML, via ctypes: present exactly when nvidia-smi is, and
        no Python dependency (nvidia-ml-py is NOT installed in a Studio env). None
        when it cannot be loaded."""
        if os.name == "nt":
            # nvml.dll can sit in NVSMI\ rather than a DLL search path, as nvidia-smi.exe
            # does (see _nvidia_smi_executable in utils/hardware/nvidia.py).
            names = ["nvml.dll"]
            for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432")):
                if root:
                    names.append(os.path.join(root, "NVIDIA Corporation", "NVSMI", "nvml.dll"))
            windir = os.environ.get("SystemRoot")
            if windir:
                names.append(os.path.join(windir, "System32", "nvml.dll"))
        else:
            names = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for name in names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        return None

    # ctypes has no subprocess timeout and a wedged driver blocks nvmlInit_v2 forever.
    _NVML_PROBE_TIMEOUT_SECONDS = 10

    @classmethod
    def _probe_nvml_nvlink_topology(cls) -> Optional[dict]:
        """_probe_nvml_nvlink_topology_inner under a wall-clock bound. A stalled NVML
        call cannot be cancelled, so the thread is abandoned as a daemon and the caller
        falls through to the shell-out, which has a timeout of its own."""
        box = {}

        def _run():
            box["matrix"] = cls._probe_nvml_nvlink_topology_inner()

        worker = threading.Thread(target = _run, daemon = True, name = "nvml-nvlink-probe")
        worker.start()
        worker.join(cls._NVML_PROBE_TIMEOUT_SECONDS)
        if worker.is_alive():
            logger.debug(
                f"NVML NVLink probe still running after "
                f"{cls._NVML_PROBE_TIMEOUT_SECONDS}s; abandoning it"
            )
            return None
        return box.get("matrix")

    @classmethod
    def _probe_nvml_nvlink_topology_inner(cls) -> Optional[dict]:
        """NVML per-pair evidence: an NVSwitch link proves only attachment, so every GPU linked is not every pair reachable; None when inconclusive, then shell out."""
        lib = cls._nvml_library()
        if lib is None:
            return None
        initialised = False
        try:
            try:
                get_status = lib.nvmlDeviceGetP2PStatus
                get_count = lib.nvmlDeviceGetCount_v2
                get_handle = lib.nvmlDeviceGetHandleByIndex_v2
                get_link_state = lib.nvmlDeviceGetNvLinkState
                nvml_init = lib.nvmlInit_v2
            except AttributeError as e:
                # An older driver missing these still has the working shell-out.
                logger.debug(f"NVML lacks a required symbol: {e}")
                return None
            for fn in (get_status, get_count, get_handle, get_link_state, nvml_init):
                fn.restype = ctypes.c_int
            if nvml_init() != cls._NVML_SUCCESS:
                return None
            initialised = True

            count = ctypes.c_uint()
            if get_count(ctypes.byref(count)) != cls._NVML_SUCCESS:
                return None
            if count.value < 2:
                return None  # peer copies need two devices; nothing to say

            handles = []
            for i in range(count.value):
                handle = ctypes.c_void_p()
                if get_handle(ctypes.c_uint(i), ctypes.byref(handle)) != cls._NVML_SUCCESS:
                    return None  # partial inventory: never approve the rest
                handles.append(handle)

            # Per-device active links, used only to contradict a positive.
            active_links = []
            for handle in handles:
                seen = 0
                for link in range(cls._NVML_MAX_LINKS):
                    state = ctypes.c_uint()
                    rc = get_link_state(handle, ctypes.c_uint(link), ctypes.byref(state))
                    if rc != cls._NVML_SUCCESS:
                        break  # past this device's last link, or unsupported
                    if state.value == cls._NVML_FEATURE_ENABLED:
                        seen += 1
                active_links.append(seen)

            matrix: dict = {}
            for a in range(count.value):
                for b in range(count.value):
                    if a == b:
                        continue
                    status = ctypes.c_uint()
                    rc = get_status(
                        handles[a],
                        handles[b],
                        ctypes.c_uint(cls._NVML_P2P_CAPS_INDEX_NVLINK),
                        ctypes.byref(status),
                    )
                    if rc != cls._NVML_SUCCESS:
                        return None  # unknown for one pair is unknown for all
                    linked = status.value == cls._NVML_P2P_STATUS_OK
                    if not linked and status.value not in cls._NVML_P2P_STATUS_DEFINITE_NEGATIVES:
                        # NO-NVLINK would read as conclusive and rob `topo -m` of the say.
                        logger.debug(
                            f"NVML P2P status {status.value} for GPU {a}<->{b} is not a "
                            "definite answer; treating the topology as unreadable"
                        )
                        return None
                    # An OK on a PCIe-only box becomes unknown, not a false positive.
                    if linked and not (active_links[a] and active_links[b]):
                        logger.debug(
                            f"NVML says GPU {a}<->{b} is NVLink but they report "
                            f"{active_links[a]}/{active_links[b]} active links; "
                            "treating the topology as unreadable"
                        )
                        return None
                    matrix[(a, b)] = cls._NVML_NVLINK_LABEL if linked else cls._NVML_NO_NVLINK_LABEL

            # By key, not count: right-sized with wrong keys passes a length check.
            expected = {(a, b) for a in range(count.value) for b in range(count.value) if a != b}
            if set(matrix) != expected:
                return None
            return matrix
        except Exception as e:
            logger.debug(f"NVML NVLink probe failed: {e}")
            return None
        finally:
            if initialised:
                try:
                    lib.nvmlShutdown()
                except Exception:
                    pass

    @staticmethod
    def _probe_nvlink_topology() -> Optional[dict]:
        """Keyed by nvidia-smi PCI index, not CUDA ordinal; only GPU<n> rows and columns are read; None when the topology cannot be established."""
        try:
            from utils.hardware import gpu_query

            result = gpu_query.run_nvidia_smi(
                ["nvidia-smi", "topo", "-m"],
                cache = False,
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                env = child_env_without_native_path_secret(),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode != 0:
                return None
            columns: list[int] = []
            matrix: dict = {}
            for raw in result.stdout.splitlines():
                line = LlamaCppBackend._ANSI_ESCAPE_RE.sub("", raw)
                if line.strip().lower().startswith("legend"):
                    break
                tokens = line.split()
                if not tokens:
                    continue
                if not columns:
                    # Header: the leading run of GPU<n> labels names the columns.
                    for token in tokens:
                        m = LlamaCppBackend._TOPO_GPU_LABEL_RE.match(token)
                        if m is None:
                            break
                        columns.append(int(m.group(1)))
                    # Data rows lead with GPU<n> too, so one column is more likely
                    # a missing header than a real table: keep looking.
                    if len(columns) < 2:
                        columns = []
                    continue
                row = LlamaCppBackend._TOPO_GPU_LABEL_RE.match(tokens[0])
                if row is None:
                    continue  # NIC row, or the blank line before the legend
                source = int(row.group(1))
                labels = tokens[1 : 1 + len(columns)]
                if len(labels) != len(columns):
                    return None  # truncated row: refuse to guess
                for col, label in zip(columns, labels):
                    if col != source:
                        matrix[(source, col)] = label.upper()
            # Require every directed pair: truncated rows can look uniformly NVLinked.
            expected = {(a, b) for a in columns for b in columns if a != b}
            if set(matrix) != expected:
                logger.debug(
                    f"nvidia-smi topo -m: {len(matrix)} pairs for {len(columns)} "
                    f"GPUs, expected {len(expected)}; treating as unreadable"
                )
                return None
            return matrix or None
        except Exception as e:
            logger.debug(f"nvidia-smi topo probe failed: {e}")
            return None

    @classmethod
    def _probe_interconnect_matrix(cls) -> Optional[dict]:
        """NVML first, nvidia-smi topo -m only as fallback; UNSLOTH_P2P_TOPO_CROSSCHECK=1 runs both."""
        matrix = cls._probe_nvml_nvlink_topology()
        if matrix is not None and os.environ.get("UNSLOTH_P2P_TOPO_CROSSCHECK") == "1":
            topo = cls._probe_nvlink_topology()
            if topo is not None:
                nvml_yes = {k for k, v in matrix.items() if cls._label_is_nvlink(v)}
                topo_yes = {k for k, v in topo.items() if cls._label_is_nvlink(v)}
                if nvml_yes != topo_yes:
                    logger.warning(
                        "P2P topology cross-check disagreement: NVML-only pairs "
                        f"{sorted(nvml_yes - topo_yes)}, topo-only pairs "
                        f"{sorted(topo_yes - nvml_yes)}. Using topo -m. Please "
                        "report this with `nvidia-smi topo -m` output (#10613)."
                    )
                    return topo
        if matrix is not None:
            return matrix
        return cls._probe_nvlink_topology()

    @classmethod
    def prime_nvlink_topology(cls) -> None:
        """NVML only: a background prime must not spawn nvidia-smi (it can run its full timeout) and publishes only a success, never a miss."""
        if cls._NVLINK_TOPO_CACHE is not None:
            return
        if os.environ.get("UNSLOTH_P2P_TOPO_CROSSCHECK") == "1":
            # Do not publish here: only the load path runs the cross-check, and warm always primes first.
            return
        matrix = cls._probe_nvml_nvlink_topology()
        if matrix is None:
            return
        with cls._NVLINK_TOPO_LOCK:
            if cls._NVLINK_TOPO_CACHE is None:
                cls._NVLINK_TOPO_CACHE = (matrix,)

    @classmethod
    def _nvlink_topology(
        cls,
        refresh = False,
        cache_failure = True,
    ) -> Optional[dict]:
        """Probe runs unlocked; the generation check stops a slow result overwriting a newer
        invalidation."""
        if not refresh and cls._NVLINK_TOPO_CACHE is not None:
            return cls._NVLINK_TOPO_CACHE[0]
        with cls._NVLINK_TOPO_LOCK:
            generation = cls._NVLINK_TOPO_GENERATION
            if refresh:
                cls._NVLINK_TOPO_GENERATION += 1
                generation = cls._NVLINK_TOPO_GENERATION
            elif cls._NVLINK_TOPO_CACHE is not None:
                return cls._NVLINK_TOPO_CACHE[0]
        probed = cls._probe_interconnect_matrix()
        with cls._NVLINK_TOPO_LOCK:
            if generation == cls._NVLINK_TOPO_GENERATION:
                if probed is None and not cache_failure:
                    # A miss probed mid driver init would keep P2P off for the process.
                    return None
                existing = cls._NVLINK_TOPO_CACHE
                if probed is None and existing is not None and existing[0] is not None:
                    # Without this a racing transient failure would erase a concurrent pass's good matrix.
                    return existing[0]
                cls._NVLINK_TOPO_CACHE = (probed,)
                return probed
            # A refresh started mid-pass owns the answer; discard this one as stale.
            cached = cls._NVLINK_TOPO_CACHE
        return probed if cached is None else cached[0]

    @staticmethod
    def _running_virtualized() -> bool:
        """True under a hypervisor, where the IOMMU veto does not apply; False when unknown, keeping it."""
        try:
            with open("/proc/cpuinfo", "r", encoding = "utf-8", errors = "replace") as f:
                for line in f:
                    if line.startswith("flags") and ":" in line:
                        return "hypervisor" in line.split(":", 1)[1].split()
        except Exception:
            pass
        return False

    # One directory per IOMMU group, each with a `type`. Overridable for tests.
    _IOMMU_GROUPS_ROOT = "/sys/kernel/iommu_groups"

    # True if _get_gpu_memory ids are nvidia-smi PCI indices, False if torch ordinals; None until
    # probed. They coincide only under PCI_BUS_ID.
    _GPU_IDS_ARE_PCI_INDICES = None
    # index -> inherited uuid, when CUDA_VISIBLE_DEVICES named the NVML rows by uuid.
    _NVML_ROWS: list = []  # the last NVML inventory; _child_visibility_for reads it

    # MIG cards that same probe saw (None = discovery unknown), so pinning uses its snapshot.
    _SMI_MIG_INDICES: Optional[frozenset] = None

    _SMI_UUID_ROWS: list = []  # (index, uuid) from the last nvidia-smi memory probe

    # Boot-time property: cached for the default root only; an explicit root (tests) re-reads.
    _IOMMU_CACHE = None

    @classmethod
    def _iommu_is_translating_cached(cls) -> Optional[bool]:
        if cls._IOMMU_CACHE is None:
            try:
                cls._IOMMU_CACHE = (cls._iommu_is_translating(),)
            except Exception as e:
                logger.debug(f"IOMMU probe failed: {e}")
                cls._IOMMU_CACHE = (None,)
        return cls._IOMMU_CACHE[0]

    @staticmethod
    def _iommu_is_translating(root = None) -> Optional[bool]:
        """True if any IOMMU group translates DMA (bare-metal P2P then drops writes silently); None when groups exist but none is readable."""
        root = root or LlamaCppBackend._IOMMU_GROUPS_ROOT
        try:
            groups = os.listdir(root)
        except Exception:
            # Absent = nothing translates; present but unreadable = unknown.
            return None if os.path.isdir(root) else False
        if not groups:
            return False  # IOMMU compiled in but off
        kinds = []
        for group in groups:
            try:
                with open(
                    os.path.join(root, group, "type"),
                    "r",
                    encoding = "utf-8",
                    errors = "replace",
                ) as f:
                    kinds.append(f.read().strip().upper())
            except OSError:
                continue  # pre-5.x kernels have no `type`; other groups may
        if not kinds:
            return None
        return any(not kind.startswith("IDENTITY") for kind in kinds)

    @staticmethod
    def _mig_snapshot_rows():
        """R570+ MIG rows _get_gpu_memory sized under this mask and order, else None (index = CUDA id)."""
        snapshot = getattr(LlamaCppBackend, "_MIG_DEVICE_SNAPSHOT", None)
        if snapshot is not None and snapshot[:2] == (
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            os.environ.get("CUDA_DEVICE_ORDER"),
        ):
            return snapshot[2]
        return None

    @classmethod
    def _p2p_veto_reason(
        cls,
        gpu_indices = None,
        launch_order_pinned = False,
        ids_are_pci_indices = None,
    ) -> Optional[str]:
        """Fail-closed wrapper around _p2p_veto_reason_inner: anything unexpected
        reads as "do not set P2P", since losing the tuning beats an exception
        escaping into load_model and failing the load."""
        try:
            return cls._p2p_veto_reason_inner(gpu_indices, launch_order_pinned, ids_are_pci_indices)
        except Exception as e:
            logger.debug(f"peer-fabric check failed: {e}")
            return f"the peer-fabric check could not complete ({type(e).__name__})"

    @classmethod
    def _p2p_veto_reason_inner(
        cls,
        gpu_indices = None,
        launch_order_pinned = False,
        ids_are_pci_indices = None,
    ) -> Optional[str]:
        """Reason GGML_CUDA_P2P must stay off, or None once every pair has a confirmed NV# link. Fails closed; device names and driver peer checks are not proof."""
        if os.environ.get("UNSLOTH_DISABLE_DC_P2P") == "1":
            return "disabled by UNSLOTH_DISABLE_DC_P2P=1"
        if os.environ.get("UNSLOTH_FORCE_DC_P2P") == "1":
            return None
        mig_rows = cls._mig_snapshot_rows()
        if mig_rows is not None:
            # MIG instances have no NVLink matrix row; whole cards keep it via physical_index.
            by_id = {int(r["index"]): r for r in mig_rows}
            selected = (
                sorted(by_id) if gpu_indices is None else sorted(set(int(i) for i in gpu_indices))
            )
            if any(i not in by_id or by_id[i].get("mig") for i in selected):
                return "MIG device UUIDs cannot be verified by the physical-card NVLink matrix"
            gpu_indices = [int(by_id[i]["physical_index"]) for i in selected]
            ids_are_pci_indices = True

        def _pcie(reason: str) -> str:
            """Appended to every PCIe peer veto; on bare-metal Linux a translating IOMMU makes
            copies unsupported."""
            if cls._iommu_is_translating_cached() and not cls._running_virtualized():
                reason += (
                    "; bare-metal Linux with a translating IOMMU, where CUDA "
                    "does not support PCIe peer copies at all"
                )
            return reason

        if not cls._all_selected_gpus_match(cls._NVLINK_FABRIC_GPU_RE, gpu_indices):
            return _pcie("no NVLink-capable part in the selection, so peer copies would cross PCIe")

        matrix = cls._nvlink_topology()
        if not matrix:
            return "neither NVML nor nvidia-smi topo -m gave a usable interconnect matrix"

        gpu_ids = sorted({i for pair in matrix for i in pair})

        # gpu_indices are already nvidia-smi indices (same enumeration as topo -m). Remapping them as
        # CUDA ordinals could make a PCIe pair look NVLinked under FASTEST_FIRST.
        if gpu_indices is not None:
            # Without a PCI_BUS_ID pin the child may resolve these ids to other cards, so refuse,
            # unless the whole box is uniformly NVLinked and the mapping cannot matter.
            uniformly_nvlinked = all(cls._label_is_nvlink(v) for v in matrix.values())
            if not (
                launch_order_pinned
                or os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
                or uniformly_nvlinked
            ):
                return _pcie(
                    "the child's device order is not pinned to PCI_BUS_ID and the "
                    "box is not uniformly NVLinked, so the verified GPUs may not be "
                    "the ones it runs on; set CUDA_DEVICE_ORDER=PCI_BUS_ID to "
                    "enable P2P here"
                )
            # topo -m proves nvidia-smi enumerated the selection; torch ordinals misread it under FASTEST_FIRST.
            pci_ids = (
                cls._GPU_IDS_ARE_PCI_INDICES is True
                if ids_are_pci_indices is None
                else ids_are_pci_indices
            )
            # As above: every mapping agrees on a uniform box, and stray ids still veto.
            if cls._matrix_is_nvml(matrix) and not pci_ids and not uniformly_nvlinked:
                return _pcie(
                    "the GPU selection came from torch rather than nvidia-smi, so "
                    "its ids are CUDA ordinals and cannot be matched against the "
                    "NVML interconnect matrix"
                )
            selected = sorted(set(gpu_indices))
        else:
            # No explicit selection: filter the matrix to the visible mask, but only when the mask maps to
            # PCI indices (numeric entries are CUDA ordinals); otherwise check the whole box.
            visible = cls._resolve_visible_physical_ids()
            mappable = os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
            selected = (
                [i for i in gpu_ids if i in set(visible)]
                if visible is not None and mappable
                else list(gpu_ids)
            )

        if len(selected) < 2:
            return "fewer than two GPUs resolved in the interconnect matrix"

        for a in selected:
            for b in selected:
                if a == b:
                    continue
                label = matrix.get((a, b))
                if label is None:
                    return f"GPU {a} and GPU {b} are absent from the interconnect matrix"
                if not cls._label_is_nvlink(label):
                    return _pcie(f"GPU {a} to GPU {b} is {label}, not NVLink")
        return None

    @staticmethod
    def _effective_gpu_count(gpu_indices = None) -> int:
        """GPUs llama-server will use: len(selection), else the visible CUDA
        device count (None = every visible GPU). 0 on error so multi-GPU tuning
        stays off when the count is unknown."""
        if gpu_indices is not None:
            return len(gpu_indices)
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            return 0
        return 0

    # ggml tests GGML_CUDA_P2P for presence, not value, so these would ENABLE it; only absence is off.
    _FALSY_ENV_VALUES = frozenset({"", "0", "false", "off", "no"})

    # Property of the host, not the model, so warn once rather than per launch.
    _warned_no_nvlink = False

    # The loaded template refuses an empty render. Reset on every load.
    _empty_chat_render_refused = False

    @staticmethod
    def _sanitize_p2p_env(env: dict) -> Optional[str]:
        """Drops an inherited GGML_CUDA_P2P (ggml tests presence, not value); call on every launch path, since consumer GPUs never reach the DC gate."""
        value = env.get("GGML_CUDA_P2P")
        if value is None:
            return None
        if (
            str(value).strip().lower() in LlamaCppBackend._FALSY_ENV_VALUES
            or os.environ.get("UNSLOTH_DISABLE_DC_P2P") == "1"
        ):
            del env["GGML_CUDA_P2P"]
            logger.info(
                "Dropped inherited GGML_CUDA_P2P=%r from the llama-server "
                "environment: llama.cpp tests this variable for presence, not "
                "value, so passing it through would have ENABLED peer copies "
                "(#10613)",
                value,
            )
            return value
        return None

    @staticmethod
    def _p2p_user_opted_out() -> bool:
        """Whether the USER asked for peer copies off. Read from os.environ, not a
        child env _sanitize_p2p_env may already have cleaned, so the intent survives
        the variable and the default cannot put it back."""
        return (
            os.environ.get("UNSLOTH_DISABLE_DC_P2P") == "1"
            or str(os.environ.get("GGML_CUDA_P2P", "1")).strip().lower()
            in LlamaCppBackend._FALSY_ENV_VALUES
        )

    @staticmethod
    def _apply_datacenter_env(
        env: dict,
        gpu_indices = None,
        p2p_opted_out = False,
        launch_order_pinned = False,
        ids_are_pci_indices = None,
    ) -> bool:
        """Injects DC tuning via setdefault (user wins); GGML_CUDA_P2P only on a confirmed NVLink fabric; falsy values removed; UNSLOTH_DISABLE_DC_TUNING=1 opts out."""
        if LlamaCppBackend._sanitize_p2p_env(env) is not None:
            p2p_opted_out = True
        if os.environ.get("UNSLOTH_DISABLE_DC_TUNING") == "1":
            return False
        if not LlamaCppBackend._is_datacenter_gpu(gpu_indices):
            return False

        def _apply(name: str, value: str) -> str:
            supplied = name in env
            env.setdefault(name, value)
            return f"{name}={env[name]}" + (" (user)" if supplied else "")

        applied = [_apply("GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F", "1")]
        if LlamaCppBackend._effective_gpu_count(gpu_indices) > 1:
            # Moves no data between GPUs, so it is not gated on the fabric.
            applied.append(_apply("CUDA_SCALE_LAUNCH_QUEUES", "4x"))
            # User asked for off: skip the probe (up to 10s).
            veto = (
                "GGML_CUDA_P2P was turned off in the environment"
                if p2p_opted_out
                else LlamaCppBackend._p2p_veto_reason(
                    gpu_indices, launch_order_pinned, ids_are_pci_indices
                )
            )
            if veto is None:
                applied.append(_apply("GGML_CUDA_P2P", "1"))
            elif "GGML_CUDA_P2P" in env:
                # A truthy user-supplied GGML_CUDA_P2P stands, but it steers straight into the known P2P failure.
                logger.warning(
                    "GGML_CUDA_P2P is set in the environment (%s), so peer copies stay "
                    "ON despite: %s. Unset it entirely -- not =0, which reads as ON "
                    "upstream -- if model output is garbled (#10613).",
                    env["GGML_CUDA_P2P"],
                    veto,
                )
            elif p2p_opted_out or os.environ.get("UNSLOTH_DISABLE_DC_P2P") == "1":
                # Asked for off: the fabric was never inspected, so do not warn about it.
                logger.info("GGML_CUDA_P2P left unset: %s", veto)
            elif not LlamaCppBackend._warned_no_nvlink:
                LlamaCppBackend._warned_no_nvlink = True
                logger.warning(
                    "Multi-GPU data-center box without a confirmed NVLink fabric: %s. "
                    "Leaving GGML_CUDA_P2P unset. A PCIe peer copy can be discarded "
                    "silently while CUDA still reports success, which surfaces as "
                    "garbled output rather than an error (#10613). Verify with "
                    "scripts/p2p_integrity_probe.py; if it passes, set "
                    "UNSLOTH_FORCE_DC_P2P=1 to opt back in.",
                    veto,
                )
        logger.info(
            "Data-center GPU detected: applied DC llama.cpp env tuning (%s)",
            ", ".join(applied),
        )
        return True

    @staticmethod
    def _visible_devices_mask(env_name: str) -> Optional[set[int]]:
        """Empty mask hides every device, unlike an unset variable (None); blank entries are dropped."""
        raw = os.environ.get(env_name)
        if raw is None:
            return None
        try:
            return set(int(x.strip()) for x in raw.split(",") if x.strip())
        except ValueError:
            if (
                env_name != "CUDA_VISIBLE_DEVICES"
                or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                return None
        from utils.hardware import nvidia

        ids = nvidia.resolve_uuid_mask(raw.strip())
        return set(ids) if ids is not None else None

    @staticmethod
    def _vulkan_pin_args(gpu_indices: Optional[Iterable[int]]) -> list[str]:
        """Pin with --device Vulkan<i>; GGML_VK_VISIBLE_DEVICES uses raw indices, not the compact
        ordinals."""
        if not gpu_indices:
            return []
        return ["--device", ",".join(f"Vulkan{i}" for i in gpu_indices)]
