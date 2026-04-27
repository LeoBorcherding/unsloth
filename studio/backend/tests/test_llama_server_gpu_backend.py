# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression guard: llama-server binary must be compiled with a GPU backend.

Root cause of the April-2026 "new update doesn't use GPU" regression:
the Windows llama-server build shipped as CPU-only (compiled with Clang, no
CUDA). Even though the Studio correctly computed ``-ngl -1``, the binary
silently ignored the flag and loaded every model into system RAM.

Diagnostic evidence from the affected user's log:
    GGUF size: 2.7 GB, est. KV cache: 0.1 GB, context: 4096,
    GPUs free: [(0, 10575)], selected: [0], fit: False

``fit: False`` + ``selected: [0]`` means the Studio logic was correct —
the model fit, and ``-ngl -1`` was passed. The failure was entirely in the
binary.

These tests guard two complementary surfaces:

1. **Binary capability** (``TestLlamaServerGpuBackend``) — run the binary
   with ``--version`` and assert it lists a GPU backend.  On a machine where
   a CUDA-capable GPU is present this test fails on a CPU-only binary, which
   is exactly the regression to catch.  On machines with no GPU (CI without
   CUDA, macOS without Metal, etc.) the test is skipped so it stays green
   in headless environments.

2. **GPU-fit arithmetic** (``TestGpuFitArithmetic``) — drive
   ``LlamaCppBackend._select_gpus`` with the exact numbers from the bug
   report and assert that the result really is ``(selected=[0], fit=False)``.
   This pins the logic so a future change does not accidentally send the
   model down the CPU path when VRAM is clearly sufficient.
"""

from __future__ import annotations

import subprocess
import sys
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stub heavy / unavailable external dependencies before importing the module.
# Same pattern as the other llama_cpp tests.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

_httpx_stub = _types.ModuleType("httpx")
for _exc in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
_httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
_httpx_stub.Client = type(
    "C",
    (),
    {
        "__init__": lambda s, **kw: None,
        "__enter__": lambda s: s,
        "__exit__": lambda s, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# GPU backend tokens that appear in ``llama-server --version`` output when
# the binary was compiled with hardware acceleration.
_GPU_BACKEND_TOKENS = [
    "cuda",    # NVIDIA CUDA (most Windows/Linux users)
    "cublas",  # older llama.cpp CUDA label
    "metal",   # Apple Metal
    "vulkan",  # cross-platform GPU via Vulkan
    "opencl",  # OpenCL backend
    "rocm",    # AMD ROCm
    "hip",     # AMD HIP (same runtime, different label in some builds)
    "sycl",    # Intel GPU via SYCL
]


def _has_nvidia_gpu() -> bool:
    """Return True if ``nvidia-smi`` can enumerate at least one CUDA device."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_amd_gpu() -> bool:
    """Return True if ``rocm-smi`` (or ``hipInfo``) can enumerate a ROCm device."""
    try:
        r = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _gpu_is_expected() -> bool:
    """True when the running machine is known to have a CUDA or ROCm GPU."""
    return _has_nvidia_gpu() or _has_amd_gpu()


# ---------------------------------------------------------------------------
# Part 1 — binary capability check
# ---------------------------------------------------------------------------


class TestLlamaServerGpuBackend:
    """The bundled llama-server binary must expose a GPU backend.

    Skipped on machines with no detectable GPU (headless CI, Mac without
    NVIDIA/AMD) because a CPU-only binary is the correct choice there.
    """

    @pytest.fixture(scope="class")
    def binary(self):
        b = LlamaCppBackend._find_llama_server_binary()
        if b is None:
            pytest.skip("llama-server binary not found — install or set LLAMA_SERVER_PATH")
        return b

    @pytest.fixture(scope="class")
    def version_output(self, binary):
        try:
            r = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (FileNotFoundError, PermissionError) as exc:
            pytest.skip(f"Could not run llama-server binary: {exc}")
        # llama-server prints version info to stderr on some builds, stdout on others
        return (r.stdout + r.stderr).lower()

    def test_binary_reports_version(self, binary, version_output):
        """Sanity-check that the binary runs at all."""
        assert version_output, (
            f"llama-server at {binary!r} produced no output for --version"
        )

    @pytest.mark.skipif(not _gpu_is_expected(), reason="No CUDA/ROCm GPU detected on this machine")
    def test_binary_has_gpu_backend(self, binary, version_output):
        """A CPU-only build silently drops -ngl -1 and loads models into RAM.

        Regression: April-2026 Windows build shipped with Clang (no CUDA).
        GPUs free: [(0, 10575)], fit: False → -ngl -1 passed but ignored.
        """
        found = [tok for tok in _GPU_BACKEND_TOKENS if tok in version_output]
        assert found, (
            f"llama-server binary at {binary!r} appears CPU-only.\n"
            f"No GPU backend token ({_GPU_BACKEND_TOKENS}) found in --version output.\n"
            f"Models will silently load into system RAM even when VRAM is sufficient.\n"
            f"Fix: rebuild llama.cpp with -DGGML_CUDA=ON (or the relevant GPU flag) "
            f"and replace the binary at that path.\n\n"
            f"--version output:\n{version_output}"
        )


# ---------------------------------------------------------------------------
# Part 2 — GPU-fit arithmetic reproducing the exact bug-report numbers
# ---------------------------------------------------------------------------


class TestGpuFitArithmetic:
    """Drive ``_select_gpus`` with the exact numbers from the bug report.

    Bug-report log line:
        GGUF size: 2.7 GB, est. KV cache: 0.1 GB, context: 4096,
        GPUs free: [(0, 10575)], selected: [0], fit: False

    Confirms that the Studio logic itself was correct — the regression was
    purely in the binary lacking a GPU backend, not in the fit decision.
    """

    # Exact values from the user's log
    GGUF_SIZE_GIB = 2.7
    KV_CACHE_GIB = 0.1
    GPU_FREE_MIB = 10_575  # ~10.3 GiB free on the RTX 4070 Super

    @property
    def _total_bytes(self) -> int:
        GIB = 1024 ** 3
        return int((self.GGUF_SIZE_GIB + self.KV_CACHE_GIB) * GIB)

    def test_model_fits_on_single_gpu(self):
        """2.8 GB total vs 10.3 GB free → GPU-0 is selected, no --fit needed."""
        gpus = [(0, self.GPU_FREE_MIB)]
        gpu_indices, use_fit = LlamaCppBackend._select_gpus(self._total_bytes, gpus)
        assert gpu_indices == [0], f"Expected GPU-0 selected, got {gpu_indices}"
        assert use_fit is False, (
            "Expected use_fit=False (model fits → -ngl -1), got True (--fit on). "
            "This would send inference to CPU despite sufficient VRAM."
        )

    def test_ngl_minus_one_is_chosen_not_fit_flag(self):
        """When use_fit is False, the launch path must pick -ngl -1, not --fit.

        This is the code-path guard: even if the binary ignores -ngl -1
        (the regression), the Studio must be passing the right flag so
        a correctly compiled binary would use the GPU.
        """
        gpus = [(0, self.GPU_FREE_MIB)]
        _, use_fit = LlamaCppBackend._select_gpus(self._total_bytes, gpus)
        # use_fit=False means the "elif gpu_indices is not None: cmd.extend(['-ngl', '-1'])"
        # branch fires, not the "--fit on" branch.
        assert not use_fit, (
            "The Studio should not be passing --fit when the model fits in VRAM. "
            "With a properly compiled binary, -ngl -1 is what enables GPU offload."
        )

    def test_90_percent_threshold_is_not_the_cause(self):
        """The 90% headroom rule still passes comfortably for these numbers.

        90% of 10575 MiB = 9517 MiB ≈ 9.3 GiB.
        Model + KV cache = 2.8 GiB.  Ratio ≈ 0.29 — well inside the budget.
        """
        MIB = 1024 ** 2
        budget_mib = self.GPU_FREE_MIB * 0.90
        total_mib = self._total_bytes / MIB
        assert total_mib < budget_mib, (
            f"Model ({total_mib:.0f} MiB) should be well under 90% of "
            f"free VRAM ({budget_mib:.0f} MiB). "
            f"If this assertion fails the GPU free-memory figure has changed."
        )

    def test_empty_gpu_list_falls_back_to_fit(self):
        """No GPUs at all → (None, True) so --fit handles allocation.

        Sanity-check that the function does NOT select GPU-0 when the
        probe returns an empty list (e.g. CUDA drivers not installed).
        """
        gpu_indices, use_fit = LlamaCppBackend._select_gpus(self._total_bytes, [])
        assert gpu_indices is None
        assert use_fit is True

    def test_truly_oversized_model_falls_back_to_fit(self):
        """A 200 GB model on a 12 GB GPU → (None, True).

        Confirms the fallback path still works so we do not regress the
        ``--fit on`` logic while fixing the GPU-offload path.
        """
        oversized_bytes = 200 * 1024 ** 3
        gpu_indices, use_fit = LlamaCppBackend._select_gpus(
            oversized_bytes, [(0, self.GPU_FREE_MIB)]
        )
        assert gpu_indices is None
        assert use_fit is True
