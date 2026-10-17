"""The XPU fast_dequantize / fast_gemv hand FP4 and single-quant (no double quant) weights to
bitsandbytes; their own kernels only decode double-quantized NF4. Runs the XPU branch's functions
on CPU with the bitsandbytes ctypes kernels stubbed out."""

import ast
import contextlib
import textwrap
from pathlib import Path

import pytest
import torch

bnb_functional = pytest.importorskip("bitsandbytes.functional")

import unsloth  # noqa: F401
from unsloth.kernels import utils as kernel_utils


class _KernelReached(Exception):
    pass


def _kernel(*args, **kwargs):
    raise _KernelReached


def _xpu_functions():
    source = Path(kernel_utils.__file__).read_text(encoding = "utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "DEVICE_TYPE == 'xpu' and HAS_XPU_STREAM":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in ("fast_dequantize", "fast_gemv"):
                    yield textwrap.dedent(ast.get_source_segment(source, fn))


@pytest.fixture
def xpu():
    namespace = dict(vars(kernel_utils))
    namespace.update(
        XPU_STREAMS = {None: None, 0: None},
        torch_gpu_device = lambda device: contextlib.nullcontext(),
        cdequantize_blockwise_fp32 = _kernel,
        cdequantize_blockwise_fp16_nf4 = _kernel,
        cdequantize_blockwise_bf16_nf4 = _kernel,
        cdequantize_blockwise_fp32_nf4 = _kernel,
        cgemm_4bit_inference_naive_fp16 = _kernel,
        cgemm_4bit_inference_naive_bf16 = _kernel,
        bnb_functional = bnb_functional,
    )
    sources = list(_xpu_functions())
    assert len(sources) == 2
    for src in sources:
        exec(compile(src, kernel_utils.__file__, "exec"), namespace)
    return namespace


def _quantize(quant_type, double_quant):
    torch.manual_seed(0)
    w = torch.randn(64, 128, dtype = torch.bfloat16)
    return bnb_functional.quantize_4bit(w, quant_type = quant_type, compress_statistics = double_quant)


@pytest.mark.parametrize("quant_type, double_quant", [("fp4", True), ("fp4", False), ("nf4", False)])
def test_dequantize_falls_back_to_bitsandbytes(xpu, quant_type, double_quant):
    W, state = _quantize(quant_type, double_quant)
    torch.testing.assert_close(xpu["fast_dequantize"](W, state), bnb_functional.dequantize_4bit(W, state))


@pytest.mark.parametrize("quant_type, double_quant", [("fp4", True), ("fp4", False), ("nf4", False)])
def test_gemv_falls_back_to_a_dequantized_matmul(xpu, quant_type, double_quant):
    W, state = _quantize(quant_type, double_quant)
    X = torch.randn(1, 1, 128, dtype = torch.bfloat16)
    expected = X @ bnb_functional.dequantize_4bit(W, state).t()
    torch.testing.assert_close(xpu["fast_gemv"](X, W, state), expected)


def test_double_quant_nf4_still_uses_the_xpu_kernels(xpu):
    W, state = _quantize("nf4", True)
    with pytest.raises(_KernelReached):
        xpu["fast_dequantize"](W, state)
    with pytest.raises(_KernelReached):
        xpu["fast_gemv"](torch.randn(1, 1, 128, dtype = torch.bfloat16), W, state)
