# GPT-OSS-20B Training Issue - Discord User "floaty"

## Issue Summary

**Reporter:** floaty (Discord)  
**Date:** April 3-6, 2026  
**Hardware:** RTX 3090 24GB VRAM  
**OS:** Ubuntu 24.04  
**Model:** unsloth/gpt-oss-20b (4-bit QLoRA)  
**Dataset:** HuggingFaceH4/Multilingual-Thinking  

### Problem Description

Training GPT-OSS-20B with QLoRA causes loss to collapse to near-zero (~1e-5) within 50-300 training steps, resulting in token looping during inference.

### Key Symptoms

1. **Abnormally high initial loss:** ~9.874 (expected: ~2.79)
2. **Rapid loss collapse:** Drops to <0.001 within 200-300 steps
3. **Token looping in inference:** Model repeats same words/tokens
4. **High evaluation loss:** Eval loss stays high (~7-13) while training loss drops
5. **Flash Attention 2 warning:** "Your Flash Attention 2 installation seems to be broken. Using Xformers instead."

### Environment Differences

**Floaty's System (BROKEN):**
- Initial loss: **9.874**
- Flash Attention: `Xformers = 0.0.35, FA2 = False` (broken)
- Issue persists across multiple machines
- Torch 2.10.0+cu128, CUDA 8.6, Toolkit 12.8

**Leo's Colab (WORKING):**
- Initial loss: **2.79**
- Flash Attention: `Xformers = None, FA2 = False`
- No loss collapse
- Standard Colab environment

### Attempted Solutions

✅ **Tried:**
1. Removed `train_on_responses_only()` masking (correct - GPT-OSS uses two-channel format)
2. Used official HF dataset directly (ruled out dataset corruption)
3. Removed `reasoning_effort="medium"` parameter (correct per official notebook)
4. Added `bf16=True` to training config (correct for RTX 3090)
5. Tested on multiple machines (issue persists)
6. Uninstalled xformers (issue persists)
7. Currently: Compiling flash-attn from scratch

❌ **Did Not Solve:**
- Learning rate reduction (2e-4 → 2e-5)
- Dataset format changes
- Configuration adjustments

### Current Hypothesis

The **abnormally high initial loss (9.874 vs 2.79)** strongly suggests:

1. **Attention mechanism issue:** Broken FA2 falling back to Xformers may not properly handle GPT-OSS's custom attention pattern
2. **Gradient instability:** High initial loss → unstable gradients → rapid collapse
3. **Environment-specific bug:** Issue is tied to specific CUDA/Torch/FA2 configuration on user's system

### Critical Observation

The initial loss difference (9.874 vs 2.79) is **3.5x higher** than expected, indicating the model is not properly processing inputs from the very first batch. This is **before** any training occurs, ruling out learning rate or training dynamics as root cause.

## Debugging Resources Created

### 1. Colab Notebook
**Location:** `studio/GPT_OSS_20B_Training_Debug_Floaty.ipynb`

**Features:**
- Complete training script matching floaty's configuration
- Enhanced diagnostic logging for every step
- Initial loss tracking and alerts
- Token looping detection
- Environment version checks
- Gradient norm monitoring
- Debugging checklist

**Key Diagnostic Cells:**
- Cell 1: Environment and version diagnostics
- Cell 2: Model loading with config inspection
- Cell 3: LoRA parameter counting
- Cell 4-5: Dataset processing with sample output
- Cell 6: Trainer setup with configuration display
- Cell 7: **Enhanced training loop with initial loss alerts**
- Cell 8: Inference testing with loop detection
- Cell 9: Model saving

### 2. Branch
**Name:** `support/discord-floaty-gpt-oss-training-debug`

## Next Steps for Investigation

### Immediate Actions:
1. **Have floaty run the debug notebook** and share full output
2. **Compare initial loss values** between floaty's run and Leo's Colab
3. **Check if flash-attn compilation from source fixes issue**

### If Issue Persists:

#### Test #1: Force attention implementation
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    max_seq_length=3072,
    load_in_4bit=True,
    attn_implementation="sdpa",  # Force PyTorch SDPA
)
```

#### Test #2: Disable flex attention
```python
import os
os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"
```

#### Test #3: Check model weight initialization
```python
# After model loading, before training
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: mean={param.mean():.6f}, std={param.std():.6f}")
```

#### Test #4: Verify tokenizer output
```python
# Compare tokenizer output between environments
sample = train_dataset[0]["text"]
tokens = tokenizer(sample, return_tensors="pt")
print(f"Input shape: {tokens['input_ids'].shape}")
print(f"First 50 tokens: {tokens['input_ids'][0][:50]}")
```

## Related Code Locations

- unsloth/models/_utils.py L232-233 - GPT-OSS flex attention exclusion
- unsloth/trainer.py L59-60 - Flex attention configuration
- tests/saving/gpt-oss-merge/train_and_merge.py - Official test without train_on_responses_only

## References

- Discord conversation: April 3-6, 2026
- Official GPT-OSS notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb
- Leo's working Colab: https://colab.research.google.com/drive/1fz8r1q8HAnORMl02IKLenrxZDDHRofIZ
- GPT-OSS docs: https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/

## Status

🔴 **ACTIVE** - Awaiting flash-attn compilation results and debug notebook output from floaty

---

**Created:** 2026-04-06  
**Last Updated:** 2026-04-06  
**Assignee:** Leo [UnAI]  
**Priority:** High (user blocked, affects training quality)
