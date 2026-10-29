# Live monitor accuracy: NVIDIA (Colab L4)

- Machine: Google Colab L4 VM (`fl-l4-pr-15630-monitor-76833`), NVIDIA L4 22.5 GiB (23034 MiB), driver 580.82.07, CUDA 13.0 (nvcc 13.0.88), torch 2.11.0+cu130, Linux 6.6.159 / glibc 2.39, 12 vCPU, 53 GiB RAM, no cgroup memory cap (`memory.max` = max).
- Commit: `36f7fce0a907170a6402ed81a91d03293ea86292` (`studio/live-monitor-redesign`, the VM's checkout).
- Phases (20 samples each, 1 s apart, `/api/system?gpu_max_age=1` + `/api/inference/status` and ground truth read in the same loop on the VM):
  1. idle
  2. `unsloth/Qwen3-8B-GGUF` Q4_K_M loaded, ctx 8192
  3. same + Qwen3-1.7B LoRA training (bf16, bs 2, sampled from step 11 on)
  4. idle after `/api/train/stop` + `/api/inference/unload`
- Ground truth: `nvidia-smi --query-gpu` and `--query-compute-apps`, `/proc/meminfo` (MemTotal - MemAvailable), summed RSS of the server pid's process tree, llama-server's startup log.

## Errors (monitor minus truth, GiB unless noted, all 80 samples)

| metric | median abs | max abs | notes |
|---|---|---|---|
| RAM used | 0.005 | 0.021 | sampling skew only |
| RAM total | 0.002 | 0.002 | rounding to 2 dp |
| Unsloth RAM (process tree) | 0.000 | 0.004 | exact; includes the training subprocess and cloudflared |
| VRAM used | 0.004 | 0.004 | rounding; idle reads 0.00 vs 3 MiB |
| VRAM total | 0.004 | 0.004 | 22.49 vs 22.494 |
| Model row vs llama-server weights (log) | 0.000 | 0.000 | 4455 MiB, exact |
| Model row vs llama-server process VRAM | 1.526 | 1.526 | by design: weights only, see below |
| Other in use vs (used - weights) | 0.003 | 0.004 | exact by construction |
| CPU % (vs psutil in the sampler) | 0.2 pt | 8.9 pt | different sampling windows; spikes, not bias |

Per-phase numbers are in `logs/stats.txt`; every sample is in `samples.csv`.

## Model / Unsloth split vs per-process truth

In phases 2 and 3, llama-server (pid 8593) holds **6018 MiB** per `nvidia-smi --query-compute-apps`. Its log:

| buffer | MiB |
|---|---|
| CUDA0 model (weights) | 4455.34 |
| CUDA0 KV cache (8192 ctx, 4 slots, unified) | 1152.00 |
| CUDA0 compute | 190.71 |
| CUDA context + rest (6018 - above) | ~220 |

The model row reads **4.35 GiB = 4455 MiB**: exactly the weights buffer (`reclaimable_gpu_memory_gb` parses the `model buffer size` lines, `backend/core/inference/llama_cpp.py:8366`). The other 1.49 GiB of llama-server (KV 1.13, compute 0.19, context ~0.21) lands in "Other in use", as the hint says.

Phase 3: "Other in use" 5.86 GiB vs truth 5.855 = llama-server non-weights 1.53 + training process (pid 16659) 4.06 + the Studio server's own CUDA context (pid 916) 0.25 + ~0.02 driver. The training run's VRAM is in "Other in use", in full.

Phase 4: "Other in use" 0.26 GiB is the Studio server process itself (pid 916, 254 MiB of CUDA context kept after training). Nothing else is on the GPU.

The RAM split is exact: Unsloth 5.84 GiB in phase 3 = server 2.0 + training subprocess 3.0 + llama-server 0.76 + cloudflared and the pip resource tracker.

## Verdict

No wrong number. Every monitor field matches ground truth to rounding (max 0.021 GiB, which is skew between two reads). The model row is weights only by design; here the unattributed part of llama-server is 1.49 GiB (25% of the process), 1.13 GiB of it KV cache.

Copy nit, not a numbers bug: the "Other in use" hint says the driver does not report VRAM per app (`frontend/src/i18n/locales/en.ts:2459`). On NVIDIA it does (`--query-compute-apps` / NVML), which is how this check was done. It holds for ROCm on Windows and Apple, not NVIDIA. Unsloth's own server CUDA context (0.18 to 0.25 GiB) also sits in "Other in use", which the hint doesn't mention.

## Files

- Screenshots (device_scale_factor 3): `1-idle-*`, `2-model-*`, `3-model-train-*`, `4-idle-after-*` `.png` here (1-simple, 2-detailed, 3-unsloth-share, 4-share-and-per-model, 5-everything, 6-full-page, 7/8 settings), with the monitor's text in matching `.txt`.
- The HARs here have the password redacted; they still carry JWTs for a VM that no longer exists.
- `logs/`: raw per-sample JSONL (system + inference + nvidia-smi + meminfo + process tree), llama-server log, Studio server log, serve/boot logs, the sampler and analysis scripts.
- Full VM export: `COLAB LOGS EXPORT/pr-15630/fl-l4-pr-15630-monitor-76833/20261029-052023-final/`.

Caveat: the first training run (600 steps) finished while a `colab exec` call hung, so phase 3 was taken on a second run (max_steps 3000, stopped with `/api/train/stop` after sampling). VM time about 58 min.
