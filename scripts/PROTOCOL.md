# Live monitor accuracy protocol (PR unslothai/unsloth#15630)

Branch: `studio/live-monitor-redesign` on LeoBorcherding/unsloth (head 36f7fce0a9 or later).
Run the branch Unsloth Studio isolated (own studio home, own port). Never touch a stock install.

## What the monitor shows, and where each number comes from
| monitor row | API field |
|---|---|
| CPU % | `/api/system` `cpu.usage_percent` |
| RAM used / total | `memory.total_gb - memory.available_gb` / `memory.total_gb` |
| Unsloth (RAM) | `memory.process_tree_used_mb` (server + child processes RSS); floored at the summed model size on unified memory |
| VRAM / per-GPU rows | `gpu.devices[i].vram_used_gb`, `memory_total_gb` (dedicated cards only; iGPUs excluded) |
| model rows (by model) | `/api/inference/status` `resident_mib` per loaded model |
Call `/api/system?gpu_max_age=1` (what the floating monitor sends at a 1 s refresh).

## Phases
1. idle (nothing loaded)
2. chat model loaded (a GGUF of 4-8 GB)
3. chat model + a training run (Qwen3-1.7B LoRA on unsloth/alpaca-cleaned, max_steps ~600) if memory allows
4. idle again after unload/stop

## Per phase: 20 samples, 1 s apart, each sample records at the same instant
- `/api/system?gpu_max_age=1` JSON (keep raw), `/api/inference/status` JSON (keep raw)
- ground truth:
  - RAM: Linux `/proc/meminfo` (MemTotal - MemAvailable); macOS `vm_stat`/`sysctl hw.memsize`; Windows `Win32_OperatingSystem`
  - per-GPU VRAM: NVIDIA `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits`;
    AMD Linux `amd-smi metric --mem-usage --json` (or `rocm-smi --showmeminfo vram --json`)
  - per-process VRAM: NVIDIA `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`
    (lets you check the "model" and "Other in use" split exactly)
  - process tree RSS: sum of RSS for the server pid and all descendants (`ps -o rss= -p <pids>`)
  - llama-server's own log lines for model/KV/compute buffer sizes per device
- Write `samples.csv` (one row per sample, monitor value next to ground truth, both in GiB).

## Report (`summary.md`)
- machine (GPU, driver, OS, ROCm/CUDA version), branch commit
- per metric: median and max absolute error (GiB or %), and what explains any gap
- what the model row covers vs the per-process truth (weights only? KV cache?)
- HD screenshots (device_scale_factor 3) of the floating monitor in each phase: simple, detailed, share, share+by model.
  `gr33n_shots.py` here does it: `uv run --no-project --with playwright python gr33n_shots.py <base_url> <password_file> <out_dir> <tag>`
