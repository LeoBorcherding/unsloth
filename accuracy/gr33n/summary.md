# GR33N accuracy (2x Radeon AI PRO R9700 32 GB + Radeon iGPU, Windows 11, ROCm)

Branch head `36f7fce0a9`. 20 samples per phase (~3.7 s apart; `Get-Counter` is slow). Monitor = `/api/system?gpu_max_age=1`; truth = `Win32_OperatingSystem`, process working sets over the server's tree, and Windows `GPU Adapter Memory` / `GPU Process Memory` dedicated-usage counters.

## onegpu-chat

| reading | monitor GiB (median) | truth GiB (median) | median diff | max abs diff |
|---|---|---|---|---|
| RAM used | 30.14 | 30.15 | -0.003 | 0.016 |
| Unsloth RAM (tree working set) | 0.62 | 0.61 | +0.012 | 0.020 |

Per card, last sample: monitor GPU 0 = 10.02, GPU 1 = n/a (hidden); Windows adapter counters (largest two) = 10.015;1.749. Unsloth tree on the GPUs (Windows per-process counter) = 10.013, of which llama-server = 9.834; model weights per the monitor = 6.462.

## twogpu-idle

| reading | monitor GiB (median) | truth GiB (median) | median diff | max abs diff |
|---|---|---|---|---|
| RAM used | 31.53 | 31.53 | -0.000 | 0.205 |
| Unsloth RAM (tree working set) | 0.88 | 0.88 | +0.000 | 0.066 |

Per card, last sample: monitor GPU 0 = 0, GPU 1 = 1.77; Windows adapter counters (largest two) = 1.771;0.154. Unsloth tree on the GPUs (Windows per-process counter) = 0.138, of which llama-server = 0; model weights per the monitor = 0.

## twogpu-chat

| reading | monitor GiB (median) | truth GiB (median) | median diff | max abs diff |
|---|---|---|---|---|
| RAM used | 33.59 | 33.59 | -0.012 | 0.135 |
| Unsloth RAM (tree working set) | 2.90 | 2.91 | -0.001 | 0.004 |

Per card, last sample: monitor GPU 0 = 6.62, GPU 1 = 7.76; Windows adapter counters (largest two) = 7.756;6.625. Unsloth tree on the GPUs (Windows per-process counter) = 12.722, of which llama-server = 12.227; model weights per the monitor = 6.462.

## twogpu-chat-train

| reading | monitor GiB (median) | truth GiB (median) | median diff | max abs diff |
|---|---|---|---|---|
| RAM used | 37.93 | 38.02 | -0.003 | 0.180 |
| Unsloth RAM (tree working set) | 5.27 | 5.24 | +0.034 | 0.122 |

Per card, last sample: monitor GPU 0 = 5.49, GPU 1 = 11.81; Windows adapter counters (largest two) = 11.807;5.488. Unsloth tree on the GPUs (Windows per-process counter) = 15.637, of which llama-server = 9.834; model weights per the monitor = 6.462.

## /api/system latency at a 1 s refresh (measured on GR33N itself, 30 polls)

A fresh GPU read on Windows ROCm takes about 1.4 s (the counters behind it are slow), 3.3 s for the first full resolve.

| backend | request | median | slow polls (>500 ms) | max |
|---|---|---|---|---|
| `36f7fce0a9` | `?gpu_max_age=1` | ~1380 ms (every poll probed; from the screenshot HAR, through the SSH tunnel) | 34/34 | 1.5 s |
| `a5f0b24a54` | `?gpu_max_age=1` | 17 ms | 4/30 | 3.3 s (the first, full resolve) |
| `a5f0b24a54` | no age (main's 10 s cache) | 10 ms | 3/30 | 1.4 s |

`a5f0b24a54` spaces fresh reads at least four probe lengths apart, so on this box the GPU rows refresh about every 6 s at the 1 s setting, at the same cost as main's default; on the Mac (a 50 ms probe) they still refresh every second. CSVs: `latency-*.csv`.
