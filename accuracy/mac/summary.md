# Mac accuracy (Apple M5, 32 GB unified, macOS 26)

Branch head `36f7fce0a9`. 20 samples at 1 s per phase. Monitor = `/api/system?gpu_max_age=1`; truth = `vm_stat` (free+inactive+speculative, the same definition psutil uses), `ps` RSS summed over the server's process tree, `ioreg` "In use system memory". Max |diff| is a single sample where memory moved between the two reads.

## mac-idle

| reading | monitor GiB (median) | truth GiB (median) | median diff | max abs diff |
|---|---|---|---|---|
| RAM used | 19.81 | 19.80 | -0.004 | 1.404 |
| Unsloth (server + children) | 0.25 | 0.25 | +0.000 | 0.000 |
| GPU in use | 0.65 | 0.66 | -0.017 | 1.037 |

## mac-chat

| reading | monitor GiB (median) | truth GiB (median) | median diff | max abs diff |
|---|---|---|---|---|
| RAM used | 27.81 | 27.78 | -0.003 | 0.548 |
| Unsloth (server + children) | 9.38 | 9.38 | +0.000 | 0.017 |
| GPU in use | 10.62 | 10.66 | -0.038 | 0.200 |

Model resident (gemma-4-12b UD-Q4_K_XL): 9.22 GiB; Unsloth's share is floored at that.
