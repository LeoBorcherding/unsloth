# Mint accuracy (RX 9060 XT 16 GB, gfx1200, Linux Mint 22.1, kernel 7.0, in-kernel amdgpu)

Branch head `36f7fce0a9`. torch 2.11.0+rocm7.13.0 (AMD wheels), llama.cpp b11963 ROCm. 80 samples over idle, chat, training, idle-after. Truth: the kernel's amdgpu VRAM counters (what amd-smi reads), `/proc/meminfo`, RSS over the server's process tree, and per-process VRAM from fdinfo.

| metric (80 samples) | median err GiB | max err GiB |
|---|---|---|
| RAM used | 0.002 | 0.039 (memory moved between reads) |
| RAM total | 0.002 | 0.002 |
| Unsloth RAM | 0.000 | 0.020 |
| VRAM used | 0.004 | 0.004 |
| VRAM total | 0.002 | 0.002 |

- The model row (4.16 GiB) matches llama-server's model buffer in its own log (4264 MiB): weights only. KV cache, compute buffers and the vision projector (1.72 GiB) land in "Other in use", as its hint says.
- Training VRAM also lands in "Other in use" (4.97 GiB shown). fdinfo says 4.83 GiB of that is Unsloth's trainer and server, so on Linux per-process VRAM is readable without root: a possible follow-up.
- Training on a 16 GB card unloaded the chat model ("insufficient VRAM"), so chat plus training together was not measured here (it was on GR33N).
- Install note, not this PR: with no rocminfo or amd-smi on the box, `--local` fell back to CPU torch until `UNSLOTH_ROCM_GFX_ARCH=gfx1200` was set.
