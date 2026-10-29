# Media and logs for unslothai/unsloth#15630 (Live monitor redesign)

Screenshots, accuracy samples and perf traces behind the PR body. Not code; nothing here is meant to merge.

| folder | what |
|---|---|
| `shots/<machine>/` | HD (3x) Playwright captures of the floating Live monitor and Settings > System |
| `accuracy/<machine>/` | the monitor's `/api/system` readings sampled next to each OS's own counters; `summary.md` per machine |
| `perf/` | 30 s idle and drag traces at a 1 s refresh, before and after the perf commits (Chromium CDP, WebKit pointer lag) |
| `scripts/` | the samplers and the shot runner; `PROTOCOL.md` is the procedure |

Machines: Apple M5 Mac (unified memory), GR33N (2x Radeon AI PRO R9700 + iGPU, Windows 11 ROCm), Mint (RX 9060 XT, Linux), Colab L4 (NVIDIA, Linux).
