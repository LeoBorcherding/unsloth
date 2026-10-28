import sys, os
sys.path.insert(0, os.path.expanduser("~/.claude/skills/matplotlib-charts"))
import pro_style as ps
ps.apply()

# 1. MTP draft depth, gemma-4-12B-it-qat UD-Q4_K_XL, Colab L4
labels = ["Off", "1", "2", "3", "4", "6", "8"]
vals = [33.4, 52.4, 62.2, 67.2, 72.3, 74.0, 69.5]
fig, ax = ps.new(10, 5.6)
colors = [ps.NEUTRAL["light"]] + [ps.CAT[0]] * 6
colors[labels.index("6")] = ps.CAT[1]
bars = ax.bar(labels, vals, color=colors, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=11)
ax.set_xlabel("MTP draft tokens")
ax.set_ylim(0, 85); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ps.finish(fig, "MTP more than doubles gemma 12B's speed, peaking at 6 draft tokens",
          "Generation tok/s per row of the MTP draft depth sweep, 2 measured runs + 1 warm-up, 128 tokens",
          "Source: Benchmarks page on Unsloth Studio, Colab L4 22.5 GB, llama.cpp b11876, gemma-4-12B-it-qat UD-Q4_K_XL, 2026-10-27",
          out="11808_mtp_depth")

# 2. MoE RAM offload, Qwen3-30B-A3B, Colab L4
x = [0, 12, 24, 36, 48]
q4 = [97.3, 53.4, 35.9, 29.2, 23.3]
q8 = [None, None, 24.2, 19.1, 15.8]
fig, ax = ps.new(10, 5.6)
ax.plot(x, q4, marker="o", color=ps.CAT[0], lw=2.4, label="UD-Q4_K_XL (17.7 GB, fits)")
ax.plot(x[2:], q8[2:], marker="o", color=ps.CAT[1], lw=2.4, label="UD-Q8_K_XL (36 GB, bigger than VRAM)")
ax.scatter([12], [0.8], marker="x", s=90, color=ps.STATUS["critical"], zorder=5)
ax.text(12, 3.5, "Q8 out of memory", ha="center", fontsize=10, color=ps.STATUS["critical"])
for xi, v in zip(x, q4):
    ax.text(xi, v + 3, f"{v:.1f}", ha="center", fontsize=10, color=ps.CAT[0])
for xi, v in zip(x[2:], q8[2:]):
    ax.text(xi, v - 6.5, f"{v:.1f}", ha="center", fontsize=10, color=ps.CAT[1])
ax.set_xticks(x); ax.set_xticklabels(["All on GPU", "12 of 48", "24 of 48", "36 of 48", "All 48"])
ax.set_xlabel("Expert layers on CPU"); ax.set_ylabel("tok/s"); ax.set_ylim(0, 110)
ps.legend_top(ax)
ps.finish(fig, "Experts on CPU let a 36 GB MoE run on a 22.5 GB card, at a speed cost",
          "Generation tok/s per RAM offload row, Qwen3-30B-A3B, 8K context",
          "Source: Benchmarks page on Unsloth Studio, Colab L4 22.5 GB + 53 GB RAM, llama.cpp b11876, 2026-10-27",
          out="11808_moe_offload")

# 3. llama-bench: tab vs direct, Qwen3-4B Q4_K_M, Colab T4
rows = ["llama-bench run directly", "The tab's API", "The tab in the UI"]
pp = [2088.37, 2059.07, 2026.59]; tg = [70.04, 68.87, 67.03]
fig, ax = ps.new(10, 4.6)
import numpy as np
y = np.arange(len(rows)); h = 0.36
pct_pp = [v / pp[0] * 100 for v in pp]; pct_tg = [v / tg[0] * 100 for v in tg]
ax.barh(y - h / 2, pct_pp, h, color=ps.CAT[0], label="pp512")
ax.barh(y + h / 2, pct_tg, h, color=ps.CAT[2], label="tg128")
for yi, p, raw in zip(y, pct_pp, pp):
    ax.text(p + 0.6, yi - h / 2, f"{p:.1f}%  ({raw:,.0f} t/s)", va="center", fontsize=10)
for yi, p, raw in zip(y, pct_tg, tg):
    ax.text(p + 0.6, yi + h / 2, f"{p:.1f}%  ({raw:.1f} t/s)", va="center", fontsize=10)
ax.set_yticks(y); ax.set_yticklabels(rows); ax.invert_yaxis()
ax.set_xlim(0, 125); ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
ps.legend_top(ax)
ps.finish(fig, "The llama-bench tab lands within 5% of llama-bench run by hand",
          "Throughput as a share of the direct run, -p 512 -n 128 -r 5, flash attention auto",
          "Source: Unsloth Studio llama-bench tab, Colab T4 15 GB, llama.cpp b11876, Qwen3-4B Q4_K_M, 2026-10-27",
          out="12522_ui_vs_direct")
ps.contact_sheet(["11808_mtp_depth.png", "11808_moe_offload.png", "12522_ui_vs_direct.png"], "contact.png", cols=1)
