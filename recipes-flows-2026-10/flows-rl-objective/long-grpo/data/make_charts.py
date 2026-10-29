# Reward curves for the long GRPO runs: python make_charts.py <csv> <label> <out> [commit]
import sys
sys.path.insert(0, "/Users/leoborcherding/.claude/skills/matplotlib-charts")
import pandas as pd
import pro_style as ps
csv, label, out = sys.argv[1:4]
commit = sys.argv[4] if len(sys.argv) > 4 else ""
d = pd.read_csv(csv).dropna(subset=["reward"]).sort_values("step")
w = max(5, len(d) // 20)
roll = d[["reward", "rewards/exact_answer/mean", "rewards/soft_xml_format/mean"]].rolling(w, min_periods=1).mean()
ps.apply()
fig, ax = ps.new(10, 5.6)
ax.plot(d["step"], d["reward"], color=ps.INK_3, lw=0.6, alpha=0.45)
ax.plot(d["step"], roll["reward"], color=ps.CAT[0], lw=2.2, label=f"total reward ({w}-step mean)")
ax.plot(d["step"], roll["rewards/exact_answer/mean"], color=ps.CAT[1], lw=1.6, label="exact-answer (max 2.0)")
ax.plot(d["step"], roll["rewards/soft_xml_format/mean"], color=ps.CAT[2], lw=1.6, label="soft-xml-format (max 0.5)")
ax.axhline(2.5, color=ps.INK_3, lw=0.8, ls="--"); ax.text(d["step"].iloc[0], 2.53, "best possible 2.5", fontsize=9, color=ps.INK_2, va="bottom")
ax.set_ylim(0, 2.75); ax.set_xlabel("training step"); ax.set_ylabel("mean reward per completion")
ps.legend_top(ax)
a, b = roll["reward"].iloc[min(w, len(d) - 1)], roll["reward"].iloc[-1]
verb = "climbs" if b > a + 0.25 else ("falls" if b < a - 0.25 else "stays flat")
ps.finish(fig, f"{label}: GRPO reward {verb}, {a:.2f} to {b:.2f} of 2.5",
          "gsm8k main train, exact-answer + soft-xml-format, 8 completions per prompt, one prompt per step",
          f"Source: /api/train/status metric_history.rl from the Flows Fine-tune node{', stack ' + commit if commit else ''}", out=out)
print(out, len(d), a, b)
