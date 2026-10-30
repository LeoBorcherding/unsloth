# flows-closed-loop pass10 (#15941)

Shot at local merge 6ec62bb767 = flows-closed-loop 0922d92659 + flows-studio-theme 21bddd3f59 (throwaway shots/pass10-flows-closed-loop).
Benchmark: SetFit/emotion, Qwen2.5-0.5B LoRA 60 steps, 10% held out, Recipe held-out reference match on 40 rows: 57.5% (23/40) adapter on, 40% adapter off.
Loop: Qwen3-4B writer, 80 rows, 3 rounds, 20 steps, max_rows 24: test score 83.33, 83.33, 75 on 12 test rows; rows_from_failures 0 each round (about 8.5 min).
Machine: Colab T4 (Tesla T4, 15 GB), Linux, VM fl-t4-nst-flows-shots-99121. Headless Chromium 1440x900 @2x, scaled to 900 wide. Light and dark.
