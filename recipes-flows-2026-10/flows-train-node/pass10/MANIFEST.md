# flows-train-node pass10 (#15611)

Shot at local merge 743e50a647 = flows-train-node 43c7b84a48 + flows-studio-theme a11afb6db6 (throwaway shots/pass10-flows-train-node). BEFORE = flows-base d9a50abf72.
Real runs: SFT unsloth/Qwen3-0.6B, yahma/alpaca-cleaned rows 0-300, 10% held out, 30 steps, loss 1.093, 271 trained / 30 held out, Chat With Checkpoint answered.
GRPO Qwen2.5-0.5B-Instruct on gsm8k, 40 steps, 4 generations, 200 tokens, about 14 min; Try the rewards row 1 scored 2.50/2.5.
Machine: Colab T4 (Tesla T4, 15 GB), Linux, VM fl-t4-nst-flows-shots-99121. Headless Chromium 1440x900 @2x, scaled to 900 wide. Light and dark.
