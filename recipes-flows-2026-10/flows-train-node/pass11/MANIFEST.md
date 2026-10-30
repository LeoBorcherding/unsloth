# flows-train-node pass11 (#15611)

Shot at flows-stack-all f6360f9380 (merges every Flows PR; includes flows-studio-theme 8073f6ccd2, so the Train page pickers from #15609 commit 350f4151d4). Not this branch alone.
Machine: M5 MacBook Pro, macOS 26, Unsloth Studio web from `install.sh --local` in its own home, port 8931. Headless Chromium 1440x900 @2x (Advanced at 1440x4400, cut into three columns at section breaks). Light and dark. No training run; flow is openai/gsm8k (main, train) into Fine-tune on unsloth/Qwen3-0.6B.

| file | scene |
|---|---|
| p11-finetune-canvas | Hugging Face Dataset, Fine-tune and Preview cards on the canvas |
| p11-finetune-model-picker | Fine-tune Settings (simple fields) with the Train page model picker closed, then open |
| p11-finetune-advanced | Fine-tune Settings with Advanced open, every Train page setting |
| p11-hf-dataset-picker | Hugging Face Dataset Settings with subset and split dropdowns, then the dataset picker open on a search |
