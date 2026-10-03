# Recipes: Benchmark card that scores the model a Train card just produced

> **Draft, scope only. No code yet.** This branch holds this file so the PR can exist; it gets deleted when the real code lands.

Follows #7395. Part of the closed-loop pipeline (`recipes-closed-loop`) and builds on #11646 Phase 3.

## Why

A recipe can generate data and (with #7395) train on it, but nothing tells you whether the dataset made the model better without leaving Recipe Studio.

## What it does

- One input handle, wired from a Train card. It defaults to the latest finished run's adapter, or you pick a model by hand.
- Two kinds of benchmark:
  - **Standard:** MMLU, GSM8K, HellaSwag and TruthfulQA via lm-eval, the same harness as #11646 Phase 3, so the two don't grow separate eval stacks.
  - **Held-out recipe split:** score the model on a slice of the generated dataset it never trained on, using a validator or an LLM judge. This is the one that measures what the recipe is for.
- The card shows the score, the delta against the base model, and a link to the full results.
- An output handle carries the score and the failed examples, so a later card (the loop controller, or a regenerate step) can use them.

## Depends on

- #7395 (Train card).
- #11646 Phase 3 for the lm-eval path, and #7703 for logprobs on non-GGUF models.
- The held-out split doesn't need lm-eval, so it can ship first.

## Open questions

- The held-out split has to be carved out before training, so the Train card needs a "hold out N%" option. That's a small change to #7395.
- An LLM judge needs a model loaded while the trained adapter is evaluated, which is a VRAM question on a single GPU.
