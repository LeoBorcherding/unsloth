Scripts and raw results behind fork PR 779 (MergeKit in Unsloth Studio / Desktop).

- `labonne/`: the three Labonne 7B merges (`labonne_runs.py`, `run_all.sh`), the newer-model runs
  (`newer_runs.py`), the MoE runs (`moe_runs.py`). They expect the branch checked out at
  `/content/unsloth` on a Colab VM with Unsloth Studio installed, and get copied into
  `studio/backend/tests/manual/` before running. Each writes one JSON per merge.
- `newer/`: JSON results for the Qwen3 / SmolLM3 dense merges and the two MoE merges.
- `charts*.py`: rebuild the PNGs in this branch's root with the thick-charts skill.
