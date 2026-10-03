# Recipes: data generation workflows that can iterate on a dataset, not just produce one

> **Draft, scope only. No code yet.** This branch holds this file so the PR can exist; it gets deleted when the real code lands.

Part of the closed-loop pipeline (`recipes-closed-loop`).

## Why

Today a recipe runs once and writes a dataset. For the loop, generation has to be something you can steer between runs.

## Proposed

- **Starter templates** for the common shapes (instruction/response, multi-turn chat, tool calls, reasoning traces), each a ready graph you can edit.
- **Sample, inspect, run.** Preview 20 rows, tweak, then commit to the full run. Preview exists today; the gap is making that iteration fast and keeping the full run's config in sync with the preview's.
- **Quality pass before training:** dedupe, length and format checks, and validator pass rates shown as stats on the card, so a bad dataset is caught before it costs a training run.
- **Versioned runs:** each execution keeps its config, seed and row count, so the loop can say which dataset version produced which score.
- **Regenerate from failures** (later): take the failed examples from a Benchmark card and generate more rows aimed at them.

## Order

Templates and the quality pass don't depend on anything and can ship first. Versioned runs are needed by the loop controller. Regenerate-from-failures needs the Benchmark card.
