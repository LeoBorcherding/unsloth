# Recipes: closed-loop pipeline (generate, train, benchmark, regenerate until a target score)

> **Draft, scope only. No code yet.** This branch holds this file so the PR can exist; it gets deleted when the real code lands.

## Why

Recipe Studio can generate a dataset, and #7395 adds a Train card that fine-tunes on the recipe's output. This PR is the umbrella for the rest of the loop:

```
Generate dataset -> Train model -> Benchmark model -> Regenerate dataset -> Benchmark ...
```

repeated until a target benchmark score is reached, or a run or time budget runs out.

## Pieces

| piece | branch / PR | state |
|---|---|---|
| Train card | #7395 | open |
| Benchmark card: scores the trained adapter inside the recipe | `recipes-benchmark-card` | draft, scope only |
| Data generation workflows: regenerate can act on what the benchmark found | `recipes-datagen-workflows` | draft, scope only |
| Super cards: collapse a sub-graph into one card so the loop stays readable | `recipes-super-cards` | draft, scope only |
| Wire routing: logic-diagram wiring, including the loop's feedback edge | `recipes-wire-routing` | draft, scope only |
| Loop controller | this PR | not started |

## What this PR adds: the loop controller

- A Loop card (or loop settings on a super card) with a target metric, a max iteration count, and stop on plateau.
- A per-iteration history: dataset version, adapter, score. It's stored so it survives a restart, since a loop can run for hours.
- A feedback edge from the Benchmark card back to the generate step. It follows the wire-routing rules (out right/bottom, around the graph, in top/left).

## Open questions

- What does "regenerate" change between iterations: more rows, different seeds, or prompts rewritten from the failures? The first version can just be "more rows with a new seed"; feeding failures back in comes later.
- Where does iteration history live: next to the recipe execution records (`RecipeExecutionRecord`), or a new table?

## Depends on

#7395, then the Benchmark card. Super cards and wire routing aren't required, but without them the loop is hard to read.
