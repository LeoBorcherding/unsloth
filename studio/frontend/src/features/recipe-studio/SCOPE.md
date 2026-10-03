# Recipes: super cards that collapse a sub-graph into one card

> **Draft, scope only. No code yet.** This branch holds this file so the PR can exist; it gets deleted when the real code lands.

Part of the closed-loop pipeline (`recipes-closed-loop`).

## Why

A full loop recipe (seed, samplers, LLM columns, validators, train, benchmark) gets too big to read on one canvas. A super card works like a Factorio factory: small on the outside, the whole build on the inside.

Recipe Studio has no grouping today. The node kinds are sampler, tool_config, llm, model_provider, model_config, expression, validator, markdown_note and seed, with no parent or subflow concept, so this is new work.

## What it does

- Select nodes and choose "Combine into super card". The selection collapses into one card. Any edge crossing the selection boundary becomes an input or output handle on that card.
- Double-click to open it and edit the inside. A breadcrumb leads back out.
- The outside of the card shows a summary: what it takes, what it produces, and the status of the last run.
- Save a super card as a reusable block in the Add-a-step sheet ("Generate + validate chat data", "Train + benchmark").
- Round-trips through recipe JSON export and import, and "Ungroup" puts the nodes back unchanged.

## Scope

UI and graph structure only. The payload sent to the backend flattens super cards back into the nodes they contain, so the backend needs no changes.

A super card's handles follow the wire-routing rules (`recipes-wire-routing`): inputs on top/left, outputs on right/bottom.

## Open questions

- React Flow sub-flows (`parentId`) versus a separate nested graph per super card. Sub-flows are less code, but the inside view with a breadcrumb wants its own canvas.
- Nesting: allowed eventually; the first version can stop at one level.
