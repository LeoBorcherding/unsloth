# Recipes: route wires like a logic diagram (in from top/left, out from right/bottom, never under a card)

> **Draft, scope only. No code yet.** This branch holds this file so the PR can exist; it gets deleted when the real code lands.

## Why

Recipe Studio wiring gets unreadable once a recipe has more than a few cards. Wires run underneath cards, cross each other for no reason, and enter or leave cards from any side. A recipe should read like a digital logic schematic or a multiplexer: signals flow in one direction, wires are orthogonal and evenly spaced, and you can trace any connection by eye.

## What causes it today

- **Every card has in and out handles on all four sides.** `recipe-graph-node.tsx` registers `dataIn`/`dataOutLeft` on the left, `dataInTop`/`dataOutTop` on top, `dataOut`/`dataInRight` on the right, and `dataOutBottom`/`dataInBottom` on the bottom, plus semantic handles on several sides. `recipe-graph-aux-node.tsx` does the same. A connection can start or end anywhere, so there's no flow direction.
- **Edges ignore the other cards.** `RecipeGraphSemanticEdge` and `DataEdge` use React Flow's `getSmoothStepPath` / `getBezierPath`. Those draw a path between two points and know nothing about obstacles, so a wire goes straight under whatever card sits in its way.
- **Parallel wires overlap.** Several edges from one card share the same path until they split, so you can't tell how many there are.

## The rules this PR enforces

1. **Inputs enter from the top or left. Outputs leave from the right or bottom.** No exceptions, like pins on a mux.
2. **Wires never pass under a card.** They route around a card's bounding box with a small clearance margin.
3. **Wires are orthogonal:** horizontal and vertical segments only, on a grid.
4. **Parallel wires get their own lanes,** spaced apart instead of stacked on top of each other.
5. **Crossings are explicit.** Where two wires have to cross, one draws a small hop arc (the schematic convention), so a crossing never looks like a junction. A fan-out point gets a junction dot.

## Plan

- **Handles:** drop `dataOutLeft`, `dataOutTop`, `dataInRight`, `dataInBottom` and the matching aux and semantic handles. Keep target handles on top/left and source handles on right/bottom.
- **Migration:** saved recipes that use a removed handle id get remapped on import (`utils/import/importer.ts`) and on load, to the nearest legal handle (`dataOutLeft` becomes `dataOut`, `dataInBottom` becomes `dataInTop`, and so on). The edge is kept, never dropped.
- **Router:** a custom edge that computes an obstacle-avoiding orthogonal path from the measured node rects:
  - a grid A* or visibility-graph search over node bounding boxes plus clearance, with a penalty for bends and crossings so it prefers clean paths;
  - lane assignment after routing, so wires that share a corridor are offset;
  - results cached per edge and recomputed only when a node the path touches moves, with a debounce while dragging so the canvas stays smooth.

  A small hand-written router instead of libavoid or ELK: libavoid is a WASM port with an LGPL licence question, and ELK routes well only when it also owns the layout. ELK stays the fallback if the hand-written router can't keep up on big recipes.
- **Auto layout:** the existing dagre auto layout (`layout-controls.tsx`) gets set to left-to-right with rank spacing that leaves room for routing lanes, so "Auto layout" and the router agree.
- **Drawing:** square corners, consistent stroke widths, a hop arc at crossings, and dots at fan-out junctions. The active/selected dash styles from `RecipeGraphSemanticEdge` stay as they are.

## Out of scope

- Payload and backend: edges mean the same thing as before, so nothing sent to the backend changes.
- Super cards (`recipes-super-cards`). Their in/out handles will follow these same rules, which is part of why this lands first.

## Risks and open questions

- **Performance on large recipes.** Routing every edge on every drag frame isn't viable. The plan is to route the dragged node's edges live with a simple path and do the full reroute on drop. This needs measuring on a recipe with 50+ cards.
- **Feedback edges.** The closed loop (benchmark feeding back into regenerate) needs a wire that runs backwards. It leaves from the right or bottom, goes around the outside of the graph and comes back in from the top or left, like a feedback line in a schematic. It must not break rule 1.
- **Card placement users chose by hand.** The router never moves cards. If a route is impossible (a card fully boxed in), it falls back to the shortest orthogonal path and flags the edge instead of hiding it.

## Testing (planned)

- Unit tests for the router: no segment intersects a node rect, every path starts on a right/bottom handle and ends on a top/left one, no two parallel segments overlap.
- Import tests: an old recipe using each removed handle id loads with every edge intact and remapped.
- Before/after screenshots of the same messy recipe, plus a timing number for rerouting a 50-card recipe on drop.
