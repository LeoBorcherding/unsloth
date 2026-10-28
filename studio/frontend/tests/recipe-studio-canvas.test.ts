// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "@typescript/typescript6";

import { registerBundlerResolver } from "./helpers/kit.ts";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

registerBundlerResolver();

const RECIPE_STUDIO = new URL(
  "../src/features/recipe-studio/",
  import.meta.url,
);

const xyflow = await import("@xyflow/react");
const zustand = await import("zustand");
const dagre = await import("@dagrejs/dagre");

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyRecord = Record<string, any>;

// The real store; only the block registry (which pulls in every dialog) is cut off.
function loadStore(): AnyRecord {
  const mod = loadWithStubs<AnyRecord>(
    new URL("stores/recipe-studio.ts", RECIPE_STUDIO),
    {
      "@xyflow/react": xyflow,
      zustand,
      "@dagrejs/dagre": dagre,
      "../blocks/registry": { getBlockDefinition: () => null },
    },
    { relativePassthrough: true },
  );
  return mod.useRecipeStudioStore;
}

function jsxProps(relative: string, tagName: string): Map<string, ts.Expression | null>[] {
  const path = new URL(relative, RECIPE_STUDIO);
  const source = ts.createSourceFile(
    relative,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );
  const found: Map<string, ts.Expression | null>[] = [];
  const visit = (node: ts.Node) => {
    const tag = openingTag(node);
    if (tag && tag.tagName.getText(source).startsWith(tagName)) {
      const props = new Map<string, ts.Expression | null>();
      for (const attr of tag.attributes.properties) {
        if (ts.isJsxAttribute(attr)) {
          const init = attr.initializer;
          props.set(
            attr.name.getText(source),
            init && ts.isJsxExpression(init) ? (init.expression ?? null) : null,
          );
        }
      }
      found.push(props);
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return found;
}

function hfSeed(): AnyRecord {
  return {
    id: "n1",
    kind: "seed",
    name: "seed",
    seed_source_type: "hf",
    hf_repo_id: "unsloth/alpaca-cleaned",
    hf_split: "train",
    hf_path: "unsloth/alpaca-cleaned/train.parquet",
    seed_columns: ["instruction", "output"],
    seed_preview_rows: [{ instruction: "a", output: "b" }],
    sampling_strategy: "ordered",
    selection_type: "none",
  };
}

function storeWithSeed(): AnyRecord {
  const store = loadStore();
  const seed = hfSeed();
  store.setState({
    configs: { n1: seed },
    nodes: [
      {
        id: "n1",
        type: "builder",
        position: { x: 0, y: 0 },
        data: {},
      },
    ],
  });
  return store;
}

test("Delete and Backspace both remove a selected block", () => {
  const [flow] = jsxProps("recipe-studio-page.tsx", "ReactFlow");
  assert.ok(flow, "the page renders <ReactFlow>");
  assert.equal(flow.get("deleteKeyCode")?.getText(), "CANVAS_DELETE_KEYS");
});

test("each block header has a delete button", () => {
  const buttons = jsxProps("components/recipe-graph-node.tsx", "Button");
  const deleteButton = buttons.find((props) =>
    (props.get("aria-label")?.getText() ?? "").includes("Delete"),
  );
  assert.ok(deleteButton, "a Button labelled Delete <name>");
  assert.match(
    deleteButton.get("onClick")?.getText() ?? "",
    /deleteElements\(/,
    "it removes the node through React Flow, same path as the key",
  );
  // The canvas lock turns off selection, so the keys can't delete; the button mustn't either.
  assert.match(
    deleteButton.get("disabled")?.getText() ?? "",
    /canvasLocked/,
    "the button is off while the canvas is locked",
  );
  const source = readFileSync(
    new URL(
      "../src/features/recipe-studio/components/recipe-graph-node.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    source,
    /const canvasLocked = !useStore\(\s*\(state\) => state\.elementsSelectable/,
    "canvasLocked follows React Flow's elementsSelectable",
  );
});

test("the canvas and fit view can zoom out past 0.5x", async () => {
  const [flow] = jsxProps("recipe-studio-page.tsx", "ReactFlow");
  assert.ok(flow?.has("minZoom"), "<ReactFlow> sets minZoom");
  const { buildFitViewOptions, CANVAS_MIN_ZOOM } = await import(
    "../src/features/recipe-studio/utils/graph/fit-view.ts"
  );
  assert.ok(
    typeof CANVAS_MIN_ZOOM === "number" && CANVAS_MIN_ZOOM < 0.5,
    `CANVAS_MIN_ZOOM is ${CANVAS_MIN_ZOOM}`,
  );
  assert.equal(buildFitViewOptions([]).minZoom, CANVAS_MIN_ZOOM);
});

test("picking the seed type that's already there keeps the configured seed", () => {
  const store = storeWithSeed();
  store.getState().addSeedNode("seed_hf", undefined, true);

  const seed = store.getState().configs.n1;
  assert.equal(seed.hf_path, "unsloth/alpaca-cleaned/train.parquet");
  assert.equal(seed.hf_repo_id, "unsloth/alpaca-cleaned");
  assert.deepEqual(seed.seed_columns, ["instruction", "output"]);
  assert.equal(seed.seed_preview_rows.length, 1);
  assert.equal(store.getState().activeConfigId, "n1", "it opens the seed");
  assert.equal(store.getState().dialogOpen, true);
});

test("switching a configured seed to another type asks first", async () => {
  const { seedSwitchLosesData } = (await import(
    "../src/features/recipe-studio/stores/helpers/seed-replace.ts"
  )) as AnyRecord;
  const seed = hfSeed();
  assert.equal(seedSwitchLosesData(seed, "seed_hf"), false, "same type");
  assert.equal(seedSwitchLosesData(seed, "seed_local"), true, "configured HF to CSV");
  assert.equal(
    seedSwitchLosesData(
      { ...seed, hf_repo_id: "", hf_path: "", seed_columns: [], seed_preview_rows: [] },
      "seed_local",
    ),
    false,
    "an empty seed switches without a prompt",
  );

  // The store still switches when the user confirms.
  const store = storeWithSeed();
  store.getState().addSeedNode("seed_local", undefined, true);
  const switched = store.getState().configs.n1;
  assert.equal(switched.seed_source_type, "local");
  assert.deepEqual(switched.seed_columns, []);
});

function columnsOnlyRecipe(count: number): string {
  const columns = [];
  for (let i = 0; i < count; i += 1) {
    columns.push(
      i === 0
        ? { name: "c0", column_type: "sampler", sampler_type: "uuid", params: {} }
        : { name: `c${i}`, column_type: "expression", expr: `{{ c${i - 1} }}`, dtype: "str" },
    );
  }
  return JSON.stringify({ recipe: { columns } });
}

function overlappingPairs(nodes: AnyRecord[], height: number): number {
  let pairs = 0;
  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      const a = nodes[i];
      const b = nodes[j];
      const aw = a.measured?.width ?? 320;
      const bw = b.measured?.width ?? 320;
      if (
        a.position.x < b.position.x + bw &&
        b.position.x < a.position.x + aw &&
        a.position.y < b.position.y + height &&
        b.position.y < a.position.y + height
      ) {
        pairs += 1;
      }
    }
  }
  return pairs;
}

test("a pasted recipe with no ui positions is laid out once its cards are measured", async () => {
  const { importRecipePayload } = await import(
    "../src/features/recipe-studio/utils/import/index.ts"
  );
  const result = importRecipePayload(columnsOnlyRecipe(17));
  assert.deepEqual(result.errors, []);
  const store = loadStore();
  store.getState().loadRecipe(result.snapshot);

  // What React Flow does once the cards render: real cards are taller than the 140 px step.
  const cardHeight = 260;
  store.setState({
    nodes: store
      .getState()
      .nodes.map((node: AnyRecord) => ({
        ...node,
        measured: { width: 320, height: cardHeight },
      })),
  });
  // What the canvas does on its first fit.
  if (store.getState().autoLayoutPending) {
    store.getState().applyLayout();
  }

  assert.equal(
    overlappingPairs(store.getState().nodes, cardHeight),
    0,
    "no two cards overlap",
  );
  assert.equal(store.getState().autoLayoutPending, false);
});

test("a recipe with stored positions keeps them", async () => {
  const { importRecipePayload } = await import(
    "../src/features/recipe-studio/utils/import/index.ts"
  );
  const text = readFileSync(
    new URL(
      "../src/features/data-recipes/learning-recipes/pdf-grounded-qa.json",
      import.meta.url,
    ),
    "utf8",
  );
  const result = importRecipePayload(text);
  assert.deepEqual(result.errors, []);
  assert.ok(result.snapshot);
  assert.equal(Boolean(result.snapshot.needsLayout), false);
  const store = loadStore();
  store.getState().loadRecipe(result.snapshot);
  assert.equal(Boolean(store.getState().autoLayoutPending), false);
  const before = result.snapshot.nodes.map((node: AnyRecord) => node.position);
  const after = store.getState().nodes.map((node: AnyRecord) => node.position);
  assert.deepEqual(after, before);
});

test("selecting, measuring and drag frames don't rebuild the payload; the drop does", async () => {
  const { payloadNodesChanged } = (await import(
    "../src/features/recipe-studio/utils/payload/payload-nodes.ts"
  )) as AnyRecord;
  const base = [
    { id: "n1", position: { x: 0, y: 0 }, style: { width: 320 }, data: {} },
    { id: "n2", position: { x: 400, y: 0 }, style: { width: 320 }, data: {} },
  ];
  const selected = base.map((node) => ({ ...node, selected: node.id === "n1" }));
  const measured = base.map((node) => ({
    ...node,
    measured: { width: 320, height: 180 },
  }));
  const dragging = [
    { ...base[0], position: { x: 30, y: 40 }, dragging: true },
    base[1],
  ];
  const dropped = [{ ...base[0], position: { x: 30, y: 40 } }, base[1]];

  assert.equal(payloadNodesChanged(base, selected), false);
  assert.equal(payloadNodesChanged(base, measured), false);
  assert.equal(payloadNodesChanged(base, dragging), false);
  assert.equal(payloadNodesChanged(base, dropped), true);
  assert.equal(payloadNodesChanged(base, [base[0]]), true, "a removed block");
});

test("the available references list measures a bounded number of chips", () => {
  const { AvailableReferencesInline } = loadWithStubs<AnyRecord>(
    new URL("components/shared/available-references-inline.tsx", RECIPE_STUDIO),
    {
      "@/components/ui/badge": { Badge: "Badge" },
      react: {
        useLayoutEffect: () => undefined,
        useRef: (current: unknown) => ({ current }),
        useState: (value: unknown) => [value, () => undefined],
      },
      "react/jsx-runtime": stubJsxRuntime(),
    },
  );
  const entries = Array.from({ length: 500 }, (_, i) => ({
    name: `column_${i}`,
    source: "column",
  }));
  const tree = AvailableReferencesInline({ entries }) as StubElement;

  let badges = 0;
  const walk = (node: unknown) => {
    if (Array.isArray(node)) {
      node.forEach(walk);
      return;
    }
    if (!(node && typeof node === "object" && "props" in node)) {
      return;
    }
    const element = node as StubElement;
    if (element.type === "Badge") {
      badges += 1;
    }
    walk(element.props.children);
  };
  walk(tree);

  // 500 nodes each listing 500 columns twice was 500k chips per import.
  assert.ok(badges <= 200, `rendered ${badges} chips for 500 columns`);
});
