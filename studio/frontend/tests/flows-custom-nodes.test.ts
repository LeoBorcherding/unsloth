// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { catalog, doc, edge, node } from "./flows-editor-fixtures.ts";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const api = await import("../src/features/flows/api.ts");
const { useCatalogStore, getSpec } = await import(
  "../src/features/flows/stores/catalog-store.ts"
);
const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const C = await import("../src/features/flows/stores/custom-nodes-store.ts");
const { createNode } = await import("../src/features/flows/editor/node-factory.ts");
const { paramStillValid, portSummary, updateNodesTo } = await import(
  "../src/features/flows/editor/custom-node-actions.ts"
);

import type { CustomNodeEntry, NodeSpec } from "../src/features/flows/types.ts";

function customSpec(type: string, outputs: string[], params = ["times"]): NodeSpec {
  return {
    type,
    v: 1,
    title: "Repeat text",
    category: "text",
    description: "",
    inputs: [{ id: "text", type: "text", label: "Text", optional: false }],
    outputs: [
      ...outputs.map((id) => ({ id, type: "text" as const, label: id })),
      { id: "log", type: "text", label: "Log" },
    ],
    params: params.map((id) => ({ id, kind: "number" as const, default: 2, label: id })),
    lane: "cpu",
    model_kind: null,
    cacheable: false,
    side_effect: true,
    icon: "braces",
    player: false,
    entrypoint: false,
    flow_in: false,
    flow_out: [],
    requires_approval: ["python"],
    handler: "custom.script",
    code_param: "_script",
    custom: { hash: `sha256:${type}` },
  };
}

function entry(id: string, spec: NodeSpec, source: string): CustomNodeEntry {
  return { id, name: spec.title, source, hash: `sha256:${spec.type}`, spec, error: null, createdAt: 0, updatedAt: 0 };
}

const OLD = customSpec("custom.naaaaaaaaaaaaaaaa", ["text", "length"]);
const NEW = customSpec("custom.nbbbbbbbbbbbbbbbb", ["text"], ["times", "shout"]);

test.beforeEach(() => {
  useCatalogStore.getState().setCatalog(catalog);
  C.useCustomNodesStore.setState({ entries: [], loaded: false });
});

test("library specs join the catalog's specs but not its palette groups", () => {
  C.useCustomNodesStore.getState().setEntries([entry("cn_1", OLD, "old source")]);
  assert.equal(getSpec(OLD.type)?.title, "Repeat text");
  assert.ok(!useCatalogStore.getState().catalog?.nodes.some((s) => s.type === OLD.type));
  assert.ok(useCatalogStore.getState().catalog?.categories.some((c) => c.id === "custom"));
});

test("a placed custom node carries its script, library id and hash", () => {
  C.useCustomNodesStore.getState().setEntries([entry("cn_1", OLD, "old source")]);
  const placed = createNode(OLD.type, [0, 0]);
  assert.ok(placed);
  assert.equal(placed.type, OLD.type);
  assert.deepEqual(placed.params, {
    times: 2,
    _script: "old source",
    _script_id: "cn_1",
    _script_hash: `sha256:${OLD.type}`,
  });
  // A spec whose script is not in the library (a flow opened from elsewhere) cannot be placed blank.
  useCatalogStore.getState().addSpecs([NEW]);
  assert.equal(createNode(NEW.type, [0, 0]), null);
});

test("a changed library script is detected and the node updates in place", () => {
  const d = doc(
    [
      node("t", "input.text"),
      node("c", OLD.type, [0, 0], { times: 5, _script: "old", _script_id: "cn_1", _script_hash: "x" }),
      node("o", "output.result"),
    ],
    [edge("e1", "t", "text", "c", "text"), edge("e2", "c", "length", "o", "value")],
  );
  useDocStore.getState().load(d);
  const newer = entry("cn_1", NEW, "new source");
  const placed = useDocStore.getState().graph.nodes.c;
  assert.equal(C.newerLibraryVersion(placed, [newer]), newer);
  assert.equal(C.newerLibraryVersion(placed, [entry("cn_1", OLD, "old")]), null);
  assert.equal(C.newerLibraryVersion(placed, [entry("cn_2", NEW, "x")]), null);

  assert.equal(updateNodesTo(newer), 1);
  const updated = useDocStore.getState().graph.nodes.c;
  assert.equal(updated.type, NEW.type);
  assert.equal(updated.params.times, 5);
  assert.equal(updated.params.shout, 2);
  assert.equal(updated.params._script, "new source");
  // The wire from an output the new version dropped goes; the one into a kept input stays.
  assert.deepEqual(Object.keys(useDocStore.getState().graph.edges), ["e1"]);
});

test("an update drops wires and param values the new version's types rule out", () => {
  const retyped = customSpec("custom.ncccccccccccccccc", ["text"]);
  retyped.inputs = [{ id: "text", type: "number", label: "Text", optional: false }];
  retyped.params = [
    { id: "times", kind: "select", default: "two", label: "times", options: ["one", "two"] },
    { id: "limit", kind: "number", default: 3, label: "limit", min: 0, max: 10 },
  ];
  useDocStore.getState().load(
    doc(
      [
        node("t", "input.text"),
        node("c", OLD.type, [0, 0], { times: 5, limit: 50, _script: "old", _script_id: "cn_1", _script_hash: "x" }),
      ],
      [edge("e1", "t", "text", "c", "text")],
    ),
  );
  assert.equal(updateNodesTo(entry("cn_1", retyped, "retyped source")), 1);
  const updated = useDocStore.getState().graph.nodes.c;
  assert.deepEqual(Object.keys(useDocStore.getState().graph.edges), []);
  assert.equal(updated.params.times, "two");
  assert.equal(updated.params.limit, 3);
  assert.ok(paramStillValid(retyped.params[1], 10) && !paramStillValid(retyped.params[1], 11));
  assert.ok(paramStillValid(retyped.params[0], "one") && !paramStillValid(retyped.params[0], 1));
});

test("opening a flow parses scripts the library does not hold", async () => {
  const calls: string[] = [];
  api.setFlowsTransport(async (url, init) => {
    calls.push(`${init?.method ?? "GET"} ${url}`);
    const body =
      url.endsWith("/custom-nodes/parse")
        ? { spec: NEW, hash: "h" }
        : { nodes: [entry("cn_1", OLD, "old")] };
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
  try {
    await C.hydrateCustomSpecs(
      doc([
        node("a", OLD.type, [0, 0], { _script: "old" }),
        node("b", NEW.type, [0, 0], { _script: "new" }),
      ]),
    );
  } finally {
    api.setFlowsTransport(null);
  }
  assert.deepEqual(calls, ["GET /api/flows/custom-nodes", "POST /api/flows/custom-nodes/parse"]);
  assert.ok(getSpec(OLD.type) && getSpec(NEW.type));
});

test("the port summary reads like the card", () => {
  assert.equal(portSummary(OLD), "text (text) -> text (text), length (text)");
});

test("the palette has the Custom section with upload, template and the sandbox facts", () => {
  const read = (p: string) =>
    readFileSync(new URL(`../src/features/flows/editor/${p}`, import.meta.url), "utf8");
  assert.match(read("palette.tsx"), /<CustomNodesSection/);
  const src = read("custom-nodes.tsx");
  for (const id of ["custom-node-upload", "custom-node-template", "custom-node-libraries", "custom-node-isolation"]) {
    assert.match(src, new RegExp(id));
  }
  assert.match(src, /Uploading never runs the script/);
  assert.match(read("flows-editor-page.tsx"), /await hydrateCustomSpecs\(record\.doc\)/);
});
