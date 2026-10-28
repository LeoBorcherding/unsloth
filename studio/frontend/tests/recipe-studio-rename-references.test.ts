// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const sync = await import(
  "../src/features/recipe-studio/stores/helpers/reference-sync.ts"
);
const { findInvalidJinjaReferences } = await import(
  "../src/features/recipe-studio/utils/refs.ts"
);
const variables = await import("../src/features/recipe-studio/utils/variables.ts");

type NodeConfig = Parameters<typeof sync.applyRenameToConfig>[0];
type Configs = Record<string, NodeConfig>;
type Processor = Parameters<typeof sync.applyRenameToProcessors>[0][number];
type NameEditState = Parameters<typeof sync.applyNameEdit>[0];

function block(id: string, fields: Record<string, unknown>): NodeConfig {
  return { id, ...fields } as unknown as NodeConfig;
}

function column(id: string, name: string): NodeConfig {
  return block(id, { kind: "sampler", sampler_type: "category", name, values: [] });
}

function llm(id: string, name: string, prompt: string, extra = {}): NodeConfig {
  return block(id, {
    kind: "llm",
    llm_type: "text",
    name,
    prompt,
    system_prompt: "",
    model_alias: "",
    ...extra,
  });
}

function byId(...configs: NodeConfig[]): Configs {
  return Object.fromEntries(configs.map((config) => [config.id, config]));
}

// What updateConfig does on each keystroke in a name field.
function typeName(
  configs: Configs,
  id: string,
  keystrokes: string[],
  processors: Processor[] = [],
): NameEditState {
  let state: NameEditState = { configs, processors, renameSources: {} };
  for (const name of keystrokes) {
    const oldName = state.configs[id].name;
    const next = { ...state.configs, [id]: { ...state.configs[id], name } as NodeConfig };
    state = sync.applyNameEdit({ ...state, configs: next }, id, oldName, name);
  }
  return state;
}

function prompt(configs: Configs, id: string): string {
  return (configs[id] as { prompt: string }).prompt;
}

test("clearing a name mid-rename keeps the references and the next name picks them up", () => {
  const configs = byId(
    column("c1", "topic"),
    llm("q", "q", "Write about {{ topic }} and {{ ab }}"),
  );
  const out = typeName(configs, "c1", ["", "s", "su", "subject"]);
  assert.equal(prompt(out.configs, "q"), "Write about {{ subject }} and {{ ab }}");
});

test("clearing a model, tool or provider name doesn't unlink what uses it", () => {
  const configs = byId(
    block("prov", { kind: "model_provider", name: "local" }),
    block("mc", { kind: "model_config", name: "m", provider: "local", model: "x" }),
    block("tp", { kind: "tool_config", name: "calc" }),
    llm("q", "q", "hi", { model_alias: "m", tool_alias: "calc" }),
  );
  let out = typeName(configs, "mc", ["", "m2"]);
  out = typeName(out.configs, "tp", ["", "calc2"]);
  out = typeName(out.configs, "prov", ["", "remote"]);
  assert.equal((out.configs.q as { model_alias: string }).model_alias, "m2");
  assert.equal((out.configs.q as { tool_alias: string }).tool_alias, "calc2");
  assert.equal((out.configs.mc as { provider: string }).provider, "remote");
});

test("typing through another column's name doesn't take over its references", () => {
  const configs = byId(column("a", "a"), column("ab", "ab"), llm("q", "q", "X {{ a }} Y {{ ab }}"));
  const out = typeName(configs, "ab", ["a", "ac"]);
  assert.equal(prompt(out.configs, "q"), "X {{ a }} Y {{ ac }}");
});

test("a rename reaches dotted, filtered and statement references", () => {
  const configs = byId(
    column("t", "topic"),
    llm(
      "q",
      "q",
      `{{ topic | upper }} {% if topic %}y{% endif %} {{topic.title}} {{ ticket.topic }} {{ "topic" }} topic`,
    ),
  );
  const renamed = sync.applyRenameToConfigs(configs, "topic", "subject");
  assert.equal(
    prompt(renamed, "q"),
    `{{ subject | upper }} {% if subject %}y{% endif %} {{subject.title}} {{ ticket.topic }} {{ "topic" }} topic`,
  );
});

test("a rename reaches schema transform processors", () => {
  const configs = byId(column("t", "topic"), llm("q", "q", "{{ topic }}"));
  const processors: Processor[] = [
    {
      id: "p1",
      processor_type: "schema_transform",
      name: "chat",
      template: '{"messages": [{"role": "user", "content": "{{ topic }}"}]}',
    },
  ];
  const out = typeName(configs, "t", ["subject"], processors);
  assert.equal(
    out.processors[0].template,
    '{"messages": [{"role": "user", "content": "{{ subject }}"}]}',
  );
});

test("a dotted reference into an object column isn't flagged", () => {
  assert.deepEqual(
    findInvalidJinjaReferences("{{ customer.first_name }}", ["customer"], ["customer"]),
    [],
  );
  assert.deepEqual(
    findInvalidJinjaReferences("{{ topic.first_name }}", ["topic", "customer"], ["customer"]),
    ["topic.first_name"],
  );
});

test("object columns are person samplers, structured and judge LLMs, validators and seed columns", () => {
  const configs = byId(
    block("p", { kind: "sampler", sampler_type: "person_from_faker", name: "customer" }),
    column("c", "topic"),
    llm("s", "ticket", "", { llm_type: "structured" }),
    llm("j", "score", "", { llm_type: "judge" }),
    block("v", { kind: "validator", name: "check" }),
    llm("t", "answer", ""),
    block("seed", { kind: "seed", name: "seed", seed_columns: ["meta", " "] }),
  );
  assert.deepEqual(
    variables.getNestedReferenceRoots(configs, "t").sort(),
    ["check", "customer", "meta", "score", "ticket"],
  );
});

test("a column named like a filter, test or keyword argument only renames the column", () => {
  const configs = byId(
    column("t", "title"),
    llm(
      "q",
      "q",
      "{{ topic | title }} {{ title }} {% if title is not title %}{{ x | truncate(title=5) }}{% endif %}",
    ),
  );
  const renamed = sync.applyRenameToConfigs(configs, "title", "headline");
  assert.equal(
    prompt(renamed, "q"),
    "{{ topic | title }} {{ headline }} {% if headline is not title %}{{ x | truncate(title=5) }}{% endif %}",
  );
});

test("a filter block's name stays when a column of that name is renamed", () => {
  const configs = byId(column("u", "upper"), llm("q", "q", "{% filter upper %}{{ upper }}{% endfilter %}"));
  const renamed = sync.applyRenameToConfigs(configs, "upper", "shout");
  assert.equal(prompt(renamed, "q"), "{% filter upper %}{{ shout }}{% endfilter %}");
});

test("renaming a note never moves a block's references", () => {
  const configs = byId(
    block("n", { kind: "markdown_note", name: "topic" }),
    column("c", "topic"),
    llm("q", "q", "Write about {{ topic }}"),
  );
  const out = typeName(configs, "n", ["notes"]);
  assert.equal(prompt(out.configs, "q"), "Write about {{ topic }}");
});

test("a block waiting on a taken name catches up when the other block is renamed or removed", () => {
  const configs = byId(column("a", "a"), column("b", "b"), llm("q", "q", "X {{ a }} Y {{ b }}"));
  // a -> b while b is taken: a's references wait on "a".
  const waiting = typeName(configs, "a", ["b"]);
  assert.equal(prompt(waiting.configs, "q"), "X {{ a }} Y {{ b }}");

  const renamed = sync.reconcileRenameSources(
    sync.applyNameEdit(
      { ...waiting, configs: { ...waiting.configs, b: { ...waiting.configs.b, name: "c" } as NodeConfig } },
      "b",
      "b",
      "c",
    ),
  );
  assert.equal(prompt(renamed.configs, "q"), "X {{ b }} Y {{ c }}");
  assert.deepEqual(renamed.renameSources, {});

  const { b: _removed, ...rest } = waiting.configs;
  // What removing a node does: drop the removed block's references, then reconcile.
  const removed = sync.reconcileRenameSources({
    ...waiting,
    configs: sync.applyRemovalToConfigs(rest, "b"),
  });
  assert.match(prompt(removed.configs, "q"), /^X \{\{ b \}\} Y\s*$/);
  assert.deepEqual(removed.renameSources, {});
});

test("a name another block's references still wait on stays taken", () => {
  const configs = byId(column("a", "a"), column("b", "b"), llm("q", "q", "X {{ a }} Y {{ b }}"));
  // a -> b while b is taken, then b is cleared before its new name is typed.
  let state = typeName(configs, "a", ["b"]);
  const cleared = { ...state.configs, b: { ...state.configs.b, name: "" } as NodeConfig };
  state = sync.reconcileRenameSources(sync.applyNameEdit({ ...state, configs: cleared }, "b", "b", ""));
  assert.equal(prompt(state.configs, "q"), "X {{ a }} Y {{ b }}");

  const named = { ...state.configs, b: { ...state.configs.b, name: "c" } as NodeConfig };
  state = sync.reconcileRenameSources(sync.applyNameEdit({ ...state, configs: named }, "b", "", "c"));
  assert.equal(prompt(state.configs, "q"), "X {{ b }} Y {{ c }}");
  assert.deepEqual(state.renameSources, {});
});
