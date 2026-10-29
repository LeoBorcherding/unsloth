// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  SCRIPT_HASH_PARAM,
  SCRIPT_ID_PARAM,
  SCRIPT_PARAM,
  newerLibraryVersion,
} from "../stores/custom-nodes-store";
import { useDocStore } from "../stores/doc-store";
import type { CustomNodeEntry, NodeSpec } from "../types";

/** The node a script turns into, as one line per side: "text (text) -> text (text), length (number)". */
export function portSummary(spec: NodeSpec): string {
  const side = (ports: NodeSpec["inputs"]) =>
    ports
      .filter((p) => !p.param && p.id !== "log")
      .map((p) => `${p.id} (${p.type})`)
      .join(", ") || "nothing";
  return `${side(spec.inputs)} -> ${side(spec.outputs)}`;
}

/** Swap every node of an older version of `entry` in the open flow for the library's, keeping the
 *  param values and the wires that still have a port to land on. */
export function updateNodesTo(entry: CustomNodeEntry): number {
  const spec = entry.spec;
  if (!spec) return 0;
  const store = useDocStore.getState();
  const stale = Object.values(store.graph.nodes).filter(
    (node) => newerLibraryVersion(node, [entry]) === entry,
  );
  const inputs = new Set(spec.inputs.map((p) => p.id));
  const outputs = new Set(spec.outputs.map((p) => p.id));
  const params = new Set(spec.params.map((p) => p.id));
  for (const node of stale) {
    const dropped = Object.values(store.graph.edges)
      .filter(
        (edge) =>
          (edge.target === node.id && !inputs.has(edge.target_port)) ||
          (edge.source === node.id && !outputs.has(edge.source_port)),
      )
      .map((edge) => edge.id);
    if (dropped.length) store.removeElements({ edges: dropped });
    const kept = Object.fromEntries(
      Object.entries(node.params).filter(([key]) => params.has(key)),
    );
    const fresh = Object.fromEntries(
      spec.params
        .filter((p) => !(p.id in kept))
        .map((p) => [p.id, p.default]),
    );
    useDocStore.getState().updateNode(node.id, { type: spec.type, v: spec.v });
    useDocStore.getState().updateParams(node.id, {
      ...kept,
      ...fresh,
      [SCRIPT_PARAM]: entry.source,
      [SCRIPT_ID_PARAM]: entry.id,
      [SCRIPT_HASH_PARAM]: entry.hash,
    });
  }
  return stale.length;
}
