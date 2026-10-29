// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  SCRIPT_HASH_PARAM,
  SCRIPT_ID_PARAM,
  SCRIPT_PARAM,
  newerLibraryVersion,
} from "../stores/custom-nodes-store";
import { getSpec } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import type { CustomNodeEntry, NodeSpec, ParamSpec } from "../types";
import { checkConnection, inHandle, outHandle } from "./connection-rules";

/** The node a script turns into, as one line per side: "text (text) -> text (text), length (number)". */
export function portSummary(spec: NodeSpec): string {
  const side = (ports: NodeSpec["inputs"]) =>
    ports
      .filter((p) => !p.param && p.id !== "log")
      .map((p) => `${p.id} (${p.type})`)
      .join(", ") || "nothing";
  return `${side(spec.inputs)} -> ${side(spec.outputs)}`;
}

/** Whether a kept param value still fits the updated param, so `run()` never gets one its header rules out. */
export function paramStillValid(param: ParamSpec, value: unknown): boolean {
  switch (param.kind) {
    case "text":
    case "textarea":
      return typeof value === "string";
    case "bool":
      return typeof value === "boolean";
    case "select":
      return typeof value === "string" && (param.options ?? []).includes(value);
    case "number":
    case "int":
      return (
        typeof value === "number" &&
        Number.isFinite(value) &&
        (param.kind !== "int" || Number.isInteger(value)) &&
        (param.min === undefined || value >= param.min) &&
        (param.max === undefined || value <= param.max)
      );
    default:
      return false;
  }
}

/** Swap every node of an older version of `entry` in the open flow for the library's, keeping the
 *  param values that still fit and the wires the connection rules still allow. */
export function updateNodesTo(entry: CustomNodeEntry): number {
  const spec = entry.spec;
  if (!spec) return 0;
  const store = useDocStore.getState();
  const stale = Object.values(store.graph.nodes).filter(
    (node) => newerLibraryVersion(node, [entry]) === entry,
  );
  for (const node of stale) {
    const nodes = useDocStore.getState().graph.nodes;
    const specOf = (id: string) => (id === node.id ? spec : getSpec(nodes[id]?.type ?? ""));
    const dropped = Object.values(useDocStore.getState().graph.edges)
      .filter((edge) => edge.source === node.id || edge.target === node.id)
      .filter(
        (edge) =>
          !checkConnection(
            edge.source,
            specOf(edge.source),
            outHandle(edge.source_port),
            edge.target,
            specOf(edge.target),
            inHandle(edge.target_port),
          ).ok,
      )
      .map((edge) => edge.id);
    if (dropped.length) useDocStore.getState().removeElements({ edges: dropped });
    const params = Object.fromEntries(
      spec.params.map((p) => [
        p.id,
        p.id in node.params && paramStillValid(p, node.params[p.id])
          ? node.params[p.id]
          : p.default,
      ]),
    );
    useDocStore.getState().updateNode(node.id, { type: spec.type, v: spec.v });
    useDocStore.getState().updateParams(node.id, {
      ...params,
      [SCRIPT_PARAM]: entry.source,
      [SCRIPT_ID_PARAM]: entry.id,
      [SCRIPT_HASH_PARAM]: entry.hash,
    });
  }
  return stale.length;
}
