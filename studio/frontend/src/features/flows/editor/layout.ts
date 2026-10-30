// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InternalNode } from "@xyflow/react";
import {
  isSetupId,
  setupLayoutNodes,
  withLaidOutSetup,
} from "../recipe/recipe-setup";
import { type DocGraph, useDocStore } from "../stores/doc-store";
import type { LayoutRequest, LayoutResult } from "./layout-core";
import { REROUTE_SIZE } from "./reroutes";
import { foldLayout } from "./super-cards";

let worker: Worker | null = null;
let nextId = 0;
const pending = new Map<
  number,
  { resolve: (r: LayoutResult) => void; reject: (e: Error) => void }
>();

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./layout.worker.ts", import.meta.url), {
    type: "module",
  });
  worker.onmessage = (
    event: MessageEvent<{ id: number; result?: LayoutResult; error?: string }>,
  ) => {
    const entry = pending.get(event.data.id);
    if (!entry) return;
    pending.delete(event.data.id);
    if (event.data.error) entry.reject(new Error(event.data.error));
    else entry.resolve(event.data.result ?? {});
  };
  return worker;
}

/** Nodes and reroutes, joined by the drawn wires, so reroutes are laid out in line and kept. */
export function layoutRequest(
  graph: DocGraph,
  measured: (id: string) => { width?: number; height?: number } | undefined,
  direction: "LR" | "TB",
): LayoutRequest {
  return {
    direction,
    nodes: [
      ...graph.nodeOrder.map((id) => ({
        id,
        width: measured(id)?.width ?? 260,
        height: measured(id)?.height ?? 140,
      })),
      ...graph.rerouteOrder.map((id) => ({ id, ...REROUTE_SIZE })),
    ],
    edges: [
      ...graph.edgeOrder
        .filter((id) => !graph.wires[id])
        .map((id) => ({
          source: graph.edges[id].source,
          target: graph.edges[id].target,
        })),
      ...graph.wireOrder.map((id) => ({
        source: graph.wires[id].source,
        target: graph.wires[id].target,
      })),
    ],
  };
}

export async function autoLayout(
  getInternalNode: (id: string) => InternalNode | undefined,
  direction: "LR" | "TB" = "LR",
): Promise<void> {
  const { graph } = useDocStore.getState();
  const request = layoutRequest(
    graph,
    (id) => getInternalNode(id)?.measured,
    direction,
  );
  // Setup cards take part as nodes, so the layout leaves room for them and saves where they went.
  const setup = setupLayoutNodes(graph, (id) => getInternalNode(id)?.measured);
  request.nodes.push(...setup.nodes);
  request.edges.push(...setup.edges);
  const id = ++nextId;
  const folded = foldLayout(request, useDocStore.getState(), (key) =>
    getInternalNode(key)?.measured,
  );
  const result = await new Promise<LayoutResult>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, request: folded.request });
  });
  const nodes = Object.fromEntries(
    Object.entries(folded.unfold(result)).filter(([key]) => !isSetupId(key)),
  );
  useDocStore.getState().moveNodes(nodes);
  const groups = withLaidOutSetup(useDocStore.getState().graph, result);
  if (groups.length) useDocStore.getState().patchGraph({ groups });
}
