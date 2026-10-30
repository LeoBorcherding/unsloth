// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InternalNode } from "@xyflow/react";
import { getSpec } from "../stores/catalog-store";
import {
  type DocGraph,
  GROUP_KEY,
  NOTE_KEY,
  useDocStore,
} from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import { CARD_MIN_WIDTH } from "./card-size";
import { inHandle, outHandle } from "./connection-rules";
import type { LayoutRequest, LayoutResult } from "./layout-core";
import { foldLayout, readSuperCards } from "./super-cards";
import {
  type TidyContext,
  fittedCardSize,
  tidyPatch,
  tidyRequest,
} from "./tidy";

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

function runLayout(request: LayoutRequest): Promise<LayoutResult> {
  const id = ++nextId;
  return new Promise<LayoutResult>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, request });
  });
}

function rfElement(id: string): HTMLElement | null {
  return document.querySelector<HTMLElement>(
    `.react-flow__node[data-id="${CSS.escape(id)}"]`,
  );
}

/** How much of an element's content is cut off, the most of it or any descendant. */
function hiddenBelow(root: HTMLElement): number {
  let hidden = root.scrollHeight - root.clientHeight;
  for (const el of root.querySelectorAll<HTMLElement>("*"))
    if (el.clientHeight) hidden = Math.max(hidden, el.scrollHeight - el.clientHeight);
  return hidden;
}

/** Empty space at the bottom of a sized card's output area. */
function slackBelow(grow: HTMLElement): number {
  let bottom = 0;
  for (const child of grow.children)
    bottom = Math.max(
      bottom,
      (child as HTMLElement).offsetTop + (child as HTMLElement).offsetHeight,
    );
  return grow.clientHeight - (bottom - grow.offsetTop);
}

/**
 * Each card's size after fitting: one with output is sized to show it (up to a cap), an oversized
 * one shrinks, and a card with nothing to show goes back to its automatic size. A card in a group
 * never grows out of the frame.
 */
function fitCards(
  graph: DocGraph,
  measured: (id: string) => { width: number; height: number } | undefined,
): Record<string, [number, number] | null> {
  const out: Record<string, [number, number] | null> = {};
  for (const id of graph.nodeOrder) {
    const node = graph.nodes[id];
    const size = measured(id);
    const card = rfElement(id);
    // Zoomed far out a card draws no body, so there is nothing to measure.
    if (node.collapsed || !size || !card?.querySelector("[data-card-body]"))
      continue;
    const grow = card.querySelector<HTMLElement>("[data-card-grow]");
    if (!grow) {
      if (node.size) out[id] = null;
      continue;
    }
    const spec = getSpec(node.type);
    let fit = fittedCardSize(
      size,
      hiddenBelow(grow),
      node.size ? slackBelow(grow) : 0,
      spec ? CARD_MIN_WIDTH : 0,
    );
    if (!fit) continue;
    const frame = node.group ? graph.groups[node.group]?.rect : undefined;
    if (frame) {
      const pad = 12;
      fit = [
        Math.min(fit[0], frame[0] + frame[2] - node.pos[0] - pad),
        Math.min(fit[1], frame[1] + frame[3] - node.pos[1] - pad),
      ];
      if (fit[0] < size.width && fit[1] < size.height && !node.size) continue;
    }
    out[id] = fit;
  }
  return out;
}

/** Tidy up: fit cards to their output, lay out cards and groups, then move notes and put reroutes back. Frames and notes keep their sizes. */
export async function autoLayout(
  getInternalNode: (id: string) => InternalNode | undefined,
  direction: "LR" | "TB" = "LR",
): Promise<void> {
  const state = useDocStore.getState();
  const { graph } = state;
  const measured = (id: string) => {
    const m = getInternalNode(id)?.measured;
    return m?.width && m.height ? { width: m.width, height: m.height } : undefined;
  };
  const sizes = fitCards(graph, measured);
  // A closed super card is laid out as one block, even when its cards sit in a frame.
  const folded = new Set(
    readSuperCards(state.meta.editor, graph)
      .filter((card) => !card.open)
      .flatMap((card) => card.members),
  );
  const view: DocGraph = folded.size
    ? {
        ...graph,
        nodes: Object.fromEntries(
          Object.entries(graph.nodes).map(([id, node]) => [
            id,
            folded.has(id) ? { ...node, group: null } : node,
          ]),
        ),
      }
    : graph;
  const ctx: TidyContext = {
    direction,
    graph: view,
    wireStyle: useUiStore.getState().wireStyle,
    size: (id) => {
      const fit = sizes[id];
      return fit
        ? { width: fit[0], height: fit[1] }
        : (measured(id) ?? { width: 260, height: 140 });
    },
    portY: (id, port, side) => {
      const handle = getInternalNode(id)?.internals.handleBounds?.[
        side === "out" ? "source" : "target"
      ]?.find((h) => h.id === (side === "out" ? outHandle(port) : inHandle(port)));
      return handle ? handle.y + handle.height / 2 : undefined;
    },
  };
  const fold = foldLayout(tidyRequest(ctx), state, (key) =>
    getInternalNode(key)?.measured,
  );
  const result = fold.unfold(await runLayout(fold.request));
  const patch = tidyPatch(ctx, result, sizes);
  // Frames and notes folded into a super card moved with it.
  for (const [key, [x, y]] of Object.entries(result)) {
    if (key.startsWith(GROUP_KEY) && graph.groups[key.slice(GROUP_KEY.length)]) {
      const rect = graph.groups[key.slice(GROUP_KEY.length)].rect;
      patch.groups[key.slice(GROUP_KEY.length)] = [x, y, rect[2], rect[3]];
    } else if (key.startsWith(NOTE_KEY) && graph.notes[key.slice(NOTE_KEY.length)]) {
      const rect = graph.notes[key.slice(NOTE_KEY.length)].rect;
      patch.notes[key.slice(NOTE_KEY.length)] = [x, y, rect[2], rect[3]];
    }
  }
  useDocStore.getState().applyTidy(patch);
}
