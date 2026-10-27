// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FitViewOptions, Node } from "@xyflow/react";

export const FIT_VIEW_MAX_ZOOM = 1.1;
export const FIT_VIEW_PADDING = 0.12;
export const FIT_VIEW_DURATION_MS = 340;
/** The run progress island docks top-center over the canvas (about 141 px tall expanded). */
export const RUN_ISLAND_CLEARANCE_PX = 160;

/** Fit padding that keeps nodes out from under the run progress island while it shows. */
export function fitViewPadding(
  runIslandVisible: boolean,
): FitViewOptions["padding"] {
  return runIslandVisible
    ? {
        top: `${RUN_ISLAND_CLEARANCE_PX}px`,
        bottom: FIT_VIEW_PADDING,
        x: FIT_VIEW_PADDING,
      }
    : FIT_VIEW_PADDING;
}

function isMarkdownNoteNode(node: Node): boolean {
  if (node.type !== "builder") {
    return false;
  }
  if (!node.data || typeof node.data !== "object") {
    return false;
  }
  return (node.data as { kind?: string }).kind === "note";
}

function isAuxNode(node: Node): boolean {
  return node.type === "aux";
}

/** Fit targets without notes and aux nodes, falling back to all nodes if none remain. */
export function getFitViewTargetNodes(nodes: Node[]): Node[] {
  const primary = nodes.filter(
    (node) => !(isMarkdownNoteNode(node) || isAuxNode(node)),
  );
  return primary.length > 0 ? primary : nodes;
}

/** All fitView call sites go through this so zoom, padding and filtering stay consistent. */
export function buildFitViewOptions(
  nodes: Node[],
  overrides?: Partial<FitViewOptions>,
): FitViewOptions {
  const targets = getFitViewTargetNodes(nodes);
  return {
    duration: FIT_VIEW_DURATION_MS,
    maxZoom: FIT_VIEW_MAX_ZOOM,
    padding: FIT_VIEW_PADDING,
    nodes: targets.map((n) => ({ id: n.id })),
    ...overrides,
  };
}
