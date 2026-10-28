// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipeNode } from "../../types";
import { readNodeWidth } from "../rf-node-dimensions";

/**
 * Whether `next` changes what the payload reads from nodes (order, id, position,
 * width). Selection and measuring don't, and drag frames wait for the drop.
 */
export function payloadNodesChanged(
  prev: RecipeNode[],
  next: RecipeNode[],
): boolean {
  if (prev === next || next.some((node) => node.dragging)) {
    return false;
  }
  if (prev.length !== next.length) {
    return true;
  }
  return next.some((node, index) => {
    const before = prev[index];
    return (
      before.id !== node.id ||
      before.position.x !== node.position.x ||
      before.position.y !== node.position.y ||
      readNodeWidth(before) !== readNodeWidth(node)
    );
  });
}
