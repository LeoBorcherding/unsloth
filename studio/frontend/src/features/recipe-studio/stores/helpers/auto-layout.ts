// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge } from "@xyflow/react";
import type { LayoutDirection, NodeConfig, RecipeNode } from "../../types";
import { deriveDisplayGraph } from "../../utils/graph/derive-display-graph";
import { getLayoutedElements } from "../../utils/layout";
import { applyLayoutDirectionToNodes } from "../recipe-studio-helpers";
import {
  centerModelInfraNodes,
  optimizeModelInfraEdgeHandles,
} from "./model-infra-layout";

/** The Auto layout pass: dagre over the data blocks, then model blocks centred on their users. */
export function layoutRecipeGraph(input: {
  nodes: RecipeNode[];
  edges: Edge[];
  configs: Record<string, NodeConfig>;
  layoutDirection: LayoutDirection;
  llmAuxVisibility: Record<string, boolean>;
}): { nodes: RecipeNode[]; edges: Edge[] } {
  const isTopBottom = input.layoutDirection === "TB";
  const displayGraph = deriveDisplayGraph({
    nodes: input.nodes,
    edges: input.edges,
    configs: input.configs,
    layoutDirection: input.layoutDirection,
    auxNodePositions: {},
    llmAuxVisibility: input.llmAuxVisibility,
  });
  const { nodes } = getLayoutedElements(displayGraph.nodes, displayGraph.edges, {
    direction: input.layoutDirection,
    nodesep: isTopBottom ? 120 : 80,
    ranksep: isTopBottom ? 140 : 80,
    configs: input.configs,
  });
  const layoutedPositions = new Map(
    nodes.map((node) => [node.id, node.position] as const),
  );
  const nextNodes = input.nodes.map((node) => {
    const position = layoutedPositions.get(node.id);
    if (!position) {
      return node;
    }
    return { ...node, position };
  });
  const centeredNodes = centerModelInfraNodes(
    nextNodes,
    input.edges,
    input.configs,
    input.layoutDirection,
  );
  return {
    edges: optimizeModelInfraEdgeHandles(
      input.edges,
      centeredNodes,
      input.configs,
      input.layoutDirection,
    ),
    nodes: applyLayoutDirectionToNodes(
      centeredNodes,
      input.configs,
      input.layoutDirection,
    ),
  };
}
