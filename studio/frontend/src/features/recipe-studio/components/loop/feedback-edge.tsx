// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  BaseEdge,
  EdgeLabelRenderer,
  type EdgeProps,
  type ReactFlowState,
  useStore,
} from "@xyflow/react";
import { type ReactElement, memo } from "react";

const RADIUS = 14;
const GAP = 56;
const OUT = 32;
// The label sits just past the seed end of the wire, where the loop closes.
const LABEL_OFFSET = 40;

function floorBetween(left: number, right: number, start: number) {
  return (state: ReactFlowState): number => {
    let floor = start;
    for (const node of state.nodeLookup.values()) {
      const x = node.internals.positionAbsolute.x;
      const width = node.measured.width ?? 0;
      if (x + width < left || x > right) {
        continue;
      }
      floor = Math.max(floor, node.internals.positionAbsolute.y + (node.measured.height ?? 0));
    }
    return floor;
  };
}

/**
 * The loop's back edge: out of the Benchmark card, under every card it passes, and up into the
 * failures seed, so the cycle reads as one loop instead of a wire crossing the recipe.
 */
export const RecipeFeedbackEdge = memo(function RecipeFeedbackEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  markerEnd,
  selected,
  data,
}: EdgeProps): ReactElement {
  const active = Boolean((data as { active?: boolean } | undefined)?.active);
  const out = sourceX + OUT;
  const floor = useStore(
    floorBetween(Math.min(targetX, out), Math.max(targetX, out), Math.max(sourceY, targetY)),
  );
  const y = floor + GAP;
  const r = RADIUS;
  const path = [
    `M ${sourceX} ${sourceY}`,
    `L ${out - r} ${sourceY}`,
    `Q ${out} ${sourceY} ${out} ${sourceY + r}`,
    `L ${out} ${y - r}`,
    `Q ${out} ${y} ${out - r} ${y}`,
    `L ${targetX + r} ${y}`,
    `Q ${targetX} ${y} ${targetX} ${y - r}`,
    `L ${targetX} ${targetY}`,
  ].join(" ");

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        style={{
          stroke: "var(--primary)",
          strokeWidth: active || selected ? 2.6 : 2.1,
          strokeDasharray: "8 6",
          opacity: active || selected ? 1 : 0.75,
          animation: active ? "dashdraw 0.6s linear infinite reverse" : undefined,
        }}
      />
      <EdgeLabelRenderer>
        <div
          className="nodrag nopan pointer-events-none absolute rounded-full border border-primary/30 bg-background px-2.5 py-1 text-ui-11 font-medium text-primary shadow-xs"
          style={{
            transform: `translate(0, -50%) translate(${targetX + LABEL_OFFSET}px, ${y}px)`,
          }}
        >
          Failed examples feed the next round
        </div>
      </EdgeLabelRenderer>
    </>
  );
});
