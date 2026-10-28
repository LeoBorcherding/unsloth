// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactElement, useCallback } from "react";
import {
  Panel,
  useReactFlow,
  useUpdateNodeInternals,
} from "@xyflow/react";
import { Button } from "@/components/ui/button";
import { buildFitViewOptions, fitViewPadding } from "../../utils/graph/fit-view";

type LayoutControlsProps = {
  direction: "LR" | "TB";
  onLayout: () => void;
  onToggleDirection: () => void;
  runIslandVisible?: boolean;
};

export function LayoutControls({
  direction,
  onLayout,
  onToggleDirection,
  runIslandVisible = false,
}: LayoutControlsProps): ReactElement {
  const { fitView, getNodes } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();

  const refreshNodeInternals = useCallback(() => {
    const nodeIds = getNodes().map((node) => node.id);
    if (nodeIds.length > 0) {
      updateNodeInternals(nodeIds);
    }
  }, [getNodes, updateNodeInternals]);

  const handleLayout = useCallback(() => {
    onLayout();
    requestAnimationFrame(() => {
      refreshNodeInternals();
      requestAnimationFrame(() => {
        fitView(
          buildFitViewOptions(getNodes(), {
            padding: fitViewPadding(runIslandVisible),
          }),
        );
      });
    });
  }, [fitView, getNodes, onLayout, refreshNodeInternals, runIslandVisible]);

  const handleToggleDirection = useCallback(() => {
    onToggleDirection();
    requestAnimationFrame(() => {
      onLayout();
      requestAnimationFrame(() => {
        refreshNodeInternals();
        requestAnimationFrame(() => {
          fitView(
            buildFitViewOptions(getNodes(), {
              padding: fitViewPadding(runIslandVisible),
            }),
          );
        });
      });
    });
  }, [
    fitView,
    getNodes,
    onLayout,
    onToggleDirection,
    refreshNodeInternals,
    runIslandVisible,
  ]);

  return (
    <Panel position="top-left" className="m-3 flex items-center gap-2">
      <Button size="sm" className="corner-squircle" variant="secondary" onClick={handleLayout}>
        Auto layout
      </Button>
      <Button size="sm" className="corner-squircle" variant="outline" onClick={handleToggleDirection}>
        {direction}
      </Button>
    </Panel>
  );
}
