// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipeStudioState } from "../stores/recipe-studio";
import { buildRecipePayload } from "../utils/payload/build-payload";
import type { RecipePayload } from "../utils/payload/types";

/**
 * The recipe as the store holds it right now. A round patches the failures seed and starts the run
 * in the same tick, before React re-renders, so the page's memoized payload would still point at
 * the previous round's seed file.
 */
export function loopRunPayload(state: Pick<RecipeStudioState, "configs" | "nodes" | "edges" | "processors" | "layoutDirection" | "auxNodePositions">): RecipePayload {
  const built = buildRecipePayload(
    state.configs,
    state.nodes,
    state.edges,
    state.processors,
    state.layoutDirection,
    state.auxNodePositions,
  );
  if (built.errors.length > 0) {
    throw new Error(built.errors[0]);
  }
  return built.payload;
}

