// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext } from "react";
import type { RecipeLoopView } from "../../hooks/use-recipe-loop";

/** The page drives the loop (it owns the recipe runs); canvas cards read it from here. */
export const RecipeLoopContext = createContext<RecipeLoopView | null>(null);

export function useRecipeLoopView(): RecipeLoopView | null {
  return useContext(RecipeLoopContext);
}
