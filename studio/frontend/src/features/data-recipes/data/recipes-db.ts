// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createEmptyRecipePayload } from "@/features/recipe-studio";
import { normalizeNonEmptyName } from "@/utils";
import { useCallback, useEffect, useState } from "react";
import type { RecipeRecord, SaveRecipeInput } from "../types";
import { importLegacyRecipes } from "./legacy-import";
import { RecipeApiError, recipeRequest } from "./recipes-api";

const recentRecipeCache = new Map<string, RecipeRecord>();
// Start Empty recipes the server hasn't seen yet; the first save creates them.
const draftRecipes = new Map<string, RecipeRecord>();
const listeners = new Set<(recipes: RecipeRecord[]) => void>();
let cachedRecipeList: RecipeRecord[] = [];
let recipeListReady = false;
let recipeListRequest: Promise<RecipeRecord[]> | null = null;
let mutationVersion = 0;

function publishRecipeList(recipes: RecipeRecord[]): RecipeRecord[] {
  cachedRecipeList = [...recipes].sort((a, b) => b.updatedAt - a.updatedAt);
  recipeListReady = true;
  // The list is authoritative: a recipe deleted elsewhere must not reopen from the cache.
  recentRecipeCache.clear();
  for (const recipe of cachedRecipeList) {
    recentRecipeCache.set(recipe.id, recipe);
  }
  for (const listener of listeners) {
    listener(cachedRecipeList);
  }
  return cachedRecipeList;
}

async function fetchRecipeList(): Promise<RecipeRecord[]> {
  await importLegacyRecipes();
  for (;;) {
    // A save or delete that lands while the GET is in flight makes its answer stale.
    const version = mutationVersion;
    const { recipes } = await recipeRequest<{ recipes: RecipeRecord[] }>("");
    if (version === mutationVersion) return publishRecipeList(recipes);
  }
}

export function listRecipes(): Promise<RecipeRecord[]> {
  recipeListRequest ??= fetchRecipeList().finally(() => {
    recipeListRequest = null;
  });
  return recipeListRequest;
}

export function preloadRecipes(): Promise<RecipeRecord[]> {
  return recipeListReady ? Promise.resolve(cachedRecipeList) : listRecipes();
}

export async function getRecipe(id: string): Promise<RecipeRecord | undefined> {
  await importLegacyRecipes();
  try {
    const record = await recipeRequest<RecipeRecord>(
      `/${encodeURIComponent(id)}`,
    );
    recentRecipeCache.set(id, record);
    return record;
  } catch (error) {
    if (error instanceof RecipeApiError && error.status === 404) {
      return undefined;
    }
    throw error;
  }
}

export function getCachedRecipe(id: string): RecipeRecord | null {
  return recentRecipeCache.get(id) ?? draftRecipes.get(id) ?? null;
}

export function primeRecipeCache(record: RecipeRecord): void {
  recentRecipeCache.set(record.id, record);
}

export async function saveRecipe(
  input: SaveRecipeInput,
): Promise<RecipeRecord> {
  const now = Date.now();
  const id = input.id ?? crypto.randomUUID();
  const draft = draftRecipes.get(id);
  const existing = input.id
    ? (recentRecipeCache.get(input.id) ??
      draft ??
      (await getRecipe(input.id)))
    : undefined;
  let record: RecipeRecord;
  try {
    record = await recipeRequest<RecipeRecord>(`/${encodeURIComponent(id)}`, {
      method: "PUT",
      body: JSON.stringify({
        id,
        name: normalizeNonEmptyName(input.name),
        payload: input.payload,
        createdAt: existing?.createdAt ?? now,
        updatedAt: now,
        learningRecipeId: input.learningRecipeId ?? existing?.learningRecipeId,
        learningRecipeTitle:
          input.learningRecipeTitle ?? existing?.learningRecipeTitle,
        // Another window may have saved since this copy was read: the server refuses with 409.
        baseUpdatedAt: draft
          ? undefined
          : (input.baseUpdatedAt ?? existing?.updatedAt),
      }),
    });
  } catch (error) {
    if (error instanceof RecipeApiError && error.status === 409) {
      recentRecipeCache.delete(id);
    }
    throw error;
  }
  mutationVersion += 1;
  draftRecipes.delete(id);
  recentRecipeCache.set(id, record);
  if (recipeListReady) {
    publishRecipeList([
      record,
      ...cachedRecipeList.filter((recipe) => recipe.id !== id),
    ]);
  }
  return record;
}

export async function deleteRecipe(id: string): Promise<void> {
  await recipeRequest<void>(`/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  mutationVersion += 1;
  recentRecipeCache.delete(id);
  if (recipeListReady) {
    publishRecipeList(cachedRecipeList.filter((recipe) => recipe.id !== id));
  }
}

/** `name`, or `name (2)`, `name (3)`... when a listed recipe already has it. */
export function uniqueRecipeName(name: string): string {
  const taken = new Set(cachedRecipeList.map((recipe) => recipe.name));
  if (!taken.has(name)) return name;
  let suffix = 2;
  while (taken.has(`${name} (${suffix})`)) suffix += 1;
  return `${name} (${suffix})`;
}

/** An empty recipe kept client-side until its first edit saves it. */
export function createRecipeDraft(): RecipeRecord {
  const now = Date.now();
  const draft: RecipeRecord = {
    id: crypto.randomUUID(),
    name: uniqueRecipeName("Unnamed"),
    payload: createEmptyRecipePayload(),
    createdAt: now,
    updatedAt: now,
  };
  draftRecipes.set(draft.id, draft);
  return draft;
}

export function createRecipeFromLearningRecipe(input: {
  templateId: string;
  templateTitle: string;
  payload: RecipeRecord["payload"];
}): Promise<RecipeRecord> {
  return saveRecipe({
    name: uniqueRecipeName(input.templateTitle),
    payload: input.payload,
    learningRecipeId: input.templateId,
    learningRecipeTitle: input.templateTitle,
  });
}

export function useRecipes(enabled = true): {
  recipes: RecipeRecord[];
  ready: boolean;
  /** The last list fetch failed; `ready` stays false until one succeeds. */
  error: string | null;
  retry: () => void;
} {
  const [recipes, setRecipes] = useState<RecipeRecord[]>(cachedRecipeList);
  const [ready, setReady] = useState(recipeListReady);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const retry = useCallback(() => {
    setError(null);
    setAttempt((value) => value + 1);
  }, []);

  useEffect(() => {
    if (!enabled) return;
    const listener = (value: RecipeRecord[]) => {
      setRecipes(value);
      setReady(true);
      setError(null);
    };
    listeners.add(listener);
    listRecipes().catch((error) => {
      // biome-ignore lint/suspicious/noConsole: the page keeps its cached list
      console.error("Load data recipes failed:", error);
      setError(error instanceof Error ? error.message : String(error));
    });
    return () => {
      listeners.delete(listener);
    };
  }, [attempt, enabled]);

  return { recipes, ready, error, retry };
}
