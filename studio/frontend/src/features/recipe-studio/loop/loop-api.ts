// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type { LoopRecord } from "./types";

function loopsUrl(recipeId: string): string {
  return `/api/data-recipe/recipes/${encodeURIComponent(recipeId)}/loops`;
}

export async function listRecipeLoops(recipeId: string): Promise<LoopRecord[]> {
  const res = await authFetch(loopsUrl(recipeId));
  if (!res.ok) throw new Error(`Load loops failed (${res.status})`);
  const { loops } = (await res.json()) as { loops: LoopRecord[] };
  return loops;
}

async function putLoop(record: LoopRecord): Promise<void> {
  const res = await authFetch(
    `${loopsUrl(record.recipeId)}/${encodeURIComponent(record.id)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(record),
    },
  );
  if (!res.ok) throw new Error(`Save loop failed (${res.status})`);
}

const pending = new Map<string, LoopRecord>();
const draining = new Set<string>();

/** Saves in order; a burst of updates collapses to the newest record. */
export function saveRecipeLoop(record: LoopRecord): void {
  pending.set(record.id, record);
  if (draining.has(record.id)) return;
  draining.add(record.id);
  void (async () => {
    try {
      let next = pending.get(record.id);
      while (next) {
        pending.delete(record.id);
        try {
          await putLoop(next);
        } catch (error) {
          // biome-ignore lint/suspicious/noConsole: background persistence must not stop the loop
          console.error("Save recipe loop failed:", error);
        }
        next = pending.get(record.id);
      }
    } finally {
      draining.delete(record.id);
    }
  })();
}
