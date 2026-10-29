// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { listCustomNodes, parseCustomNode } from "../api";
import type { CustomNodeEntry, FlowDocument, FlowNode } from "../types";
import { useCatalogStore } from "./catalog-store";

export const CUSTOM_TYPE_PREFIX = "custom.n";
export const SCRIPT_PARAM = "_script";
export const SCRIPT_ID_PARAM = "_script_id";
export const SCRIPT_HASH_PARAM = "_script_hash";

export function isCustomType(type: string): boolean {
  return type.startsWith(CUSTOM_TYPE_PREFIX);
}

interface CustomNodesState {
  entries: CustomNodeEntry[];
  loaded: boolean;
  setEntries: (entries: CustomNodeEntry[]) => void;
}

export const useCustomNodesStore = create<CustomNodesState>((set) => ({
  entries: [],
  loaded: false,
  setEntries: (entries) => {
    useCatalogStore
      .getState()
      .addSpecs(entries.flatMap((entry) => (entry.spec ? [entry.spec] : [])));
    set({ entries, loaded: true });
  },
}));

export async function loadCustomNodes(): Promise<CustomNodeEntry[]> {
  const { nodes } = await listCustomNodes();
  useCustomNodesStore.getState().setEntries(nodes);
  return nodes;
}

/** The params a new node of a library entry's type starts with: the script travels in the node. */
export function customNodeSeed(type: string): Record<string, unknown> | null {
  const entry = useCustomNodesStore
    .getState()
    .entries.find((e) => e.spec?.type === type);
  if (!entry) return null;
  return {
    [SCRIPT_PARAM]: entry.source,
    [SCRIPT_ID_PARAM]: entry.id,
    [SCRIPT_HASH_PARAM]: entry.hash,
  };
}

/** The library entry a node came from, when the library holds a different version of it. */
export function newerLibraryVersion(
  node: FlowNode,
  entries: CustomNodeEntry[],
): CustomNodeEntry | null {
  if (!isCustomType(node.type)) return null;
  const entry = entries.find((e) => e.id === node.params[SCRIPT_ID_PARAM]);
  if (!entry?.spec || entry.spec.type === node.type) return null;
  return entry;
}

/** Specs for the custom nodes a document embeds, so it opens without the library holding them. */
export async function hydrateCustomSpecs(doc: FlowDocument): Promise<void> {
  try {
    await loadCustomNodes();
  } catch {
    // The library is optional for opening a flow; embedded scripts still parse below.
  }
  const known = useCatalogStore.getState().specs;
  const missing = new Map<string, string>();
  for (const node of doc.nodes) {
    const source = node.params[SCRIPT_PARAM];
    if (isCustomType(node.type) && !known[node.type] && typeof source === "string") {
      missing.set(node.type, source);
    }
  }
  const parsed = await Promise.all(
    [...missing.values()].map((source) =>
      parseCustomNode(source)
        .then((found) => found.spec)
        .catch(() => null),
    ),
  );
  useCatalogStore
    .getState()
    .addSpecs(parsed.flatMap((spec) => (spec ? [spec] : [])));
}
