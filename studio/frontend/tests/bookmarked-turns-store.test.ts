// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { useBookmarkedTurnsStore } from "../src/features/chat/stores/bookmarked-turns-store.ts";

test("deleting chats drops only their bookmarks", () => {
  const store = useBookmarkedTurnsStore.getState();
  store.toggleBookmarkedTurn("kept", "u1");
  store.toggleBookmarkedTurn("gone", "u1");
  store.toggleBookmarkedTurn("gone", "u2");
  store.clearThreads(["gone", "never-bookmarked"]);
  assert.deepEqual(useBookmarkedTurnsStore.getState().bookmarkedByThread, {
    kept: ["u1"],
  });
});
