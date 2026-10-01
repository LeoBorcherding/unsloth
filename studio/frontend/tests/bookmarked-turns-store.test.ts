// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { useBookmarkedTurnsStore } from "../src/features/chat/stores/bookmarked-turns-store.ts";

const store = () => useBookmarkedTurnsStore.getState();

test("deleted chats leave no bookmark records behind", () => {
  useBookmarkedTurnsStore.setState({ bookmarkedByThread: {} });
  store().toggleBookmarkedTurn("t1", "u1");
  store().toggleBookmarkedTurn("t1", "u2");
  store().toggleBookmarkedTurn("t2", "u9");
  store().toggleBookmarkedTurn("t3", "u5");
  store().forgetThreads(["t1", "t3", "never-bookmarked"]);
  assert.deepEqual(store().bookmarkedByThread, { t2: ["u9"] });
});

test("forgetting chats with no bookmarks keeps the state object", () => {
  useBookmarkedTurnsStore.setState({ bookmarkedByThread: { t2: ["u9"] } });
  const before = store().bookmarkedByThread;
  store().forgetThreads(["t1"]);
  assert.equal(store().bookmarkedByThread, before);
});

test("every chat delete route forgets the deleted chats' bookmarks", async () => {
  // A project delete or a clear-all removes chats without going through deleteChatItems.
  const { readSrcAsync } = await import("./helpers/kit.ts");
  for (const file of [
    "features/chat/hooks/use-chat-sidebar-items.ts",
    "features/chat/hooks/use-chat-projects.ts",
    "features/chat/utils/clear-all-chats.ts",
  ]) {
    assert.match(await readSrcAsync(file), /forgetThreads\(/, `${file} leaves bookmarks behind`);
  }
});

test("a project delete forgets the chats the backend deleted, not an earlier listing", async () => {
  // A chat moved into the project after any client-side listing is still deleted with it.
  const { loadWithStubs } = await import("./helpers/module-stubs.ts");
  const forgotten: string[][] = [];
  const projects = loadWithStubs<{
    deleteChatProject: (id: string) => Promise<void>;
  }>(
    new URL("../src/features/chat/hooks/use-chat-projects.ts", import.meta.url),
    {
      react: {},
      "../api/chat-api": { CHAT_PROJECTS_UPDATED_EVENT: "projects-updated" },
      "../stores/bookmarked-turns-store": {
        useBookmarkedTurnsStore: {
          getState: () => ({
            forgetThreads: (ids: string[]) => forgotten.push(ids),
          }),
        },
      },
      "../utils/chat-history-storage": {
        deleteStoredChatProject: async () => ({
          deletedThreadIds: ["listed", "moved-in"],
          sandboxesKept: [],
        }),
        listStoredChatThreads: async (args?: { projectId?: string }) =>
          args?.projectId ? [{ id: "listed" }] : [],
      },
      "../utils/offer-kept-sandbox-files": {
        offerToDeleteKeptSandboxes: () => {},
      },
    },
  );
  await projects.deleteChatProject("p1");
  assert.deepEqual(forgotten.flat().sort(), ["listed", "moved-in"]);
});
