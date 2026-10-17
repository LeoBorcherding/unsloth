// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Only a refresh the backend actually rejects ends the session; a 5xx or a dropped
// connection on /api/auth/refresh keeps the tokens and stays on the page.

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import type * as AuthApi from "../src/features/auth/api.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type RefreshReply = number | "network";

function setup(t: TestContext, refreshReply: RefreshReply, isTauri = false) {
  const tokens = {
    access: "ACCESS_X" as string | null,
    refresh: "REFRESH_X" as string | null,
  };
  const location = { pathname: "/chat", href: "/chat" };
  let refreshPosts = 0;
  let autoAuthCalls = 0;
  const previousWindow = globalThis.window;
  const previousFetch = globalThis.fetch;
  globalThis.window = { location } as unknown as Window & typeof globalThis;
  globalThis.fetch = async (input) => {
    const path = String(input);
    if (path.endsWith("/api/auth/refresh")) {
      refreshPosts += 1;
      if (refreshReply === "network") throw new TypeError("Failed to fetch");
      return new Response("{}", { status: refreshReply });
    }
    if (path.endsWith("/api/auth/status")) {
      return Response.json({ requires_password_change: false });
    }
    return new Response("{}", { status: 401 });
  };
  t.after(() => {
    globalThis.window = previousWindow;
    globalThis.fetch = previousFetch;
  });
  const api = loadWithStubs<typeof AuthApi>(
    new URL("../src/features/auth/api.ts", import.meta.url),
    {
      "@/lib/api-base": {
        apiUrl: (path: string) => path,
        getApiPort: () => 8888,
        isTauri,
      },
      "@/lib/account-transition": { accountTransitionPending: () => false },
      "@tauri-apps/api/core": { invoke: async () => false },
      "./tauri-auto-auth": {
        tauriAutoAuth: async () => {
          autoAuthCalls += 1;
          return false;
        },
      },
      "./session": {
        getAuthToken: () => tokens.access,
        getRefreshToken: () => tokens.refresh,
        clearAuthTokens: () => {
          tokens.access = null;
          tokens.refresh = null;
        },
        storeAuthTokens: (access: string, refresh: string) => {
          tokens.access = access;
          tokens.refresh = refresh;
        },
        mustChangePassword: () => false,
        setMustChangePassword: () => {},
      },
    },
  );
  return {
    api,
    tokens,
    location,
    refreshPosts: () => refreshPosts,
    autoAuthCalls: () => autoAuthCalls,
  };
}

const settle = () => new Promise<void>((resolve) => setImmediate(resolve));

for (const reply of [503, 502, "network"] as const) {
  test(`refresh -> ${reply} keeps the session`, async (t) => {
    const env = setup(t, reply);
    assert.equal(await env.api.refreshSession(), false);
    assert.equal(env.tokens.refresh, "REFRESH_X");

    const response = await env.api.authFetch("/api/models");
    await settle();
    assert.equal(response.status, 401);
    assert.deepEqual(env.tokens, { access: "ACCESS_X", refresh: "REFRESH_X" });
    assert.equal(env.location.href, "/chat");
  });
}

test("refresh -> 401 still ends the session and redirects", async (t) => {
  const env = setup(t, 401);
  const response = await env.api.authFetch("/api/models");
  await settle();
  assert.equal(response.status, 401);
  assert.deepEqual(env.tokens, { access: null, refresh: null });
  assert.equal(env.location.href, "/login");
});

test("Unsloth Desktop: a lost refresh is sent once and keeps the session", async (t) => {
  const env = setup(t, "network", true);
  const response = await env.api.authFetch("/api/models");
  assert.equal(response.status, 401);
  assert.equal(env.refreshPosts(), 1);
  assert.equal(env.autoAuthCalls(), 0);
  assert.deepEqual(env.tokens, { access: "ACCESS_X", refresh: "REFRESH_X" });
});
