// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// On a multi-account Desktop the shell answers desktop-login with login_required, so a failed
// request must not clear the page's session and ask the shell again.

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import type * as AuthApi from "../src/features/auth/api.ts";

function fixture(
  t: TestContext,
  opts: {
    loginRequired: boolean;
    pathname: string;
    access: string | null;
    mustChange: boolean;
    api: Response;
  },
) {
  const previousWindow = globalThis.window;
  const previousFetch = globalThis.fetch;
  const location = { pathname: opts.pathname, search: "", href: "" };
  const tokens = { access: opts.access, refresh: null as string | null };
  let change = opts.mustChange;
  let clears = 0;
  let shellAsks = 0;
  globalThis.window = { location } as unknown as Window & typeof globalThis;
  globalThis.fetch = async (path) =>
    String(path).endsWith("/status")
      ? Response.json({ login_mode: "multi", requires_password_change: false })
      : opts.api.clone();
  t.after(async () => {
    for (let i = 0; i < 5; i++) await new Promise<void>((resolve) => setImmediate(resolve));
    globalThis.window = previousWindow;
    globalThis.fetch = previousFetch;
  });
  const api = loadWithStubs<typeof AuthApi>(
    new URL("../src/features/auth/api.ts", import.meta.url),
    {
      "@/lib/api-base": { apiUrl: (path: string) => path, getApiPort: () => null, isTauri: true },
      "@/lib/account-transition": { accountTransitionPending: () => false },
      "./session": {
        getAuthToken: () => tokens.access,
        getRefreshToken: () => tokens.refresh,
        getAuthSessionMark: () => null,
        ensureAuthSessionMark: () => "mark",
        mustChangePassword: () => change,
        setMustChangePassword: (required: boolean) => { change = required; },
        storeAuthTokens: () => undefined,
        clearAuthTokens: () => {
          clears++;
          tokens.access = null;
          tokens.refresh = null;
          change = false;
        },
      },
      "./tauri-auto-auth": {
        isTauriLoginRequired: () => opts.loginRequired,
        tauriAutoAuth: async () => {
          shellAsks++;
          return !opts.loginRequired;
        },
      },
    },
  );
  const settle = async () => {
    for (let i = 0; i < 5; i++) await new Promise<void>((resolve) => setImmediate(resolve));
  };
  return {
    api, location, tokens, settle,
    get clears() { return clears; },
    get shellAsks() { return shellAsks; },
    get change() { return change; },
  };
}

const passwordChangeRequired = () =>
  Response.json({ detail: "Password change required" }, { status: 403 });

test("a setup-code session keeps its token when a poller hits Password change required", async (t) => {
  const f = fixture(t, {
    loginRequired: true,
    pathname: "/change-password",
    access: "alice-setup",
    mustChange: true,
    api: passwordChangeRequired(),
  });
  const response = await f.api.authFetch("/api/hub/active-downloads");
  await f.settle();
  assert.equal(response.status, 403);
  assert.equal(f.tokens.access, "alice-setup");
  assert.equal(f.clears, 0);
  assert.equal(f.shellAsks, 0);
  assert.equal(f.change, true);
  assert.equal(f.location.href, "");
});

test("a setup-code session elsewhere is sent to change its password, not to login", async (t) => {
  const f = fixture(t, {
    loginRequired: true,
    pathname: "/chat",
    access: "alice-setup",
    mustChange: true,
    api: passwordChangeRequired(),
  });
  await f.api.authFetch("/api/hub/active-downloads");
  await f.settle();
  assert.equal(f.tokens.access, "alice-setup");
  assert.equal(f.shellAsks, 0);
  assert.equal(f.location.href, "/change-password");
});

test("a signed-out multi-account window does not clear or ask the shell on each 401", async (t) => {
  const f = fixture(t, {
    loginRequired: true,
    pathname: "/login",
    access: null,
    mustChange: false,
    api: Response.json({ detail: "Not authenticated" }, { status: 401 }),
  });
  for (let i = 0; i < 3; i++) {
    assert.equal((await f.api.authFetch("/api/hub/active-downloads")).status, 401);
  }
  await f.settle();
  // Each clear fires the session event that restarts the download poller at once.
  assert.equal(f.clears, 0);
  assert.equal(f.shellAsks, 0);
  assert.equal(f.location.href, "");
});

test("single-account Desktop still recovers a rejected session through the shell", async (t) => {
  const f = fixture(t, {
    loginRequired: false,
    pathname: "/chat",
    access: "owner-stale",
    mustChange: false,
    api: Response.json({ detail: "expired" }, { status: 401 }),
  });
  await f.api.authFetch("/api/hub/active-downloads");
  await f.settle();
  assert.equal(f.clears, 1);
  assert.equal(f.shellAsks, 1);
});
