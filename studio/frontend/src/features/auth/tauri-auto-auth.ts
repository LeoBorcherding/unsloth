// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import {
  clearAuthTokens,
  hasAuthToken,
  hasRefreshToken,
  mustChangePassword,
  setMustChangePassword,
  storeAuthTokens,
} from "./session";
import { refreshSession, refreshSessionOutcome } from "./api";
import { setLoginMode } from "./login-client";
import {
  OWNER_BROWSER_ACCOUNT,
  transitionBrowserAccount,
} from "@/lib/account-transition";

type DesktopAuthResponse =
  | { access_token: string; refresh_token: string }
  | { login_required: true; login_mode: "multi" };

type TauriAutoAuthOptions = {
  force?: boolean;
};

// Route guards can call this concurrently; dedupe so first-launch password change cannot race.
let pending: { promise: Promise<boolean>; force: boolean } | null = null;
let lastTauriAuthFailure: string | null = null;
let tauriLoginRequired = false;

export function isTauriLoginRequired(): boolean {
  return tauriLoginRequired;
}

// Set by an explicit Desktop log out so the shell secret does not sign the owner straight back in.
// "unsloth"-prefixed, so an account switch's localStorage purge also clears it.
export const DESKTOP_SIGNED_OUT_KEY = "unsloth.desktop-signed-out.v1";

export function isDesktopSignedOut(): boolean {
  try {
    return window.localStorage.getItem(DESKTOP_SIGNED_OUT_KEY) === "1";
  } catch {
    return false;
  }
}

export function markDesktopSignedOut(): void {
  tauriLoginRequired = true;
  try {
    window.localStorage.setItem(DESKTOP_SIGNED_OUT_KEY, "1");
  } catch {
    // Without storage the in-memory flag still holds for this window.
  }
}

export function clearDesktopSignedOut(): void {
  try {
    window.localStorage.removeItem(DESKTOP_SIGNED_OUT_KEY);
  } catch {
    // Nothing stored to clear.
  }
}

const TAURI_AUTH_FAILURE_FALLBACK =
  "Desktop authentication failed. Update or repair the managed Unsloth install, then restart Unsloth.";
const BACKEND_NOT_READY_MESSAGE = "Backend is not ready";

function authFailureMessage(error: unknown): string {
  if (typeof error === "string" && error) return error;
  if (error instanceof Error && error.message) return error.message;
  return TAURI_AUTH_FAILURE_FALLBACK;
}

export function getTauriAuthFailure(): string | null {
  return lastTauriAuthFailure;
}

export function clearTauriAuthFailure(): void {
  lastTauriAuthFailure = null;
}

function setTauriAuthFailure(error: unknown): void {
  lastTauriAuthFailure = authFailureMessage(error);
  window.dispatchEvent(
    new CustomEvent("tauri-auth-failed", { detail: lastTauriAuthFailure }),
  );
}

function isBackendNotReady(error: unknown): boolean {
  return authFailureMessage(error).includes(BACKEND_NOT_READY_MESSAGE);
}

async function doTauriAutoAuth(options: TauriAutoAuthOptions): Promise<boolean> {
  // Desktop must handle password-change state internally in Rust.
  if (!options.force && hasAuthToken() && !mustChangePassword()) {
    clearTauriAuthFailure();
    return true;
  }

  if (!options.force && hasRefreshToken()) {
    const refreshed = await refreshSession();
    if (refreshed && hasAuthToken() && !mustChangePassword()) {
      clearTauriAuthFailure();
      return true;
    }
  }

  if (isDesktopSignedOut()) {
    tauriLoginRequired = true;
    clearTauriAuthFailure();
    const { router } = await import("@/app/router");
    await router.navigate({ to: "/login", replace: true });
    return options.force === true;
  }

  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const tokens = await invoke<DesktopAuthResponse>("desktop_auth");
    if ("login_required" in tokens) {
      tauriLoginRequired = true;
      // A managed account signs in in the document; do not discard the session it holds.
      const outcome = hasRefreshToken()
        ? await refreshSessionOutcome()
        : "rejected";
      if (outcome === "ok" && hasAuthToken()) {
        clearTauriAuthFailure();
        if (mustChangePassword()) {
          const { router } = await import("@/app/router");
          await router.navigate({ to: "/change-password", replace: true });
        }
        return true;
      }
      // Keep the session; the forced startup probe still releases the startup screen.
      if (outcome === "unavailable") {
        clearTauriAuthFailure();
        return options.force === true;
      }
      clearAuthTokens();
      clearTauriAuthFailure();
      // The forced startup probe must release the startup screen so the login form can mount; ordinary API recovery still returns false.
      const { router } = await import("@/app/router");
      await router.navigate({ to: "/login", replace: true });
      return options.force === true;
    }
    tauriLoginRequired = false;
    // Owner tokens after the last managed account is deleted must not inherit its browser data.
    await transitionBrowserAccount(OWNER_BROWSER_ACCOUNT, "/chat", () => {
      storeAuthTokens(tokens.access_token, tokens.refresh_token);
      setMustChangePassword(false);
      setLoginMode("single");
    });
    clearTauriAuthFailure();
    return true;
  } catch (error) {
    if (isBackendNotReady(error)) return false;
    setTauriAuthFailure(error);
    return false;
  }
}

/** Silently authenticate in Tauri desktop mode; also true when only a forced startup probe verified the shell. */
export function tauriAutoAuth(
  options: TauriAutoAuthOptions = {},
): Promise<boolean> {
  if (!isTauri) return Promise.resolve(false);
  const force = options.force === true;
  if (!pending || (force && !pending.force)) {
    let promise: Promise<boolean>;
    promise = doTauriAutoAuth({ force }).finally(() => {
      if (pending?.promise === promise) pending = null;
    });
    pending = { promise, force };
  }
  if (!force && pending.force) {
    // An API retry may share the forced probe, but shell readiness alone must not authorize it.
    return pending.promise.then(
      (ready) => ready && (!tauriLoginRequired || hasAuthToken()),
    );
  }
  return pending.promise;
}

/** "Continue as owner" on the Desktop login page: the shell secret signs the owner in, single mode only. */
export async function continueAsDesktopOwner(): Promise<boolean> {
  clearDesktopSignedOut();
  await tauriAutoAuth({ force: true });
  return !tauriLoginRequired && hasAuthToken();
}

/** "Forgot password?" on the Desktop login page: the shell proves the owner with its local secret, which the
 *  backend accepts only over direct loopback. Signs the owner in with the new password's session. */
export async function resetDesktopOwnerPassword(newPassword: string): Promise<boolean> {
  const { invoke } = await import("@tauri-apps/api/core");
  let tokens: DesktopAuthResponse;
  try {
    tokens = await invoke<DesktopAuthResponse>("desktop_reset_password", { newPassword });
  } catch (error) {
    throw new Error(authFailureMessage(error));
  }
  if ("login_required" in tokens) throw new Error(TAURI_AUTH_FAILURE_FALLBACK);
  clearDesktopSignedOut();
  return transitionBrowserAccount(OWNER_BROWSER_ACCOUNT, "/chat", () => {
    storeAuthTokens(tokens.access_token, tokens.refresh_token);
    setMustChangePassword(false);
  });
}
