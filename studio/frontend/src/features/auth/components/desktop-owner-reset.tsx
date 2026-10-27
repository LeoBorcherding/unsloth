// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useState } from "react";
import type { ReactElement, SyntheticEvent } from "react";
import { resetDesktopOwnerPassword } from "../tauri-auto-auth";

/** Desktop only: the owner's forgotten password, reset with the shell's local secret. `onDone` gets whether the
 *  account switch already reloaded the page. */
export function DesktopOwnerReset({ onDone }: { onDone: (replaced: boolean) => void }): ReactElement {
  const [open, setOpen] = useState(false);
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const invalid = next.length < 8 || /\s/.test(next) || next !== confirm;

  if (!open) {
    return (
      <Button type="button" variant="link" className="mx-auto flex w-fit" onClick={() => setOpen(true)}>
        Forgot password?
      </Button>
    );
  }

  async function submit(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    if (invalid || busy) return;
    setBusy(true);
    setError(null);
    try {
      onDone(await resetDesktopOwnerPassword(next));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Password reset failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="space-y-4" onSubmit={submit}>
      <p className="text-sm text-muted-foreground">
        Set a new password for the installation owner. This signs out every other session and revokes its API keys.
        Other accounts: ask the owner for a new setup code.
      </p>
      <div className="space-y-2">
        <Label htmlFor="owner-reset-password">New owner password</Label>
        <Input
          id="owner-reset-password"
          type="password"
          autoComplete="new-password"
          value={next}
          onChange={(event) => setNext(event.target.value)}
          minLength={8}
          required
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="owner-reset-confirm">Confirm password</Label>
        <Input
          id="owner-reset-confirm"
          type="password"
          autoComplete="new-password"
          value={confirm}
          onChange={(event) => setConfirm(event.target.value)}
          minLength={8}
          required
        />
      </div>
      <p className={`text-xs ${error ? "text-destructive" : "text-muted-foreground"}`} aria-live="polite">
        {error ??
          (/\s/.test(next)
            ? "New password cannot contain spaces."
            : confirm && next !== confirm
              ? "Please ensure passwords match."
              : "Must be at least 8 characters.")}
      </p>
      <div className="flex justify-center gap-2">
        <Button type="button" variant="ghost" disabled={busy} onClick={() => setOpen(false)}>
          Cancel
        </Button>
        <Button type="submit" variant="dark" disabled={busy || invalid}>
          {busy ? "Please wait..." : "Reset password"}
        </Button>
      </div>
    </form>
  );
}
