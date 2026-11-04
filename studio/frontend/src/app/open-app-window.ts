// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { translate } from "@/i18n";
import { toast } from "@/lib/toast";

/** Another desktop window on the same backend (src-tauri/src/app_windows.rs). */
export async function openAppWindow(): Promise<void> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("open_app_window");
  } catch (error) {
    toast.error(translate("desktopShell.newWindowFailed", { error: String(error) }));
  }
}
