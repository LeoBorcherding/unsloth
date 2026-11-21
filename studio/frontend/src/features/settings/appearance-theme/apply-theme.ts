// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import {
  applyAppearanceSnapshot,
  applyThemeBackground,
  portableGlass,
  readAppearanceSnapshot,
  readLoadedAppearanceSnapshot,
  restoreBackground,
  withLoadableFonts,
} from "./snapshot";
import { type AppearanceTheme, planThemeApply, undoTarget } from "./theme-file";

const K = "settings.appearance.themeFile";

/** Shared by import and saved profiles: apply, then a toast whose Undo puts back the whole look. */
export async function applyTheme(
  t: ReturnType<typeof useT>,
  theme: AppearanceTheme,
  opts: { applyLayout: boolean; applyAssets: boolean },
  onApplied?: () => void,
): Promise<void> {
  const before = await readLoadedAppearanceSnapshot();
  const [loadable, badFonts] = opts.applyAssets
    ? await withLoadableFonts(theme)
    : [theme, []];
  for (const path of badFonts)
    toast.warning(t(`${K}.warning.invalid`, { path }));
  const next = portableGlass(planThemeApply(before, loadable, opts));
  applyAppearanceSnapshot(next);
  if (
    next.background !== "keep" &&
    !(await applyThemeBackground(next.background))
  ) {
    toast.warning(t(`${K}.warning.backgroundSkipped`));
  }
  onApplied?.();
  toast.success(
    t(`${K}.applied`, { name: theme.name || t(`${K}.importTitle`) }),
    {
      action: {
        label: t(`${K}.undo`),
        onClick: () => {
          applyAppearanceSnapshot(undoTarget(before, readAppearanceSnapshot()));
          const restoring =
            next.background === "keep"
              ? Promise.resolve(true)
              : restoreBackground(before.background);
          void restoring.then((ok) =>
            ok
              ? toast(t(`${K}.undone`))
              : toast.warning(t(`${K}.undoneWithoutBackground`)),
          );
        },
      },
    },
  );
}
