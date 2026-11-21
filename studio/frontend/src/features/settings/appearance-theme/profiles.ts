// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  type StateStorage,
  createJSONStorage,
  persist,
} from "zustand/middleware";
import {
  type AppearanceSnapshot,
  type AppearanceTheme,
  captureTheme,
  cleanThemeName,
  fontKey,
  FONT_FIELDS,
  parseThemeFile,
} from "./theme-file.ts";

export const MAX_PROFILES = 30;
const MAX_ID = 64;

/** `theme` is a theme file object, read back through `parseThemeFile` like an import. */
export type AppearanceProfile = { id: string; theme: Record<string, unknown> };

/** Imported fonts are kept by name and the background is left out, so a profile stays a few KB. */
export function captureProfile(
  snapshot: AppearanceSnapshot,
  opts: { name: string; includeLayout: boolean },
): Record<string, unknown> {
  return captureTheme(
    {
      ...snapshot,
      // No imported fonts to drop, so their selections survive; the fonts stay on this device.
      customization: { ...snapshot.customization, importedFonts: [] },
      background: null,
    },
    { ...opts, includeAssets: false },
  );
}

export function parseProfile(
  profile: AppearanceProfile,
): AppearanceTheme | null {
  const result = parseThemeFile(JSON.stringify(profile.theme));
  return result.ok ? result.theme : null;
}

export const profileName = (profile: AppearanceProfile) =>
  parseProfile(profile)?.name ?? "";

/** Drops anything that isn't a profile or fails the import checks. */
export function readProfiles(value: unknown): AppearanceProfile[] {
  if (!Array.isArray(value)) return [];
  const seen = new Set<string>();
  const out: AppearanceProfile[] = [];
  for (const entry of value) {
    if (out.length >= MAX_PROFILES) break;
    if (typeof entry !== "object" || entry === null) continue;
    const { id, theme } = entry as Partial<AppearanceProfile>;
    if (typeof id !== "string" || !id || id.length > MAX_ID || seen.has(id))
      continue;
    if (typeof theme !== "object" || theme === null || Array.isArray(theme))
      continue;
    if (!parseProfile({ id, theme })) continue;
    // A hand-edited entry could carry assets; keep only what a saved profile holds.
    const rest = { ...theme };
    delete rest.fonts;
    delete rest.background;
    seen.add(id);
    out.push({ id, theme: rest });
  }
  return out;
}

/** Export carries the imported fonts the profile selects, if they are still on this device. */
export function profileExportFile(
  theme: AppearanceTheme,
  importedFonts: AppearanceSnapshot["customization"]["importedFonts"],
): Record<string, unknown> {
  const selected = new Set(
    FONT_FIELDS.map((field) => fontKey(theme.customization[field])),
  );
  const used = importedFonts.filter((font) => selected.has(fontKey(font.name)));
  return captureTheme(
    {
      palette: theme.palette,
      customization: { ...theme.customization, importedFonts: used },
      glass: { ...theme.glass, style: "frosted" },
      background: null,
    },
    {
      name: theme.name,
      includeLayout: theme.hasLayout,
      includeAssets: used.length > 0,
    },
  );
}

const newId = () =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

// Set by the storage below, so a save that didn't fit can be rolled back and reported.
let writeFailed = false;
const reportingLocalStorage: StateStorage = {
  getItem: (name) => {
    try {
      return window.localStorage.getItem(name);
    } catch {
      return null;
    }
  },
  setItem: (name, value) => {
    try {
      window.localStorage.setItem(name, value);
      writeFailed = false;
    } catch {
      writeFailed = true;
    }
  },
  removeItem: (name) => {
    try {
      window.localStorage.removeItem(name);
    } catch {
      // ignore
    }
  },
};

export type SaveProfileResult = "saved" | "full" | "storage";

type ProfilesState = {
  profiles: AppearanceProfile[];
  save: (theme: Record<string, unknown>) => SaveProfileResult;
  rename: (id: string, name: string) => boolean;
  remove: (id: string) => void;
  restore: (profile: AppearanceProfile, index: number) => SaveProfileResult;
};

export const useAppearanceProfilesStore = create<ProfilesState>()(
  persist(
    (set, get) => {
      /** Applies `next`, or puts the old list back when storage refuses it. */
      const commit = (next: AppearanceProfile[]) => {
        const prev = get().profiles;
        writeFailed = false;
        set({ profiles: next });
        if (!writeFailed) return true;
        set({ profiles: prev });
        return false;
      };
      return {
        profiles: [],
        save: (theme) => {
          const profiles = get().profiles;
          if (profiles.length >= MAX_PROFILES) return "full";
          return commit([...profiles, { id: newId(), theme }])
            ? "saved"
            : "storage";
        },
        rename: (id, name) => {
          const clean = cleanThemeName(name);
          if (!clean) return false;
          return commit(
            get().profiles.map((p) =>
              p.id === id ? { ...p, theme: { ...p.theme, name: clean } } : p,
            ),
          );
        },
        remove: (id) => {
          commit(get().profiles.filter((p) => p.id !== id));
        },
        restore: (profile, index) => {
          const profiles = get().profiles.filter((p) => p.id !== profile.id);
          // A save made since the delete can fill the list; never evict it to undo.
          if (profiles.length >= MAX_PROFILES) return "full";
          profiles.splice(Math.min(index, profiles.length), 0, profile);
          return commit(profiles) ? "saved" : "storage";
        },
      };
    },
    {
      name: "unsloth_appearance_profiles",
      version: 1,
      storage: createJSONStorage(() => reportingLocalStorage),
      partialize: (state) => ({ profiles: state.profiles }),
      // Validated on every rehydrate, as an imported file is, so a corrupt entry is dropped.
      merge: (persisted, current) => ({
        ...current,
        profiles: readProfiles(
          (persisted as Partial<ProfilesState> | undefined)?.profiles,
        ),
      }),
    },
  ),
);

if (
  typeof window !== "undefined" &&
  typeof window.addEventListener === "function"
) {
  window.addEventListener("storage", (event) => {
    if (
      event.key === useAppearanceProfilesStore.persist.getOptions().name &&
      event.newValue !== null
    ) {
      void useAppearanceProfilesStore.persist.rehydrate();
    }
  });
}
