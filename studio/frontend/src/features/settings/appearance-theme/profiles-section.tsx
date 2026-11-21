// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useT } from "@/i18n";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { toast } from "@/lib/toast";
import {
  Download01Icon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useId, useMemo, useState } from "react";

import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { useAppearanceCustomStore } from "../stores/appearance-custom-store";
import {
  type AppearanceProfile,
  MAX_PROFILES,
  captureProfile,
  parseProfile,
  profileExportFile,
  profileName,
  useAppearanceProfilesStore,
} from "./profiles";
import { downloadThemeText, readAppearanceSnapshot } from "./snapshot";
import { MAX_THEME_NAME } from "./theme-file";
import { applyTheme } from "./apply-theme";
import { Toggle } from "./theme-file-section";

const K = "settings.appearance.profiles";
const F = "settings.appearance.themeFile";

type Editing =
  { kind: "save" } | { kind: "rename"; profile: AppearanceProfile };

function NameForm({
  editing,
  onClose,
}: {
  editing: Editing;
  onClose: () => void;
}) {
  const t = useT();
  const id = useId();
  const { save, rename } = useAppearanceProfilesStore.getState();
  const [name, setName] = useState(
    editing.kind === "rename" ? profileName(editing.profile) : "",
  );
  const [includeLayout, setIncludeLayout] = useState(false);

  function submit() {
    const fallback = t(`${K}.namePlaceholder`);
    if (editing.kind === "rename") {
      if (!name.trim()) return;
      if (!rename(editing.profile.id, name)) {
        toast.error(t(`${K}.storageError`));
        return;
      }
      onClose();
      return;
    }
    const theme = captureProfile(readAppearanceSnapshot(), {
      name: name.trim() || fallback,
      includeLayout,
    });
    const result = save(theme);
    if (result === "full") {
      toast.error(t(`${K}.full`, { count: MAX_PROFILES }));
      return;
    }
    if (result === "storage") {
      toast.error(t(`${K}.storageFull`));
      return;
    }
    onClose();
    toast.success(t(`${K}.saved`, { name: String(theme.name) }));
  }

  return (
    <form
      className="flex flex-col gap-6"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <DialogHeader>
        <DialogTitle>
          {t(editing.kind === "save" ? `${K}.saveTitle` : `${K}.renameTitle`)}
        </DialogTitle>
      </DialogHeader>
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor={`${id}-name`}>{t(`${K}.name`)}</Label>
          <Input
            id={`${id}-name`}
            autoFocus={true}
            value={name}
            maxLength={MAX_THEME_NAME}
            placeholder={t(`${K}.namePlaceholder`)}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        {editing.kind === "save" ? (
          <Toggle
            id={`${id}-layout`}
            label={t(`${F}.exportIncludeLayout`)}
            checked={includeLayout}
            onChange={setIncludeLayout}
            disabled={false}
          />
        ) : null}
      </div>
      <DialogFooter>
        <Button type="button" variant="outline" onClick={onClose}>
          {t("common.cancel")}
        </Button>
        <Button
          type="submit"
          variant="dark"
          disabled={editing.kind === "rename" && !name.trim()}
        >
          {t(editing.kind === "save" ? `${K}.save` : `${K}.rename`)}
        </Button>
      </DialogFooter>
    </form>
  );
}

export function ProfilesSection() {
  const t = useT();
  const profiles = useAppearanceProfilesStore((s) => s.profiles);
  const names = useMemo(() => profiles.map(profileName), [profiles]);
  const [editing, setEditing] = useState<Editing | null>(null);
  const [open, setOpen] = useState(false);
  const edit = (next: Editing) => {
    setEditing(next);
    setOpen(true);
  };
  const [busy, setBusy] = useState(false);

  async function apply(profile: AppearanceProfile) {
    const theme = parseProfile(profile);
    if (!theme) return;
    setBusy(true);
    try {
      await applyTheme(t, theme, {
        applyLayout: theme.hasLayout,
        applyAssets: true,
      });
    } finally {
      setBusy(false);
    }
  }

  async function exportProfile(profile: AppearanceProfile) {
    const theme = parseProfile(profile);
    if (!theme) return;
    const file = profileExportFile(
      theme,
      useAppearanceCustomStore.getState().customization.importedFonts,
    );
    try {
      const fileName = await downloadThemeText(
        `${JSON.stringify(file, null, 2)}\n`,
        theme.name,
      );
      if (fileName) toast.success(t(`${F}.exported`, { name: fileName }));
    } catch {
      toast.error(t(`${F}.exportFailed`));
    }
  }

  function remove(profile: AppearanceProfile, index: number) {
    if (!useAppearanceProfilesStore.getState().remove(profile.id)) {
      toast.error(t(`${K}.storageError`));
      return;
    }
    toast(t(`${K}.deleted`, { name: names[index] }), {
      action: {
        label: t(`${F}.undo`),
        onClick: () => {
          const result = useAppearanceProfilesStore
            .getState()
            .restore(profile, index);
          if (result === "full")
            toast.error(t(`${K}.full`, { count: MAX_PROFILES }));
          else if (result === "storage") toast.error(t(`${K}.storageError`));
        },
      },
    });
  }

  return (
    <SettingsSection title={t(`${K}.title`)}>
      <SettingsRow
        label={t(`${K}.saveLabel`)}
        description={t(`${K}.saveDescription`)}
      >
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => edit({ kind: "save" })}
        >
          {t(`${K}.save`)}
        </Button>
      </SettingsRow>
      {profiles.length === 0 ? (
        <p className="py-3 text-sm text-muted-foreground">{t(`${K}.empty`)}</p>
      ) : (
        <ul aria-label={t(`${K}.listLabel`)} className="flex flex-col">
          {profiles.map((profile, index) => (
            <li key={profile.id} className="group flex items-center gap-3 py-2">
              <span
                className="min-w-0 flex-1 truncate text-sm font-medium text-foreground"
                title={names[index]}
              >
                {names[index]}
              </span>
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={busy}
                onClick={() => void apply(profile)}
              >
                {t(`${K}.apply`)}
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild={true}>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="size-7 p-0 max-sm:size-9"
                    aria-label={t(`${K}.actionsFor`, { name: names[index] })}
                  >
                    <HugeiconsIcon
                      icon={MoreHorizontalIcon}
                      className="size-4"
                    />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem
                    onClick={() => edit({ kind: "rename", profile })}
                  >
                    <HugeiconsIcon
                      icon={PencilEdit02Icon}
                      className="size-3.5 mr-2"
                    />
                    {t(`${K}.rename`)}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => void exportProfile(profile)}>
                    <HugeiconsIcon
                      icon={Download01Icon}
                      className="size-3.5 mr-2"
                    />
                    {t(`${K}.export`)}
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onClick={() => remove(profile, index)}
                    className="text-destructive focus:text-destructive"
                  >
                    <HugeiconsIcon icon={TrashIcon} className="size-3.5 mr-2" />
                    {t(`${K}.delete`)}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </li>
          ))}
        </ul>
      )}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent onCloseAutoFocus={() => setEditing(null)}>
          {editing ? (
            <NameForm editing={editing} onClose={() => setOpen(false)} />
          ) : null}
        </DialogContent>
      </Dialog>
    </SettingsSection>
  );
}
