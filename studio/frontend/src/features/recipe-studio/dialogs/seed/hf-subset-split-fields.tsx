// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import type { ReactElement } from "react";
import type { SeedConfig } from "../../types";
import { FieldLabel } from "../shared/field-label";

type HfSubsetSplitFieldsProps = {
  config: SeedConfig;
  onUpdate: (patch: Partial<SeedConfig>) => void;
  disabled?: boolean;
};

// A new subset or split points at different files, so the loaded path and fields go until the next Load.
const CLEARED_LOAD: Partial<SeedConfig> = {
  hf_path: "",
  seed_columns: [],
  seed_drop_columns: [],
  seed_preview_rows: [],
};

export function HfSubsetSplitFields({
  config,
  onUpdate,
  disabled,
}: HfSubsetSplitFieldsProps): ReactElement {
  const subsetId = `${config.id}-hf-subset`;
  const splitId = `${config.id}-hf-split`;
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <div className="grid gap-1.5">
        <FieldLabel
          label="Subset (optional)"
          htmlFor={subsetId}
          hint="Config name for datasets with several, e.g. main or socratic for openai/gsm8k."
        />
        <Input
          id={subsetId}
          className="nodrag"
          disabled={disabled}
          placeholder="default"
          value={config.hf_subset ?? ""}
          onChange={(event) => onUpdate({ ...CLEARED_LOAD, hf_subset: event.target.value })}
        />
      </div>
      <div className="grid gap-1.5">
        <FieldLabel label="Split" htmlFor={splitId} hint="Which split to read, e.g. train or test." />
        <Input
          id={splitId}
          className="nodrag"
          disabled={disabled}
          placeholder="train"
          value={config.hf_split ?? ""}
          onChange={(event) => onUpdate({ ...CLEARED_LOAD, hf_split: event.target.value })}
        />
      </div>
    </div>
  );
}
