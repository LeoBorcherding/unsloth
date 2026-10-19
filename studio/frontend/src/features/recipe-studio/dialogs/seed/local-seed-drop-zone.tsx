// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useRef } from "react";
import { CloudUploadIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNativeFileDrop } from "@/features/native-intents";
import { isTauri } from "@/lib/api-base";
import { LOCAL_SEED_UPLOAD_MAX_LABEL } from "./upload-limits";

export const LOCAL_SEED_ACCEPT = ".csv,.json,.jsonl,.parquet";
// Unsloth Desktop hands a drop to the chat attachment reader: no Parquet, CSV up to 20 MB.
const NATIVE_DROP_ACCEPT = ".csv,.json,.jsonl";

type LocalSeedDropZoneProps = {
  onFile: (file: File) => void;
  disabled?: boolean;
};

export function LocalSeedDropZone({ onFile, disabled }: LocalSeedDropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const { ref, dragging, dragHandlers } = useNativeFileDrop({
    onFiles: (files) => {
      if (files[0]) onFile(files[0]);
    },
    accept: isTauri ? NATIVE_DROP_ACCEPT : LOCAL_SEED_ACCEPT,
    multiple: false,
    disabled,
    disabledReason: "This block is busy. Try the drop again in a moment.",
  });

  const handleClick = useCallback(() => {
    if (!disabled) inputRef.current?.click();
  }, [disabled]);

  return (
    <>
      <div
        className={`nodrag flex flex-col items-center justify-center rounded-md border-2 border-dashed px-4 py-6 text-center transition-colors ${
          dragging
            ? "border-ring-strong bg-primary/5"
            : "border-muted-foreground/25 hover:border-muted-foreground/50"
        } ${disabled ? "cursor-default opacity-50" : "cursor-pointer"}`}
        ref={ref}
        {...dragHandlers}
        onClick={handleClick}
      >
        <HugeiconsIcon
          icon={CloudUploadIcon}
          className="text-muted-foreground mb-2 size-8"
        />
        <p className="text-muted-foreground text-sm">
          Drop a file here or click to browse
        </p>
        <p className="text-muted-foreground/60 mt-1 text-xs">
          CSV, JSON, JSONL, Parquet - up to {LOCAL_SEED_UPLOAD_MAX_LABEL}
        </p>
        {isTauri && (
          <p className="text-muted-foreground/60 text-xs">
            Drop CSV (up to 20 MB) or JSON; click to browse for Parquet.
          </p>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={LOCAL_SEED_ACCEPT}
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
          event.target.value = "";
        }}
      />
    </>
  );
}
