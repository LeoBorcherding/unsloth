// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Whether a cache row is chattable. Every field is optional so an older backend keeps trying. */
export function isChattableCachedRepo(repo: {
  partial?: boolean;
  task?: string | null;
  model_format?: string | null;
  capabilities?: { can_chat?: boolean } | null;
}): boolean {
  return (
    repo.partial !== true &&
    repo.capabilities?.can_chat !== false &&
    // Adapters fetch base weights from the Hub, and can_chat is true on file layout alone.
    repo.model_format !== "adapter" &&
    // Cached diffusion and speech repos report can_chat true on file format alone.
    !NON_CHAT_TASKS.has(repo.task ?? "")
  );
}

// Chat models are tagged text-generation or null, hence a denylist; audio tasks reply with speech.
export const NON_CHAT_TASKS: ReadonlySet<string> = new Set([
  "text-to-image",
  "text-to-video",
  "image-diffusion-unsupported",
  "text-to-speech",
  "text-to-audio",
  "audio-to-audio",
  "automatic-speech-recognition",
]);
