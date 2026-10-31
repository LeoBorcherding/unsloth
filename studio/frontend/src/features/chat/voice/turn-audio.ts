// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The chat model's own voice (voice-models.ts VOICE_CHAT_MODEL_VOICE): the turn's recorded
// speech goes to the model with its transcript, and the reply comes back as speech chunks.
// Dependency-free so the node test runner loads it directly.

import { pcmWavBase64 } from "../utils/wav-chunks.ts";

/** The speech of the turn being said, fragment by fragment, until the turn is sent. */
export function createTurnAudio() {
  let chunks: Float32Array[] = [];
  let rate = 0;
  return {
    add(samples: Float32Array, sampleRate: number): void {
      if (rate && sampleRate !== rate) chunks = [];
      rate = sampleRate;
      chunks.push(samples);
    },
    /** The turn as one WAV (base64), emptied; null when nothing was said. */
    take(): string | null {
      if (!chunks.length || !rate) return null;
      const total = chunks.reduce((n, c) => n + c.length, 0);
      const merged = new Float32Array(total);
      let at = 0;
      for (const c of chunks) {
        merged.set(c, at);
        at += c.length;
      }
      chunks = [];
      return pcmWavBase64(merged, rate);
    },
    clear(): void {
      chunks = [];
    },
    get size(): number {
      return chunks.length;
    },
  };
}

/** The user message's audio attachment for a turn's WAV. */
export function voiceTurnAttachment(wavBase64: string, id: string) {
  return {
    id,
    type: "file" as const,
    name: "voice.wav",
    contentType: "audio/wav",
    content: [{ type: "audio" as const, audio: { data: wavBase64, format: "wav" as const } }],
    status: { type: "complete" as const },
  };
}

type Part = { type: string };
type Message = { role: string; content?: readonly Part[]; attachments?: readonly { content?: readonly Part[] }[] };

/** Whether the newest user message carries audio: its reply is the spoken one. */
export function lastUserTurnHasAudio(messages: readonly Message[]): boolean {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role !== "user") continue;
    const parts = [...(m.content ?? []), ...(m.attachments ?? []).flatMap((a) => a.content ?? [])];
    return parts.some((p) => p.type === "audio");
  }
  return false;
}

const AUDIO_PLAYER = /<audio-player\b[^>]*\/>/g;

/** Reply text without the spoken reply's player markup, which is not something to say. */
export function withoutAudioPlayer(text: string): string {
  return text.includes("<audio-player") ? text.replace(AUDIO_PLAYER, "").trim() : text;
}
