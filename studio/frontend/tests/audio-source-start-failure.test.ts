// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as AudioSourceModule from "../src/features/audio/hooks/use-audio-source.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
const audioSourceState = await import(
  "../src/features/audio/hooks/audio-source-state.ts"
);

/** useAudioSource under a minimal React, with a recorder whose start() throws. */
function setup() {
  const actions: { type: string; message?: string }[] = [];
  const timers = new Set<number>();
  let nextTimer = 0;
  (globalThis as { window?: unknown }).window = {
    setTimeout: () => {
      timers.add(++nextTimer);
      return nextTimer;
    },
    clearTimeout: (id: number) => timers.delete(id),
  };
  const tracks: { stopped: boolean }[] = [];
  let micOpens = 0;
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      mediaDevices: {
        getUserMedia: async () => {
          micOpens += 1;
          const track = {
            stopped: false,
            stop: () => {
              track.stopped = true;
            },
          };
          tracks.push(track);
          return { active: false, getTracks: () => [track] };
        },
      },
    },
  });
  const cells: { v: unknown }[] = [];
  let at = 0;
  const cell = <T>(init: () => T): { v: T } => {
    const i = at++;
    if (!(i in cells)) cells[i] = { v: init() };
    return cells[i] as { v: T };
  };
  const react = {
    useRef: (value: unknown) => cell(() => ({ current: value })).v,
    useState: (value: unknown) => [cell(() => value).v, () => {}],
    useReducer: (_reducer: unknown, initial: unknown) => [
      cell(() => initial).v,
      (action: { type: string; message?: string }) => actions.push(action),
    ],
    useCallback: (fn: unknown) => fn,
    useEffect: () => {},
  };
  const mod = loadWithStubs<typeof AudioSourceModule>(
    new URL("../src/features/audio/hooks/use-audio-source.ts", import.meta.url),
    {
      react,
      "@/features/chat/adapters/pcm-recorder": {
        createAudioRecorder: () => ({
          state: "inactive",
          mimeType: "audio/webm",
          addEventListener: () => {},
          stop: () => {},
          start: () => {
            throw Object.assign(new Error("stream ended"), {
              name: "NotSupportedError",
            });
          },
        }),
      },
      "@/features/chat/adapters/studio-web-speech-dictation-adapter": {
        isMissingDeviceError: () => false,
      },
      "@/features/settings/stores/voice-settings-store": {
        useVoiceSettingsStore: { getState: () => ({ micDeviceId: "default" }) },
      },
      "@/lib/api-base": { isTauri: false },
      "../api": {},
      "../audio-run-request": {},
      "../components/waveform-decode": {},
      "./audio-source-state": audioSourceState,
    },
  );
  const hook = () => {
    at = 0;
    return mod.useAudioSource({ value: null, onChange: () => {} });
  };
  return { hook, actions, timers, tracks, opens: () => micOpens };
}

test("a recorder that refuses to start releases the mic and lets the user retry", async () => {
  const { hook, actions, timers, tracks, opens } = setup();

  await assert.doesNotReject(hook().startRecording());
  assert.equal(actions.at(-1)?.type, "fail");
  assert.ok(actions.at(-1)?.message);
  assert.ok(!actions.some((action) => action.type === "record-start"));
  assert.ok(
    tracks.every((track) => track.stopped),
    "every mic track is stopped",
  );
  assert.equal(timers.size, 0, "the recording limit timer is cleared");

  await assert.doesNotReject(hook().startRecording());
  assert.equal(opens(), 2, "a second click reaches the microphone again");
});
