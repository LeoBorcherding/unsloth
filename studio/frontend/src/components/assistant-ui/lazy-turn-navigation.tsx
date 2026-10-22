// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { lazy, Suspense, type ComponentProps } from "react";
import type {
  BookmarkTurnMenuItem as MenuItem,
  TurnNavigator as Navigator,
} from "./turn-navigation";
import { useChatPreferencesStore } from "@/features/chat";

// Off the startup bundle: turn navigation is off by default, so nothing loads until it is turned on.
const NavigatorComponent = lazy(() =>
  import("./turn-navigation").then((module) => ({
    default: module.TurnNavigator,
  })),
);
const ButtonComponent = lazy(() =>
  import("./turn-navigation").then((module) => ({
    default: module.BookmarkTurnButton,
  })),
);
const MenuItemComponent = lazy(() =>
  import("./turn-navigation").then((module) => ({
    default: module.BookmarkTurnMenuItem,
  })),
);

const useEnabled = () =>
  useChatPreferencesStore((state) => state.showTurnNavigation);

export function TurnNavigator(props: ComponentProps<typeof Navigator>) {
  if (!useEnabled()) return null;
  return (
    <Suspense fallback={null}>
      <NavigatorComponent {...props} />
    </Suspense>
  );
}

export function BookmarkTurnButton() {
  if (!useEnabled()) return null;
  return (
    <Suspense fallback={null}>
      <ButtonComponent />
    </Suspense>
  );
}

export function BookmarkTurnMenuItem(props: ComponentProps<typeof MenuItem>) {
  if (!useEnabled()) return null;
  return (
    <Suspense fallback={null}>
      <MenuItemComponent {...props} />
    </Suspense>
  );
}
