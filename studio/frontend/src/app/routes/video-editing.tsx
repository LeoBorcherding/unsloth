// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const VideoEditingPage = lazyRouteComponent(
  () => import("@/features/video-editing/video-editing-page"),
  "VideoEditingPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/video-editing",
  staticData: { titleKey: "shell.navigation.videoEditing" },
  beforeLoad: () => requireAuth(),
  component: VideoEditingPage,
});
