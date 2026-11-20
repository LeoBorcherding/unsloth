// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

/** An object URL for a blob, revoked when the blob changes or the component unmounts. */
export function useObjectUrl(blob: Blob | null | undefined): string | null {
  // Effect state, not a memo: StrictMode would revoke a memoized URL and blank the image.
  const [state, setState] = useState<{ blob: Blob; url: string } | null>(null);
  useEffect(() => {
    if (!blob) {
      setState(null);
      return;
    }
    const next = URL.createObjectURL(blob);
    setState({ blob, url: next });
    return () => URL.revokeObjectURL(next);
  }, [blob]);
  // Tied to the blob, so the previous blob's soon-revoked URL never renders for a frame.
  return blob && state?.blob === blob ? state.url : null;
}
