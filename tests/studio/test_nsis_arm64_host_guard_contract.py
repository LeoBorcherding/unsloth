# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ARM64 installer must refuse an x64 host: RunningX64 is true on both, so nothing else stops it."""

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "studio/src-tauri/windows/installer.nsi"


@pytest.fixture(scope = "module")
def on_init() -> list[str]:
    text = TEMPLATE.read_text(encoding = "utf-8")
    body = re.search(r"^Function \.onInit\n(.*?)^FunctionEnd", text, re.S | re.M)
    assert body, "expected a .onInit function"
    return [l.strip() for l in body.group(1).splitlines()]


ARM64_ONLY = '!if "${ARCH}" == "arm64"'


def _guard(on_init: list[str]) -> list[str]:
    assert ARM64_ONLY in on_init, ".onInit has no arm64-only block"
    start = on_init.index(ARM64_ONLY)
    end = on_init.index("!endif", start)
    return on_init[start:end]


def test_guard_is_scoped_to_the_arm64_build(on_init: list[str]) -> None:
    assert any("IsNativeARM64" in l for l in _guard(on_init))


def test_guard_quits_with_an_error_level_and_a_silent_safe_message(on_init: list[str]) -> None:
    guard = _guard(on_init)
    assert "${IfNot} ${IsNativeARM64}" in guard
    box = next(l for l in guard if l.startswith("MessageBox"))
    assert "/SD IDOK" in box, "a /S install must not block on the dialog"
    assert "Unsloth-Desktop-Windows.exe" in box
    level = next(l for l in guard if l.startswith("SetErrorLevel"))
    assert level.split()[1] != "0"
    assert guard.index(level) < guard.index("Quit")


def test_guard_runs_before_the_existing_install_is_touched(on_init: list[str]) -> None:
    _guard(on_init)
    guard_at = on_init.index(ARM64_ONLY)
    for later in ("!insertmacro SetContext", "Call RestorePreviousInstallLocation"):
        assert guard_at < on_init.index(later)
