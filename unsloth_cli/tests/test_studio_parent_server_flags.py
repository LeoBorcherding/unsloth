# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio -p/-H/-f/-q <subcommand>` must fail loudly: Typer never forwards them."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer
from click.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.commands import studio as studio_mod


@pytest.fixture
def group(monkeypatch):
    group = typer.main.get_command(studio_mod.studio_app)
    calls = []
    for name in ("run", "stop"):
        monkeypatch.setattr(group.commands[name], "callback", lambda **kw: calls.append(kw))
    group.calls = calls
    return group


@pytest.mark.parametrize(
    "parent, fixed",
    [
        (["-p", "9000"], "run --port 9000"),
        (["-H", "0.0.0.0"], "run --host 0.0.0.0"),
        (["-f", "dist"], "run --frontend dist"),
        (["-q"], "run --silent"),
    ],
)
def test_parent_server_flags_before_run_are_rejected(group, parent, fixed):
    result = CliRunner().invoke(group, [*parent, "run", "--model", "x"])

    assert result.exit_code == 2, result.output
    assert fixed in result.output
    assert group.calls == []


def test_parent_port_before_a_subcommand_without_it_says_so(group):
    result = CliRunner().invoke(group, ["-p", "9000", "stop"])

    assert result.exit_code == 2, result.output
    assert "`unsloth studio stop` does not take it" in result.output
    assert group.calls == []


def test_run_without_parent_server_flags_still_runs(group):
    result = CliRunner().invoke(group, ["run", "--model", "x", "-p", "9000", "-H", "0.0.0.0"])

    assert result.exit_code == 0, result.output
    assert (group.calls[0]["port"], group.calls[0]["host"]) == (9000, "0.0.0.0")
