"""`unsloth studio run` passes unknown llama-server flags through. Click must not read `-ctv`,
`-dev`, `-nkvo` or `-lv` as clusters of its own `-v`/`-y` shorts and pass the leftovers on."""

from __future__ import annotations

import pytest
import typer

from unsloth_cli.commands import studio


def _parse(argv):
    group = typer.main.get_command(studio.studio_app)
    return group.commands["run"].make_context("run", list(argv))


@pytest.mark.parametrize(
    "flags",
    [["-ctv", "q8_0", "-ctk", "q8_0"], ["-dev", "CUDA1"], ["-nkvo"], ["-lv", "3"], ["-fa", "on"]],
)
def test_llama_server_flags_reach_the_tail_intact(flags):
    ctx = _parse(["--model", "x", *flags])
    assert ctx.args == flags
    assert ctx.params["verbose"] is False
    assert ctx.params["yes"] is False


def test_exact_short_flags_are_still_unsloths():
    found, rest = studio._consume_exact_short_flags(["-v", "-ctv", "q8_0", "--", "-v"], "-v")
    assert found is True
    assert rest == ["-ctv", "q8_0", "--", "-v"]
    assert studio._consume_exact_short_flags(["-dev", "CUDA1"], "-v") == (False, ["-dev", "CUDA1"])
