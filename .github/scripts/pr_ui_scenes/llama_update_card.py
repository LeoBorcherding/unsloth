"""Scene: the llama.cpp / whisper.cpp runtime update card (PR 14436).

The card only renders when an update is on offer, so the update endpoints are mocked
with Playwright routes (same payload shapes as tests/studio/playwright_update_banner_layout.py):
a llama.cpp + whisper.cpp bundle, no app update.

Three pairs, each a clip of the bottom-right corner where the card sits:
  0  collapsed card
  1  the details toggle clicked (BEFORE "llama.cpp changes", AFTER "What's new")
  2  after skipping (BEFORE the X button, AFTER "Skip these versions"): BEFORE shows
     nothing, AFTER shows the Undo notice
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

NO_APP_UPDATE = {
    "current_version": "2026.8.7",
    "latest_version": "2026.8.7",
    "update_available": False,
    "install_source": "pypi",
    "can_show_web_notification": False,
    "release_notes_url": None,
    "checked_at": "2099-01-01T00:00:00Z",
    "reason": None,
    "error": None,
}
IDLE_JOB = {
    "state": "idle",
    "message": "",
    "from_tag": None,
    "to_tag": None,
    "reload_required": None,
    "error": None,
    "progress": None,
    "finished_at": None,
}
BUNDLE = {
    "supported": True,
    "update_available": True,
    "llama_update_available": True,
    "update_component": "llama",
    "installed_tag": "b11724",
    "latest_tag": "b11734-mix-c208e5f",
    "update_size_bytes": 28 * 1024 * 1024,
    "source_build": False,
    "component": "llama.cpp",
    "whisper": {
        "update_available": True,
        "installed_tag": "v1.9.1",
        "latest_tag": "v1.9.2",
        "update_size_bytes": 11 * 1024 * 1024,
        "skip_reason": None,
    },
    "job": IDLE_JOB,
}
CHANGELOG = {
    "matched": True,
    "installed_tag": "b11724",
    "latest_tag": "b11734-mix-c208e5f",
    "changes": [
        {
            "summary": "model: add GLM-5-Next",
            "links": [
                {"label": "#27754", "url": "https://github.com/ggml-org/llama.cpp/pull/27754"}
            ],
        },
        {"summary": "llama: batched readahead for lazily read gather tables", "links": []},
        {"summary": "MTP for Qwen3.8-Flash-Next", "links": []},
    ],
    "total_changes": 3,
    "truncated": False,
    "release_url": "https://github.com/unslothai/llama.cpp/releases/tag/b11734-mix-c208e5f",
    "error": None,
}

VIEW = (1280, 860)
CLIP = {"x": 760, "y": 360, "width": 520, "height": 500}


def _stub(payload: dict):
    async def handler(route) -> None:
        await route.fulfill(status = 200, content_type = "application/json", body = json.dumps(payload))

    return handler


async def _card_facts(page) -> dict:
    return await page.evaluate("""() => {
      const card = document.querySelector('[data-testid="llama-update-banner"]');
      const skipped = document.querySelector('[data-testid="llama-update-skipped"]');
      return {
        card: !!card,
        buttons: card ? [...card.querySelectorAll('button')].map(b => (b.innerText || b.getAttribute('aria-label') || '').trim()) : [],
        text: card ? card.innerText.replace(/\\s+/g, ' ').trim().slice(0, 400) : null,
        skipped_notice: skipped ? skipped.innerText.replace(/\\s+/g, ' ').trim() : null,
      };
    }""")


async def drive(
    session: Session, out_dir: Path, label: str, **_: object
) -> tuple[list[Path], dict]:
    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    shots: list[Path] = []
    facts: dict = {}
    async with open_chat(session.base_url, init_scripts = [init], viewport = VIEW, headless = True) as sp:
        page = sp.page
        await page.emulate_media(color_scheme = "dark")
        await page.route("**/api/studio/update-status*", _stub(NO_APP_UPDATE))
        await page.route("**/api/llama/update-status*", _stub(BUNDLE))
        await page.route("**/api/llama/update-changelog*", _stub(CHANGELOG))
        await page.goto(f"{session.base_url}/", wait_until = "domcontentloaded")
        card = page.locator('[data-testid="llama-update-banner"]')
        await card.wait_for(state = "visible", timeout = 30_000)
        await page.wait_for_timeout(1_500)

        async def snap(name: str) -> None:
            path = out_dir / f"{label.lower()}_{name}.png"
            await page.screenshot(path = str(path), clip = CLIP)
            shots.append(path)

        facts["collapsed"] = await _card_facts(page)
        await snap("0_collapsed")

        await page.locator('[data-testid="llama-update-changelog-toggle"]').click()
        await page.wait_for_timeout(1_500)
        facts["details_open"] = await _card_facts(page)
        await snap("1_details_open")

        skip = page.locator('[data-testid="llama-update-skip-button"]')
        if await skip.count():
            await skip.click()
            facts["skip_control"] = "Skip button"
        else:
            await page.locator(
                '[data-testid="llama-update-banner"] button[aria-label^="Dismiss"]'
            ).click()
            facts["skip_control"] = "X button"
        await page.wait_for_timeout(1_000)
        facts["after_skip"] = await _card_facts(page)
        await snap("2_after_skip")
    return shots, facts


if __name__ == "__main__":
    import argparse

    from pr_ui_scenes._common import studio_session

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required = True)
    ap.add_argument("--home", type = Path, required = True)
    ap.add_argument("--password", required = True)
    ap.add_argument("--out", type = Path, required = True)
    ap.add_argument("--label", default = "AFTER")
    a = ap.parse_args()
    a.out.mkdir(parents = True, exist_ok = True)
    s = studio_session(a.url, a.home, a.password)
    print(asyncio.run(drive(s, a.out, a.label)))
