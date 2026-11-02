"""Measure the real footer with deterministic generation frames and two branches."""

import json

from pr_ui_scenes import chat_stream_content_parity as fixture
from studio_test_kit.ui import send_prompt


def scripts(_):
    first = [
        (0, fixture._HEAD),
        (0.1, fixture._delta("A short reply for the footer check.")),
        (0.1, fixture._delta(finish = "stop")),
        (0, fixture._DONE),
    ]
    second = [(0, fixture._HEAD), (0.1, fixture._delta("A short reply for the footer check."))]
    for count in (2, 123456):
        second.append(
            (
                4,
                fixture._sse(
                    {
                        "type": "generation_progress",
                        "phase": "decode",
                        "tokens": count,
                        "tok_per_s": 123.4,
                        "draft_n": count,
                        "draft_accepted": count,
                    }
                ),
            )
        )
    second.extend([(8, fixture._delta(finish = "stop")), (0, fixture._DONE)])
    return [first, second]


READ = """() => {
 const root = [...document.querySelectorAll('[data-role="assistant"]')].at(-1);
 const picker = root?.querySelector('.aui-branch-picker-root');
 const line = root?.querySelector('[data-slot="live-generation-line"]');
 const box = picker?.getBoundingClientRect();
 const footer = root?.querySelector('.aui-assistant-message-footer')?.getBoundingClientRect();
 return {picker: picker?.innerText, live: line?.innerText,
         x: box && box.x - footer.x, y: box && box.y - footer.y,
         height: box?.height};
}"""


async def turn(page, out_dir, label, facts, _):
    shots = []
    await send_prompt(fixture._SP(page), "Give a short reply.")
    await fixture._await_stream_end(page, 0)
    root = page.locator('[data-role="assistant"]').last
    await root.hover()
    await root.get_by_role("button", name = "Refresh", exact = True).click()
    await fixture._wait_for(page, "() => window.__run.count === 2", "regeneration did not start")
    await fixture._wait_for(
        page,
        "() => !!document.querySelector('[data-role=assistant] .aui-branch-picker-root')",
        "regeneration has no branch picker",
    )
    facts["before_live"] = await page.evaluate(READ)
    await fixture._wait_for(
        page,
        "() => !!document.querySelector('[data-slot=live-generation-line]')",
        "no live generation line",
    )
    facts["short_live"] = await page.evaluate(READ)
    shots.append(await fixture._shot(page, out_dir, f"{label}_0_short.png"))
    await fixture._wait_for(
        page,
        "() => /123[,.\\s]?456/.test(document.querySelector('[data-slot=live-generation-line]')?.innerText || '')",
        "live count did not grow",
    )
    facts["long_live"] = await page.evaluate(READ)
    shots.append(await fixture._shot(page, out_dir, f"{label}_1_long.png"))
    await page.set_viewport_size({"width": 400, "height": 900})
    facts["narrow_live"] = await page.evaluate(READ)
    shots.append(await fixture._shot(page, out_dir, f"{label}_2_narrow.png"))
    await fixture._await_stream_end(page, 1)
    facts["finished"] = await page.evaluate(READ)
    shots.append(await fixture._shot(page, out_dir, f"{label}_3_finished.png"))
    facts["reply"] = await page.evaluate("() => window.__parity.replyText()")
    return shots


async def drive(session, out_dir, label, **kwargs):
    old_scripts, old_turn = fixture._scripts, fixture._turn_thread
    old_settings = dict(fixture._SETTINGS)
    fixture._scripts, fixture._turn_thread = scripts, turn
    fixture._SETTINGS["unsloth_chat_preferences"] = json.dumps(
        {"state": {"showGenerationDetails": True, "showReplyTimes": False}, "version": 0}
    )
    try:
        return await fixture.drive(session, out_dir, label, payload = "plain_thread", external = True)
    finally:
        fixture._scripts, fixture._turn_thread = old_scripts, old_turn
        fixture._SETTINGS.clear()
        fixture._SETTINGS.update(old_settings)
