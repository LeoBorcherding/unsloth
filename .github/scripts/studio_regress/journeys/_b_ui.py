"""Chat-page UI helpers shared by the fork-B journeys (Playwright, stable accessible names)."""

from __future__ import annotations

import time

from ..contract import StepFailed, StepUnreachable

ASSISTANT = "[data-role=assistant]"
USER = "[data-role=user]"
STOP = 'button[aria-label="Stop generating"], button[aria-label*="Stop"]'


async def need(locator, what, timeout_ms=15_000):
    try:
        await locator.first.wait_for(state="visible", timeout=timeout_ms)
    except Exception:
        raise StepUnreachable(f"{what} not found") from None
    return locator.first


async def open_run_settings(page):
    """Open the run-settings panel. The closed panel stays in the DOM (off-canvas), so openness
    is judged by the Open button, not by the panel's own controls."""
    opener = page.get_by_role("button", name="Open run settings")
    try:
        await opener.first.wait_for(state="visible", timeout=5_000)
        await opener.first.click()
    except Exception:
        pass  # already open: the opener is hidden while the panel shows
    box = await need(page.get_by_role("textbox", name="Context Length"), "Context Length field")
    await page.wait_for_function(
        "el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.right <= innerWidth; }",
        arg=await box.element_handle(), timeout=10_000)


async def set_field(page, name, value):
    """Type `value` into a labelled field and prove it stuck. Some fields swap their input
    element on the first keystroke (Context Length goes from "Auto" to a numeric input), so on
    a loaded host the rest of the keystrokes can land on the detached one and the field commits
    "4" (clamped to 128) instead of 4096. Re-locate and retype until the committed value is
    the one asked for."""
    want = str(value)
    got = None
    for _ in range(3):
        box = await need(page.get_by_role("textbox", name=name), f"{name} field")
        await box.click()
        await box.press("Control+a")
        await box.press_sequentially(want, delay=20)
        await page.wait_for_timeout(150)
        box = await need(page.get_by_role("textbox", name=name), f"{name} field")
        if (await box.input_value()).replace(",", "") != want:
            continue
        await box.press("Tab")
        await page.wait_for_timeout(150)
        got = await (await need(page.get_by_role("textbox", name=name), f"{name} field")).input_value()
        if got.replace(",", "") == want or _same_number(got, want):
            return got
    return got if got is not None else await (await need(page.get_by_role("textbox", name=name),
                                                           f"{name} field")).input_value()


def _same_number(a, b):
    try:
        return float(str(a).replace(",", "")) == float(str(b).replace(",", ""))
    except ValueError:
        return False


async def field(page, name):
    box = await need(page.get_by_role("textbox", name=name), f"{name} field")
    return await box.input_value()


async def send(page, text):
    """Type and submit with Enter (the send button's accessible name changes with context state,
    e.g. once the window is nearly full; Enter is what a user does either way)."""
    box = await need(page.get_by_role("textbox", name="Message input"), "Message input")
    await box.click()
    await box.fill(text)
    await box.press("Enter")


async def wait_reply(page, n_before, timeout_s=180):
    """Wait until an assistant row beyond `n_before` exists and streaming stopped."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        n = await page.locator(ASSISTANT).count()
        streaming = await page.locator(STOP).count() and await page.locator(STOP).first.is_visible()
        if n > n_before and not streaming:
            txt = (await page.locator(ASSISTANT).nth(n - 1).inner_text()).strip()
            if txt:
                return txt
        await page.wait_for_timeout(500)
    raise StepFailed(f"no finished assistant reply within {timeout_s}s")
