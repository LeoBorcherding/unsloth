"""Journey 10: image generation on the smallest pipeline Studio accepts (tiny SDXL, local).

Steps: load (API) -> Images page -> fixed params (prompt, 256x256, 1 step, cfg 0, seed 0) ->
Generate -> result + gallery facts (image sha for the same seed) -> image menu -> Recipe ->
long generation + Stop -> unload. Loads with speed_mode off (see _diffusion.DETERMINISTIC_LOAD).
Video is journeys/diffusion_video.py.
"""

from __future__ import annotations

import asyncio
import re

from studio_regress.contract import Journey, Step, StepFailed, StepUnreachable
from studio_regress.journeys import _diffusion as d

MASKS = ("[data-sonner-toaster]",)


async def load(ctx):
    await ctx.api.post("/api/inference/images/unload")
    await ctx.api.post("/api/inference/images/load", json=d.image_load_body(ctx))
    s = await d.wait_images_loaded(ctx)
    ctx.state["gallery0"] = len(await d.gallery(ctx))
    return {"family": s.get("family"), "model_kind": s.get("model_kind"), "device": s.get("device"),
            "dtype": s.get("dtype"), "workflows": s.get("workflows"), "supports_lora": s.get("supports_lora")}


async def open_images(ctx):
    p = ctx.page
    await p.goto(ctx.base_url + "/images", wait_until="domcontentloaded")
    try:
        await p.get_by_role("button", name="Generate", exact=True).first.wait_for(timeout=30_000)
    except Exception as e:
        raise StepUnreachable(f"/images has no Generate button: {e}") from None
    await d.dismiss_toasts(p)
    return {"url": p.url.replace(ctx.base_url, "")}


async def _fill(page, label, value):
    loc = page.locator(f'input[aria-label="{label}"], input[placeholder="{label}"]')
    if await loc.count() == 0:
        raise StepUnreachable(f"input {label!r} not found")
    await loc.first.fill(str(value))
    await loc.first.press("Tab")


async def set_params(ctx):
    p = ctx.page
    box = p.locator("textarea").first
    await box.fill(d.PROMPT)
    for label, value in (("Width", 256), ("Height", 256), ("Steps", 1), ("Guidance", 0),
                         ("Random if empty", 0)):
        await _fill(p, label, value)
    await p.wait_for_timeout(300)
    vals = {}
    for label in ("Width", "Height", "Steps", "Guidance"):
        vals[label.lower()] = await p.locator(f'input[aria-label="{label}"], input[placeholder="{label}"]').first.input_value()
    return vals


async def generate(ctx):
    p = ctx.page
    await p.get_by_role("button", name="Generate", exact=True).first.click()
    await asyncio.sleep(1)
    await d.wait_generation_idle(ctx)
    imgs = await d.gallery(ctx)
    if len(imgs) <= ctx.state.get("gallery0", 0):
        raise StepFailed("Generate produced no gallery image")
    img = imgs[0]
    # Compared fact: with speed_mode off the decoded pixels for seed 0 are bit-identical (measured 5/5
    # in one load, again after a reload, and across four Studio processes on fresh homes).
    sha, size = await d.image_sha(ctx, img)
    await p.wait_for_timeout(1500)
    await d.dismiss_toasts(p)
    await d.mask_results(p)
    return {"width": img.get("width"), "height": img.get("height"), "steps": img.get("steps"),
            "seed": img.get("seed"), "image_sha": sha, "decoded_size": size, "gallery_count": len(imgs)}


async def _hover_result(p):
    """The Recipe / Download / menu toolbar only renders while the result image is hovered."""
    imgs = p.locator("img")
    best, area = None, 0
    for i in range(await imgs.count()):
        b = await imgs.nth(i).bounding_box()
        if b and b["width"] * b["height"] > area:
            best, area = imgs.nth(i), b["width"] * b["height"]
    if best is not None:
        await best.hover()
        await p.wait_for_timeout(300)


async def image_menu(ctx):
    p = ctx.page
    await _hover_result(p)
    btn = p.get_by_role("button", name="More actions for this image")
    if await btn.count() == 0:
        raise StepUnreachable("no image action menu")
    await btn.first.click()
    await p.wait_for_timeout(500)
    items = [t.strip() for t in await p.get_by_role("menuitem").all_inner_texts()]
    await d.mask_results(p)
    return {"menu_items": items}


async def recipe(ctx):
    p = ctx.page
    await p.keyboard.press("Escape")
    await _hover_result(p)
    btn = p.get_by_role("button").filter(has_text=re.compile(r"^\s*Recipe\s*$"))
    if await btn.count() == 0:
        raise StepUnreachable("no Recipe button")
    await btn.first.click()
    await p.wait_for_timeout(700)
    dlg = p.locator("[role=dialog]")
    text = (await dlg.first.inner_text()) if await dlg.count() else ""
    await d.mask_results(p)
    return {"recipe_has_prompt": d.PROMPT in text, "recipe_has_seed": "0" in text}


async def stop(ctx):
    p = ctx.page
    await p.keyboard.press("Escape")
    await _fill(p, "Steps", 100)
    await _fill(p, "Width", 1024)
    await _fill(p, "Height", 1024)
    await p.get_by_role("button", name="Generate", exact=True).first.click()
    active = False
    for _ in range(20):
        if (await ctx.api.get("/api/inference/images/generate-progress")).get("active"):
            active = True
            break
        await asyncio.sleep(0.25)
    stop_btn = p.get_by_role("button").filter(has_text="Stop")
    has_stop = await stop_btn.count() > 0
    if has_stop:
        await stop_btn.last.click()
    else:
        await ctx.api.post("/api/inference/images/generate/cancel")
    after = await d.wait_generation_idle(ctx, timeout_s=90)
    await p.wait_for_timeout(1000)
    await d.dismiss_toasts(p)
    await d.mask_results(p)
    return {"was_active": active, "stop_control": has_stop, "after_active": bool(after.get("active"))}


async def unload(ctx):
    s = await ctx.api.post("/api/inference/images/unload")
    return {"loaded": s.get("loaded")}


JOURNEY = Journey(
    name="diffusion_gen", tier="gpu", needs=("tiny_sdxl",), routes=("/images",),
    steps=(
        Step("s01_load", load, shot=False, timeout_s=300),
        Step("s02_images_page", open_images, masks=MASKS),
        Step("s03_params", set_params, masks=MASKS),
        Step("s04_generate", generate, masks=MASKS, timeout_s=180),
        Step("s05_image_menu", image_menu, masks=MASKS),
        Step("s06_recipe", recipe, masks=MASKS),
        Step("s07_stop", stop, masks=MASKS, timeout_s=150),
        Step("s08_unload", unload, shot=False),
    ),
)
