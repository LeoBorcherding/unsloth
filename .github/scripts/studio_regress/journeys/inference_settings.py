"""Journey 2: GGUF inference settings. Load at ctx 2048, change the context in the run-settings
panel and re-apply (effective ctx read back from /api/inference/status), custom sampling +
system prompt, save a preset, reload the page and prove it persisted, one short reply, Stop, then
re-apply a context while the load reply is lost in transit (no rollback of a load that landed).

Model: unsloth/gemma-3-270m-it-GGUF UD-Q4_K_XL (CPU-capable, 254 MiB)."""

from __future__ import annotations

from ..contract import Journey, Step, StepFailed
from . import _b_common as b
from . import _b_ui as ui

PRESET = "regress-preset"
SYSTEM = "You are a terse assistant. Answer in one short sentence."


async def load_2048(ctx):
    repo, variant = b.model(ctx, "gguf_small", b.GGUF_270M)
    await b.unload_all(ctx)
    st = await b.load_gguf(ctx, repo, variant, 2048)
    eff = b.effective_ctx(st)
    if eff != 2048:
        raise StepFailed(f"loaded with max_seq_length 2048, status reports context {eff}")
    await b.goto(ctx, "/chat")
    await ui.need(ctx.page.get_by_role("textbox", name="Message input"), "chat composer", 60_000)
    return {"model": repo, "variant": variant, "context_length": eff,
            "requested_context_length": st.get("requested_context_length")}


async def open_settings(ctx):
    await ui.open_run_settings(ctx.page)
    return {"context_field": await ui.field(ctx.page, "Context Length"),
            "temperature_field": await ui.field(ctx.page, "Temperature")}


async def reapply_4096(ctx):
    typed = await ui.set_field(ctx.page, "Context Length", 4096)
    await (await ui.need(ctx.page.get_by_role("button", name="Reload model"), "Reload model")).click()
    st = await b.poll(lambda: b.status(ctx),
                      lambda s: s.get("requested_context_length") == 4096 and not s.get("loading")
                      and s.get("loaded"), 300)
    eff = b.effective_ctx(st)
    if eff != 4096:
        raise StepFailed(f"re-applied 4096, effective context is {eff}")
    await ctx.page.wait_for_timeout(800)
    return {"typed": typed, "context_length": eff, "field_after": await ui.field(ctx.page, "Context Length")}


async def custom_sampling(ctx):
    t = await ui.set_field(ctx.page, "Temperature", "0.3")
    p = await ui.set_field(ctx.page, "Top P", "0.9")
    seed = await ui.set_field(ctx.page, "Seed", "3407")  # fixed seed: the reply is comparable
    sp = await ui.need(ctx.page.get_by_role("textbox", name="System prompt"), "System prompt")
    await sp.fill(SYSTEM)
    if t not in ("0.3", "0.30") or p not in ("0.9", "0.90"):
        raise StepFailed(f"fields did not keep the values: temperature={t} top_p={p}")
    return {"temperature": t, "top_p": p, "seed": seed}


async def save_preset(ctx):
    name = await ui.need(ctx.page.get_by_role("textbox", name="Inference preset name"), "preset name")
    await name.fill(PRESET)
    await (await ui.need(ctx.page.get_by_role("button", name="Save current settings as"), "Save preset")).click()
    s = (await b.poll(lambda: b.get(ctx, "/api/chat/settings"),
                      lambda d: any(p.get("name") == PRESET
                                    for p in (d.get("settings") or {}).get("customPresets") or []),
                      30, 1.0))["settings"]
    pre = next(p for p in s["customPresets"] if p["name"] == PRESET)["params"]
    if (pre.get("temperature"), pre.get("topP"), pre.get("systemPrompt")) != (0.3, 0.9, SYSTEM):
        raise StepFailed(f"saved preset params differ: {pre}")
    return {"active_preset": s.get("activePreset"), "preset_params": {
        k: pre.get(k) for k in ("temperature", "topP", "systemPrompt", "maxSeqLength")}}


async def reload_persisted(ctx):
    await ctx.page.reload(wait_until="domcontentloaded")
    await ui.need(ctx.page.get_by_role("textbox", name="Message input"), "chat composer", 60_000)
    await ui.open_run_settings(ctx.page)
    t, p = await ui.field(ctx.page, "Temperature"), await ui.field(ctx.page, "Top P")
    sp = await ctx.page.get_by_role("textbox", name="System prompt").first.input_value()
    if (t, p, sp) != ("0.3", "0.9", SYSTEM):
        raise StepFailed(f"after reload: temperature={t} top_p={p} system={sp!r}")
    return {"temperature": t, "top_p": p, "system_prompt_kept": True}


async def short_reply(ctx):
    close = ctx.page.get_by_role("button", name="Close run settings")
    if await close.count():
        await close.first.click()
    n = await ctx.page.locator(ui.ASSISTANT).count()
    await ui.send(ctx.page, "Say hello in three words.")
    txt = await ui.wait_reply(ctx.page, n)
    return {"reply_nonempty": bool(txt), "assistant_rows": n + 1}


async def stop_generation(ctx):
    n = await ctx.page.locator(ui.ASSISTANT).count()
    await ui.send(ctx.page, "Count from 1 to 500, one number per line.")
    stop = ctx.page.locator(ui.STOP).first
    await ui.need(ctx.page.locator(ui.STOP), "Stop button", 60_000)
    await stop.click()
    await stop.wait_for(state="hidden", timeout=30_000)
    st = await b.status(ctx)
    if not st.get("loaded"):
        raise StepFailed("model unloaded after Stop")
    await ctx.page.wait_for_timeout(500)
    return {"stopped": True, "assistant_rows": await ctx.page.locator(ui.ASSISTANT).count(),
            "rows_before": n}



async def lost_load_reply(ctx):
    """Re-apply a new context while the load's HTTP reply is lost in transit. The request reaches
    the server and the load completes there; only the answer is cut (connection reset), which is
    what a page reload mid-load or a dropped keepalive body looks like to the client. A client
    that treats that unknown outcome as a failure rolls the server back to the previous context
    (unsloth#11729); one that does not keeps the context it asked for."""
    page = ctx.page
    before = b.effective_ctx(await b.status(ctx))
    target = 3072 if before != 3072 else 2560
    seen = {"n": 0, "cut": False}

    async def cut_first_reply(route):
        if seen["n"] == 0 and route.request.method == "POST":
            seen["n"] += 1
            try:
                await route.fetch(timeout=300_000)   # the server performs the load
            except Exception:
                pass
            await route.abort("connectionreset")
            seen["cut"] = True
        else:
            await route.continue_()

    await page.route("**/api/inference/load", cut_first_reply)
    try:
        await ui.open_run_settings(page)
        await ui.set_field(page, "Context Length", target)
        await (await ui.need(page.get_by_role("button", name="Reload model"), "Reload model")).click()
        await b.poll(lambda: b.status(ctx), lambda s: seen["cut"], 330)
        if not seen["cut"]:
            raise StepFailed("Reload model sent no /api/inference/load within 330s")
        # A rollback, if the client sends one, starts right after the failed reply: give it time
        # to start and finish before reading the outcome.
        await page.wait_for_timeout(8_000)
        st = await b.poll(lambda: b.status(ctx), lambda s: s.get("loaded") and not s.get("loading"), 300)
    finally:
        await page.unroute("**/api/inference/load")
    await page.wait_for_timeout(800)
    final = b.effective_ctx(st)
    return {"previous_context": before, "requested_context": target, "final_context": final,
            "rolled_back": final != target, "load_requests_cut": seen["n"]}


JOURNEY = Journey(
    name="inference_settings", tier="model", needs=("gguf_small",), routes=("/chat",),
    steps=(
        Step("load_2048", load_2048, timeout_s=400),
        Step("open_settings", open_settings),
        Step("reapply_4096", reapply_4096, timeout_s=400),
        Step("custom_sampling", custom_sampling),
        Step("save_preset", save_preset),
        Step("reload_persisted", reload_persisted),
        Step("short_reply", short_reply, masks=b.VOLATILE_MASKS, mask_reason="tok/s, elapsed",
             timeout_s=240),
        Step("stop_generation", stop_generation, masks=b.VOLATILE_MASKS + b.GENERATED_TEXT_MASKS,
             mask_reason="partial stream length varies", timeout_s=240),
        # Last: it may leave the two sides on different contexts on purpose.
        Step("lost_load_reply", lost_load_reply, masks=b.VOLATILE_MASKS + b.GENERATED_TEXT_MASKS,
             mask_reason="earlier replies", timeout_s=700),
    ),
)
