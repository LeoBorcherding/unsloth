import sys, json, time
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE, PW, OUT, TAG = sys.argv[1], Path(sys.argv[2]).read_text().strip(), Path(sys.argv[3]), sys.argv[4]
OUT.mkdir(parents=True, exist_ok=True)
console = []
def state(page, **kw):
    page.evaluate("""(kw) => { const s = JSON.parse(localStorage.getItem('unsloth_monitor_overlay') || '{"state":{},"version":0}');
      Object.assign(s.state, {isOpen: true, refreshMs: 1000}, kw); localStorage.setItem('unsloth_monitor_overlay', JSON.stringify(s));
      localStorage.removeItem('unsloth_live_monitor_width'); localStorage.removeItem('unsloth_live_monitor_position'); }""", kw)
    page.reload(); page.wait_for_timeout(8000)
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=3, color_scheme="dark",
                        record_har_path=str(OUT / f"{TAG}-network.har"))
    page = ctx.new_page()
    page.on("console", lambda m: console.append({"t": time.time(), "type": m.type, "text": m.text}))
    page.goto(BASE + "/login")
    page.get_by_label("Password", exact=False).first.fill(PW); page.keyboard.press("Enter")
    page.wait_for_url("**/chat**", timeout=30000)
    mon = lambda: page.get_by_test_id("floating-monitor")
    off = dict(showUnslothShare=False, showModelBreakdown=False, showDetails=False)
    shots = [("1-simple", off),
             ("2-detailed", dict(off, showDetails=True)),
             ("3-unsloth-share", dict(off, showUnslothShare=True)),
             ("4-share-and-per-model", dict(off, showUnslothShare=True, showModelBreakdown=True)),
             ("5-everything", dict(showUnslothShare=True, showModelBreakdown=True, showDetails=True))]
    for name, kw in shots:
        state(page, **kw)
        mon().screenshot(path=str(OUT / f"{TAG}-{name}.png"))
        (OUT / f"{TAG}-{name}.txt").write_text(mon().inner_text())
    later = page.get_by_role("button", name="Later")
    if later.count(): later.first.dispatch_event("click"); page.wait_for_timeout(500)
    page.screenshot(path=str(OUT / f"{TAG}-6-full-page.png"))
    state(page, isOpen=False, showUnslothShare=True, showModelBreakdown=True)
    page.get_by_role("button", name="Settings").first.click(); page.wait_for_timeout(1500)
    page.get_by_text("System", exact=True).first.click(); page.wait_for_timeout(8000)
    page.screenshot(path=str(OUT / f"{TAG}-7-settings-top.png"))
    page.get_by_text("Show Unsloth's share").first.scroll_into_view_if_needed(); page.wait_for_timeout(1000)
    page.screenshot(path=str(OUT / f"{TAG}-8-settings-share.png"))
    page.get_by_text("Refresh rate").first.scroll_into_view_if_needed(); page.wait_for_timeout(1000)
    page.screenshot(path=str(OUT / f"{TAG}-9-settings-floating-window.png"))
    ctx.close(); b.close()
(OUT / f"{TAG}-console.json").write_text(json.dumps(console, indent=1))
print("ok", len(console), "console lines")
