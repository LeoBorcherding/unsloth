import sys, json, time, statistics as st
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE, PW, OUT, TAG = sys.argv[1], Path(sys.argv[2]).read_text().strip(), Path(sys.argv[3]), sys.argv[4]
REFRESH = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
res = {"tag": TAG, "refresh_ms": REFRESH}
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--enable-gpu-rasterization", "--ignore-gpu-blocklist"])
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.goto(BASE + "/login"); page.get_by_label("Password", exact=False).first.fill(PW); page.keyboard.press("Enter"); page.wait_for_url("**/chat**")
    page.evaluate("""(ms)=>{localStorage.removeItem('unsloth_live_monitor_position');const s=JSON.parse(localStorage.getItem('unsloth_monitor_overlay')||'{"state":{},"version":0}');Object.assign(s.state,{isOpen:true,refreshMs:ms,showUnslothShare:true,showModelBreakdown:true,showDetails:true});localStorage.setItem('unsloth_monitor_overlay',JSON.stringify(s));}""", REFRESH)
    page.reload(); page.wait_for_timeout(5000)
    cdp = ctx.new_cdp_session(page); cdp.send("Performance.enable")
    # React commit counter on the monitor subtree: count DOM mutations as a proxy.
    page.evaluate("""()=>{window.__mut=0;const m=document.querySelector('[data-testid=floating-monitor]');new MutationObserver(r=>{window.__mut+=r.length}).observe(m,{subtree:true,childList:true,characterData:true,attributes:true});
      window.__lt=[];new PerformanceObserver(l=>{for(const e of l.getEntries())window.__lt.push(e.duration)}).observe({type:'longtask',buffered:false});}""")
    # 1) idle polling cost over 30 s
    reqs = []
    page.on("requestfinished", lambda r: reqs.append(r) if "/api/" in r.url else None)
    m0 = {x["name"]: x["value"] for x in cdp.send("Performance.getMetrics")["metrics"]}
    t0 = time.time(); page.wait_for_timeout(30000); dt = time.time() - t0
    m1 = {x["name"]: x["value"] for x in cdp.send("Performance.getMetrics")["metrics"]}
    by = {}
    for r in reqs:
        path = r.url.split("?")[0].split(BASE)[-1]
        t = r.timing; sz = r.sizes()
        e = by.setdefault(path, {"count": 0, "ms": [], "bytes": 0})
        e["count"] += 1; e["ms"].append(t["responseEnd"]); e["bytes"] += sz["responseBodySize"] + sz["responseHeadersSize"]
    res["idle_30s"] = {
        "seconds": round(dt, 1),
        "requests": {k: {"count": v["count"], "per_min": round(v["count"] * 60 / dt, 1), "median_ms": round(st.median(v["ms"]), 1), "bytes_per_min": int(v["bytes"] * 60 / dt)} for k, v in by.items()},
        "script_ms": round((m1["ScriptDuration"] - m0["ScriptDuration"]) * 1000, 1),
        "layout_ms": round((m1["LayoutDuration"] - m0["LayoutDuration"]) * 1000, 1),
        "style_ms": round((m1["RecalcStyleDuration"] - m0["RecalcStyleDuration"]) * 1000, 1),
        "layouts": int(m1["LayoutCount"] - m0["LayoutCount"]),
        "dom_mutations": page.evaluate("window.__mut"),
        "long_tasks": page.evaluate("window.__lt.length"),
    }
    # 2) fast left-right drag from the card background, rAF intervals
    page.evaluate("window.__mut=0;window.__lt=[]")
    box = page.get_by_test_id("floating-monitor-cpu").bounding_box()
    x, y = box["x"] + 30, box["y"] + 8
    page.evaluate("""()=>{window.__f=[];let last=performance.now();const tick=t=>{window.__f.push(t-last);last=t;if(window.__run)requestAnimationFrame(tick)};window.__run=true;requestAnimationFrame(tick)}""")
    m0 = {x_["name"]: x_["value"] for x_ in cdp.send("Performance.getMetrics")["metrics"]}
    cdp.send("Tracing.start", {"categories": "devtools.timeline,blink.user_timing,v8.execute", "transferMode": "ReturnAsStream"})
    page.mouse.move(x, y); page.mouse.down()
    for rep in range(6):
        for i in range(40):
            page.mouse.move(x - (i * 15 if rep % 2 == 0 else (39 - i) * 15), y)
    page.mouse.up()
    page.wait_for_timeout(300)
    page.evaluate("window.__run=false")
    m1 = {x_["name"]: x_["value"] for x_ in cdp.send("Performance.getMetrics")["metrics"]}
    done = {}
    cdp.on("Tracing.tracingComplete", lambda e: done.update(e))
    cdp.send("Tracing.end")
    for _ in range(100):
        if done: break
        page.wait_for_timeout(100)
    if done.get("stream"):
        chunks = []
        while True:
            r = cdp.send("IO.read", {"handle": done["stream"]}); chunks.append(r["data"])
            if r.get("eof"): break
        (OUT / f"{TAG}-drag-trace.json").write_text("".join(chunks))
    f = [v for v in page.evaluate("window.__f")[2:]]
    f.sort()
    res["drag"] = {
        "frames": len(f), "median_frame_ms": round(st.median(f), 2), "p95_frame_ms": round(f[int(len(f) * .95) - 1], 2), "max_frame_ms": round(f[-1], 2),
        "frames_over_20ms": sum(1 for v in f if v > 20),
        "script_ms": round((m1["ScriptDuration"] - m0["ScriptDuration"]) * 1000, 1),
        "layout_ms": round((m1["LayoutDuration"] - m0["LayoutDuration"]) * 1000, 1),
        "style_ms": round((m1["RecalcStyleDuration"] - m0["RecalcStyleDuration"]) * 1000, 1),
        "layouts": int(m1["LayoutCount"] - m0["LayoutCount"]),
        "style_recalcs": int(m1["RecalcStyleCount"] - m0["RecalcStyleCount"]),
        "dom_mutations": page.evaluate("window.__mut"),
        "long_tasks": page.evaluate("window.__lt"),
    }
    b.close()
(OUT / f"{TAG}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
