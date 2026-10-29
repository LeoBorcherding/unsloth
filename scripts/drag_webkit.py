import sys, json, statistics as st
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE, PW, OUT, TAG = sys.argv[1], Path(sys.argv[2]).read_text().strip(), Path(sys.argv[3]), sys.argv[4]
with sync_playwright() as p:
    b = p.webkit.launch(headless=True); page = b.new_page(viewport={"width": 1440, "height": 900})
    page.goto(BASE + "/login"); page.get_by_label("Password", exact=False).first.fill(PW); page.keyboard.press("Enter"); page.wait_for_url("**/chat**")
    page.evaluate("""()=>{localStorage.removeItem('unsloth_live_monitor_position');const s=JSON.parse(localStorage.getItem('unsloth_monitor_overlay')||'{"state":{},"version":0}');Object.assign(s.state,{isOpen:true,refreshMs:1000,showUnslothShare:true,showModelBreakdown:true,showDetails:true});localStorage.setItem('unsloth_monitor_overlay',JSON.stringify(s));}""")
    page.reload(); page.wait_for_timeout(5000)
    box = page.get_by_test_id("floating-monitor-cpu").bounding_box(); x, y = box["x"] + 30, box["y"] + 8
    # Pointer-to-paint lag: on each frame, how far the card's painted x is from the last pointer x.
    page.evaluate("""()=>{window.__f=[];window.__lag=[];window.__px=null;let last=performance.now();
      addEventListener('pointermove',e=>{window.__px=e.clientX},{capture:true});
      const m=document.querySelector('[data-testid=floating-monitor]');let x0=null,p0=null;
      const tick=t=>{window.__f.push(t-last);last=t;const r=m.getBoundingClientRect();
        if(window.__px!==null){if(x0===null){x0=r.left;p0=window.__px}window.__lag.push(Math.abs((window.__px-p0)-(r.left-x0)))}
        if(window.__run)requestAnimationFrame(tick)};window.__run=true;requestAnimationFrame(tick)}""")
    page.mouse.move(x, y); page.mouse.down()
    for rep in range(6):
        for i in range(40):
            page.mouse.move(x - (i * 15 if rep % 2 == 0 else (39 - i) * 15), y)
            page.wait_for_timeout(8)
    page.mouse.up(); page.wait_for_timeout(300); page.evaluate("window.__run=false")
    f = sorted(page.evaluate("window.__f")[2:]); lag = sorted(page.evaluate("window.__lag"))
    res = {"tag": TAG, "engine": "webkit (headless)", "frames": len(f), "median_frame_ms": round(st.median(f), 2),
           "p95_frame_ms": round(f[int(len(f) * .95) - 1], 2), "max_frame_ms": round(f[-1], 2), "frames_over_20ms": sum(1 for v in f if v > 20),
           "pointer_to_card_px_median": round(st.median(lag), 1) if lag else None, "pointer_to_card_px_p95": round(lag[int(len(lag) * .95) - 1], 1) if lag else None}
    b.close()
(OUT / f"{TAG}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res))
