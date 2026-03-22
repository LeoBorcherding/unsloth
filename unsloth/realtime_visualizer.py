# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Real-time neuron tracer — Studio integration
============================================

Provides a live web dashboard that shows activation heatmaps, gradient norms
and LoRA growth *during* training, updating after every captured step via
Server-Sent Events (SSE).  No JS framework, no websockets — just a background
HTTP thread + EventSource in the browser.

Usage::

    from unsloth.activation_capture import ActivationCaptureConfig, ActivationCapture
    from unsloth.realtime_visualizer import RealtimeActivationCallback, NeuronTracerServer

    capture_cfg = ActivationCaptureConfig(
        output_dir="run/activations",
        capture_interval=5,
    )
    capture = ActivationCapture(model, capture_cfg)

    # Drop-in replacement for ActivationCaptureCallback:
    callback = RealtimeActivationCallback(capture, port=7863, auto_open=True)

    trainer = SFTTrainer(model=model, ..., callbacks=[callback])
    trainer.train()
    # Dashboard stays live at http://127.0.0.1:7863 until the process exits.
"""

__all__ = [
    "NeuronTracerServer",
    "RealtimeActivationCallback",
]

import json
import logging
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Optional
from urllib.parse import urlparse

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .activation_capture import ActivationCapture, ActivationCaptureCallback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML template — dark Studio theme, live SSE feed
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🦥 Unsloth — Neuron Tracer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:         #07070c;
  --bg-card:    #0d0d14;
  --bg-input:   #11111a;
  --border:     rgba(255,255,255,0.07);
  --border-hi:  rgba(99,102,241,0.5);
  --text:       #e8e8f0;
  --text2:      #888898;
  --text3:      #484858;
  --blue:       #6366f1;
  --blue-dim:   rgba(99,102,241,0.12);
  --cyan:       #22d3ee;
  --green:      #10b981;
  --green-dim:  rgba(16,185,129,0.12);
  --purple:     #a855f7;
  --purple-dim: rgba(168,85,247,0.12);
  --orange:     #f59e0b;
  --red:        #ef4444;
  --radius:     12px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', system-ui, sans-serif;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  overflow-x: hidden;
}
a { color: var(--blue); text-decoration: none; }
/* ── top bar ── */
.topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 20px;
  height: 52px;
  background: var(--bg-card);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  position: sticky;
  top: 0;
  z-index: 100;
}
.brand { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 1rem; letter-spacing: -0.02em; }
.brand-emoji { font-size: 1.25rem; }
.brand-sep { color: var(--text3); font-weight: 300; font-size: 1.1rem; }
.brand-sub { font-weight: 500; color: var(--text2); font-size: 0.9rem; }
.topbar-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px;
  border-radius: 20px;
  border: 1px solid var(--border);
  font-size: 0.7rem;
  font-family: 'JetBrains Mono', monospace;
  background: var(--bg-input);
  color: var(--text2);
}
.badge .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--text3);
  transition: background 0.3s;
}
.badge.live .dot { background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse-dot 1.5s ease-in-out infinite; }
.badge.done .dot { background: var(--blue); box-shadow: 0 0 6px var(--blue); }
.badge.error .dot { background: var(--red); }
@keyframes pulse-dot { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
/* ── layout ── */
.main {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 280px;
  grid-template-rows: 1fr;
  gap: 0;
  overflow: hidden;
  height: calc(100vh - 52px);
}
/* ── viz column ── */
.viz-col {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-right: 1px solid var(--border);
}
.panel-tabs {
  display: flex;
  gap: 0;
  background: var(--bg-card);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  padding: 0 16px;
}
.tab-btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 12px 14px;
  border: none;
  background: transparent;
  color: var(--text3);
  font-family: inherit;
  font-size: 0.78rem;
  font-weight: 500;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.15s;
  margin-bottom: -1px;
}
.tab-btn:hover { color: var(--text2); }
.tab-btn.active { color: var(--text); border-bottom-color: var(--blue); }
.tab-btn .tab-icon {
  width: 18px; height: 18px; border-radius: 5px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.65rem;
}
.tab-btn .tab-icon.act { background: var(--blue-dim); color: var(--blue); }
.tab-btn .tab-icon.grad { background: var(--green-dim); color: var(--green); }
.tab-btn .tab-icon.lora { background: var(--purple-dim); color: var(--purple); }
.canvas-area {
  flex: 1;
  overflow: auto;
  padding: 16px;
  background: var(--bg);
  display: flex;
  align-items: flex-start;
  justify-content: center;
}
canvas { display: block; border-radius: 8px; }
/* ── controls bar ── */
.controls-bar {
  background: var(--bg-card);
  border-top: 1px solid var(--border);
  padding: 12px 20px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 14px;
}
.play-btn {
  width: 34px; height: 34px; border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--bg-input);
  color: var(--text);
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.15s;
  flex-shrink: 0;
}
.play-btn:hover { border-color: var(--blue); }
.play-btn.playing { background: var(--blue); border-color: var(--blue); }
.slider-wrap { flex: 1; display: flex; flex-direction: column; gap: 5px; }
.slider-track {
  position: relative; height: 6px;
  background: var(--bg-input);
  border-radius: 3px; overflow: visible;
}
.slider-fill {
  position: absolute; top: 0; left: 0; height: 100%;
  background: linear-gradient(90deg, var(--blue), var(--cyan));
  border-radius: 3px;
  pointer-events: none;
  transition: width 0.1s;
}
input[type=range] {
  width: 100%; height: 6px; -webkit-appearance: none;
  background: transparent; position: relative; z-index: 2; cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
  background: var(--text); cursor: pointer;
  box-shadow: 0 0 0 2px var(--blue), 0 2px 6px rgba(0,0,0,0.4);
  margin-top: -4px;
  transition: transform 0.1s;
}
input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.15); }
input[type=range]::-webkit-slider-runnable-track { height: 6px; background: transparent; }
.slider-meta {
  display: flex; justify-content: space-between;
  font-size: 0.68rem; font-family: 'JetBrains Mono', monospace;
}
.sl-step { color: var(--text); }
.sl-loss { color: var(--cyan); }
.sl-speed { color: var(--text3); }
.ctrl-btns { display: flex; gap: 6px; }
.ctrl-btn {
  padding: 5px 11px; border: 1px solid var(--border);
  background: transparent; color: var(--text2);
  font-size: 0.7rem; font-family: inherit; border-radius: 7px;
  cursor: pointer; transition: all 0.15s;
}
.ctrl-btn:hover { border-color: var(--border-hi); color: var(--text); }
.ctrl-btn.active { background: var(--blue-dim); border-color: var(--border-hi); color: var(--blue); }
/* ── side panel ── */
.side-panel {
  display: flex; flex-direction: column;
  background: var(--bg-card);
  overflow-y: auto;
}
.side-section {
  padding: 14px 16px;
  border-bottom: 1px solid var(--border);
}
.side-title {
  font-size: 0.65rem; font-weight: 600; color: var(--text3);
  text-transform: uppercase; letter-spacing: 0.07em;
  margin-bottom: 10px;
}
/* model info */
.model-row { display: flex; flex-direction: column; gap: 4px; }
.model-name { font-size: 0.85rem; font-weight: 600; color: var(--text); word-break: break-all; }
.model-meta { font-size: 0.7rem; color: var(--text2); font-family: 'JetBrains Mono', monospace; }
/* mini sparklines */
.metric-rows { display: flex; flex-direction: column; gap: 5px; }
.metric-row { display: flex; align-items: center; gap: 8px; }
.metric-lbl {
  font-size: 0.62rem; color: var(--text3);
  font-family: 'JetBrains Mono', monospace;
  width: 28px; flex-shrink: 0; text-align: right;
}
.spark-outer { flex: 1; height: 14px; background: var(--bg-input); border-radius: 3px; overflow: hidden; }
.spark-inner { height: 100%; border-radius: 3px; transition: width 0.15s; }
.spark-inner.grad { background: linear-gradient(90deg, #064e3b, var(--green)); }
.spark-inner.lora { background: linear-gradient(90deg, #3b0764, var(--purple)); }
.spark-val { font-size: 0.6rem; color: var(--text3); font-family: 'JetBrains Mono', monospace; width: 38px; text-align: right; }
/* loss chart */
.loss-canvas-wrap { border-radius: 6px; overflow: hidden; background: var(--bg); }
/* legend */
.legend-row { display: flex; align-items: center; gap: 7px; font-size: 0.68rem; color: var(--text2); margin-bottom: 5px; }
.legend-row:last-child { margin-bottom: 0; }
.legend-bar { width: 52px; height: 6px; border-radius: 3px; }
.legend-bar.act { background: linear-gradient(90deg,#1e3a5f,#3b82f6,#f59e0b,#ef4444); }
.legend-bar.grad { background: linear-gradient(90deg,var(--bg),#059669,var(--green)); }
.legend-bar.lora { background: linear-gradient(90deg,var(--bg),#7c3aed,var(--purple)); }
/* loading overlay */
#overlay {
  position: fixed; inset: 0; z-index: 200;
  background: rgba(7,7,12,0.92);
  backdrop-filter: blur(8px);
  display: flex; align-items: center; justify-content: center;
  flex-direction: column; gap: 16px;
}
#overlay.hidden { display: none; }
.spinner {
  width: 36px; height: 36px; border-radius: 50%;
  border: 3px solid var(--border);
  border-top-color: var(--blue);
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.overlay-text { font-size: 0.9rem; color: var(--text2); }
/* shortcuts */
.shortcuts { font-size: 0.65rem; color: var(--text3); text-align: center; padding: 8px 16px; }
.shortcuts kbd {
  display: inline-block; background: var(--bg-input);
  border: 1px solid var(--border); border-radius: 4px;
  padding: 1px 5px; font-family: 'JetBrains Mono', monospace;
  font-size: 0.6rem; margin: 0 1px;
}
/* diff mode banner */
#diff-banner {
  display: none; padding: 4px 12px;
  background: rgba(99,102,241,0.12);
  border-bottom: 1px solid rgba(99,102,241,0.25);
  font-size: 0.7rem; color: var(--blue); text-align: center;
}
#diff-banner.visible { display: block; }
/* responsive */
@media (max-width: 700px) {
  .main { grid-template-columns: 1fr; grid-template-rows: 1fr auto; }
  .side-panel { height: auto; overflow: visible; border-right: none; border-top: 1px solid var(--border); }
}
</style>
</head>
<body>

<div id="overlay">
  <div class="spinner"></div>
  <div class="overlay-text" id="overlay-text">Connecting to training run…</div>
</div>

<!-- ── top bar ─────────────────────────────────────────── -->
<div class="topbar">
  <div class="brand">
    <span class="brand-emoji">🦥</span>
    <span>Unsloth</span>
    <span class="brand-sep">|</span>
    <span class="brand-sub">Neuron Tracer</span>
  </div>
  <div class="topbar-right">
    <div class="badge" id="status-badge">
      <span class="dot"></span>
      <span id="status-text">connecting</span>
    </div>
    <div class="badge" id="model-badge" style="display:none">
      <span id="model-badge-text"></span>
    </div>
  </div>
</div>

<!-- ── main layout ─────────────────────────────────────── -->
<div class="main">

  <!-- viz column -->
  <div class="viz-col">
    <div id="diff-banner">⊕ Showing change relative to pre-training baseline (step 0)</div>
    <div class="panel-tabs">
      <button class="tab-btn active" data-tab="act" onclick="switchTab('act')">
        <span class="tab-icon act">⚡</span> Activations <span id="act-dims" style="color:var(--text3);font-size:0.65rem;margin-left:2px"></span>
      </button>
      <button class="tab-btn" data-tab="grad" id="tab-grad" style="display:none" onclick="switchTab('grad')">
        <span class="tab-icon grad">∇</span> Gradients
      </button>
      <button class="tab-btn" data-tab="lora" id="tab-lora" style="display:none" onclick="switchTab('lora')">
        <span class="tab-icon lora">◇</span> LoRA
      </button>
    </div>
    <div class="canvas-area" id="canvas-area">
      <canvas id="c-act"></canvas>
      <canvas id="c-grad" style="display:none"></canvas>
      <canvas id="c-lora" style="display:none"></canvas>
    </div>
    <div class="controls-bar">
      <button class="play-btn" id="btn-play" title="Play / Pause  (Space)">▶</button>
      <div class="slider-wrap">
        <div class="slider-track">
          <div class="slider-fill" id="slider-fill"></div>
          <input type="range" id="slider" min="0" value="0" max="0">
        </div>
        <div class="slider-meta">
          <span class="sl-step" id="sl-step">Step —</span>
          <span class="sl-loss" id="sl-loss"></span>
          <span class="sl-speed" id="sl-speed">1.0×</span>
        </div>
      </div>
      <div class="ctrl-btns">
        <button class="ctrl-btn" id="btn-diff" onclick="toggleDiff()">Absolute</button>
        <button class="ctrl-btn" id="btn-slower" onclick="changeSpeed(-1)">−</button>
        <button class="ctrl-btn" id="btn-faster" onclick="changeSpeed(+1)">+</button>
      </div>
    </div>
    <div class="shortcuts">
      <kbd>Space</kbd> Play/Pause &nbsp; <kbd>←</kbd><kbd>→</kbd> Step &nbsp; <kbd>D</kbd> Diff &nbsp; <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> Tabs
    </div>
  </div>

  <!-- side panel -->
  <div class="side-panel">
    <!-- model info -->
    <div class="side-section">
      <div class="side-title">Model</div>
      <div class="model-row">
        <div class="model-name" id="sp-model">—</div>
        <div class="model-meta" id="sp-meta">Loading…</div>
      </div>
    </div>

    <!-- grad norms -->
    <div class="side-section" id="sp-grad-section" style="display:none">
      <div class="side-title">Gradient Norms <span style="float:right;font-size:0.6rem;color:var(--text3)" id="sp-grad-step"></span></div>
      <div class="metric-rows" id="sp-grad-rows"></div>
    </div>

    <!-- lora norms -->
    <div class="side-section" id="sp-lora-section" style="display:none">
      <div class="side-title">LoRA Growth  <span style="float:right;font-size:0.6rem;color:var(--text3)">||B·A||<sub>F</sub></span></div>
      <div class="metric-rows" id="sp-lora-rows"></div>
    </div>

    <!-- live loss -->
    <div class="side-section" id="sp-loss-section">
      <div class="side-title">Loss</div>
      <div class="loss-canvas-wrap"><canvas id="c-loss" width="248" height="80"></canvas></div>
    </div>

    <!-- legend -->
    <div class="side-section">
      <div class="side-title">Legend</div>
      <div class="legend-row">
        <span style="color:var(--text2);font-size:0.7rem;width:52px">Activation</span>
        <span style="font-size:0.62rem;color:var(--text3)">low</span>
        <div class="legend-bar act"></div>
        <span style="font-size:0.62rem;color:var(--text3)">high</span>
      </div>
      <div class="legend-row" id="leg-grad" style="display:none">
        <span style="color:var(--text2);font-size:0.7rem;width:52px">Gradient</span>
        <span style="font-size:0.62rem;color:var(--text3)">idle</span>
        <div class="legend-bar grad"></div>
        <span style="font-size:0.62rem;color:var(--text3)">active</span>
      </div>
      <div class="legend-row" id="leg-lora" style="display:none">
        <span style="color:var(--text2);font-size:0.7rem;width:52px">LoRA</span>
        <span style="font-size:0.62rem;color:var(--text3)">zero</span>
        <div class="legend-bar lora"></div>
        <span style="font-size:0.62rem;color:var(--text3)">high</span>
      </div>
    </div>
  </div><!-- /side-panel -->

</div><!-- /main -->

<script>
// ============================================================
//  State
// ============================================================
let META = null, RECORDS = [];
let curIdx = 0, playing = false, frameMs = 300;
let showDiff = false, activeTab = 'act';
let globalMax = 1e-9, globalGradMax = 1e-9, globalLoraMax = 1e-9;
let isLive = false, trainingDone = false;

// ============================================================
//  SSE Connection
// ============================================================
const sse = new EventSource('/stream');

sse.addEventListener('meta', e => {
  META = JSON.parse(e.data);
  initViz();
  setStatus('live', 'live');
  document.getElementById('overlay').classList.add('hidden');
});

sse.addEventListener('record', e => {
  const rec = JSON.parse(e.data);
  RECORDS.push(rec);
  updateGlobalMaxes(rec);
  // update side panel live stats
  updateSidePanel(rec);
  updateLossChart();
  // if following live (at the end) advance
  if (playing || curIdx === RECORDS.length - 2) {
    setIdx(RECORDS.length - 1);
  } else {
    slider.max = RECORDS.length - 1;
    updateSliderFill();
  }
});

sse.addEventListener('done', () => {
  trainingDone = true;
  isLive = false;
  setStatus('done', `done · ${RECORDS.length} steps`);
  stopPlay();
});

sse.onerror = () => {
  if (!trainingDone) setStatus('error', 'disconnected');
};

function setStatus(cls, text) {
  const badge = document.getElementById('status-badge');
  badge.className = 'badge ' + cls;
  document.getElementById('status-text').textContent = text;
}

// ============================================================
//  Colour helpers
// ============================================================
function lerp(a,b,t) { return a+(b-a)*t; }

function heatColor(v) {
  let r,g,b;
  if (v < 0.33) {
    const t = v*3;
    r = Math.round(lerp(0x1a,0x22,t)); g = Math.round(lerp(0x2a,0xd3,t)); b = Math.round(lerp(0x6c,0xee,t));
  } else if (v < 0.66) {
    const t = (v-0.33)*3;
    r = Math.round(lerp(0x22,0xf5,t)); g = Math.round(lerp(0xd3,0x9e,t)); b = Math.round(lerp(0xee,0x0b,t));
  } else {
    const t = (v-0.66)*3;
    r = Math.round(lerp(0xf5,0xef,t)); g = Math.round(lerp(0x9e,0x44,t)); b = Math.round(lerp(0x0b,0x44,t));
  }
  return `rgb(${r},${g},${b})`;
}
function gradColor(v) {
  return `rgb(${Math.round(lerp(0x06,0x10,v))},${Math.round(lerp(0x1a,0xb9,v))},${Math.round(lerp(0x10,0x81,v))})`;
}
function loraColor(v) {
  return `rgb(${Math.round(lerp(0x0a,0xa8,v))},${Math.round(lerp(0x06,0x55,v))},${Math.round(lerp(0x14,0xf7,v))})`;
}

// ============================================================
//  Global max trackers
// ============================================================
function updateGlobalMaxes(rec) {
  if (META) {
    for (let l=0;l<META.num_layers;l++) {
      const vals = rec.layers[String(l)];
      if (!vals) continue;
      for (const v of vals.mean_abs) if (v>globalMax) globalMax=v;
    }
  }
  if (rec.grad_norms) for (const v of Object.values(rec.grad_norms)) if (v>globalGradMax) globalGradMax=v;
  if (rec.lora_norms) for (const v of Object.values(rec.lora_norms)) if (v>globalLoraMax) globalLoraMax=v;
}

// ============================================================
//  Init after META arrives
// ============================================================
const ACT_CELL=11, ACT_GAP=2, ACT_LGAP=5, ACT_PAD=44, ACT_LBLW=40, ACT_TOPP=36;
let colW, rowH, gridW, gridH, cAct, ctxAct, cGrad, ctxGrad, cLora, ctxLora;
const GRAD_BAR_H=28, LORA_ROW_H=24;

function initViz() {
  const N=META.num_layers, C=META.captured_channels.length;
  document.getElementById('act-dims').textContent = `${N}L×${C}ch`;

  // model badge
  const mname = (META.model_name||'unknown').split('/').pop();
  document.getElementById('model-badge-text').textContent = mname;
  document.getElementById('model-badge').style.display = 'inline-flex';
  document.getElementById('sp-model').textContent = META.model_name || 'unknown';
  document.getElementById('sp-meta').textContent =
    `${N} layers · ${C} channels · hidden ${META.hidden_size||'?'}`;

  // show tabs / legend rows for available data
  if (META.capture_gradients) {
    document.getElementById('tab-grad').style.display='inline-flex';
    document.getElementById('leg-grad').style.display='flex';
    document.getElementById('sp-grad-section').style.display='block';
    buildGradRows(N);
  }
  if (META.capture_lora_norms && (META.lora_targets||[]).length>0) {
    document.getElementById('tab-lora').style.display='inline-flex';
    document.getElementById('leg-lora').style.display='flex';
    document.getElementById('sp-lora-section').style.display='block';
    buildLoraRows(META.lora_targets);
  }

  // canvas sizes
  colW  = ACT_CELL+ACT_GAP;
  rowH  = ACT_CELL+ACT_GAP;
  gridW = N*colW + (N-1)*(ACT_LGAP-ACT_GAP) + ACT_LBLW;
  gridH = C*rowH;

  cAct = document.getElementById('c-act');
  ctxAct = cAct.getContext('2d');
  cAct.width  = gridW + ACT_PAD*2;
  cAct.height = gridH + ACT_PAD + ACT_TOPP;

  cGrad = document.getElementById('c-grad');
  ctxGrad = cGrad.getContext('2d');
  cGrad.width  = gridW + ACT_PAD*2;
  cGrad.height = GRAD_BAR_H + ACT_TOPP + ACT_PAD + 20;

  const loraTargets = META.lora_targets||[];
  cLora = document.getElementById('c-lora');
  ctxLora = cLora.getContext('2d');
  cLora.width  = gridW + ACT_PAD*2;
  cLora.height = loraTargets.length*(LORA_ROW_H+4) + ACT_TOPP + ACT_PAD;

  isLive = true;
}

// ============================================================
//  Side-panel spark rows
// ============================================================
function buildGradRows(nLayers) {
  const container = document.getElementById('sp-grad-rows');
  container.innerHTML = '';
  // show every 4th layer to save space
  const step = Math.max(1, Math.floor(nLayers/8));
  for (let l=0;l<nLayers;l+=step) {
    const row = document.createElement('div');
    row.className='metric-row';
    row.innerHTML=`
      <span class="metric-lbl">L${l}</span>
      <div class="spark-outer"><div class="spark-inner grad" id="sg-${l}" style="width:0%"></div></div>
      <span class="spark-val" id="sgv-${l}">—</span>`;
    container.appendChild(row);
  }
}
function buildLoraRows(targets) {
  const container = document.getElementById('sp-lora-rows');
  container.innerHTML = '';
  for (const t of targets) {
    const row = document.createElement('div');
    row.className='metric-row';
    const lbl = t.replace('_proj','');
    row.innerHTML=`
      <span class="metric-lbl">${lbl}</span>
      <div class="spark-outer"><div class="spark-inner lora" id="sl-${t}" style="width:0%"></div></div>
      <span class="spark-val" id="slv-${t}">—</span>`;
    container.appendChild(row);
  }
}

function updateSidePanel(rec) {
  if (!META) return;
  const N=META.num_layers;
  const step=Math.max(1,Math.floor(N/8));
  // gradients
  if (rec.grad_norms) {
    for (let l=0;l<N;l+=step) {
      const v=rec.grad_norms[String(l)]||0;
      const norm=Math.min(1,v/globalGradMax);
      const el=document.getElementById(`sg-${l}`);
      const elv=document.getElementById(`sgv-${l}`);
      if(el) el.style.width=`${norm*100}%`;
      if(elv) elv.textContent=v.toFixed(2);
    }
    document.getElementById('sp-grad-step').textContent=`step ${rec.step}`;
  }
  // lora
  if (rec.lora_norms && META.lora_targets) {
    for (const t of META.lora_targets) {
      // aggregate across all layers for this target in the side panel
      let sum=0,cnt=0;
      for (const k of Object.keys(rec.lora_norms)) {
        if(k.endsWith('.'+t)){ sum+=rec.lora_norms[k]; cnt++; }
      }
      const avg=cnt>0?sum/cnt:0;
      const norm=Math.min(1,avg/globalLoraMax);
      const el=document.getElementById(`sl-${t}`);
      const elv=document.getElementById(`slv-${t}`);
      if(el) el.style.width=`${norm*100}%`;
      if(elv) elv.textContent=avg.toFixed(3);
    }
  }
}

// ============================================================
//  Loss mini-chart
// ============================================================
const cLoss = document.getElementById('c-loss');
const ctxLoss = cLoss.getContext('2d');
function updateLossChart() {
  const W=cLoss.width, H=cLoss.height;
  ctxLoss.clearRect(0,0,W,H);
  ctxLoss.fillStyle='#0a0a0f';
  ctxLoss.fillRect(0,0,W,H);

  const pts = RECORDS.filter(r=>r.loss!=null);
  if (pts.length<2) return;
  const maxL = Math.max(...pts.map(r=>r.loss));
  const minL = Math.min(...pts.map(r=>r.loss));
  const range = maxL-minL||1;
  const pad=8;

  ctxLoss.strokeStyle='rgba(34,211,238,0.2)';
  ctxLoss.lineWidth=1;
  for (let i=1;i<4;i++) {
    const y=pad+(H-pad*2)*i/3;
    ctxLoss.beginPath(); ctxLoss.moveTo(pad,y); ctxLoss.lineTo(W-pad,y); ctxLoss.stroke();
  }

  // gradient fill
  const grad=ctxLoss.createLinearGradient(0,pad,0,H-pad);
  grad.addColorStop(0,'rgba(34,211,238,0.25)');
  grad.addColorStop(1,'rgba(34,211,238,0)');
  ctxLoss.beginPath();
  for (let i=0;i<pts.length;i++) {
    const x=pad+(W-pad*2)*i/(pts.length-1);
    const y=pad+(H-pad*2)*(1-(pts[i].loss-minL)/range);
    if(i===0) ctxLoss.moveTo(x,y); else ctxLoss.lineTo(x,y);
  }
  ctxLoss.lineTo(pad+(W-pad*2),H-pad);
  ctxLoss.lineTo(pad,H-pad);
  ctxLoss.closePath();
  ctxLoss.fillStyle=grad;
  ctxLoss.fill();

  // line
  ctxLoss.beginPath();
  ctxLoss.strokeStyle='#22d3ee';
  ctxLoss.lineWidth=1.5;
  for (let i=0;i<pts.length;i++) {
    const x=pad+(W-pad*2)*i/(pts.length-1);
    const y=pad+(H-pad*2)*(1-(pts[i].loss-minL)/range);
    if(i===0) ctxLoss.moveTo(x,y); else ctxLoss.lineTo(x,y);
  }
  ctxLoss.stroke();
  // current dot
  const last=pts[pts.length-1];
  const lx=pad+(W-pad*2)*(pts.length-1)/(pts.length-1);
  const ly=pad+(H-pad*2)*(1-(last.loss-minL)/range);
  ctxLoss.beginPath();
  ctxLoss.arc(lx,ly,3.5,0,Math.PI*2);
  ctxLoss.fillStyle='#22d3ee';
  ctxLoss.fill();
}

// ============================================================
//  Drawing
// ============================================================
function getActVals(rec, l) {
  const d = rec.layers[String(l)];
  return d ? d.mean_abs : null;
}

function normAct(rec, l) {
  const vals = getActVals(rec, l);
  if (!vals) return null;
  if (!showDiff) return vals.map(v=>v/globalMax);
  const base = getActVals(RECORDS[0], l);
  return vals.map((v,i)=>Math.max(0,Math.min(1,(v-(base?base[i]:0))/globalMax*0.5+0.5)));
}

function drawAct(idx) {
  if (!META||!ctxAct) return;
  const N=META.num_layers, C=META.captured_channels.length;
  const rec=RECORDS[idx];
  const W=cAct.width, H=cAct.height;
  ctxAct.fillStyle='#0a0a0f'; ctxAct.fillRect(0,0,W,H);

  // layer headers (every 4)
  ctxAct.font='500 9px "JetBrains Mono",monospace';
  ctxAct.fillStyle='#6366f1';
  for (let l=0;l<N;l+=4) {
    const x=ACT_PAD+ACT_LBLW+l*(colW+ACT_LGAP-ACT_GAP);
    ctxAct.fillText(l,x+ACT_CELL/2-4,ACT_PAD+12);
  }
  // channel labels (every 8)
  ctxAct.fillStyle='#5c5c6e';
  ctxAct.font='9px "JetBrains Mono",monospace';
  for (let c=0;c<C;c+=8) {
    const y=ACT_TOPP+ACT_PAD+c*rowH+ACT_CELL-1;
    ctxAct.fillText(META.captured_channels[c],8,y);
  }
  // cells
  for (let l=0;l<N;l++) {
    const x0=ACT_PAD+ACT_LBLW+l*(colW+ACT_LGAP-ACT_GAP);
    const nv=normAct(rec,l);
    for (let c=0;c<C;c++) {
      const v=nv?nv[c]:0;
      ctxAct.fillStyle=heatColor(v);
      ctxAct.beginPath();
      ctxAct.roundRect(x0,ACT_TOPP+ACT_PAD+c*rowH,ACT_CELL,ACT_CELL,2);
      ctxAct.fill();
    }
  }
}

function drawGrad(idx) {
  if (!META||!ctxGrad) return;
  const N=META.num_layers, rec=RECORDS[idx];
  const W=cGrad.width;
  ctxGrad.fillStyle='#0a0a0f'; ctxGrad.fillRect(0,0,W,cGrad.height);

  const barW=(W-ACT_PAD*2-ACT_LBLW)/N-2;
  for (let l=0;l<N;l++) {
    const gv=rec.grad_norms?(rec.grad_norms[String(l)]||0):0;
    const norm=Math.min(1,gv/globalGradMax);
    const x=ACT_PAD+ACT_LBLW+l*(barW+2);
    const bh=GRAD_BAR_H*norm;
    const y=ACT_TOPP+GRAD_BAR_H-bh;
    if(norm>0.5){ ctxGrad.shadowColor='#10b981'; ctxGrad.shadowBlur=norm*8; }
    ctxGrad.fillStyle=gradColor(norm);
    ctxGrad.beginPath(); ctxGrad.roundRect(x,y,barW-1,Math.max(bh,2),2); ctxGrad.fill();
    ctxGrad.shadowBlur=0;
    if(l%4===0){
      ctxGrad.fillStyle='#5c5c6e'; ctxGrad.font='9px "JetBrains Mono",monospace';
      ctxGrad.fillText(l,x+barW/2-4,ACT_TOPP+GRAD_BAR_H+14);
    }
  }
}

function drawLora(idx) {
  if (!META||!ctxLora) return;
  const loraTargets=META.lora_targets||[];
  if (!loraTargets.length) return;
  const N=META.num_layers, rec=RECORDS[idx];
  const W=cLora.width;
  ctxLora.fillStyle='#0a0a0f'; ctxLora.fillRect(0,0,W,cLora.height);
  const barW=(W-ACT_PAD*2-ACT_LBLW)/N-2;
  for (let t=0;t<loraTargets.length;t++) {
    const target=loraTargets[t];
    const rowY=ACT_TOPP+t*(LORA_ROW_H+4);
    ctxLora.fillStyle='#a855f7';
    ctxLora.font='500 9px "JetBrains Mono",monospace';
    ctxLora.fillText(target.replace('_proj',''),8,rowY+LORA_ROW_H/2+3);
    for (let l=0;l<N;l++) {
      const key=`${l}.${target}`;
      const lv=rec.lora_norms?(rec.lora_norms[key]||0):0;
      const norm=Math.min(1,lv/globalLoraMax);
      const x=ACT_PAD+ACT_LBLW+l*(barW+2);
      const bh=(LORA_ROW_H-4)*norm;
      const y=rowY+(LORA_ROW_H-4)-bh;
      if(norm>0.5){ ctxLora.shadowColor='#a855f7'; ctxLora.shadowBlur=norm*6; }
      ctxLora.fillStyle=loraColor(norm);
      ctxLora.beginPath(); ctxLora.roundRect(x,y,barW-1,Math.max(bh,1),2); ctxLora.fill();
      ctxLora.shadowBlur=0;
    }
  }
}

function draw(idx) {
  if (!META||!RECORDS.length) return;
  const rec=RECORDS[idx];
  drawAct(idx);
  drawGrad(idx);
  drawLora(idx);
  document.getElementById('sl-step').textContent=`Step ${rec.step}`;
  document.getElementById('sl-loss').textContent=rec.loss!=null?`loss ${rec.loss.toFixed(4)}`:'';
  updateSliderFill();
}

function updateSliderFill() {
  const pct=RECORDS.length>1?(curIdx/(RECORDS.length-1))*100:0;
  document.getElementById('slider-fill').style.width=`${pct}%`;
}

// ============================================================
//  Tab switching
// ============================================================
function switchTab(tab) {
  activeTab=tab;
  ['act','grad','lora'].forEach(t=>{
    document.querySelector(`[data-tab=${t}]`).classList.toggle('active',t===tab);
    document.getElementById(`c-${t}`).style.display=t===tab?'block':'none';
  });
}

// ============================================================
//  Playback
// ============================================================
let timer=null;

function setIdx(i) {
  curIdx=Math.max(0,Math.min(RECORDS.length-1,i));
  slider.value=curIdx;
  draw(curIdx);
}

function tick() {
  if(!playing) return;
  setIdx(curIdx+1);
  if(curIdx>=RECORDS.length-1&&trainingDone){ stopPlay(); return; }
  timer=setTimeout(tick,frameMs);
}
function startPlay() {
  playing=true;
  document.getElementById('btn-play').classList.add('playing');
  document.getElementById('btn-play').textContent='⏸';
  timer=setTimeout(tick,frameMs);
}
function stopPlay() {
  playing=false;
  clearTimeout(timer);
  document.getElementById('btn-play').classList.remove('playing');
  document.getElementById('btn-play').textContent='▶';
}
function togglePlay() {
  if(playing) stopPlay();
  else { if(curIdx>=RECORDS.length-1&&trainingDone) setIdx(0); startPlay(); }
}

document.getElementById('btn-play').onclick=togglePlay;
const slider=document.getElementById('slider');
slider.oninput=()=>{ stopPlay(); setIdx(+slider.value); };

function changeSpeed(dir) {
  if(dir>0) frameMs=Math.max(50,frameMs-50);
  else frameMs=Math.min(2000,frameMs+50);
  document.getElementById('sl-speed').textContent=`${(300/frameMs).toFixed(1)}×`;
}

function toggleDiff() {
  showDiff=!showDiff;
  const btn=document.getElementById('btn-diff');
  btn.textContent=showDiff?'Diff':'Absolute';
  btn.classList.toggle('active',showDiff);
  document.getElementById('diff-banner').classList.toggle('visible',showDiff);
  draw(curIdx);
}

// ============================================================
//  Keyboard
// ============================================================
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT') return;
  switch(e.key){
    case ' ': e.preventDefault(); togglePlay(); break;
    case 'ArrowLeft': e.preventDefault(); stopPlay(); setIdx(curIdx-1); break;
    case 'ArrowRight': e.preventDefault(); stopPlay(); setIdx(curIdx+1); break;
    case 'd': case 'D': toggleDiff(); break;
    case '1': switchTab('act'); break;
    case '2': if(META&&META.capture_gradients) switchTab('grad'); break;
    case '3': if(META&&(META.lora_targets||[]).length>0) switchTab('lora'); break;
  }
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP / SSE server
# ---------------------------------------------------------------------------

class _NeuronTracerHTTPHandler(BaseHTTPRequestHandler):
    """Minimal handler: serves the dashboard HTML and an SSE stream."""

    def log_message(self, format, *args):
        # Silence default access log; training console is noisy enough
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._serve_html()
        elif path == "/stream":
            self._serve_sse()
        else:
            self.send_error(404)

    def _serve_html(self):
        body = _HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        viz: "NeuronTracerServer" = self.server.visualizer  # type: ignore[attr-defined]

        # 1. Send metadata immediately
        self._send_event("meta", json.dumps(viz.metadata, separators=(",", ":")))

        # 2. Replay all records captured so far (catch-up on reconnect)
        with viz._lock:
            replay = list(viz._records)

        for rec in replay:
            self._send_event("record", json.dumps(rec, separators=(",", ":")))

        # 3. Subscribe to future records
        q: queue.Queue = queue.Queue()
        with viz._lock:
            viz._clients.append(q)

        try:
            while True:
                try:
                    msg = q.get(timeout=15)
                except queue.Empty:
                    # Send SSE keep-alive comment
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue

                if msg is None:
                    # Training finished
                    self._send_event("done", "{}")
                    break

                self._send_event("record", msg)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with viz._lock:
                try:
                    viz._clients.remove(q)
                except ValueError:
                    pass

    def _send_event(self, event_type: str, data: str):
        payload = f"event: {event_type}\ndata: {data}\n\n"
        self.wfile.write(payload.encode())
        self.wfile.flush()


class _ThreadedHTTPServer(HTTPServer):
    """Allows the visualizer reference to be stored on the server."""

    def __init__(self, server_address, handler_class, visualizer: "NeuronTracerServer"):
        super().__init__(server_address, handler_class)
        self.visualizer = visualizer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NeuronTracerServer:
    """Background HTTP server that pushes live activation data to a browser.

    Typically you use :class:`RealtimeActivationCallback` which owns this
    server automatically — but you can also manage it directly:

    Example::

        server = NeuronTracerServer(port=7863)
        server.set_metadata(capture.build_metadata())   # dict from ActivationCapture
        server.start(auto_open=True)

        # … in your training loop:
        server.push(activation_record_dict)

        server.finish()  # signals browser that training is done
    """

    def __init__(self, port: int = 7863):
        self.port = port
        self.metadata: Optional[dict] = None
        self._records: List[dict] = []
        self._clients: List[queue.Queue] = []
        self._lock = threading.Lock()
        self._server: Optional[_ThreadedHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._done = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_metadata(self, metadata: dict):
        self.metadata = metadata

    def start(self, auto_open: bool = False):
        """Bind the server and start the background thread."""
        if self._server is not None:
            return  # already running

        self._server = _ThreadedHTTPServer(("127.0.0.1", self.port), _NeuronTracerHTTPHandler, self)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://127.0.0.1:{self.port}"
        print(f"\n🦥 Unsloth Neuron Tracer → {url}\n")

        if auto_open:
            # Small delay so the server is ready before the browser hits it
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    # ------------------------------------------------------------------
    # Data push
    # ------------------------------------------------------------------

    def push(self, record: dict):
        """Enqueue a new activation record and broadcast it to all SSE clients."""
        data = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._records.append(record)
            for q in self._clients:
                q.put_nowait(data)

    def finish(self):
        """Signal all connected browsers that training has completed."""
        if self._done:
            return
        self._done = True
        with self._lock:
            for q in self._clients:
                q.put_nowait(None)  # sentinel → "done" event

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.finish()
        # Keep the server alive so the user can still view the replay


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------

class RealtimeActivationCallback(ActivationCaptureCallback):
    """Drop-in replacement for :class:`~unsloth.activation_capture.ActivationCaptureCallback`
    that **also** streams every captured record to a live browser dashboard.

    Inherits all capture logic from :class:`ActivationCaptureCallback`.

    Args:
        capture:    An :class:`~unsloth.activation_capture.ActivationCapture` instance.
        port:       Local port for the dashboard server (default: 7863).
        auto_open:  If True, open the browser automatically when training starts.

    Example::

        from unsloth.activation_capture import ActivationCaptureConfig, ActivationCapture
        from unsloth.realtime_visualizer import RealtimeActivationCallback

        cfg = ActivationCaptureConfig(output_dir="run/activations", capture_interval=5)
        cap = ActivationCapture(model, cfg)

        callback = RealtimeActivationCallback(cap, port=7863, auto_open=True)
        trainer = SFTTrainer(model=model, ..., callbacks=[callback])
        trainer.train()
    """

    def __init__(
        self,
        capture: ActivationCapture,
        port: int = 7863,
        auto_open: bool = True,
    ):
        super().__init__(capture)
        self._server = NeuronTracerServer(port=port)
        self._auto_open = auto_open

    # Override to also push records to the server
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Build metadata for the server from the capture object
        meta = self._build_metadata_dict()
        self._server.set_metadata(meta)
        self._server.start(auto_open=self._auto_open)
        # Delegate to parent (attaches hooks, arms step-0 capture)
        super().on_train_begin(args, state, control, **kwargs)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Flush writes to JSONL AND returns the record dict
        record = self._flush_and_return()
        if record is not None:
            self._server.push(record)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Final flush
        record = self._flush_and_return()
        if record is not None:
            self._server.push(record)
        self.capture.detach()
        self._server.finish()
        url = f"http://127.0.0.1:{self._server.port}"
        print(
            f"🦥 Unsloth: Neuron Tracer replay available at {url}\n"
            f"   Activation log: '{self.capture.config.output_dir}'"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_metadata_dict(self) -> dict:
        """Re-read metadata.json written by ActivationCapture.__init__."""
        import os
        meta_path = os.path.join(self.capture.config.output_dir, "metadata.json")
        try:
            with open(meta_path) as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback: build a minimal dict from capture state
            return {
                "model_name": "unknown",
                "num_layers": len(self.capture._layers),
                "hidden_size": self.capture._hidden_size,
                "intermediate_size": self.capture._intermediate_size,
                "captured_channels": self.capture._sampled_channels,
                "capture_interval": self.capture.config.capture_interval,
                "max_channels": self.capture.config.max_channels,
                "capture_mlp_out": self.capture.config.capture_mlp_out,
                "capture_gradients": self.capture.config.capture_gradients,
                "capture_lora_norms": self.capture.config.capture_lora_norms,
                "lora_targets": [],
            }

    def _flush_and_return(self) -> Optional[dict]:
        """Like ActivationCapture.flush() but returns the record instead of only
        writing it to disk, so we can also push it to the SSE server."""
        cap = self.capture
        if not cap._buffer and not cap._grad_buffer:
            cap._should_capture = False
            return None

        record = {
            "step": cap._step,
            "loss": cap._loss,
            "layers": {str(k): v for k, v in cap._buffer.items()},
        }
        if cap._grad_buffer:
            record["grad_norms"] = {str(k): v for k, v in cap._grad_buffer.items()}
        if cap.config.capture_lora_norms and cap._lora_modules:
            lora_norms = cap._compute_lora_norms()
            if lora_norms:
                record["lora_norms"] = {
                    f"{li}.{tgt}": norm
                    for li, targets in lora_norms.items()
                    for tgt, norm in targets.items()
                }

        # Write to JSONL (same as original flush)
        with open(cap._log_path, "a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

        cap._buffer.clear()
        cap._grad_buffer.clear()
        cap._should_capture = False
        return record
