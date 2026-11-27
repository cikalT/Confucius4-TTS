"""Waveform trim editor for the Gradio web UI (gr.HTML + canvas, no extra libraries).

The component value is a JSON string:
    {"url", "duration", "mins", "maxs", "start", "end", "nonce"}
Python fills it when a reference clip is uploaded/recorded; the browser draws the
waveform, lets the user drag a selection and fires an ``input`` event with the new
start/end whenever a drag ends.
"""

import json
import uuid
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import numpy as np

from confuciustts.utils.audio_io import load_audio

BINS = 1600

EMPTY = json.dumps({})


def waveform_value(path, max_seconds=None):
    """Build the editor value for an audio file (or EMPTY when there is none)."""
    if not path:
        return EMPTY
    wav, sr = load_audio(path)
    wav = wav.mean(dim=0).numpy()
    n = len(wav)
    if n == 0:
        return EMPTY
    duration = n / sr
    bins = min(BINS, n)
    edges = np.linspace(0, n, bins + 1).astype(np.int64)[:-1]
    mins = np.minimum.reduceat(wav, edges)
    maxs = np.maximum.reduceat(wav, edges)
    peak = max(float(np.abs(wav).max()), 1e-6)
    end = duration if not max_seconds else min(duration, float(max_seconds))
    return json.dumps({
        "url": "gradio_api/file=" + quote(Path(path).as_posix(), safe="/:"),
        "duration": round(duration, 3),
        "mins": np.round(mins / peak * 100).astype(int).tolist(),
        "maxs": np.round(maxs / peak * 100).astype(int).tolist(),
        "start": 0.0,
        "end": round(end, 3),
        "nonce": uuid.uuid4().hex,
    })


def selection_of(value):
    """(start, end) from an editor value, or (0, 0) when empty."""
    try:
        d = json.loads(value or "{}")
        return round(float(d.get("start", 0)), 2), round(float(d.get("end", 0)), 2)
    except (ValueError, TypeError):
        return 0.0, 0.0


def with_selection(value, start, end):
    """Return the editor value with a new selection (clamped to the clip)."""
    try:
        d = json.loads(value or "{}")
    except ValueError:
        return gr.update()
    if not d.get("mins"):
        return gr.update()
    dur = float(d["duration"])
    s = min(max(float(start or 0), 0.0), dur)
    e = float(end or 0)
    e = dur if e <= 0 else min(e, dur)
    if e - s < 0.1:
        return gr.update()
    d["start"], d["end"] = round(s, 3), round(e, 3)
    return json.dumps(d)


HTML = """
<div class="wf">
  <div class="wf-head">
    <span class="wf-title">Pilih bagian rekaman</span>
    <span class="wf-sel"></span>
  </div>
  <div class="wf-stage">
    <canvas class="wf-canvas"></canvas>
    <div class="wf-empty">Upload atau rekam audio di atas, lalu geser area hijau untuk memilih bagian yang dipakai.</div>
  </div>
  <div class="wf-bar">
    <button data-act="play" class="wf-btn wf-primary">&#9654; Putar seleksi</button>
    <button data-act="stop" class="wf-btn">&#9632; Stop</button>
    <button data-act="all" class="wf-btn">Pilih semua</button>
    <span class="wf-hint">Seret di area kosong untuk seleksi baru &middot; tarik tepi untuk mengubah &middot; tarik tengah untuk menggeser &middot; klik dua kali untuk putar</span>
    <span class="wf-total"></span>
  </div>
</div>
"""

CSS = """
.wf {background:#0b1622; border:1px solid #1f3b57; border-radius:8px; padding:8px 10px 10px; color:#cfe3f5;
     font-family: ui-sans-serif, system-ui, sans-serif; user-select:none;}
.wf-head {display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px;}
.wf-title {font-weight:600; font-size:14px;}
.wf-sel {font-variant-numeric:tabular-nums; font-size:13px;}
.wf-sel .ok {color:#7ee08a;} .wf-sel .warn {color:#f4c26b;}
.wf-stage {position:relative; height:170px; border-radius:6px; overflow:hidden; background:#0e2235;}
.wf-canvas {width:100%; height:100%; display:block; touch-action:none;}
.wf-empty {position:absolute; inset:0; display:flex; align-items:center; justify-content:center; text-align:center;
           padding:0 24px; color:#7f9bb5; font-size:13px;}
.wf-bar {display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:8px;}
.wf-btn {background:#16324c; color:#d8ecff; border:1px solid #2a5378; border-radius:6px; padding:5px 12px;
         font-size:13px; cursor:pointer;}
.wf-btn:hover {background:#1e4466;}
.wf-btn:disabled {opacity:.4; cursor:default;}
.wf-primary {background:#1f7a52; border-color:#2fa36e;} .wf-primary:hover {background:#25905f;}
.wf-hint {color:#7f9bb5; font-size:12px; flex:1; min-width:200px;}
.wf-total {color:#7f9bb5; font-size:12px; font-variant-numeric:tabular-nums;}
"""

JS = r"""
const S = {data: null, start: 0, end: 0, drag: null, raf: 0, playing: false};
const audio = new Audio();
audio.preload = 'auto';
element.__wf = {S, audio};  // handle for debugging from the console
const HANDLE = 7;           // px tolerance for grabbing a selection edge
const TOP = 18;             // px reserved for the time ruler

const $ = (q) => element.querySelector(q);
const fmt = (t) => {
  t = Math.max(0, t);
  const m = Math.floor(t / 60), s = t - m * 60;
  return m + ':' + (s < 10 ? '0' : '') + s.toFixed(2);
};

function parse() {
  let d = null;
  try { d = JSON.parse(props.value || '{}'); } catch (e) { d = null; }
  if (!d || !d.mins || !d.mins.length) { S.data = null; stop(); audio.removeAttribute('src'); return; }
  if (!S.data || S.data.nonce !== d.nonce) { stop(); audio.src = d.url; }
  S.data = d; S.start = d.start; S.end = d.end;
}

function geom() {
  const cv = $('.wf-canvas');
  if (!cv) return null;
  const w = cv.clientWidth, h = cv.clientHeight;
  return {cv, w, h};
}
const t2x = (t, w) => (t / S.data.duration) * w;
const x2t = (x, w) => Math.min(Math.max(x / w, 0), 1) * S.data.duration;

function draw() {
  const g = geom(); if (!g) return;
  const {cv, w, h} = g;
  const empty = $('.wf-empty');
  if (empty) empty.style.display = S.data ? 'none' : 'flex';
  for (const b of element.querySelectorAll('.wf-btn')) b.disabled = !S.data;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  }
  const c = cv.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  c.fillStyle = '#0e2235'; c.fillRect(0, 0, w, h);
  const sel = $('.wf-sel'), total = $('.wf-total');
  // only touch the DOM when text changes, or the MutationObserver below would loop
  const setHTML = (el, v) => { if (el && el.__v !== v) { el.__v = v; el.innerHTML = v; } };
  if (!S.data) { setHTML(sel, ''); setHTML(total, ''); return; }

  const d = S.data, n = d.mins.length, mid = TOP + (h - TOP) / 2, amp = (h - TOP) / 2 - 6;
  const sx = t2x(S.start, w), ex = t2x(S.end, w);

  // selection body
  c.fillStyle = 'rgba(38, 150, 100, 0.42)';
  c.fillRect(sx, TOP, ex - sx, h - TOP);

  // waveform bars, one per pixel column
  for (let x = 0; x < w; x++) {
    const i0 = Math.floor(x / w * n), i1 = Math.max(i0 + 1, Math.floor((x + 1) / w * n));
    let lo = 0, hi = 0;
    for (let i = i0; i < i1 && i < n; i++) { if (d.mins[i] < lo) lo = d.mins[i]; if (d.maxs[i] > hi) hi = d.maxs[i]; }
    const inSel = x >= sx && x <= ex;
    const a = Math.max(hi, -lo) / 100;
    if (inSel) c.fillStyle = a > 0.6 ? '#b9f5c4' : '#6fd38a';
    else c.fillStyle = a > 0.6 ? '#e7c08a' : (a > 0.3 ? '#a9c6e6' : '#6d8fb3');
    c.fillRect(x, mid - hi / 100 * amp, 1, Math.max(1, (hi - lo) / 100 * amp));
  }
  // centre line
  c.fillStyle = 'rgba(255,255,255,0.35)'; c.fillRect(0, mid, w, 1);

  // ruler
  c.fillStyle = '#0a1826'; c.fillRect(0, 0, w, TOP);
  const step = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60].find((s) => (s / d.duration) * w >= 70) || 120;
  c.font = '10px ui-sans-serif, system-ui, sans-serif'; c.textBaseline = 'middle';
  for (let t = 0; t <= d.duration + 1e-6; t += step) {
    const x = Math.round(t2x(t, w)) + 0.5;
    c.fillStyle = '#35597c'; c.fillRect(x, TOP - 5, 1, 5);
    c.fillStyle = '#7f9bb5'; c.fillText(fmt(t), x + 3, TOP / 2);
  }
  // selection strip on the ruler + edges + handles
  c.fillStyle = '#1f9a63'; c.fillRect(sx, 0, ex - sx, 5);
  c.fillStyle = '#e8fff0';
  c.fillRect(Math.round(sx), TOP, 1.5, h - TOP); c.fillRect(Math.round(ex) - 1.5, TOP, 1.5, h - TOP);
  for (const x of [sx, ex]) {
    c.fillStyle = '#f4b860';
    c.beginPath(); c.moveTo(x - 6, 0); c.lineTo(x + 6, 0); c.lineTo(x, 9); c.closePath(); c.fill();
    c.fillStyle = '#e8fff0'; c.fillRect(x - 3, mid - 14, 6, 28);
  }
  // playhead
  if (S.playing) {
    const px = t2x(audio.currentTime, w);
    c.fillStyle = '#ffffff'; c.fillRect(px, TOP, 1.5, h - TOP);
  }

  const len = S.end - S.start, ok = len >= 3 && len <= 15;
  setHTML(sel, fmt(S.start) + ' &rarr; ' + fmt(S.end) +
    ' &middot; <span class="' + (ok ? 'ok' : 'warn') + '">' + len.toFixed(1) + ' dtk' +
    (len < 3 ? ' (terlalu pendek)' : len > 15 ? ' (lebih dari 15 dtk)' : '') + '</span>');
  setHTML(total, 'Total ' + fmt(d.duration));
}

function commit() {
  if (!S.data) return;
  const d = Object.assign({}, S.data, {start: +S.start.toFixed(3), end: +S.end.toFixed(3)});
  S.data = d;
  props.value = JSON.stringify(d);
  trigger('input');
}

function stop() {
  S.playing = false; cancelAnimationFrame(S.raf);
  try { audio.pause(); } catch (e) {}
}
function play() {
  if (!S.data) return;
  stop();
  audio.currentTime = S.start;
  audio.play().then(() => {
    S.playing = true;
    const tick = () => {
      if (!S.playing) return;
      if (audio.currentTime >= S.end || audio.ended) { stop(); draw(); return; }
      draw(); S.raf = requestAnimationFrame(tick);
    };
    tick();
  }).catch((e) => { S.lastError = String(e); });
}

function hit(x, w) {
  const sx = t2x(S.start, w), ex = t2x(S.end, w);
  if (Math.abs(x - sx) <= HANDLE) return 'start';
  if (Math.abs(x - ex) <= HANDLE) return 'end';
  if (x > sx && x < ex) return 'move';
  return 'new';
}

element.addEventListener('pointerdown', (e) => {
  const cv = e.target.closest && e.target.closest('.wf-canvas');
  if (!cv || !S.data) return;
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, w = r.width, t = x2t(x, w);
  const kind = hit(x, w);
  S.drag = {kind, t0: t, s0: S.start, e0: S.end};
  if (kind === 'new') { S.start = t; S.end = t; }
  cv.setPointerCapture(e.pointerId);
  e.preventDefault();
});
element.addEventListener('pointermove', (e) => {
  const cv = e.target.closest && e.target.closest('.wf-canvas');
  if (!cv || !S.data) return;
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, w = r.width, t = x2t(x, w);
  const D = S.data.duration, MIN = 0.1;
  if (!S.drag) {
    const k = hit(x, w);
    cv.style.cursor = (k === 'start' || k === 'end') ? 'ew-resize' : (k === 'move' ? 'grab' : 'crosshair');
    return;
  }
  const g = S.drag;
  if (g.kind === 'start') S.start = Math.min(Math.max(t, 0), S.end - MIN);
  else if (g.kind === 'end') S.end = Math.max(Math.min(t, D), S.start + MIN);
  else if (g.kind === 'move') {
    const len = g.e0 - g.s0;
    let s = g.s0 + (t - g.t0); s = Math.min(Math.max(s, 0), D - len);
    S.start = s; S.end = s + len; cv.style.cursor = 'grabbing';
  } else { S.start = Math.min(g.t0, t); S.end = Math.max(g.t0, t); }
  draw();
});
const endDrag = () => {
  if (!S.drag) return;
  const g = S.drag; S.drag = null;
  if (S.end - S.start < 0.1) { S.start = g.s0; S.end = g.e0; }  // a plain click: keep the old selection
  draw(); commit();
};
element.addEventListener('pointerup', endDrag);
element.addEventListener('pointercancel', endDrag);
element.addEventListener('dblclick', (e) => { if (e.target.closest && e.target.closest('.wf-canvas')) play(); });
element.addEventListener('click', (e) => {
  const b = e.target.closest && e.target.closest('[data-act]');
  if (!b || !S.data) return;
  const act = b.getAttribute('data-act');
  if (act === 'play') play();
  else if (act === 'stop') { stop(); draw(); }
  else if (act === 'all') { S.start = 0; S.end = S.data.duration; draw(); commit(); }
});

// Redraw when the template re-renders, the size changes, or Python sends a new value.
new ResizeObserver(() => draw()).observe(element);
new MutationObserver(() => { if (!element.__wfDrawing) { element.__wfDrawing = true; requestAnimationFrame(() => { element.__wfDrawing = false; draw(); }); } })
  .observe(element, {childList: true, subtree: true});
watch('value', () => { parse(); draw(); });
parse(); draw();
"""


def waveform_editor():
    return gr.HTML(value=EMPTY, html_template=HTML, css_template=CSS, js_on_load=JS,
                   apply_default_css=False, elem_id="wf-editor")
