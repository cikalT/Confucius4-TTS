#!/usr/bin/env python3
"""Confucius4-TTS Gradio WebUI with a voice library.

Tabs:
  * Generate      - pick a saved voice, type text, tweak settings, synthesize.
  * Clone suara   - upload/record a reference clip, trim/clean it, save it as a
                    named voice together with its default generation settings.
  * Panduan       - what every setting does.

Voices are stored under ./voices/<slug>/ (reference.wav + voice.json).
All models are auto-downloaded via HuggingFace Hub on first run.

Usage:
    python webui.py --host 127.0.0.1 --port 7860
    python webui.py --backend vllm          # CUDA + vLLM only
"""

import argparse
import asyncio
import inspect
import json
import random
import re
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch

# Make the local package importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent))

from confuciustts.utils.audio_io import decode_with_av, load_audio, save_audio
from confuciustts.utils.common import resolve_device
from webui_waveform import selection_of, waveform_editor, waveform_value, with_selection


# --------------------------------------------------------------------------- #
# Static data
# --------------------------------------------------------------------------- #
LANGUAGES = [
    ("Indonesia", "id"), ("English", "en"), ("Melayu", "ms"), ("中文 (Mandarin)", "zh"),
    ("日本語 (Jepang)", "ja"), ("한국어 (Korea)", "ko"), ("ไทย (Thai)", "th"),
    ("Tiếng Việt (Vietnam)", "vi"), ("Deutsch (Jerman)", "de"), ("Français (Prancis)", "fr"),
    ("Español (Spanyol)", "es"), ("Português", "pt"), ("Italiano", "it"), ("Русский (Rusia)", "ru"),
]
DEFAULT_TEXT = "Halo, selamat datang. Ini adalah contoh suara yang dibuat dengan Confucius4-TTS."
DEFAULT_TEST_TEXT = "Halo, ini adalah contoh suara hasil kloning. Bagaimana menurut Anda?"

# Generation settings: name -> default. Names match ConfuciusTTS.generate() kwargs,
# except "seed" which the UI handles itself.
DEFAULTS = {
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": 30,
    "repetition_penalty": 10.0,
    "num_beams": 3,
    "n_timesteps": 25,
    "inference_cfg_rate": 0.7,
    "max_length": 1520,
    "max_text_tokens_per_segment": 80,
    "cross_fade_duration": 0.3,
    "raw": False,
    "seed": -1,
}
SETTING_KEYS = list(DEFAULTS)

# Uploaded media longer than this is cut (only the reference clip matters).
MAX_EXTRACT_SECONDS = 600

DEFAULT_PRESET = "Seimbang (default)"
PRESETS = {
    DEFAULT_PRESET: {},
    "Stabil": {"temperature": 0.6, "top_p": 0.7, "top_k": 20},
    "Ekspresif": {"temperature": 1.0, "top_p": 0.9, "top_k": 50},
    "Cepat": {"num_beams": 1, "n_timesteps": 16},
    "Kualitas tinggi": {"n_timesteps": 40, "num_beams": 3},
}

CSS = """
.gradio-container {max-width: 1280px !important; margin: 0 auto;}
.setting-card {padding: 12px 14px !important;}
.setting-card h4 {margin: 0 0 4px 0;}
#gen-btn, #clone-btn {min-height: 52px; font-size: 1.05em;}
.hint {color: var(--body-text-color-subdued); font-size: 0.92em;}
"""


# --------------------------------------------------------------------------- #
# Voice library
# --------------------------------------------------------------------------- #
class VoiceLibrary:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _slug(self, name: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "suara"
        slug, n = base, 2
        while (self.root / slug).exists():
            slug, n = f"{base}-{n}", n + 1
        return slug

    def list(self):
        voices = []
        for meta_path in self.root.glob("*/voice.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta["slug"] = meta_path.parent.name
            voices.append(meta)
        return sorted(voices, key=lambda v: v.get("created", ""), reverse=True)

    def choices(self):
        return [(v["name"], v["slug"]) for v in self.list()]

    def get(self, slug: str):
        if not slug:
            return None
        meta_path = self.root / slug / "voice.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["slug"] = slug
        meta["reference"] = str(self.root / slug / "reference.wav")
        return meta

    def create(self, name, description, language, wav, sr, settings, source_duration):
        slug = self._slug(name)
        folder = self.root / slug
        folder.mkdir(parents=True)
        save_audio(folder / "reference.wav", torch.from_numpy(wav).unsqueeze(0), sr)
        meta = {
            "name": name,
            "description": description,
            "language": language,
            "created": datetime.now().isoformat(timespec="seconds"),
            "duration": round(len(wav) / sr, 2),
            "source_duration": round(source_duration, 2),
            "settings": settings,
        }
        (folder / "voice.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return slug

    def update_settings(self, slug, settings):
        meta_path = self.root / slug / "voice.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["settings"] = settings
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def delete(self, slug):
        folder = (self.root / slug).resolve()
        if folder.parent == self.root.resolve() and folder.exists():
            shutil.rmtree(folder)


def process_reference(path, trim_start, trim_end, remove_silence, top_db, normalize, max_seconds):
    """Load a clip, make it mono, trim it and optionally clean it. Returns (wav, sr, source_dur)."""
    wav, sr = load_audio(path)
    wav = wav.mean(dim=0).numpy().astype(np.float32)
    source_dur = len(wav) / sr

    start = int(max(0.0, float(trim_start or 0)) * sr)
    end = int(float(trim_end) * sr) if trim_end and float(trim_end) > 0 else len(wav)
    wav = wav[start:end]
    if remove_silence and len(wav):
        import librosa
        wav, _ = librosa.effects.trim(wav, top_db=float(top_db))
    if max_seconds and len(wav) > int(max_seconds * sr):
        wav = wav[: int(max_seconds * sr)]
    if normalize and len(wav):
        peak = float(np.abs(wav).max())
        if peak > 0:
            wav = wav * (10 ** (-1 / 20) / peak)  # peak at -1 dBFS
    return wav, sr, source_dur


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def free_accelerator_cache(device: str):
    acc = getattr(torch, device, None) if device in ("cuda", "xpu") else None
    if acc is not None and acc.is_available():
        acc.synchronize()
        acc.empty_cache()


def vram_summary(device: str) -> str:
    acc = getattr(torch, device, None) if device in ("cuda", "xpu") else None
    if acc is None or not acc.is_available():
        return device
    gb = 1024 ** 3
    return (f"VRAM alloc {acc.memory_allocated() / gb:.1f} GB, reserved {acc.memory_reserved() / gb:.1f} GB, "
            f"peak {acc.max_memory_allocated() / gb:.1f} GB")


def settings_from_values(values):
    s = dict(zip(SETTING_KEYS, values))
    if s["seed"] is None or s["seed"] == "":
        s["seed"] = -1
    for k in ("top_k", "num_beams", "n_timesteps", "max_length", "max_text_tokens_per_segment", "seed"):
        s[k] = int(s[k])
    # Browsers send whole-number slider values as ints; HF requires floats here.
    for k in ("temperature", "top_p", "repetition_penalty", "inference_cfg_rate", "cross_fade_duration"):
        s[k] = float(s[k])
    s["raw"] = bool(s["raw"])
    return s


def values_from_settings(settings):
    merged = {**DEFAULTS, **(settings or {})}
    return [merged[k] for k in SETTING_KEYS]


def build_settings_panel(title, note):
    """Visible settings panel. Returns (preset_radio, reset_btn, [components in SETTING_KEYS order])."""
    with gr.Group():
        gr.Markdown(f"### {title}\n<span class='hint'>{note}</span>")
        with gr.Row(equal_height=False):
            preset = gr.Radio(list(PRESETS), value=DEFAULT_PRESET, label="Preset cepat",
                              info="Titik awal. Slider di bawah tetap bisa diubah setelah memilih preset.",
                              scale=4)
            reset = gr.Button("↺ Kembalikan ke default", size="sm", scale=0, min_width=200)
    with gr.Row(equal_height=False):
        with gr.Column(elem_classes="setting-card"):
            gr.Markdown("#### Gaya & stabilitas")
            temperature = gr.Slider(0.1, 1.5, value=DEFAULTS["temperature"], step=0.05, label="Variasi / temperature",
                                    info="Rendah = stabil & konsisten. Tinggi = lebih ekspresif, tapi bisa salah ucap. (≈ Stability di ElevenLabs, dibalik)")
            top_p = gr.Slider(0.1, 1.0, value=DEFAULTS["top_p"], step=0.05, label="Top-p",
                              info="Batas peluang token yang boleh dipilih. Rendah = aman, tinggi = beragam.")
            top_k = gr.Slider(1, 100, value=DEFAULTS["top_k"], step=1, label="Top-k",
                              info="Jumlah kandidat token tiap langkah. Rendah = aman, tinggi = beragam.")
            repetition_penalty = gr.Slider(1.0, 20.0, value=DEFAULTS["repetition_penalty"], step=0.5,
                                           label="Penalti pengulangan",
                                           info="Mencegah suara mengulang/macet. Turunkan jika suara terdengar terburu-buru.")
        with gr.Column(elem_classes="setting-card"):
            gr.Markdown("#### Kualitas & kecepatan")
            num_beams = gr.Slider(1, 5, value=DEFAULTS["num_beams"], step=1, label="Jumlah beam",
                                  info="Lebih banyak = lebih rapi tapi lebih lambat. 1 = tercepat.")
            n_timesteps = gr.Slider(4, 64, value=DEFAULTS["n_timesteps"], step=1, label="Langkah difusi",
                                    info="Detail akustik. Lebih banyak = lebih halus tapi lebih lambat.")
            inference_cfg_rate = gr.Slider(0.0, 2.0, value=DEFAULTS["inference_cfg_rate"], step=0.05,
                                           label="Kekuatan kemiripan / CFG",
                                           info="Seberapa kuat hasil mengikuti kondisi suara. (≈ Similarity di ElevenLabs)")
            max_length = gr.Slider(200, 1520, value=DEFAULTS["max_length"], step=10, label="Panjang token maksimal",
                                   info="Batas panjang audio per potongan teks. Biarkan maksimal kecuali ingin membatasi.")
        with gr.Column(elem_classes="setting-card"):
            gr.Markdown("#### Teks panjang & lainnya")
            max_text_tokens_per_segment = gr.Slider(20, 200, value=DEFAULTS["max_text_tokens_per_segment"], step=5,
                                                    label="Panjang potongan teks",
                                                    info="Teks panjang dipecah per potongan. Kecil = lebih stabil, besar = intonasi lebih menyambung.")
            cross_fade_duration = gr.Slider(0.0, 1.5, value=DEFAULTS["cross_fade_duration"], step=0.05,
                                            label="Jeda antar potongan (detik)",
                                            info="Jeda hening + fade di antara potongan teks.")
            raw = gr.Checkbox(value=DEFAULTS["raw"], label="Lewati normalisasi teks",
                              info="Centang agar teks dibaca apa adanya (angka/singkatan tidak diubah otomatis). Tulis angka sebagai kata.")
            seed = gr.Number(value=DEFAULTS["seed"], precision=0, label="Seed",
                             info="-1 = acak tiap kali. Isi angka (lihat status hasil) untuk mengulang hasil yang sama persis.")
    comps = [temperature, top_p, top_k, repetition_penalty, num_beams, n_timesteps, inference_cfg_rate,
             max_length, max_text_tokens_per_segment, cross_fade_duration, raw, seed]
    # order must match SETTING_KEYS
    ordered = dict(zip(["temperature", "top_p", "top_k", "repetition_penalty", "num_beams", "n_timesteps",
                        "inference_cfg_rate", "max_length", "max_text_tokens_per_segment",
                        "cross_fade_duration", "raw", "seed"], comps))
    comps = [ordered[k] for k in SETTING_KEYS]

    def apply_preset(name):
        return values_from_settings(PRESETS.get(name, {}))

    preset.input(apply_preset, inputs=preset, outputs=comps)
    reset.click(lambda: [DEFAULT_PRESET] + values_from_settings({}), outputs=[preset] + comps)
    return preset, reset, comps


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def build_demo(model, library: VoiceLibrary, device: str, backend: str):
    loop = asyncio.new_event_loop()
    accepted = set(inspect.signature(model.generate).parameters)

    def run_generate(text, lang, ref_path, settings):
        """Synthesize; returns (wav path, markdown status)."""
        s = dict(settings)
        seed = s.pop("seed")
        if seed is None or seed < 0:
            seed = random.randint(0, 2**31 - 1)
        random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
        kwargs = {k: v for k, v in s.items() if k in accepted}
        ignored = [k for k in s if k not in accepted]

        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] generate: {len(text)} chars, lang={lang}, "
              f"ref={Path(ref_path).parent.name}, seed={seed}", flush=True)
        try:
            audio = model.generate(text, lang, ref_path, verbose=False, **kwargs)
            if asyncio.iscoroutine(audio):  # vLLM backend is async
                audio = loop.run_until_complete(audio)
        except Exception as exc:
            raise gr.Error(f"Gagal generate: {type(exc).__name__}: {exc}") from exc
        finally:
            # Return cached blocks from this request's shapes to the driver. Without this the
            # allocator keeps growing across requests until VRAM spills into shared system
            # memory, which makes generation crawl (seen on a 12 GB Arc B580).
            free_accelerator_cache(device)
        elapsed = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}]   done: {audio.shape[-1] / model.sample_rate:.1f}s audio "
              f"in {elapsed:.1f}s, {vram_summary(device)}", flush=True)

        out = Path(tempfile.gettempdir()) / f"confucius4_{uuid.uuid4().hex[:10]}.wav"
        save_audio(out, audio, model.sample_rate)
        dur = audio.shape[-1] / model.sample_rate
        status = (f"**Selesai.** {dur:.1f} detik audio dalam {elapsed:.1f} detik "
                  f"({elapsed / max(dur, 1e-6):.2f}× waktu nyata) · seed `{seed}`")
        if ignored:
            status += f"\n\n<span class='hint'>Backend {backend} mengabaikan: {', '.join(ignored)}</span>"
        return str(out), status

    def normalized_preview(text, lang, raw):
        if raw or not hasattr(model, "normalizer"):
            return text
        try:
            return model.normalizer.normalize(text, language=lang)
        except Exception:
            return text

    def voice_info_md(meta):
        if not meta:
            return "<span class='hint'>Belum ada suara dipilih. Buat dulu di tab **Clone suara**.</span>"
        desc = f" · {meta['description']}" if meta.get("description") else ""
        return (f"**{meta['name']}**{desc}  \n<span class='hint'>Referensi {meta.get('duration', '?')} detik · "
                f"dibuat {meta.get('created', '?').replace('T', ' ')}</span>")

    # ------------------------------------------------------------------ UI
    gpu = torch.xpu.get_device_name(0) if device == "xpu" else (
        torch.cuda.get_device_name(0) if device == "cuda" else "CPU")
    with gr.Blocks(title="Confucius4-TTS") as demo:
        gr.Markdown(
            "# Confucius4-TTS\n"
            "Kloning suara dan text-to-speech untuk 14 bahasa. "
            f"<span class='hint'>Perangkat: {gpu} ({device}) · backend: {backend}</span>"
        )

        with gr.Tabs() as tabs:
            # ============================================================ Generate
            with gr.Tab("Generate", id="generate"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        with gr.Row(equal_height=False):
                            voice = gr.Dropdown(library.choices(), label="Suara",
                                                value=(library.choices() or [(None, None)])[0][1],
                                                info="Suara hasil kloning yang tersimpan.", scale=4)
                            refresh = gr.Button("↻ Muat ulang", size="sm", scale=0, min_width=120)
                        voice_info = gr.Markdown()
                        text = gr.Textbox(label="Teks", value=DEFAULT_TEXT, lines=6, max_lines=20,
                                          placeholder="Ketik teks yang ingin diucapkan...")
                        lang = gr.Dropdown(LANGUAGES, value="id", label="Bahasa teks",
                                           info="Bahasa dari teks di atas. Suara bisa dipakai lintas bahasa.")
                        gen_btn = gr.Button("Generate suara", variant="primary", elem_id="gen-btn")
                    with gr.Column(scale=2):
                        gen_audio = gr.Audio(label="Hasil", type="filepath", interactive=False)
                        gen_status = gr.Markdown("<span class='hint'>Hasil akan muncul di sini.</span>")
                        norm_text = gr.Textbox(label="Teks yang dibaca model (setelah normalisasi)",
                                               lines=3, interactive=False)
                        with gr.Accordion("Audio referensi suara ini", open=False):
                            ref_player = gr.Audio(type="filepath", interactive=False, show_label=False)
                _, _, gen_settings = build_settings_panel(
                    "Settings generate",
                    "Terisi otomatis dari setting default suara yang dipilih. Ubah sesuka hati untuk generate ini.")
                with gr.Row():
                    save_to_voice = gr.Button("Simpan settings ini sebagai default suara", size="sm")
                    delete_voice = gr.Button("Hapus suara ini", size="sm", variant="stop")
                    voice_msg = gr.Markdown()

            # ============================================================ Clone
            with gr.Tab("Clone suara", id="clone"):
                gr.Markdown("### 1. Rekaman referensi\n"
                            "<span class='hint'>Paling bagus 3–10 detik, satu pembicara, jernih, tanpa "
                            "musik/gema. Emosi & gaya bicara rekaman ikut ditiru.</span>")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        # gr.Audio's own upload box rejects many valid files by MIME type
                        # (e.g. .mpeg is reported as video/mpeg), so uploads go through a
                        # plain file drop zone and are decoded server-side.
                        file_in = gr.File(label="Upload audio atau video", file_count="single",
                                          type="filepath", height=110)
                        gr.Markdown("<span class='hint'>Format apa saja: wav, mp3, m4a, ogg, flac, wma, "
                                    "mp4, mpeg, mov, mkv, webm, dll. Dari video, audionya diambil otomatis "
                                    f"(maks. {MAX_EXTRACT_SECONDS // 60} menit pertama).</span>")
                        ref_in = gr.Audio(sources=["microphone"], type="filepath",
                                          label="…atau rekam langsung (hasil upload juga diputar di sini)")
                    with gr.Column(scale=2):
                        clone_name = gr.Textbox(label="Nama suara", placeholder="mis. Narator Budi")
                        clone_lang = gr.Dropdown(LANGUAGES, value="id", label="Bahasa default")
                        clone_desc = gr.Textbox(label="Catatan (opsional)", placeholder="mis. tenang, cocok untuk narasi")
                gr.Markdown("### 2. Potong & bersihkan rekaman")
                wave = waveform_editor()
                with gr.Row(equal_height=False):
                    trim_start = gr.Number(value=0, label="Mulai (detik)", minimum=0, step=0.1,
                                           info="Ikut berubah saat area hijau digeser.")
                    trim_end = gr.Number(value=0, label="Sampai (detik)", minimum=0, step=0.1,
                                         info="0 = sampai akhir")
                    with gr.Column():
                        remove_silence = gr.Checkbox(value=True, label="Buang hening di awal & akhir")
                        top_db = gr.Slider(10, 60, value=30, step=1, label="Sensitivitas hening (dB)",
                                           info="Kecil = memotong lebih agresif.")
                    with gr.Column():
                        normalize = gr.Checkbox(value=True, label="Samakan volume (puncak -1 dB)")
                        max_seconds = gr.Slider(3, 30, value=15, step=1, label="Durasi maksimal (detik)",
                                                info="Sisa setelah ini dibuang. Disarankan ≤ 10–15 detik.")
                _, _, clone_settings = build_settings_panel(
                    "3. Settings default suara",
                    "Disimpan bersama suara dan otomatis dipakai di tab Generate (masih bisa diubah di sana).")
                gr.Markdown("### 4. Simpan & coba")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        test_text = gr.Textbox(label="Kalimat uji", value=DEFAULT_TEST_TEXT, lines=2)
                        make_test = gr.Checkbox(value=True, label="Langsung buat contoh suara setelah disimpan")
                        clone_btn = gr.Button("Simpan suara", variant="primary", elem_id="clone-btn")
                    with gr.Column(scale=2):
                        clone_status = gr.Markdown()
                        processed_ref = gr.Audio(label="Referensi setelah dibersihkan", type="filepath",
                                                 interactive=False)
                        test_audio = gr.Audio(label="Contoh suara hasil kloning", type="filepath",
                                              interactive=False)

            # ============================================================ Guide
            with gr.Tab("Panduan", id="guide"):
                gr.Markdown(GUIDE_MD)

        # -------------------------------------------------------------- events
        def on_voice_change(slug):
            meta = library.get(slug)
            if not meta:
                return [voice_info_md(None), None, gr.update()] + [gr.update()] * len(SETTING_KEYS)
            return ([voice_info_md(meta), meta["reference"], meta.get("language", "id")]
                    + values_from_settings(meta.get("settings")))

        voice.change(on_voice_change, inputs=voice, outputs=[voice_info, ref_player, lang] + gen_settings)

        def on_refresh(current):
            choices = library.choices()
            values = [c[1] for c in choices]
            return gr.update(choices=choices, value=current if current in values else (values[0] if values else None))

        refresh.click(on_refresh, inputs=voice, outputs=voice)
        # Voices may have been added since the server started: reload the list on page open.
        demo.load(on_refresh, inputs=voice, outputs=voice).then(
            on_voice_change, inputs=voice, outputs=[voice_info, ref_player, lang] + gen_settings)

        def on_generate(slug, txt, language, *vals):
            meta = library.get(slug)
            if not meta:
                raise gr.Error("Pilih suara dulu. Belum punya? Buat di tab Clone suara.")
            if not txt or not txt.strip():
                raise gr.Error("Teksnya masih kosong.")
            settings = settings_from_values(vals)
            preview = normalized_preview(txt, language, settings["raw"])
            path, status = run_generate(txt, language, meta["reference"], settings)
            return path, status, preview

        gen_btn.click(on_generate, inputs=[voice, text, lang] + gen_settings,
                      outputs=[gen_audio, gen_status, norm_text])

        def on_save_settings(slug, *vals):
            if not library.get(slug):
                raise gr.Error("Pilih suara dulu.")
            library.update_settings(slug, settings_from_values(vals))
            gr.Info("Settings disimpan sebagai default suara ini.")
            return ""

        save_to_voice.click(on_save_settings, inputs=[voice] + gen_settings, outputs=voice_msg)

        def on_delete(slug):
            if not slug:
                return gr.update(), ""
            meta = library.get(slug)
            library.delete(slug)
            choices = library.choices()
            gr.Info(f"Suara '{meta['name'] if meta else slug}' dihapus.")
            return gr.update(choices=choices, value=choices[0][1] if choices else None), ""

        delete_voice.click(on_delete, inputs=voice, outputs=[voice, voice_msg],
                           js="(v) => (v && confirm('Hapus suara ini secara permanen?')) ? v : null")

        def on_clone(ref, name, language, desc, t_start, t_end, rm_sil, db, norm, max_s, test_txt, do_test, *vals):
            if not ref:
                raise gr.Error("Upload atau rekam audio referensi dulu.")
            if not name or not name.strip():
                raise gr.Error("Beri nama untuk suara ini.")
            try:
                wav, sr, src_dur = process_reference(ref, t_start, t_end, rm_sil, db, norm, max_s)
            except Exception as exc:
                raise gr.Error(f"Audio tidak bisa dibaca/diproses: {exc}") from exc
            dur = len(wav) / sr
            if dur < 1.5:
                raise gr.Error(f"Hasil potongan hanya {dur:.1f} detik. Minimal ±3 detik; cek pengaturan potong/hening.")
            settings = settings_from_values(vals)
            slug = library.create(name.strip(), (desc or "").strip(), language, wav, sr, settings, src_dur)
            meta = library.get(slug)

            status = f"**Suara '{meta['name']}' tersimpan.** Referensi {dur:.1f} detik (asli {src_dur:.1f} detik)."
            if dur < 3:
                status += " Agak pendek; hasil kloning bisa kurang mirip."
            elif dur > 15:
                status += " Agak panjang; proses bisa lebih lambat."
            sample = None
            if do_test and test_txt and test_txt.strip():
                sample, gen_status_md = run_generate(test_txt, language, meta["reference"], settings)
                status += "\n\n" + gen_status_md
            status += "\n\nBuka tab **Generate** untuk memakai suara ini."
            choices = library.choices()
            return status, meta["reference"], sample, gr.update(choices=choices, value=slug)

        # Waveform editor <-> trim fields (both directions use .input, so no feedback loop)
        def on_ref_change(path, max_s):
            try:
                value = waveform_value(path, max_s)
            except Exception as exc:
                gr.Warning(f"Waveform tidak bisa ditampilkan: {exc}")
                value = waveform_value(None)
            start, end = selection_of(value)
            return value, start, end

        ref_in.change(on_ref_change, inputs=[ref_in, max_seconds], outputs=[wave, trim_start, trim_end])

        def on_any_upload(file):
            """Decode any audio file or video's audio track into a WAV for the audio box."""
            path = file if isinstance(file, str) else getattr(file, "name", None)
            if not path:
                return gr.update()
            try:
                wav, sr = decode_with_av(path, max_seconds=MAX_EXTRACT_SECONDS)
            except ImportError as exc:
                raise gr.Error("Butuh paket PyAV: pip install av") from exc
            except ValueError as exc:  # no audio stream
                raise gr.Error("File ini tidak punya trek audio (video tanpa suara).") from exc
            except Exception as exc:
                raise gr.Error(f"File ini tidak bisa dibaca sebagai audio/video: {exc}") from exc
            if len(wav) < sr * 0.5:
                raise gr.Error("Trek audionya kosong atau terlalu pendek.")
            if len(wav) >= MAX_EXTRACT_SECONDS * sr:
                gr.Info(f"File panjang: hanya {MAX_EXTRACT_SECONDS // 60} menit pertama yang diambil.")
            out = Path(tempfile.gettempdir()) / f"confucius4_extract_{uuid.uuid4().hex[:10]}.wav"
            save_audio(out, torch.from_numpy(wav).unsqueeze(0), sr)
            return str(out)

        file_in.upload(on_any_upload, inputs=file_in, outputs=ref_in)
        file_in.clear(lambda: None, outputs=ref_in)
        wave.input(selection_of, inputs=wave, outputs=[trim_start, trim_end])
        trim_start.input(with_selection, inputs=[wave, trim_start, trim_end], outputs=wave)
        trim_end.input(with_selection, inputs=[wave, trim_start, trim_end], outputs=wave)

        clone_btn.click(
            on_clone,
            inputs=[ref_in, clone_name, clone_lang, clone_desc, trim_start, trim_end, remove_silence, top_db,
                    normalize, max_seconds, test_text, make_test] + clone_settings,
            outputs=[clone_status, processed_ref, test_audio, voice],
        )

    return demo


GUIDE_MD = """
## Cara pakai
1. **Clone suara**: upload atau rekam 3–10 detik suara, beri nama, lalu klik **Simpan suara**.
   Rekaman dibersihkan (potong, buang hening, samakan volume) lalu disimpan di folder `voices/`.
2. **Generate**: pilih suara, ketik teks, pilih bahasa teks, lalu klik **Generate suara**.
   Setting terisi dari default suara. Ubah seperlunya, lalu klik *Simpan settings ini sebagai default suara*
   jika ingin dipakai terus.

## Arti setting
| Setting | Padanan ElevenLabs | Efek |
|---|---|---|
| Variasi / temperature, Top-p, Top-k | Stability (dibalik) | Rendah = stabil dan konsisten. Tinggi = ekspresif, tapi lebih sering salah ucap |
| Kekuatan kemiripan / CFG | Similarity | Seberapa kuat hasil mengikuti kondisi suara |
| Penalti pengulangan | – | Mencegah suara mengulang atau macet |
| Jumlah beam, Langkah difusi | – | Kualitas lawan kecepatan |
| Panjang potongan teks, Jeda antar potongan | – | Cara teks panjang dipecah dan disambung |
| Lewati normalisasi teks | – | Teks dibaca apa adanya (angka tidak diubah jadi kata) |
| Seed | Seed | Isi angka yang sama untuk mengulang hasil yang persis sama |

**Tidak tersedia di model ini:** pengatur kecepatan bicara dan *style exaggeration*.
Emosi dan gaya bicara diambil dari rekaman referensi, jadi pilih rekaman dengan nada yang Anda inginkan.

Nilai preset adalah titik awal, bukan hasil uji. Bandingkan beberapa variasi untuk suara Anda.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Confucius4-TTS Gradio WebUI")
    parser.add_argument("--config", default="config/inference_config.yaml", help="Inference config YAML path.")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--host", default="127.0.0.1", help="Gradio server host (0.0.0.0 to expose on LAN).")
    parser.add_argument("--voices_dir", default="voices", help="Where cloned voices are stored.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4,
                        help="vLLM GPU memory utilization (default 0.4).")
    parser.add_argument("--backend", choices=("auto", "hf", "vllm"), default="auto",
                        help="T2S backend: vllm (CUDA only), hf (HuggingFace, works on "
                             "CUDA / Intel XPU / CPU), or auto (vllm on CUDA if installed, else hf).")
    parser.add_argument("--device", default="auto", help="cuda, xpu (Intel Arc), cpu or auto.")
    parser.add_argument("--inbrowser", action="store_true", help="Open the UI in the default browser when ready.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    backend = args.backend
    if backend == "auto":
        backend = "hf"
        if device == "cuda":
            try:
                import vllm  # noqa: F401
                backend = "vllm"
            except ImportError:
                pass
    if backend == "vllm":
        from confuciustts.cli.inference_vllm import ConfuciusTTSVLLM
        model = ConfuciusTTSVLLM(
            config_path=args.config,
            gpu_memory_utilization=args.gpu_memory_utilization,
            device=device,
        )
    else:
        from confuciustts.cli.inference import ConfuciusTTS
        model = ConfuciusTTS(config_path=args.config, device=device)
    print(f"Loaded. backend={backend} device={device} sample_rate={model.sample_rate}")

    library = VoiceLibrary(Path(args.voices_dir))
    demo = build_demo(model, library, device, backend)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, inbrowser=args.inbrowser,
        theme=gr.themes.Soft(primary_hue="indigo", font=["Inter", "Arial", "sans-serif"]), css=CSS,
    )


if __name__ == "__main__":
    main()
