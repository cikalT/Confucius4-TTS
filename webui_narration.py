"""'Narasi Video' tab: per-line narration for niche-vids films.

Input is the script made while building the baseline video:
  * lines.json            [{"id": "L01", "sec": "hook", "text": ..., "tag": "[excited]", "est": 5.0}, ...]
  * naskah-tts.txt / tts-elevenlabs-v2.txt / tts-elevenlabs-v3.txt   ("L01 | [tag] text" lines, "# SECTION")
  * tts-gemini.txt        ("----- S1 · HOOK -----" sections, one line per sentence)
  * any other .txt        (split into sentences -> L01, L02, ...)

Every line is synthesized on its own and saved immediately to narration/<film>/voice/<id>.wav,
so the work survives a stop or a restart; loading the same script again resumes. The result is
a ZIP with voice/L01.wav ... that drops into the film folder, where prep_voice.py picks it up.

Audio tags ([excited], [nervous], ...) are not read by the model: they are removed from the
spoken text, and each tag can be mapped to its own reference voice instead.
"""

import json
import re
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np

from confuciustts.utils.audio_io import save_audio

NARR_ROOT = Path("narration")
STATUS_LABEL = {"belum": "belum", "sedang": "sedang dibuat…", "selesai": "selesai", "ulang": "perlu dibuat ulang"}
SEC_RE = re.compile(r"^\s*#\s+(.+?)\s*$")
LINE_RE = re.compile(r"^\s*(L\d{1,3})\s*\|\s*(.+?)\s*$", re.I)
SECTION_RE = re.compile(r"^\s*-{3,}\s*S\d+\s*[·.:-]?\s*([^-(]*)", re.I)
TAG_RE = re.compile(r"\[[^\]]+\]")
LEADING_TAGS_RE = re.compile(r"^\s*((?:\[[^\]]+\]\s*)+)")
GENERIC_NAMES = {"lines", "naskah-tts", "naskah", "tts-elevenlabs-v2", "tts-elevenlabs-v3", "tts-gemini", "script"}
FILM_BUDGET_S = 195  # PIPELINE-FILM.md: keep the narration under ~195 s (films are max 3 minutes)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def est_seconds(text):
    """Same estimate as niche-vids' lines.json (≈ 0.35 + characters / 13.5)."""
    return round(0.35 + len(text) / 13.5, 2)


def clean_text(text):
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", text)).strip()


def split_tags(text):
    m = LEADING_TAGS_RE.match(text)
    tags = TAG_RE.findall(m.group(1)) if m else []
    tags = [t.lower() for t in tags if t.lower() not in ("[long pause]", "[pause]")]
    return (tags[0] if tags else ""), clean_text(text)


def slug(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", (name or "").strip()).strip("-.") or "narasi"


def guess_film(raw, filename):
    for pat in (r"niche-vids[\\/](?:film[\\/]\w+[\\/])?([A-Za-z0-9_.-]+)[\\/]voice",
                r"NASKAH TTS\s*[—–-]\s*([A-Za-z0-9_.-]+)"):
        m = re.search(pat, raw)
        if m:
            return m.group(1)
    stem = Path(filename or "").stem
    return slug(stem) if stem and stem.lower() not in GENERIC_NAMES else ""


def content_name(lines):
    """Stable name for scripts without a film name (e.g. lines.json): same script -> same project."""
    import hashlib
    words = re.findall(r"[a-z0-9]+", (lines[0]["text"] if lines else "").lower())[:4]
    digest = hashlib.sha1("\n".join(l["text"] for l in lines).encode("utf-8")).hexdigest()[:6]
    return slug("narasi-" + "-".join(words) + "-" + digest)


def _line(i, text, sec="", tag="", est=None, kind="", line_id=None):
    t, cleaned = split_tags(text)
    tag = (tag or t or "").lower()
    return {"id": (line_id or f"L{i:02d}").upper(), "sec": (sec or "").strip().lower(), "tag": tag,
            "text": cleaned, "est": float(est) if est else est_seconds(cleaned), "kind": kind or ""}


def parse_script(path, filename=""):
    """Return (film_name_guess, lines, source_label)."""
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    film = guess_film(raw, filename)
    stripped = raw.lstrip()

    if str(filename).lower().endswith(".json") or stripped.startswith(("[", "{")):
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("lines", [])
        lines = []
        for i, d in enumerate(data, 1):
            text = d.get("text") or d.get("sub") or ""
            if not text.strip():
                continue
            lines.append(_line(i, text, d.get("sec", ""), d.get("tag", ""), d.get("est"), d.get("kind", ""), d.get("id")))
        return film, lines, "lines.json"

    # "L01 | [tag] text" (naskah-tts.txt, tts-elevenlabs CARA A); first occurrence of each id wins
    sec, found = "", {}
    for ln in raw.splitlines():
        m_sec = SEC_RE.match(ln)
        if m_sec:
            sec = m_sec.group(1)
            continue
        m = LINE_RE.match(ln)
        if m and m.group(1).upper() not in found:
            found[m.group(1).upper()] = _line(0, m.group(2), sec, line_id=m.group(1))
    if found:
        return film, list(found.values()), "naskah (Lxx | teks)"

    # "----- S1 · HOOK -----" sections, one sentence per line (tts-gemini.txt)
    if any(SECTION_RE.match(ln) for ln in raw.splitlines()):
        sec, lines, inside = "", [], False
        for ln in raw.splitlines():
            m = SECTION_RE.match(ln)
            if m:
                sec, inside = m.group(1).strip(" ·-"), True
                continue
            if inside and ln.strip():
                lines.append(_line(len(lines) + 1, ln, sec))
        return film, lines, "naskah per bagian"

    # plain text: one line per sentence
    text = re.sub(r"\s+", " ", raw).strip()
    # "…" is a pause inside a sentence in these scripts ("Nomor satu… Kapal Nomor Nol!"), not an end
    sentences = [s for s in re.split(r"(?<=[.!?])(?<!\.\.\.)\s+", text) if s.strip()]
    return film, [_line(i, s) for i, s in enumerate(sentences, 1)], "teks biasa (dipecah per kalimat)"


# --------------------------------------------------------------------------- #
# Project store (narration/<film>/project.json + voice/<id>.wav)
# --------------------------------------------------------------------------- #
class Project:
    def __init__(self, film):
        self.film = slug(film)
        self.dir = NARR_ROOT / self.film
        self.voice_dir = self.dir / "voice"
        self.path = self.dir / "project.json"
        self.data = {"film": self.film, "lang": "id", "voice": None, "tag_voices": {}, "lines": [], "source": ""}
        if self.path.exists():
            self.data.update(json.loads(self.path.read_text(encoding="utf-8")))

    @property
    def lines(self):
        return self.data["lines"]

    def line(self, line_id):
        return next((l for l in self.lines if l["id"] == line_id), None)

    def wav(self, line_id):
        return self.voice_dir / f"{line_id}.wav"

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data["updated"] = datetime.now().isoformat(timespec="seconds")
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def merge(self, new_lines, source):
        """Take the new script; keep finished audio for lines whose text did not change."""
        old = {l["id"]: l for l in self.lines}
        merged = []
        for n in new_lines:
            o = old.get(n["id"])
            if o and o.get("status") == "selesai" and o.get("text") == n["text"] and self.wav(n["id"]).exists():
                merged.append({**o, "sec": n["sec"], "tag": n["tag"], "est": n["est"], "kind": n["kind"]})
            else:
                merged.append({**n, "status": "ulang" if (o and self.wav(n["id"]).exists()) else "belum"})
        self.data["lines"] = merged
        self.data["source"] = source
        self.save()

    def tags(self):
        counts = {}
        for l in self.lines:
            if l.get("tag"):
                counts[l["tag"]] = counts.get(l["tag"], 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])


ALIAS_FILE = NARR_ROOT / "_aliases.json"


def get_alias(name):
    try:
        target = json.loads(ALIAS_FILE.read_text(encoding="utf-8")).get(name)
    except (OSError, ValueError):
        return None
    return target if target and (NARR_ROOT / target / "project.json").exists() else None


def set_alias(old, new):
    try:
        data = json.loads(ALIAS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    for k, v in list(data.items()):  # chains: a -> old becomes a -> new
        if v == old:
            data[k] = new
    data[old] = new
    NARR_ROOT.mkdir(parents=True, exist_ok=True)
    ALIAS_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")


def speech_seconds(audio_np, sr, thr_db=-42):
    """Length without leading/trailing silence (what prep_voice.py will keep)."""
    x = np.asarray(audio_np, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    win = max(1, int(0.01 * sr))
    e = np.sqrt(np.convolve(x * x, np.ones(win) / win, mode="same") + 1e-12)
    idx = np.where(e > e.max() * 10 ** (thr_db / 20))[0]
    return round((idx[-1] - idx[0]) / sr, 2) if idx.size else 0.0


def flags(lines):
    """Mark lines whose length is far from the script estimate (model rambled or cut off)."""
    done = [l for l in lines if l.get("status") == "selesai" and l.get("dur") and l.get("est")]
    ratios = [l["dur"] / l["est"] for l in done]
    med = float(np.median(ratios)) if len(ratios) >= 3 else 1.0
    out = {}
    for l in done:
        r = l["dur"] / l["est"]
        if r > 1.7 * med and l["dur"] - l["est"] * med > 1.5:
            out[l["id"]] = "⚠ terlalu panjang"
        elif r < 0.5 * med:
            out[l["id"]] = "⚠ terlalu pendek"
    return out


# --------------------------------------------------------------------------- #
# Tab
# --------------------------------------------------------------------------- #
HEADERS = ["#", "ID", "Bagian", "Tag", "Teks", "Status", "Durasi (s)", "Perkiraan (s)", "Cek"]


def build_narration_tab(ctx):
    """ctx: library, synthesize(text, lang, ref, settings), sample_rate, languages,
    build_settings_panel, settings_from_values."""
    library = ctx.library

    def table(proj):
        if not proj or not proj.lines:
            return [["", "", "", "", "Upload naskah dulu.", "", "", "", ""]]
        fl = flags(proj.lines)
        return [[i, l["id"], l.get("sec", ""), l.get("tag", ""), l["text"], STATUS_LABEL.get(l.get("status"), ""),
                 l.get("dur", ""), l.get("est", ""), fl.get(l["id"], "")] for i, l in enumerate(proj.lines, 1)]

    def summary(proj, note=""):
        if not proj or not proj.lines:
            return "<span class='hint'>Belum ada naskah.</span>"
        n = len(proj.lines)
        done = [l for l in proj.lines if l.get("status") == "selesai"]
        total = sum(l.get("dur") or 0 for l in done)
        est_total = sum(l.get("est") or 0 for l in proj.lines)
        fl = flags(proj.lines)
        voice = library.get(proj.data.get("voice") or "")
        md = (f"**{proj.film}** · {n} baris · selesai **{len(done)}/{n}**"
              + (f" · <span style='color:#d97706'>⚠ {len(fl)} perlu dicek</span>" if fl else "")
              + f"  \nDurasi suara {total:.1f} s dari baris selesai · perkiraan naskah {est_total:.0f} s"
              + f" · batas film ±{FILM_BUDGET_S} s"
              + (f"  \nSuara utama: **{voice['name']}**" if voice else "  \nSuara utama: belum dipilih")
              + f" · sumber: {proj.data.get('source', '')}")
        if len(done) == n and total > FILM_BUDGET_S:
            md += f"  \n<span style='color:#d97706'>Total {total:.0f} s melebihi ±{FILM_BUDGET_S} s; pendekkan naskah.</span>"
        return md + (f"  \n{note}" if note else "")

    def line_choices(proj):
        return [(f"{l['id']} · {l['text'][:60]}", l["id"]) for l in proj.lines] if proj else []

    def ref_for(proj, line):
        slug_ = (proj.data.get("tag_voices") or {}).get(line.get("tag") or "") or proj.data.get("voice")
        meta = library.get(slug_ or "")
        if not meta:
            raise gr.Error("Pilih suara utama dulu (atau suara untuk tag ini sudah dihapus).")
        return meta

    # ------------------------------------------------------------------ layout
    gr.Markdown("Buat narasi film per kalimat dari naskah video (`lines.json`, `naskah-tts.txt`, "
                "`tts-elevenlabs-v3.txt`, atau teks biasa). Tiap baris disimpan begitu selesai, jadi bisa "
                "dihentikan dan dilanjutkan kapan saja. Hasil akhirnya ZIP berisi `voice/L01.wav …` untuk "
                "diekstrak ke folder film di niche-vids.")
    film_state = gr.State("")
    tags_state = gr.State([])
    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            script_in = gr.File(label="1. Upload naskah", file_count="single", type="filepath", height=110)
            with gr.Row():
                film_name = gr.Textbox(label="Nama film (folder)", placeholder="mis. pilih-1-kapal-misteri", scale=3,
                                       info="Terisi otomatis. Naskah dengan nama sama melanjutkan progres sebelumnya.")
                lang = gr.Dropdown(ctx.languages, value="id", label="Bahasa", scale=1)
        with gr.Column(scale=2):
            gr.Markdown("**2. Suara**")
            with gr.Row(equal_height=False):
                voice = gr.Dropdown(library.choices(), label="Suara utama", scale=4,
                                    info="Dipakai semua baris, kecuali tag yang dipetakan di bawah.")
                refresh_voices = gr.Button("↻", size="sm", scale=0, min_width=48)
            gr.Markdown("<span class='hint'>Suara per tag emosi (opsional): pilih klip referensi dengan nada "
                        "yang cocok, mis. suara ceria untuk [excited].</span>")

            @gr.render(inputs=[film_state, tags_state])
            def tag_mapping(film, tags):
                proj = Project(film) if film else None
                if not proj or not tags:
                    gr.Markdown("<span class='hint'>Naskah ini tidak memakai tag emosi.</span>")
                    return
                current = proj.data.get("tag_voices") or {}
                choices = [("— suara utama —", "")] + library.choices()
                with gr.Row():
                    for tag, count in tags:
                        dd = gr.Dropdown(choices, value=current.get(tag, ""), label=f"{tag} · {count} baris",
                                         min_width=160, key=f"tagvoice-{film}-{tag}")

                        def set_tag(slug_, tag=tag, film=film):
                            p = Project(film)
                            tv = p.data.setdefault("tag_voices", {})
                            if slug_:
                                tv[tag] = slug_
                            else:
                                tv.pop(tag, None)
                            p.save()

                        dd.input(set_tag, inputs=dd)

    _, _, settings = ctx.build_settings_panel(
        "3. Settings narasi",
        "Berlaku untuk baris yang dibuat berikutnya. 'Lewati normalisasi' aktif karena naskah niche-vids "
        "sudah menulis angka sebagai kata.",
        base={"raw": True})

    gr.Markdown("### 4. Buat suara per baris")
    with gr.Row():
        run_btn = gr.Button("▶ Generate baris yang belum", variant="primary", scale=3)
        flagged_btn = gr.Button("Ulang baris ⚠", scale=2)
        all_btn = gr.Button("Ulang semua", scale=1)
        stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
    info = gr.Markdown("<span class='hint'>Belum ada naskah.</span>")
    grid = gr.Dataframe(headers=HEADERS, value=table(None), interactive=False, wrap=True,
                        column_widths=["4%", "6%", "9%", "9%", "42%", "11%", "7%", "7%", "10%"])

    gr.Markdown("### 5. Cek & ulang satu baris\n<span class='hint'>Klik baris di tabel atau pilih di sini. "
                "Teks bisa diedit sebelum dibuat ulang.</span>")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            line_pick = gr.Dropdown([], label="Baris")
            line_text = gr.Textbox(label="Teks yang dibaca", lines=2)
            with gr.Row():
                line_redo = gr.Button("Simpan teks & buat ulang baris ini", variant="primary")
        with gr.Column(scale=2):
            line_audio = gr.Audio(label="Suara baris ini", type="filepath", interactive=False)
            line_info = gr.Markdown()

    gr.Markdown("### 6. Download")
    with gr.Row(equal_height=False):
        zip_btn = gr.Button("Buat ZIP folder voice/", variant="primary", scale=1)
        zip_file = gr.File(label="ZIP", interactive=False, scale=2)
    zip_info = gr.Markdown()

    # ------------------------------------------------------------------ handlers
    def load_script(path, name_in):
        if not path:
            return gr.update(), "", [], table(None), summary(None), gr.update(choices=[]), gr.update()
        try:
            guess, lines, source = parse_script(path, Path(path).name)
        except Exception as exc:
            raise gr.Error(f"Naskah tidak bisa dibaca: {exc}") from exc
        if not lines:
            raise gr.Error("Tidak ada kalimat yang ditemukan di file ini.")
        named = bool(guess)
        if not guess:
            alias = get_alias(content_name(lines))  # this script was given a film name before
            named = bool(alias)
            guess = alias or content_name(lines)
        proj = Project(guess)
        proj.merge(lines, source)
        if not proj.data.get("voice") and library.choices():
            proj.data["voice"] = library.choices()[0][1]
            proj.save()
        resumed = sum(1 for l in proj.lines if l.get("status") == "selesai")
        note = f"Melanjutkan: {resumed} baris sudah ada suaranya." if resumed else ""
        if not named:
            note += ("  \nFile ini tidak menyebut nama film. Ketik nama folder film di kolom 'Nama film' "
                     "lalu tekan Enter (nama ZIP mengikuti nama ini).")
        return (proj.film, proj.film, proj.tags(), table(proj), summary(proj, note),
                gr.update(choices=line_choices(proj), value=proj.lines[0]["id"]),
                gr.update(value=proj.data.get("voice"), choices=library.choices()))

    script_in.upload(load_script, inputs=[script_in, film_name],
                     outputs=[film_name, film_state, tags_state, grid, info, line_pick, voice])

    def rename(new_name, film):
        """Use another folder name: move the project, or merge it into an existing one."""
        new = slug(new_name)
        if not film or new == film:
            return gr.update(value=film), film, gr.update(), gr.update(), gr.update(), gr.update()
        src, dst = Project(film), Project(new)
        if not dst.path.exists():
            dst.dir.parent.mkdir(parents=True, exist_ok=True)
            src.dir.rename(dst.dir)
            dst = Project(new)
            dst.data["film"] = new
            dst.save()
        else:
            # keep dst's finished lines, take the script from src, then carry over src audio
            dst.merge([{k: l[k] for k in ("id", "sec", "tag", "text", "est", "kind")} for l in src.lines],
                      src.data.get("source", ""))
            for l in src.lines:
                d = dst.line(l["id"])
                if (l.get("status") == "selesai" and src.wav(l["id"]).exists() and d
                        and d.get("status") != "selesai" and d["text"] == l["text"]):
                    dst.voice_dir.mkdir(parents=True, exist_ok=True)
                    src.wav(l["id"]).replace(dst.wav(l["id"]))
                    d.update({k: l[k] for k in ("status", "dur", "seed", "voice", "gen_s", "settings") if k in l})
            for k in ("voice", "tag_voices", "lang"):
                dst.data[k] = dst.data.get(k) or src.data.get(k)
            dst.save()
            shutil.rmtree(src.dir, ignore_errors=True)
        set_alias(film, new)
        return new, new, dst.tags(), table(dst), summary(dst), gr.update(choices=line_choices(dst))

    film_name.submit(rename, inputs=[film_name, film_state],
                     outputs=[film_name, film_state, tags_state, grid, info, line_pick])

    def set_voice(slug_, film, lang_):
        if not film:
            return gr.update()
        proj = Project(film)
        proj.data["voice"], proj.data["lang"] = slug_, lang_
        proj.save()
        return summary(proj)

    voice.input(set_voice, inputs=[voice, film_state, lang], outputs=info)
    lang.input(set_voice, inputs=[voice, film_state, lang], outputs=info)
    refresh_voices.click(lambda v, t: (gr.update(choices=library.choices(), value=v), list(t or [])),
                         inputs=[voice, tags_state], outputs=[voice, tags_state])

    def synth_line(proj, line, settings_vals):
        meta = ref_for(proj, line)
        audio, seed, elapsed, _ = ctx.synthesize(line["text"], proj.data.get("lang", "id"), meta["reference"],
                                                 ctx.settings_from_values(settings_vals))
        proj.voice_dir.mkdir(parents=True, exist_ok=True)
        save_audio(proj.wav(line["id"]), audio, ctx.sample_rate)
        line.update(status="selesai", dur=speech_seconds(audio.cpu().numpy(), ctx.sample_rate), seed=seed,
                    voice=meta["slug"], gen_s=round(elapsed, 1),
                    settings=ctx.settings_from_values(settings_vals))
        proj.save()

    def run(film, mode, *vals):
        if not film:
            raise gr.Error("Upload naskah dulu.")
        proj = Project(film)
        if not proj.data.get("voice"):
            raise gr.Error("Pilih suara utama dulu.")
        if mode == "flagged":
            ids = set(flags(proj.lines))
        elif mode == "all":
            ids = {l["id"] for l in proj.lines}
        else:
            ids = {l["id"] for l in proj.lines if l.get("status") != "selesai"}
        todo = [l for l in proj.lines if l["id"] in ids]
        if not todo:
            yield table(proj), summary(proj, "Tidak ada baris yang perlu dibuat.")
            return
        t_start = time.time()
        for k, line in enumerate(todo, 1):
            line["status"] = "sedang"  # shown only; saved status changes after the line is done
            yield table(proj), summary(proj, f"Membuat {line['id']} ({k}/{len(todo)}) …")
            synth_line(proj, line, vals)
        yield table(proj), summary(proj, f"Selesai: {len(todo)} baris dalam {time.time() - t_start:.0f} s.")

    def run_pending(film, *vals):
        yield from run(film, "pending", *vals)

    def run_flagged(film, *vals):
        yield from run(film, "flagged", *vals)

    def run_all(film, *vals):
        if not film:  # confirm dialog was cancelled
            yield gr.update(), gr.update()
            return
        yield from run(film, "all", *vals)

    run_ev = run_btn.click(run_pending, inputs=[film_state] + settings, outputs=[grid, info])
    flag_ev = flagged_btn.click(run_flagged, inputs=[film_state] + settings, outputs=[grid, info])
    all_ev = all_btn.click(run_all, inputs=[film_state] + settings, outputs=[grid, info],
                           js="(f, ...v) => confirm('Buat ulang SEMUA baris? Suara lama ditimpa.') ? [f, ...v] : [null, ...v]")

    def after_stop(film):
        proj = Project(film) if film else None
        return table(proj), summary(proj, "Dihentikan. Baris yang sudah selesai tersimpan; klik Generate untuk lanjut.")

    stop_btn.click(after_stop, inputs=film_state, outputs=[grid, info], cancels=[run_ev, flag_ev, all_ev])

    def show_line(line_id, film):
        proj = Project(film) if film else None
        line = proj.line(line_id) if proj and line_id else None
        if not line:
            return "", None, ""
        wav = proj.wav(line_id)
        voice_meta = library.get(line.get("voice") or "")
        details = (f"{STATUS_LABEL.get(line.get('status'), '')} · tag {line.get('tag') or '-'} · "
                   f"durasi {line.get('dur', '-')} s / perkiraan {line.get('est', '-')} s"
                   + (f" · suara {voice_meta['name']}" if voice_meta else "")
                   + (f" · seed `{line['seed']}`" if line.get("seed") is not None else ""))
        return line["text"], (str(wav) if wav.exists() and line.get("status") == "selesai" else None), details

    line_pick.change(show_line, inputs=[line_pick, film_state], outputs=[line_text, line_audio, line_info])

    def pick_row(film, evt: gr.SelectData):
        proj = Project(film) if film else None
        if not proj or evt.index is None:
            return gr.update()
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        return proj.lines[row]["id"] if 0 <= row < len(proj.lines) else gr.update()

    grid.select(pick_row, inputs=film_state, outputs=line_pick)

    def redo_line(line_id, text, film, *vals):
        proj = Project(film) if film else None
        line = proj.line(line_id) if proj and line_id else None
        if not line:
            raise gr.Error("Pilih baris dulu.")
        tag, cleaned = split_tags(text or "")
        if not cleaned:
            raise gr.Error("Teksnya kosong.")
        if cleaned != line["text"]:
            line["text"], line["est"] = cleaned, est_seconds(cleaned)
            if tag:
                line["tag"] = tag
        synth_line(proj, line, vals)
        text_out, wav, details = show_line(line_id, film)
        return table(proj), summary(proj, f"{line_id} dibuat ulang."), text_out, wav, details

    line_redo.click(redo_line, inputs=[line_pick, line_text, film_state] + settings,
                    outputs=[grid, info, line_text, line_audio, line_info])

    def make_zip(film):
        if not film:
            raise gr.Error("Upload naskah dulu.")
        proj = Project(film)
        done = [l for l in proj.lines if l.get("status") == "selesai" and proj.wav(l["id"]).exists()]
        if not done:
            raise gr.Error("Belum ada baris yang selesai.")
        missing = [l["id"] for l in proj.lines if l not in done]
        out = Path(tempfile.gettempdir()) / f"{proj.film}-voice.zip"
        info_json = {"film": proj.film, "dibuat": datetime.now().isoformat(timespec="seconds"),
                     "model": "Confucius4-TTS", "bahasa": proj.data.get("lang"),
                     "suara_utama": proj.data.get("voice"), "suara_per_tag": proj.data.get("tag_voices"),
                     "baris": {l["id"]: {"teks": l["text"], "tag": l.get("tag"), "suara": l.get("voice"),
                                         "seed": l.get("seed"), "durasi": l.get("dur")} for l in done},
                     "belum": missing}
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for l in done:
                z.write(proj.wav(l["id"]), f"voice/{l['id']}.wav")
            z.writestr("voice/confucius-narasi.json", json.dumps(info_json, indent=1, ensure_ascii=False))
        total = sum(l.get("dur") or 0 for l in done)
        msg = (f"**{out.name}**: {len(done)} file, {total:.1f} s suara. Ekstrak ke folder film "
               f"(mis. `D:\\niche-vids\\film\\proses\\{proj.film}\\`) supaya jadi `voice\\L01.wav …`, "
               "lalu render seperti biasa (prep_voice.py membaca folder voice/).")
        if missing:
            msg += (f"  \n<span style='color:#d97706'>Belum ada suara untuk {len(missing)} baris: "
                    f"{', '.join(missing[:12])}{' …' if len(missing) > 12 else ''}. Film memakai perkiraan "
                    "durasi untuk baris itu.</span>")
        return str(out), msg

    zip_btn.click(make_zip, inputs=film_state, outputs=[zip_file, zip_info])
