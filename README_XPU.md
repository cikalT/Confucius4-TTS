# Running Confucius4-TTS on Intel Arc (XPU) — Windows

Tested: Intel Arc B580 (12 GB), driver 32.0.101.9150, Windows 11, Python 3.10,
torch 2.14.1+xpu, torchaudio 2.11.0+xpu, transformers 4.52.4.

## Cara cepat (klik saja)

Butuh: Windows 10/11 64-bit, GPU Intel Arc dengan driver terbaru, internet, dan ruang disk
bebas sekitar 8 GB di folder proyek ditambah 10 GB di drive user (model + cache).

1. **`1_setup.bat`**: jalankan sekali (15–30 menit, tergantung internet). Isinya:
   - cek Windows, GPU Arc + driver, RAM, ruang disk, dan koneksi;
   - pasang `uv`, Python 3.10, PyTorch 2.14.1 XPU, dan semua dependensi ke folder `.venv`;
   - unduh semua model (sekitar 5 GB);
   - tes generate satu kalimat (hasilnya `outputs\selftest.wav`).

   Kalau ada yang gagal, cukup jalankan lagi; setup aman diulang.
2. **`2_run.bat`**: jalankan setiap kali mau memakai. Browser terbuka otomatis di
   http://127.0.0.1:7860. Biarkan jendelanya terbuka, dan tutup jendela itu untuk berhenti.
   Model dimuat dari cache lokal, jadi tidak butuh internet. Kalau WebUI sudah terbuka di
   jendela lain, `2_run.bat` hanya membuka browser, karena dua proses model tidak muat di
   VRAM 12 GB.

Opsi: `2_run.bat --host 0.0.0.0` membuka WebUI untuk HP/laptop lain di jaringan yang sama;
`2_run.bat --port 7861` memakai port lain.

## Manual setup (uv)

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.14.1 --index-url https://download.pytorch.org/whl/xpu
# torchaudio's newest XPU wheel (2.11) pins torch 2.11; the parts used here are pure
# PyTorch and work on 2.14, so install it without deps.
uv pip install --python .venv\Scripts\python.exe --no-deps torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/xpu
# runtime deps without torch/torchaudio, so the XPU build is not replaced
uv pip install --python .venv\Scripts\python.exe -r requirements_xpu.txt
.venv\Scripts\python.exe -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0), hasattr(torch.xpu, 'XPUGraph'))"
```

torch 2.7.0+xpu (with torchaudio 2.7.0+xpu) also works, but has no `XPUGraph`, so
generation falls back to the slower eager path automatically.

## Run

```powershell
# CLI (device defaults to auto: cuda > xpu > cpu)
.venv\Scripts\python.exe example.py --prompt_wav ref.wav --text "Halo, apa kabar?" --lang id --out out.wav

# Web UI (uses the HuggingFace backend automatically when not on CUDA+vLLM)
.venv\Scripts\python.exe webui.py --host 127.0.0.1 --port 7860
```

### Web UI (Indonesian)

- **Clone suara**: upload or record a reference clip and pick the part to keep in a
  waveform editor (`webui_waveform.py`): drag on empty space for a new selection, drag the
  edges to resize, drag the middle to move it, play the selection, and double-click to play.
  Start/end stay in sync with the number fields. Uploads go through a plain file drop zone
  that accepts any file (gr.Audio's own upload rejects valid files by MIME type, e.g. an MP3
  saved as `.mpeg` is reported as `video/mpeg`). The server decodes audio, or the audio
  track of a video, with PyAV (`av` in requirements.txt, first 10 minutes). The audio box
  below it is for microphone recording and playback. Then optionally strip leading and trailing
  silence, peak-normalize and cap the length, name the voice, and save it with default
  generation settings. It can also synthesize a test sentence right after saving.
  Voices are stored in `voices/<name>/` (`reference.wav` + `voice.json`); `--voices_dir`
  changes the location.
- **Generate**: pick a saved voice. Its settings load into the panel, where they can be
  changed for this generation or saved back as the voice's defaults. Shows the text after
  normalization, the seed used (enter it again for an identical result) and the speed.
- Every working generation parameter is visible as a slider, with presets (Seimbang,
  Stabil, Ekspresif, Cepat, Kualitas tinggi) as starting points.
  `edge_fade_duration` / `edge_pad_duration` are not shown because `generate()` ignores them.
- The default host is now `127.0.0.1`; pass `--host 0.0.0.0` to open it on the LAN.

Models (~5 GB) download to `%USERPROFILE%\.cache\huggingface` on first run.
Only one model process fits in 12 GB VRAM: if two run at once, Windows spills to shared
system memory and generation becomes 50-100x slower.

## Changes vs upstream

- `confuciustts/utils/common.py`: `get_default_device()` / `resolve_device()` (cuda > xpu > cpu).
- `inference.py`, `run_inference.py`, `example.py`: `--device` defaults to `auto`, accepts `xpu`.
- `confuciustts/llm/graph_decode.py` + hooks in `llm.py`: on XPU, T2S decoding uses a
  static KV cache and replays each decode step from a captured `torch.xpu.XPUGraph`
  (one graph per 256-token context bucket). On Windows each kernel launch costs
  ~30-60 us and a GPT-2 step issues ~600 kernels, so eager decoding is launch-bound.
  HF `generate` (beam sampling, logits processors) is unchanged; only the forward is
  swapped. Greedy and beam outputs are token-identical to the eager path.
  Disable with `CONFUCIUS_GRAPH_DECODE=0`.
- `inference.py`: on XPU the mel is padded to 256-frame buckets before BigVGAN and the
  output trimmed. oneDNN recompiles conv kernels for each new input length (~3 s each);
  bucketing lets them be reused (vocoder 3 s → 0.1–0.4 s once a bucket is warm).
- `confuciustts/utils/audio_io.py`: wav load/save via soundfile (torchaudio >= 2.9 needs
  torchcodec for `load`/`save`).
- `webui.py`: rewritten with a voice library (Clone / Generate / Panduan tabs) and all
  settings; `--backend auto|hf|vllm`, `--device`, `--voices_dir`. vLLM is CUDA-only, so on
  XPU the HuggingFace `ConfuciusTTS` backend is used (settings vLLM does not accept are
  skipped, and the UI says which).

## Performance (B580, fp32)

| | before | now |
|---|---|---|
| T2S, beam 3 | 25.5 ms/token | 13.1 ms/token |
| T2S, greedy | 20.5 ms/token | 6.3 ms/token |
| 5–6 s utterance, warm | 12–14 s (~2.3× real time) | 5–6 s (~1× real time) |

Memory: on Arc under Windows, running out of VRAM does not raise OOM. The driver silently
spills into shared system RAM, generation slows down 10-100x, and eventually the device can be
lost. Three things keep usage flat (~7 GB peak, also for 30 s multi-segment texts):
the beam-search KV-cache reorder uses bucket-sized temporaries (exact, growing sizes
fragmented the allocator by several GB per long utterance), `empty_cache()` runs between
text segments, and the web UI calls it after every request. The UI also logs each request with
its VRAM usage.

The first request with a new length is slower (graph capture + vocoder kernel compile,
~11 s); later requests reuse them. Peak VRAM ~8–10 GB. bf16 autocast did not help.
Remaining time: HF beam-search bookkeeping (~7 ms/token) and the S2A flow-matching stage
(~1.3–2 s). `server.py` and `example_vllm.py` still need vLLM (CUDA only).
