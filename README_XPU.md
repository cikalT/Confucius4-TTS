# Running Confucius4-TTS on Intel Arc (XPU) — Windows

Tested: Intel Arc B580 (12 GB), driver 32.0.101.9150, Windows 11, Python 3.10,
torch 2.14.1+xpu, torchaudio 2.11.0+xpu, transformers 4.52.4.

## Setup (uv)

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.14.1 --index-url https://download.pytorch.org/whl/xpu
# torchaudio's newest XPU wheel (2.11) pins torch 2.11; the parts used here are pure
# PyTorch and work on 2.14, so install it without deps.
uv pip install --python .venv\Scripts\python.exe --no-deps torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/xpu
# everything else from requirements.txt EXCEPT torch/torchaudio (so the XPU build is not replaced)
Get-Content requirements.txt | Where-Object { $_ -notmatch '^\s*(torch|torchaudio)==' } | Set-Content -Encoding utf8 $env:TEMP\req_xpu.txt
uv pip install --python .venv\Scripts\python.exe -r $env:TEMP\req_xpu.txt gradio
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
- `webui.py`: `--backend auto|hf|vllm` and `--device`. vLLM is CUDA-only, so on XPU the
  HuggingFace `ConfuciusTTS` backend is used.

## Performance (B580, fp32)

| | before | now |
|---|---|---|
| T2S, beam 3 | 25.5 ms/token | 13.1 ms/token |
| T2S, greedy | 20.5 ms/token | 6.3 ms/token |
| 5–6 s utterance, warm | 12–14 s (~2.3× real time) | 5–6 s (~1× real time) |

The first request with a new length is slower (graph capture + vocoder kernel compile,
~11 s); later requests reuse them. Peak VRAM ~8–10 GB. bf16 autocast did not help.
Remaining time: HF beam-search bookkeeping (~7 ms/token) and the S2A flow-matching stage
(~1.3–2 s). `server.py` and `example_vllm.py` still need vLLM (CUDA only).
