"""Audio file I/O that does not depend on torchaudio's backend.

torchaudio >= 2.9 routes load/save through torchcodec, which is not available for
every platform/accelerator build (e.g. torch XPU on Windows). soundfile covers
wav/flac/ogg/mp3 (libsndfile >= 1.1); torchaudio is used only as a fallback.
"""

import numpy as np
import soundfile as sf
import torch


def load_audio(path: str):
    """Return (waveform (C, T) float32 tensor, sample_rate)."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
        return torch.from_numpy(np.ascontiguousarray(data.T)), sr
    except Exception:
        import torchaudio
        return torchaudio.load(str(path))


def save_audio(path: str, wav: torch.Tensor, sample_rate: int):
    """Save a (C, T) or (T,) tensor as a 16-bit PCM file (format from extension)."""
    wav = wav.detach().float().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    sf.write(str(path), wav.clamp(-1, 1).numpy().T, sample_rate, subtype="PCM_16")
