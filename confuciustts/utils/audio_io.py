"""Audio file I/O that does not depend on torchaudio's backend.

torchaudio >= 2.9 routes load/save through torchcodec, which is not available for
every platform/accelerator build (e.g. torch XPU on Windows). soundfile covers
wav/flac/ogg/mp3 (libsndfile >= 1.1). Anything else (m4a/aac, wma, amr, and the audio
track of video files such as mp4/mpeg/mov/mkv/webm) is decoded with PyAV (bundled
FFmpeg) when it is installed; torchaudio is the last fallback.
"""

from typing import Optional

import numpy as np
import soundfile as sf
import torch


def decode_with_av(path: str, target_sr: Optional[int] = None, max_seconds: Optional[float] = None):
    """Decode the first audio stream of any FFmpeg-readable file.

    Returns (mono float32 numpy array, sample_rate). Raises ValueError if the file
    has no audio stream.
    """
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("file tidak berisi trek audio")
        stream = container.streams.audio[0]
        sr = int(target_sr or stream.codec_context.sample_rate or 16000)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        limit = int(max_seconds * sr) if max_seconds else None
        chunks, total = [], 0
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                arr = out.to_ndarray().reshape(-1)
                chunks.append(arr)
                total += len(arr)
            if limit and total >= limit:
                break
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    if limit:
        wav = wav[:limit]
    return wav, sr


def load_audio(path: str):
    """Return (waveform (C, T) float32 tensor, sample_rate)."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
        return torch.from_numpy(np.ascontiguousarray(data.T)), sr
    except Exception:
        pass
    try:
        wav, sr = decode_with_av(path)
        return torch.from_numpy(wav).unsqueeze(0), sr
    except ImportError:
        import torchaudio
        return torchaudio.load(str(path))


def save_audio(path: str, wav: torch.Tensor, sample_rate: int):
    """Save a (C, T) or (T,) tensor as a 16-bit PCM file (format from extension)."""
    wav = wav.detach().float().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    sf.write(str(path), wav.clamp(-1, 1).numpy().T, sample_rate, subtype="PCM_16")
