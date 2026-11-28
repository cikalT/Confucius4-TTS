"""Setup self-test: download all models, load them on the GPU and synthesize one sentence.

Run by 1_setup.bat. Exit code 0 means the install is complete and working.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def synthetic_reference(path: Path, sr: int = 16000, seconds: float = 6.0):
    """A voice-like test signal (pitched syllables with pauses); no real speaker needed."""
    t = np.arange(int(sr * seconds)) / sr
    voiced = ((t > 0.5) & (t < 2.8)) | ((t > 3.3) & (t < 5.6))
    env = voiced * (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)) * (0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 0.37 * t)))
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.8 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    rng = np.random.default_rng(0)
    wav = (np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.2 * rng.uniform(-1, 1, t.size)) * env * 0.35
    import soundfile as sf
    sf.write(str(path), wav.astype(np.float32), sr)


def main():
    from confuciustts.utils.common import get_default_device

    device = get_default_device()
    print(f"  torch {torch.__version__}, device: {device}", end="")
    if device == "xpu":
        print(f" ({torch.xpu.get_device_name(0)}), XPUGraph: {hasattr(torch.xpu, 'XPUGraph')}")
    else:
        print()

    print("  Mengunduh & memuat model (pertama kali sekitar 5 GB, bisa beberapa menit) ...", flush=True)
    t0 = time.time()
    from confuciustts.cli.inference import ConfuciusTTS
    from confuciustts.utils.audio_io import save_audio
    model = ConfuciusTTS(config_path="config/inference_config.yaml", device=device)
    print(f"  Model siap dalam {time.time() - t0:.0f} detik.", flush=True)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    ref = out_dir / "selftest_reference.wav"
    synthetic_reference(ref)

    print("  Tes generate satu kalimat ...", flush=True)
    t0 = time.time()
    audio = model.generate("Halo, instalasi berhasil dan siap dipakai.", "id", str(ref))
    elapsed = time.time() - t0
    dur = audio.shape[-1] / model.sample_rate
    if audio.numel() == 0 or torch.isnan(audio).any() or float(audio.abs().max()) < 1e-4:
        print("  [GAGAL] Hasil audio kosong atau rusak.")
        return 1
    out = out_dir / "selftest.wav"
    save_audio(out, audio, model.sample_rate)
    print(f"  OK: {dur:.1f} detik audio dalam {elapsed:.1f} detik -> {out.relative_to(ROOT)}")
    if device == "cpu":
        print("  [PERINGATAN] Berjalan di CPU. Generate akan sangat lambat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
