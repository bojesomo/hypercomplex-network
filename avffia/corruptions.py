"""Test-time corruptions for the robustness protocol of U-FFIA (Cui et al., 2024, Sec. V-D / VI).

* Audio: noise added to the whole clip at SNR in {-10, -5, 0, 10, 20} dB (U-FFIA Table III).
  U-FFIA used recorded *bubble and pump* noise. Point ``noise_dir`` at such recordings
  (wav/flac). Without them, white Gaussian noise is used, which is a different and
  not-comparable condition; the result keys say which was used.
* Video: darkness (pixel intensities scaled down) and additive Gaussian noise with variance up
  to 0.2.
* Missing modality (U-FFIA's single-modality test conditions) is handled by the model's
  ``missing`` switch in ``train.robustness``, not by corrupting the inputs.

All corruptions are deterministic given ``seed``, so every model sees identical corrupted inputs.
"""
from __future__ import annotations

import glob
import os
from typing import Callable, Dict, List, Optional

import torch


class AudioNoiseBank:
    def __init__(self, noise_dir: Optional[str], sample_rate: int):
        self.clips: List[torch.Tensor] = []
        if noise_dir:
            import torchaudio
            for f in sorted(glob.glob(os.path.join(noise_dir, '**', '*.wav'), recursive=True) +
                            glob.glob(os.path.join(noise_dir, '**', '*.flac'), recursive=True)):
                w, sr = torchaudio.load(f)
                w = w.mean(0)
                if sr != sample_rate:
                    w = torchaudio.functional.resample(w, sr, sample_rate)
                self.clips.append(w)
        self.kind = 'recorded' if self.clips else 'white'

    def sample(self, n: int, length: int, g: torch.Generator) -> torch.Tensor:
        if not self.clips:
            return torch.randn(n, length, generator=g)
        out = []
        for _ in range(n):
            c = self.clips[int(torch.randint(len(self.clips), (1,), generator=g))]
            if c.numel() < length:
                c = c.repeat(length // c.numel() + 1)
            s = int(torch.randint(c.numel() - length + 1, (1,), generator=g))
            out.append(c[s:s + length])
        return torch.stack(out)


def add_noise_snr(wav: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    p_sig = wav.pow(2).mean(-1, keepdim=True).clamp_min(1e-10)
    p_noise = noise.pow(2).mean(-1, keepdim=True).clamp_min(1e-10)
    return wav + noise * torch.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))


def build_corruptions(sample_rate: int, noise_dir: Optional[str] = None, seed: int = 0,
                      snrs=(-10, -5, 0, 10, 20), dark=(0.5, 0.25), gauss_var=(0.05, 0.1, 0.2)
                      ) -> Dict[str, Callable[[dict, int], dict]]:
    """Return ``{name: fn(batch, batch_idx) -> batch}``. Batches hold ``waveform`` (B, L) and
    ``video`` (B, 3, T, H, W) in [0, 1]."""
    bank = AudioNoiseBank(noise_dir, sample_rate)
    out = {}

    def gen(i):
        return torch.Generator().manual_seed(seed * 100003 + i)

    for snr in snrs:
        def f(b, i, snr=snr):
            w = b['waveform']
            n = bank.sample(w.shape[0], w.shape[-1], gen(i)).to(w.device, w.dtype)
            return {**b, 'waveform': add_noise_snr(w, n, snr)}
        out[f'audio_{bank.kind}_snr{snr:+d}dB'] = f
    for k in dark:
        out[f'video_dark_x{k}'] = lambda b, i, k=k: {**b, 'video': b['video'] * k}
    for var in gauss_var:
        def f(b, i, var=var):
            v = b['video']
            n = torch.randn(v.shape, generator=gen(i)).to(v.device, v.dtype)
            return {**b, 'video': (v + n * var ** 0.5).clamp(0, 1)}
        out[f'video_gauss_var{var}'] = f
    return out
