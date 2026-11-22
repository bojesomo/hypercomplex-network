"""Audio front-ends (run on the GPU, inside the model) and video normalisation helpers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


class LogMel(nn.Module):
    """Waveform ``(B, L)`` to log-mel ``(B, 1, T, F)``, with SpecAugment while training.

    The U-FFIA reference set-up is sr=64 kHz, window 2048, hop 1024 and 64 mel bins
    (its ``fmax=128000`` is above Nyquist, so it is clipped to sr/2 here).
    """

    def __init__(self, sample_rate=64000, n_fft=2048, hop=1024, n_mels=64, fmin=50.0, fmax=None,
                 freq_mask=8, time_mask=16, n_masks=2, normalize='bn'):
        super().__init__()
        fmax = min(fmax or sample_rate / 2, sample_rate / 2)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft, hop_length=hop, f_min=fmin, f_max=fmax,
            n_mels=n_mels, power=2.0, center=True, pad_mode='reflect', norm='slaney', mel_scale='slaney')
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0)
        self.fmask = torchaudio.transforms.FrequencyMasking(freq_mask) if freq_mask else None
        self.tmask = torchaudio.transforms.TimeMasking(time_mask) if time_mask else None
        self.n_masks = n_masks
        # 'bn': learned per-bin BatchNorm over mel bins (as in PANNs), for trainable backbones.
        # 'instance': per-clip standardisation, with no learned statistics (for frozen backbones).
        self.normalize = 'bn' if normalize is True else normalize
        self.bn = nn.BatchNorm2d(n_mels) if self.normalize == 'bn' else None

    def forward(self, wav):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = self.to_db(self.mel(wav.float()))  # (B, F, T)
        if self.training:
            for _ in range(self.n_masks):
                if self.fmask is not None:
                    x = self.fmask(x)
                if self.tmask is not None:
                    x = self.tmask(x)
        x = x.transpose(1, 2).unsqueeze(1)  # (B, 1, T, F)
        if self.bn is not None:
            x = self.bn(x.transpose(1, 3)).transpose(1, 3)
        elif self.normalize == 'instance':
            x = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-5)
        return x


class KaldiFbank(nn.Module):
    """AST/BEATs-style 128-bin Kaldi fbank at 16 kHz: ``(B, L)`` to ``(B, T, 128)``."""

    def __init__(self, in_sample_rate=64000, sample_rate=16000, n_mels=128, mean=-4.2677393, std=4.5689974,
                 freq_mask=24, time_mask=48):
        super().__init__()
        self.resample = (torchaudio.transforms.Resample(in_sample_rate, sample_rate)
                         if in_sample_rate != sample_rate else nn.Identity())
        self.sample_rate, self.n_mels, self.mean, self.std = sample_rate, n_mels, mean, std
        self.fmask = torchaudio.transforms.FrequencyMasking(freq_mask) if freq_mask else None
        self.tmask = torchaudio.transforms.TimeMasking(time_mask) if time_mask else None

    def forward(self, wav):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            wav = self.resample(wav.float())
            feats = []
            for w in wav:
                w = w - w.mean()
                feats.append(torchaudio.compliance.kaldi.fbank(
                    w.unsqueeze(0), htk_compat=True, sample_frequency=self.sample_rate, use_energy=False,
                    window_type='hanning', num_mel_bins=self.n_mels, dither=0.0, frame_shift=10))
            x = torch.stack(feats)  # (B, T, F)
        if self.training:
            x = x.transpose(1, 2)
            if self.fmask is not None:
                x = self.fmask(x)
            if self.tmask is not None:
                x = self.tmask(x)
            x = x.transpose(1, 2)
        return (x - self.mean) / (2 * self.std)


class VideoNormalize(nn.Module):
    """Video in ``[0, 1]``, shape ``(B, C, T, H, W)``: per-channel normalisation and an optional resize."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD, size=None):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1, 1), persistent=False)
        self.size = size

    def forward(self, x):
        if self.size is not None and tuple(x.shape[-2:]) != tuple(self.size):
            B, C, T, H, W = x.shape
            x = F.interpolate(x.transpose(1, 2).reshape(B * T, C, H, W), size=self.size, mode='bilinear',
                              align_corners=False).view(B, T, C, *self.size).transpose(1, 2)
        return (x - self.mean) / self.std
