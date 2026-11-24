"""CNN-native cross-modal mixer for pretrained audio and video CNNs (AV-MixNet).

One mixer sits after a stage of both encoders and runs *read -> mix -> write*:

  read   Each feature map is pooled to a small token grid that keeps **time** (audio: T x Fa
         frequency tokens; video: T x g x g spatial tokens), then projected to a shared width D
         by hypercomplex linear layers. Each token gets a modality embedding, a slot embedding, and
         a **shared time embedding**, so audio and video tokens from the same moment line up.
  mix    One of:
           'full' joint self-attention over all audio and video tokens;
           'sra'  SegFormer-style efficient attention: queries are all tokens, keys/values are
                  reduced to one token per (modality, time step);
           'time' time-axis mixing: one token per (modality, time step);
           'hcf'  hypercomplex component fusion: learned time queries read audio and video in
                  parallel and are mixed by one Hamilton product (avffia.fusion).
         'full', 'sra' and 'time' are followed by a Mix-FFN (a depth-wise temporal conv between
         the two linear layers, as in SegFormer).
  write  Any combination of:
           'xattn' each modality's tokens cross-attend to the mixed tokens; the result is
                   projected to the stage's channels and upsampled onto the feature map;
           'film'  a time-resolved, zero-initialised scale-and-shift of the map per channel;
           'guide' cross-modal guidance maps: audio at time t produces a spatial attention map
                   over the video frame (where the feeding sound points), and video at time t
                   produces a frequency map over the spectrogram (which bands matter).
         The sum of the writes is added to each map through a learnable per-channel layer scale.

The output scale is initialised by the model's depth schedule (small for early mixers, larger
for later ones), so training starts close to the two pretrained CNNs.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import HyperComponentFusion
from .hyper import HyperAttention, LayerScale, hlinear


def sinusoidal_time(T: int, dim: int, device) -> torch.Tensor:
    """(T, dim) sinusoidal embedding of normalised time (t + 0.5) / T in [0, 1].
    It depends on position in the clip, not on frame index, so audio and video tokens from
    the same moment get the same embedding even at different frame rates."""
    t = (torch.arange(T, device=device, dtype=torch.float32) + 0.5) / T
    freqs = torch.exp(torch.linspace(0, math.log(1000.0), dim // 2, device=device))
    ang = t[:, None] * freqs[None] * math.pi
    return torch.cat([ang.sin(), ang.cos()], -1)


def _to_canonical(x: torch.Tensor, layout: str) -> torch.Tensor:
    """Audio -> (B, C, T, F); video (B, C, T, H, W) unchanged."""
    return x.transpose(2, 3) if layout == 'ft' else x


def _from_canonical(x: torch.Tensor, layout: str) -> torch.Tensor:
    return x.transpose(2, 3) if layout == 'ft' else x


class MixFFN(nn.Module):
    """SegFormer Mix-FFN over a (B, T, P, D) token grid: linear -> depth-wise conv over time -> GELU -> linear."""

    def __init__(self, dim: int, ratio: float = 2.0, n_divs: int = 4):
        super().__init__()
        hidden = int(dim * ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = hlinear(dim, hidden, n_divs)
        self.dw = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = hlinear(hidden, dim, n_divs)

    def forward(self, x):  # (B, T, P, D)
        B, T, P, D = x.shape
        h = self.fc1(self.norm(x))
        h = self.dw(h.permute(0, 2, 3, 1).reshape(B * P, -1, T)).reshape(B, P, -1, T).permute(0, 3, 1, 2)
        return self.fc2(F.gelu(h))


class CrossModalMixer(nn.Module):
    def __init__(self, c_a: int, c_v: int, dim: int = 192, n_divs: int = 4, num_heads: int = 4,
                 layout_a: str = 'ft', layout_v: str = 'thw', mix: str = 'sra',
                 write: Sequence[str] = ('xattn', 'film'), time_tokens: int = 16, audio_freq_tokens: int = 2,
                 video_grid: int = 2, scale_init: float = 0.1, hcf_algebra: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        assert mix in ('full', 'sra', 'time', 'hcf'), mix
        assert set(write) <= {'xattn', 'film', 'guide'} and write, write
        self.mix, self.write = mix, tuple(write)
        self.layout_a, self.layout_v = layout_a, layout_v
        self.time_tokens = time_tokens
        self.pa = 1 if mix == 'time' else audio_freq_tokens
        self.g = 1 if mix == 'time' else video_grid
        self.pv = self.g * self.g
        self.dim = dim

        # read
        self.read_a = nn.Sequential(hlinear(c_a, dim, n_divs), nn.LayerNorm(dim))
        self.read_v = nn.Sequential(hlinear(c_v, dim, n_divs), nn.LayerNorm(dim))
        self.mod_a = nn.Parameter(torch.zeros(dim))
        self.mod_v = nn.Parameter(torch.zeros(dim))
        self.slot_a = nn.Parameter(torch.zeros(self.pa, dim))
        self.slot_v = nn.Parameter(torch.zeros(self.pv, dim))
        for p in (self.mod_a, self.mod_v, self.slot_a, self.slot_v):
            nn.init.trunc_normal_(p, std=0.02)

        # mix
        if mix == 'hcf':
            self.query_bias = nn.Parameter(torch.zeros(dim))
            self.hcf = HyperComponentFusion(dim, num_heads, n_divs, algebra=hcf_algebra, mlp_ratio=mlp_ratio)
        else:
            self.norm_mix = nn.LayerNorm(dim)
            self.attn = HyperAttention(dim, num_heads, n_divs)
            self.ls_attn = LayerScale(dim, 1e-1)
            self.ffn = MixFFN(dim, mlp_ratio, n_divs)
            self.ls_ffn = LayerScale(dim, 1e-1)

        # write
        if 'xattn' in self.write:
            self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.xattn_a = HyperAttention(dim, num_heads, n_divs)
            self.xattn_v = HyperAttention(dim, num_heads, n_divs)
            self.out_a = hlinear(dim, c_a, n_divs)
            self.out_v = hlinear(dim, c_v, n_divs)
        if 'film' in self.write:
            self.film_a = hlinear(dim, 2 * c_a, n_divs)
            self.film_v = hlinear(dim, 2 * c_v, n_divs)
            for m in (self.film_a, self.film_v):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        if 'guide' in self.write:
            self.guide_a = hlinear(dim, c_a, n_divs)  # video time vector -> audio channel query
            self.guide_v = hlinear(dim, c_v, n_divs)  # audio time vector -> video channel query
        self.scale_a = nn.Parameter(scale_init * torch.ones(c_a))
        self.scale_v = nn.Parameter(scale_init * torch.ones(c_v))
        self.record_maps = False
        self.last_maps: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ read
    def _read(self, a, v, T):
        """a: (B, C, T_a, F); v: (B, C, T_v, H, W) -> token grids (B, T, Pa, D), (B, T, Pv, D)."""
        pa = F.adaptive_avg_pool2d(a, (T, self.pa)) + F.adaptive_max_pool2d(a, (T, self.pa))
        pv = F.adaptive_avg_pool3d(v, (T, self.g, self.g)) + F.adaptive_max_pool3d(v, (T, self.g, self.g))
        ta = self.read_a(pa.permute(0, 2, 3, 1))                               # (B, T, Pa, D)
        tv = self.read_v(pv.flatten(3).permute(0, 2, 3, 1))                    # (B, T, Pv, D)
        te = sinusoidal_time(T, self.dim, a.device)[None, :, None]             # shared time embedding
        ta = ta + te + self.mod_a + self.slot_a[None, None]
        tv = tv + te + self.mod_v + self.slot_v[None, None]
        return ta, tv

    # ------------------------------------------------------------------ mix
    def _mix(self, ta, tv):
        B, T, Pa, D = ta.shape
        if self.mix == 'hcf':
            z = (sinusoidal_time(T, D, ta.device) + self.query_bias)[None].expand(B, -1, -1)
            z = self.hcf(z, ta.flatten(1, 2), tv.flatten(1, 2))               # (B, T, D)
            return z, z, z, z
        u = torch.cat([ta, tv], 2)                                             # (B, T, P, D)
        x = u.flatten(1, 2)
        if self.mix == 'sra':  # keys/values reduced to one token per (modality, time step)
            kv = torch.cat([ta.mean(2), tv.mean(2)], 1)
            x = x + self.ls_attn(self.attn(self.norm_mix(x), self.norm_mix(kv)))
        else:
            x = x + self.ls_attn(self.attn(self.norm_mix(x)))
        u = x.view(B, T, -1, D)
        u = u + self.ls_ffn(self.ffn(u))
        ia, iv = u[:, :, :Pa], u[:, :, Pa:]
        return u.flatten(1, 2), ia.mean(2), iv.mean(2), u.mean(2)

    # ------------------------------------------------------------------ write
    @staticmethod
    def _resize_time(x, T):  # (B, T0, C) -> (B, T, C)
        return x if x.shape[1] == T else F.interpolate(x.transpose(1, 2), size=T, mode='linear',
                                                       align_corners=False).transpose(1, 2)

    def _write(self, x, tokens, own_t, other_t, i_all, which: str):
        """x: canonical map; tokens: this modality's read tokens (B, T, P, D)."""
        B, C = x.shape[:2]
        Tx = x.shape[2]
        bcast = [1] * (x.dim() - 3)  # trailing (F) or (H, W) dims
        delta = 0.0
        if 'xattn' in self.write:
            q = tokens.flatten(1, 2)
            attn = self.xattn_a if which == 'a' else self.xattn_v
            out = (self.out_a if which == 'a' else self.out_v)(attn(self.norm_q(q), self.norm_kv(i_all)))
            T, P = tokens.shape[1], tokens.shape[2]
            grid = out.view(B, T, P, C)
            if which == 'a':
                grid = grid.permute(0, 3, 1, 2)                                 # (B, C, T, Pa)
                delta = delta + F.interpolate(grid, size=x.shape[2:], mode='bilinear', align_corners=False)
            else:
                grid = grid.permute(0, 3, 1, 2).reshape(B, C, T, self.g, self.g)
                delta = delta + F.interpolate(grid, size=x.shape[2:], mode='trilinear', align_corners=False)
        if 'film' in self.write:
            gb = self._resize_time((self.film_a if which == 'a' else self.film_v)(own_t), Tx)
            g_, b_ = gb.chunk(2, -1)                                            # (B, Tx, C)
            g_ = g_.transpose(1, 2).reshape(B, C, Tx, *bcast)
            b_ = b_.transpose(1, 2).reshape(B, C, Tx, *bcast)
            delta = delta + x * g_ + b_
        if 'guide' in self.write:
            q = self._resize_time((self.guide_a if which == 'a' else self.guide_v)(other_t), Tx)
            q = q.transpose(1, 2).reshape(B, C, Tx, *bcast)
            m = torch.sigmoid((x * q).sum(1, keepdim=True) / math.sqrt(C))     # (B, 1, Tx, ...)
            if self.record_maps:
                self.last_maps['audio_freq' if which == 'a' else 'video_spatial'] = m.detach()
            delta = delta + x * (2 * m - 1)
        scale = (self.scale_a if which == 'a' else self.scale_v).view(1, C, *([1] * (x.dim() - 2)))
        return x + scale * delta

    # ------------------------------------------------------------------ forward
    def forward(self, a_map, v_map, keep: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        a = _to_canonical(a_map, self.layout_a)
        v = v_map
        T = max(1, min(self.time_tokens, a.shape[2], v.shape[2]))
        ta, tv = self._read(a, v, T)
        if keep is not None:  # modality dropout / missing modality: the dropped side writes nothing in
            ta = ta * keep[0].view(-1, 1, 1, 1)
            tv = tv * keep[1].view(-1, 1, 1, 1)
        i_all, ia, iv, summary = self._mix(ta, tv)
        a_new = self._write(a, ta, ia, iv, i_all, 'a')
        v_new = self._write(v, tv, iv, ia, i_all, 'v')
        return _from_canonical(a_new, self.layout_a), v_new, summary
