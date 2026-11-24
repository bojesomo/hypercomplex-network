"""HyperAVNet: a hypercomplex audio-visual network for fish feeding intensity assessment.

Layout::

    waveform --> audio backbone --stage tokens a_0 ... a_{S-1}--+
                                                                |  hypercomplex projections
    video ----> video backbone --stage tokens v_0 ... v_{S-1}--+  (C_s -> D, 1/n params)
                                                                |
          bottleneck tokens Z --[fusion @ s in fusion_stages]--> Z'   (hcf | mbt | concat)
                      |                   (stage-wise students: Z' is fed back into both streams)
                      +--> head(Z, a_last, v_last) -> logits     (+ audio-only / video-only aux heads)

``fusion_stages`` is the ablation axis from deep to early: ``[3]`` is late fusion only,
``[2, 3]``, then ``[1, 2, 3]``, then ``[0, 1, 2, 3]`` is progressive fusion from the first stage
on. Indices can be negative (``-1`` is the last stage).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .backbones import Backbone, build_backbone
from .fusion import build_fusion
from .hyper import hlinear


class StageProjector(nn.Module):
    """Stage tokens ``(B, N, C_s)`` to ``(B, N, D)``: hypercomplex linear, LayerNorm and a modality embedding."""

    def __init__(self, cin: int, dim: int, n_divs: int):
        super().__init__()
        self.proj = hlinear(cin, dim, n_divs)
        self.norm = nn.LayerNorm(dim)
        self.mod = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mod, std=0.02)

    def forward(self, x):
        return self.norm(self.proj(x)) + self.mod


class HyperAVNet(nn.Module):
    def __init__(self, audio: dict, video: dict, num_classes: int = 4, dim: int = 256, n_divs: int = 4,
                 fusion: str = 'hcf', fusion_stages: Sequence[int] = (-1,), fusion_algebra: int = 4,
                 num_bottleneck: int = 8, num_heads: int = 4, mlp_ratio: float = 2.0, drop: float = 0.0,
                 drop_path: float = 0.1, inject: bool = True, modality_dropout: float = 0.1,
                 head_dropout: float = 0.2, aux_heads: bool = True):
        super().__init__()
        self.audio: Backbone = build_backbone(audio)
        self.video: Backbone = build_backbone(video)
        Sa, Sv = len(self.audio.stage_dims), len(self.video.stage_dims)
        assert Sa == Sv, f'audio ({Sa}) and video ({Sv}) backbones must expose the same number of stages'
        self.num_stages = S = Sa
        self.fusion_stages = sorted({s % S for s in fusion_stages})
        self.proj_stages = sorted(set(self.fusion_stages) | {S - 1})
        self.inject = inject and self.audio.stagewise and self.video.stagewise
        self.modality_dropout = modality_dropout
        self.missing: Optional[str] = None  # set to 'audio' / 'video' for missing-modality tests
        self.dim = dim

        self.proj_a = nn.ModuleDict({str(s): StageProjector(self.audio.stage_dims[s], dim, n_divs)
                                     for s in self.proj_stages})
        self.proj_v = nn.ModuleDict({str(s): StageProjector(self.video.stage_dims[s], dim, n_divs)
                                     for s in self.proj_stages})
        self.bottleneck = nn.Parameter(torch.zeros(1, num_bottleneck, dim))
        nn.init.trunc_normal_(self.bottleneck, std=0.02)
        dpr = torch.linspace(0, drop_path, len(self.fusion_stages)).tolist() if self.fusion_stages else []
        self.fuse = nn.ModuleDict({
            str(s): build_fusion(fusion, dim=dim, num_heads=num_heads, n_divs=n_divs, algebra=fusion_algebra,
                                 mlp_ratio=mlp_ratio, drop=drop, drop_path=dpr[i])
            for i, s in enumerate(self.fusion_stages)})

        if self.inject:  # cross-modal feedback into the backbone streams (zero-initialised = identity)
            self.inj_a = nn.ModuleDict({str(s): hlinear(dim, self.audio.stage_dims[s], n_divs)
                                        for s in self.fusion_stages if s < S - 1})
            self.inj_v = nn.ModuleDict({str(s): hlinear(dim, self.video.stage_dims[s], n_divs)
                                        for s in self.fusion_stages if s < S - 1})
            for m in list(self.inj_a.values()) + list(self.inj_v.values()):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        self.norm_z, self.norm_a, self.norm_v = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.head = nn.Sequential(hlinear(3 * dim, dim, n_divs), nn.GELU(), nn.Dropout(head_dropout),
                                  nn.Linear(dim, num_classes))
        self.aux_heads = aux_heads
        if aux_heads:
            self.head_a = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(dim, num_classes))
            self.head_v = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(dim, num_classes))

    # ------------------------------------------------------------------------------
    def _keep_masks(self, B: int, device) -> Optional[tuple]:
        """Per-sample keep masks ``(keep_a, keep_v)`` of shape (B, 1, 1), or None (keep all).

        * Training: modality dropout. Each sample drops audio with probability p and video with
          probability p, never both; in U-FFIA's notation p_av = 1 - 2p and p_a = p_v = p.
        * ``self.missing in {'audio', 'video'}``: that modality is treated as absent (test-time
          missing-modality evaluation).

        The same mask is applied at every fusion input and at the fused head, so a modality
        that is "missing" is missing everywhere. The auxiliary unimodal heads always see their
        own modality.
        """
        if self.missing in ('audio', 'video'):
            keep_a = torch.full((B, 1, 1), float(self.missing != 'audio'), device=device)
            keep_v = torch.full((B, 1, 1), float(self.missing != 'video'), device=device)
            return keep_a, keep_v
        if not self.training or self.modality_dropout <= 0:
            return None
        r = torch.rand(B, device=device)
        keep_a = (r >= self.modality_dropout).float().view(B, 1, 1)
        keep_v = ((r < self.modality_dropout) | (r >= 2 * self.modality_dropout)).float().view(B, 1, 1)
        return keep_a, keep_v

    def _fuse(self, s: int, z, a_tok, v_tok, keep):
        a, v = self.proj_a[str(s)](a_tok), self.proj_v[str(s)](v_tok)
        if keep is not None:
            a, v = a * keep[0], v * keep[1]
        return self.fuse[str(s)](z, a, v)

    def _heads(self, z, a_last, v_last, fused_tokens, keep) -> Dict[str, torch.Tensor]:
        zf = self.norm_z(z).mean(1)
        af = self.norm_a(a_last).mean(1)
        vf = self.norm_v(v_last).mean(1)
        af_f, vf_f = (af, vf) if keep is None else (af * keep[0][:, 0], vf * keep[1][:, 0])
        out = {'logits': self.head(torch.cat([zf, af_f, vf_f], -1)), 'feat': zf, 'feat_a': af, 'feat_v': vf,
               'fused_tokens': fused_tokens}
        if self.aux_heads:
            out['logits_a'] = self.head_a(af)
            out['logits_v'] = self.head_v(vf)
        return out

    # ------------------------------------------------------------------------------
    def forward_from_tokens(self, a_stages: List[torch.Tensor], v_stages: List[torch.Tensor]):
        """Side-network fusion over precomputed (or frozen) backbone stage tokens."""
        B = a_stages[0].shape[0]
        keep = self._keep_masks(B, a_stages[0].device)
        z = self.bottleneck.expand(B, -1, -1)
        fused = {}
        for s in self.fusion_stages:
            z = self._fuse(s, z, a_stages[s], v_stages[s], keep)
            fused[s] = z
        S = self.num_stages - 1
        a_last = self.proj_a[str(S)](a_stages[S])
        v_last = self.proj_v[str(S)](v_stages[S])
        return self._heads(z, a_last, v_last, fused, keep)

    def forward(self, waveform: torch.Tensor, video: torch.Tensor):
        if not self.inject:
            return self.forward_from_tokens(self.audio(waveform), self.video(video))

        # Stage-wise interleaved execution with cross-modal feedback (early fusion).
        ha, hv = self.audio.prepare(waveform), self.video.prepare(video)
        keep = self._keep_masks(waveform.shape[0], waveform.device)
        z = self.bottleneck.expand(waveform.shape[0], -1, -1)
        fused = {}
        S = self.num_stages
        for s in range(S):
            ha, hv = self.audio.run_stage(s, ha), self.video.run_stage(s, hv)
            if str(s) in self.fuse:
                z = self._fuse(s, z, self.audio.tokens(s, ha), self.video.tokens(s, hv), keep)
                fused[s] = z
                if str(s) in self.inj_a:
                    g = z.mean(1)
                    ha = self.audio.modulate(ha, torch.tanh(self.inj_a[str(s)](g)))
                    hv = self.video.modulate(hv, torch.tanh(self.inj_v[str(s)](g)))
        a_last = self.proj_a[str(S - 1)](self.audio.tokens(S - 1, ha))
        v_last = self.proj_v[str(S - 1)](self.video.tokens(S - 1, hv))
        return self._heads(z, a_last, v_last, fused, keep)


def build_model(cfg: dict) -> nn.Module:
    """``arch: hyperavnet`` (default) or ``arch: mixnet`` (pretrained CNNs + cross-modal mixers)."""
    cfg = dict(cfg)
    arch = cfg.pop('arch', 'hyperavnet')
    if arch == 'mixnet':
        from .mixnet import AVMixNet
        return AVMixNet(**cfg)
    return HyperAVNet(**cfg)
