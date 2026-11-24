"""AV-MixNet: pretrained audio and video CNNs with cross-modal mixers between their stages.

    waveform -> audio CNN stage 0 -> ... -> stage s -> [mixer_s] -> stage s+1 -> ...
    video    -> video CNN stage 0 -> ... -> stage s -> [mixer_s] -> stage s+1 -> ...
                                                         |
                                       summary tokens (B, T, D) per mixed stage
                                                         |
              multi-scale head (SegFormer-style all-MLP aggregation over mixed stages) -> logits

* The mixers (avffia.cnn_mixer.CrossModalMixer) write back into both feature maps, so later
  stages of each CNN see cross-modal information (early/mid fusion, not side fusion).
* Their output scales follow a depth schedule (scale_start for the first mixer to scale_end for
  the last) and are learnable; the learned values are reported by ``mixer_scales()``.
* Same output dictionary as HyperAVNet (logits, logits_a/v, feat*), so train.py, distillation,
  robustness and the paper scripts work unchanged. Modality dropout and the ``missing`` switch
  behave as in HyperAVNet.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .cnn_mixer import CrossModalMixer
from .hyper import hlinear


class AVMixNet(nn.Module):
    def __init__(self, audio: dict, video: dict, num_classes: int = 4, dim: int = 192, n_divs: int = 4,
                 mix_stages: Sequence[int] = (1, 2, 3), mixer: Optional[dict] = None,
                 scale_start: float = 0.01, scale_end: float = 0.1, head_time: int = 8,
                 modality_dropout: float = 0.1, head_dropout: float = 0.2, aux_heads: bool = True, num_heads: int = 4):
        super().__init__()
        self.audio = build_backbone(audio)
        self.video = build_backbone(video)
        for name, b in (('audio', self.audio), ('video', self.video)):
            assert b.stagewise and b.map_layout, f'{name} backbone must be stage-wise with a map_layout'
        S = len(self.audio.stage_dims)
        assert S == len(self.video.stage_dims), 'audio and video backbones need the same number of stages'
        self.num_stages = S
        self.mix_stages = sorted({s % S for s in mix_stages})
        self.dim, self.head_time = dim, head_time
        self.modality_dropout = modality_dropout
        self.missing: Optional[str] = None

        mcfg = dict(mixer or {})
        k = len(self.mix_stages)
        scales = [scale_start + (scale_end - scale_start) * i / max(1, k - 1) for i in range(k)]
        self.mixers = nn.ModuleDict({
            str(s): CrossModalMixer(self.audio.stage_dims[s], self.video.stage_dims[s], dim, n_divs, num_heads,
                                    self.audio.map_layout, 'thw', scale_init=scales[i], **mcfg)
            for i, s in enumerate(self.mix_stages)})

        # multi-scale aggregation of the mixed stages (SegFormer all-MLP decoder idea)
        self.agg = nn.Sequential(nn.LayerNorm(dim * max(1, k)), hlinear(dim * max(1, k), dim, n_divs), nn.GELU())
        # last-stage unimodal features
        self.proj_a = nn.Sequential(hlinear(self.audio.stage_dims[-1], dim, n_divs), nn.LayerNorm(dim))
        self.proj_v = nn.Sequential(hlinear(self.video.stage_dims[-1], dim, n_divs), nn.LayerNorm(dim))
        self.norm_z = nn.LayerNorm(dim)
        self.head = nn.Sequential(hlinear(3 * dim, dim, n_divs), nn.GELU(), nn.Dropout(head_dropout),
                                  nn.Linear(dim, num_classes))
        self.aux_heads = aux_heads
        if aux_heads:
            self.head_a = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(dim, num_classes))
            self.head_v = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(dim, num_classes))

    # ------------------------------------------------------------------ helpers
    def _keep_masks(self, B, device):
        """Same semantics as HyperAVNet._keep_masks: returns (keep_a, keep_v) of shape (B,) or None."""
        if self.missing in ('audio', 'video'):
            return (torch.full((B,), float(self.missing != 'audio'), device=device),
                    torch.full((B,), float(self.missing != 'video'), device=device))
        if not self.training or self.modality_dropout <= 0:
            return None
        r = torch.rand(B, device=device)
        return ((r >= self.modality_dropout).float(),
                ((r < self.modality_dropout) | (r >= 2 * self.modality_dropout)).float())

    def mixer_scales(self) -> Dict[str, Dict[str, float]]:
        """Mean learned output scale per mixer and modality: where the model chose to fuse."""
        return {s: {'audio': m.scale_a.detach().abs().mean().item(), 'video': m.scale_v.detach().abs().mean().item()}
                for s, m in self.mixers.items()}

    def set_record_maps(self, flag: bool = True):
        for m in self.mixers.values():
            m.record_maps = flag

    # ------------------------------------------------------------------ forward
    def forward(self, waveform: torch.Tensor, video: torch.Tensor):
        B = waveform.shape[0]
        keep = self._keep_masks(B, waveform.device)
        ha, hv = self.audio.prepare(waveform), self.video.prepare(video)
        summaries: Dict[int, torch.Tensor] = {}
        for s in range(self.num_stages):
            ha, hv = self.audio.run_stage(s, ha), self.video.run_stage(s, hv)
            if str(s) in self.mixers:
                ha, hv, summaries[s] = self.mixers[str(s)](ha, hv, keep)

        # multi-scale head over the mixed stages
        seqs = [F.interpolate(summaries[s].transpose(1, 2), size=self.head_time, mode='linear',
                              align_corners=False).transpose(1, 2) for s in self.mix_stages]
        z = self.agg(torch.cat(seqs, -1))                                     # (B, head_time, D)
        zf = self.norm_z(z.mean(1) + z.amax(1))
        af = self.proj_a(self.audio.tokens(self.num_stages - 1, ha)).mean(1)
        vf = self.proj_v(self.video.tokens(self.num_stages - 1, hv)).mean(1)
        af_f, vf_f = (af, vf) if keep is None else (af * keep[0][:, None], vf * keep[1][:, None])
        out = {'logits': self.head(torch.cat([zf, af_f, vf_f], -1)), 'feat': zf, 'feat_a': af, 'feat_v': vf,
               'fused_tokens': summaries}
        if self.aux_heads:
            out['logits_a'] = self.head_a(af)
            out['logits_v'] = self.head_v(vf)
        return out
