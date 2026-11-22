"""Training objectives: supervised AV loss and multi-level knowledge distillation."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyper import hlinear


class AVLoss(nn.Module):
    """Cross-entropy on the fused logits, plus down-weighted audio-only and video-only auxiliary losses.

    The auxiliary heads make each branch learn on its own, which counters modality imbalance:
    without them the video stream tends to dominate and the audio branch under-trains.
    """

    def __init__(self, aux_weight: float = 0.3, label_smoothing: float = 0.1):
        super().__init__()
        self.aux_weight = aux_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {'ce': self.ce(out['logits'], target)}
        if self.aux_weight > 0 and 'logits_a' in out:
            losses['aux'] = self.aux_weight * 0.5 * (self.ce(out['logits_a'], target) + self.ce(out['logits_v'], target))
        return losses


def kd_kl(student_logits, teacher_logits, T: float, style: str = 'hinton'):
    """``hinton``: both sides softened by T, scaled by T^2 (Hinton et al., 2015).
    ``uffia``: only the teacher is softened, KL(softmax(t / T) || softmax(s)), as in U-FFIA eq. (9)
    (Cui et al., 2024). They argue this suits cross-model KD, where the logit scales differ."""
    if style == 'uffia':
        return F.kl_div(F.log_softmax(student_logits, -1), F.softmax(teacher_logits / T, -1), reduction='batchmean')
    return F.kl_div(F.log_softmax(student_logits / T, -1), F.softmax(teacher_logits / T, -1),
                    reduction='batchmean') * T * T


class DistillLoss(nn.Module):
    """Teacher to student distillation at three levels.

    1. **Logits**: temperature-scaled KL on the fused logits, plus each unimodal auxiliary head
       distilled from the teacher's matching head (audio to audio, video to video).
    2. **Embeddings**: cosine alignment of the pooled fused, audio and video features, through
       small hypercomplex projectors (student dim to teacher dim).
    3. **Relations** (optional): match the within-batch similarity structure of the fused
       embeddings (RKD-style). This needs no projector and is robust to dimension mismatch.
    """

    def __init__(self, student_dim: int, teacher_dim: int, n_divs: int = 4, T: float = 2.0, alpha: float = 0.7,
                 beta: float = 1.0, gamma: float = 0.5, unimodal_weight: float = 0.5, style: str = 'hinton'):
        # U-FFIA's own setting is style='uffia', T=2.5, with CE and KD weighted 0.5 / 0.5 (alpha=1.0)
        super().__init__()
        self.style = style
        self.T, self.alpha, self.beta, self.gamma, self.uw = T, alpha, beta, gamma, unimodal_weight
        self.proj = nn.ModuleDict({k: hlinear(student_dim, teacher_dim, n_divs) for k in ('feat', 'feat_a', 'feat_v')})

    @staticmethod
    def _relation(x):
        x = F.normalize(x, dim=-1)
        return x @ x.t()

    def forward(self, s: Dict[str, torch.Tensor], t: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        kd = kd_kl(s['logits'], t['logits'], self.T, self.style)
        if 'logits_a' in s and 'logits_a' in t:
            kd = kd + self.uw * 0.5 * (kd_kl(s['logits_a'], t['logits_a'], self.T, self.style) +
                                       kd_kl(s['logits_v'], t['logits_v'], self.T, self.style))
        losses['kd'] = self.alpha * kd
        if self.beta > 0:
            feat = 0.0
            for k, p in self.proj.items():
                feat = feat + (1 - F.cosine_similarity(p(s[k]), t[k].detach(), dim=-1)).mean()
            losses['feat'] = self.beta * feat / len(self.proj)
        if self.gamma > 0 and s['feat'].shape[0] > 1:
            losses['rel'] = self.gamma * F.smooth_l1_loss(self._relation(s['feat']), self._relation(t['feat']).detach())
        return losses
