from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def classification_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict:
    pred = logits.argmax(-1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(target.tolist(), pred.tolist()):
        cm[t, p] += 1
    tp = cm.diag().float()
    prec = tp / cm.sum(0).clamp(min=1)
    rec = tp / cm.sum(1).clamp(min=1)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
    return {'acc': (pred == target).float().mean().item(), 'macro_f1': f1.mean().item(),
            'macro_precision': prec.mean().item(), 'macro_recall': rec.mean().item(),
            'per_class_f1': f1.tolist(), 'confusion': cm.tolist()}


def count_flops(model: nn.Module, *inputs) -> int:
    """Forward FLOPs (multiply and add counted separately, i.e. 2 x MACs) from PyTorch's FlopCounterMode.

    Note: the "FLOPs" in the U-FFIA paper's tables are multiply-accumulates (MACs). Checked:
    torchvision S3D at 20x224^2 is 22.4 GMACs here vs 22.8 "GFLOPs" in U-FFIA Table II. Divide
    by 2 (``gmacs`` in train.py's stats) before comparing with U-FFIA numbers."""
    from torch.utils.flop_counter import FlopCounterMode
    was = model.training
    model.eval()
    with FlopCounterMode(display=False) as fc:
        model(*inputs)
    model.train(was)
    return fc.get_total_flops()
