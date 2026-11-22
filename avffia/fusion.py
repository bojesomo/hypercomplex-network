"""Hypercomplex audio-visual fusion blocks.

Every fusion block updates a small set of shared *bottleneck* tokens ``Z`` (B, Nb, D) from the
audio tokens ``A`` (B, Na, D) and video tokens ``V`` (B, Nv, D) of one backbone stage. Stacking
blocks at several stages gives early, mid, late or progressive fusion (see
``HyperAVNet.fusion_stages``).

* ``hcf``: **Hypercomplex Component Fusion** (the proposed block). ``Z`` cross-attends to each
  modality separately, giving ``za`` and ``zv``. Then the modality-specific and interaction
  views are treated as the components of one hypercomplex number,

      q = Z e0 + za e1 + zv e2 + (za * zv) e3          (quaternion, algebra = 4)
        (+ (za - zv) e4 + (Z*za) e5 + (Z*zv) e6 + (Z*za*zv) e7  for octonion, algebra = 8)

  and are mixed by one hypercomplex linear layer. The Hamilton product makes every output
  component a signed combination of every input component, using one shared set of weights,
  so cross-modal interactions come at ``1/algebra`` of the parameter cost of a dense mixer.
* ``mbt``: an MBT-style attention bottleneck (Nagrani et al., 2021; also used as the U-FFIA
  baseline) built from hypercomplex transformer blocks.
* ``concat``: a gated concatenation baseline (real or hypercomplex linear).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .hyper import DropPath, HyperAttention, HyperBlock, HyperMlp, LayerScale, hlinear


class HyperComponentFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, n_divs: int = 4, algebra: int = 4, mlp_ratio: float = 2.0,
                 drop: float = 0.0, drop_path: float = 0.0, ls_init: float = 1e-2):
        super().__init__()
        assert algebra in (2, 4, 8), 'component algebra must be complex (2), quaternion (4) or octonion (8)'
        self.algebra = algebra
        self.norm_z = nn.LayerNorm(dim)
        self.norm_a, self.norm_v = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn_a = HyperAttention(dim, num_heads, n_divs, proj_drop=drop)
        self.attn_v = HyperAttention(dim, num_heads, n_divs, proj_drop=drop)
        self.norm_c = nn.LayerNorm(dim)
        # The component mixer: one Hamilton product over the `algebra` components.
        # With n_divs == 1 (the real-valued ablation) it is a dense layer.
        self.mix = hlinear(algebra * dim, dim, algebra if n_divs > 1 else 1)
        self.ls_mix = LayerScale(dim, ls_init)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = HyperMlp(dim, int(dim * mlp_ratio), n_divs, drop)
        self.ls_mlp = LayerScale(dim, ls_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def components(self, z, za, zv):
        n = self.norm_c
        comps = [z, za, zv, n(za) * n(zv), za - zv, n(z) * n(za), n(z) * n(zv), n(z) * n(za) * n(zv)]
        if self.algebra == 2:
            return [za, zv]
        return comps[:self.algebra]

    def forward(self, z, a, v):
        zn = self.norm_z(z)
        za = z + self.attn_a(zn, self.norm_a(a))
        zv = z + self.attn_v(zn, self.norm_v(v))
        q = torch.cat(self.components(z, za, zv), dim=-1)
        z = z + self.drop_path(self.ls_mix(self.mix(q)))
        z = z + self.drop_path(self.ls_mlp(self.mlp(self.norm_m(z))))
        return z


class HyperMBTFusion(nn.Module):
    """Attention bottleneck: ``[A; Z]`` and ``[V; Z]`` each pass through a hypercomplex transformer
    block, and the two bottleneck updates are averaged."""

    def __init__(self, dim: int, num_heads: int = 4, n_divs: int = 4, mlp_ratio: float = 2.0,
                 drop: float = 0.0, drop_path: float = 0.0, **_):
        super().__init__()
        self.block_a = HyperBlock(dim, num_heads, mlp_ratio, n_divs, drop, drop_path, ls_init=1e-2)
        self.block_v = HyperBlock(dim, num_heads, mlp_ratio, n_divs, drop, drop_path, ls_init=1e-2)

    def forward(self, z, a, v):
        nb = z.shape[1]
        za = self.block_a(torch.cat([a, z], 1))[:, -nb:]
        zv = self.block_v(torch.cat([v, z], 1))[:, -nb:]
        return 0.5 * (za + zv)


class ConcatFusion(nn.Module):
    """Baseline: pooled ``[Z, mean(A), mean(V)]`` through a gated (hyper)linear layer."""

    def __init__(self, dim: int, n_divs: int = 4, **_):
        super().__init__()
        self.norm = nn.LayerNorm(3 * dim)
        self.fc = hlinear(3 * dim, 2 * dim, n_divs)

    def forward(self, z, a, v):
        g = torch.cat([z, a.mean(1, keepdim=True).expand_as(z), v.mean(1, keepdim=True).expand_as(z)], -1)
        h, gate = self.fc(self.norm(g)).chunk(2, -1)
        return z + h * torch.sigmoid(gate)


FUSIONS = {'hcf': HyperComponentFusion, 'mbt': HyperMBTFusion, 'concat': ConcatFusion}


def build_fusion(kind: str, **kw) -> nn.Module:
    return FUSIONS[kind](**kw)
