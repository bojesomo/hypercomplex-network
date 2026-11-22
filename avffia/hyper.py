"""Hypercomplex building blocks for the AV-FFIA models.

All the parameterised layers here use the ``fast_hypercomplex`` library from this
repository: a layer with ``n_divs = n`` shares its weights across the ``n``
hypercomplex components (via the Hamilton-like product matrix of that algebra), so it
has ``1/n`` of the parameters of the equivalent real layer.

``n_divs = 1`` falls back to the plain real-valued layer, so every block can be ablated
against its real counterpart with the same code path.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_hypercomplex import HyperConv, HyperLinear
from fast_hypercomplex.ops import fast_hypercomplex

try:  # timm >= 0.9
    from timm.layers import DropPath
except ImportError:  # pragma: no cover
    from timm.models.layers import DropPath


# --------------------------------------------------------------------------------------
# Factories: return a hypercomplex layer when the shapes allow it, otherwise a real one.
# --------------------------------------------------------------------------------------
def _divisible(n: int, *dims: int) -> bool:
    return n > 1 and all(d % n == 0 for d in dims)


def hlinear(in_features: int, out_features: int, n_divs: int = 4, bias: bool = True) -> nn.Module:
    """Hypercomplex linear layer (``1/n_divs`` params), or ``nn.Linear`` if not divisible."""
    if _divisible(n_divs, in_features, out_features):
        return HyperLinear(in_features, out_features, bias=bias, n_divs=n_divs)
    return nn.Linear(in_features, out_features, bias=bias)


def hconv(in_channels: int, out_channels: int, kernel_size, dim: int = 2, n_divs: int = 4,
          stride=1, padding=0, bias: bool = True) -> nn.Module:
    """Hypercomplex 1d/2d/3d convolution, or the real ``nn.ConvNd`` if not divisible."""
    if _divisible(n_divs, in_channels, out_channels):
        return HyperConv(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         n_divs=n_divs, bias=bias, operation=f'convolution{dim}d')
    conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]
    return conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)


# --------------------------------------------------------------------------------------
# Deployment: fold hypercomplex weights into ordinary dense layers (no runtime overhead).
# --------------------------------------------------------------------------------------
@torch.no_grad()
def reparameterize_hypercomplex(model: nn.Module) -> nn.Module:
    """Replace every ``HyperLinear`` / ``HyperConv`` with an equivalent ``nn.Linear`` /
    ``nn.ConvNd`` holding the expanded weight, in place.

    Training keeps the ``1/n`` parameter count (it is what is stored and optimised). For
    inference, building the Hamilton kernel on every forward pass is wasted work, so this
    turns it into a plain dense kernel. The outputs are identical.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, HyperLinear):
            w = fast_hypercomplex(child.weight, child.n_divs, comp_mat=child.comp_mat)
            new = nn.Linear(w.shape[1], w.shape[0], bias=child.bias is not None)
            new.weight.copy_(w)
            if child.bias is not None:
                new.bias.copy_(child.bias)
            setattr(model, name, new.to(w.device))
        elif isinstance(child, HyperConv):
            w = fast_hypercomplex(child.weight, child.n_divs, comp_mat=child.comp_mat)
            dim = len(child.kernel_size)
            conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]
            new = conv(w.shape[1] * child.groups, w.shape[0], child.kernel_size, stride=child.stride,
                       padding=child.padding, dilation=child.dilation, groups=child.groups,
                       bias=child.bias is not None)
            new.weight.copy_(w)
            if child.bias is not None:
                new.bias.copy_(child.bias)
            setattr(model, name, new.to(w.device))
        else:
            reparameterize_hypercomplex(child)
    return model


# --------------------------------------------------------------------------------------
# Transformer-style blocks
# --------------------------------------------------------------------------------------
class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class HyperMlp(nn.Module):
    def __init__(self, dim: int, hidden: int, n_divs: int = 4, drop: float = 0.0, out_dim: Optional[int] = None):
        super().__init__()
        self.fc1 = hlinear(dim, hidden, n_divs)
        self.act = nn.GELU()
        self.fc2 = hlinear(hidden, out_dim or dim, n_divs)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class HyperAttention(nn.Module):
    """Multi-head (self- or cross-) attention whose q/k/v/out projections are hypercomplex."""

    def __init__(self, dim: int, num_heads: int = 4, n_divs: int = 4, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.q = hlinear(dim, dim, n_divs, bias=qkv_bias)
        self.kv = hlinear(dim, 2 * dim, n_divs, bias=qkv_bias)
        self.proj = hlinear(dim, dim, n_divs)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        context = x if context is None else context
        h = self.num_heads
        q = self.q(x).view(B, N, h, C // h).transpose(1, 2)
        k, v = self.kv(context).view(B, context.shape[1], 2, h, C // h).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class HyperBlock(nn.Module):
    """Pre-norm transformer block (optionally with cross-attention) using hypercomplex layers."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, n_divs: int = 4,
                 drop: float = 0.0, drop_path: float = 0.0, cross: bool = False, ls_init: float = 1e-4):
        super().__init__()
        self.cross = cross
        self.norm1 = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim) if cross else nn.Identity()
        self.attn = HyperAttention(dim, num_heads, n_divs, proj_drop=drop)
        self.ls1 = LayerScale(dim, ls_init)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = HyperMlp(dim, int(dim * mlp_ratio), n_divs, drop)
        self.ls2 = LayerScale(dim, ls_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, context: Optional[torch.Tensor] = None):
        ctx = self.norm_ctx(context) if (self.cross and context is not None) else None
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), ctx)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


# --------------------------------------------------------------------------------------
# Convolutional blocks for the lightweight hypercomplex student backbones
# --------------------------------------------------------------------------------------
class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, dim: int, ratio: int = 4, n_divs: int = 4):
        super().__init__()
        hidden = max(8, channels // ratio)
        hidden = int(math.ceil(hidden / n_divs) * n_divs) if n_divs > 1 else hidden
        self.dim = dim
        self.fc1 = hlinear(channels, hidden, n_divs)
        self.fc2 = hlinear(hidden, channels, n_divs)

    def forward(self, x):
        s = x.mean(dim=tuple(range(2, x.dim())))
        s = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))
        return x * s.view(*s.shape, *([1] * (x.dim() - 2)))


class HyperInvertedResidual(nn.Module):
    """MobileNetV2/EfficientNet-style inverted residual, 2D or 3D.

    The two point-wise (1x1) convolutions hold almost all the parameters, so these are the
    hypercomplex ones. The depth-wise convolution is already cheap and stays real.
    """

    def __init__(self, cin: int, cout: int, stride=1, expand: int = 4, kernel=3, dim: int = 2,
                 n_divs: int = 4, se: bool = True, drop_path: float = 0.0):
        super().__init__()
        hidden = cin * expand
        bn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        dw = nn.Conv2d if dim == 2 else nn.Conv3d
        if isinstance(kernel, int):
            kernel = (kernel,) * dim
        if isinstance(stride, int):
            stride = (stride,) * dim
        pad = tuple(k // 2 for k in kernel)
        self.use_res = cin == cout and all(s == 1 for s in stride)
        self.pw = nn.Sequential(hconv(cin, hidden, 1, dim, n_divs, bias=False), bn(hidden), nn.SiLU())
        self.dw = nn.Sequential(dw(hidden, hidden, kernel, stride, pad, groups=hidden, bias=False), bn(hidden), nn.SiLU())
        self.se = SqueezeExcite(hidden, dim, n_divs=n_divs) if se else nn.Identity()
        self.pwl = nn.Sequential(hconv(hidden, cout, 1, dim, n_divs, bias=False), bn(cout))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        y = self.pwl(self.se(self.dw(self.pw(x))))
        return x + self.drop_path(y) if self.use_res else y


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
