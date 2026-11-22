"""Audio and video backbones with one shared multi-stage interface.

Every backbone returns a list of token tensors, one per stage, ordered from early (shallow)
to deep: ``[(B, N_0, C_0), ..., (B, N_{S-1}, C_{S-1})]``. ``stage_dims`` gives ``C_s``.
This is what lets the fusion network attach at any depth.

There are two families:

* **Hypercomplex students** (``HyperAudioNet``, ``HyperVideoNet``) are light, trained from
  scratch or by distillation. They are *stage-wise* (``stagewise = True``): the AV model runs
  them one stage at a time and can feed fused cross-modal information back into each stream,
  which gives true early fusion.
* **Pretrained wrappers** (timm, torchvision, Hugging Face) are used as feature extractors,
  frozen or partly fine-tuned. They expose their intermediate stages, and the hypercomplex
  fusion network runs beside them as a side network.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frontends import (IMAGENET_MEAN, IMAGENET_STD, KINETICS_MEAN, KINETICS_STD, KaldiFbank, LogMel,
                        VideoNormalize)
from .hyper import HyperInvertedResidual, hconv


class Backbone(nn.Module):
    stage_dims: List[int]
    stagewise: bool = False

    def __init__(self, freeze: bool = False, unfreeze: Sequence[str] = ()):
        super().__init__()
        self.freeze = freeze
        self.unfreeze = tuple(unfreeze)

    # -- freezing ---------------------------------------------------------------------
    def _apply_freeze(self, module: nn.Module):
        if not self.freeze:
            return
        for name, p in module.named_parameters():
            p.requires_grad = any(u in name for u in self.unfreeze)

    @property
    def fully_frozen(self) -> bool:
        return self.freeze and not self.unfreeze

    def train(self, mode: bool = True):
        super().train(mode)
        if self.fully_frozen:
            # Keep the BN statistics and dropout of a frozen extractor fixed, but let the
            # front-end apply SpecAugment when training.
            for m in self.modules():
                if m is not self:
                    m.training = isinstance(m, (LogMel, KaldiFbank)) and mode
        return self

    def forward(self, x) -> List[torch.Tensor]:
        if self.fully_frozen:
            with torch.no_grad():
                return [t.detach() for t in self.forward_stages(x)]
        return self.forward_stages(x)

    def forward_stages(self, x) -> List[torch.Tensor]:  # pragma: no cover - abstract
        raise NotImplementedError


def _grid_pool_tokens(fmap: torch.Tensor, grid: Optional[int]) -> torch.Tensor:
    """``(B, C, *spatial)`` to ``(B, N, C)``, average-pooling the trailing two dims to ``grid x grid``."""
    if grid is not None:
        if fmap.dim() == 4:
            fmap = F.adaptive_avg_pool2d(fmap, (min(grid, fmap.shape[-2]), min(grid, fmap.shape[-1])))
        elif fmap.dim() == 5:
            fmap = F.adaptive_avg_pool3d(fmap, (fmap.shape[2], min(grid, fmap.shape[-2]), min(grid, fmap.shape[-1])))
    return fmap.flatten(2).transpose(1, 2)


# ======================================================================================
# Hypercomplex student backbones (stage-wise)
# ======================================================================================
class _StagewiseNet(Backbone):
    stagewise = True

    def prepare(self, x):
        raise NotImplementedError

    def run_stage(self, i: int, h):
        return self.stages[i](h)

    def tokens(self, i: int, h) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def modulate(h, gate):
        """Channel-wise cross-modal modulation ``h * (1 + gate)``, with ``gate`` of shape ``(B, C)``."""
        return h * (1 + gate.view(*gate.shape, *([1] * (h.dim() - 2))))

    def forward_stages(self, x):
        h = self.prepare(x)
        out = []
        for i in range(len(self.stages)):
            h = self.run_stage(i, h)
            out.append(self.tokens(i, h))
        return out


class HyperAudioNet(_StagewiseNet):
    """Hypercomplex inverted-residual CNN on log-mel spectrograms.

    Input: waveform ``(B, L)``. Stage tokens are frequency-pooled time frames ``(B, T_s, C_s)``.
    """

    def __init__(self, dims=(48, 96, 160, 256), depths=(2, 2, 3, 2), strides=(2, 2, 2, 1), stem=32,
                 expand=4, n_divs=4, drop_path=0.05, frontend: Optional[dict] = None, **kw):
        super().__init__(**kw)
        self.frontend = LogMel(**(frontend or {}))
        self.stem = nn.Sequential(nn.Conv2d(1, stem, 3, 2, 1, bias=False), nn.BatchNorm2d(stem), nn.SiLU())
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        stages, cin, k = [], stem, 0
        for d, s, c in zip(depths, strides, dims):
            blocks = []
            for j in range(d):
                blocks.append(HyperInvertedResidual(cin, c, s if j == 0 else 1, expand, 3, 2, n_divs, drop_path=dpr[k]))
                cin, k = c, k + 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.stage_dims = list(dims)
        self._apply_freeze(self)

    def prepare(self, wav):
        return self.stem(self.frontend(wav))

    def tokens(self, i, h):  # h: (B, C, T, F) -> mean over frequency (and keep max for transients)
        return (h.mean(-1) + h.amax(-1)).transpose(1, 2)


class HyperVideoNet(_StagewiseNet):
    """Hypercomplex 3D inverted-residual CNN (X3D/MoViNet-like budget).

    Input: video ``(B, 3, T, H, W)`` in ``[0, 1]``. Stage tokens are ``T_s x g x g`` pooled cells.
    """

    def __init__(self, dims=(48, 96, 160, 256), depths=(1, 2, 3, 2),
                 strides=((1, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)), stem=32, expand=4, n_divs=4,
                 drop_path=0.05, grid=2, size=None, **kw):
        super().__init__(**kw)
        self.norm = VideoNormalize(IMAGENET_MEAN, IMAGENET_STD, size=tuple(size) if size else None)
        self.stem = nn.Sequential(nn.Conv3d(3, stem, (3, 3, 3), (1, 2, 2), (1, 1, 1), bias=False),
                                  nn.BatchNorm3d(stem), nn.SiLU())
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        stages, cin, k = [], stem, 0
        for d, s, c in zip(depths, strides, dims):
            blocks = []
            for j in range(d):
                blocks.append(HyperInvertedResidual(cin, c, tuple(s) if j == 0 else 1, expand, 3, 3, n_divs,
                                                    drop_path=dpr[k]))
                cin, k = c, k + 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.stage_dims = list(dims)
        self.grid = grid
        self._apply_freeze(self)

    def prepare(self, video):
        return self.stem(self.norm(video))

    def tokens(self, i, h):
        return _grid_pool_tokens(h, self.grid)


# ======================================================================================
# Pretrained audio backbones
# ======================================================================================
class TimmAudioBackbone(Backbone):
    """Any timm CNN (``features_only``) on a 1-channel log-mel image, e.g. ``efficientnet_b0``,
    ``mobilenetv3_large_100``, ``convnext_tiny``, ``mobilenetv4_conv_medium``."""

    def __init__(self, model_name='efficientnet_b0', pretrained=True, out_indices=(1, 2, 3, 4),
                 frontend: Optional[dict] = None, **kw):
        super().__init__(**kw)
        import timm
        fe = {'n_mels': 128, **(frontend or {})}
        fe.setdefault('normalize', 'instance' if self.freeze else 'bn')
        self.frontend = LogMel(**fe)
        self.net = timm.create_model(model_name, pretrained=pretrained, features_only=True, in_chans=1,
                                     out_indices=tuple(out_indices))
        self.stage_dims = list(self.net.feature_info.channels())
        self._apply_freeze(self.net)

    def forward_stages(self, wav):
        return [(f.mean(-1) + f.amax(-1)).transpose(1, 2) for f in self.net(self.frontend(wav))]


class ASTBackbone(Backbone):
    """Audio Spectrogram Transformer (Hugging Face ``transformers``), AudioSet-pretrained.

    Position embeddings are interpolated to the AV-FFIA clip length (2 s gives 200 fbank
    frames) rather than padding every clip to 10 s, which cuts the cost about 5x.
    """

    def __init__(self, model_name='MIT/ast-finetuned-audioset-10-10-0.4593', layers=(3, 6, 9, 12),
                 in_sample_rate=64000, clip_seconds=2.0, pretrained=True, **kw):
        super().__init__(**kw)
        from transformers import ASTConfig, ASTModel
        self.frontend = KaldiFbank(in_sample_rate=in_sample_rate)
        self.net = ASTModel.from_pretrained(model_name) if pretrained else ASTModel(ASTConfig())
        n_frames = int(round(clip_seconds * 100))
        self._resize_pos_embed(n_frames)
        self.layers = list(layers)
        self.stage_dims = [self.net.config.hidden_size] * len(self.layers)
        self._apply_freeze(self.net)

    @torch.no_grad()
    def _resize_pos_embed(self, n_frames):
        cfg, emb = self.net.config, self.net.embeddings
        f_old, t_old = emb.get_shape(cfg)
        cfg.max_length = n_frames
        f_new, t_new = emb.get_shape(cfg)
        pos = emb.position_embeddings  # (1, 2 + f*t, D)
        cls, grid = pos[:, :2], pos[:, 2:]
        D = grid.shape[-1]
        grid = grid.reshape(1, f_old, t_old, D).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(f_new, t_new), mode='bilinear', align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, f_new * t_new, D)
        emb.position_embeddings = nn.Parameter(torch.cat([cls, grid], 1))
        self.n_frames = n_frames

    def forward_stages(self, wav):
        x = self.frontend(wav)
        T = x.shape[1]
        x = F.pad(x, (0, 0, 0, self.n_frames - T)) if T < self.n_frames else x[:, :self.n_frames]
        hs = self.net(input_values=x, output_hidden_states=True).hidden_states
        return [hs[i][:, 2:] for i in self.layers]  # drop CLS/distillation tokens


# ======================================================================================
# Pretrained video backbones
# ======================================================================================
class TimmFrameBackbone(Backbone):
    """An ImageNet 2D timm backbone applied per frame, plus a trainable *hypercomplex
    temporal adapter* per stage (depth-wise temporal conv and a hypercomplex 1x1x1 conv).

    Good efficient choices: ``mobilenetv4_conv_medium``, ``efficientvit_b1``,
    ``convnext_tiny``, ``efficientnet_b0``.
    """

    def __init__(self, model_name='mobilenetv4_conv_medium', pretrained=True, out_indices=(1, 2, 3, 4),
                 grid=2, size=None, n_divs=4, temporal_adapter=True, **kw):
        super().__init__(**kw)
        import timm
        self.net = timm.create_model(model_name, pretrained=pretrained, features_only=True,
                                     out_indices=tuple(out_indices))
        cfg = self.net.pretrained_cfg if hasattr(self.net, 'pretrained_cfg') else {}
        self.norm = VideoNormalize(cfg.get('mean', IMAGENET_MEAN), cfg.get('std', IMAGENET_STD),
                                   size=tuple(size) if size else None)
        self.stage_dims = list(self.net.feature_info.channels())
        self.grid = grid
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Conv3d(c, c, (3, 1, 1), padding=(1, 0, 0), groups=c, bias=False),
                          nn.GELU(), hconv(c, c, 1, 3, n_divs)) if temporal_adapter else nn.Identity()
            for c in self.stage_dims])
        for a in self.adapters:
            if temporal_adapter:
                nn.init.zeros_(a[-1].weight)
                nn.init.zeros_(a[-1].bias)
        self._apply_freeze(self.net)

    def forward(self, x):  # the adapters stay trainable even when the 2D net is frozen
        return self.forward_stages(x)

    def forward_stages(self, video):
        video = self.norm(video)
        B, C, T, H, W = video.shape
        frames = video.transpose(1, 2).reshape(B * T, C, H, W)
        if self.fully_frozen:
            with torch.no_grad():
                feats = self.net(frames)
        else:
            feats = self.net(frames)
        out = []
        for f, ad in zip(feats, self.adapters):
            g = self.grid or f.shape[-1]
            f = F.adaptive_avg_pool2d(f, (min(g, f.shape[-2]), min(g, f.shape[-1])))
            f = f.view(B, T, *f.shape[1:]).transpose(1, 2)  # (B, C, T, g, g)
            f = f + ad(f)
            out.append(f.flatten(2).transpose(1, 2))
        return out


class TorchvisionVideoBackbone(Backbone):
    """Kinetics-400 pretrained 3D CNNs from torchvision: ``r2plus1d_18``, ``r3d_18``, ``mc3_18``, ``s3d``.

    S3D is the video backbone of the U-FFIA reference model.
    """

    _STAGES = {
        'r2plus1d_18': ['layer1', 'layer2', 'layer3', 'layer4'],
        'r3d_18': ['layer1', 'layer2', 'layer3', 'layer4'],
        'mc3_18': ['layer1', 'layer2', 'layer3', 'layer4'],
        's3d': ['features.3', 'features.6', 'features.12', 'features.15'],
    }

    def __init__(self, model_name='r2plus1d_18', pretrained=True, grid=2, size=None, **kw):
        super().__init__(**kw)
        import torchvision.models.video as tvv
        self.net = getattr(tvv, model_name)(weights='DEFAULT' if pretrained else None)
        self.norm = VideoNormalize(KINETICS_MEAN, KINETICS_STD, size=tuple(size) if size else None)
        self.stage_names = self._STAGES[model_name]
        self.grid = grid
        self._feats = {}
        modules = dict(self.net.named_modules())
        for n in self.stage_names:
            modules[n].register_forward_hook(lambda m, i, o, n=n: self._feats.__setitem__(n, o))
        with torch.no_grad():
            self.net.eval()
            self._run(torch.zeros(1, 3, 16, 112, 112))
            self.stage_dims = [self._feats[n].shape[1] for n in self.stage_names]
        self.net.train()
        self._apply_freeze(self.net)

    def _run(self, x):  # run the trunk only (skip pooling and classifier)
        n = self.net
        if hasattr(n, 'layer4'):
            n.layer4(n.layer3(n.layer2(n.layer1(n.stem(x)))))
        else:
            n.features(x)

    def forward_stages(self, video):
        self._feats.clear()
        self._run(self.norm(video))
        return [_grid_pool_tokens(self._feats[n], self.grid) for n in self.stage_names]


class VideoMAEBackbone(Backbone):
    """VideoMAE (Hugging Face), Kinetics-pretrained. Tokens are spatially pooled to ``grid x grid``
    per tubelet time step to keep fusion cheap."""

    def __init__(self, model_name='MCG-NJU/videomae-base-finetuned-kinetics', layers=(3, 6, 9, 12), grid=7,
                 size=(224, 224), pretrained=True, **kw):
        super().__init__(**kw)
        from transformers import VideoMAEConfig, VideoMAEModel
        self.net = VideoMAEModel.from_pretrained(model_name) if pretrained else VideoMAEModel(VideoMAEConfig())
        self.norm = VideoNormalize(IMAGENET_MEAN, IMAGENET_STD, size=tuple(size))
        cfg = self.net.config
        self.t_patches = cfg.num_frames // cfg.tubelet_size
        self.s_patches = cfg.image_size // cfg.patch_size
        self.layers, self.grid = list(layers), grid
        self.stage_dims = [cfg.hidden_size] * len(self.layers)
        self._apply_freeze(self.net)

    def forward_stages(self, video):
        x = self.norm(video).transpose(1, 2)  # (B, T, C, H, W)
        hs = self.net(pixel_values=x, output_hidden_states=True).hidden_states
        out = []
        for i in self.layers:
            h = hs[i]
            B, N, D = h.shape
            h = h.view(B, self.t_patches, self.s_patches, self.s_patches, D).permute(0, 4, 1, 2, 3)
            out.append(_grid_pool_tokens(h, self.grid))
        return out


BACKBONES = {
    'hyper_audio': HyperAudioNet,
    'timm_audio': TimmAudioBackbone,
    'ast': ASTBackbone,
    'hyper_video': HyperVideoNet,
    'timm_frames': TimmFrameBackbone,
    'tv_video': TorchvisionVideoBackbone,
    'videomae': VideoMAEBackbone,
}


def build_backbone(cfg: dict) -> Backbone:
    cfg = dict(cfg)
    return BACKBONES[cfg.pop('type')](**cfg)
