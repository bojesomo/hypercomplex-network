"""Offline smoke tests (random weights, tiny inputs): python -m pytest -q tests"""
import copy
import importlib.util

import pytest
import torch

from avffia.fusion import HyperComponentFusion
from avffia.hyper import HyperAttention, count_parameters, hlinear, reparameterize_hypercomplex
from avffia.losses import AVLoss, DistillLoss
from avffia.model import HyperAVNet

WAV = lambda b=2: torch.randn(b, 64000)          # 1 s at 64 kHz keeps tests fast
VID = lambda b=2, s=64, t=8: torch.rand(b, 3, t, s, s)
HYPER_A = {'type': 'hyper_audio', 'dims': [16, 32, 48, 64], 'depths': [1, 1, 1, 1], 'stem': 16}
HYPER_V = {'type': 'hyper_video', 'dims': [16, 32, 48, 64], 'depths': [1, 1, 1, 1], 'stem': 16}


def _check(m, wav, vid):
    out = m(wav, vid)
    assert out['logits'].shape == (wav.shape[0], 4)
    loss = sum(AVLoss()(out, torch.tensor([0, 1][:wav.shape[0]])).values())
    loss.backward()
    return out


def test_hlinear_param_reduction():
    assert count_parameters(hlinear(64, 64, 4)) == 64 * 64 // 4 + 64
    assert count_parameters(hlinear(64, 64, 1)) == 64 * 64 + 64
    assert isinstance(hlinear(6, 64, 4), torch.nn.Linear)  # falls back when not divisible


def test_hypercomplex_attention_and_component_fusion():
    attn = HyperAttention(32, 4, 4)
    assert attn(torch.randn(2, 5, 32), torch.randn(2, 7, 32)).shape == (2, 5, 32)
    for alg in (2, 4, 8):
        f = HyperComponentFusion(32, 4, 4, algebra=alg)
        assert f(torch.randn(2, 3, 32), torch.randn(2, 9, 32), torch.randn(2, 6, 32)).shape == (2, 3, 32)


@pytest.mark.parametrize('n_divs', [1, 2, 4, 8])
@pytest.mark.parametrize('fusion', ['hcf', 'mbt', 'concat'])
@pytest.mark.parametrize('stages', [[3], [2, 3], [0, 1, 2, 3], [0]])
def test_student_fusion_ablation_grid(n_divs, fusion, stages):
    m = HyperAVNet(audio={**HYPER_A, 'n_divs': n_divs}, video={**HYPER_V, 'n_divs': n_divs}, dim=32,
                   n_divs=n_divs, fusion=fusion, fusion_stages=stages, fusion_algebra=min(max(n_divs, 2), 8))
    _check(m, WAV(), VID())


def test_inject_vs_side_and_from_tokens_match():
    torch.manual_seed(0)
    m = HyperAVNet(audio=HYPER_A, video=HYPER_V, dim=32, fusion_stages=[1, 2, 3], inject=True).eval()
    wav, vid = WAV(), VID()
    with torch.no_grad():
        a, b = m(wav, vid)['logits'], m.forward_from_tokens(m.audio(wav), m.video(vid))['logits']
    # feedback is zero-initialised, so at init the stage-wise and side paths agree
    assert torch.allclose(a, b, atol=1e-5)


def test_reparameterize_is_exact():
    m = HyperAVNet(audio=HYPER_A, video=HYPER_V, dim=32, fusion_stages=[0, 3]).eval()
    for p in m.parameters():  # move away from zero/identity init so the check is meaningful
        p.data.add_(0.01 * torch.randn_like(p))
    wav, vid = WAV(), VID()
    with torch.no_grad():
        ref = m(wav, vid)['logits']
        dep = reparameterize_hypercomplex(copy.deepcopy(m))(wav, vid)['logits']
    assert torch.allclose(ref, dep, atol=1e-4)


def test_distill_loss():
    s = HyperAVNet(audio=HYPER_A, video=HYPER_V, dim=32)
    t = HyperAVNet(audio=HYPER_A, video=HYPER_V, dim=64).eval()
    wav, vid = WAV(), VID()
    with torch.no_grad():
        to = t(wav, vid)
    kd = DistillLoss(32, 64, 4)
    sum(kd(s(wav, vid), to).values()).backward()


def test_timm_and_torchvision_backbones():
    m = HyperAVNet(audio={'type': 'timm_audio', 'model_name': 'mobilenetv2_100', 'pretrained': False,
                          'freeze': True},
                   video={'type': 'timm_frames', 'model_name': 'mobilenetv4_conv_small', 'pretrained': False,
                          'freeze': True},
                   dim=32, fusion_stages=[0, 1, 2, 3])
    out = _check(m, WAV(), VID(t=4))
    assert all(not p.requires_grad for p in m.audio.net.parameters())
    m = HyperAVNet(audio=HYPER_A, video={'type': 'tv_video', 'model_name': 'r2plus1d_18', 'pretrained': False},
                   dim=32, fusion_stages=[2, 3])
    _check(m, WAV(), VID(s=64, t=8))
    m = HyperAVNet(audio=HYPER_A, video={'type': 'tv_video', 'model_name': 's3d', 'pretrained': False},
                   dim=32, fusion_stages=[3])
    _check(m, WAV(), VID(s=112, t=16))


@pytest.mark.skipif(importlib.util.find_spec('transformers') is None, reason='transformers not installed')
def test_hf_backbones_random_init():
    m = HyperAVNet(audio={'type': 'ast', 'pretrained': False, 'layers': [1, 2], 'in_sample_rate': 64000,
                          'clip_seconds': 1.0, 'freeze': True, 'unfreeze': ['encoder.layer.11.']},
                   video={'type': 'videomae', 'pretrained': False, 'layers': [1, 2], 'grid': 2, 'freeze': True},
                   dim=48, fusion_stages=[0, 1])
    _check(m, WAV(), VID(s=224, t=16))
