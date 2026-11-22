"""Write one self-contained config per teacher candidate (for scripts/run_teacher_ablation.sh).

    python scripts/make_teacher_configs.py --out runs_teacher/configs [--set train.epochs=1 ...]

Each teacher is the PANNs-CNN10 + S3D teacher config with its audio and/or video backbone swapped.
The same HCF fusion network, data pipeline and training recipe are used for all, so the only
variable is the pretrained backbone pair.
"""
import argparse
import copy
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import _merge, load_config  # noqa: E402

P = 'pretrained'
TEACHERS = {
    # name: (base config, model overrides)
    'T1_cnn10_s3d': ('configs/teacher_panns_s3d.yaml', {}),
    'T2_cnn14_s3d': ('configs/teacher_panns_s3d.yaml',
                     {'audio': {'model_name': 'cnn14', 'checkpoint': f'{P}/Cnn14_mAP=0.431.pth'}}),
    # R(2+1)D was pretrained at 112x112 (and costs ~166 GMACs at 224), so it resizes its input
    'T3_cnn10_r2plus1d': ('configs/teacher_panns_s3d.yaml', {'video': {'model_name': 'r2plus1d_18', 'size': [112, 112]}}),
    'T4_cnn10_convnext_frames': ('configs/teacher_panns_s3d.yaml',
                                 {'video': {'type': 'timm_frames', 'model_name': 'convnext_tiny.in12k_ft_in1k',
                                            'out_indices': [0, 1, 2, 3], 'grid': 2}}),
    'T5_effnet_mel_s3d': ('configs/teacher_panns_s3d.yaml',
                          {'audio': {'type': 'timm_audio', 'model_name': 'efficientnet_b0.ra_in1k',
                                     'frontend': {'sample_rate': 64000, 'n_fft': 2048, 'hop': 1024, 'n_mels': 128}}}),
    'T6_ast_videomae': ('configs/teacher_ast_videomae.yaml', {}),
    # X3D: best classic 3D CNN on animal-behaviour video (Animal Kingdom), about 4.7 GMACs at 16x224
    'T7_cnn10_x3d_m': ('configs/teacher_panns_s3d.yaml',
                       {'video': {'type': 'pv_video', 'model_name': 'x3d_m', 'grid': 2}}),
    # one-modality-at-a-time transformer swaps (AST 0.736 and ViViT 0.743 under-performed in U-FFIA)
    'T8_ast_s3d': ('configs/teacher_panns_s3d.yaml',
                   {'audio': {'type': 'ast', 'model_name': 'MIT/ast-finetuned-audioset-10-10-0.4593',
                              'layers': [3, 6, 9, 12], 'in_sample_rate': 64000}}),
    'T9_cnn10_videomae': ('configs/teacher_panns_s3d.yaml',
                          {'video': {'type': 'videomae', 'model_name': 'MCG-NJU/videomae-base-finetuned-kinetics',
                                     'layers': [3, 6, 9, 12], 'grid': 7}}),
    # Not yet implemented (needs external code): HMKD's EfficientAT (mn10_as) + MoViNet-A0 teacher,
    # BEATs (attentive multi-layer pooling), V-JEPA 2 / VideoMAE V2, CAV-MAE (joint AV).
}


def build(name, base, over, sets):
    cfg = load_config(base, sets)
    for mod, o in over.items():
        if 'type' in o and o['type'] != cfg['model'][mod].get('type'):
            cfg['model'][mod] = copy.deepcopy(o)  # new backbone type: replace, do not merge
        else:
            cfg['model'][mod] = _merge(cfg['model'][mod], o)
    cfg['name'] = f'teacher_{name}'
    return cfg


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, (base, over) in TEACHERS.items():
        if a.only and name not in a.only:
            continue
        path = os.path.join(a.out, f'{name}.yaml')
        with open(path, 'w') as f:
            yaml.safe_dump(build(name, base, over, a.set), f, sort_keys=False)
        print(path)
