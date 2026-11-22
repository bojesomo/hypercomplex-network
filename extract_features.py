"""Cache multi-stage tokens from frozen pretrained backbones, for very cheap fusion training.

With frozen backbones the per-epoch cost is dominated by running them. Running them once,
over ``--views`` augmented views of each training clip, means the hypercomplex fusion network
(well under 1M parameters) then trains in minutes::

    python extract_features.py -c configs/frozen_timm_side.yaml --out feats/timm --views 4
    python train.py -c configs/frozen_timm_side.yaml --set data.cached_features=feats/timm

Note: stage tokens are stored in fp16; with ``grid`` pooling they stay small (tens of KB per clip).
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import Dataset

from avffia.data import AVFFIADataset, make_loader
from avffia.model import build_model


class CachedTokenDataset(Dataset):
    def __init__(self, root: str, split: str):
        self.files = sorted(os.path.join(root, split, f) for f in os.listdir(os.path.join(root, split)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i])
        v = torch.randint(len(d['a']), ()).item()  # random augmented view
        return {'a_tokens': [t.float() for t in d['a'][v]], 'v_tokens': [t.float() for t in d['v'][v]],
                'target': d['target']}


@torch.no_grad()
def main():
    from train import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', type=int, default=4, help='augmented views per training clip')
    ap.add_argument('--set', action='append', default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    dcfg = cfg['data']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg['model']).to(device).eval()
    trainable = [n for n, p in list(model.audio.named_parameters()) + list(model.video.named_parameters())
                 if p.requires_grad]
    if trainable:
        print(f'WARNING: {len(trainable)} trainable backbone tensors (e.g. {trainable[0]}) will be cached at '
              'their initial values; set freeze: true and temporal_adapter: false for cached training.')
    for split in ('train', 'val', 'test'):
        ds = AVFFIADataset(dcfg['manifest'], split, dcfg['sample_rate'], dcfg['seconds'], dcfg['num_frames'],
                           dcfg['size'], dcfg['crop'], cache_dir=dcfg.get('cache_dir'), cv_fold=dcfg.get('cv_fold'))
        views = args.views if split == 'train' else 1
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        store = [{'a': [], 'v': [], 'target': int(it['label'])} for it in ds.items]
        for _ in range(views):
            loader = make_loader(ds, 32, False, dcfg.get('num_workers', 8))
            k = 0
            for b in loader:
                a = model.audio(b['waveform'].to(device))
                v = model.video(b['video'].to(device))
                for j in range(b['target'].shape[0]):
                    store[k + j]['a'].append([t[j].half().cpu() for t in a])
                    store[k + j]['v'].append([t[j].half().cpu() for t in v])
                k += b['target'].shape[0]
        for i, s in enumerate(store):
            torch.save(s, os.path.join(args.out, split, f'{i:06d}.pt'))
        print(split, len(store), 'clips cached')


if __name__ == '__main__':
    main()
