"""Parameters, FLOPs and latency of a config's model: python scripts/profile_model.py -c configs/student_hyper.yaml"""
import argparse
import copy
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from avffia.hyper import count_parameters, reparameterize_hypercomplex  # noqa: E402
from avffia.metrics import count_flops  # noqa: E402
from avffia.model import build_model  # noqa: E402
from train import load_config  # noqa: E402


@torch.no_grad()
def latency(m, inputs, reps=20):
    m.eval()
    for _ in range(3):
        m(*inputs)
    if inputs[0].is_cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        m(*inputs)
    if inputs[0].is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True)
ap.add_argument('--set', action='append', default=[])
ap.add_argument('--batch', type=int, default=1)
ap.add_argument('--out', help='also save the JSON here, e.g. runs/profile_student_hyper.json (read by make_paper_assets.py)')
args = ap.parse_args()
cfg = load_config(args.config, args.set)
d = cfg['data']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = build_model(cfg['model']).to(dev)
x = (torch.randn(args.batch, int(d['sample_rate'] * d['seconds']), device=dev),
     torch.rand(args.batch, 3, d['num_frames'], d['crop'], d['crop'], device=dev))
dep = reparameterize_hypercomplex(copy.deepcopy(m))
parts = {k: count_parameters(getattr(m, k)) for k in ('audio', 'video')}
parts['fusion+heads'] = count_parameters(m) - sum(parts.values())
res = {'name': cfg.get('name'), 'device': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu',
       'batch': args.batch, 'params_M': count_parameters(m) / 1e6,
       'stored_params_by_part_M': {k: v / 1e6 for k, v in parts.items()},
       'deployed_dense_params_M': count_parameters(dep) / 1e6,
       'GFLOPs_per_clip': count_flops(m, *x) / 1e9 / args.batch,
       'latency_ms': latency(m, x), 'latency_deployed_ms': latency(dep, x)}
res['GMACs_per_clip'] = res['GFLOPs_per_clip'] / 2
print(json.dumps(res, indent=1))
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
