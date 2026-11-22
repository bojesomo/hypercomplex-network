"""Train and evaluate HyperAVNet on AV-FFIA (supervised, or distilled from a teacher).

    # 1) build the split manifest once (same split rule and seed as U-FFIA)
    python train.py --build-manifest --video-root /data/AV-FFIA --audio-root /data/AV-FFIA_audio \
        --manifest data/avffia.csv

    # 2) teacher (pretrained backbones, hypercomplex side-fusion)
    python train.py -c configs/teacher_ast_videomae.yaml

    # 3) efficient hypercomplex student, distilled from the teacher
    python train.py -c configs/student_hyper_kd.yaml

    # ablation override, e.g. fusion depth sweep
    python train.py -c configs/student_hyper_kd.yaml --set model.fusion_stages=[1,2,3] --set name=fs123

    # smoke test without data
    python train.py -c configs/student_hyper.yaml --synthetic --set train.epochs=1
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from avffia.data import AVFFIADataset, SyntheticAVDataset, build_manifest, make_loader
from avffia.hyper import count_parameters, reparameterize_hypercomplex
from avffia.losses import AVLoss, DistillLoss
from avffia.metrics import classification_metrics, count_flops
from avffia.model import build_model


# ----------------------------------------------------------------------------- config
def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop('_base_', None)
    if base:
        root = load_config(os.path.join(os.path.dirname(path), base))
        cfg = _merge(root, cfg)
    for ov in overrides:
        key, val = ov.split('=', 1)
        d = cfg
        *parents, last = key.split('.')
        for k in parents:
            d = d.setdefault(k, {})
        d[last] = yaml.safe_load(val)
    return cfg


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- optim
def param_groups(model, lr, wd, backbone_lr_mult):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith(('audio.net', 'video.net'))  # pretrained weights
        no_wd = p.ndim <= 1 or name.endswith('.bias') or 'bottleneck' in name or name.endswith('.mod')
        key = (is_backbone, no_wd)
        if key not in groups:
            groups[key] = {'params': [], 'lr': lr * (backbone_lr_mult if is_backbone else 1.0),
                           'weight_decay': 0.0 if no_wd else wd}
        groups[key]['params'].append(p)
    for g in groups.values():
        g['base_lr'] = g['lr']
    return list(groups.values())


def cosine_lr(opt, step, total, warmup, min_ratio=0.01):
    if step < warmup:
        f = (step + 1) / warmup
    else:
        t = (step - warmup) / max(1, total - warmup)
        f = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    for g in opt.param_groups:
        g['lr'] = g['base_lr'] * f


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


# ----------------------------------------------------------------------------- monitoring
class Progress:
    """In-epoch console monitoring. A tqdm bar on an interactive terminal; otherwise (nohup,
    redirected logs, or no tqdm) a plain text line every ``log_every`` steps.

    Usage: ``for b in (bar := Progress(loader, 'ep 3/40')): ...; bar.update(loss=0.53, acc=0.81)``
    """

    def __init__(self, iterable, desc: str, enabled: bool = True, log_every: int = 50, leave: bool = True):
        self.it, self.desc, self.log_every = iterable, desc, max(1, log_every)
        self.total = len(iterable) if hasattr(iterable, '__len__') else None
        self.bar, self.i, self.t0 = None, 0, time.time()
        self.enabled = enabled
        if enabled and sys.stderr.isatty():
            try:
                import shutil
                from tqdm import tqdm
                # Some consoles (IDE run windows, emulated ptys) report a 0x0 terminal; tqdm then
                # draws nothing (0 rows hides the bar), so fall back to a fixed size there.
                ts = shutil.get_terminal_size((0, 0))
                size = dict(dynamic_ncols=True) if ts.columns > 0 else dict(ncols=120)
                if ts.lines <= 0:
                    size['nrows'] = 40
                self.bar = tqdm(iterable, desc=desc, total=self.total, leave=leave, mininterval=0.5,
                                smoothing=0.1, **size)
            except ImportError:
                pass

    def __iter__(self):
        for x in (self.bar if self.bar is not None else self.it):
            self.i += 1
            yield x

    def update(self, **stats):
        if not self.enabled:
            return
        fmt = {k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in stats.items()}
        if self.bar is not None:
            self.bar.set_postfix(fmt, refresh=False)
        elif self.i % self.log_every == 0 or self.i == self.total:
            el = time.time() - self.t0
            eta = el / self.i * (self.total - self.i) if self.total else 0
            print(f'{self.desc} [{self.i}/{self.total or "?"}] ' + ' '.join(f'{k}={v}' for k, v in fmt.items()) +
                  f' | {el:.0f}s elapsed, eta {eta:.0f}s', flush=True)

    def write(self, msg: str):
        (self.bar.write if self.bar is not None else print)(msg)

    def close(self):
        if self.bar is not None:
            self.bar.close()


def gpu_mem() -> str:
    return f'{torch.cuda.max_memory_allocated() / 2**30:.1f}G' if torch.cuda.is_available() else '-'


# ----------------------------------------------------------------------------- loops
def to_device(batch, device):
    mv = lambda v: v.to(device, non_blocking=True) if torch.is_tensor(v) else [mv(x) for x in v] if isinstance(v, list) else v
    return {k: mv(v) for k, v in batch.items()}


def forward(model, b):
    if 'a_tokens' in b:  # cached frozen-backbone features
        return model.forward_from_tokens(b['a_tokens'], b['v_tokens'])
    return model(b['waveform'], b['video'])


@torch.no_grad()
def evaluate(model, loader, device, num_classes, amp_dtype, corrupt=None, desc='eval', progress=True):
    model.eval()
    logits, la, lv, ys = [], [], [], []
    correct = seen = 0
    bar = Progress(loader, f'  {desc}', enabled=progress, log_every=10 ** 9, leave=False)
    for i, b in enumerate(bar):
        b = to_device(b, device)
        if corrupt is not None:
            b = corrupt(b, i)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = forward(model, b)
        logits.append(out['logits'].float().cpu())
        correct += (logits[-1].argmax(-1) == b['target'].cpu()).sum().item()
        seen += b['target'].shape[0]
        bar.update(acc=correct / max(1, seen))
        if 'logits_a' in out:
            la.append(out['logits_a'].float().cpu())
            lv.append(out['logits_v'].float().cpu())
        ys.append(b['target'].cpu())
    bar.close()
    y = torch.cat(ys)
    res = classification_metrics(torch.cat(logits), y, num_classes)
    if la:
        res['audio_only_acc'] = classification_metrics(torch.cat(la), y, num_classes)['acc']
        res['video_only_acc'] = classification_metrics(torch.cat(lv), y, num_classes)['acc']
    return res


def robustness(model, loader, device, num_classes, amp_dtype, cfg):
    """U-FFIA-style robustness protocol: audio noise at fixed SNRs, video darkness and Gaussian
    noise, and missing modalities. Returns ``{condition: accuracy}``."""
    from avffia.corruptions import build_corruptions
    rc = cfg.get('robustness', {})
    res = {}
    cached = 'a_tokens' in next(iter(loader))  # cached features: input corruptions do not apply
    for name, fn in ({} if cached else build_corruptions(cfg['data']['sample_rate'], rc.get('noise_dir'),
                                                          rc.get('seed', 0))).items():
        res[name] = evaluate(model, loader, device, num_classes, amp_dtype, corrupt=fn, desc=name)['acc']
    for m in ('audio', 'video'):
        if hasattr(model, 'missing'):
            model.missing = m
            res[f'missing_{m}'] = evaluate(model, loader, device, num_classes, amp_dtype, desc=f'missing_{m}')['acc']
            model.missing = None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', default='configs/student_hyper.yaml')
    ap.add_argument('--set', action='append', default=[], help='override, e.g. model.n_divs=8')
    ap.add_argument('--synthetic', action='store_true', help='use random data (smoke test)')
    ap.add_argument('--build-manifest', action='store_true')
    ap.add_argument('--video-root')
    ap.add_argument('--audio-root')
    ap.add_argument('--manifest', default='data/avffia.csv')
    ap.add_argument('--protocol', default='uffia', choices=['uffia', 'kfold', 'grouped'])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--group-level', type=int, default=1, help='dirs above the class folder that define a session')
    ap.add_argument('--test-per-class', type=int, default=700)
    ap.add_argument('--val-per-class', type=int, default=700)
    ap.add_argument('--eval-only', help='checkpoint to evaluate on the test split')
    ap.add_argument('--robustness', action='store_true', help='also run the noise / missing-modality protocol')
    args = ap.parse_args()

    if args.build_manifest:
        print('manifest written to', build_manifest(args.video_root, args.manifest, args.audio_root,
                                                    test_per_class=args.test_per_class, val_per_class=args.val_per_class,
                                                    protocol=args.protocol, folds=args.folds,
                                                    group_level=args.group_level))
        return

    cfg = load_config(args.config, args.set)
    seed_all(cfg.get('seed', 25))
    dcfg, tcfg = cfg['data'], cfg['train']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(tcfg.get('amp', 'bf16')) if device.type == 'cuda' else None
    out_dir = os.path.join(cfg.get('output_dir', 'runs'), cfg.get('name', 'run'))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)

    # data
    if args.synthetic:
        mk = lambda n, s: SyntheticAVDataset(n, dcfg['sample_rate'], dcfg['seconds'], dcfg['num_frames'], dcfg['crop'], seed=s)
        ds = {'train': mk(32, 0), 'val': mk(16, 1), 'test': mk(16, 2)}
    elif dcfg.get('cached_features'):
        from extract_features import CachedTokenDataset
        assert not cfg.get('distill'), 'distillation needs raw inputs for the teacher'
        ds = {s: CachedTokenDataset(dcfg['cached_features'], s) for s in ('train', 'val', 'test')}
    else:
        ds = {s: AVFFIADataset(dcfg.get('manifest', args.manifest), s, dcfg['sample_rate'], dcfg['seconds'],
                               dcfg['num_frames'], dcfg['size'], dcfg['crop'], cache_dir=dcfg.get('cache_dir'),
                               cv_fold=dcfg.get('cv_fold'))
              for s in ('train', 'val', 'test')}
    nw = 0 if args.synthetic else dcfg.get('num_workers', 8)
    loaders = {s: make_loader(d, tcfg['batch_size'], s == 'train', nw) for s, d in ds.items()}

    # model
    model = build_model(cfg['model']).to(device)
    num_classes = cfg['model'].get('num_classes', 4)
    wav0 = torch.zeros(1, int(dcfg['sample_rate'] * dcfg['seconds']), device=device)
    vid0 = torch.zeros(1, 3, dcfg['num_frames'], dcfg['crop'], dcfg['crop'], device=device)
    stats = {'params_total': count_parameters(model), 'params_trainable': count_parameters(model, True)}
    try:
        stats['gflops'] = count_flops(model, wav0, vid0) / 1e9
        stats['gmacs'] = stats['gflops'] / 2  # the unit U-FFIA reports as "FLOPs"
    except Exception as e:  # pragma: no cover
        stats['gflops'] = f'n/a ({e})'
    print(json.dumps(stats))

    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=device))
        res = evaluate(model, loaders['test'], device, num_classes, amp_dtype)
        if args.robustness:
            res['robustness'] = robustness(model, loaders['test'], device, num_classes, amp_dtype, cfg)
        print(json.dumps(res, indent=1))
        return

    # teacher (optional)
    teacher, kd = None, None
    if cfg.get('distill'):
        dc = cfg['distill']
        tc = load_config(dc['teacher_config'])
        teacher = build_model(tc['model']).to(device).eval()
        if dc.get('teacher_ckpt'):
            teacher.load_state_dict(torch.load(dc['teacher_ckpt'], map_location=device))
        for p in teacher.parameters():
            p.requires_grad_(False)
        kd = DistillLoss(model.dim, teacher.dim, cfg['model'].get('n_divs', 4), **dc.get('loss', {})).to(device)

    crit = AVLoss(tcfg.get('aux_weight', 0.3), tcfg.get('label_smoothing', 0.1))
    params = param_groups(model, tcfg['lr'], tcfg.get('weight_decay', 0.05), tcfg.get('backbone_lr_mult', 0.1))
    if kd is not None:
        params.append({'params': list(kd.parameters()), 'lr': tcfg['lr'], 'base_lr': tcfg['lr'], 'weight_decay': 0.0})
    opt = torch.optim.AdamW(params, lr=tcfg['lr'], betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
    ema = EMA(model, tcfg['ema']) if tcfg.get('ema') else None
    epochs = tcfg['epochs']
    total = epochs * len(loaders['train'])
    warmup = int(tcfg.get('warmup_epochs', 1) * len(loaders['train']))

    best, step, log = -1.0, 0, []
    show = tcfg.get('progress', True)
    log_every = tcfg.get('log_every', 50)
    for ep in range(epochs):
        model.train()
        t0, run = time.time(), {}
        correct = seen = 0
        bar = Progress(loaders['train'], f'ep {ep + 1}/{epochs}', enabled=show, log_every=log_every)
        for b in bar:
            b = to_device(b, device)
            cosine_lr(opt, step, total, warmup)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = forward(model, b)
                losses = crit(out, b['target'])
                if teacher is not None:
                    with torch.no_grad():
                        tout = teacher(b['waveform'], b['video'])
                    losses.update(kd({k: v.float() for k, v in out.items() if torch.is_tensor(v)},
                                     {k: v.float() for k, v in tout.items() if torch.is_tensor(v)}))
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), tcfg.get('clip_grad', 5.0))
            scaler.step(opt)
            scaler.update()
            if ema:
                ema.update(model)
            step += 1
            for k, v in losses.items():
                run[k] = run.get(k, 0.0) + v.item()
            correct += (out['logits'].argmax(-1) == b['target']).sum().item()
            seen += b['target'].shape[0]
            k = bar.i  # running means over the epoch so far; components are in the epoch summary
            extra = {'kd': run['kd'] / k} if 'kd' in run else {}
            bar.update(loss=sum(run.values()) / k, **extra, acc=correct / seen,
                       lr=f"{opt.param_groups[0]['lr']:.1e}", gnorm=float(gnorm),
                       sps=f"{seen / (time.time() - t0):.0f}/s", mem=gpu_mem())
        bar.close()
        n = max(1, len(loaders['train']))
        eval_model = ema.model if ema else model
        val = evaluate(eval_model, loaders['val'], device, num_classes, amp_dtype, desc='val', progress=show)
        rec = {'epoch': ep, 'time': round(time.time() - t0, 1), **{k: v / n for k, v in run.items()},
               'val_acc': val['acc'], 'val_f1': val['macro_f1'],
               'val_audio_acc': val.get('audio_only_acc'), 'val_video_acc': val.get('video_only_acc')}
        log.append(rec)
        with open(os.path.join(out_dir, 'log.jsonl'), 'a') as f:  # full record per epoch
            f.write(json.dumps(rec) + '\n')
        improved = val['acc'] > best
        if improved:
            best = val['acc']
            torch.save(eval_model.state_dict(), os.path.join(out_dir, 'best.pt'))
        losses_s = ' '.join(f'{k} {v / n:.4f}' for k, v in run.items())
        uni = (f" | A {val['audio_only_acc']:.4f} V {val['video_only_acc']:.4f}"
               if val.get('audio_only_acc') is not None else '')
        print(f"ep {ep + 1:>3}/{epochs} | {losses_s} | train acc {correct / max(1, seen):.4f} | "
              f"val acc {val['acc']:.4f} f1 {val['macro_f1']:.4f}{uni} | best {best:.4f}"
              f"{' *' if improved else ''} | {rec['time']:.0f}s", flush=True)

    model.load_state_dict(torch.load(os.path.join(out_dir, 'best.pt'), map_location=device))
    test = evaluate(model, loaders['test'], device, num_classes, amp_dtype, desc='test', progress=show)
    if args.robustness or cfg.get('robustness', {}).get('enabled', False):
        test['robustness'] = robustness(model, loaders['test'], device, num_classes, amp_dtype, cfg)
    # deployable model: hypercomplex kernels folded into dense layers (same outputs, faster)
    torch.save(reparameterize_hypercomplex(copy.deepcopy(model).cpu()).state_dict(), os.path.join(out_dir, 'deploy.pt'))
    result = {'name': cfg.get('name'), **stats, 'best_val_acc': best, 'test': test}
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump({'result': result, 'log': log}, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
