"""AV-FFIA data pipeline.

The reference U-FFIA code (github.com/FishMaster93/U-FFIA) finds clips at
``<root>/**/<class>/*.mp4`` with class folders ``none|strong|medium|weak``, uses labels
``none=0, strong=1, medium=2, weak=3``, and splits each class with
``np.random.RandomState(seed)``: the first 700 shuffled clips go to test, the next 700 to
val, the rest to train. ``build_manifest`` follows exactly that rule (same seed means same
split), so results can be compared with the published numbers. It writes a CSV manifest,
which is then the single source of truth.

Dataset facts (U-FFIA paper, Sec. V): 27,000 two-second clips of *Oplegnathus punctatus*, from one
recirculating tank, with 6,750 clips per class. Audio is from a hydrophone at 256 kHz, which
U-FFIA downsamples to 64 kHz. Video is 25 fps (50 frames per clip). The paper reports a
21,000 / 2,800 / 2,800 split; the released code's rule above gives 21,400 / 2,800 / 2,800
(6,750 x 4 - 5,600). The 400-clip difference is the paper's own inconsistency; val and test
are identical either way.

Audio comes from a sibling ``.wav`` (looked up under ``audio_root`` with the same relative
path or stem) or, failing that, from the mp4's own audio track.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CLASSES = ['none', 'strong', 'medium', 'weak']  # index = U-FFIA label id
LABELS = {c: i for i, c in enumerate(CLASSES)}
# ordinal intensity (none < weak < medium < strong), handy for analysis
INTENSITY = {'none': 0, 'weak': 1, 'medium': 2, 'strong': 3}


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------
def _find_audio(video_path: str, video_root: str, audio_root: Optional[str]) -> str:
    if audio_root:
        # U-FFIA / Zenodo convention (utils/get_audioname.py in the U-FFIA repo): the audio lives in
        # <audio_root>/<d3>/<d2>/<class>/*.wav, mirroring the last three video directories, and
        # pairs with the video whose file stem matches after the first 9 characters.
        parts = os.path.normpath(video_path).split(os.sep)
        vid_id = os.path.splitext(parts[-1])[0][9:]
        for base in (audio_root, os.path.join(audio_root, 'audio_dataset')):
            d = os.path.join(base, *parts[-4:-1])
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith('.wav') and os.path.splitext(f)[0][9:] == vid_id:
                        return os.path.join(d, f)
        rel = os.path.relpath(os.path.splitext(video_path)[0], video_root)
        for ext in ('.wav', '.flac'):
            cand = os.path.join(audio_root, rel + ext)
            if os.path.exists(cand):
                return cand
        stem = os.path.basename(os.path.splitext(video_path)[0])
        hits = glob.glob(os.path.join(audio_root, '**', stem + '.wav'), recursive=True)
        if hits:
            return hits[0]
    cand = os.path.splitext(video_path)[0] + '.wav'
    return cand if os.path.exists(cand) else ''  # '' means use the mp4 audio track


def _session(video_path: str, video_root: str, level: int) -> str:
    """Recording-session id: the ``level`` directories directly above the class folder."""
    rel = os.path.relpath(video_path, video_root).split(os.sep)[:-2]  # drop class folder + file
    return '/'.join(rel[-level:]) if level > 0 else ''


def build_manifest(video_root: str, out_csv: str, audio_root: Optional[str] = None, seed: int = 25,
                   test_per_class: int = 700, val_per_class: int = 700, protocol: str = 'uffia',
                   folds: int = 5, group_level: int = 1) -> str:
    """Write the clip manifest. ``protocol``:

    * ``uffia``: the U-FFIA code split. Each class is shuffled with ``RandomState(seed)``, then
      700 clips go to test, 700 to val and the rest to train. Files are **sorted** first: the
      U-FFIA code uses unsorted ``os.listdir``, so its split is not portable across file systems.
      The sorted version is deterministic, and the manifest hash identifies it.
    * ``kfold``: stratified ``folds``-fold CV (split column ``fold0`` ... ``foldK-1``), the
      HMMF-Net protocol. Select a fold at training time with ``data.cv_fold``.
    * ``grouped``: session-disjoint. Whole recording sessions (the ``group_level`` directories
      above the class folder, e.g. a recording date) are assigned to test, val or train, so
      adjacent 2 s clips cut from the same minute of recording cannot leak across the split.
      All published AV-FFIA results use random clip-level splits.
    """
    rows = []
    rng = np.random.RandomState(seed)
    per_class = {cls: sorted(glob.glob(os.path.join(video_root, '**', cls, '*.mp4'), recursive=True))
                 for cls in CLASSES}
    if protocol == 'grouped':
        sessions = sorted({_session(v, video_root, group_level) for vs in per_class.values() for v in vs})
        rng.shuffle(sessions)
        n_test, n_val = test_per_class * len(CLASSES), val_per_class * len(CLASSES)
        counts = {}
        for vs in per_class.values():
            for v in vs:
                k = _session(v, video_root, group_level)
                counts[k] = counts.get(k, 0) + 1
        assign, n = {}, {'test': 0, 'val': 0}
        for k in sessions:  # fill test, then val, each with whole sessions; the rest is train
            dst = 'test' if n['test'] < n_test else 'val' if n['val'] < n_val else 'train'
            assign[k] = dst
            if dst != 'train':
                n[dst] += counts[k]
        if len(sessions) < 3:
            raise ValueError(f'only {len(sessions)} sessions at group_level={group_level}; use a deeper level')
    for cls, vids in per_class.items():
        vids = list(vids)
        if protocol != 'grouped':
            np.random.RandomState(seed).shuffle(vids)
        for i, v in enumerate(vids):
            if protocol == 'uffia':
                split = 'test' if i < test_per_class else 'val' if i < test_per_class + val_per_class else 'train'
            elif protocol == 'kfold':
                split = f'fold{i % folds}'
            elif protocol == 'grouped':
                split = assign[_session(v, video_root, group_level)]
            else:
                raise ValueError(protocol)
            rows.append({'video': v, 'audio': _find_audio(v, video_root, audio_root), 'label': LABELS[cls],
                         'split': split, 'session': _session(v, video_root, max(group_level, 1))})
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['video', 'audio', 'label', 'split', 'session'])
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for r in rows:
        summary.setdefault(r['split'], [0] * len(CLASSES))[int(r['label'])] += 1
    digest = hashlib.sha256('\n'.join(f"{os.path.relpath(r['video'], video_root)},{r['split']}"
                                       for r in rows).encode()).hexdigest()[:16]
    print(f'protocol={protocol} clips={len(rows)} missing_audio={sum(not r["audio"] for r in rows)} '
          f'split_hash={digest}')
    for k in sorted(summary):
        print(f'  {k}: {sum(summary[k])} ({", ".join(f"{c}={n}" for c, n in zip(CLASSES, summary[k]))})')
    return out_csv


def read_manifest(path: str, split: str, cv_fold: Optional[int] = None, val_every: int = 8) -> List[Dict]:
    """Rows of one split. For a k-fold manifest with ``cv_fold=f``: fold f is the test set, and
    every ``val_every``-th clip of the remaining folds (deterministically) forms val."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if cv_fold is None:
        return [r for r in rows if r['split'] == split]
    test = [r for r in rows if r['split'] == f'fold{cv_fold}']
    rest = [r for r in rows if r['split'] != f'fold{cv_fold}']
    if split == 'test':
        return test
    return [r for i, r in enumerate(rest) if (i % val_every == 0) == (split == 'val')]


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------
def load_audio(path: str, video_path: str, sample_rate: int, seconds: float) -> torch.Tensor:
    import torchaudio
    n = int(sample_rate * seconds)
    wav, sr = None, None
    if path:
        try:
            wav, sr = torchaudio.load(path)
        except Exception:  # fall back to soundfile
            import soundfile as sf
            data, sr = sf.read(path, dtype='float32', always_2d=True)
            wav = torch.from_numpy(data.T)
    else:  # the mp4's own audio track
        import av
        with av.open(video_path) as c:
            stream = c.streams.audio[0]
            sr = stream.rate
            wav = torch.from_numpy(np.concatenate([f.to_ndarray().reshape(f.layout.nb_channels if f.format.is_planar
                                   else 1, -1) for f in c.decode(stream)], axis=1).astype(np.float32))
    wav = wav.float().mean(0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.numel() < n:
        wav = F.pad(wav, (0, n - wav.numel()))
    return wav[:n]


def load_video(path: str, num_frames: int, size: int) -> torch.Tensor:
    """Return uint8 ``(T, C, H, W)``: ``num_frames`` uniformly sampled frames, short side resized to ``size``."""
    frames = None
    try:
        import decord  # fastest, if available
        vr = decord.VideoReader(path)
        idx = np.linspace(0, len(vr) - 1, num_frames).round().astype(int)
        frames = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
    except ImportError:  # PyAV (torchvision >= 0.24 no longer ships video decoding)
        import av
        with av.open(path) as c:
            v = np.stack([f.to_ndarray(format='rgb24') for f in c.decode(video=0)])
        idx = np.linspace(0, len(v) - 1, num_frames).round().astype(int)
        frames = torch.from_numpy(v[idx]).permute(0, 3, 1, 2)
    T, C, H, W = frames.shape
    s = size / min(H, W)
    frames = F.interpolate(frames.float(), size=(round(H * s), round(W * s)), mode='bilinear',
                           align_corners=False, antialias=True)
    return frames.clamp(0, 255).to(torch.uint8)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class AVFFIADataset(Dataset):
    """Returns ``{'waveform': (L,), 'video': (3, T, crop, crop) float in [0,1], 'target': int}``.

    ``cache_dir`` stores the decoded (resized, uint8) frames and resampled audio of each clip
    after the first read. AV-FFIA clips are short, so later epochs are I/O-light.
    """

    def __init__(self, manifest: str, split: str, sample_rate: int = 64000, seconds: float = 2.0,
                 num_frames: int = 16, size: int = 128, crop: int = 112, train: Optional[bool] = None,
                 cache_dir: Optional[str] = None, audio_gain_db: float = 6.0, cv_fold: Optional[int] = None):
        self.items = read_manifest(manifest, split, cv_fold)
        self.split, self.sr, self.seconds = split, sample_rate, seconds
        self.num_frames, self.size, self.crop = num_frames, size, crop
        self.train = (split == 'train') if train is None else train
        self.cache_dir = os.path.join(cache_dir, f'sr{sample_rate}_t{num_frames}_s{size}') if cache_dir else None
        self.audio_gain_db = audio_gain_db

    def __len__(self):
        return len(self.items)

    def _load(self, i):
        it = self.items[i]
        if self.cache_dir:
            p = os.path.join(self.cache_dir, hashlib.md5(it['video'].encode()).hexdigest() + '.pt')
            if os.path.exists(p):
                return torch.load(p)
        sample = {'waveform': load_audio(it['audio'], it['video'], self.sr, self.seconds),
                  'frames': load_video(it['video'], self.num_frames, self.size)}
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(sample, p)
        return sample

    def __getitem__(self, i):
        s = self._load(i)
        wav, frames = s['waveform'].clone(), s['frames']
        T, C, H, W = frames.shape
        c = self.crop
        if self.train:
            # video: random crop, horizontal flip, mild brightness/contrast jitter
            y, x = random.randint(0, H - c), random.randint(0, W - c)
            frames = frames[..., y:y + c, x:x + c].float() / 255
            if random.random() < 0.5:
                frames = frames.flip(-1)
            frames = ((frames - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.1, 0.1)).clamp(0, 1)
            # audio: random gain and random circular time shift
            wav = wav * 10 ** (random.uniform(-self.audio_gain_db, self.audio_gain_db) / 20)
            wav = torch.roll(wav, random.randint(0, wav.numel() - 1))
        else:
            y, x = (H - c) // 2, (W - c) // 2
            frames = frames[..., y:y + c, x:x + c].float() / 255
        return {'waveform': wav, 'video': frames.permute(1, 0, 2, 3).contiguous(),
                'target': int(self.items[i]['label'])}


class SyntheticAVDataset(Dataset):
    """Random AV clips with weakly class-dependent structure, for smoke tests and benchmarking without the data."""

    def __init__(self, n=64, sample_rate=64000, seconds=2.0, num_frames=16, crop=112, num_classes=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.y = torch.randint(0, num_classes, (n,), generator=g)
        self.n, self.L, self.T, self.c, self.sr = n, int(sample_rate * seconds), num_frames, crop, sample_rate

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        y = int(self.y[i])
        g = torch.Generator().manual_seed(i)
        t = torch.arange(self.L) / self.sr
        wav = 0.1 * torch.randn(self.L, generator=g) + 0.2 * (y + 1) * torch.sin(2 * np.pi * 500 * (y + 1) * t)
        vid = (torch.rand(3, self.T, self.c, self.c, generator=g) * 0.5 + 0.1 * y).clamp(0, 1)
        return {'waveform': wav, 'video': vid, 'target': y}


def make_loader(ds: Dataset, batch_size: int, train: bool, num_workers: int = 8) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
