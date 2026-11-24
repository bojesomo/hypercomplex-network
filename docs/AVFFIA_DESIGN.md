# HyperAVNet: hypercomplex audio-visual fish feeding intensity assessment (AV-FFIA)

This document describes the model, the training recipe and the ablation protocol in `avffia/`,
`train.py`, `extract_features.py`, `configs/` and `scripts/`. Everything builds on the
`fast_hypercomplex` layers in this repository (Bojesomo et al., IEEE SPM 2024).

## 1. Task and protocol (kept identical to U-FFIA for comparability)

* AV-FFIA: 27,000 synchronised 2 s audio and video clips of *Oplegnathus punctatus* (6,750 per
  class). Hydrophone audio at 256 kHz, downsampled to 64 kHz; video at 25 fps. Label ids follow
  the U-FFIA code: `none=0, strong=1, medium=2, weak=3`.
* Split: for each class, shuffle with `np.random.RandomState(25)`, then take 700 test clips,
  700 val clips and the rest as train. `python train.py --build-manifest ...` reproduces this rule
  and writes `data/avffia.csv`. This gives 21,400 train clips; the paper's text says 21,000,
  which does not add up to 27,000. Val and test (2,800 each) agree.
* Audio: 2 s at 64 kHz, log-mel with window 2048, hop 1024 and 128 bins (U-FFIA §VI-A), with
  optional SimPF spectral pooling (`simpf: 0.5`). Video: 16 uniformly sampled frames (U-FFIA's
  S3D used 20; the U-FFIA model 4).
* Robustness (U-FFIA Table III protocol, automatic after training): audio noise at SNR −10, −5,
  0, 10 and 20 dB (use U-FFIA-style bubble and pump recordings via `robustness.noise_dir`;
  otherwise white noise, which is not comparable), video darkness and Gaussian noise, and
  missing audio or missing video.
* Metrics: accuracy (the U-FFIA headline metric), macro-F1, per-class F1 and the confusion matrix,
  plus audio-only and video-only accuracy from the auxiliary heads, parameters and FLOPs.
  Model selection is on val accuracy; test is reported once.

### Published results to beat (U-FFIA, arXiv:2309.05058v3, Tables I–IV)

The compute unit in U-FFIA's tables is multiply-accumulates (MACs), although they label it
FLOPs. Checked: torchvision S3D at 20×224² gives 22.4 GMACs with our counter, against 22.8 in
their Table II. `train.py` reports both `gflops` and `gmacs`.

| model | modality | acc. | params (M) | GMACs |
|---|---|---|---|---|
| Pre-CNN10 (best audio) | A | 0.859 | 5.0 | 3.347 |
| Pre-AST | A | 0.736 | 85.3 | 12.440 |
| Pretrained-S3D (best video, 20 frames) | V | 0.898 | 7.9 | 22.826 |
| Pre-ViViT | V | 0.743 | 78.2 | 77.344 |
| **Cross-attention (SOTA)** | AV | **0.917** | 35.082 | 60.933 |
| Self-attention | AV | 0.892 | 47.695 | 68.404 |
| MBT | AV | 0.724 | 47.689 | 68.690 |
| U-FFIA + KD | AV | 0.907 | 21.626 | 15.488 |

AV accuracy under bubble and pump noise, from −10 to 20 dB: cross-attention 0.727 / 0.768 /
0.805 / 0.834 / 0.867; U-FFIA + KD 0.726 / 0.779 / 0.806 / 0.835 / 0.876.

### Later work: the true state of the art (literature review, Nov 2026)

The full review, with sources and a verified/snippet label on every number, is in
[reports/AV FFIA state of the art.md](../reports/AV%20FFIA%20state%20of%20the%20art.md).
0.917 is **not** the current bar. The later AV results all use different protocols:

| model | AV acc. | params / compute | protocol | source |
|---|---|---|---|---|
| LDMF (2026) | 98.22% | 7.10M | filtered subset (6,016 clips) | snippet |
| HMKD (Inf. Proc. Agric. 2026; teacher EfficientAT + MoViNet-A0) | 97.47 ± 0.61% | ≈60% < teacher | 26,843 pairs, holdout | snippet |
| HMMF-Net (arXiv 2611.09224) | 96.77 ± 0.27% | 3.987M / 1.002 "GFLOPs" | 5-fold CV, 27,067 clips | snippet |
| **DynaGate-AV** (GitHub) | **96.5% / 96.4%** | 5.64 / 5.29 "GFLOPs" | **official code split**, 19,963 clips | verified (repo) |
| ACAF-Net (Animals 2025) | 95.4% | n/r | random clip split | snippet |

Single-modality bests are still about 0.86 (audio) and 0.90 (video). Re-runs of the same
baseline vary by about 1.5 pp. The public Zenodo release may hold only 19,963 clips. All
published splits are random at clip level, so they carry a leakage risk.

**Targets:** > 96.5% on the official split and > 96.8% under 5-fold CV, with the fewest
parameters, plus the first session-disjoint result. Protocols: `--protocol uffia | kfold |
grouped` in `train.py --build-manifest`, and `data.cv_fold` for k-fold.

**Teacher ablation:** run `scripts/run_teacher_ablation.sh` (T1–T9, see
`scripts/make_teacher_configs.py`): CNN10 / CNN14 / EfficientNet-mel / AST × S3D / R(2+1)D /
ConvNeXt-frames / X3D-M / VideoMAE, plus T10 = HMKD's EfficientAT mn10_as + MoViNet-A0
(5.1M params, 1.05 GMACs). Each teacher is trained, then a student is distilled from
it, plus a no-KD reference.

What this implies for the design:
1. Pretrained CNNs beat transformers on this data, so the teacher is **PANNs CNN10 + S3D**
   (`configs/teacher_panns_s3d.yaml`). The AST + VideoMAE teacher is kept as an ablation only.
2. MBT fails on AV-FFIA (0.724), which U-FFIA attributes to sequential bottleneck attention.
   HCF reads both modalities in parallel and mixes them in one Hamilton product.
3. The targets are **> 0.917 accuracy** and better noise robustness, at about 2 M parameters
   and about 1.2 GMACs: 10× / 13× below U-FFIA and 16× / 52× below cross-attention.

## 2. Architecture

```
waveform (B, 128000) --> audio backbone --> a_0, a_1, a_2, a_3   (stage tokens, shallow -> deep)
video (B, 3, 16, H, W) -> video backbone --> v_0, v_1, v_2, v_3
                                     |  StageProjector_s : HyperLinear(C_s -> D) + LN + modality emb.
 bottleneck tokens Z (8 x D) --[ Fusion_s for s in fusion_stages ]--> Z
                                     |  (stage-wise students) g_s = tanh(HyperLinear(mean Z)) --> h * (1 + g_s)
 head: HyperLinear([mean Z, mean a_3, mean v_3]) -> GELU -> Linear -> 4 logits
 aux heads: audio-only logits, video-only logits
```

### 2.1 Hypercomplex everywhere parameters live

`n_divs = n` turns every projection (point-wise convs, attention q/k/v/o, MLPs, projectors, the
head) into a hypercomplex layer, which has `1/n` of the real layer's parameters through the
Hamilton-product weight sharing of `fast_hypercomplex`. `n = 1` is the real-valued ablation with
the same code path. Depth-wise convolutions (already about 1% of the parameters) stay real.

### 2.2 Hypercomplex Component Fusion (HCF), the proposed fusion block

At each fusion stage the bottleneck tokens `Z` cross-attend to the audio tokens (`za`) and the
video tokens (`zv`) separately, using hypercomplex attention. The two views and their
interactions then become the **components of one hypercomplex number**:

```
quaternion (algebra = 4): q = Z·e0 + za·e1 + zv·e2 + (za ⊙ zv)·e3
octonion   (algebra = 8): + (za − zv)·e4 + (Z ⊙ za)·e5 + (Z ⊙ zv)·e6 + (Z ⊙ za ⊙ zv)·e7
Z ← Z + LayerScale(HyperLinear_algebra(q))  →  Z ← Z + LayerScale(HyperMLP(LN Z))
```

The Hamilton product forces *every* output component to be a signed mix of *every* input
component through one shared weight set. Cross-modal interaction terms are therefore built
into the algebra rather than learned by a dense `4D x D` mixer. The mixer has `D²` parameters
whatever the algebra, so the octonion version adds interaction terms at no parameter cost.
`mbt` (an MBT bottleneck, one of the fusion baselines benchmarked in the U-FFIA paper, where it
scored 0.724; built here from hypercomplex blocks) and
`concat` are included as fusion-operator ablations.

### 2.3 Fusion depth: the deep-to-early ablation axis

`fusion_stages` chooses where fusion happens: `[3]` is late fusion; `[2,3]`, `[1,2,3]` and
`[0,1,2,3]` push progressive fusion earlier; `[k]` alone isolates single-stage fusion at depth
`k`. With the stage-wise hypercomplex students (`inject: true`), the fused tokens are fed
**back** into both streams through zero-initialised hypercomplex gates, `h ← h·(1+tanh(W·mean Z))`.
This is true early fusion: video features at later stages are conditioned on the audio, and
vice versa. Zero initialisation means training starts exactly at the late-fusion model
(this is unit-tested).

### 2.4 Backbones: pretrained models used as teachers or feature extractors

| key | model | role |
|---|---|---|
| `hyper_audio` | hypercomplex inverted-residual CNN on log-mel (0.75M at n=4) | efficient student |
| `hyper_video` | hypercomplex 3D inverted-residual CNN (0.82M at n=4) | efficient student |
| `panns` | PANNs CNN10 / CNN14, AudioSet (official checkpoint names); best audio model on AV-FFIA | **teacher** / extractor |
| `ast` | AST, AudioSet-pretrained, position embeddings interpolated to 2 s; 0.736 in U-FFIA | ablation |
| `videomae` | VideoMAE-B, Kinetics-pretrained, tokens grid-pooled; ViViT got 0.743 in U-FFIA | ablation |
| `timm_audio` | any timm CNN on 1-channel mel (EfficientNet, MobileNetV2/V4, ConvNeXt) | extractor |
| `timm_frames` | any timm 2D net per frame, plus a hypercomplex temporal adapter | extractor |
| `tv_video` | torchvision Kinetics S3D (U-FFIA's video teacher, 0.898), R(2+1)D-18, R3D-18, MC3-18 | **teacher** / extractor |

`freeze: true` with `unfreeze: [...]` gives frozen or partly fine-tuned extractors. The fusion
network then runs as a hypercomplex *side network* over their intermediate layers (cheap
ladder-style tuning). `extract_features.py` caches those stage tokens once (with several
augmented views), after which fusion training takes minutes.

### 2.5 Training objectives

* `AVLoss`: label-smoothed CE on the fused logits, plus 0.3 × the mean CE of the audio-only and
  video-only heads (counters modality imbalance; the weaker audio branch would otherwise
  under-train).
* Modality dropout (p = 0.1 per modality; U-FFIA notation p_av = 1 − 2p, p_a = p_v = p). One
  per-sample mask applies at every fusion input and at the fused head, and the same switch
  (`model.missing`) simulates a missing modality at test time. U-FFIA Table V shows that heavy
  dropout costs clean AV accuracy; the auxiliary heads let HyperAVNet keep p small.
* `DistillLoss` (student ← teacher): KL at T = 2 on the fused logits (`style: uffia` gives
  U-FFIA's eq. 9, with T = 2.5 on the teacher only), plus unimodal head-to-head KL;
  cosine alignment of the pooled fused, audio and video embeddings through hypercomplex
  projectors; and RKD-style batch-relation matching.
* AdamW, cosine schedule with warmup, EMA weights (0.999), bf16 AMP, gradient clipping. Pretrained
  weights train at 0.1× the learning rate.

### 2.6 Deployment

`reparameterize_hypercomplex(model)` folds every Hamilton kernel into a plain dense
`Linear`/`Conv`. The outputs are identical (tested), and CPU latency of the default student
drops from 65 ms to 46 ms. `train.py` writes this as `deploy.pt`; load it by building the model,
calling `reparameterize_hypercomplex`, then `load_state_dict`.

**Honest efficiency note.** Hypercomplex layers cut **parameters and memory** (`1/n`), not
FLOPs: the expanded kernel is dense. Default student (`configs/student_hyper.yaml`, 16×112²
frames + 2 s audio): **2.14M params at n=4** against 7.88M for the identical real network (n=1),
both at 2.36 GFLOPs = **1.18 GMACs** per clip. That is 1.09 GMACs with SimPF 0.5, 0.69 at 8
frames and 1.42 at 20 frames. To reduce compute, shrink `dims`, `num_frames` or the input size.

## 3. Recommended recipe for SOTA

1. **Teacher**: `configs/teacher_panns_s3d.yaml`. PANNs CNN10 (AudioSet; put
   `Cnn10_mAP=0.380.pth` from Zenodo 3987831 in `pretrained/`) + S3D (Kinetics), fine-tuned
   with HCF fusion over their last four stages: 15.9M params, 21.5 GMACs.
2. **Student**: `configs/student_hyper_kd.yaml`. A 2.1M-parameter quaternion HyperAVNet with
   progressive fusion `[1,2,3]` and feedback, distilled from the teacher. This is the
   efficient model to report against SOTA.
3. **Cheap alternative**: `configs/frozen_timm_side.yaml` with `extract_features.py` (frozen
   ImageNet backbones; only about 1.2M trainable fusion parameters).

Things to tune on AV-FFIA, on val only: the frame count (U-FFIA's S3D used 20), the input
resolution, SimPF on or off, `n_divs ∈ {4, 8}`, `num_bottleneck`, modality dropout, and the KD
style and temperature.

## 4. Ablation protocol (`scripts/run_ablations.sh`)

| group | axis | runs |
|---|---|---|
| A | fusion depth, deep to early | `[3]`, `[2,3]`, `[1,2,3]`, `[0,1,2,3]`; single `[0]`, `[1]`, `[2]`; `[1,2,3]` without feedback |
| B | algebra of the network | n = 1, 2, 4, 8, 16 |
| C | fusion operator | HCF-complex, HCF-quaternion, HCF-octonion, MBT, concat |
| D | regularisers | modality dropout p ∈ {0, 0.1, 0.25} (cf. U-FFIA Table V), no auxiliary heads |
| E | distillation | the A–D winners with and without `distill` (base: `configs/student_hyper_kd.yaml`); multi-level KD vs U-FFIA eq. (9) |
| F | robustness | automatic after each run: SNR −10…20 dB, darkness, Gaussian noise, missing modality |
| G | audio front-end | SimPF k = 0.5 vs none |

```bash
bash scripts/run_ablations.sh configs/student_hyper.yaml
bash scripts/run_ablations.sh configs/student_hyper_kd.yaml output_dir=runs_kd
python scripts/summarize.py runs > results.md
```

Run each configuration with 3 seeds (`--set seed=...`, keeping the split seed fixed by
rebuilding the manifest only once), and report mean ± std. The differences between AV-FFIA
models are small.

## 5. Results table template

| model | modality | params (M) | GMACs | test acc | macro-F1 |
|---|---|---|---|---|---|
| Pre-CNN10 [U-FFIA] | A | 5.0 | 3.347 | 0.859 | n/r |
| Pretrained-S3D [U-FFIA] | V | 7.9 | 22.826 | 0.898 | n/r |
| Cross-attention [U-FFIA], SOTA | AV | 35.082 | 60.933 | 0.917 | n/r |
| U-FFIA + KD [U-FFIA] | AV | 21.626 | 15.488 | 0.907 | n/r |
| Late-concat real baseline (this pipeline) | AV | 14.938 | 21.252 | | |
| Teacher: PANNs CNN10 + S3D + HCF | AV | 15.874 | 21.483 | | |
| **HyperAVNet-Q (KD)** | AV | 2.137 | 1.180 | | |
| HyperAVNet-O (KD) | AV | 1.235 | 1.180 | | |

## Pretrained model downloads

All links, and how each was verified, are in [PRETRAINED_MODELS.md](PRETRAINED_MODELS.md).
Run `bash scripts/download_pretrained.sh --cache`.
