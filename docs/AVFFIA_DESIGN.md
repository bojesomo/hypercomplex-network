# HyperAVNet: hypercomplex audio-visual fish feeding intensity assessment (AV-FFIA)

This document describes the model, the training recipe and the ablation protocol in `avffia/`,
`train.py`, `extract_features.py`, `configs/` and `scripts/`. Everything builds on the
`fast_hypercomplex` layers in this repository (Bojesomo et al., IEEE SPM 2024).

## 1. Task and protocol (kept identical to U-FFIA for comparability)

* AV-FFIA: about 27k paired audio and video clips, 4 classes. Label ids follow the U-FFIA code:
  `none=0, strong=1, medium=2, weak=3`.
* Split: for each class, shuffle with `np.random.RandomState(25)`, then take 700 test clips,
  700 val clips and the rest as train. `python train.py --build-manifest ...` reproduces this rule
  and writes `data/avffia.csv`.
* Audio: 2 s at 64 kHz (as in U-FFIA). Video: 16 uniformly sampled frames.
* Metrics: accuracy (the U-FFIA headline metric), macro-F1, per-class F1 and the confusion matrix,
  plus audio-only and video-only accuracy from the auxiliary heads, parameters and FLOPs.
  Model selection is on val accuracy; test is reported once.

> **Check against the paper before claiming SOTA.** The arXiv paper (2309.05058) was not
> reachable from the build environment, so the published U-FFIA numbers are *not* copied here.
> Fill table 6 from the paper, and also run `configs/uffia_like_baseline.yaml` (MobileNetV2 +
> S3D + MBT, real-valued) through this exact pipeline, so the comparison is like-for-like.

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
`mbt` (an MBT bottleneck, the U-FFIA baseline fusion, built here from hypercomplex blocks) and
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
| `ast` | AST, AudioSet-pretrained, position embeddings interpolated to 2 s (about 5x cheaper than padding to 10 s) | teacher / extractor |
| `videomae` | VideoMAE-B, Kinetics-pretrained, tokens grid-pooled | teacher / extractor |
| `timm_audio` | any timm CNN on 1-channel mel (EfficientNet, MobileNetV2/V4, ConvNeXt) | extractor |
| `timm_frames` | any timm 2D net per frame, plus a hypercomplex temporal adapter | extractor |
| `tv_video` | torchvision Kinetics R(2+1)D-18, R3D-18, MC3-18, S3D (the U-FFIA video backbone) | extractor / baseline |

`freeze: true` with `unfreeze: [...]` gives frozen or partly fine-tuned extractors. The fusion
network then runs as a hypercomplex *side network* over their intermediate layers (cheap
ladder-style tuning). `extract_features.py` caches those stage tokens once (with several
augmented views), after which fusion training takes minutes.

### 2.5 Training objectives

* `AVLoss`: label-smoothed CE on the fused logits, plus 0.3 × the mean CE of the audio-only and
  video-only heads (counters modality imbalance; the weaker audio branch would otherwise
  under-train).
* Modality dropout (p = 0.1 per modality) at the fusion inputs: robustness when one modality is
  uninformative (turbid water, rain or pump noise).
* `DistillLoss` (student ← teacher): KL at T = 2 on the fused logits, plus unimodal head-to-head KL;
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
both at 2.37 GFLOPs per clip. To reduce FLOPs, shrink `dims`, `num_frames` or the input size.

## 3. Recommended recipe for SOTA

1. **Teacher**: `configs/teacher_ast_videomae.yaml`. AST + VideoMAE-B with the last two blocks
   unfrozen, and HCF side-fusion at layers 3, 6, 9 and 12. This aims at the highest possible
   accuracy; its size does not matter.
2. **Student**: `configs/student_hyper_kd.yaml`. A 2.1M-parameter quaternion HyperAVNet with
   progressive fusion `[1,2,3]` and feedback, distilled from the teacher. This is the
   efficient model to report against SOTA.
3. **Cheap alternative**: `configs/frozen_timm_side.yaml` with `extract_features.py` (frozen
   ImageNet backbones; only about 1.2M trainable fusion parameters).

Things to tune on AV-FFIA (they are dataset-specific and unverified here): the frame count
(the U-FFIA code hints at 20 frames), the input resolution, the audio hop (512 is used here
instead of U-FFIA's 1024, for finer time resolution), `n_divs ∈ {4, 8}`, `num_bottleneck`, and
the KD temperature.

## 4. Ablation protocol (`scripts/run_ablations.sh`)

| group | axis | runs |
|---|---|---|
| A | fusion depth, deep to early | `[3]`, `[2,3]`, `[1,2,3]`, `[0,1,2,3]`; single `[0]`, `[1]`, `[2]`; `[1,2,3]` without feedback |
| B | algebra of the network | n = 1, 2, 4, 8, 16 |
| C | fusion operator | HCF-complex, HCF-quaternion, HCF-octonion, MBT, concat |
| D | regularisers | no modality dropout, no auxiliary heads |
| E | distillation | the A–D winners with and without `distill` (use `configs/student_hyper_kd.yaml` as the base) |

```bash
bash scripts/run_ablations.sh configs/student_hyper.yaml
bash scripts/run_ablations.sh configs/student_hyper_kd.yaml output_dir=runs_kd
python scripts/summarize.py runs > results.md
```

Run each configuration with 3 seeds (`--set seed=...`, keeping the split seed fixed by
rebuilding the manifest only once), and report mean ± std. The differences between AV-FFIA
models are small.

## 5. Results table template

| model | modality | params (M) | GFLOPs | test acc | macro-F1 |
|---|---|---|---|---|---|
| U-FFIA paper: audio-only best | A | *(from paper)* | | | |
| U-FFIA paper: video-only best | V | *(from paper)* | | | |
| U-FFIA paper: AV (MBT) | AV | *(from paper)* | | | |
| U-FFIA-like baseline (this pipeline) | AV | | | | |
| Teacher: AST + VideoMAE + HCF | AV | | | | |
| **HyperAVNet-Q (KD)** | AV | 2.14 | 2.37 | | |
| HyperAVNet-O (KD) | AV | 1.24 | 2.37 | | |
