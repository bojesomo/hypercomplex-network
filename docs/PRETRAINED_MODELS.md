# Pretrained models: download links

Run `bash scripts/download_pretrained.sh` (add `--cache` to also pre-fetch the library-managed
weights). File checkpoints go into `pretrained/`; the configs point there.

"Verified" says how each link was checked (in Nov 2026):

- **downloaded**: fetched and loaded into our wrapper.
- **library source**: the URL is copied from the installed library's own code.
- **README / port**: copied from the authors' code or README.

## File checkpoints (`pretrained/`)

| Model | Role | Download | Verified |
|---|---|---|---|
| PANNs **CNN10** (AudioSet, mAP 0.380) | audio teacher T1, T3, T4, T7, T9 (U-FFIA's audio teacher) | https://zenodo.org/record/3987831/files/Cnn10_mAP%3D0.380.pth?download=1 | Zenodo record and file-name pattern from `panns_inference` source; Cnn10 name from the PANNs README |
| PANNs **CNN14** (AudioSet, mAP 0.431) | audio teacher T2 | https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1 | library source (`panns_inference`) |
| **EfficientAT mn10_as** (AudioSet, mAP 0.471, 4.88M) | audio teacher T10 (HMKD's audio teacher) | https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt | downloaded and loaded (all keys) |
| EfficientAT mn04_as (mAP 0.432, 0.98M) | optional smaller audio teacher | https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt | library source (EfficientAT `models/mn/model.py`) |
| **MoViNet-A0 Base** (Kinetics-600) | video teacher T10 (HMKD's video teacher) | https://raw.githubusercontent.com/Atze00/MoViNet-pytorch/main/weights/modelA0_statedict_v3 | downloaded and loaded (strict) |

Other EfficientAT sizes (`mn05/20/30/40_as`) are listed in `EFFICIENTAT_CKPT`
(`avffia/backbones.py`), from the same release.

## Library-managed weights (downloaded automatically on first use)

| Model | Role | Download (as used by the library) | Verified |
|---|---|---|---|
| torchvision **S3D** (Kinetics-400) | video teacher T1, T2, T5, T8 (U-FFIA's video teacher) | https://download.pytorch.org/models/s3d-d76dad2f.pth | library source (torchvision 0.29 `S3D_Weights`) |
| torchvision R(2+1)D-18 (Kinetics-400) | video teacher T3 | https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth | library source |
| pytorchvideo **X3D-M** (Kinetics-400) | video teacher T7 | https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth | library source (`pytorchvideo.models.hub.x3d`) |
| AST (AudioSet) | audio teacher T6, T8 | https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 | model id from the HF hub |
| VideoMAE-B (Kinetics-400) | video teacher T6, T9 | https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics | model id from the HF hub |
| ConvNeXt-T (IN-12k → IN-1k) | per-frame video teacher T4 | https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k | library source (timm `pretrained_cfg`) |
| EfficientNet-B0 (IN-1k) | mel-image audio teacher T5 | https://huggingface.co/timm/efficientnet_b0.ra_in1k | library source (timm) |
| MobileNetV4-M (IN-1k) | frozen-extractor video (`frozen_timm_side.yaml`) | https://huggingface.co/timm/mobilenetv4_conv_medium.e500_r256_in1k | library source (timm) |

## Dataset

| Item | Link |
|---|---|
| AV-FFIA, part 1 (audio ≈17.2 GB, video ≈21.8 GB) | https://zenodo.org/records/11059975 |
| AV-FFIA, further video parts and noise bundle | https://zenodo.org/records/11060195 · https://zenodo.org/records/11060370 · https://zenodo.org/records/14619756 |
| U-FFIA code (split rule, CNN10 / S3D teacher checkpoints on Google Drive) | https://github.com/FishMaster93/U-FFIA |

The Zenodo record ids come from the literature review: see `reports/AV FFIA state of the art.md`,
which notes that the public release may hold fewer clips than the paper states. After download,
check the clip count and the split hash that `python train.py --build-manifest` prints.

## Vendored code (MIT)

`avffia/third_party/efficientat` comes from https://github.com/fschmid56/EfficientAT and
`avffia/third_party/movinets` from https://github.com/Atze00/MoViNet-pytorch. Each folder keeps
its upstream `LICENSE`. MoViNet's config needs `fvcore` (`pip install fvcore`).
