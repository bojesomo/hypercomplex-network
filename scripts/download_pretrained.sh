#!/usr/bin/env bash
# Download every pretrained checkpoint used by the HyperAVNet teachers / extractors.
#   bash scripts/download_pretrained.sh            # file checkpoints into ./pretrained/
#   bash scripts/download_pretrained.sh --cache    # also pre-fill torchvision / timm / HF / pytorchvideo caches
# Links are documented (with how each was verified) in docs/PRETRAINED_MODELS.md.
set -euo pipefail
DIR=${PRETRAINED_DIR:-pretrained}
mkdir -p "$DIR"
FAILED=()
get() {  # get <url> <output file>; failures are collected, not fatal
  if [[ -s "$DIR/$2" ]]; then echo "ok    $2 (exists)"; return; fi
  echo "fetch $2"
  if curl -fsSL --retry 3 -o "$DIR/$2.part" "$1"; then mv "$DIR/$2.part" "$DIR/$2"; echo "ok    $2"
  else rm -f "$DIR/$2.part"; FAILED+=("$2  <-  $1"); echo "FAIL  $2"; fi
}
# PANNs (Kong et al., 2020), AudioSet, Zenodo record 3987831
get 'https://zenodo.org/record/3987831/files/Cnn10_mAP%3D0.380.pth?download=1' 'Cnn10_mAP=0.380.pth'
get 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1' 'Cnn14_mAP=0.431.pth'
# EfficientAT (Schmid et al., ICASSP 2023), AudioSet, GitHub release v0.0.1
get 'https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt' 'mn10_as_mAP_471.pt'
get 'https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt' 'mn04_as_mAP_432.pt'
# MoViNet-A0 Base (Kondratyuk et al., CVPR 2021), Kinetics-600, MoViNet-pytorch port
get 'https://raw.githubusercontent.com/Atze00/MoViNet-pytorch/main/weights/modelA0_statedict_v3' 'modelA0_statedict_v3'

if [[ "${1:-}" == "--cache" ]]; then
  # These libraries download into their own caches on first use; this just pre-fetches them.
  python - <<'PY'
import torchvision.models.video as tvv
tvv.s3d(weights='DEFAULT'); tvv.r2plus1d_18(weights='DEFAULT'); print('torchvision: S3D, R(2+1)D-18')
import timm
for n in ['efficientnet_b0.ra_in1k', 'convnext_tiny.in12k_ft_in1k', 'mobilenetv4_conv_medium.e500_r256_in1k']:
    timm.create_model(n, pretrained=True); print('timm:', n)
try:
    import pytorchvideo.models.hub as pvh
    pvh.x3d_m(pretrained=True); print('pytorchvideo: x3d_m')
except ImportError:
    print('pytorchvideo not installed (pip install pytorchvideo) - skipping X3D')
try:
    from transformers import ASTModel, VideoMAEModel
    ASTModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
    VideoMAEModel.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics'); print('HF: AST, VideoMAE-B')
except ImportError:
    print('transformers not installed - skipping AST / VideoMAE')
PY
fi
ls -la "$DIR"
if (( ${#FAILED[@]} )); then
  printf '\nThese downloads failed (network policy or host down); fetch them manually into %s/:\n' "$DIR"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
