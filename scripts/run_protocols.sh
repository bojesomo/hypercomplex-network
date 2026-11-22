#!/usr/bin/env bash
# Evaluate one model configuration under the three AV-FFIA protocols (paper Table "protocols"):
#   P_uffia_s<k>    official U-FFIA split rule (sorted, hashed), one run per seed
#   P_kfold_fold<k> 5-fold stratified cross-validation (HMMF-Net protocol)
#   P_grouped_s<k>  session-disjoint split (whole recording dates held out)
# Usage: VIDEO_ROOT=/data/AV-FFIA AUDIO_ROOT=/data/AV-FFIA/audio_dataset bash scripts/run_protocols.sh [config]
set -euo pipefail
CFG=${1:-configs/student_hyper_kd.yaml}
OUT=${OUT:-runs_proto}
SEEDS=${SEEDS:-"0 1 2"}
mkdir -p data
for p in uffia kfold grouped; do
  python train.py --build-manifest --video-root "$VIDEO_ROOT" --audio-root "$AUDIO_ROOT" \
      --manifest "data/avffia_$p.csv" --protocol "$p" --group-level "${GROUP_LEVEL:-1}"
done
for sd in $SEEDS; do
  python train.py -c "$CFG" --set name="P_uffia_s$sd" --set seed="$sd" --set output_dir="$OUT" --set data.manifest=data/avffia_uffia.csv
  python train.py -c "$CFG" --set name="P_grouped_s$sd" --set seed="$sd" --set output_dir="$OUT" --set data.manifest=data/avffia_grouped.csv
done
for k in 0 1 2 3 4; do
  python train.py -c "$CFG" --set name="P_kfold_fold$k" --set output_dir="$OUT" \
      --set data.manifest=data/avffia_kfold.csv --set data.cv_fold=$k
done
python scripts/make_paper_assets.py
