#!/usr/bin/env bash
# Teacher ablation for AV-FFIA. For each teacher T (scripts/make_teacher_configs.py):
#   1) train T: gives T's AV, audio-only and video-only accuracy (plus robustness)
#   2) distil the HyperAVNet student from T: gives the student's accuracy
# so the write-up can report "teacher -> student" and test whether stronger teachers make
# stronger students. A no-KD student is trained as the reference.
# Usage: bash scripts/run_teacher_ablation.sh [extra overrides, e.g. train.epochs=30]
#        TRAIN_ARGS=--synthetic ONLY="T1_cnn10_s3d" bash scripts/run_teacher_ablation.sh train.epochs=1
set -euo pipefail
EXTRA=("$@")
OUT=${OUT:-runs_teacher}
SETS=("${EXTRA[@]/#/--set=}")
python scripts/make_teacher_configs.py --out "$OUT/configs" ${ONLY:+--only $ONLY} "${SETS[@]}"
for cfg in "$OUT"/configs/*.yaml; do
  name=$(basename "$cfg" .yaml)
  python train.py ${TRAIN_ARGS:-} -c "$cfg" --set output_dir="$OUT" "${SETS[@]}"
  python train.py ${TRAIN_ARGS:-} -c configs/student_hyper_kd.yaml --set name="student_from_$name" \
      --set output_dir="$OUT" --set distill.teacher_config="$cfg" \
      --set distill.teacher_ckpt="$OUT/teacher_$name/best.pt" "${SETS[@]}"
done
python train.py ${TRAIN_ARGS:-} -c configs/student_hyper.yaml --set name=student_no_kd --set output_dir="$OUT" "${SETS[@]}"
python scripts/summarize.py "$OUT"
