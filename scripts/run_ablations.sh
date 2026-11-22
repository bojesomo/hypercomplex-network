#!/usr/bin/env bash
# Ablation suite for the hypercomplex AV model on AV-FFIA. Each run writes runs/<name>/result.json;
# summarise with: python scripts/summarize.py runs
# Usage: bash scripts/run_ablations.sh [base_config] [extra --set overrides...]
set -euo pipefail
CFG=${1:-configs/student_hyper.yaml}; shift || true
EXTRA=("$@")
# TRAIN_ARGS (env) is passed through, e.g. TRAIN_ARGS=--synthetic for a dry run.
run() { local name=$1; shift; python train.py ${TRAIN_ARGS:-} -c "$CFG" --set name="$name" "$@" "${EXTRA[@]/#/--set=}"; }

# A. Fusion depth, deep -> early (progressive fusion from stage k onwards)
run A_fs3        --set "model.fusion_stages=[3]"
run A_fs23       --set "model.fusion_stages=[2,3]"
run A_fs123      --set "model.fusion_stages=[1,2,3]"
run A_fs0123     --set "model.fusion_stages=[0,1,2,3]"
#    single-stage fusion at each depth (where is fusion most useful?)
run A_fs0        --set "model.fusion_stages=[0]"
run A_fs1        --set "model.fusion_stages=[1]"
run A_fs2        --set "model.fusion_stages=[2]"
#    early fusion with versus without cross-modal feedback into the streams
run A_fs123_noinj --set "model.fusion_stages=[1,2,3]" --set model.inject=false

# B. Algebra of the whole network (1 real, 2 complex, 4 quaternion, 8 octonion, 16 sedenion)
for n in 1 2 4 8 16; do
  alg=$(( n < 2 ? 4 : (n > 8 ? 8 : n) ))
  run B_n${n} --set model.n_divs=$n --set model.audio.n_divs=$n --set model.video.n_divs=$n \
              --set model.fusion_algebra=$alg
done

# C. Fusion operator (at the best depth from A)
run C_hcf_q  --set model.fusion=hcf --set model.fusion_algebra=4
run C_hcf_o  --set model.fusion=hcf --set model.fusion_algebra=8
run C_hcf_c  --set model.fusion=hcf --set model.fusion_algebra=2
run C_mbt    --set model.fusion=mbt
run C_concat --set model.fusion=concat

# D. Regularisers
run D_no_moddrop --set model.modality_dropout=0.0
run D_no_aux     --set train.aux_weight=0.0
