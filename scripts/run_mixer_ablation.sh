#!/usr/bin/env bash
# AV-MixNet ablations (EfficientAT mn10 + X3D-M). Runs are named M_<variant>_s<seed> so that
# scripts/make_paper_assets.py can average seeds.
# Usage: SEEDS="0 1 2" bash scripts/run_mixer_ablation.sh [extra overrides...]
#        TRAIN_ARGS=--synthetic SEEDS=0 bash scripts/run_mixer_ablation.sh train.epochs=1 train.batch_size=2
set -euo pipefail
CFG=configs/mixnet_effat_x3d.yaml
EXTRA=("$@")
run() { local name=$1; shift; for sd in ${SEEDS:-0}; do
  python train.py ${TRAIN_ARGS:-} -c "$CFG" --set name="M_${name}_s${sd}" --set seed="$sd" "$@" "${EXTRA[@]/#/--set=}"; done; }

# reference points
run late_only       --set "model.mix_stages=[3]"                    # fusion only after the last stage
run no_mixer_scale1 --set model.scale_start=0.1 --set model.scale_end=0.1   # flat schedule
# mix operator (write fixed: xattn+film)
run mix_sra   --set model.mixer.mix=sra
run mix_full  --set model.mixer.mix=full
run mix_time  --set model.mixer.mix=time
run mix_hcf   --set model.mixer.mix=hcf
# write operator (mix fixed: sra)
run write_xattn --set "model.mixer.write=[xattn]"
run write_film  --set "model.mixer.write=[film]"
run write_guide --set "model.mixer.write=[guide]"
run write_all   --set "model.mixer.write=[xattn,film,guide]"
# where to mix
run stages_23   --set "model.mix_stages=[2,3]"
run stages_0123 --set "model.mix_stages=[0,1,2,3]"
# training schedule
run no_freeze   --set train.freeze_fraction=0.0
run freeze50    --set train.freeze_fraction=0.5
run bn_train    --set model.audio.bn_eval=false --set model.video.bn_eval=false
