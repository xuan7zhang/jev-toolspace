#!/bin/bash
# GPU TAG MODELDIR [GD_NOTHINK=1]: one session: lots (anchor) + full + Jev A/B/C at threshold 0.5 (no K)  -> tgb4/jev/thr/eval_thr_<tag>.json
GPU=$1 TAG=$2 DIR=$3 NT=${4:-}
T4=/datasets/omni_pretraining/gta2/results/taco/tgb4; W=/datasets/omni_pretraining/wt_tgb; O=$T4/jev/thr
source /datasets/omni_pretraining/gta2/scripts_extra/new_model_env.sh
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$W GD_EAGER=1 GD_UTIL=${GD_UTIL:-0.85} HF_HUB_OFFLINE=1 $NT OMP_NUM_THREADS=4
cd $W; F="it/s\]\|^INFO\|^WARNING\|^(EngineCore"
X=""; for c in A B C; do X="$X --extra-mask jev${c}_thr=$O/tgb_jev${c}_thr0p5_masks_$TAG.json"; done
[ -s $O/eval_thr_$TAG.json ] || $NEW_PY -m tgb.eval_baseline_masks --dir $T4 --mask $T4/jev/tgb_lots_masks_$TAG.json --method lots --out $O/eval_thr_$TAG.json --model $DIR --tag $TAG $X 2>&1 | grep -v "$F"
