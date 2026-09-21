#!/bin/bash
# GPU TAG MODELDIR STAGE [GD_NOTHINK=1]   STAGE: traces | jev | eval | all
#   traces  fitting traces of the executor (GPU, no API)          -> $T4/jev/fit_traces_TAG.json
#   jev     Jev conditions A,B,C (API, paid; needs JEV_API_KEY)   -> $T4/jev/full/tgb_jev{A,B,C}_masks_TAG.json + run log
#   eval    one session: LOTS, Keep all, Random, Jev A/B/C, Trace judge, Dense, Tool2Vec -> $T4/jev/eval_jev_on_TAG.json
GPU=$1 TAG=$2 DIR=$3 STAGE=${4:-all} NT=${5:-}
T4=/datasets/omni_pretraining/gta2/results/taco/tgb4 W=/datasets/omni_pretraining/wt_tgb O=$T4/jev
source /datasets/omni_pretraining/gta2/scripts_extra/new_model_env.sh
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$W GD_EAGER=1 GD_UTIL=${GD_UTIL:-0.85} HF_HUB_OFFLINE=1 $NT
cd $W; mkdir -p $O; F="it/s\]\|^INFO\|^WARNING\|^(EngineCore"
if [[ $STAGE == traces || $STAGE == all ]]; then
  [ -s $O/fit_traces_$TAG.json ] || $NEW_PY -m tgb.jev_fit_traces --dir $T4 --model $DIR --tag $TAG 2>&1 | grep -v "$F"
fi
if [[ $STAGE == jev || $STAGE == all ]]; then
  source ~/.jev_api_env
  $NEW_PY -m jev.tgb_run --dir $T4 --tag $TAG --conds A,B,C --traces $O/fit_traces_$TAG.json --out-dir $O/full 2>&1 | tail -5
fi
if [[ $STAGE == eval || $STAGE == all ]]; then
  $NEW_PY -m jev.lots_masks --dir $T4 --tag $TAG --out-dir $O
  X=""; for c in A B C; do X="$X --extra-mask jev$c=$O/full/tgb_jev${c}_masks_$TAG.json"; done
  for b in random:$O/tgb_random_masks_$TAG.json tracejudge:$T4/tgb_tracejudge_task_masks_$TAG.json dense:$T4/tgb_dense_base_task_masks_$TAG.json tool2vec:$T4/tgb_tool2vec_task_masks_$TAG.json; do
    [ -s ${b#*:} ] && X="$X --extra-mask ${b%%:*}=${b#*:}"; done
  $NEW_PY -m tgb.eval_baseline_masks --dir $T4 --mask $O/tgb_lots_masks_$TAG.json --method lots --out $O/eval_jev_on_$TAG.json --model $DIR --tag $TAG $X 2>&1 | grep -v "$F"
fi
