#!/bin/bash
# LG TG JB TAG...  all comparator arms + Jev arms on read_arith,count_arith (t1) for each model in turn, one session per model
: "${1:?LG}" "${2:?TG}" "${3:?JB}"; LG=$1 TG=$2 JB=$3; shift 3; X=/datasets/omni_pretraining/gta2/scripts_extra; J=/datasets/omni_pretraining/gta2/results/jev; IO=/datasets/omni_pretraining/gta2/results/inject_opt
declare -A M=([3b]=Qwen2.5-3B-Instruct [7b]=Qwen2.5-7B-Instruct [14b]=Qwen2.5-14B-Instruct [q3_8b]=Qwen3-8B [q35_9b]=Qwen3.5-9B [phi4]=phi-4 [gemma4_12b]=gemma-4-12B-it)
declare -A NT=([q3_8b]=1 [q35_9b]=1 [phi4]=1 [gemma4_12b]=1)
P=$((17000 + RANDOM % 800))
for t in "$@"; do
  [ -s $IO/typebase1_${t}_jva.json ] && { echo "$t done"; continue; }; echo "=== $t start $(date)"
  EM="jevA=$J/$t/menus_A_{type}.json,jevB=$J/$t/menus_B_{type}.json,jevC=$J/$t/menus_C_{type}.json"
  EX=""; [ -n "${NT[$t]:-}" ] && EX="NOTHINK=1"; [ $t = gemma4_12b ] && EX="$EX GTA_GEMMA_ADAPTER=1"
  env API=0 MODEL=/datasets/omni_pretraining/gta2/models/${M[$t]} TYPES=read_arith,count_arith SPLIT=t1 FRESH=jva ARMS=notools,full,random,router,bm25,dense,tool2vec,tto,beam,jevA,jevB,jevC GTA_EXTRA_MENUS="$EM" TAG=jva_$t MENU_TAG=$t GTA_OBS_ROLE=user UTIL=0.85 MAXLEN=16384 LGPU=$LG TGPU=$TG LPORT=$((P+1)) TPORT=$((P+2)) PPORT=$((P+3)) JOB=$JB $EX bash $X/gta_jev_lane.sh
  P=$((P+10)); echo "=== $t end $(date) $( [ -s $IO/typebase1_${t}_jva.json ] && echo ok || echo MISSING)"
done
