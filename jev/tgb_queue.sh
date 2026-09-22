#!/bin/bash
# GPU JOB TAG[:NOTHINK]...  run tgb_lane.sh (traces -> Jev C -> one-session eval) for each tag in turn on one GPU inside SLURM job JOB
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
GPU=$1 JOB=$2; shift 2; T4=/datasets/omni_pretraining/gta2/results/taco/tgb4
declare -A M=([7b]=Qwen2.5-7B-Instruct [14b]=Qwen2.5-14B-Instruct [llama8b]=Llama-3.1-8B-Instruct [mistral7b]=Mistral-7B-Instruct-v0.3 [q35_9b]=Qwen3.5-9B [q3_8b]=Qwen3-8B [phi4]=phi-4 [gemma4_12b]=gemma-4-12B-it)
declare -A NT=([q35_9b]=GD_NOTHINK=1 [q3_8b]=GD_NOTHINK=1 [phi4]=GD_NOTHINK=1 [gemma4_12b]=GD_NOTHINK=1)
for t in "$@"; do
  [ -s $T4/jev/eval_jev_on_$t.json ] && { echo "$t done already"; continue; }
  echo "=== $t start $(date)"
  srun --overlap --jobid=$JOB -N1 -n1 --cpus-per-task=8 bash /datasets/omni_pretraining/wt_tgb/jev/tgb_lane.sh $GPU $t /datasets/omni_pretraining/gta2/models/${M[$t]} all ${NT[$t]:-} </dev/null >> $T4/jev/queue_$t.log 2>&1
  echo "=== $t end $(date) $( [ -s $T4/jev/eval_jev_on_$t.json ] && echo ok || echo MISSING)"
done
