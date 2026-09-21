#!/usr/bin/env bash
# GTA comparator arms on web_fact / count_arith under the per-type fifth protocol (gta_type_baselines.py), derived from gta_kfit_lane.sh.
# Original header: GTA description ablation for one model: arms full / own(mask) / nodesc (every description
# emptied) / demote (descriptions kept only for the LOTS read_arith menu), same server, same day.
# Self-contained: starts its own tool server (GPU TGPU, port TPORT), proxy (PPORT) and model
# server (GPU LGPU, port LPORT) inside job JOB.
#   MODEL TAG LGPU TGPU LPORT TPORT PPORT JOB [NOTHINK=1] [MENU_TAG=7b] bash gta_phi_lane.sh
set -uo pipefail
API=${API:-0}; [ "$API" = 1 ] && APIX="GTA_USE_API=1 " || APIX=""
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
unset SLURM_JOB_ID SLURM_JOBID SLURM_STEP_ID SLURM_NODELIST SLURM_JOB_NODELIST SLURM_NNODES SLURM_NTASKS SLURM_PROCID SLURM_LOCALID SLURM_CPUS_PER_TASK SLURM_JOB_CPUS_PER_NODE
BIG=/datasets/omni_pretraining/gta2; X=$BIG/scripts_extra; S=/project/6101776/xzhan576/gta2-envlab/scripts
: "${MODEL:?}" "${TAG:?}" "${LGPU:?}" "${TGPU:?}" "${LPORT:?}" "${TPORT:?}" "${PPORT:?}" "${JOB:?}"; MENU_TAG=${MENU_TAG:-7b}
LOG=$BIG/results/gta_typebase${SPLIT:+_$SPLIT}_${TAG}_$(date +%m%d_%H%M).log; exec >>"$LOG" 2>&1
echo "=== gta_phi $TAG start $(date) lgpu=$LGPU tgpu=$TGPU ports $LPORT/$TPORT/$PPORT job=$JOB"
probe(){ timeout 60 srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 bash -c "curl -s -m 5 -o /dev/null -w '%{http_code}' $1" 2>/dev/null; }
S_T=gphi_tool_$TAG; S_P=gphi_proxy_$TAG; S_L=gphi_llm_$TAG
if [ "$(probe http://127.0.0.1:$TPORT/docs)" != "200" ]; then
  tmux kill-session -t $S_T 2>/dev/null
  tmux new-session -d -s $S_T "export SLURM_CONF=$SLURM_CONF; srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 --cpus-per-task=4 bash -c '${APIX}GTA_TOOL_GPU=$TGPU GTA_TOOL_PORT=$TPORT bash $S/start_toolserver.sh' 2>&1 | tee -a $BIG/results/gphi_tool_$TAG.log"
fi
if [ "$(probe http://127.0.0.1:$PPORT/proxy_config)" != "200" ]; then
  tmux kill-session -t $S_P 2>/dev/null
  tmux new-session -d -s $S_P "export SLURM_CONF=$SLURM_CONF; srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 --cpus-per-task=2 bash -c '${APIX}GTA_TOOL_PORT=$TPORT GTA_PROXY_PORT=$PPORT GTA_PROXY_LOG=$BIG/results/proxy_calls_gphi_$TAG.jsonl bash $S/start_proxy.sh' 2>&1 | tee -a $BIG/results/gphi_proxy_$TAG.log"
fi
if [ "$(probe http://127.0.0.1:$LPORT/v1/models)" != "200" ]; then
  tmux kill-session -t $S_L 2>/dev/null
  tmux new-session -d -s $S_L "export SLURM_CONF=$SLURM_CONF; srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 --cpus-per-task=4 bash -c 'MODEL=$MODEL NAME=$TAG PORT=$LPORT GPUS=$LGPU TP=1 ${NOTHINK:+NOTHINK=1} bash /datasets/omni_pretraining/gta2/scripts_extra/serve_new_util.sh' 2>&1 | tee -a $BIG/results/gphi_llm_$TAG.log"
fi
for i in $(seq 1 90); do [ "$(probe http://127.0.0.1:$TPORT/docs)" = "200" ] && [ "$(probe http://127.0.0.1:$PPORT/proxy_config)" = "200" ] && [ "$(probe http://127.0.0.1:$LPORT/v1/models)" = "200" ] && break; sleep 20; done
[ "$(probe http://127.0.0.1:$TPORT/docs)" = "200" ] && [ "$(probe http://127.0.0.1:$PPORT/proxy_config)" = "200" ] && [ "$(probe http://127.0.0.1:$LPORT/v1/models)" = "200" ] || { echo "$TAG: services never came up (tool $(probe http://127.0.0.1:$TPORT/docs) proxy $(probe http://127.0.0.1:$PPORT/proxy_config) llm $(probe http://127.0.0.1:$LPORT/v1/models))"; exit 1; }
echo "=== $TAG services up $(date +%H:%M)"
timeout 86400 srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 --cpus-per-task=4 bash -c "
source $S/common_env.sh; conda activate \$GTA_BIG/envs/opencompass
export ${APIX}GTA_EXTRA_MENUS=${GTA_EXTRA_MENUS:-} GTA_TRAJ_SFX=${GTA_TRAJ_SFX:-} ${GTA_GEMMA_ADAPTER:+GTA_GEMMA_ADAPTER=1} GTA_OBS_ROLE=${GTA_OBS_ROLE:-system} GTA_MODEL_NAME=$TAG GTA_LLM_URL=http://127.0.0.1:$LPORT/v1/chat/completions GTA_TOOLSERVER=http://127.0.0.1:$PPORT GTA_EVAL_MODES=end GTA_TOOLMETA=\$GTA_BIG/data/gta_dataset/toolmeta.json GTA_TEMP=0 GTA_ROUTER_PORT=$LPORT
cd $X && python3 gta_type_baselines.py --tag ${MENU_TAG} --types ${TYPES:-web_fact,count_arith} ${SPLIT:+--split $SPLIT} ${ARMS:+--arms $ARMS} ${FRESH:+--fresh $FRESH} ${EXTRA_ARGS:-}"
rc=$?
for s in $S_L $S_T $S_P; do tmux kill-session -t $s 2>/dev/null; done
echo "=== gta_typebase $TAG done rc=$rc $(date)"
exit $rc
