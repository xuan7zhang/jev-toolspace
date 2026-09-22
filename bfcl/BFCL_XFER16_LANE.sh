#!/usr/bin/env bash
# Experiment 3: cross-model reuse with independent fitting/evaluation.  Target KEY serves the 160 test records (primary-class fifth split)
# under tier demotion with a FIXED K=16 for every class, using the pooled ranking of each SOURCE model (kf_scores_<src>.json, fitted on that
# model's 10 build records per class) -- including its own -- plus full documentation, 3 seeds.  Nothing is refit.
# Results -> mt/xfer16/xfer16_<KEY>.json.     KEY MODEL NAME GPUS PORT JOB [NOTHINK=1] [SOURCES=a,b,c] [K=16] bash BFCL_XFER16_LANE.sh
set -uo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
unset SLURM_JOB_ID SLURM_JOBID SLURM_STEP_ID SLURM_NODELIST SLURM_JOB_NODELIST SLURM_NNODES SLURM_NTASKS SLURM_PROCID SLURM_LOCALID SLURM_CPUS_PER_TASK SLURM_JOB_CPUS_PER_NODE
B=/datasets/omni_pretraining/bfcl/runs; ENV=/project/aip-xli135/xzhan576/BFCL-Env; X=/datasets/omni_pretraining/gta2/scripts_extra
PY=/datasets/omni_pretraining/envs/vllm_new/bin/python; SITE=/datasets/omni_pretraining/bfcl/site_py312
: "${KEY:?}" "${MODEL:?}" "${NAME:?}" "${GPUS:?}" "${PORT:?}" "${JOB:?}"; SOURCES=${SOURCES:-qwen3-4b,qwen3.5-9b,gemma-4-12b}; K=${K:-16}
W=$B/mt/xfer16; mkdir -p $W; LOG=$B/xfer16_${KEY}_$(date +%m%d_%H%M).log; exec >>"$LOG" 2>&1
echo "=== xfer16 $KEY start $(date) gpus=$GPUS port=$PORT job=$JOB K=$K sources=$SOURCES"
probe(){ timeout 60 srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 bash -c "curl -sf -m 5 localhost:$PORT/v1/models" 2>/dev/null </dev/null | grep -q "\"$NAME\""; }
S=xfs_$(echo $KEY | tr . _)_$PORT
if probe; then echo "server up, reusing"; else
  tmux kill-session -t $S 2>/dev/null
  tmux new-session -d -s $S "export SLURM_CONF=$SLURM_CONF; srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 --cpus-per-task=4 bash -c 'MODEL=$MODEL NAME=$NAME PORT=$PORT GPUS=$GPUS TP=${TP:-1} ${NOTHINK:+NOTHINK=1} MAXLEN=${MAXLEN:-40960} bash $X/serve_new.sh' 2>&1 | tee -a $B/xfer16_serve_$KEY.log"
  for i in $(seq 1 90); do probe && break; sleep 20; done
fi
probe || { echo "server never came up"; exit 1; }
sed -e "s/--jobid=\${MT_JOBID:-[0-9]*}/--jobid=\${MT_JOBID:-$JOB}/g" -e "s|^PY=.*|PY=$PY|" -e "s|/datasets/omni_pretraining/bfcl/site:|$SITE:|g" $B/arm_once_persist.sh > $B/arm_xf_${KEY}.sh; chmod +x $B/arm_xf_${KEY}.sh
cd $ENV && PYTHONPATH=$SITE:$ENV KEY=$KEY NAME=$NAME PORT=$PORT JOB=$JOB ARM=$B/arm_xf_${KEY}.sh SOURCES=$SOURCES K=$K $PY - <<'ZPY'
import json, os
from mtlots import task_search
R="/datasets/omni_pretraining/bfcl/runs/mt"; W=f"{R}/xfer16"
KEY,NAME,PORT,JOB,ARM,SRC,K=(os.environ[k] for k in ("KEY","NAME","PORT","JOB","ARM","SOURCES","K")); SRC=SRC.split(",")
class A: work=W; arm_script=ARM
sp=json.load(open(f"{R}/class_fifth_split.json")); classes=sorted(sp["test"]); test=sp["test"]; te=[r for c in classes for r in test[c]]
full_raw=json.load(open(f"{R}/lots_all_scores.json"))
out=f"{W}/xfer16_{KEY}.json"; res=json.load(open(out)) if os.path.exists(out) else {"target":KEY,"K":int(K),"protocol":"rank from source's 10 build records per class; deploy on the 40 test records per class; tier demotion, fixed K","runs":{}}
def save(): json.dump(res,open(out,"w"),indent=1)
def failed(tag):
    p=f"{R}/search/{tag}/score/{NAME}/multi_turn/BFCL_v4_multi_turn_base_score.json"
    if not os.path.exists(p): return None
    rows=[json.loads(l) for l in open(p) if l.strip()]; return sorted({r["id"] for r in rows[1:] if "id" in r and not r.get("valid",False)})
def env(seed,**kw): e={"MT_MODEL":KEY,"MT_PORT":PORT,"MT_JOBID":JOB,"MT_SEED":str(seed),"MT_CAT":"multi_turn_base"}; e.update(kw); return e
for seed in (0,1,2):
    k=f"full_s{seed}"
    if res["runs"].get(k,{}).get("acc") is None:
        tag=f"XF_{KEY}_full_s{seed}"; a=task_search.run_arm({r:sorted(full_raw[r]) for r in te}, te, tag, A, extra_env=env(seed)); res["runs"][k]={"acc":a,"failed":failed(tag)}; save(); print(f"[{KEY}] {k}: {a}",flush=True)
    for src in SRC:
        k=f"from_{src}_s{seed}"
        if res["runs"].get(k,{}).get("acc") is not None: continue
        per,fl={},[]
        for c in classes:
            tag=f"XF_{KEY}_from_{src}_{c}_K{K}_s{seed}"
            per[c]=task_search.run_arm(None, test[c], tag, A, extra_env=env(seed,MT_ARM="tier",MT_SCORES=f"{R}/kfit/kf_scores_{src}.json",MT_TIER_K=str(K),MT_COMPRESS="1")); fl+=failed(tag) or []
        ok=[v for v in per.values() if v is not None]
        res["runs"][k]={"acc":(sum(per[c]*len(test[c]) for c in classes)/len(te) if len(ok)==len(classes) else None),"by_class":per,"failed":fl}; save(); print(f"[{KEY}] {k}: {res['runs'][k]['acc']}",flush=True)
print("=== xfer16",KEY,"summary:",{k:v["acc"] for k,v in res["runs"].items()},flush=True)
ZPY
tmux kill-session -t $S 2>/dev/null; echo "=== xfer16 $KEY done $(date)"
