#!/usr/bin/env bash
# Jev comparators on the BFCL primary-class fifth protocol, one executor, one server session:
# full (same-session control) + tier demotion with Jev scores (A per record; B/C class spaces, zero-fill; Bpm/Cpm pooling-rule ablation),
# K*_c from kfit_<KEY>.json, 3 repeats (arm_once_persist pins the padding seed to 0, so repeats measure run-to-run variation).
# Results go to mt/jev/eval_<KEY>.json; kfit_<KEY>.json is read only.
#   KEY MODEL NAME GPU PORT JOB [UTIL=0.55] bash BFCL_JEV_LANE.sh
set -uo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
unset SLURM_JOB_ID SLURM_JOBID SLURM_STEP_ID SLURM_NODELIST SLURM_JOB_NODELIST SLURM_NNODES SLURM_NTASKS SLURM_PROCID SLURM_LOCALID SLURM_CPUS_PER_TASK SLURM_JOB_CPUS_PER_NODE
B=/datasets/omni_pretraining/bfcl/runs; ENV=/project/aip-xli135/xzhan576/BFCL-Env
PY=/datasets/omni_pretraining/envs/vllm_new/bin/python; SITE=/datasets/omni_pretraining/bfcl/site_py312
: "${KEY:?}" "${MODEL:?}" "${NAME:?}" "${GPU:?}" "${PORT:?}" "${JOB:?}"
LOG=$B/jev_${KEY}_$(date +%m%d_%H%M).log; exec >>"$LOG" 2>&1
echo "=== jev $KEY start $(date) gpu=$GPU port=$PORT job=$JOB"
probe(){ timeout 60 srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 bash -c "curl -sf -m 5 localhost:$PORT/v1/models" 2>/dev/null </dev/null | grep -q "\"$NAME\""; }
S=jvs_$(echo $KEY | tr . _)_$PORT
if probe; then echo "server up, reusing"; else
  tmux kill-session -t $S 2>/dev/null
  tmux new-session -d -s $S "export SLURM_CONF=$SLURM_CONF; srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 --cpus-per-task=4 bash -c 'MODEL=$MODEL NAME=$NAME PORT=$PORT GPUS=$GPU TP=1 ${NOTHINK:+NOTHINK=1} MAXLEN=${MAXLEN:-40960} bash /datasets/omni_pretraining/gta2/scripts_extra/serve_new.sh' 2>&1 | tee -a $B/jev_serve_$KEY.log"
  for i in $(seq 1 90); do probe && break; sleep 20; done
fi
probe || { echo "server never came up"; exit 1; }
sed -e "s/--jobid=\${MT_JOBID:-[0-9]*}/--jobid=\${MT_JOBID:-$JOB}/g" -e "s|^PY=.*|PY=$PY|" -e "s|/datasets/omni_pretraining/bfcl/site:|$SITE:|g" $B/arm_once_persist.sh > $B/arm_jev_${KEY}.sh; chmod +x $B/arm_jev_${KEY}.sh
cd $ENV && PYTHONPATH=$SITE:$ENV KEY=$KEY NAME=$NAME PORT=$PORT JOB=$JOB ARM=$B/arm_jev_${KEY}.sh $PY - <<'ZPY'
import json, os
from mtlots import task_search
R="/datasets/omni_pretraining/bfcl/runs/mt"; W=f"{R}/jev"; J=f"{W}/full"
KEY,NAME,PORT,JOB,ARM=(os.environ[k] for k in ("KEY","NAME","PORT","JOB","ARM"))
class A: work=W; arm_script=ARM
kf=json.load(open(f"{R}/kfit/kfit_{KEY}.json")); classes=sorted(kf["train"]); test=kf["test"]; Kstar=kf["K_star"]
te_ids=[r for c in classes for r in test[c]]; full_raw=json.load(open(f"{R}/lots_all_scores.json"))
out=f"{W}/eval_{KEY}.json"; res=json.load(open(out)) if os.path.exists(out) else {"model":KEY,"K_star":Kstar,"test":test,"runs":{}}
def save(): json.dump(res,open(out,"w"),indent=1)
def failed(tag):
    p=f"{R}/search/{tag}/score/{NAME}/multi_turn/BFCL_v4_multi_turn_base_score.json"
    if not os.path.exists(p): return None
    rows=[json.loads(l) for l in open(p) if l.strip()]; return sorted({r["id"] for r in rows[1:] if "id" in r and not r.get("valid",False)})
def env(seed,**kw): e={"MT_MODEL":KEY,"MT_PORT":PORT,"MT_JOBID":JOB,"MT_SEED":str(seed),"MT_CAT":"multi_turn_base"}; e.update(kw); return e
for rep in (0,1,2):
    for arm in ("full","jevA","jevB","jevC","jevBpm","jevCpm"):
        k=f"{arm}_r{rep}"
        if k in res["runs"] and res["runs"][k].get("acc") is not None: continue
        if arm=="full":
            tag=f"JV_{KEY}_test_full_r{rep}"; a=task_search.run_arm({r:sorted(full_raw[r]) for r in te_ids}, te_ids, tag, A, extra_env=env(rep))
            res["runs"][k]={"acc":a,"failed":failed(tag)}
        else:
            per,fl={},[]
            for c in classes:
                tag=f"JV_{KEY}_test_{c}_{arm}_r{rep}"
                per[c]=task_search.run_arm(None, test[c], tag, A, extra_env=env(rep,MT_ARM="tier",MT_SCORES=f"{J}/{arm}_scores_{KEY}.json",MT_TIER_K=str(Kstar[c]),MT_COMPRESS="1"))
                fl+=failed(tag) or []
            ok=[v for v in per.values() if v is not None]
            res["runs"][k]={"acc":(sum(per[c]*len(test[c]) for c in classes)/len(te_ids) if len(ok)==len(classes) else None),"by_class":per,"failed":fl}
        save(); print(f"[{KEY}] {k}: {res['runs'][k]['acc']}",flush=True)
print("=== jev",KEY,"summary:",{k:v["acc"] for k,v in res["runs"].items()},flush=True)
ZPY
tmux kill-session -t $S 2>/dev/null; echo "=== jev $KEY done $(date)"
