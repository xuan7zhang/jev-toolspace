#!/usr/bin/env bash
# router arm for the Jev BFCL table against the SAME server as BFCL_JEVBASE_LANE_v2.sh (port 8953): tierset with the router's named tools
# (MT_TIER_SET) and a matching MT_SCORES (named tools first), 3 repeats; merged into eval_base_<KEY>.json as router_r*.
set -uo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
unset SLURM_JOB_ID SLURM_JOBID SLURM_STEP_ID SLURM_NODELIST SLURM_JOB_NODELIST SLURM_NNODES SLURM_NTASKS SLURM_PROCID SLURM_LOCALID SLURM_CPUS_PER_TASK SLURM_JOB_CPUS_PER_NODE
B=/datasets/omni_pretraining/bfcl/runs; ENV=/project/aip-xli135/xzhan576/BFCL-Env; PY=/datasets/omni_pretraining/envs/vllm_new/bin/python; SITE=/datasets/omni_pretraining/bfcl/site_py312
KEY=qwen2.5-7b NAME=Qwen2.5-7B-Instruct PORT=8953 JOB=5478555; exec >>$B/jevbase_router_fix.log 2>&1
cd $ENV && PYTHONPATH=$SITE:$ENV KEY=$KEY NAME=$NAME PORT=$PORT JOB=$JOB ARM=$B/arm_jb_${KEY}.sh $PY - <<'ZPY'
import json, os
from mtlots import task_search
R="/datasets/omni_pretraining/bfcl/runs/mt"; W=f"{R}/jev"
KEY,NAME,PORT,JOB,ARM=(os.environ[k] for k in ("KEY","NAME","PORT","JOB","ARM"))
class A: work=W; arm_script=ARM
kf=json.load(open(f"{R}/kfit/kfit_{KEY}.json")); te=[r for c in sorted(kf["test"]) for r in kf["test"][c]]
out=f"{W}/eval_base_router_{KEY}.json"; res=json.load(open(out)) if os.path.exists(out) else {}
for rep in (0,1,2):
    k=f"router_r{rep}"
    if res.get(k,{}).get("acc") is not None: continue
    tag=f"JB_{KEY}_test_routerfix_r{rep}"
    a=task_search.run_arm(None, te, tag, A, extra_env={"MT_MODEL":KEY,"MT_PORT":PORT,"MT_JOBID":JOB,"MT_SEED":str(rep),"MT_CAT":"multi_turn_base","MT_ARM":"tierset",
        "MT_TIER_SET":f"{W}/base_router_mask_{KEY}.json","MT_SCORES":f"{W}/base_router_scores_{KEY}.json","MT_COMPRESS":"1"})
    p=f"{R}/search/{tag}/score/{NAME}/multi_turn/BFCL_v4_multi_turn_base_score.json"
    fl=sorted({json.loads(l)["id"] for l in list(open(p))[1:] if l.strip() and not json.loads(l).get("valid",False)}) if os.path.exists(p) else None
    res[k]={"acc":a,"failed":fl}; json.dump(res,open(out,"w"),indent=1); print(k,a,flush=True)
ZPY
