#!/usr/bin/env bash
# Comparators for the Jev BFCL table, one executor, one server session, 3 repeats, same K*_c and tier demotion as the Jev arms:
#   full (anchor) | random (kf_rand_<KEY>_<class>_s<r>) | bm25 (kf_bm25_<KEY>) | dense (BGE, class train queries) |
#   usage (call counts in the class's full-menu build traces) | tool2vec (e5 mean of 20 synthetic queries per tool written by the served model,
#   class mean similarity over build records) | router (served model names tools per test record; named tools keep full docs, MT_ARM=tierset)
# kfit_<KEY>.json is read only; results -> mt/jev/eval_base_<KEY>.json.   KEY MODEL NAME GPU PORT JOB [UTIL=0.55] bash BFCL_JEVBASE_LANE.sh
set -uo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
unset SLURM_JOB_ID SLURM_JOBID SLURM_STEP_ID SLURM_NODELIST SLURM_JOB_NODELIST SLURM_NNODES SLURM_NTASKS SLURM_PROCID SLURM_LOCALID SLURM_CPUS_PER_TASK SLURM_JOB_CPUS_PER_NODE
B=/datasets/omni_pretraining/bfcl/runs; ENV=/project/aip-xli135/xzhan576/BFCL-Env
PY=/datasets/omni_pretraining/envs/vllm_new/bin/python; SITE=/datasets/omni_pretraining/bfcl/site_py312
: "${KEY:?}" "${MODEL:?}" "${NAME:?}" "${GPU:?}" "${PORT:?}" "${JOB:?}"
LOG=$B/jevbase_${KEY}_$(date +%m%d_%H%M).log; exec >>"$LOG" 2>&1
echo "=== jevbase $KEY start $(date) gpu=$GPU port=$PORT job=$JOB"
probe(){ timeout 60 srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 bash -c "curl -sf -m 5 localhost:$PORT/v1/models" 2>/dev/null </dev/null | grep -q "\"$NAME\""; }
S=jbs_$(echo $KEY | tr . _)_$PORT
if probe; then echo "server up, reusing"; else
  tmux kill-session -t $S 2>/dev/null
  tmux new-session -d -s $S "export SLURM_CONF=$SLURM_CONF; srun --overlap --jobid=$JOB --nodes=1 --ntasks=1 --cpus-per-task=4 bash -c 'MODEL=$MODEL NAME=$NAME PORT=$PORT GPUS=$GPU TP=1 ${NOTHINK:+NOTHINK=1} MAXLEN=${MAXLEN:-40960} bash /datasets/omni_pretraining/gta2/scripts_extra/serve_new.sh' 2>&1 | tee -a $B/jevbase_serve_$KEY.log"
  for i in $(seq 1 90); do probe && break; sleep 20; done
fi
probe || { echo "server never came up"; exit 1; }
sed -e "s/--jobid=\${MT_JOBID:-[0-9]*}/--jobid=\${MT_JOBID:-$JOB}/g" -e "s|^PY=.*|PY=$PY|" -e "s|/datasets/omni_pretraining/bfcl/site:|$SITE:|g" $B/arm_once_persist.sh > $B/arm_jb_${KEY}.sh; chmod +x $B/arm_jb_${KEY}.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
NODE=$(squeue -j $JOB -h -o %N)
cd $ENV && PYTHONPATH=$SITE:$ENV NODE=$NODE KEY=$KEY NAME=$NAME PORT=$PORT JOB=$JOB ARM=$B/arm_jb_${KEY}.sh SITE=$SITE PY=$PY $PY - <<'ZPY'
import json, os, re, glob, collections, subprocess, urllib.request, random, statistics as st
import numpy as np
from mtlots import task_search
R="/datasets/omni_pretraining/bfcl/runs/mt"; W=f"{R}/jev"; KF=f"{R}/kfit"; D="/datasets/omni_pretraining/bfcl/site/bfcl_eval/data"
KEY,NAME,PORT,JOB,ARM,SITE,PY,NODE=(os.environ[k] for k in ("KEY","NAME","PORT","JOB","ARM","SITE","PY","NODE"))
class A: work=W; arm_script=ARM
kf=json.load(open(f"{KF}/kfit_{KEY}.json")); classes=sorted(kf["train"]); train=kf["train"]; test=kf["test"]; Kstar=kf["K_star"]; pooled=kf["pooled"]
te_ids=[r for c in classes for r in test[c]]; full_raw=json.load(open(f"{R}/lots_all_scores.json"))
out=f"{W}/eval_base_{KEY}.json"; res=json.load(open(out)) if os.path.exists(out) else {"model":KEY,"K_star":Kstar,"test":test,"runs":{},"spaces":{}}
def save(): json.dump(res,open(out,"w"),indent=1)
docs={}
for f in glob.glob(f"{D}/multi_turn_func_doc/*.json"):
    for l in open(f):
        if l.strip(): d=json.loads(l); docs[d["name"]]=d
recs={}
for l in open(f"{D}/BFCL_v4_multi_turn_base.json"):
    if l.strip(): r=json.loads(l); recs[r["id"]]=r
def users(rid): return [m["content"] for turn in recs[rid]["question"] for m in turn if m.get("role")=="user"]
def doctext(t):
    d=docs[t]; ps=d.get("parameters",{}).get("properties",{})
    return " ".join([re.sub(r"(?<!^)(?=[A-Z])"," ",t).replace("_"," "), d.get("description",""), " ".join(f"{k} {v.get('description','')}" for k,v in ps.items())])
def per_class_file(name, sc_by_class):
    p=f"{W}/base_{name}_{KEY}.json"; json.dump({r:sc_by_class[c] for c in classes for r in test[c]},open(p,"w"))
    res["spaces"][name]={c:sorted(sc_by_class[c],key=lambda t:(-sc_by_class[c][t],t))[:Kstar[c]] for c in classes}; save(); return p
import torch; torch.set_num_threads(1); torch.set_num_interop_threads(1)
from transformers import AutoTokenizer, AutoModel
def encoder(path, pool):
    tk=AutoTokenizer.from_pretrained(path); m=AutoModel.from_pretrained(path).eval()
    def emb(texts, prefix=""):
        out=[]
        with torch.no_grad():
            for i in range(0,len(texts),32):
                b=tk([prefix+x for x in texts[i:i+32]],padding=True,truncation=True,max_length=512,return_tensors="pt"); h=m(**b).last_hidden_state
                v=h[:,0] if pool=="cls" else (h*b["attention_mask"].unsqueeze(-1)).sum(1)/b["attention_mask"].sum(1,keepdim=True)
                out.append(torch.nn.functional.normalize(v,dim=-1).numpy())
        return np.concatenate(out)
    return emb
files={}
# dense (BGE-base, as BFCL_DENSE_LANE.sh)
if not os.path.exists(f"{W}/base_dense_{KEY}.json"):
    emb=encoder("/datasets/omni_pretraining/gta2/models/bge-base-en-v1.5","cls"); sc={}
    for c in classes:
        tools=list(pooled[c]); tv=emb([doctext(t) for t in tools]); qv=emb([u for r in train[c] for u in users(r)],"Represent this sentence for searching relevant passages: ")
        m=(qv@tv.T).mean(0); sc[c]={t:float(m[i]) for i,t in enumerate(tools)}
    per_class_file("dense",sc)
files["dense"]=f"{W}/base_dense_{KEY}.json"
# usage frequency (as BFCL_USAGE_LANE.sh)
if not os.path.exists(f"{W}/base_usage_{KEY}.json"):
    rf=glob.glob(f"{R}/search/KF_{KEY}_train_full_s0/*/multi_turn/*result.json")[0]; rows={json.loads(l)["id"]:json.loads(l) for l in open(rf) if l.strip()}; sc={}
    for c in classes:
        cnt=collections.Counter(); cov=collections.Counter()
        for rid in train[c]:
            names=[n for n in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", json.dumps(rows[rid].get("result"))) if n in pooled[c]]; cnt.update(names); cov.update(set(names))
        sc[c]={t:cnt[t]+cov[t]/1000.0 for t in pooled[c]}
    per_class_file("usage",sc)
files["usage"]=f"{W}/base_usage_{KEY}.json"
# tool2vec: 20 synthetic queries per tool from the served model (tool2vec.py gen prompt, T=0.7, seed 0), e5-base-v2 mean pooling
if not os.path.exists(f"{W}/base_tool2vec_{KEY}.json"):
    qf=f"{W}/tool2vec_queries_{KEY}.json"; Q=json.load(open(qf)) if os.path.exists(qf) else {}
    alltools=sorted({t for c in classes for t in pooled[c]})
    for t in alltools:
        if t in Q: continue
        d=(docs[t].get("description") or "").split("Tool description:",1)[-1].strip(); msg=(f"You are helping build a tool retrieval dataset.\nTool: {t}: {d}\n\n"
             "Write 20 diverse, realistic user questions that would require calling this tool. Vary the wording, the numbers and the situation. One question per line, numbered 1 to 20, no other text.")
        body=json.dumps({"model":NAME,"temperature":0.7,"seed":0,"max_tokens":1200,"messages":[{"role":"user","content":msg}]}).encode()
        txt=json.loads(urllib.request.urlopen(urllib.request.Request(f"http://{NODE}:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"}),timeout=300).read())["choices"][0]["message"]["content"]
        Q[t]=[x for x in (re.sub(r"^\s*\d+[.)]\s*","",l).strip() for l in txt.split("\n")) if len(x)>10][:20]; json.dump(Q,open(qf,"w"))
    emb=encoder("/datasets/omni_pretraining/gta2/models/e5-base-v2","mean"); tv={t:emb(Q[t],"query: ").mean(0) for t in alltools if Q.get(t)}
    tv={t:v/np.linalg.norm(v) for t,v in tv.items()}; sc={}
    for c in classes:
        qv=emb([" ".join(users(r)) for r in train[c]],"query: ").mean(0)
        sc[c]={t:float(qv@tv[t]) for t in pooled[c] if t in tv}
    per_class_file("tool2vec",sc)
files["tool2vec"]=f"{W}/base_tool2vec_{KEY}.json"
# router: the served model names the tools each test record needs (mtlots.router_mask, pool 30, seed 0); named tools keep full docs
rmask=f"{W}/base_router_mask_{KEY}.json"
if not os.path.exists(rmask):
    idp=f"{W}/ids_router_{KEY}.json"; json.dump({"multi_turn_base":te_ids},open(idp,"w"))
    subprocess.run(["srun","--overlap",f"--jobid={JOB}","--nodes=1","--ntasks=1","--cpus-per-task=4","bash","-c",
        f"export PYTHONPATH={SITE}:/project/aip-xli135/xzhan576/BFCL-Env MT_MODEL={KEY} MT_PORT={PORT} HF_HOME=/datasets/omni_pretraining/gta2/hf_home HF_HUB_OFFLINE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; cd /project/aip-xli135/xzhan576/BFCL-Env && {PY} -m mtlots.router_mask --cat multi_turn_base --ids {idp} --seed 0 --out {rmask}"],stdin=subprocess.DEVNULL)
rm_=json.load(open(rmask)); json.dump({r:{t:1.0 for t in v} for r,v in rm_.items()},open(f"{W}/base_router_scores_{KEY}.json","w"))
res["router_kept"]=st.mean(len(v) for v in rm_.values()); save(); print(f"[{KEY}] router mean named {res['router_kept']:.1f}",flush=True)
def env(seed,**kw): e={"MT_MODEL":KEY,"MT_PORT":PORT,"MT_JOBID":JOB,"MT_SEED":str(seed),"MT_CAT":"multi_turn_base"}; e.update(kw); return e
def failed(tag):
    p=f"{R}/search/{tag}/score/{NAME}/multi_turn/BFCL_v4_multi_turn_base_score.json"
    if not os.path.exists(p): return None
    rows=[json.loads(l) for l in open(p) if l.strip()]; return sorted({r["id"] for r in rows[1:] if "id" in r and not r.get("valid",False)})
for rep in (0,1,2):
    for arm in ("full","random","bm25","dense","usage","tool2vec","router"):
        k=f"{arm}_r{rep}"
        if k in res["runs"] and res["runs"][k].get("acc") is not None: continue
        if arm=="full":
            tag=f"JB_{KEY}_test_full_r{rep}"; a=task_search.run_arm({r:sorted(full_raw[r]) for r in te_ids}, te_ids, tag, A, extra_env=env(rep)); res["runs"][k]={"acc":a,"failed":failed(tag)}
        elif arm=="router":
            tag=f"JB_{KEY}_test_router_r{rep}"; a=task_search.run_arm(None, te_ids, tag, A, extra_env=env(rep,MT_ARM="tierset",MT_TIER_SET=rmask,MT_SCORES=f"{W}/base_router_scores_{KEY}.json",MT_COMPRESS="1")); res["runs"][k]={"acc":a,"failed":failed(tag)}
        else:
            per,fl={},[]
            for c in classes:
                sf={"random":f"{KF}/kf_rand_{KEY}_{c}_s{rep}.json","bm25":f"{KF}/kf_bm25_{KEY}.json"}.get(arm) or files[arm]
                tag=f"JB_{KEY}_test_{c}_{arm}_r{rep}"
                per[c]=task_search.run_arm(None, test[c], tag, A, extra_env=env(rep,MT_ARM="tier",MT_SCORES=sf,MT_TIER_K=str(Kstar[c]),MT_COMPRESS="1")); fl+=failed(tag) or []
            ok=[v for v in per.values() if v is not None]
            res["runs"][k]={"acc":(sum(per[c]*len(test[c]) for c in classes)/len(te_ids) if len(ok)==len(classes) else None),"by_class":per,"failed":fl}
        save(); print(f"[{KEY}] {k}: {res['runs'][k]['acc']}",flush=True)
print("=== jevbase",KEY,"summary:",{k:v["acc"] for k,v in res["runs"].items()},flush=True)
ZPY
tmux kill-session -t $S 2>/dev/null; echo "=== jevbase $KEY done $(date)"
