"""Jev tool judgments on GTA task types (t1 split: inject_opt/{type}_split.json), executor Qwen2.5-7B.
Candidates = the 7-tool pool of gta_type_baselines.py (+ GoogleSearch when GTA_USE_API=1, i.e. web_fact), described by toolmeta.json;
budget K = the Tool2Vec/Dense budget (results/tool2vec/K_t1.json).  Jev sees request text only (no image), like the router and BM25.
  A  per test request: question + tool descriptions -> noul per tool -> top-K
  B  per type: A's call on the type's train requests, mean -> top-K, frozen
  C  per type: the executor's full-menu train trajectories (tool outputs in call order + final answer) -> "does the answer rely on T's
     output" for the tools that were called; mean over train requests (not called = 0); tools never called sink below, canonical order
Writes results/jev/menus_{A,B,C}_{type}.json (test id -> tool list) for gta_type_baselines.py (GTA_EXTRA_MENUS).  No reference chain is read.
    python -m jev.gta_run --types read_arith,count_arith          (GTA_USE_API unset)
    GTA_USE_API=1 python -m jev.gta_run --types web_fact
"""
import argparse, ast, collections, concurrent.futures as cf, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev.client import JevClient, PRICE_PER_M_INPUT
from jev.scoring import PROMPTS, score_request, score_trace, top_k
BIG = "/datasets/omni_pretraining/gta2"; OUT = f"{BIG}/results/inject_opt"; DSP = f"{BIG}/data/gta_dataset"
DS = json.load(open(f"{DSP}/dataset.json")); META = json.load(open(f"{DSP}/toolmeta.json"))
CAND = ["OCR", "ImageDescription", "Calculator", "CountGivenObject", "RegionAttributeDescription", "TextToBbox", "Solver"]
if os.environ.get("GTA_USE_API") == "1": CAND = CAND + ["GoogleSearch"]
FULL_PRED = f"{BIG}/results/p_q7bphi_full/20260828_172124/predictions/q7bphi/gta_bench_end.json"      # 7B full-menu run, search disabled
WF_TRAIN = f"{OUT}/trainfull_7b_web_fact_api.json"                                                      # 7B full-menu train run, search enabled
OUT_CHARS = 600

def desc(t): m = META[t]; return (m.get("description") or "") if isinstance(m, dict) else str(m)
def question(k):
    for m in DS[str(k)]["dialogs"]:
        if m.get("role") == "user":
            c = m.get("content"); return c if isinstance(c, str) else " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return ""

def trace(v):
    """(outputs [(tool, text)], final answer) from one prediction record"""
    msgs = (v.get("prediction") or [[]])[0] or []; queue, outs, ans = [], [], ""
    for m in msgs:
        tc = m.get("tool_calls")
        if isinstance(tc, str):
            try: tc = ast.literal_eval(tc)
            except Exception: tc = []
        if tc: queue += [c.get("function", {}).get("name") for c in tc]; continue
        if m.get("role") == "tool":
            outs.append((queue.pop(0) if queue else "?", str(m.get("content"))[:OUT_CHARS])); continue
        if m.get("role") == "assistant" and m.get("content") and "thought" not in str(m.get("meta", "")): ans = m["content"]
    return [(t, o) for t, o in outs if t in CAND], ans

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--types", required=True); ap.add_argument("--out-dir", default=f"{BIG}/results/jev")
    ap.add_argument("--model", default="jev-latest"); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); K = json.load(open(f"{BIG}/results/tool2vec/K_t1.json"))
    cl = JevClient(model=a.model, cache=f"{a.out_dir}/jev_cache.jsonl"); tools = [(t, desc(t)) for t in CAND]
    log = json.load(open(f"{a.out_dir}/gta_jev_run_7b.json")) if os.path.exists(f"{a.out_dir}/gta_jev_run_7b.json") else {}
    for ty in a.types.split(","):
        sp = json.load(open(f"{OUT}/{ty}_split.json")); train, test = [str(x) for x in sp["train"]], [str(x) for x in sp["test"]]; k = int(K[ty])
        calls = []
        def req(i):
            s, r = score_request(cl, question(i), tools, tag=i); calls.append(("build" if i in train else "online", r)); return i, s
        with cf.ThreadPoolExecutor(a.workers) as ex: rq = dict(ex.map(req, train + test))
        if ty == "web_fact":
            w = json.load(open(WF_TRAIN)); p = json.load(open(w["pred"])); pred = {str(i): p[str(j)] for j, i in enumerate(w["ids"]) if str(j) in p}
        else: pred = json.load(open(FULL_PRED))
        def trc(i):
            outs, ans = trace(pred.get(i, {}))
            if not outs: return i, {}, 0
            s, r = score_trace(cl, question(i), outs, ans, tools, tag=i); calls.append(("build", r)); return i, s, len(outs)
        with cf.ThreadPoolExecutor(a.workers) as ex: tc = {i: (s, n) for i, s, n in ex.map(trc, train)}
        mA = {i: top_k(rq[i], k, CAND) for i in test}
        B = {t: sum(rq[i][t] for i in train) / len(train) for t in CAND}; mB = top_k(B, k, CAND)
        called = {t for i in train for t in tc[i][0]}
        C = {t: sum(tc[i][0].get(t) or 0.0 for i in train) / len(train) for t in called}
        mC = top_k(C, k, CAND); mC = mC + [t for t in CAND if t not in mC][: k - len(mC)]
        for n, m in (("A", mA), ("B", {i: mB for i in test}), ("C", {i: mC for i in test})):
            json.dump(m, open(f"{a.out_dir}/menus_{n}_{ty}.json", "w"))
        cost = collections.defaultdict(collections.Counter)
        for ph, r in calls: cost[ph].update(calls=1, input_tokens=r.get("usage", {}).get("input_tokens", 0))
        log[ty] = dict(K=k, n_train=len(train), n_test=len(test), candidates=CAND, B_scores=B, C_scores=C, B_space=mB, C_space=mC,
                       C_traces_with_calls=sum(1 for i in train if tc[i][1]), A_menus=mA, per_request_A={i: rq[i] for i in test},
                       per_request_B={i: rq[i] for i in train}, per_request_C={i: tc[i][0] for i in train},
                       cost={ph: dict(v, usd=v["input_tokens"] * PRICE_PER_M_INPUT / 1e6) for ph, v in cost.items()},
                       models=sorted({r.get("model") for _, r in calls}), prompts=PROMPTS, trace_source=(WF_TRAIN if ty == "web_fact" else FULL_PRED),
                       api=os.environ.get("GTA_USE_API") == "1", finished=time.strftime("%F %T"))
        print(ty, "K", k, "B", mB, "C", mC, "C traces with calls", log[ty]["C_traces_with_calls"], "/", len(train), dict(log[ty]["cost"]), flush=True)
        json.dump(log, open(f"{a.out_dir}/gta_jev_run_7b.json", "w"), indent=1)

if __name__ == "__main__": main()
