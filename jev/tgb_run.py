"""Jev tool selection on TGB (15-tool FULL_MENU_V4, per family first 80 fit / last 320 evaluation).
  A  request router      each evaluation request: question + tool docs -> per-tool noul -> top-K_f          (online, per request)
  B  task space          same call on the family's 80 fitting requests, family mean -> top-K_f, frozen       (build once)
  C  output-aware space  fitting trace (question, shown tool outputs, frozen answer y) -> "does y rely on tool" noul,
                         family mean over requests where the tool's output was shown -> top-K_f, frozen          (build once)
K_f is the LOTS budget of the executor (tgb_kfit_<tag>.json, fitted on fitting requests only) = budget-matched.
C: tools never shown in any fitting trace of the family have no evidence; they rank after every observed tool in canonical
order (the Trace-judge rule).  gt_tools/gold are never read here.
    JEV_MOCK=1 python -m jev.tgb_run --dir <tgb4> --tag 7b --conds A,B,C --limit-fit 3 --limit-eval 3 --out-dir <dir>
"""
import argparse, collections, concurrent.futures as cf, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev.client import JevClient, PRICE_PER_M_INPUT
from jev.scoring import PROMPTS, score_request, score_trace, top_k, aggregate

def tgb_menu_docs():
    from tgb import tools_v2, tools_v4; from tgb.coalitions_v4 import FULL_MENU_V4; from tgb.tool2vec_masks import tool_docs
    tools_v2.register(); tools_v4.register(); d = tool_docs()          # same first-docstring-line text as Tool2Vec/Dense/router
    return list(FULL_MENU_V4), [(t, d.get(t, "")) for t in FULL_MENU_V4]

def pmap(fn, items, workers):
    with cf.ThreadPoolExecutor(workers) as ex: return list(ex.map(fn, items))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--conds", default="A,B,C")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--model", default="jev-latest"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=600); ap.add_argument("--limit-fit", type=int, default=80); ap.add_argument("--limit-eval", type=int, default=320)
    ap.add_argument("--traces", default=None, help="fit_traces_<tag>.json (condition C)")
    a = ap.parse_args(); conds = a.conds.split(","); os.makedirs(a.out_dir, exist_ok=True)
    order, tools = tgb_menu_docs()
    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json"))); fam = collections.OrderedDict()
    for t in tasks: fam.setdefault(t["family"], []).append(t)
    fams = sorted(fam); K = {f: int(v) for f, v in json.load(open(os.path.join(a.dir, f"tgb_kfit_{a.tag}.json")))["K_star"].items()}
    fit = {f: fam[f][:80][:a.limit_fit] for f in fams}; ev = {f: fam[f][80:][:a.limit_eval] for f in fams}
    cl = JevClient(model=a.model, cache=os.path.join(a.out_dir, "jev_cache.jsonl"), rpm=a.rpm)
    calls = []; out = {"config": dict(vars(a), jev_model_requested=a.model, tool_order=order, tool_docs=dict(tools), K_star=K, prompts=PROMPTS,
                                      mock=cl.mock, started=time.strftime("%F %T"), price_per_M_input=PRICE_PER_M_INPUT)}
    def req(t):
        s, r = score_request(cl, t["question"], tools, tag=t["id"]); calls.append((t["id"], r)); return t["id"], s
    rq = {}
    if "A" in conds or "B" in conds:
        todo = [t for f in fams for t in (fit[f] if "B" in conds else []) + (ev[f] if "A" in conds else [])]
        rq = dict(pmap(req, todo, a.workers))
    masks, per_req, spaces = {}, {}, {}
    if "A" in conds:
        masks["A"] = {t["id"]: top_k(rq[t["id"]], K[t["family"]], order) for f in fams for t in ev[f]}
        per_req["A"] = {t["id"]: rq[t["id"]] for f in fams for t in ev[f]}
    if "B" in conds:
        agg = {f: aggregate([rq[t["id"]] for t in fit[f]], order) for f in fams}
        spaces["B"] = {f: dict(space=top_k(agg[f], K[f], order), mean_score=agg[f]) for f in fams}
        masks["B"] = {t["id"]: spaces["B"][f]["space"] for f in fams for t in ev[f]}
        per_req["B_fit"] = {t["id"]: rq[t["id"]] for f in fams for t in fit[f]}
    if "C" in conds:
        tr = {r["id"]: r for r in json.load(open(a.traces))["traces"]}
        def trc(t):
            r = tr[t["id"]]; s, c = score_trace(cl, t["question"], [tuple(o) for o in r["outputs"]], r["answer"], tools, tag=t["id"])
            calls.append((t["id"], c)); return t["id"], s
        tc = dict(pmap(trc, [t for f in fams for t in fit[f]], a.workers)); spaces["C"] = {}
        for f in fams:
            agg = aggregate([tc[t["id"]] for t in fit[f]], order); obs = top_k(agg, K[f], order)
            fill = [t for t in order if t not in agg][: K[f] - len(obs)]
            spaces["C"][f] = dict(space=obs + fill, mean_score=agg, n_observed=len(agg), filled_unobserved=fill,
                                  support={t: sum(t in tc[x["id"]] for x in fit[f]) for t in agg})
        masks["C"] = {t["id"]: spaces["C"][f]["space"] for f in fams for t in ev[f]}
        per_req["C_fit"] = tc
    for c, m in masks.items(): json.dump(m, open(os.path.join(a.out_dir, f"tgb_jev{c}_masks_{a.tag}.json"), "w"))
    # cost: phase = build (fit requests) vs online (eval requests); cached calls count once, at their original cost
    ids_fit = {t["id"] for f in fams for t in fit[f]}; acc = collections.defaultdict(lambda: collections.Counter())
    for tid, r in calls:
        ph = "build" if tid in ids_fit else "online"; u = r.get("usage", {})
        acc[ph].update(calls=1, input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0), cached=int(r["cached"]))
        acc[ph]["latency_ms"] += int(1000 * r.get("latency_s", 0))
    cost = {ph: dict(v, usd=v["input_tokens"] * PRICE_PER_M_INPUT / 1e6, mean_latency_s=v["latency_ms"] / 1000 / max(1, v["calls"] - v["cached"])) for ph, v in acc.items()}
    models = sorted({r.get("model") for _, r in calls})
    out.update(jev_model_returned=models, cost=cost, spaces=spaces, per_request_scores=per_req, finished=time.strftime("%F %T"))
    json.dump(out, open(os.path.join(a.out_dir, f"tgb_jev_run_{a.tag}.json"), "w"), indent=1)
    print(json.dumps(dict(models=models, cost=cost, spaces={c: {f: s["space"] for f, s in v.items()} for c, v in spaces.items()}), indent=1))

if __name__ == "__main__": main()
