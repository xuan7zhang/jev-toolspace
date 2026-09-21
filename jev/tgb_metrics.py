"""Summary tables for one same-session evaluation (eval_jev_on_<tag>.json).  gt_tools are read here only, for evaluation.
  selection: required-tool recall |S∩G|/|G|, complete-chain coverage [G⊆S], |S|        (requests with empty G: recall/coverage undefined, excluded)
  downstream: accuracy per family and macro over families; paired stratified bootstrap of Δ macro-acc vs Keep all and vs LOTS
  failure buckets: missed upstream tool, generic tools selected outside G, full chain covered but wrong while Keep all right
    python -m jev.tgb_metrics --dir <tgb4> --tag 7b [--B 2000]
"""
import argparse, collections, json, os, random, statistics as st
GENERIC = ["Calculator", "GoogleSearch", "DocRetrieve", "TableQuery"]
MASKS = {"lots": "jev/tgb_lots_masks_{t}.json", "random": "jev/tgb_random_masks_{t}.json", "jevA": "jev/full/tgb_jevA_masks_{t}.json",
         "jevB": "jev/full/tgb_jevB_masks_{t}.json", "jevC": "jev/full/tgb_jevC_masks_{t}.json", "tracejudge": "tgb_tracejudge_task_masks_{t}.json", "tracejudge_task": "tgb_tracejudge_task_masks_{t}.json",
         "dense": "tgb_dense_base_task_masks_{t}.json", "tool2vec": "tgb_tool2vec_task_masks_{t}.json"}
KIND = {"full": "static", "lots": "build-once", "random": "build-once", "jevA": "per-request", "jevB": "build-once", "jevC": "build-once",
        "tracejudge": "build-once", "dense": "build-once", "tool2vec": "build-once"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--eval", default=None); a = ap.parse_args()
    ev = json.load(open(a.eval or os.path.join(a.dir, "jev", f"eval_jev_on_{a.tag}.json")))["heldout"]
    T = {t["id"]: t for t in json.load(open(os.path.join(a.dir, "tgb_tasks.json")))}
    from tgb.coalitions_v4 import FULL_MENU_V4
    masks = {m: json.load(open(os.path.join(a.dir, p.format(t=a.tag)))) for m, p in MASKS.items() if m in ev}
    masks["full"] = {i: list(FULL_MENU_V4) for i in ev["full"]["per"]}
    ids = sorted(ev["full"]["per"]); fams = sorted({T[i]["family"] for i in ids}); byf = {f: [i for i in ids if T[i]["family"] == f] for f in fams}
    rows, per_req = {}, {}
    for m in ev:
        S = masks[m]; c = ev[m]["per"]; G = {i: set(T[i]["gt_tools"]) for i in ids}
        rec = {i: dict(correct=c[i], size=len(S[i]), recall=(len(G[i] & set(S[i])) / len(G[i]) if G[i] else None),
                       cover=(int(G[i] <= set(S[i])) if G[i] else None)) for i in ids}
        per_req[m] = rec
        def fam_stat(f, k):
            v = [rec[i][k] for i in byf[f] if rec[i][k] is not None]; return st.mean(v) if v else None
        pf = {f: {k: fam_stat(f, k) for k in ("correct", "recall", "cover", "size")} for f in fams}
        mac = {k: st.mean([pf[f][k] for f in fams if pf[f][k] is not None]) for k in ("correct", "recall", "cover", "size")}
        rows[m] = dict(kind=KIND.get(m, "?"), macro=mac, micro_acc=ev[m]["accuracy"], prompt_tokens=ev[m]["prompt_tokens_per_request"], per_family=pf)
    rng = random.Random(0)
    def macro_acc(m, samp): return st.mean(st.mean(per_req[m][i]["correct"] for i in samp[f]) for f in fams)
    boots = collections.defaultdict(list)
    for _ in range(a.B):
        samp = {f: [rng.choice(byf[f]) for _ in byf[f]] for f in fams}
        base = {r: macro_acc(r, samp) for r in ("full", "lots") if r in per_req}
        for m in per_req:
            for r in base: boots[(m, r)].append(macro_acc(m, samp) - base[r])
    for (m, r), v in boots.items():
        v.sort(); rows[m][f"d_vs_{r}"] = dict(mean=st.mean(v), lo=v[int(.025 * len(v))], hi=v[int(.975 * len(v)) - 1])
    fail = {}
    for m in per_req:
        S = masks[m]; missed_up = gen_extra = cov_wrong = 0; n_up = 0; ex = collections.defaultdict(list)
        for i in ids:
            t = T[i]; G = set(t["gt_tools"]); up = set(t.get("upstream") or []) & G if isinstance(t.get("upstream"), list) else set()
            if not up and len(t["plan"]) > 1: up = {s["tool"] for s in t["plan"][:-1]} & G       # tools whose output feeds a later step
            if up:
                n_up += 1
                if not up <= set(S[i]): missed_up += 1; ex["missed_upstream"].append(i)
            if set(GENERIC) & (set(S[i]) - G): gen_extra += 1
            if G and G <= set(S[i]) and not per_req[m][i]["correct"] and per_req["full"][i]["correct"]: cov_wrong += 1; ex["covered_but_wrong"].append(i)
        fail[m] = dict(missed_upstream_rate=missed_up / max(1, n_up), n_with_upstream=n_up, generic_outside_gold_rate=gen_extra / len(ids),
                       covered_but_wrong_vs_full=cov_wrong, generic_selected_share={g: sum(g in S[i] for i in ids) / len(ids) for g in GENERIC},
                       examples={k: v[:5] for k, v in ex.items()})
    out = dict(tag=a.tag, n=len(ids), families=fams, rows=rows, failures=fail, bootstrap=dict(B=a.B, stratified_by="family", seed=0))
    json.dump(out, open(os.path.join(a.dir, "jev", f"summary_{a.tag}.json"), "w"), indent=1)
    print(f"{'method':11s} {'kind':11s} {'acc':>6s} {'Δfull [95%]':>22s} {'ΔLOTS [95%]':>22s} {'recall':>6s} {'cover':>6s} {'|S|':>5s} {'tok':>6s}")
    for m, r in rows.items():
        d = lambda k: f"{100*r[k]['mean']:+6.2f} [{100*r[k]['lo']:+.1f},{100*r[k]['hi']:+.1f}]" if k in r else ""
        mc = r["macro"]; print(f"{m:11s} {r['kind']:11s} {100*mc['correct']:6.2f} {d('d_vs_full'):>22s} {d('d_vs_lots'):>22s} {mc['recall']:6.3f} {mc['cover']:6.3f} {mc['size']:5.1f} {r['prompt_tokens']:6.0f}")

if __name__ == "__main__": main()
