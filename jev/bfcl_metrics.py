"""BFCL summary for mt/jev/eval_<key>.json.  Gold tools (possible_answer ground-truth calls) are read here only, for evaluation.
Full-doc set of a record = the top-K*_c of its menu under the arm's scores, ordered exactly as mtlots/menu.py orders them.
  accuracy (mean of repeats, per class and overall), required-tool recall / complete-chain coverage of the full-doc set,
  executor input tokens per record (from BFCL result files)
    python -m jev.bfcl_metrics --key qwen2.5-7b
"""
import argparse, glob, json, os, re, statistics as st
R = "/datasets/omni_pretraining/bfcl/runs/mt"; DATA = "/datasets/omni_pretraining/bfcl/site/bfcl_eval/data"
CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--key", required=True); a = ap.parse_args()
    ev = json.load(open(f"{R}/jev/eval_{a.key}.json")); run = json.load(open(f"{R}/jev/full/bfcl_jev_run_{a.key}.json"))
    test, K = ev["test"], ev["K_star"]; classes = sorted(test); ids = [r for c in classes for r in test[c]]
    cls = {r: c for c in classes for r in test[c]}; menus = run["menus"]
    gold = {}
    for l in open(f"{DATA}/possible_answer/BFCL_v4_multi_turn_base.json"):
        x = json.loads(l); gold[x["id"]] = {n for t in x["ground_truth"] for c in t for n in CALL.findall(c)}
    arms = sorted({k.rsplit("_r", 1)[0] for k in ev["runs"]}, key=lambda s: ("full", "jevA", "jevB", "jevC", "jevBpm", "jevCpm").index(s))
    def fullset(arm, rid):
        if arm == "full": return set(menus[rid])
        sc = json.load(open(f"{R}/jev/full/{arm}_scores_{a.key}.json")).get(rid, {})
        floor = min([v for v in sc.values() if v is not None], default=0.0) - 1.0
        order = sorted(menus[rid], key=lambda t: (-(sc.get(t) if sc.get(t) is not None else floor), t))
        return set(order[: K[cls[rid]]])
    def tokens(arm, rep):
        pat = f"{R}/search/JV_{a.key}_test_full_r{rep}" if arm == "full" else f"{R}/search/JV_{a.key}_test_*_{arm}_r{rep}"
        v = []
        for d in glob.glob(pat):
            for f in glob.glob(f"{d}/*/multi_turn/BFCL_v4_multi_turn_base_result.json"):
                for l in open(f):
                    try: x = json.loads(l)
                    except Exception: continue
                    def flat(z): return sum(flat(y) for y in z) if isinstance(z, list) else (z or 0)
                    v.append(flat(x.get("input_token_count")))
        return st.mean(v) if v else None
    rows = {}
    for arm in arms:
        reps = [ev["runs"][k] for k in sorted(ev["runs"]) if k.rsplit("_r", 1)[0] == arm and ev["runs"][k].get("acc") is not None]
        if not reps: continue
        fs = {r: fullset(arm, r) for r in ids}
        rec = {c: st.mean(len(gold[r] & fs[r]) / len(gold[r]) for r in test[c]) for c in classes}
        cov = {c: st.mean(gold[r] <= fs[r] for r in test[c]) for c in classes}
        acc_c = {c: st.mean(100 * (1 - sum(r in set(x["failed"]) for r in test[c]) / len(test[c])) for x in reps) for c in classes}
        rows[arm] = dict(n_rep=len(reps), acc=st.mean(x["acc"] for x in reps), acc_reps=[x["acc"] for x in reps], acc_by_class=acc_c,
                         recall=st.mean(rec.values()), coverage=st.mean(cov.values()), recall_by_class=rec, coverage_by_class=cov,
                         full_docs=st.mean(len(fs[r]) for r in ids), input_tokens=tokens(arm, 0))
    json.dump(dict(key=a.key, K_star=K, rows=rows), open(f"{R}/jev/summary_{a.key}.json", "w"), indent=1)
    print(f"{'arm':8s} {'acc':>6s} {'reps':>20s} {'recall':>6s} {'cover':>6s} {'#full':>5s} {'in_tok':>8s}  " + " ".join(c[:8] for c in classes))
    for arm, r in rows.items():
        print(f"{arm:8s} {r['acc']:6.2f} {str([round(x,2) for x in r['acc_reps']]):>20s} {r['recall']:6.3f} {r['coverage']:6.3f} {r['full_docs']:5.1f} {r['input_tokens'] or 0:8.0f}  "
              + " ".join(f"{r['acc_by_class'][c]:8.1f}" for c in classes))

if __name__ == "__main__": main()
