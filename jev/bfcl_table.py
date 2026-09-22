"""BFCL comparison table: baselines (eval_base_<key>.json + eval_base_router_<key>.json) and Jev (eval_<key>.json), each 3 repeats.
Recall = share of the reference calls' tools that keep full docs (gold read here only); tokens = executor input tokens per record (repeat 0).
    python -m jev.bfcl_table --key qwen2.5-7b
"""
import argparse, glob, json, re, statistics as st
R = "/datasets/omni_pretraining/bfcl/runs/mt"; DATA = "/datasets/omni_pretraining/bfcl/site/bfcl_eval/data"; CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
ap = argparse.ArgumentParser(); ap.add_argument("--key", required=True); a = ap.parse_args(); K_ = a.key
base = json.load(open(f"{R}/jev/eval_base_{K_}.json")); jev = json.load(open(f"{R}/jev/eval_{K_}.json")); rt = json.load(open(f"{R}/jev/eval_base_router_{K_}.json"))
menus = json.load(open(f"{R}/jev/full/bfcl_jev_run_{K_}.json"))["menus"]; K = base["K_star"]; test = base["test"]; cls = {r: c for c in test for r in test[c]}
gold = {}
for l in open(f"{DATA}/possible_answer/BFCL_v4_multi_turn_base.json"):
    x = json.loads(l); gold[x["id"]] = {n for t in x["ground_truth"] for c in t for n in CALL.findall(c)}
def order(sc, rid):
    floor = min([v for v in sc.values() if v is not None], default=0.0) - 1.0
    return sorted(menus[rid], key=lambda t: (-(sc.get(t) if sc.get(t) is not None else floor), t))
def recall(fs): return st.mean(len(gold[r] & fs[r]) / len(gold[r]) for r in cls)
def tokens(pat):
    v = []
    for f in glob.glob(f"{R}/search/{pat}/*/multi_turn/BFCL_v4_multi_turn_base_result.json"):
        for l in open(f):
            try: x = json.loads(l)
            except Exception: continue
            flat = lambda z: sum(flat(y) for y in z) if isinstance(z, list) else (z or 0); v.append(flat(x.get("input_token_count")))
    return st.mean(v) if v else None
rows = []
def add(name, accs, fs, tok):
    ok = [x for x in accs if x is not None]
    rows.append(dict(method=name, acc=st.mean(ok) if ok else None, reps=accs, recall=(recall(fs) if fs else 1.0), tokens=tok))
add("Full documentation (anchor, baseline session)", [base["runs"].get(f"full_r{i}",{}).get("acc") for i in range(3)], None, tokens(f"JB_{K_}_test_full_r0"))
KF = f"{R}/kfit"; W = f"{R}/jev"
src = {"Random": [f"{KF}/kf_rand_{K_}_{{c}}_s0.json"], "BM25": [f"{KF}/kf_bm25_{K_}.json"], "Dense": [f"{W}/base_dense_{K_}.json"],
       "Usage Frequency": [f"{W}/base_usage_{K_}.json"], "Tool2Vec": [f"{W}/base_tool2vec_{K_}.json"]}
arm = {"Random": "random", "BM25": "bm25", "Dense": "dense", "Usage Frequency": "usage", "Tool2Vec": "tool2vec"}
for n, (p,) in src.items():
    sc = {}
    for c in test:
        s = json.load(open(p.replace("{c}", c)))
        for r in test[c]: sc[r] = s.get(r, {})
    fs = {r: set(order(sc[r], r)[: K[cls[r]]]) for r in cls}
    add(n, [base["runs"].get(f"{arm[n]}_r{i}",{}).get("acc") for i in range(3)], fs, tokens(f"JB_{K_}_test_*_{arm[n]}_r0"))
rm = json.load(open(f"{W}/base_router_mask_{K_}.json"))
add("LLM as router", [rt[f"router_r{i}"]["acc"] for i in range(3)], {r: set(rm[r]) for r in cls}, tokens(f"JB_{K_}_test_routerfix_r0"))
add("Full documentation (anchor, Jev session)", [jev["runs"][f"full_r{i}"]["acc"] for i in range(3)], None, tokens(f"JV_{K_}_test_full_r0"))
for n, t in (("Jev-A", "jevA"), ("Jev-B", "jevB"), ("Jev-C", "jevC")):
    s = json.load(open(f"{W}/full/{t}_scores_{K_}.json")); fs = {r: set(order(s.get(r, {}), r)[: K[cls[r]]]) for r in cls}
    add(n, [jev["runs"][f"{t}_r{i}"]["acc"] for i in range(3)], fs, tokens(f"JV_{K_}_test_*_{t}_r0"))
json.dump(rows, open(f"{W}/bfcl_table_{K_}.json", "w"), indent=1)
for r in rows: print(f"{r['method']:45s} {r['acc'] or 0:6.2f} {str([x if x is None else round(x,2) for x in r['reps']]):22s} recall {r['recall']:.2f} tok {r['tokens'] or 0:8.0f}")
