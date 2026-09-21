"""Jev tool judgments on BFCL v4 multi_turn_base under the primary-class fifth protocol
(4 API classes x 10 build / 40 test records, mt/class_fifth_split.json).  Each record's menu is the one the executor is
served: its involved classes padded to 30 tools with menu.py's seed-0 padding.  Deployment = documentation demotion:
the top-K*_c tools keep full docs, the rest are compressed (MT_ARM=tier), K*_c from kfit_<key>.json.
  A  per record: user turns + tool docs -> noul per menu tool                         (online, 1 call per record)
  B  per class: A's call on the class's 10 build records, mean over records where the tool is on the menu (build once)
  C  per class: the executor's build trace under the full menu (calls, outputs, replies) -> noul per CALLED tool;
     tools never called in the class have no score and sink below every scored tool (menu.py rule, as for other trace methods)
Writes scores files for MT_SCORES (id -> tool -> score); nothing reads gold answers.
    PYTHONPATH=<bfcl site>:<BFCL-Env>:<wt_tgb> python -m jev.bfcl_run --key qwen2.5-7b --out-dir <dir> [--conds A,B,C] [--limit N]
"""
import argparse, collections, concurrent.futures as cf, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev.client import JevClient, PRICE_PER_M_INPUT
from jev.scoring import PROMPTS, score_request, score_conversation, aggregate
R = "/datasets/omni_pretraining/bfcl/runs/mt"; DATA = "/datasets/omni_pretraining/bfcl/site/bfcl_eval/data"
CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
MAP = {"GorillaFileSystem": "gorilla_file_system.json", "MathAPI": "math_api.json", "MessageAPI": "message_api.json", "TwitterAPI": "posting_api.json",
       "TicketAPI": "ticket_api.json", "TradingBot": "trading_bot.json", "TravelAPI": "travel_booking.json", "VehicleControlAPI": "vehicle_control.json"}
OUT_CHARS = 400

def load_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]

def doc_text(d):
    desc = (d.get("description") or "").strip()
    m = re.match(r"This tool belongs to the (.+?) API", desc); tag = f"[{m.group(1)}] " if m else ""
    body = desc.split("Tool description:", 1)[1].strip() if "Tool description:" in desc else desc
    params = ", ".join(d.get("parameters", {}).get("properties", {}))
    return f"{tag}{body}" + (f" Parameters: {params}." if params else "")

def menus(recs, ids):
    from mtlots.menu import _pad
    docs = {c: load_jsonl(f"{DATA}/multi_turn_func_doc/{f}") for c, f in MAP.items()}
    out = {}
    for rid in ids:
        e = recs[rid]; menu = _pad(e, docs, e["involved_classes"], 30, random.Random(f"{rid}/0"))
        out[rid] = [(d["name"], doc_text(d)) for d in menu]
    return out

def user_turns(e): return [m["content"] for turn in e["question"] for m in turn if m.get("role") == "user"]

def conversation(row, names):
    conv, called = [], set()
    for t in row.get("inference_log", []):
        if not isinstance(t, dict): continue
        for m in t.get("begin_of_turn_query", []):
            if isinstance(m, dict) and m.get("role") == "user": conv.append({"role": "user", "content": str(m.get("content"))})
        for s in sorted((k for k in t if k.startswith("step_")), key=lambda k: int(k.split("_")[1])):
            for m in t[s]:
                if not isinstance(m, dict) or m.get("role") not in ("assistant", "tool"): continue
                c = str(m.get("content"))
                if m["role"] == "assistant": called |= set(CALL.findall(c)) & names
                conv.append({"role": m["role"], "content": c[:OUT_CHARS]})
    while len(json.dumps(conv)) > 90000: conv = conv[:len(conv) // 2] + conv[len(conv) // 2 + 2:]   # stay under the 32k-token state limit
    return conv, called

def zero_fill(per_record, menu_tools, observed):
    """task score = mean over ALL build records, a record where the tool is off the menu / not called counting 0;
    tools never scored in the class stay unscored (they sink, menu.py rule)"""
    out = {}
    for t in menu_tools:
        if any(p.get(t) is not None for p in per_record):
            out[t] = sum(p.get(t) or 0.0 for p in per_record) / len(per_record)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--key", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--conds", default="A,B,C"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="jev-latest"); ap.add_argument("--train-result", default=None)
    a = ap.parse_args(); conds = a.conds.split(","); os.makedirs(a.out_dir, exist_ok=True)
    sp = json.load(open(f"{R}/class_fifth_split.json")); classes = sorted(sp["train"])
    train = {c: sp["train"][c][: a.limit or None] for c in classes}; test = {c: sp["test"][c][: a.limit or None] for c in classes}
    K = json.load(open(f"{R}/kfit/kfit_{a.key}.json"))["K_star"]
    recs = {r["id"]: r for r in load_jsonl(f"{DATA}/BFCL_v4_multi_turn_base.json")}
    ids = [r for c in classes for r in train[c] + test[c]]; M = menus(recs, ids)
    cl = JevClient(model=a.model, cache=os.path.join(a.out_dir, "jev_cache.jsonl"))
    calls, per, spaces = [], {}, {}
    def req(rid):
        turns = "\n".join(f"[turn {i + 1}] {u}" for i, u in enumerate(user_turns(recs[rid])))
        s, r = score_request(cl, turns, M[rid], tag=rid); calls.append((rid, r)); return rid, s
    fit_ids = [r for c in classes for r in train[c]]; test_ids = [r for c in classes for r in test[c]]
    todo = (fit_ids if "B" in conds else []) + (test_ids if "A" in conds else [])
    with cf.ThreadPoolExecutor(a.workers) as ex: rq = dict(ex.map(req, todo))
    if "A" in conds:
        json.dump({r: rq[r] for r in test_ids}, open(f"{a.out_dir}/jevA_scores_{a.key}.json", "w")); per["A"] = {r: rq[r] for r in test_ids}
    if "B" in conds:
        mt = {c: sorted({t for r in train[c] for t, _ in M[r]}) for c in classes}
        spaces["B"] = {c: zero_fill([rq[r] for r in train[c]], mt[c], None) for c in classes}
        spaces["Bpm"] = {c: aggregate([rq[r] for r in train[c]], mt[c]) for c in classes}
        for n in ("B", "Bpm"): json.dump({r: spaces[n][c] for c in classes for r in test[c]}, open(f"{a.out_dir}/jev{n}_scores_{a.key}.json", "w"))
        per["B_fit"] = {r: rq[r] for r in fit_ids}
    if "C" in conds:
        tr = a.train_result or f"{R}/search/KF_{a.key}_train_full_s0/" + os.listdir(f"{R}/search/KF_{a.key}_train_full_s0")[0] + "/multi_turn/BFCL_v4_multi_turn_base_result.json"
        rows = {x["id"]: x for x in load_jsonl(tr)}; conv_meta = {}
        def trc(rid):
            conv, called = conversation(rows[rid], {t for t, _ in M[rid]})
            s, r = score_conversation(cl, conv, sorted(called), tag=rid)
            if r: calls.append((rid, r))
            conv_meta[rid] = dict(n_msgs=len(conv), called=sorted(called)); return rid, s
        with cf.ThreadPoolExecutor(a.workers) as ex: tc = dict(ex.map(trc, fit_ids))
        mt = {c: sorted({t for r in train[c] for t, _ in M[r]}) for c in classes}
        spaces["C"] = {c: zero_fill([tc[r] for r in train[c]], mt[c], None) for c in classes}
        spaces["Cpm"] = {c: aggregate([tc[r] for r in train[c]], mt[c]) for c in classes}
        for n in ("C", "Cpm"): json.dump({r: spaces[n][c] for c in classes for r in test[c]}, open(f"{a.out_dir}/jev{n}_scores_{a.key}.json", "w"))
        per["C_fit"] = tc; per["C_meta"] = conv_meta; per["C_trace_file"] = tr
    acc = collections.defaultdict(collections.Counter)
    for rid, r in calls:
        ph = "build" if rid in set(fit_ids) else "online"; u = r.get("usage", {})
        acc[ph].update(calls=1, input_tokens=u.get("input_tokens", 0), cached=int(r["cached"])); acc[ph]["latency_ms"] += int(1000 * r.get("latency_s", 0))
    cost = {ph: dict(v, usd=v["input_tokens"] * PRICE_PER_M_INPUT / 1e6) for ph, v in acc.items()}
    top = {cnd: {c: sorted(v[c], key=lambda t: -v[c][t])[: K[c]] for c in classes} for cnd, v in spaces.items()}
    json.dump(dict(config=dict(vars(a), K_star=K, prompts=PROMPTS, mock=cl.mock, menu_rule="involved classes padded to 30, menu.py seed 0", aggregation="B/C: zero-fill mean over all build records (primary, fixed before any test run); Bpm/Cpm: mean over records where scored (pooling-rule ablation)",
                               finished=time.strftime("%F %T")), jev_model_returned=sorted({r.get("model") for _, r in calls}), cost=cost,
                   spaces=spaces, top_K=top, per_record=per, menus={r: [t for t, _ in M[r]] for r in ids}),
              open(f"{a.out_dir}/bfcl_jev_run_{a.key}.json", "w"), indent=1)
    print(json.dumps(dict(models=sorted({r.get("model") for _, r in calls}), cost=cost, top_K=top), indent=1))

if __name__ == "__main__": main()
