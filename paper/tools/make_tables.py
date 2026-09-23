"""Generate paper/tabs/{tgb,bfcl,gta}.tex from result files: one table per benchmark, one column per executor, rows = methods.
Missing results print as --.  Sources: TGB tgb4/jev/eval_jev_on_<tag>.json; BFCL mt/jev/eval_<key>.json (+eval_base_<key>.json, router
from eval_base_router_<key>.json for 7B); GTA inject_opt/typebase1_<tag>_jva.json.   python make_tables.py <out dir>"""
import json, os, statistics as st, sys, glob
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
T4 = "/datasets/omni_pretraining/gta2/results/taco/tgb4"; R = "/datasets/omni_pretraining/bfcl/runs/mt"; IO = "/datasets/omni_pretraining/gta2/results/inject_opt"
def ld(p): return json.load(open(p)) if os.path.exists(p) else None
def fmt(v, best=False): return "--" if v is None else (f"\\textbf{{{v:.1f}}}" if best else f"{v:.1f}")
def table(rows, cols, cells, caption, label, colhead, extra_hdr=""):
    L = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\fitwidth{",
         "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule", extra_hdr + "Method & " + " & ".join(colhead) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if r == "MIDRULE": L.append(r"\midrule"); continue
        vals = [cells.get((r, c)) for c in cols]
        L.append(r + " & " + " & ".join(fmt(v, v is not None and v == max([x for x in [cells.get((rr, c)) for rr in rows if rr != "MIDRULE"] if x is not None], default=None) and r not in ("Keep all", "Full documentation")) for v, c in zip(vals, cols)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]; return "\n".join(L) + "\n"
# ---- TGB
tgb_models = [("7b", "Qwen2.5-7B"), ("14b", "Qwen2.5-14B"), ("llama8b", "Llama3.1-8B"), ("mistral7b", "Mistral-7B"), ("q35_9b", "Qwen3.5-9B"), ("q3_8b", "Qwen3-8B"), ("phi4", "phi-4"), ("gemma4_12b", "Gemma4-12B")]
arm = {"Keep all": "full", "Random": "random", "Dense": "dense", "Tool2Vec": "tool2vec", "Trace judge": "tracejudge", "Jev-A (per request)": "jevA", "Jev-B (per task)": "jevB", "Jev-C (per task, traces)": "jevC"}
rows = ["Keep all", "Random", "Dense", "Tool2Vec", "Trace judge", "MIDRULE", "Jev-A (per request)", "Jev-B (per task)", "Jev-C (per task, traces)"]
tasks = ld(f"{T4}/tgb_tasks.json"); fam = {t["id"]: t["family"] for t in tasks}; fams = sorted(set(fam.values()))
cells = {}
for tag, name in tgb_models:
    e = ld(f"{T4}/jev/eval_jev_on_{tag}.json")
    if not e or "heldout" not in e: continue
    for r in rows:
        if r == "MIDRULE" or arm[r] not in e["heldout"]: continue
        per = e["heldout"][arm[r]]["per"]; cells[(r, tag)] = 100 * st.mean(st.mean(per[i] for i in per if fam[i] == f) for f in fams)
open(f"{OUT}/tgb.tex", "w").write(table(rows, [m[0] for m in tgb_models], cells,
    "TGB: accuracy (\\%) averaged over the ten families on their 320 evaluation requests each, one column per executor. All methods in a column keep the same number of tools per family (the budget fixed from build accuracy) and were evaluated in one session. Best reduced space in bold.",
    "tab:tgb", [m[1] for m in tgb_models]))
# ---- BFCL
bf_models = [("qwen2.5-7b", "Qwen2.5-7B"), ("qwen3-4b", "Qwen3-4B"), ("qwen3.5-4b", "Qwen3.5-4B"), ("qwen3.5-9b", "Qwen3.5-9B"), ("gemma-4-12b", "Gemma4-12B"), ("phi-4-mini", "Phi-4-mini")]
rows = ["Full documentation", "Random", "BM25", "Dense", "Usage Frequency", "Tool2Vec", "LLM as router", "MIDRULE", "Jev-A (per request)", "Jev-B (per task)", "Jev-C (per task, traces)"]
barm = {"Random": "random", "BM25": "bm25", "Dense": "dense", "Usage Frequency": "usage", "Tool2Vec": "tool2vec", "LLM as router": "router"}
jarm = {"Jev-A (per request)": "jevA", "Jev-B (per task)": "jevB", "Jev-C (per task, traces)": "jevC"}
cells = {}
PART = set()
def mean3(runs, key, tagc=None):
    v = [runs[f"{key}_r{i}"]["acc"] for i in range(3) if f"{key}_r{i}" in runs and runs[f"{key}_r{i}"].get("acc") is not None]
    if v and len(v) < 3 and tagc: PART.add(tagc)
    return st.mean(v) if v else None
for key, name in bf_models:
    j = ld(f"{R}/jev/eval_{key}.json"); b = ld(f"{R}/jev/eval_base_{key}.json"); rt = ld(f"{R}/jev/eval_base_router_{key}.json")
    if j: cells[("Full documentation", key)] = mean3(j["runs"], "full", key)
    for r, a in jarm.items():
        if j: cells[(r, key)] = mean3(j["runs"], a, key)
    for r, a in barm.items():
        if b: cells[(r, key)] = mean3(b["runs"], a, key)
    if rt and cells.get(("LLM as router", key)) is None: cells[("LLM as router", key)] = mean3(rt, "router", key)
open(f"{OUT}/bfcl.tex", "w").write(table(rows, [m[0] for m in bf_models], cells,
    "BFCL multi-turn: accuracy (\\%) on the 160 test records, mean of three runs, one column per executor. Every method keeps full documentation for the same $K$ tools per API class (the router documents the tools it names per record); baselines and Jev were run in two sessions per executor, each with its own full-documentation run (the two agree within one record). Best reduced space in bold. $^\\dagger$: fewer than three runs completed so far.",
    "tab:bfcl", [m[1] + ("$^\\dagger$" if m[0] in PART else "") for m in bf_models]))
# ---- GTA: one pooled accuracy per executor over the 47 test requests (32 read_arith + 15 count_arith)
gta_models = [("3b", "Qwen2.5-3B"), ("7b", "Qwen2.5-7B"), ("q3_8b", "Qwen3-8B"), ("14b", "Qwen2.5-14B"), ("q35_9b", "Qwen3.5-9B"), ("phi4", "phi-4"), ("gemma4_12b", "Gemma4-12B")]
rows = ["Keep all", "No tools", "Random", "BM25", "Dense", "Tool2Vec", "LLM as router", "TTO", "Beam Search", "MIDRULE", "Jev-A (per request)", "Jev-B (per task)", "Jev-C (per task, traces)"]
garm = {"Keep all": "full", "No tools": "notools", "Random": "random", "BM25": "bm25", "Dense": "dense", "Tool2Vec": "tool2vec", "LLM as router": "router", "TTO": "tto", "Beam Search": "beam", "Jev-A (per request)": "jevA", "Jev-B (per task)": "jevB", "Jev-C (per task, traces)": "jevC"}
N = {"read_arith": 32, "count_arith": 15}; cells = {}
for tag, name in gta_models:
    d = ld(f"{IO}/typebase1_{tag}_jva.json") or {}
    dc = ld(f"{IO}/typebase1_{tag}_jvac.json") or {}
    if "count_arith" not in d and "count_arith" in dc: d["count_arith"] = dc["count_arith"]
    if not all(ty in d for ty in N): continue
    for r in rows:
        if r == "MIDRULE" or not all(garm[r] in d[ty]["test_acc"] for ty in N): continue
        cells[(r, tag)] = 100 * sum(d[ty]["test_acc"][garm[r]] / 100 * N[ty] for ty in N) / sum(N.values())
open(f"{OUT}/gta.tex", "w").write(table(rows, [m[0] for m in gta_models], cells,
    "GTA: accuracy (\\%) over the 47 test requests of \\task{read_arith} (32) and \\task{count_arith} (15) pooled (one request is 2.1 points), two-tool spaces, all methods of a column in one session. Best reduced space in bold.",
    "tab:gta", [m[1] for m in gta_models]))
print("wrote", OUT)
