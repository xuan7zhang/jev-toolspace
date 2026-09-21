"""All figures of the Jev TGB report, from tgb4/jev/{summary_7b.json, full/tgb_jev_run_7b.json} (no model calls)."""
import json, statistics as st, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
T4 = "/datasets/omni_pretraining/gta2/results/taco/tgb4"; OUT = sys.argv[1]
S = json.load(open(f"{T4}/jev/summary_7b.json")); R = json.load(open(f"{T4}/jev/full/tgb_jev_run_7b.json"))
T = {t["id"]: t for t in json.load(open(f"{T4}/tgb_tasks.json"))}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
NAME = {"full": "Keep all", "lots": "LOTS", "jevC": "Jev-C", "tracejudge": "Trace judge", "jevA": "Jev-A", "jevB": "Jev-B",
        "tool2vec": "Tool2Vec", "dense": "Dense", "random": "Random"}
ORDER = ["jevC", "full", "tracejudge", "jevA", "jevB", "tool2vec", "dense", "random"]
COL = {"lots": "#2F6F55", "jevC": "#6FA287", "jevA": "#D08C3A", "jevB": "#E6B36A", "full": "#8C8C8C", "tracejudge": "#7A8FB8",
       "tool2vec": "#B9B9B9", "dense": "#B9B9B9", "random": "#B9B9B9"}
rows = S["rows"]

# Fig 2: macro accuracy with paired-bootstrap CI of the difference to Keep all
fig, ax = plt.subplots(figsize=(5.4, 2.1))
acc = [100 * rows[m]["macro"]["correct"] for m in ORDER]
lo = [100 * (rows[m]["d_vs_full"]["mean"] - rows[m]["d_vs_full"]["lo"]) for m in ORDER]
hi = [100 * (rows[m]["d_vs_full"]["hi"] - rows[m]["d_vs_full"]["mean"]) for m in ORDER]
ax.bar(range(len(ORDER)), acc, color=[COL[m] for m in ORDER])
ax.axhline(100 * rows["full"]["macro"]["correct"], color="#555", lw=0.7, ls="--")
for i, a in enumerate(acc): ax.text(i, a + 2, f"{a:.1f}", ha="center", fontsize=7)
ax.set_xticks(range(len(ORDER)), [NAME[m] for m in ORDER], rotation=0, fontsize=7); ax.set_ylabel("Macro accuracy (%)"); ax.set_ylim(0, 95)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_main.pdf"); plt.close(fig)

# Fig 3: per-family accuracy heatmap
fams = S["families"]; M = ["full", "jevC", "jevA", "jevB", "tracejudge", "tool2vec", "dense", "random"]
A = np.array([[100 * rows[m]["per_family"][f]["correct"] for f in fams] for m in M])
fig, ax = plt.subplots(figsize=(6.2, 2.7)); im = ax.imshow(A, cmap="Greens", vmin=0, vmax=100, aspect="auto")
for i in range(len(M)):
    for j in range(len(fams)): ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if A[i, j] > 60 else "#222")
ax.set_xticks(range(len(fams)), [f.replace("_", "\\_") if False else f for f in fams], rotation=30, ha="right", fontsize=7)
ax.set_yticks(range(len(M)), [NAME[m] for m in M], fontsize=7); ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label("Accuracy (%)", fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_family.pdf"); plt.close(fig)

cases = [("fx_settle", "ExchangeRate", "GoogleSearch"), ("table_total", "TableQuery", "DocRetrieve"), ("doc_two_facts", "DocRetrieve", "TableQuery")]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for k, (f, g, w) in enumerate(cases):
    for off, t, c in [(-0.17, g, "#2F6F55"), (0.17, w, "#D08C3A")]:
        v = [s[t] for i, s in R["per_request_scores"]["A"].items() if T[i]["family"] == f]
        ax.bar(k + off, st.mean(v), 0.32, color=c)
        ax.text(k + off, 0.02, t, rotation=90, ha="center", va="bottom", fontsize=6, color="white")
ax.set_xticks(range(3), [c[0] for c in cases], fontsize=7); ax.set_ylim(0, 1); ax.set_ylabel("Jev-A probability")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_sources.pdf"); plt.close(fig)
print("ok")
