"""Jev with its native rule: keep every tool whose probability exceeds a threshold (default 0.5), no budget K.
TGB: writes tgb4/jev/thr/tgb_jev{A,B,C}_thr<th>_masks_<tag>.json (per-request A; per-family B/C, evaluation requests only).
BFCL: writes mt/jev/thr/jev{B,C}_thr<th>_K_<key>.json (per-class K = number of tools above the threshold; 0 allowed) for the tier arm.
    python -m jev.threshold_masks --th 0.5 --tgb-tags 7b,14b,... --bfcl-keys qwen2.5-7b,...
"""
import argparse, json, os
T4 = "/datasets/omni_pretraining/gta2/results/taco/tgb4"; R = "/datasets/omni_pretraining/bfcl/runs/mt"
ap = argparse.ArgumentParser(); ap.add_argument("--th", type=float, default=0.5); ap.add_argument("--tgb-tags", default=""); ap.add_argument("--bfcl-keys", default=""); a = ap.parse_args()
th = a.th; sfx = f"thr{th:g}".replace(".", "p")
if a.tgb_tags:
    os.makedirs(f"{T4}/jev/thr", exist_ok=True); tasks = json.load(open(f"{T4}/tgb_tasks.json")); fam = {}
    for t in tasks: fam.setdefault(t["family"], []).append(t["id"])
    for tag in a.tgb_tags.split(","):
        r = json.load(open(f"{T4}/jev/full/tgb_jev_run_{tag}.json")); order = r["config"]["tool_order"]; sizes = {}
        A = r["per_request_scores"]["A"]; mA = {i: [t for t in order if (s.get(t) or 0) > th] for i, s in A.items()}
        json.dump(mA, open(f"{T4}/jev/thr/tgb_jevA_{sfx}_masks_{tag}.json", "w")); sizes["A"] = sum(map(len, mA.values())) / len(mA)
        for c in ("B", "C"):
            sp = {f: [t for t in order if (r["spaces"][c][f]["mean_score"].get(t) or 0) > th] for f in fam}
            json.dump({i: sp[f] for f in fam for i in fam[f][80:]}, open(f"{T4}/jev/thr/tgb_jev{c}_{sfx}_masks_{tag}.json", "w")); sizes[c] = {f: len(sp[f]) for f in fam}
        print("TGB", tag, sizes)
if a.bfcl_keys:
    os.makedirs(f"{R}/jev/thr", exist_ok=True)
    for key in a.bfcl_keys.split(","):
        b = json.load(open(f"{R}/jev/full/bfcl_jev_run_{key}.json"))
        for c in ("B", "C"):
            K = {cls: sum(v > th for v in b["spaces"][c][cls].values()) for cls in b["spaces"][c]}
            json.dump(K, open(f"{R}/jev/thr/jev{c}_{sfx}_K_{key}.json", "w")); print("BFCL", key, c, K)
