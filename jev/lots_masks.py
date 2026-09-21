"""Per-request mask files for the existing LOTS and Random spaces of tgb_kfit_<tag>.json (read only, nothing refit)."""
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out-dir", required=True); a = ap.parse_args()
k = json.load(open(os.path.join(a.dir, f"tgb_kfit_{a.tag}.json"))); fam = {}
for t in json.load(open(os.path.join(a.dir, "tgb_tasks.json"))): fam.setdefault(t["family"], []).append(t["id"])
for name, key in [("lots", "menu"), ("random", "random_menu")]:
    json.dump({i: k[key][f] for f in fam for i in fam[f][80:]}, open(os.path.join(a.out_dir, f"tgb_{name}_masks_{a.tag}.json"), "w"))
