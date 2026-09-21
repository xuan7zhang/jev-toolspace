"""Dump the fitting traces that the trace-based baselines read, for judges that run outside vLLM (Jev condition C).
Same path as trace_judge_masks: first 80 requests per family, full menu, run_chain with the same RNG seed, TPL + context,
greedy decoding with GD_GENTOK tokens and the same model settings.  Nothing here reads gold answers or gt_tools.
    python -m tgb.jev_fit_traces --dir <tgb4> --model <dir> --tag <tag>   -> <tgb4>/jev/fit_traces_<tag>.json
"""
import argparse, json, os, random, tempfile, zlib
from vllm import LLM, SamplingParams
from . import scenes, tools_v2, tools_v4
from .coalitions_v4 import FULL_MENU_V4
from .generate import TPL, context, CANON, CTX_CHARS
from .tools import run_chain

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    a = ap.parse_args(); tools_v2.register(); tools_v4.register()
    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json"))); fam = {}
    for t in tasks: fam.setdefault(t["family"], []).append(t)
    fit = [t for f in sorted(fam) for t in fam[f][:80]]
    kw = {"hf_overrides": {"text_config": {"allow_global_per_layer_attribute_access": True}}} if "gemma-4" in a.model.lower() else {}
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")), enforce_eager=bool(os.environ.get("GD_EAGER")),
              max_model_len=8192, trust_remote_code=True, **kw); tok = llm.get_tokenizer()
    def chat(p):
        if not os.environ.get("GD_NOTHINK"): return p
        k = dict(tokenize=False, add_generation_prompt=True)
        try: return tok.apply_chat_template([{"role": "user", "content": p}], enable_thinking=False, **k)
        except TypeError: return tok.apply_chat_template([{"role": "user", "content": p}], **k)
    rank = {t: i for i, t in enumerate(CANON)}; rec = []
    with tempfile.TemporaryDirectory(prefix="tgb_jevtr_") as tmp:
        for t in fit:
            boxes = scenes.render(t["scene"], os.path.join(tmp, "scene.png")); menu = list(FULL_MENU_V4)
            rng = random.Random(zlib.crc32(f"{t['id']}/{'+'.join(sorted(menu))}".encode()))
            outs, _ = run_chain(t, t["scene"], boxes, set(menu), rng, corrupt_tools=())
            shown = [(n, str(outs[n])[:CTX_CHARS]) for n in sorted(outs, key=lambda n: (rank.get(n, 99), n)) if outs.get(n)]
            rec.append(dict(id=t["id"], family=t["family"], question=t["question"], outputs=shown, context=context(outs, [s["tool"] for s in t["plan"]])))
    gen = llm.generate([chat(TPL.format(q=r["question"], o=r["context"])) for r in rec], SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "48")), temperature=0))
    for r, g in zip(rec, gen): r["answer"] = g.outputs[0].text.strip()
    os.makedirs(os.path.join(a.dir, "jev"), exist_ok=True)
    json.dump({"model": a.model, "tag": a.tag, "menu": list(FULL_MENU_V4), "gentok": int(os.environ.get("GD_GENTOK", "48")), "traces": rec},
              open(os.path.join(a.dir, "jev", f"fit_traces_{a.tag}.json"), "w"), indent=1)
    print(a.tag, "traces", len(rec))

if __name__ == "__main__": main()
