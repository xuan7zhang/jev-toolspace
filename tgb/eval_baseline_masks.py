"""Evaluate a frozen TGB mask and Keep-all in one model session.

The reported protocol uses the first 80 requests of every family for fitting
and the remaining 320 for evaluation.  Serving cost is the mean complete input
prompt over those 3,200 test requests, after applying the model's chat template.
"""
import argparse
import json
import os
import random
import statistics as st
import tempfile
import time
import zlib

from vllm import LLM, SamplingParams

from . import scenes, tools_v2, tools_v4
from .coalitions_v4 import FULL_MENU_V4
from .generate import TPL, context
from .score_dl import correct
from .tools import run_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", default="bm25")
    ap.add_argument("--fit-tokens", type=int, default=0)
    ap.add_argument("--fit-passes", type=int, default=1)
    ap.add_argument("--fit-seconds", type=float, default=0.0)
    ap.add_argument("--extra-mask", action="append", default=[],
                    help="NAME=PATH, evaluated in the same session as --mask and full")
    a = ap.parse_args()

    tools_v2.register()
    tools_v4.register()
    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    by_family = {}
    for task in tasks:
        by_family.setdefault(task["family"], []).append(task)
    test = [task for family in sorted(by_family)
            for task in by_family[family][80:]]
    if len(test) != 3200:
        raise RuntimeError(f"expected 3200 family-held-out tasks, got {len(test)}")

    mask = json.load(open(a.mask))
    extra = dict(spec.split("=", 1) for spec in a.extra_mask)
    extra = {name: json.load(open(path)) for name, path in extra.items()}
    for name, m in [(a.method, mask)] + list(extra.items()):
        missing = [task["id"] for task in test if task["id"] not in m]
        if missing:
            raise ValueError(f"mask {name} misses {len(missing)} held-out task ids")

    with tempfile.TemporaryDirectory(prefix="tgb_baseline_") as tmp:
        for task in test:
            task["boxes"] = scenes.render(task["scene"], os.path.join(tmp, "scene.png"))
        llm_kwargs = {}
        if "gemma-4" in a.model.lower():
            # Transformers 5 models Gemma-4 as a heterogeneous text config.
            # vLLM reads the global head_dim while building the engine, which
            # requires explicitly enabling that compatibility access.
            llm_kwargs["hf_overrides"] = {
                "text_config": {"allow_global_per_layer_attribute_access": True}
            }
        llm = LLM(model=a.model, tensor_parallel_size=1, dtype="bfloat16",
                  gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
                  enforce_eager=bool(os.environ.get("GD_EAGER")), max_model_len=4096,
                  trust_remote_code=True, **llm_kwargs)
        tok = llm.get_tokenizer()

        def wrap(prompt):
            if not os.environ.get("GD_NOTHINK"):
                return prompt
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            try:
                return tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    enable_thinking=False, **kwargs)
            except TypeError:
                return tok.apply_chat_template(
                    [{"role": "user", "content": prompt}], **kwargs)

        def run(label, menu_of):
            prompts, meta, sizes = [], [], []
            for task in test:
                menu = list(menu_of(task))
                rng = random.Random(zlib.crc32(
                    f"{task['id']}/{'+'.join(sorted(menu))}".encode()))
                outputs, _ = run_chain(task, task["scene"], task["boxes"],
                                       set(menu), rng, corrupt_tools=())
                raw = TPL.format(
                    q=task["question"],
                    o=context(outputs, [step["tool"] for step in task["plan"]]))
                prompt = wrap(raw)
                prompts.append(prompt)
                sizes.append(len(tok(prompt, add_special_tokens=False)["input_ids"]))
                meta.append((task["id"], task["gold"], task["family"], len(menu)))
            start = time.perf_counter()
            generated = llm.generate(
                prompts, SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "48")),
                                        temperature=0))
            elapsed = time.perf_counter() - start
            per = {tid: int(correct(out.outputs[0].text, gold))
                   for (tid, gold, _, _), out in zip(meta, generated)}
            per_family = {
                family: st.mean(per[tid] for tid, _, fam, _ in meta if fam == family)
                for family in sorted(by_family)}
            return dict(label=label, n=len(test), accuracy=st.mean(per.values()),
                        correct=sum(per.values()), prompt_tokens_per_request=st.mean(sizes),
                        tools_per_request=st.mean(n for _, _, _, n in meta),
                        wall_seconds=elapsed, per_family=per_family, per=per)

        arm = run(a.method, lambda task: mask[task["id"]])
        full = run("full", lambda task: FULL_MENU_V4)
        extra_arms = {name: run(name, lambda task, m=m: m[task["id"]])
                      for name, m in extra.items()}

    saved = full["prompt_tokens_per_request"] - arm["prompt_tokens_per_request"]
    result = {
        "benchmark": "TGB-v4", "protocol": "per-family first 80 fit / remaining 320 test",
        "model": a.tag, "method": a.method,
        "selection": {"prefill_tokens": a.fit_tokens, "passes": a.fit_passes,
                      "wall_seconds": a.fit_seconds},
        "heldout": {a.method: arm, "full": full, **extra_arms},
        "token_efficiency": {
            "serving_convention": "complete wrapped input prompt over all 3,200 test requests",
            "tokens_saved_per_request": saved,
            "break_even_requests": a.fit_tokens / saved if saved > 0 else None}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"model": a.tag, "method": a.method,
                      "accuracy": arm["accuracy"],
                      "prompt_tokens": arm["prompt_tokens_per_request"],
                      "full_accuracy": full["accuracy"],
                      "full_prompt_tokens": full["prompt_tokens_per_request"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
