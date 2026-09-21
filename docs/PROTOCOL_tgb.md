# Jev (TypeSafe System One) on TGB — protocol

## API (verified 2026-09-21 against docs.typesafe.ai and a live call)
- `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer $JEV_API_KEY` (key in `~/.jev_api_env`, chmod 600).
- Request `model: jev-latest`; every response logs the resolved version (`jev-1.13.0` on 2026-09-21).
- One call per request: the state holds the request and the whole 15-tool catalog in `FULL_MENU_V4` order.
  It asks one `noul` question per tool. A noul is an independent probability in [0,1], so a tool's score does not compete with the others.
  No `choice` question is used.
- Price: $0.042 per 1M input tokens; output tokens are free.
- Limits: 1,200 requests/min and 64k tokens/request. The client throttles to 600 rpm.
- Retries: exponential backoff on 429, 529 and network errors. It fails fast on 401 and 422.
- Cache: `jev_cache.jsonl`, keyed on sha256(model, state, questions). A rerun resumes from the cache and costs nothing for cached calls.
- `JEV_MOCK=1` produces deterministic fake scores, marks each record with `mock: true` and model `MOCK`, and never touches the network.

## Tool text
- Same as Tool2Vec, Dense and the router: `tool2vec_masks.tool_docs()`, the first docstring line.
- ExchangeRate and Barcode have empty docstrings, so Jev sees only their names. This matches the existing baselines and is not patched.

## Conditions (executor Qwen2.5-7B-Instruct, split: first 80 fit / last 320 evaluation per family)

| | sees | calls | aggregation | kind |
|---|---|---|---|---|
| A request router | evaluation question + tool docs | 1 per evaluation request (3,200) | top-K_f of that request's scores | per-request, online |
| B task space | fitting questions + tool docs | 1 per fitting request (800) | family mean → top-K_f, frozen | build-once |
| C output-aware space | fitting question, shown tool outputs, frozen answer y | 1 per fitting request (800) | family mean over requests where the tool was shown → top-K_f, frozen | build-once |

- **K_f** is the LOTS budget in `tgb_kfit_7b.json`, which is fitted on fitting requests only. The comparison is therefore budget-matched, and no evaluation label tunes K.
- **Ties** are broken by canonical order.
- **Condition C traces:**
  - Traces come from `tgb.jev_fit_traces`. This is the same path as the Trace judge: full menu, the same `run_chain` RNG seed, TPL + context, greedy decoding, GD_GENTOK=48.
  - Under the full menu, `run_chain` also executes every tool outside the plan with generic arguments (`tools.py:266-284`), so each trace shows all 15 outputs. All 800 fitting traces have 15 outputs, and C scores every tool.
  - The unobserved-tool rule stays in place for BFCL and GTA, where traces cover only some tools. A tool that is never shown has no score and ranks after every observed tool in canonical order, which is the Trace-judge rule. Such tools are logged as `filled_unobserved`; on TGB this list is empty.
- **C question:** does the given answer rely on this tool's output, including through a downstream tool? The prompt says not to judge correctness and not to re-answer.
- **Not a LOTS intervention.** C is an external judge's reading of a trace. LOTS instead measures the change in the executor's own answer likelihood.
- **Gold labels:** gt_tools and gold answers are read only by `tgb_metrics.py`.

## Evaluation
- `tgb.eval_baseline_masks` (existing evaluator) gained a backward-compatible `--extra-mask NAME=PATH`.
- One vLLM session evaluates LOTS, Keep all, Random, Jev A/B/C, Trace judge, Dense and Tool2Vec on the same 3,200 requests. The per-request comparison is paired.
- Existing results files are not touched. The new outputs are in `tgb4/jev/`.

## Metrics (`jev.tgb_metrics`)
- **Accuracy:** macro average over families.
- **Selection quality:**
  - Required-tool recall.
  - Complete-chain coverage (G ⊆ S). For `no_tool` the gold set is empty, so these are undefined and excluded.
  - |S|, the number of tools selected.
- **Executor cost:** executor prompt tokens per request.
- **Uncertainty:** Δ vs Keep all and vs LOTS, with a family-stratified paired bootstrap (B=2000, seed 0). The bootstrap uses existing per-request outcomes only.
- **Cost accounting:**
  - Build cost: B and C calls, tokens and USD.
  - Online cost: A calls per request.
  - Measured latency: 8 threads at 600 rpm, cache misses only.
- **Failure buckets:**
  - Missed upstream tool, where `upstream` ⊄ S.
  - Generic tools selected outside G.
  - Full chain covered but wrong while Keep all is right.
    Under TGB, `run_chain` seeds the RNG by the served menu, so a different menu redraws the tool outputs. This bucket therefore partly measures output redraw, not selection.

## Commands
```bash
# mock smoke (no network, no GPU)
JEV_MOCK=1 python -m jev.tgb_run --dir $T4 --tag 7b --conds A,B,C --limit-fit 3 --limit-eval 3 --traces <mock traces> --out-dir <tmp>
# real API smoke (2 fitting requests per family, about $0.002)
source ~/.jev_api_env; python -m jev.tgb_run --dir $T4 --tag 7b --conds B,C --limit-fit 2 --limit-eval 0 --traces $T4/jev/fit_traces_7b.json --out-dir $T4/jev/smoke_real
# full: traces (GPU) -> Jev A,B,C (API, 4,800 calls, about 9M input tokens, about $0.38) -> same-session eval (GPU) -> tables
bash jev/tgb_lane.sh <GPU> 7b /datasets/omni_pretraining/gta2/models/Qwen2.5-7B-Instruct all
python -m jev.tgb_metrics --dir $T4 --tag 7b
```
`T4=/datasets/omni_pretraining/gta2/results/taco/tgb4`; run from `/datasets/omni_pretraining/wt_tgb`.

## Extending
- `jev/client.py` and `jev/scoring.py` do not depend on the benchmark.
- A BFCL or GTA driver only supplies (request, [(tool, doc)]) and traces of the form (question, [(tool, output)], answer).
