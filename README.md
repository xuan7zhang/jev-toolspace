# jev-toolspace

Evaluating Jev (TypeSafe System One, `jev-1.13.0`) as a relevance judge for building task-level tool spaces.

Jev answers yes/no (`noul`) questions about a shared state and returns an independent probability for each one. One API call scores every tool in a menu with one question per tool, and a tool's score does not compete with the others.

| Setting | What Jev reads | When it runs |
|---|---|---|
| A (per request) | the request and the tool descriptions | every request, online |
| B (per task) | the same judgment on a task's build requests, averaged into one space per task | once per task |
| C (per task, from traces) | the executor's build traces: tool outputs and answer on TGB; calls, outputs and replies on BFCL | once per task |

## Layout
- **`jev/client.py`**: API client.
  - It reads `JEV_API_KEY` from the environment.
  - It caches every call by the hash of (model, state, questions) and resumes from that cache.
  - It throttles to 600 requests per minute and retries 429, 529 and network errors with backoff.
  - It logs model version, usage and latency for each call.
  - `JEV_MOCK=1` returns labeled mock scores offline.
- **`jev/scoring.py`**: benchmark-independent judgments and prompts: `score_request`, `score_trace`, `score_conversation`, top-K and aggregation.
- **`jev/tgb_run.py`, `jev/tgb_metrics.py`**: TGB driver and metrics. The protocol is 10 families × (80 build / 320 evaluation) with one shared budget K per family.
- **`tgb/`**: the TGB side.
  - `jev_fit_traces.py` dumps the build traces.
  - `eval_baseline_masks.py` is the TGB evaluator with an added `--extra-mask` option, so several spaces are scored in one session. The patch against the original is `eval_baseline_masks_extra_mask.patch`.
  - `tgb_lane.sh` runs all of it in order: traces, then Jev, then evaluation.
- **`jev/bfcl_run.py`, `jev/bfcl_metrics.py`, `bfcl/BFCL_JEV_LANE.sh`**: BFCL v4 `multi_turn_base`.
  - Protocol: 4 primary API classes × (10 build / 40 test).
  - Each record's menu is its involved classes, padded to 30 tools.
  - The top-K*_c tools keep full documentation and the rest are compressed.
  - Aggregation over build records treats a missing tool as 0. The mean over records where the tool appears is kept as an ablation (`Bpm`, `Cpm`).
- **`results/`**: summary files, the TGB masks and the BFCL score files. Raw API caches are not included.
- **`paper/`**: the write-up. It uses the TEA template: `bash paper/build.sh`; figures come from `paper/tools/make_figs.py`.
- **`docs/`**: the protocol notes and the TGB results.

## Dependencies
The drivers reuse existing benchmark code and do not re-implement any evaluator:
- **TGB**: the `tgb` package (scenes, tools, `run_chain`, evaluator).
- **BFCL**: `bfcl_eval` and `mtlots` (menu construction, tier demotion, arm runner).

Set `PYTHONPATH` to include those packages and this repository.

## Commands
```bash
export JEV_API_KEY=...            # never commit the key

# TGB
JEV_MOCK=1 python -m jev.tgb_run --dir $TGB --tag 7b --conds A,B,C --limit-fit 3 --limit-eval 3 --traces <traces.json> --out-dir /tmp/mock
bash tgb/tgb_lane.sh <GPU> 7b <Qwen2.5-7B-Instruct dir> all       # traces -> Jev A/B/C -> one-session evaluation
python -m jev.tgb_metrics --dir $TGB --tag 7b

# BFCL
python -m jev.bfcl_run --key qwen2.5-7b --out-dir <dir> [--limit 2]
KEY=qwen2.5-7b MODEL=<dir> NAME=Qwen2.5-7B-Instruct GPU=1 PORT=8951 JOB=<slurm job> bash bfcl/BFCL_JEV_LANE.sh
python -m jev.bfcl_metrics --key qwen2.5-7b
```

## Results (Qwen2.5-7B executor)
| Benchmark | Keep all | Best baseline | Jev-A | Jev-B | Jev-C |
|---|---|---|---|---|---|
| TGB (macro acc, 3,200 requests) | 69.5 | Trace judge 67.6 | 60.3 | 55.2 | **80.7** |
| BFCL multi-turn (160 records, 3 runs) | 11.3 | Tool2Vec 13.8 | 16.3 | **17.3** | 12.5 |
| GTA read_arith / count_arith / web_fact (32 / 15 / 17) | 6.3 / 0.0 / 0.0 | 12.5 / 6.7 / 11.8 | **15.6** / 6.7 / 0.0 | 9.4 / **20.0** / 0.0 | 9.4 / 6.7 / 5.9 |

Details, per-run numbers and cost are in `paper/main.tex` and `results/`.

## Cost
Jev input costs $0.042 per million tokens, and output tokens are free.
- TGB: all three settings together take 4,800 calls, about $0.38.
- BFCL: 240 calls, about $0.035.
