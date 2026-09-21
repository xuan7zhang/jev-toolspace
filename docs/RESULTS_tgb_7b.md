# Jev on TGB: Qwen2.5-7B executor (2026-09-21, jev-1.13.0)

- **Setup:** one vLLM session on kn076. 3,200 evaluation requests. Budget matched to LOTS K\*_f, a mean of 5.1 tools.
- **Session check:** Keep all 69.47, Dense 24.91, Tool2Vec 38.09, Trace judge 67.62 and LOTS 82.28 each match their earlier separate sessions to four digits.
- **Files:**
  - `tgb4/jev/eval_jev_on_7b.json`: per-request outcomes.
  - `tgb4/jev/summary_7b.json`: tables, bootstrap and failure buckets.
  - `tgb4/jev/full/tgb_jev_run_7b.json`: Jev scores, spaces and cost.

| method | kind | macro acc | Δ Keep all [95%] | Δ LOTS [95%] | req. recall | chain cover | \|S\| | exec. tokens/req |
|---|---|---|---|---|---|---|---|---|
| Keep all | static | 69.47 | 0 | −12.80 [−14.3, −11.3] | 1.000 | 1.000 | 15.0 | 271 |
| LOTS | build-once | **82.28** | +12.80 [+11.3, +14.3] | 0 | 1.000 | 1.000 | 5.1 | 163 |
| Jev C (output-aware) | build-once | 80.66 | +11.19 [+9.8, +12.6] | −1.62 [−2.4, −0.9] | 1.000 | 1.000 | 5.1 | 165 |
| Trace judge (7B) | build-once | 67.62 | −1.85 [−3.3, −0.3] | −14.65 | 0.944 | 0.889 | 5.1 | 156 |
| Jev A (request router) | per-request | 60.28 | −9.18 [−10.6, −7.8] | −21.99 | 0.883 | 0.729 | 5.1 | 147 |
| Jev B (task space) | build-once | 55.16 | −14.31 [−15.6, −13.0] | −27.12 | 0.852 | 0.667 | 5.1 | 141 |
| Tool2Vec | build-once | 38.09 | −31.37 | −44.17 | 0.741 | 0.444 | 5.1 | 123 |
| Dense | build-once | 24.91 | −44.55 | −57.36 | 0.370 | 0.222 | 5.1 | 122 |
| Random | build-once | 23.47 | −46.00 | −58.80 | 0.426 | 0.222 | 5.1 | 117 |

The Δ intervals come from a family-stratified paired bootstrap (B=2000). Recall and coverage exclude `no_tool`.

## Cost

| | calls | input tokens | USD | latency |
|---|---|---|---|---|
| B build | 800 | about 1.46M | about $0.06 | 0.35 s/call |
| C build | 800 | about 1.73M | about $0.07 | 0.35 s/call |
| A online | 3,200 (1 per request) | 5.85M (about 1.8k per request) | $0.246 in total, about $7.7e-5 per request | +0.31 s per request |

- Latency was measured with 8 threads at 600 rpm, on cache misses only.
- LOTS build uses 12,800 executor forward passes on the local GPU. It makes no API calls.
- All build-once methods add no online cost.

## Readings
1. **Request-level selection (A).**
   - Required-tool recall is 0.883, but the complete chain is covered for only 72.9% of requests.
   - Almost all misses are one of three systematic substitutions:
     - ExchangeRate is replaced by GoogleSearch in fx_settle (270 cases). ExchangeRate has an empty docstring.
     - TableQuery is replaced by DocRetrieve in table_total (232 cases).
     - DocRetrieve is replaced by TableQuery in doc_two_facts (203 cases).
   - In each substitution the missed tool is the upstream data source, and a missed upstream breaks the whole chain. The affected families fall to 0.3%, 22.5% and 28.4%.
2. **Aggregating A into a task space (B) is worse than per-request selection** (55.16 vs 60.28).
   - The confusions are systematic, so the family mean locks the wrong source in for every request: doc_two_facts and table_total drop to 0.
   - Aggregation helps only when per-request errors are independent. Here they are not.
3. **Output-aware judging (C).**
   - C recovers every required tool: recall 1.000, chain coverage 1.000, and no missed upstream tool.
   - C is +13.0 over the Trace judge built on the executor itself.
   - C stays 1.62 below LOTS; the interval excludes 0.
   - The gap comes from the non-required tools that fill the budget. They matter because the served menu changes the executor's context. LOTS scores these fill-in tools by the executor's own likelihood, while Jev's probabilities for them are all about 0.02–0.05 and effectively tie.
     - schedule_gap: LOTS 79.7 vs C 75.0.
     - sensor_mean: LOTS 68.4 vs C 59.7.
     - no_tool: C picks Barcode at 0.03 and scores 95.9, vs 97.5.
4. **Selection quality and downstream accuracy.**
   - Chain coverage orders the methods. Among the budget-matched spaces: coverage 1.0 gives 80.7–82.3; 0.89 gives 67.6; 0.67–0.73 gives 55–60; 0.22–0.44 gives 23–38.
   - With the chain covered, the remaining ±2 points depend on which extra tools fill the budget.
   - Executor tokens are nearly identical across budget-matched methods (141–165 vs 271 for Keep all). Serving savings come from K, not from the selector.

## Status of each claim
- **Holds under this protocol (one executor, deterministic greedy decoding, sessions reproduce to four digits):** readings 1–4 and the cost figures.
- **Untested:**
  - Other executors.
  - Registries with distractors.
  - BFCL and GTA, where traces cover only some tools and the unobserved-tool rule starts to matter.
  - Whether filling in ExchangeRate's empty docstring closes A's fx_settle gap. The docstring was left empty on purpose for parity with the baselines.
  - The Jev-specific K. K is borrowed from LOTS.
- **Reading the "covered but wrong vs Keep all" bucket:** LOTS 115, C 115, A 114, Trace judge 213. Changing the menu redraws tool outputs through the `run_chain` RNG, so this bucket mixes output redraw with execution failure.
