# Architecture_design -- Ollama LLM-scheduled industrial time-series anomaly detection agent

## Evidence-v2: per-file two-stage routing for the new benchmark run

`TSB-AD/gen_llm_routing_combined.py` now defaults to **evidence-v2**.
`--method legacy` only runs the old "source morphology -> LLM -> metric-table model
selection" experiment; old results, the source prior and the CHARM candidate pool
are not overwritten. The rest of this document's ReAct/Top-6 usage remains the
original path and should not be mixed into the same experiment group as the new
routing.

Two-stage responsibilities of the new version:

1. `Anomaly_algorithm/morphology_evidence.py` extracts, at native resolution and
   per channel, evidence of spikes, mean/trend changes, high variance, low
   volatility, periodic changes and broken channel relations. The training prefix
   is the reference background; tool calls without a reference segment are marked
   `robust_series` and do not claim the background is normal. Multiple morphologies
   can coexist; insufficient evidence returns `unknown/insufficient`. The source
   prior is only conflict-audit information: it does not override the evidence and
   does not raise confidence.
2. `evidence_routing.route_scores` chooses one detector per file based on the
   high-score regions, evidence coverage and ranking of the actual detector
   scores. It preserves the selected model's raw continuous scores -- no
   inversion, no extra smoothing, no label-driven interval expansion. The LLM is
   off by default; an optional local LLM can only review numerically gated
   candidates or fall back to the baseline, and cannot bypass the gate.
   Parse/call failures are logged with the warning and the fallback reason.

The current numerical gates are fixed heuristics (minimum quality 0.6, baseline
margin 0.1) -- **not calibrated probabilities, and no guarantee of beating the
baseline**. This version validates single selection first and does not enable the
not-yet-validated fusion. Windows use 8/16/32/64, trimmed by reference-segment
length; the current evidence is offline and non-causal, not streaming real-time
detection. A contaminated normal prefix can still be misjudged; Eva labels must
not be used to clean it.

### Offline running and the strict score contract

The new version does not call detectors, does not download weights, and does not
treat summary VUS-PR as a score cache. First generate scores using the existing
benchmark detector runners:

```text
<score-dir>/
  <model>/
    <CSV filename without .csv>.npy
```

Each array must be **one-dimensional finite scores over the full file length,
larger means more anomalous**, aligned exactly with the CSV time axis. NaN/Inf,
missing models, or length mismatches fail; there is no `dropna`, truncation,
zero-filling or silent shrinking of the candidate pool. If an old cache dropped
missing rows at generation time, the data must be repaired upstream and the cache
regenerated. CHARM's local cache may be used as candidates; the new version will
not call CHARM's remote service.

Windows PowerShell example (paths and model caches must be provided by the actual
experiment):

```powershell
python .\TSB-AD\gen_llm_routing_combined.py --method evidence `
  --dataset U --split Tuning `
  --score-dir .\TSB-AD\eval\score\uni `
  --models Syntropic Time_RCD OmniAnomaly StreamVAE `
  --baseline Syntropic --protocol-id my-frozen-candidate-config-v1 `
  --output-dir .\TSB-AD\runs\evidence_u_tuning_v1 --evaluate
```

- `--baseline` is required and is not auto-picked from Eva scores. The model names
  in the example are not a validated configuration recommendation.
- Default `--split Tuning`; after freezing the configuration use `--split Eva`,
  and do not repeatedly use Eva to choose gates or the LLM.
- `--protocol-id` identifies the frozen weights, training recipe, preprocessing
  and hyperparameters. The user must still ensure the candidate training
  protocols are comparable; a string or a score hash cannot prove a cache never
  used labels.
- `--limit 3` can be used as a small-sample check; the output marks non-full
  coverage; `--file-list` supports an independent grouped list.
- `full_coverage` only means the requested list is complete; to accept the
  official full set, check `official_full_coverage` and `official_count`
  (U-Eva 350, M-Eva 180). The CLI checks against the official manifest; a complete
  small-sample list cannot impersonate the official full set.
- `--review-model llama3.1:8b` explicitly enables the local evidence review at
  `127.0.0.1:11434`. Ollama is not required when it is not enabled. The LLM does
  not receive filenames, anomaly-position metadata or labels.
- The output directory must not exist, to prevent overwriting a previous experiment.

The output includes per-file `scores/*.npy`, `decisions.jsonl`, `run.json`. The
audit stores evidence, candidate quality, selection/fallback reasons, input and
score SHA-256, code hash and dependency versions. `--evaluate` reads labels only
**after all decisions are written**, calls the project's official metric
implementation, produces `metrics.csv`, and reports per-file equal-weight VUS-PR
(0-1) for both the fixed baseline and the routing. Missing files or invalid
metrics mark the run failed, and incomplete results are not reported as full
scores.

### Validation and ablation

```powershell
python -m unittest Architecture_design.tests.test_evidence_routing -v
python Architecture_design\tests\test_agent.py
```

Tests cover: normal noise/periodicity, spike and long-drift localization, stuck/
flatline, multivariate cancellation and relation breakage, shift/scale invariance,
long series with short spikes, score ties, strict input contracts, LLM gating,
label independence, full-coverage audit and the official VUS-PR integration.
Passing synthetic tests does not imply real benchmark gains. Compare the fixed
model, legacy, numerical evidence-v2 and the LLM review version grouped by
independent data source, and freeze the configuration before running Eva.

Hand the "tool scheduling decision" to a local Ollama LLM: the model reads the
unified tool catalog and, in a ReAct loop, decides for itself which detector to
call, with what parameters, how to set thresholds and how to organize events,
finally emitting an auditable JSON conclusion for end-to-end anomaly detection on
**TSB-AD-U** univariate time series.

The model only makes **decisions**; the actual algorithm execution all lands on
real tools already registered in the project -- every step carries `provenance`
(tool name / version / dependency versions / parameters / time), traceable and
reproducible.

## Mapping to the paper's three-layer architecture

The event-driven asynchronous diagnostic architecture in the paper (Layer1
detection into events / Layer2 running the selected toolchain / Layer3 orchestrating
asynchronous RCA); this directory implements **Layer2's "decision brain"**:

| Paper layer | Implemented in this directory |
|---|---|
| Layer1 detection into events | loaded by `dataset.split_standardize` + training-segment standardization (protocol discipline: statistics fit only on the training segment) |
| Layer2 selected toolchain | the **Ollama LLM** decides which toolchain to take (detector -> dynamic threshold -> event organization) |
| Layer3 orchestrated RCA | reserved: `causelens` / `kfasl` are already wired in as callable functions (univariate degrades to single channel) |

Unlike the paper's offline toolchain selection via orthogonal tables / factorial
design, here the LLM makes **online, interpretable per-series decisions** -- the
two are complementary: the offline design answers "which chain is optimal", while
the online agent answers "which chain this data should take and why".

## Directory structure

```
Architecture_design/
├── config.py     # model / host / timeout / round budget (env-overridable)
├── catalog.py    # aggregate the four algorithm module registries -> unified tool catalog + system-prompt text
├── client.py     # requests talking directly to Ollama's native /api/chat (no ollama/openai packages)
├── schemas.py    # Ollama function schemas for the 12 curated callable functions
├── executor.py   # tool call -> real tool run() adaptation + result summarization + workspace state
├── dataset.py    # TSB-AD-U loading / splitting / training-segment standardization
├── metrics.py    # AUC-ROC / AUC-PR / best F1 / PA-F1 (threshold-free, independent evaluation of decision quality)
├── agent.py      # ReAct main loop (model decision -> execute -> return -> conclusion)
├── main.py       # CLI end-to-end demo
└── tests/        # smoke tests (dry-run, offline)
```

## Usage

```bash
# Real LLM (default llama3.1:8b, stable native tool calling)
python Architecture_design/main.py --file 001_NAB_id_1_Facility_tr_1007_1st_2014.csv --verbose

# Specify a model
python Architecture_design/main.py --file 001_NAB_id_1_Facility_tr_1007_1st_2014.csv --model llama3.1:8b

# Offline, fixed decision script to verify the whole chain
python Architecture_design/main.py --file 001_NAB_id_1_Facility_tr_1007_1st_2014.csv --dry-run --verbose

# Smoke test
python Architecture_design/tests/test_agent.py
```

Available models (`ollama list`): `llama3.1:8b`, `qwen3-coder-30b-a3b` (more
accurate but slower), etc. Environment variables `OLLAMA_MODEL` / `OLLAMA_HOST`
can override the defaults. Note that `qwen2.5-coder:7b`'s Ollama template is
incomplete and writes tool calls into `content` (the client parses it as a
fallback, but it is not the default).

## Design decisions

1. **Expose only 12 curated functions, not 149 schemas**. A 7B-level model's tool
   calling reliability degrades markedly when there are too many functions; the
   rest are looked up by name via `query_tool`, with metadata injected into the
   system-prompt catalog (`catalog.catalog_brief()`), so the model "knows what
   exists and what it can look up".
2. **Result summarization** (`executor._digest`). The full score/mask arrays
   returned by tools are not sent back to the model; only statistics, top indices,
   alarm counts and event intervals are returned, keeping context length under
   control.
3. **Decision and evaluation are decoupled**. The model chooses a threshold to
   produce events, but evaluation uses threshold-free metrics such as AUC/AUPRC
   (continuous scores in `executor.score_history`), objectively quantifying the
   quality of the "which detector to pick" decision.
4. **Protocol discipline**. Standardization statistics are fit only on the
   training segment (`dataset.split_standardize`), and detection runs only on the
   test segment, avoiding leaking the test distribution into preprocessing.
5. **Decision trajectory as audit**. `trace` records every step
   "model decision (tool + parameters) -> execution result -> provenance", finally
   written to `runs/<file>.json`.
6. **Two stages + guardrails against noise**. A local 8B model easily confuses
   "calling tools" and "emitting JSON", so the loop is split into acting (only
   scheduling tools; reply "done" when evidence suffices) and finalize (no tools,
   emit JSON alone); plus two guardrails -- refusing "done" before any detector
   has run, and parameter cleaning (`executor` keeps only the legal keys declared
   in the schema, filtering out fields the model crams in).
7. **Label-free toolchain search (Pareto frontier / orthogonal idea)**.
   `search_toolchain` runs the "detector x hyperparam" grid once and selects the
   optimum using **label-free intrinsic metrics**: the score first-order
   autocorrelation `acf1` (good scores for segment-level anomalies should be
   smooth and persistent; measured Spearman correlation with AUC is about +0.9)
   and `runlen` (top1% segment length), taking the non-dominated frontier on the
   two objectives and returning the recommended detector. This is the online form
   of the paper's "toolchain selection" theme -- no labels, choosing the toolchain
   by intrinsic criteria.
8. **Multi-detector + rolling volatility**. The new `rolling_volatility`
   (Anomaly_algorithm 019, rolling std/range/mad) targets level-shift /
   variance-rise segment-level anomalies; `ensemble_fuse` does rank_max/rank_mean
   fusion over the default panel. On TSB-AD-U, the searched rolling_volatility
   beats isolation forest by about 0.2 AUC (e.g. 008: 0.60->0.80, 001: 0.51->0.75).

## Extension points

- **Add tools**: add a schema in `schemas.py` and a mapping + adapter in
  `executor._DETECTORS/_POST/_EXPLAIN`; the agent main loop needs no change.
- **Multivariate / RCA**: swap `dataset` for TSB-AD-M multivariate loading, and
  `causelens/kfasl` can do real multi-channel root-cause attribution; the 27 tools
  in the RCA directory can be wired in as functions the same way.
- **Hybrid online/offline decisions**: append the offline factorial optimal chain
  after the agent's conclusion as cross-validation (the paper's Layer2 "selected
  toolchain").
