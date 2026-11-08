# Morphology- & Source-Prior-Guided LLM Routing for Time-Series Anomaly Detection

A label-free routing framework for time-series anomaly detection built on top of the
[TSB-AD](https://github.com/TheDatumOrg/TSB-AD) benchmark. Instead of relying on a single
detector, it first classifies the **anomaly morphology** of each series and uses a
**source prior**, then routes the series to the strongest detector for that
morphology/source bucket, and finally verifies the choice with a local LLM.

The pipeline is **label-free at inference time**: it never touches the labels to make
routing decisions. Labels are used only once, at the very end, to report VUS-PR.

---

## What it does

1. **Morphology evidence** (`Anomaly_algorithm/`) — reference-relative, channel-preserving
   evidence extraction that classifies anomalies into kinds such as `spike`, `level_shift`,
   `trend`, `variance_change`, `flatline`, `seasonal`, etc., and produces a per-point
   `support` signal without using labels.

2. **Hierarchical routing** (`TSB-AD/eval_loop_route*.py`) — learns, per
   (morphology, source) bucket, which detector is the strongest base (`best`) and which
   complementary detector pair (`tp_model`/`fp_model`) should raise missed detections and
   suppress false alarms. Buckets fall back level-by-level:

   ```
   (fine_morph, source)  ->  fine_morph  ->  (coarse_morph, source)
        ->  coarse_morph  ->  source  ->  global
   ```

3. **Label-free zero-shot loop** — a frozen set of parameters
   (`best`, `tp_model`, `fp_model`, `K`, `support_q`, `fused_q`) is applied with a
   label-free gain criterion (`gain_tau` calibrated during tuning) to decide whether the
   loop actually helps; if not, it falls back to the base detector.

4. **LLM verification** (`TSB-AD/llm_verify_best.py`) — a local Ollama model audits every
   bucket's `best` selection on two questions:
   - `highest_score_ok`: is `best` the highest-scoring candidate?
   - `prior_conform_ok`: does `best` agree with the morphology evidence and source prior?
   
   Verification is audit-only by default (`--llm-verify`); `--llm-correct` makes it
   actually override the `best` when the LLM flags a conflict.

5. **Multi-detector pool** — `Syntropic`, `Time_RCD`, `CHARM_stat_kNN`, and `OmniAnomaly`
   (VAE reconstruction, strong on point/dense anomalies such as the `Stock` source).

---

## Repository layout

```
Anomaly_algorithm/    morphology classification + evidence extraction (label-free)
Architecture_design/  LLM agent architecture (Ollama client, schemas, routing benches)
Data_preprocessing/   50+ reusable preprocessing / feature tools
TSB-AD/               custom routing pipeline scripts (see below)
Memory_base/          runtime morphology memory (generated, git-ignored)
```

Key pipeline entry points under `TSB-AD/`:

| Script | Purpose |
|---|---|
| `gen_local_scores.py` / `gen_local_scores_multi.py` | generate per-detector scores (univariate / multivariate) |
| `eval_loop_route.py` / `eval_loop_route_multi.py` | tuning (learn routing params) + eval (frozen, label-free) |
| `llm_verify_best.py` | local-LLM audit of the `best` selection |
| `llm_source_select.py` | per-source LLM detector selection benchmark |
| `fast_vus.py` | vectorized VUS-PR evaluation |

The upstream `TSB_AD/` package and `benchmark_exp/` are **not** vendored in this repo;
install them with `pip install TSB-AD`.

---

## Installation

- Python 3.10+
- `pip install TSB-AD` (the upstream benchmark; provides the `TSB_AD` package)
- `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `requests`
- [Ollama](https://ollama.com) (optional, only for LLM verification; the routing works
  without it)

Download the datasets separately (they are not part of this repository):

- TSB-AD-U: https://www.thedatum.org/datasets/TSB-AD-U.zip
- TSB-AD-M: https://www.thedatum.org/datasets/TSB-AD-M.zip

Place them so the following paths resolve:

```
Dataset/TSB-AD-U/          # univariate CSVs (value column only)
Dataset/TSB-AD-M/          # multivariate CSVs (channels only)
Validation/TSB-AD-U/       # labels
Validation/TSB-AD-M/       # labels
TSB-AD/Datasets/File_List/ # tuning/eval split CSVs (already in this repo)
```

---

## Quick start

All commands run from the repository root (scripts locate themselves via `__file__`).

```bash
# 1) Generate per-detector scores (resumable; skips files already computed)
python TSB-AD/gen_local_scores.py --workers 2 --max-len 25000

# 2) Tuning: learn routing parameters -> runs/route_params.json
python TSB-AD/eval_loop_route.py --split tuning --workers 2

# 3) Eval: label-free evaluation with the frozen parameters
python TSB-AD/eval_loop_route.py --split eva --workers 2 --max-len 8000
```

Multivariate:

```bash
python TSB-AD/gen_local_scores_multi.py --workers 1 --max-len 8000
python TSB-AD/eval_loop_route_multi.py --split tuning --workers 2
python TSB-AD/eval_loop_route_multi.py --split eva --workers 2 --max-len 8000
```

Outputs:

- `runs/route_params.json` / `runs/route_params_multi.json` — frozen routing parameters.
- `route_progress.json` + `route_data.js` — progress / dashboard data.
- `runs/route_llm_verify.json` / `runs/route_llm_verify_multi.json` — LLM audit verdicts.

---

## Command-line options

| Option | Meaning | Default |
|---|---|---|
| `--split` | `tuning` / `eva` / `both` | `tuning` |
| `--workers` | number of processes | 2 |
| `--max-len` | downsample ceiling for very long files | 25000 (tuning), 8000 (eval) |
| `--keep-tr` | keep the training prefix in eval (with-tr) | off (strip-tr) |
| `--llm-verify` / `--no-llm-verify` | enable/disable the LLM audit | on |
| `--llm-correct` | let the LLM override `best` on conflict | off |
| `--llm-model` | local model name | `config.OLLAMA_MODEL` (`llama3.1:8b`) |
| `--llm-timeout` | per-request timeout (seconds) | 60 |

### Eval with / without the training prefix

- **Strip-tr (default)** — evaluate only the detection segment after `tr`, excluding the
  normal training prefix from the metric.
- **With-tr** — `--keep-tr` keeps the training prefix in the evaluation.

```bash
python TSB-AD/eval_loop_route.py --split eva --workers 2 --max-len 8000            # strip-tr
python TSB-AD/eval_loop_route.py --split eva --workers 2 --max-len 8000 --keep-tr  # with-tr
```

### LLM verification and correction

```bash
# Audit only (does not change routing)
python TSB-AD/eval_loop_route.py --split tuning --workers 2 --llm-verify

# Audit + correct: on conflict, override best with expected_model (drops the loop)
python TSB-AD/eval_loop_route.py --split tuning --workers 2 --llm-verify --llm-correct
```

---

## Notes

- `best` is chosen as `argmax` of the mean VUS-PR within each bucket during tuning; adding
  `OmniAnomaly` to the pool lets the routing pick it automatically for sources where it is
  strongest (e.g. `Stock`).
- `--llm-verify` is **audit-only** by default, so it does not change VUS-PR. Use
  `--llm-correct` only after checking how many buckets the LLM flags as `conflict`.

---

## License

[MIT](LICENSE) © 2026 sircao.

This project builds on [TSB-AD](https://github.com/TheDatumOrg/TSB-AD), which is
Apache-2.0 licensed. The upstream benchmark code is not included in this repository.
