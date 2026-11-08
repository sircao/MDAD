# Anomaly_algorithm -- anomaly detection algorithms and open-source project survey tool suite

Turn each project in the survey report (section 5, research project survey) into a callable tool, all following the `ToolResult`
contract and directly schedulable by an Agent. The survey list has 20 entries; after deduplication (ADBench, SOTA-Deep,
and awesome-TSAD each appear twice) it becomes **17 projects**.

All algorithm tools are written as **real implementations** (native numpy/scipy/scikit-learn equivalent implementations, or loading real weights inside the repo),
run directly to produce results, and do not depend on uninstalled third-party packages.

## Directory structure

```
Anomaly_algorithm/
├── core.py          # Reuses Data_preprocessing's unified contract + ReferenceTool (knowledge retrieval base class)
├── registry.py      # discover / get / list_tools / by_category
├── tools/           # 17 tools, one folder each (tool.py + README.md)
├── README.md
└── smoke_test.py    # Smoke test (actually runs 7 detectors)
```

## Two kinds of tools

- **Detector tools (7, real implementations)**:
  - `pyod` -- isolation forest/LOF/OCSVM/HBOS/KNN/PCA reconstruction, a unified interface outputting pointwise scores
  - `adtk` -- quantile/fixed threshold/persistent difference/level shift, outputs boolean anomaly labels
  - `merlion` -- default (z-score)/IQR/EWMA residual/isolation forest
  - `tods` -- z-score/IQR/moving-average residual/isolation forest, supports scores or labels
  - `elki` -- LOF/k-nearest-neighbor mean distance/k-th nearest-neighbor distance
  - `pysad` -- streaming ExactSTORM/KNND/moving z-score, updates scores incrementally
  - `time_rcd` -- **loads the Time-RCD encoder weights inside the repo**, pointwise zero-shot representation deviation scoring
- **Knowledge retrieval tools (10, non-algorithm entries)**: `dada` `foundation_model_eval` `adbench`
  `anomaly_detection_resources` `sota_deep_ad` `anomaly_detection_algorithms`
  `awesome_anomaly_detection` `awesome_tsad` `cran_anomaly_detection` `iad_survey`.
  These are model introductions, papers, benchmarks, indexes, and surveys (no single implementable algorithm); `run()` returns structured
  metadata (name/address/type/introduction/usage notes), serving knowledge retrieval and evidence sourcing.

## Tool list

| Category | Tools |
|---|---|
| Zero-shot and general models | `time_rcd` (real encoder) `dada` `foundation_model_eval` |
| Algorithm libraries and frameworks | `pyod` `adtk` `merlion` `tods` `elki` `pysad` |
| Benchmarks and datasets | `adbench` `awesome_tsad` |
| Algorithm and paper indexes | `anomaly_detection_resources` `sota_deep_ad` `anomaly_detection_algorithms` `awesome_anomaly_detection` `cran_anomaly_detection` `iad_survey` |

## Usage

```python
from Anomaly_algorithm.registry import get, list_tools

for row in list_tools():            # Agent reads this list: name/category/runnable/installed/origin
    print(row["name"], row["category"], row["runnable"], row["installed"])

# Detector (really runs)
scores = get("pyod").run(data, algorithm="iforest").data
events = get("adtk").run(data, detector="quantile").data
zero_shot = get("time_rcd").run(data).data     # loads real Time-RCD weights

# Knowledge retrieval
info = get("adbench").run().data
```

## Notes

- **New morphology evidence**: `morphology_evidence.extract_evidence(data, reference=normal_prefix)`
  preserves channels and original sample points, returning multi-scale, reference-relative point/segment/relationship anomaly evidence;
  `summary()` returns an auditable morphology ranking, contiguous intervals (left-closed right-open), and the insufficient-evidence state.
  The source name is recorded only as a prior-conflict note and never overrides the signal criteria. Evidence strength is not an anomaly probability.
  The old classifier in `morphology.py` is kept for legacy ablations.
- The `morphology_features` and `morphology_classify` tools can explicitly pass
  `method="evidence", reference=normal_prefix` to use the new version; input supports `(T, C)`.
  For compatibility with existing tool calls, the default remains `method="legacy"`; the new output is no longer the old flat feature dict.
  `window=0` automatically uses multiple scales; an explicit window must be an integer at least 4 and no larger than half of the shorter of reference/input.
  Data and reference need at least 8 points; non-finite input or inconsistent channel counts are rejected.
  For the new leaderboard entry, cache protocol, and validation commands see [Architecture_design](../Architecture_design/README.md).
- The contract reuses `Data_preprocessing.core.base`; `list_tools()` additionally outputs `runnable` (whether it is a
  detector) and `origin` (source).
- All detectors use native numpy/scipy/scikit-learn implementations (equivalent semantics); `time_rcd` loads
  the real encoder shards from `TS-Router-D8FF/checkpoints`; the repo shards only contain encoder parameters (the reconstruction head /
  anomaly head are not exported), so `time_rcd` uses the L2 distance from the local representation to the per-channel median representation as its zero-shot score.
- Dependencies: detectors need `numpy/scipy/scikit-learn` (installed); `time_rcd` additionally needs `torch` (installed).
