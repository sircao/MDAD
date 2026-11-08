"""Callable function schemas for Ollama (function definitions for native tool calling).

Design trade-off: the project has 149 tools in total, and stuffing all schemas
into a 7B model would severely hurt decision reliability. So only **12 curated
functions** are exposed here (7 detectors + 2 post-processing + 2 explanation +
1 knowledge lookup); the rest are looked up by name via ``query_tool``, with
metadata injected into the system-prompt catalog.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _fn(
    name: str,
    description: str,
    properties: Dict[str, Any],
    required: List[str],
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


FUNCTIONS: List[dict] = [
    _fn(
        "pyod",
        "Anomaly detection (prefer the score type): isolation forest/LOF/OCSVM/HBOS/KNN/PCA reconstruction, outputting pointwise continuous anomaly scores (larger = more anomalous). Suitable as a first call to provide scores for later thresholding/event organization.",
        {
            "algorithm": {
                "type": "string",
                "enum": ["iforest", "lof", "ocsvm", "hbos", "knn", "pca"],
                "description": "Detection algorithm; iforest (isolation forest) is a robust default.",
            }
        },
        ["algorithm"],
    ),
    _fn(
        "adtk",
        "Rule-based detection: directly outputs boolean anomaly labels, no thresholding. quantile=quantile, threshold=fixed threshold, persist=persistent difference, level_shift=level shift.",
        {
            "detector": {
                "type": "string",
                "enum": ["quantile", "threshold", "persist", "level_shift"],
                "description": "Detector type.",
            }
        },
        ["detector"],
    ),
    _fn(
        "merlion",
        "Unsupervised detection: default/zscore are z-score thresholds (boolean), iqr is interquartile range (boolean), ewma is exponentially weighted residuals (continuous scores), isolation_forest is per-column isolation forest (boolean).",
        {
            "detector": {
                "type": "string",
                "enum": ["default", "zscore", "iqr", "ewma", "isolation_forest"],
                "description": "Detector type.",
            }
        },
        ["detector"],
    ),
    _fn(
        "tods",
        "Detection: zscore/iqr/moving_average/isolation_forest. return_scores=True outputs continuous scores, False outputs boolean labels.",
        {
            "detector": {
                "type": "string",
                "enum": ["zscore", "iqr", "moving_average", "isolation_forest"],
                "description": "Detector type.",
            },
            "return_scores": {
                "type": "boolean",
                "description": "Whether to output continuous scores (set true if you will threshold later).",
            },
        },
        ["detector"],
    ),
    _fn(
        "elki",
        "Anomaly scores based on neighbor distances: lof=local outlier factor, knn=mean neighbor distance, kth_neighbor=distance to the k-th neighbor. Outputs continuous scores.",
        {
            "algorithm": {
                "type": "string",
                "enum": ["lof", "knn", "kth_neighbor"],
                "description": "Neighbor algorithm.",
            }
        },
        ["algorithm"],
    ),
    _fn(
        "pysad",
        "Streaming detection scores: exact_storm / knnd / sliding_zscore, updated point by point. Outputs continuous scores.",
        {
            "model": {
                "type": "string",
                "enum": ["exact_storm", "knnd", "sliding_zscore"],
                "description": "Streaming model.",
            }
        },
        ["model"],
    ),
    _fn(
        "time_rcd",
        "Zero-shot time-series representation deviation scoring (loads the Time-RCD encoder), outputting pointwise continuous scores. Slower on first call.",
        {"score": {"type": "string", "enum": ["deviation"], "description": "Scoring method."}},
        [],
    ),
    _fn(
        "rolling_volatility",
        "Rolling-window volatility (std/range/mad) as pointwise continuous scores, good at level-shift / variance-increase segment-level anomalies. Larger window = smoother.",
        {
            "stat": {"type": "string", "enum": ["std", "range", "mad"], "description": "Volatility statistic; std is the most general."},
            "window": {"type": "integer", "description": "Rolling window length (20/50/100 etc.)."},
        },
        [],
    ),
    _fn(
        "two_sided_volatility",
        "Two-sided volatility (direction-agnostic, will not invert): |rolling std − median rolling std| as pointwise scores, covering local anomalies at both the high-volatility and low-volatility/stuck ends. A robust fallback when the anomaly direction is uncertain.",
        {"window": {"type": "integer", "description": "Rolling window length (default 100)."}},
        [],
    ),
    _fn(
        "two_sided_knn",
        "Two-sided KNN distance (direction-agnostic, will not invert): |knn distance − median distance| as pointwise scores, covering both isolated point anomalies and dense clusters (collective).",
        {"n_neighbors": {"type": "integer", "description": "Number of neighbors (default 5)."}},
        [],
    ),
    _fn(
        "stuck",
        "Stuck detection: constant run length (values unchanged for a long time = stuck/flatline), applied directly to the raw data, specifically catching low-volatility/stuck-morphology local anomalies.",
        {"eps": {"type": "number", "description": "Maximum difference judged as unchanged (default 1e-3)."}},
        [],
    ),
    _fn(
        "tsbad_model",
        "TSB-AD strong-model detector: runs SOTA models from the TSB-AD model library with optimal hyperparameters (Syntropic=strongest for segments, OmniAnomaly=strongest for points/dense clusters, Time_RCD=baseline model, StreamVAE/Sub_PCA etc.), outputting pointwise anomaly scores. Models are routed by source/morphology to avoid direction inversion.",
        {
            "model": {
                "type": "string",
                "enum": ["Syntropic", "OmniAnomaly", "Time_RCD", "StreamVAE", "Sub_PCA", "IForest", "LOF", "TranAD", "USAD"],
                "description": "TSB-AD model name; Syntropic is strongest for segments, OmniAnomaly for points/dense clusters, Time_RCD is the baseline model.",
            }
        },
        ["model"],
    ),
    _fn(
        "dynamic_threshold",
        "Converts continuous anomaly scores into an alarm mask. method=global uses a global quantile threshold; rolling uses a rolling quantile (requires window). Detector scores must exist first.",
        {
            "method": {"type": "string", "enum": ["global", "rolling"], "description": "Thresholding method."},
            "quantile": {"type": "number", "description": "Quantile (default 0.99)."},
            "window": {"type": "integer", "description": "Window length for rolling."},
        },
        ["method"],
    ),
    _fn(
        "hysteresis_merge",
        "Debounces the alarm mask: filters events that are too short and merges adjacent alarms. Use after dynamic_threshold.",
        {
            "min_duration": {"type": "integer", "description": "Minimum event duration in points (default 1)."},
            "merge_gap": {"type": "integer", "description": "Merge two events when the gap between them is ≤ this value (default 0)."},
        },
        [],
    ),
    _fn(
        "causelens",
        "Root-cause explanation via causal graph + reconstruction error + counterfactuals (multivariate attribution; univariate series degrades to single-channel).",
        {"threshold": {"type": "number", "description": "Lag correlation threshold (default 0.4)."}},
        [],
    ),
    _fn(
        "kfasl",
        "Variance-stabilized Shapley attribution + causal regularization, outputting channel importance ranking and stability (multivariate).",
        {
            "n_samples": {"type": "integer", "description": "Shapley sample count (default 100)."},
            "n_runs": {"type": "integer", "description": "Number of repetitions (default 5)."},
        },
        [],
    ),
    _fn(
        "query_tool",
        "Query metadata of any registered tool by name (including knowledge/review/benchmark entries), for methodology reference or provenance; does not run the algorithm directly.",
        {"name": {"type": "string", "description": "Tool name, e.g. adbench, awesome_tsad, foundation_model_eval."}},
        ["name"],
    ),
    _fn(
        "search_toolchain",
        "Label-free toolchain search: runs once over the candidate detector × hyperparameter grid, builds a Pareto front using the intrinsic metrics acf1 (score persistence, correlated with AUC ≈ +0.9) and runlen (anomaly segment length), and returns the recommended detector and full ranking. Recommended to call it first to pick a detector.",
        {},
        [],
    ),
    _fn(
        "ensemble_fuse",
        "Multi-detector ensemble: rank fusion over the default panel (rolling volatility ×4 + moving average + isolation forest). method=rank_max treats any detector alarm as an anomaly (stronger when complementary), rank_mean takes the mean (more conservative). Outputs fused scores.",
        {
            "method": {
                "type": "string",
                "enum": ["rank_max", "rank_mean"],
                "description": "Fusion method.",
            }
        },
        [],
    ),
]
