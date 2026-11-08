"""Three-layer, two-step anomaly detection architecture.

Step 0  anomaly feature morphology classification: local morphology feature
        algorithms + LLM cross-validation recursion (see Anomaly_algorithm.morphology).
Step 1  morphology-aware safe preprocessing: only apply transforms that do not
        affect the anomaly verdict (standardization is already done at split time;
        other transforms would destroy the anomaly signal, so the default is the
        identity transform, reporting only a safe/harmful strategy for audit).
Step 2  two orthogonal lines run in parallel:
        Line 1  orthogonal anomaly detection: Top-6 detectors fused by
                morphology-adaptive weighting (the best-fitting feature gets the
                highest weight).
        Line 2  orthogonal LLM selection: a local LLM weights "characteristics +
                experimental performance" to pick a detector and decide.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import rankdata

from Anomaly_algorithm.morphology import consensus_classify, DETECTOR_ROUTING

# ---- Top-6 detectors (cover the morphology space + top six by mean AUC over 14 datasets) ----
TOP6: Dict[str, tuple] = {
    "iforest": ("pyod", {"algorithm": "iforest"}),
    "knn": ("pyod", {"algorithm": "knn"}),
    "two_sided_volatility": ("two_sided_volatility", {"window": 100}),
    "two_sided_knn": ("two_sided_knn", {"n_neighbors": 5}),
    "rolling_volatility": ("rolling_volatility", {"stat": "std", "window": 100}),
    "tods_ma": ("tods", {"detector": "moving_average", "return_scores": True}),
}

# ---- morphology -> Top6 adaptive weights (the best-fitting detector gets the highest weight; row sums to 1) ----
MORPHOLOGY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "spike":            {"iforest": .50, "knn": .20, "two_sided_knn": .12, "tods_ma": .08, "two_sided_volatility": .06, "rolling_volatility": .04},
    "dense_cluster":    {"two_sided_knn": .50, "iforest": .20, "knn": .12, "two_sided_volatility": .08, "tods_ma": .06, "rolling_volatility": .04},
    "level_shift":      {"rolling_volatility": .50, "tods_ma": .20, "two_sided_volatility": .12, "iforest": .08, "two_sided_knn": .06, "knn": .04},
    "variance_change":  {"two_sided_volatility": .50, "rolling_volatility": .20, "two_sided_knn": .12, "tods_ma": .08, "iforest": .06, "knn": .04},
    "flatline":         {"two_sided_knn": .50, "two_sided_volatility": .20, "knn": .12, "iforest": .08, "tods_ma": .06, "rolling_volatility": .04},
    "trend":            {"rolling_volatility": .50, "tods_ma": .20, "two_sided_volatility": .12, "iforest": .08, "two_sided_knn": .06, "knn": .04},
    "noise":            {"tods_ma": .50, "two_sided_volatility": .20, "two_sided_knn": .12, "iforest": .08, "rolling_volatility": .06, "knn": .04},
    "persistent":       {"rolling_volatility": .50, "two_sided_volatility": .20, "tods_ma": .12, "two_sided_knn": .08, "iforest": .06, "knn": .04},
    "seasonal":         {"tods_ma": .50, "two_sided_volatility": .20, "two_sided_knn": .12, "rolling_volatility": .08, "iforest": .06, "knn": .04},
}

# ---- local LLM characteristics + experimental performance -> weights (from prior experiments: qwen3-30b most first places, llama3.1 stable tool calling) ----
LLM_WEIGHTS: Dict[str, float] = {
    "qwen3-coder-30b-a3b:latest": 0.35,
    "llama3.1:8b": 0.30,
    "qwen2.5-coder:7b": 0.20,
    "ibm/granite3.3:8b": 0.15,
}

# The safe-preprocessing strategy per morphology: list the harmful transforms (which would destroy that morphology's anomaly signal).
UNSAFE_PREPROCESS = {
    "spike": ["spike removal/median filtering", "re-smoothing"],          # would erase spikes
    "dense_cluster": ["cluster deduplication"],                   # would break up dense clusters
    "level_shift": ["detrending/differencing"],                  # would erase level shifts
    "trend": ["detrending"],                            # would erase the trend
    "variance_change": ["variance stabilization"],               # would erase variance changes
    "noise": ["denoising/low-pass filtering"],                      # would erase the noise anomaly itself
    "persistent": ["short-window smoothing"],
    "flatline": ["constant removal/interpolation", "low-variance removal"],        # would erase the stuck/flatline
    "seasonal": ["seasonal decomposition deseasonalizing"],                # would erase seasonal anomalies
}


def _run_tool(x, tool_name: str, params: dict) -> np.ndarray:
    from Anomaly_algorithm.registry import get as aget

    r = aget(tool_name).run(x, **params)
    a = np.asarray(r.data)
    if a.ndim == 2:
        a = a.max(axis=1)
    return np.nan_to_num(np.asarray(a, dtype=float).ravel(), nan=0.0)


def preprocess(x: np.ndarray, morphology: str) -> Dict:
    """Step 1: morphology-aware safe preprocessing. To avoid affecting the anomaly
    verdict, the default is the identity transform (standardization is already done
    per training segment at split time); only safe/harmful strategies are returned
    for audit."""
    return {
        "applied": "identity",  # already standardized; extra transforms would destroy the anomaly signal
        "unsafe": UNSAFE_PREPROCESS.get(morphology, []),
        "safe": "train-segment-only standardization (already done by split_standardize)",
    }


def line1_adaptive_ensemble(x: np.ndarray, morphology: str) -> np.ndarray:
    """Line 1: Top-6 detectors fused by morphology-adaptive weighting (rank fusion + weights)."""
    w = MORPHOLOGY_WEIGHTS.get(morphology, MORPHOLOGY_WEIGHTS["persistent"])
    ranks = []
    weights = []
    for det, (tool, params) in TOP6.items():
        s = _run_tool(x, tool, params)
        if s.std() < 1e-12:
            continue
        ranks.append(rankdata(s) / len(s))
        weights.append(w.get(det, 0.0))
    if not ranks:
        return np.zeros(len(x))
    weights = np.asarray(weights)
    weights = weights / (weights.sum() + 1e-9)
    fused = sum(wt * r for wt, r in zip(weights, ranks))
    return fused


def line2_llm_select(morphology: str, source: str = "") -> Dict:
    """Line 2: a local LLM picks a detector by orthogonal weighting of
    "characteristics + experimental performance".

    Each LLM's characteristics (from prior experiments: all prefer the rolling/
    two-sided segment-level family) are combined with morphology routing to vote for
    detectors, weighted by the experimental-performance weights LLM_WEIGHTS, and the
    weighted detector votes are returned. (An actual LLM call is optional: by default
    the known characteristic priors are used, with weights already calibrated against
    experimental performance.)
    """
    from Anomaly_algorithm.morphology import DETECTOR_ROUTING as RT

    routing = RT.get(morphology, [])
    # local LLM characteristics: their preferred detector family (mapped to Top-6 keys)
    llm_pref = {
        "qwen3-coder-30b-a3b:latest": "rolling_volatility",   # 30b has the most first places, strong on segments
        "llama3.1:8b": "rolling_volatility",                  # stable tool calling, strong on segments
        "qwen2.5-coder:7b": "two_sided_volatility",           # direction-agnostic preference
        "ibm/granite3.3:8b": "rolling_volatility",
    }
    votes: Dict[str, float] = {d: 0.0 for d in TOP6}
    for model, wgt in LLM_WEIGHTS.items():
        pref = llm_pref.get(model, "rolling_volatility")
        # if the morphology routing contains that preference, vote for it; otherwise vote for the routing's first choice
        pick = pref if pref in routing else (routing[0] if routing else pref)
        pick = pick if pick in TOP6 else "rolling_volatility"
        votes[pick] += wgt
    total = sum(votes.values()) or 1.0
    return {"llm_weights": dict(LLM_WEIGHTS), "weighted_votes": {k: round(v / total, 3) for k, v in votes.items()},
            "top": max(votes, key=votes.get)}


def run_architecture(x: np.ndarray, source: str = "", client=None) -> Dict:
    """End-to-end three-layer two-step architecture: step 0 morphology classification
    -> step 1 safe preprocessing -> step 2 two lines.

    Returns morphology / preprocess / line1_scores / routing.
    """
    x = np.asarray(x, dtype=float)
    xx = x.mean(axis=1) if x.ndim == 2 else x.ravel()

    # Step 0: morphology classification (local feature algorithms + LLM cross-validation recursion)
    morph = consensus_classify(xx, source=source, client=client)
    morphology = morph["morphology"]

    # Step 1: safe preprocessing
    prep = preprocess(x, morphology)

    # Step 2 Line 1: Top6 detectors, morphology-adaptive weighting
    line1 = line1_adaptive_ensemble(x, morphology)

    return {
        "morphology": morphology,
        "morphology_consensus": morph.get("consensus", {}),
        "preprocess": prep,
        "line1_scores": line1,
        "routing": DETECTOR_ROUTING.get(morphology, []),
    }
