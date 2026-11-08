"""Anomaly morphology classification: locate first, then classify, then route to a detector direction.

Core difficulty: anomalies are **local** events, and global raw statistics
(kurtosis/acf1/flat_ratio) are dominated by the normal background, so they cannot
discriminate morphology. This module therefore first uses a direction-agnostic
two-sided volatility (|rolling std - median|) to **locate** the anomalous segment,
then extracts local features for "anomalous segment vs background" to judge
morphology and direction, and finally routes to the direction-matching detector
family -- avoiding phase inversion of single-direction detectors at the root.

This module is reused by tools/023_morphology_features and 024_morphology_classify.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import kendalltau, kurtosis as _kurt, skew as _skew

# Common industrial TSAD anomaly morphologies (9 classes).
MORPHOLOGY_CLASSES = [
    "spike",           # spike / point anomaly (isolated extreme point)
    "dense_cluster",   # dense cluster (dense value range, temporally scattered point anomalies)
    "level_shift",     # level shift / step change
    "trend",           # trend drift
    "variance_change", # variance change
    "noise",           # high-frequency noise
    "persistent",      # persistent / collective anomaly
    "flatline",        # flatline / low volatility
    "seasonal",        # seasonal anomaly
]

# morphology -> direction-matching detector family (for Architecture_design scheduling).
DETECTOR_ROUTING = {
    "spike": ["pyod/iforest", "pyod/knn", "tods/zscore", "two_sided_knn"],
    "dense_cluster": ["pyod/iforest", "two_sided_knn"],
    "level_shift": ["rolling_volatility", "tods/moving_average", "two_sided_volatility"],
    "trend": ["rolling_volatility", "tods/moving_average"],
    "variance_change": ["two_sided_volatility", "rolling_volatility"],
    "noise": ["tods/moving_average", "merlion/ewma"],
    "persistent": ["rolling_volatility", "two_sided_volatility"],
    "flatline": ["two_sided_knn", "flatline", "stuck"],
    "seasonal": ["tods/moving_average", "merlion/ewma"],
}

# source type -> morphology prior (the domain-prior path in the paper's Discussion:
# finance->point, sensor->segmental/spike, medical->flatline). Calibrated on 14
# representative sources by optimal detector direction.
SOURCE_PRIOR = {
    "NAB": "level_shift", "IOPS": "level_shift", "YAHOO": "spike",
    "Stock": "spike", "TODS": "dense_cluster", "UCR": "flatline",
    "SMD": "level_shift", "MSL": "level_shift", "SMAP": "level_shift",
    "MGAB": "level_shift", "NEK": "variance_change", "Power": "variance_change",
    "SED": "spike", "PSM": "variance_change", "SWaT": "variance_change",
}

# routing string -> (registered tool name, parameters). Lets morphology routing map directly to a detector.
ROUTING_TO_TOOL = {
    "pyod/iforest": ("pyod", {"algorithm": "iforest"}),
    "pyod/knn": ("pyod", {"algorithm": "knn"}),
    "tods/zscore": ("tods", {"detector": "zscore", "return_scores": True}),
    "tods/moving_average": ("tods", {"detector": "moving_average", "return_scores": True}),
    "merlion/ewma": ("merlion", {"detector": "ewma"}),
    "rolling_volatility": ("rolling_volatility", {"stat": "std", "window": 100}),
    "two_sided_volatility": ("two_sided_volatility", {"window": 100}),
    "two_sided_knn": ("two_sided_knn", {"n_neighbors": 5}),
    "stuck": ("stuck", {"eps": 1e-3}),
    "flatline": ("stuck", {"eps": 1e-3}),
}


def _acf(x: np.ndarray, lag: int = 1) -> float:
    if len(x) <= lag or np.std(x) < 1e-12:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = np.std(x[max(0, i - w): i + 1])
    return out


# ---------------------------------------------------------------------------
# Global features (for the morphology_features tool + LLM path reference)
# ---------------------------------------------------------------------------
def extract_features(x: np.ndarray, window: int = 0) -> Dict[str, float]:
    """Extract label-free global signal features from a 1-D raw series (statistical/time/frequency/morphological)."""
    x = np.asarray(x, dtype=float).ravel()
    x = np.nan_to_num(x, nan=0.0)
    n = len(x)
    s = float(np.std(x))
    eps = 1e-12
    f: Dict[str, float] = {}

    if s > eps and n > 3:
        f["kurtosis"] = float(_kurt(x))
        f["skewness"] = float(_skew(x))
    else:
        f["kurtosis"] = 0.0
        f["skewness"] = 0.0
    mx = float(np.max(np.abs(x)))
    f["max_abs"] = mx
    f["max_over_std"] = mx / s if s > eps else 0.0
    f["acf1"] = _acf(x, 1)

    if s > eps and n > 3:
        t = np.arange(n, dtype=float)
        slope = float(np.polyfit(t, x, 1)[0])
        f["slope"] = slope * n / s
        f["cusum"] = float(np.max(np.abs(np.cumsum(x - np.mean(x))))) / (s * np.sqrt(n))
        tau = kendalltau(t, x).statistic
        f["mk_tau"] = float(tau) if not np.isnan(tau) else 0.0
    else:
        f["slope"] = 0.0
        f["cusum"] = 0.0
        f["mk_tau"] = 0.0

    w = window or max(10, n // 20)
    w = max(2, min(int(w), max(n, 2)))
    rstd = _rolling_std(x, w)
    f["roll_std_mean"] = float(np.mean(rstd)) / s if s > eps else 0.0
    f["roll_std_cv"] = float(np.std(rstd)) / (float(np.mean(rstd)) + eps)

    thr = 1e-3 * (s if s > eps else 1.0)
    const = np.abs(np.diff(x)) <= thr
    f["flat_ratio"] = float(np.mean(const))
    f["flat_run"] = float(_longest_run(const))

    if n > 8:
        spec = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
        tail = spec[1:].sum()
        if tail > eps:
            half = max(1, len(spec) // 2)
            f["hfr"] = float(spec[half:].sum() / tail)
            f["spec_peak"] = float(np.max(spec[1:]) / (np.mean(spec[1:]) + eps))
        else:
            f["hfr"] = 0.0
            f["spec_peak"] = 0.0
    else:
        f["hfr"] = 0.0
        f["spec_peak"] = 0.0
    f["zero_cross_rate"] = float(np.mean(np.diff(np.sign(x)) != 0))

    q = float(np.quantile(np.abs(x), 0.99))
    f["extreme_run"] = float(_longest_run(np.abs(x) >= q))
    return f


# ---------------------------------------------------------------------------
# Multi-probe: residual / short-window variance / long-window variance / value-range density,
# each locating the segment + judging direction
# ---------------------------------------------------------------------------
def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = np.mean(x[max(0, i - w): i + 1])
    return out


def _knn_dist(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Value-range KNN distance: the mean distance of each point to its k nearest neighbors (small distance = dense cluster, large = isolated point)."""
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(x, dtype=float).reshape(-1, 1)
    kk = max(2, min(int(k) + 1, len(x)))
    nn = NearestNeighbors(n_neighbors=kk).fit(X)
    d, _ = nn.kneighbors(X)
    return d[:, 1:].mean(axis=1)


def _region_stats(x: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    r = x[mask]
    b = x[~mask]
    s_bg = float(np.std(b)) if len(b) > 1 else 1.0
    out: Dict[str, float] = {}
    out["region_len"] = float(len(r))
    out["region_frac"] = float(len(r) / len(x))
    out["std_ratio"] = float(np.std(r)) / s_bg if s_bg > 1e-12 else 0.0
    out["mean_shift"] = abs(float(np.mean(r)) - float(np.mean(b))) / s_bg if s_bg > 1e-12 else 0.0
    out["peak"] = float(np.max(np.abs(r - np.mean(b)))) / s_bg if s_bg > 1e-12 else 0.0
    out["flat_ratio"] = float(np.mean(np.abs(np.diff(r)) <= 1e-3 * s_bg)) if len(r) > 1 else 0.0
    out["acf1"] = _acf(r, 1)
    return out


def classify(x: np.ndarray, window: int = 0, source: str = "") -> Dict:
    """Multi-probe classification: first judge direction without labels (sustained deviation / dense cluster / isolated point), optionally overlaid with a source-type prior.

    The source-type prior comes from SOURCE_PRIOR and breaks through label-free
    morphology confusion (the paper's domain-prior path); without a source it falls
    back to a purely label-free criterion.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = np.nan_to_num(x, nan=0.0)
    n = len(x)
    w_short = max(5, min(20, max(5, n // 20)))
    w_long = window or max(10, n // 20)
    w_long = max(5, min(int(w_long), max(n, 5)))
    w_ma = max(3, min(w_short, 10))

    vol_s = _rolling_std(x, w_short)
    resid = np.abs(x - _rolling_mean(x, w_ma))
    med_res = float(np.median(resid)) + 1e-9
    dens = _knn_dist(x, k=5)
    med_dens = float(np.median(dens))

    # -- label-free criteria --
    dev_mask = resid > 3 * med_res
    res_run = _longest_run(dev_mask)             # sustained deviation segment length
    res_ratio = float(np.max(resid) / med_res)   # deviation strength
    dens_mask = dens < np.quantile(dens, 0.01)
    dense_strength = (med_dens - float(np.quantile(dens, 0.01))) / (med_dens + 1e-9)
    dense_val_shift = abs(float(np.mean(x[dens_mask])) - float(np.median(x))) / (float(np.std(x)) + 1e-12) if int(dens_mask.sum()) >= 2 else 0.0

    f = extract_features(x, w_long)

    morph: str
    evidence: List[str]
    region_mask: np.ndarray
    if res_run >= max(10, 2 * w_ma):
        region_mask = dev_mask
        rs = _region_stats(x, dev_mask)
        if rs["mean_shift"] > 1.0 and rs["std_ratio"] < 2.0:
            morph, evidence = "level_shift", ["sustained deviation with mean shift"]
        else:
            morph, evidence = "variance_change", ["sustained deviation (high-variance segment)"]
    elif dense_strength > 0.6 and dense_val_shift > 0.5:
        # Dense value range and off-center = dense cluster / flatline (same detector origin; both route to two_sided_knn)
        region_mask = dens_mask
        morph, evidence = "dense_cluster", ["dense value range and off-center"]
    elif res_ratio > 4:
        region_mask = dev_mask
        morph, evidence = "spike", ["transient deviation (isolated point)"]
    elif abs(f["slope"]) > 1.5:
        region_mask = np.ones(n, dtype=bool)
        morph, evidence = "trend", ["large global slope"]
    elif f["hfr"] > 0.5 and f["acf1"] < 0.2:
        region_mask = np.ones(n, dtype=bool)
        morph, evidence = "noise", ["global high frequency"]
    elif f["spec_peak"] > 10:
        region_mask = np.ones(n, dtype=bool)
        morph, evidence = "seasonal", ["strong spectral peak"]
    else:
        region_mask = np.ones(n, dtype=bool)
        morph, evidence = "persistent", ["default persistent (no strong directional signal)"]

    # -- source-type prior override (robust fallback when label-free is confused) --
    prior = SOURCE_PRIOR.get(source, "") if source else ""
    used_prior = bool(prior)
    if used_prior and morph != prior:
        evidence.append(f"source prior [{source}->{prior}] overrides label-free criterion")
        morph = prior
        region_mask = dev_mask if int(dev_mask.sum()) >= 2 else dens_mask

    lead = {
        "level_shift": res_run, "variance_change": res_run, "flatline": res_run,
        "dense_cluster": dense_strength, "spike": res_ratio, "trend": abs(f["slope"]),
        "noise": f["hfr"], "seasonal": f["spec_peak"], "persistent": 1.0,
    }[morph]
    confidence = "high" if (used_prior or lead >= (20 if morph in ("level_shift", "variance_change", "flatline") else 1.5)) else ("medium" if lead >= 1.0 else "low")

    region = _region_stats(x, region_mask) if int(region_mask.sum()) >= 2 and int(region_mask.sum()) < n else None
    return {
        "morphology": morph,
        "confidence": confidence,
        "evidence": evidence,
        "used_prior": used_prior,
        "signals": {
            "res_run": int(res_run), "res_ratio": round(res_ratio, 1),
            "dense_strength": round(dense_strength, 2), "dense_val_shift": round(dense_val_shift, 2),
        },
        "ranked": [{"morphology": morph, "evidence": evidence, "n_evidence": len(evidence)}],
        "region": region,
        "features": f,
        "routing": DETECTOR_ROUTING[morph],
        "routing_fallback": DETECTOR_ROUTING[morph],
    }


# ---------------------------------------------------------------------------
# Morphology consensus recursion: local rule classification + LLM agent in parallel; on disagreement, recurse to converge
# ---------------------------------------------------------------------------
def _build_llm_prompt(local: Dict, source: str, history: List[Dict], show_local: bool = True) -> str:
    f = local["features"]
    sig = local["signals"]
    lines = [
        "You are an expert in industrial time-series anomaly morphology classification. Given the morphology features of a standardized series, determine its anomaly morphology.",
        f"Data source type: {source or 'unknown'} (usable as a domain-prior reference, e.g. UCR is mostly flatline, Stock/TODS mostly point/dense, NAB/IOPS mostly drift, but please rely on the feature evidence).",
    ]
    if show_local:
        lines.append(f"Local rule classification: {local['morphology']} (evidence: {local.get('evidence', [])})")
    lines += [
        "",
        "Label-free morphology features:",
        f"kurtosis={f['kurtosis']:.2f} skew={f['skewness']:.2f} acf1={f['acf1']:.2f} "
        f"max/σ={f['max_over_std']:.2f} cusum={f['cusum']:.2f} slope={f['slope']:.2f}",
        f"roll_std_mean={f['roll_std_mean']:.2f} roll_std_cv={f['roll_std_cv']:.2f} "
        f"flat_ratio={f['flat_ratio']:.2f} flat_run={f['flat_run']:.0f}",
        f"high-freq energy ratio={f['hfr']:.2f} spectral peak={f['spec_peak']:.1f} zero-crossing rate={f['zero_cross_rate']:.2f}",
        f"Local signals: sustained-deviation run={sig.get('res_run')} deviation strength={sig.get('res_ratio')} "
        f"value-range density={sig.get('dense_strength')} dense-value offset={sig.get('dense_val_shift')}",
        "",
        "Optional morphologies (9 classes) and criteria: spike=isolated extreme point; dense_cluster=dense value range and off-center; "
        "level_shift=sustained mean shift; variance_change=sustained high-variance segment; flatline=low variance/stuck; "
        "trend=trend drift; noise=high-frequency noise; persistent=long sustained segment; seasonal=periodic anomaly.",
    ]
    if history:
        last = history[-1]
        lines += [
            "",
            f"Previous-round disagreement: local judged '{last['local']}', you judged '{last['llm']}'."
            "Please check the local evidence item by item, explicitly agree or rebut, and try to converge.",
        ]
    lines += ["", "Output only one JSON: {\"morphology\":\"morphology name\",\"reasoning\":\"one sentence\",\"confidence\":\"high/medium/low\"}"]
    return "\n".join(lines)


def _ask_llm(x: np.ndarray, source: str, local: Dict, client, history: List[Dict],
             show_local: bool = True) -> Dict:
    prompt = _build_llm_prompt(local, source, history, show_local=show_local)
    reply = client.chat(
        [{"role": "system", "content": "You are an anomaly morphology classification expert; output only one JSON."},
         {"role": "user", "content": prompt}]
    )
    content = reply.get("content", "") or ""
    m = re.search(r"\{.*\}", content, re.S)
    data = json.loads(m.group(0)) if m else {}
    morph = str(data.get("morphology", "persistent"))
    if morph not in MORPHOLOGY_CLASSES:
        morph = local["morphology"]
    _CONF = {"high": "high", "medium": "medium", "low": "low"}
    confidence = _CONF.get(str(data.get("confidence", "medium")).strip().lower(), "medium")
    return {
        "morphology": morph,
        "reasoning": data.get("reasoning", ""),
        "confidence": confidence,
    }


def _cross_verify_once(x: np.ndarray, source: str, client, max_iter: int, windows, model: str) -> Tuple[Dict, bool]:
    """Run one round of cross-validation with a single LLM model: "LLM independent judgment -> local check -> recursive convergence".

    Returns (result, stable). stable=True means converged or the LLM is highly confident, so a conclusion can be drawn;
    otherwise a larger model must be escalated to continue.
    """
    local = classify(x, window=windows[0])
    history: List[Dict] = []
    for it in range(max_iter):
        # Round 1: LLM judges independently (without seeing local, to avoid anchoring); afterwards cross-check against local evidence
        llm = _ask_llm(x, source, local, client, history, show_local=(it > 0))
        history.append({
            "iter": it + 1, "local": local["morphology"], "local_evidence": local["evidence"],
            "llm": llm["morphology"], "llm_reasoning": llm["reasoning"], "llm_confidence": llm["confidence"],
        })
        if llm["morphology"] == local["morphology"]:
            break
        # Disagreement: re-run local with a different window at multiple scales; the next LLM round sees new evidence + the disagreement
        next_w = windows[min(it + 1, len(windows) - 1)]
        local = classify(x, window=next_w)

    last = history[-1]
    converged = last["local"] == last["llm"]
    # LLM is the decision maker, local is the validator; on disagreement the LLM's final judgment wins
    morph = last["llm"]
    # stable = cross-validation converged and not low confidence; otherwise treated as unstable (triggers model escalation / source-prior fallback)
    stable = converged and last["llm_confidence"] != "low"

    final = classify(x, window=windows[0])
    final["morphology"] = morph
    final["routing"] = DETECTOR_ROUTING[morph]
    final["routing_fallback"] = DETECTOR_ROUTING[morph]
    final["consensus"] = {
        "converged": converged, "stable": stable, "iterations": len(history),
        "model": model, "history": history,
    }
    return final, stable


def consensus_classify(x: np.ndarray, source: str = "", client=None, max_iter: int = 3,
                       windows=(0, 10, 30, 60),
                       models=("llama3.1:8b", "qwen3-coder-30b-a3b:latest")) -> Dict:
    """Model-escalation recursion: local rules + LLM cross-validation; if the morphology is unstable, escalate to a larger model until a result is produced.

    First run cross-validation with a small model; if it does not converge and the LLM is low-confidence (unstable), escalate to the next larger model
    and continue recursing until convergence/high confidence or the models are exhausted. ``client`` carries a mutable ``model`` attribute
    (e.g. OllamaClient) and switches model by model; when None, only local runs.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = np.nan_to_num(x, nan=0.0)
    if client is None:
        # No LLM: cross-validation cannot run; fall back directly to the source prior (without a source, degrade to pure label-free)
        local = classify(x, window=windows[0], source=source)
        local["consensus"] = {"converged": True, "stable": True, "method": "local-only",
                              "fallback": "source_prior" if (source and source in SOURCE_PRIOR) else None,
                              "model": None, "history": []}
        return local

    last: Dict = {}
    for model in models:
        try:
            client.model = model
        except Exception:  # noqa: BLE001 -- clients that cannot change model reuse the same model
            pass
        result, stable = _cross_verify_once(x, source, client, max_iter, windows, model=model)
        last = result
        if stable:
            return result
        # unstable -> escalate to a larger model and continue recursing

    # All models failed to converge / low confidence -> source-prior fallback (used only here, balancing cross-validation + stability)
    if source and source in SOURCE_PRIOR:
        morph = SOURCE_PRIOR[source]
        final = classify(x, window=windows[0], source=source)
        final["morphology"] = morph
        final["routing"] = DETECTOR_ROUTING[morph]
        final["routing_fallback"] = DETECTOR_ROUTING[morph]
        final["consensus"] = {
            "converged": False, "stable": True, "fallback": "source_prior",
            "model": last.get("consensus", {}).get("model"),
            "history": last.get("consensus", {}).get("history", []),
        }
        return final
    last["consensus"]["escalated"] = True
    return last


def morphology_routed_scores(x, source: str = "", client=None, window: int = 0, max_iter: int = 3) -> Dict:
    """End-to-end morphology routing: consensus classification (local + LLM recursion) -> matching detector -> continuous anomaly scores.

    This is the landing entry for "classify morphology first, then match a detector", callable directly by the Architecture_design agent's
    step 0 or by external scripts; it returns the morphology, the hit detector, and pointwise scores.
    """
    x = np.asarray(x, dtype=float)
    xx = x.mean(axis=1) if x.ndim == 2 else x.ravel()
    c = consensus_classify(xx, source=source, client=client, max_iter=max_iter,
                           windows=(window, 10, 30, 60))
    tool_name, args = ROUTING_TO_TOOL.get(
        c["routing"][0], ("rolling_volatility", {"stat": "std", "window": 100}))
    from Anomaly_algorithm.registry import get

    r = get(tool_name).run(x, **args)
    if not r.ok:  # fall back to rolling volatility
        r = get("rolling_volatility").run(x, stat="std", window=100)
        tool_name, args = "rolling_volatility", {"stat": "std", "window": 100}
    scores = np.asarray(r.data)
    if scores.ndim == 2:
        scores = scores.max(axis=1)
    return {
        "morphology": c["morphology"],
        "confidence": c["confidence"],
        "detector": tool_name,
        "args": args,
        "scores": np.asarray(scores, dtype=float).ravel(),
    }
