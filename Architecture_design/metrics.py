"""Anomaly detection evaluation: AUC-ROC / AUC-PR / best F1 / point-adjusted F1
(PA-F1, TSB-AD convention).

Evaluation is independent of the threshold the agent itself chooses: the AUC
metrics consume continuous scores directly, so the quality of the model's "which
detector to pick" decision can be quantified objectively.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _clean(scores, y):
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(s), len(y))
    s, y = s[:n], y[:n]
    finite = np.isfinite(s)
    fill = np.median(s[finite]) if finite.any() else 0.0
    s = np.where(finite, s, fill)
    return s, y


def best_f1(scores, y) -> Dict[str, float]:
    """Best F1 over a point-wise threshold sweep (a threshold-free upper bound on detection capability)."""
    s, y = _clean(scores, y)
    if s.std() < 1e-12:
        return {"best_f1": 0.0, "best_threshold": 0.0, "precision": 0.0, "recall": 0.0}
    best = (0.0, s.min(), 0.0, 0.0)
    for thr in np.linspace(s.min(), s.max(), 200):
        pred = s >= thr
        tp = float((pred & (y > 0)).sum())
        fp = float((pred & (y == 0)).sum())
        fn = float((~pred & (y > 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[0]:
            best = (f1, thr, prec, rec)
    return {
        "best_f1": round(best[0], 4),
        "best_threshold": round(float(best[1]), 4),
        "precision": round(best[2], 4),
        "recall": round(best[3], 4),
    }


def _segments(mask: np.ndarray) -> list:
    """Turn a boolean mask into consecutive intervals [(start, end), ...]."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    segs = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        if i == prev + 1:
            prev = int(i)
        else:
            segs.append((start, prev))
            start = prev = int(i)
    segs.append((start, prev))
    return segs


def point_adjusted_f1(scores, y, k: int = 0) -> Dict[str, float]:
    """Point-adjusted F1 per the TSB-AD convention (adjust first, then compute
    point-wise, keeping the metric in [0,1]).

    For each true anomaly segment, if the prediction hits any point within its +-k
    neighborhood the whole segment is marked as detected; precision/recall/F1 are
    then computed point-wise, best over a threshold sweep.
    """
    s, y = _clean(scores, y)
    n = len(s)
    true_segs = _segments(y > 0)
    if s.std() < 1e-12 or not true_segs:
        return {"pa_f1": 0.0, "pa_precision": 0.0, "pa_recall": 0.0}
    best = (0.0, 0.0, 0.0)
    for thr in np.linspace(s.min(), s.max(), 200):
        pred = s >= thr
        if not pred.any():
            continue
        pred_adj = pred.copy()
        for a, b in true_segs:
            lo, hi = max(0, a - k), min(n, b + k + 1)
            if pred_adj[lo:hi].any():
                pred_adj[a : b + 1] = True
        tp = float((pred_adj & (y > 0)).sum())
        fp = float((pred_adj & (y == 0)).sum())
        fn = float((~pred_adj & (y > 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[0]:
            best = (f1, prec, rec)
    return {
        "pa_f1": round(best[0], 4),
        "pa_precision": round(best[1], 4),
        "pa_recall": round(best[2], 4),
    }


def evaluate(scores, y) -> Dict[str, float]:
    """Aggregate all evaluation metrics (AUC family + best F1 + PA-F1)."""
    s, y = _clean(scores, y)
    out: Dict[str, float] = {}
    if 0 < y.sum() < len(y):
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            out["auc_roc"] = round(float(roc_auc_score(y, s)), 4)
            out["auc_pr"] = round(float(average_precision_score(y, s)), 4)
        except Exception:  # noqa: BLE001
            out["auc_roc"] = float("nan")
            out["auc_pr"] = float("nan")
    else:
        out["auc_roc"] = float("nan")
        out["auc_pr"] = float("nan")
    out.update(best_f1(s, y))
    out.update(point_adjusted_f1(s, y))
    return out
