"""Run the detector panel of the "univariate architecture detection window"
unchanged on TSB-AD-M multivariate data to assess whether it is ideal.

Key mechanism: the behavior of the univariate-window detectors when given 2-D
input (rows=time, columns=variables) --
- rolling_volatility / two_sided_volatility / stuck: compute per column then
  max across columns (built-in max-pool)
- pyod(iforest/knn/lof/hbos): treat each time point as a channel-space vector,
  genuinely multivariate
- tods / merlion: return 2-D per column, then executor maxes across columns

This script uses the same detector panel as the univariate window and reports
ROC-AUC / AUC-PR / best-F1 (threshold-free) for each multivariate dataset, plus a
"single-channel oracle" reference that quantifies the loss from cross-column
max-pool: anomalies often land in only 1-2 channels, and max-pool also pulls in
random fluctuations from the remaining normal channels.

Usage::

    python Architecture_design/eval_multi_window.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Anomaly_algorithm.registry import get as aget  # noqa: E402
from Architecture_design.dataset import split_standardize_multi  # noqa: E402
from Architecture_design.metrics import evaluate  # noqa: E402

# detector panel aligned with the univariate-window executor._CANDIDATES / architecture.TOP6
DETS = {
    "rolling_std20": ("rolling_volatility", {"stat": "std", "window": 20}),
    "rolling_std50": ("rolling_volatility", {"stat": "std", "window": 50}),
    "rolling_std100": ("rolling_volatility", {"stat": "std", "window": 100}),
    "rolling_range100": ("rolling_volatility", {"stat": "range", "window": 100}),
    "tods_ma": ("tods", {"detector": "moving_average", "return_scores": True}),
    "iforest": ("pyod", {"algorithm": "iforest"}),
    "knn": ("pyod", {"algorithm": "knn"}),
    "lof": ("pyod", {"algorithm": "lof"}),
    "hbos": ("pyod", {"algorithm": "hbos"}),
    "merlion_ewma": ("merlion", {"detector": "ewma"}),
    "two_sided_volatility": ("two_sided_volatility", {"window": 100}),
    "two_sided_knn": ("two_sided_knn", {"n_neighbors": 5}),
    "stuck": ("stuck", {"eps": 1e-3}),
}

FILES = {
    "MSL": "TSB-AD-M/002_MSL_id_1_Sensor_tr_500_1st_900.csv",
    "SMAP": "TSB-AD-M/144_SMAP_id_1_Sensor_tr_2052_1st_5300.csv",
    "SMD": "TSB-AD-M/057_SMD_id_1_Facility_tr_4529_1st_4629.csv",
    "SWaT": "TSB-AD-M/171_SWaT_id_1_Sensor_tr_3749_1st_9522.csv",
    "PSM": "TSB-AD-M/115_PSM_id_1_Facility_tr_50000_1st_129872.csv",
}


def run_collapsed(x, name):
    """Same path as executor._detector_scores: 2-D results are max-collapsed across columns into 1-D."""
    tool, params = DETS[name]
    r = aget(tool).run(x, **params)
    if not r.ok:
        return None, r.error
    a = np.asarray(r.data)
    if hasattr(r.data, "iloc"):
        a = r.data.to_numpy()
    a = np.asarray(a, float)
    if a.ndim == 2:
        a = a.max(axis=1)
    return np.nan_to_num(a.ravel(), nan=0.0), None


def best_single_channel_auc(x, y):
    """Single-channel oracle: score rolling_volatility(std,100) per column and take
    the best channel's AUC.

    This is the "channel-aware" upper-bound reference: anomalies often land in only
    1-2 channels, so AUC on the correct channel is high, while cross-column max-pool
    also pulls in noise from the other normal channels.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(x)
    best = 0.0
    for c in df.columns:
        s = df[c].rolling(100, min_periods=50).std().to_numpy(dtype=float)
        s = np.nan_to_num(s, nan=0.0)[: len(y)]
        if s.std() < 1e-12 or not (0 < (y > 0).sum() < len(y)):
            continue
        try:
            a = float(roc_auc_score(y, s))
        except Exception:
            continue
        best = max(best, a, 1 - a)
    return best


def main() -> int:
    rows = {}
    for src, f in FILES.items():
        t0 = time.time()
        d = split_standardize_multi(str(ROOT / f))
        x, y = d["test_std"], d["y_test"]
        n_c = x.shape[1] if x.ndim == 2 else 1
        rows[src] = {"n_channels": n_c, "n_test": int(len(y)), "n_anom": int((y > 0).sum())}
        for name in DETS:
            s, err = run_collapsed(x, name)
            if s is None:
                rows[src][name] = {"err": err}
                continue
            rows[src][name] = evaluate(s[: len(y)], y[: len(s)])
        rows[src]["single_channel_oracle_auc"] = round(best_single_channel_auc(x, y), 3)
        rows[src]["secs"] = round(time.time() - t0, 1)
        print(f"[{src}] {n_c} channels / test {len(y)} points / anomaly {int((y>0).sum())} points, {rows[src]['secs']}s", flush=True)

    out = ROOT / "Architecture_design/runs/eval_multi_window.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Summary: each dataset's best detector (oracle over the panel) + single-channel oracle
    print("\n=== Summary (window panel after cross-column max-pool collapse) ===")
    print(f"{'Dataset':7s} {'Ch':>4s} {'Best AUC':>12s} {'1ch oracle':>14s} {'Best F1':>12s}")
    for src in FILES:
        best_auc = best_f1 = -1.0
        for name in DETS:
            v = rows[src].get(name)
            if not isinstance(v, dict) or "auc_roc" not in v:
                continue
            a = v["auc_roc"]
            if isinstance(a, (int, float)) and np.isfinite(a):
                best_auc = max(best_auc, a)
                best_f1 = max(best_f1, v.get("best_f1", 0.0) or 0.0)
        oc = rows[src].get("single_channel_oracle_auc", float("nan"))
        print(f"{src:6s} {rows[src]['n_channels']:4d} {best_auc:12.3f} {oc:14.3f} {best_f1:12.3f}")

    print("\nSAVED ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
