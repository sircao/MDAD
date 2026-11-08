"""Improved rerun: add 3 direction-agnostic detectors and compare the 14-dataset mean AUC and ensemble effect."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Anomaly_algorithm.registry import get as aget  # noqa: E402
from Architecture_design.dataset import split_standardize, split_standardize_multi  # noqa: E402

FILES = {
    "IOPS": "TSB-AD-U/260_IOPS_id_1_WebService_tr_5000_1st_11804.csv",
    "MGAB": "TSB-AD-U/225_MGAB_id_1_Synthetic_tr_25000_1st_38478.csv",
    "NAB": "TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv",
    "NEK": "TSB-AD-U/278_NEK_id_2_WebService_tr_500_1st_64.csv",
    "Power": "TSB-AD-U/302_Power_id_1_Facility_tr_7685_1st_7785.csv",
    "SED": "TSB-AD-U/234_SED_id_1_Medical_tr_3740_1st_3840.csv",
    "TODS": "TSB-AD-U/287_TODS_id_1_Synthetic_tr_500_1st_11.csv",
    "UCR": "TSB-AD-U/303_UCR_id_1_Medical_tr_2499_1st_8000.csv",
    "YAHOO": "TSB-AD-U/551_YAHOO_id_1_Synthetic_tr_500_1st_893.csv",
    "MSL": "TSB-AD-M/002_MSL_id_1_Sensor_tr_500_1st_900.csv",
    "PSM": "TSB-AD-M/115_PSM_id_1_Facility_tr_50000_1st_129872.csv",
    "SMAP": "TSB-AD-M/144_SMAP_id_1_Sensor_tr_2052_1st_5300.csv",
    "SMD": "TSB-AD-M/057_SMD_id_1_Facility_tr_4529_1st_4629.csv",
    "SWaT": "TSB-AD-M/171_SWaT_id_1_Sensor_tr_3749_1st_9522.csv",
}
MUL = {"MSL", "PSM", "SMAP", "SMD", "SWaT"}

# 12 detectors: 9 original + 3 new
DETS = {
    "iforest": ("pyod", {"algorithm": "iforest"}),
    "knn": ("pyod", {"algorithm": "knn"}),
    "lof": ("pyod", {"algorithm": "lof"}),
    "tods_ma": ("tods", {"detector": "moving_average", "return_scores": True}),
    "tods_z": ("tods", {"detector": "zscore", "return_scores": True}),
    "roll_std100": ("rolling_volatility", {"stat": "std", "window": 100}),
    "roll_range100": ("rolling_volatility", {"stat": "range", "window": 100}),
    "flatline": ("__flat__", {"window": 100}),
    "two_sided_volatility": ("two_sided_volatility", {"window": 100}),
    "two_sided_knn": ("two_sided_knn", {"n_neighbors": 5}),
    "stuck": ("stuck", {"eps": 1e-3}),
}


def run(x, name):
    tool, params = DETS[name]
    if tool == "__flat__":
        r = aget("rolling_volatility").run(x, stat="std", window=100)
        return -np.asarray(r.data).ravel()
    r = aget(tool).run(x, **params)
    if not r.ok:
        raise RuntimeError(f"{name}: {r.error}")
    a = np.asarray(r.data)
    if hasattr(r.data, "iloc"):
        a = r.data.iloc[:, 0].to_numpy()
    a = np.asarray(a, float).ravel()
    return a.max(axis=1) if a.ndim == 2 else a


def main():
    auc = {d: {} for d in DETS}
    for src, f in FILES.items():
        dd = split_standardize_multi(f) if src in MUL else split_standardize(f)
        x, y = dd["test_std"], dd["y_test"]
        for d in DETS:
            s = run(x, d)[: len(y)]
            a = float(roc_auc_score(y, np.nan_to_num(np.asarray(s, float), nan=0.0))) if 0 < (y > 0).sum() < len(y) else 0.5
            auc[d][src] = a

    print("=== 14-dataset mean AUC per detector (improved, sorted by mean descending) ===")
    means = {d: float(np.mean([auc[d][s] for s in FILES])) for d in DETS}
    for d in sorted(means, key=means.get, reverse=True):
        tag = "  <new>" if d in ("two_sided_volatility", "two_sided_knn", "stuck") else ""
        print(f"  {d:24s} mean AUC={means[d]:.3f}{tag}")

    # mean of the original 9 detectors vs the new 3 detectors
    orig = ["iforest", "knn", "lof", "tods_ma", "tods_z", "roll_std100", "roll_range100", "flatline"]
    new = ["two_sided_volatility", "two_sided_knn", "stuck"]
    print(f"\n  best of original 8 detectors = {max(means[d] for d in orig):.3f}")
    print(f"  best of new 3 detectors = {max(means[d] for d in new):.3f}")

    # rank-max ensemble (original 9 panel vs 12 panel after adding the new 3)
    from scipy.stats import rankdata
    def ensemble(panel):
        per_src = []
        for src in FILES:
            dd = split_standardize_multi(FILES[src]) if src in MUL else split_standardize(FILES[src])
            x, y = dd["test_std"], dd["y_test"]
            R = [rankdata(np.nan_to_num(np.asarray(run(x, d), float), nan=0.0)[:len(y)]) / len(y) for d in panel]
            fused = np.max(R, axis=0)
            per_src.append(float(roc_auc_score(y, fused)) if 0 < (y > 0).sum() < len(y) else 0.5)
        return float(np.mean(per_src)), per_src

    m_orig, _ = ensemble(orig + ["stuck"] if False else orig)  # original 9 panel (8 detectors + flatline)
    m_new, _ = ensemble(list(DETS.keys()))
    print(f"\n=== rank-max ensemble mean AUC ===")
    print(f"  original 8-detector panel = {m_orig:.3f}")
    print(f"  12-detector panel (with two-sided + stuck) = {m_new:.3f}")

    # per dataset: direction-corrected best (upper-bound reference)
    best = []
    for src in FILES:
        vals = {d: max(auc[d][src], 1 - auc[d][src]) for d in DETS}
        best.append(max(vals.values()))
    print(f"  direction-corrected upper bound (reference) = {np.mean(best):.3f}")


if __name__ == "__main__":
    main()
