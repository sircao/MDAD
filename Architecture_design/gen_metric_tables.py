"""Generate the four-metric TSB-AD tables: Affiliation-F / F1_T / Standard-F1 / VUS-PR.

One table per metric, rows = 9 detectors, columns = 14 datasets; within each
dataset column top1 red, top2 blue, top3 starred.

Metric definitions (aligned with the TSB-AD benchmark; see TSB-AD / "Elephant in
the room"):
- VUS-PR       : volume under the PR surface, computed as average precision;
- Affiliation-F: harmonic mean of Tatbul et al.'s affiliation precision/recall,
                 best over a threshold sweep;
- Standard-F1  : point-wise F1, best over a threshold sweep (no point adjustment);
- F1_T         : point-adjusted F1 with tolerance window T=100 (TSB-AD's customary
                 point adjustment).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Anomaly_algorithm.registry import get as aget  # noqa: E402
from Architecture_design.dataset import split_standardize, split_standardize_multi  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

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
UNI = ["IOPS", "MGAB", "NAB", "NEK", "Power", "SED", "TODS", "UCR", "YAHOO"]
MUL = ["MSL", "PSM", "SMAP", "SMD", "SWaT"]
ALL = UNI + MUL

DETECTORS = {
    "iforest": ("pyod", {"algorithm": "iforest"}),
    "knn": ("pyod", {"algorithm": "knn"}),
    "lof": ("pyod", {"algorithm": "lof"}),
    "tods_ma": ("tods", {"detector": "moving_average", "return_scores": True}),
    "tods_z": ("tods", {"detector": "zscore", "return_scores": True}),
    "roll_std50": ("rolling_volatility", {"stat": "std", "window": 50}),
    "roll_std100": ("rolling_volatility", {"stat": "std", "window": 100}),
    "roll_range100": ("rolling_volatility", {"stat": "range", "window": 100}),
    "flatline": ("__flat__", {"window": 100}),
    "two_sided_volatility": ("two_sided_volatility", {"window": 100}),
    "two_sided_knn": ("two_sided_knn", {"n_neighbors": 5}),
    "stuck": ("stuck", {"eps": 1e-3}),
}


def run_detector(x, name):
    tool, params = DETECTORS[name]
    if tool == "__flat__":
        r = aget("rolling_volatility").run(x, stat="std", window=params["window"])
        return -np.asarray(r.data).ravel()
    r = aget(tool).run(x, **params)
    a = np.asarray(r.data)
    if hasattr(r.data, "iloc"):
        a = r.data.iloc[:, 0].to_numpy()
    a = np.asarray(a, float).ravel()
    return a.max(axis=1) if a.ndim == 2 else a


def _segments(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    segs, start, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        if i == prev + 1:
            prev = int(i)
        else:
            segs.append((start, prev)); start = prev = int(i)
    segs.append((start, prev))
    return segs


def standard_f1(s, y):
    """Point-wise F1, best over a threshold sweep."""
    s = np.nan_to_num(np.asarray(s, float), nan=0.0)
    if s.std() < 1e-12:
        return 0.0
    best = 0.0
    for thr in _thresholds(s, 100):
        pred = s >= thr
        tp = float((pred & (y > 0)).sum())
        fp = float((pred & (y == 0)).sum())
        fn = float((~pred & (y > 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        best = max(best, f1)
    return best


def f1_t(s, y, T=100):
    """Point-adjusted F1 (tolerance T): a prediction touching a true anomaly segment within its +-T neighborhood marks the whole segment as detected; best over a threshold sweep."""
    s = np.nan_to_num(np.asarray(s, float), nan=0.0)
    n = len(s)
    true_segs = _segments(y > 0)
    if s.std() < 1e-12 or not true_segs:
        return 0.0
    best = 0.0
    for thr in _thresholds(s, 100):
        pred = s >= thr
        if not pred.any():
            continue
        adj = pred.copy()
        for a, b in true_segs:
            lo, hi = max(0, a - T), min(n, b + T + 1)
            if adj[lo:hi].any():
                adj[a:b + 1] = True
        tp = float((adj & (y > 0)).sum())
        fp = float((adj & (y == 0)).sum())
        fn = float((~adj & (y > 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        best = max(best, f1)
    return best


def _nearest_dist(pts, refs):
    """Distance from each point in pts to the nearest point in refs (vectorized; both already sorted)."""
    if len(pts) == 0 or len(refs) == 0:
        return np.full(len(pts), np.inf)
    j = np.clip(np.searchsorted(refs, pts), 0, len(refs) - 1)
    left = np.abs(pts - refs[np.maximum(j - 1, 0)])
    right = np.abs(pts - refs[j])
    return np.minimum(left, right)


def _thresholds(s, n=100):
    """Take n thresholds by quantile, covering the whole score distribution."""
    return np.unique(np.quantile(s, np.linspace(0, 1, n)))


def affiliation_f(s, y):
    """Tatbul et al.'s affiliation F, best over a threshold sweep."""
    s = np.nan_to_num(np.asarray(s, float), nan=0.0)
    gt = y > 0
    gt_idx = np.where(gt)[0]
    if s.std() < 1e-12 or len(gt_idx) == 0:
        return 0.0
    gaps = np.diff(gt_idx)
    gamma = float(gaps.mean()) if len(gaps) > 0 else 1.0
    if gamma <= 0:
        gamma = 1.0
    best = 0.0
    for thr in _thresholds(s, 50):
        pred_idx = np.where(s >= thr)[0]
        if len(pred_idx) == 0:
            continue
        dp = _nearest_dist(pred_idx, gt_idx)
        local_p = np.maximum(0.0, 1.0 - dp / gamma).mean()
        dr = _nearest_dist(gt_idx, pred_idx)
        local_r = np.maximum(0.0, 1.0 - dr / gamma).mean()
        f = 2 * local_p * local_r / (local_p + local_r) if local_p + local_r else 0.0
        best = max(best, float(f))
    return best


def vus_pr(s, y):
    """Volume under the PR surface, computed as average precision."""
    s = np.nan_to_num(np.asarray(s, float), nan=0.0)
    if 0 < (y > 0).sum() < len(y):
        return float(average_precision_score(y, s))
    return 0.0


METRICS = {
    "Affiliation-F": affiliation_f,
    "F1_T": f1_t,
    "Standard-F1": standard_f1,
    "VUS-PR": vus_pr,
}


def fmt_cell(v, rank):
    if rank == 1:
        return f"\\textcolor{{red}}{{\\textbf{{{v:.3f}}}}}"
    if rank == 2:
        return f"\\textcolor{{blue}}{{{v:.3f}}}"
    if rank == 3:
        return f"{v:.3f}" + "${}^{*}$"
    return f"{v:.3f}"


def main():
    # compute raw scores
    scores = {}  # src -> {det -> scores}
    for src, f in FILES.items():
        multi = src in MUL
        d = split_standardize_multi(f) if multi else split_standardize(f)
        x, y = d["test_std"], d["y_test"]
        scores[src] = {"y": y}
        for name in DETECTORS:
            scores[src][name] = run_detector(x, name)[: len(y)]

    # compute the four metrics
    res = {m: {det: {} for det in DETECTORS} for m in METRICS}
    for m, fn in METRICS.items():
        for src in ALL:
            y = scores[src]["y"]
            for det in DETECTORS:
                res[m][det][src] = fn(scores[src][det], y)

    # generate 4 tables
    NL = "\n"
    blocks = []
    for m in METRICS:
        L = []
        L.append("\\begin{table*}[t]")
        L.append("\\centering")
        L.append("\\footnotesize")
        L.append("\\setlength{\\tabcolsep}{4pt}")
        L.append("\\resizebox{\\textwidth}{!}{%")
        L.append("\\begin{tabular}{@{}l ccccccccc ccccc@{}}")
        L.append("\\toprule")
        L.append("\\multirow{2}{*}{\\textbf{Detector}} & "
                 "\\multicolumn{9}{c}{\\textbf{Univariate Datasets}} & "
                 "\\multicolumn{5}{c}{\\textbf{Multivariate Datasets}} \\\\")
        L.append("\\cmidrule(lr){2-10} \\cmidrule(lr){11-15}")
        L.append("& " + " & ".join(f"\\textbf{{{s}}}" for s in ALL) + " \\\\")
        L.append("\\midrule")
        for det in DETECTORS:
            cells = []
            for src in ALL:
                # ranking within this column
                col_vals = sorted([(d2, res[m][d2][src]) for d2 in DETECTORS], key=lambda t: -t[1])
                rank = [i for i, (d2, v) in enumerate(col_vals) if d2 == det][0] + 1
                cells.append(fmt_cell(res[m][det][src], rank))
            L.append(f"{det} & " + " & ".join(cells) + " \\\\")
        L.append("\\bottomrule")
        L.append("\\end{tabular}%")
        L.append("}")
        L.append(f"\\caption{{{m} of nine unsupervised detectors across fourteen TSB-AD datasets "
                 f"(one representative series per source). For each dataset, the best value is in "
                 f"bold red, the second best in blue, and the third best is marked with an asterisk.}}")
        L.append(f"\\label{{tab:{m.lower().replace('-', '_')}}}")
        L.append("\\end{table*}")
        blocks.append(NL.join(L))

    out = NL.join(blocks) + NL
    (ROOT / "paper/metric_tables.tex").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
