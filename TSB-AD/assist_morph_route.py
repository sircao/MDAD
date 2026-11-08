"""assist_morph_route.py — Morphology-prior routing assist: best model + direction-matched morphology detector.

The two-level routing stays label-free: it never reads labels, only uses SOURCE_PRIOR (domain prior morphology) + DETECTOR_ROUTING
(morphology→detector family) to pick a direction-matched assistant, fused point-wise onto the Syntropic base to fix the base's misses.

Base    = Syntropic per-point scores (rank-normalized)
Assistant = the direction-matched detector from morphology_routed_scores(x, source) (rank-normalized)
Fusion    = max-union (union that raises recall to cover misses) / mean / raise-only

Reports per-file mean AUC: base alone / assistant alone / three fusion variants / oracle (per-file max).
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Anomaly_algorithm.morphology import morphology_routed_scores, SOURCE_PRIOR  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent
FILES = [
    '001_NAB_id_1_Facility_tr_1007_1st_2014.csv',
    '002_NAB_id_2_WebService_tr_1500_1st_4106.csv',
    '005_NAB_id_5_Traffic_tr_594_1st_1645.csv',
    '149_Stock_id_1_Finance_tr_500_1st_7.csv',
    '151_Stock_id_3_Finance_tr_500_1st_62.csv',
    '155_Stock_id_7_Finance_tr_500_1st_18.csv',
]


def source_of(f: str) -> str:
    return f.split("_")[1]


def rank(s: np.ndarray) -> np.ndarray:
    r = rankdata(s)
    return r / len(r)


def auc(score, label) -> float:
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=int)
    if len(np.unique(y)) < 2 or np.allclose(s, s[0]):
        return 0.5
    return float(roc_auc_score(y, s))


def main():
    syn = dict(np.load(ROOT / "fullscores_Syntropic.npz"))
    rows = defaultdict(lambda: defaultdict(list))
    print(f"{'file':50s} {'morph':10s} {'assistant':14s} {'base':>6s} {'comp':>6s} "
          f"{'max':>6s} {'mean':>6s} {'raise':>6s} {'oracle':>6s}")
    for f in FILES:
        src = source_of(f)
        df = pd.read_csv(f"Datasets/TSB-AD-U/{f}").dropna()
        x = df.iloc[:, 0:-1].values.astype(float)
        label = df["Label"].astype(int).to_numpy()

        base = rank(syn[f])
        r = morphology_routed_scores(x, source=src)
        comp = rank(r["scores"])
        morph, det = r["morphology"], r["detector"]

        fused_max = np.maximum(base, comp)
        fused_mean = 0.5 * (base + comp)
        fused_raise = base + 0.5 * np.clip(comp - base, 0, None)

        b = auc(base, label)
        c = auc(comp, label)
        mx = auc(fused_max, label)
        mn = auc(fused_mean, label)
        rs = auc(fused_raise, label)
        oc = max(b, c)
        for k, v in [("base", b), ("comp", c), ("max", mx), ("mean", mn),
                     ("raise", rs), ("oracle", oc)]:
            rows[src][k].append(v)
        print(f"{f:50s} {morph:10s} {det:14s} {b:6.3f} {c:6.3f} "
              f"{mx:6.3f} {mn:6.3f} {rs:6.3f} {oc:6.3f}", flush=True)

    print("\n==== per-source mean (AUC) ====")
    for src in sorted(rows):
        line = f"{src:8s}"
        for k in ["base", "comp", "max", "mean", "raise", "oracle"]:
            line += f" {k}={np.mean(rows[src][k]):.3f}"
        print(line)

    print("\n==== overall mean (AUC) ====")
    allv = defaultdict(list)
    for src in rows:
        for k, v in rows[src].items():
            allv[k] += v
    for k in ["base", "comp", "max", "mean", "raise", "oracle"]:
        print(f"  {k:8s} {np.mean(allv[k]):.3f}")


if __name__ == "__main__":
    main()
