"""Leaderboard simulator for prior routing vs single model vs oracle.

Encode the per-source Affiliation-F matrix from the official TSB-AD leaderboard and answer:
  1. what the best single model is (the floor to beat);
  2. what per-source best (oracle routing) is (the ceiling);
  3. how far routing via "source prior → morphology → SOTA model" (SOURCE_PRIOR, from
     Anomaly_algorithm/morphology.py) can go (a purely label-free, deployable prior routing);
  4. how much each source's prior routing is short, and where — this is the space for multi-agent + Pareto to fill.
"""
from __future__ import annotations
from typing import Dict, List

SRC_UNI = ["IOPS", "MGAB", "NAB", "NEK", "Power", "SED", "Stock", "TODS", "UCR", "WSD", "YAHOO"]
SRC_MUL = ["MSL", "PSM", "SMAP", "SMD", "SWaT"]
SOURCES = SRC_UNI + SRC_MUL

# official leaderboard Affiliation-F (%), model -> 16 sources (11 univariate + 5 multivariate).
AFF: Dict[str, List[float]] = {
    "TimeRCD":   [86.98, 74.54, 93.81, 83.95, 85.02, 88.99, 41.31, 84.86, 84.34, 91.00, 95.94, 84.66, 82.31, 86.01, 91.75, 69.43],
    "DADA":      [89.37, 67.66, 86.56, 95.40, 69.79, 65.18, 98.77, 76.89, 72.21, 93.92, 92.20, 76.57, 81.27, 76.92, 83.74, 76.18],
    "MOMENT":    [87.54, 66.76, 90.45, 92.26, 75.97, 59.13, 45.26, 59.76, 75.77, 95.39, 79.99, 74.55, 65.79, 77.42, 74.00, 70.17],
    "TimesFM":   [81.88, 66.95, 79.73, 90.49, 69.88, 67.14, 97.53, 89.08, 70.03, 78.97, 91.28, 20.35, 71.24, 45.44, 62.85, 44.37],
    "Chronos":   [90.12, 67.89, 86.66, 93.63, 69.72, 67.89, 96.85, 91.96, 74.35, 90.98, 96.34, 75.52, 70.88, 72.22, 75.31, 70.43],
    "TimeMOE":   [76.34, 67.23, 80.51, 80.50, 71.19, 60.98, 63.28, 54.68, 73.56, 80.25, 69.70, 69.85, 54.74, 74.38, 69.97, 64.37],
    "MovingVar": [83.90, 67.56, 91.50, 77.81, 74.77, 67.14, 67.96, 71.57, 78.44, 89.13, 78.00, 89.43, 73.91, 88.14, 81.33, 75.55],
    "TranAD":    [83.19, 67.28, 90.28, 85.02, 71.56, 61.03, 57.94, 52.76, 73.31, 84.34, 76.08, 79.91, 73.83, 87.39, 92.20, 75.37],
    "USAD":      [71.08, 67.81, 91.54, 71.13, 76.48, 55.60, 35.92, 47.90, 76.00, 65.10, 53.05, 81.86, 57.86, 87.25, 85.09, 75.06],
    "OmniAnomaly": [80.32, 67.35, 92.35, 86.30, 78.16, 61.26, 75.24, 50.73, 73.53, 78.02, 71.31, 83.15, 58.17, 91.38, 85.82, 73.39],
    "LOF":       [81.06, 68.44, 75.75, 84.74, 66.76, 63.85, 69.74, 60.58, 73.53, 81.29, 75.63, 84.35, 61.98, 63.32, 64.13, 56.34],
    "IForest":   [52.81, 68.82, 39.84, 71.15, 0.00, 70.09, 0.06, 44.17, 50.56, 41.24, 33.30, 63.36, 63.78, 59.96, 69.71, 0.00],
    "Sub-PCA":   [75.39, 66.90, 89.29, 97.10, 71.37, 67.14, 70.63, 72.75, 76.66, 76.45, 75.85, 84.25, 71.49, 90.08, 85.80, 76.29],
    "DCdetector": [71.83, 67.91, 72.21, 62.31, 69.75, 72.20, 55.79, 57.81, 70.18, 72.79, 67.77, 67.74, 67.32, 67.10, 69.55, 71.07],
    "TFMAE":     [78.25, 67.50, 75.99, 76.91, 70.30, 68.17, 56.39, 62.83, 70.60, 80.25, 76.87, 75.70, 70.07, 75.36, 70.85, 75.72],
}
MODELS = list(AFF)

# morphology -> SOTA model (domain prior, promoting morphology.py's DETECTOR_ROUTING from "detector family"
# to "SOTA model". This is the "prior" path: it does not read the leaderboard's champion, only maps by domain knowledge).
MORPH_TO_MODEL = {
    "level_shift": "TimeRCD",
    "spike": "DADA",
    "dense_cluster": "Chronos",
    "flatline": "TimeRCD",
    "variance_change": "Sub-PCA",
    "trend": "TimeRCD",
    "noise": "MovingVar",
    "persistent": "TimeRCD",
    "seasonal": "Chronos",
}

# source -> morphology, taken directly from Anomaly_algorithm/morphology.py's SOURCE_PRIOR (WSD not in the table, added separately).
SOURCE_PRIOR = {
    "NAB": "level_shift", "IOPS": "level_shift", "YAHOO": "spike",
    "Stock": "spike", "TODS": "dense_cluster", "UCR": "flatline",
    "SMD": "level_shift", "MSL": "level_shift", "SMAP": "level_shift",
    "MGAB": "level_shift", "NEK": "variance_change", "Power": "variance_change",
    "SED": "spike", "PSM": "variance_change", "SWaT": "variance_change",
    "WSD": "spike",  # WebService trace, domain-wise biased toward point anomalies; not in the original SOURCE_PRIOR table.
}


def idx(src: str) -> int:
    return SOURCES.index(src)


def prior_model(src: str) -> str:
    return MORPH_TO_MODEL[SOURCE_PRIOR[src]]


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals)


def oracle_per_source(subset: List[str]) -> Dict[str, tuple]:
    """per-source best (model, score)."""
    out = {}
    for s in subset:
        best = max(((m, AFF[m][idx(s)]) for m in MODELS), key=lambda x: x[1])
        out[s] = best
    return out


def main() -> None:
    subsets = {"univariate 11 sources": SRC_UNI, "multivariate 5 sources": SRC_MUL, "all 16 sources": SOURCES}

    print("=" * 86)
    print("1) best single model (the floor to beat)")
    print("=" * 86)
    for name, sub in subsets.items():
        best = max(((m, mean([AFF[m][idx(s)] for s in sub])) for m in MODELS), key=lambda x: x[1])
        print(f"  {name:14s} -> {best[0]:12s} {best[1]:6.2f}")

    print()
    print("=" * 86)
    print("2) oracle routing (per-source best = ceiling)")
    print("=" * 86)
    for name, sub in subsets.items():
        o = oracle_per_source(sub)
        print(f"  {name:14s} -> {mean([v for _, v in o.values()]):6.2f}")

    print()
    print("=" * 86)
    print("3) prior routing (SOURCE_PRIOR → morphology → SOTA model, purely label-free)")
    print("=" * 86)
    print(f"  {'source':6s} {'morph':14s} {'prior model':12s} {'prior score':>8s} {'oracle model':12s} {'oracle score':>8s} {'Δ':>7s}")
    for name, sub in subsets.items():
        print(f"  -- {name} --")
        rows = []
        for s in sub:
            pm = prior_model(s)
            ps = AFF[pm][idx(s)]
            om, os_ = oracle_per_source([s])[s]
            rows.append((s, SOURCE_PRIOR[s], pm, ps, om, os_, os_ - ps))
        for s, morph, pm, ps, om, os_, d in rows:
            print(f"  {s:6s} {morph:14s} {pm:12s} {ps:8.2f} {om:12s} {os_:8.2f} {d:+7.2f}")
        print(f"  {'mean':24s} {mean([r[3] for r in rows]):8.2f}")

    print()
    print("=" * 86)
    print("4) summary: the gap between the three lines")
    print("=" * 86)
    for name, sub in subsets.items():
        single = max(mean([AFF[m][idx(s)] for s in sub]) for m in MODELS)
        oracle = mean([v for _, v in oracle_per_source(sub).values()])
        prior = mean([AFF[prior_model(s)][idx(s)] for s in sub])
        print(f"  {name:14s}  single model {single:6.2f} | prior routing {prior:6.2f} "
              f"(+{prior-single:+.2f}) | oracle {oracle:6.2f} (remaining {oracle-prior:+.2f} to fill)")


if __name__ == "__main__":
    main()
