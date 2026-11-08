"""assist_strengthened.py — Strengthened morphology + temporal complement (precision-aware, targeting VUS-PR).

Base = the model chosen by routing (currently Syntropic; can swap once CHARM_kNN is dumped).
Complement = morphology prior (SOURCE_PRIOR→DETECTOR_ROUTING) picks a direction-matched detector + classify's region_mask defines the morphology region.

Temporal + precision-aware fusion (the two dimensions previously missing under VUS-PR):
  raise    —— assistant confident-anomaly (comp>0.9) and base not confident (base<0.7) and inside the morphology region
              → raise the score (cover missed detections)
  suppress —— assistant confident-normal (comp<0.1) and base confident-anomaly (base>0.7)
              → lower the score (suppress false alarms; VUS-PR precision depends on this)

Compare against max-union (raise-only, recall-oriented) to see whether VUS-PR flips.
"""
from __future__ import annotations

import json
import sys
import contextlib
import io
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Anomaly_algorithm.morphology import classify, morphology_routed_scores  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from TSB_AD.evaluation.metrics import get_metrics  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402

ROOT = Path(__file__).resolve().parent


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


def vus(score, label, sw) -> float:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return float(get_metrics(np.asarray(score).ravel(), label.astype(int),
                                 slidingWindow=sw)["VUS-PR"])


def precision_assist(base, comp, region_mask, hi=0.9, lo=0.1, base_hi=0.7, base_lo=0.7):
    """Temporal + precision-aware fusion: raise missed detections + suppress false alarms.

    raise    = assistant confident-anomaly and base not confident-anomaly and (if present) inside the morphology region
    suppress = assistant confident-normal and base confident-anomaly
    """
    fused = base.copy()
    raise_mask = (comp > hi) & (base < base_hi) & region_mask
    fused[raise_mask] = np.maximum(base, comp)[raise_mask]
    suppress_mask = (comp < lo) & (base > base_lo)
    fused[suppress_mask] = np.minimum(base, comp)[suppress_mask]
    return fused


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Syntropic", help="base model (corresponds to fullscores_*.npz)")
    ap.add_argument("--vus-limit", type=int, default=1, help="files per source for VUS-PR")
    args = ap.parse_args()

    base_scores = dict(np.load(ROOT / f"fullscores_{args.base}.npz"))
    files = sorted(base_scores.keys())
    print(f"Base {args.base}, {len(files)} files", flush=True)

    vus_subset, cnt = set(), defaultdict(int)
    for f in files:
        if cnt[source_of(f)] < args.vus_limit:
            vus_subset.add(f)
            cnt[source_of(f)] += 1

    auc_rows = defaultdict(lambda: defaultdict(list))
    vus_rows = defaultdict(lambda: defaultdict(list))
    for f in files:
        src = source_of(f)
        df = pd.read_csv(f"Datasets/TSB-AD-U/{f}").dropna()
        x = df.iloc[:, 0:-1].values.astype(float)
        label = df["Label"].astype(int).to_numpy()

        base = rank(base_scores[f])
        r = morphology_routed_scores(x, source=src)
        comp = rank(r["scores"])
        # morphology region (classify's region_mask); all-true if unavailable
        try:
            c = classify(x[:, 0].ravel(), source=src)
            region = np.asarray(c.get("region_mask", np.ones(len(x), dtype=bool)), dtype=bool)
        except Exception:
            region = np.ones(len(x), dtype=bool)
        region = region & (region.sum() > 4)  # if the morphology region is too small, degrade to all-true
        if region.sum() <= 4:
            region = np.ones(len(x), dtype=bool)

        fused_max = np.maximum(base, comp)
        fused_pa = precision_assist(base, comp, region)

        for k, v in [("base", auc(base, label)), ("max", auc(fused_max, label)),
                     ("pa", auc(fused_pa, label))]:
            auc_rows[src][k].append(v)

        if f in vus_subset:
            sw = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
            vus_rows[src]["base"].append(vus(base, label, sw))
            vus_rows[src]["max"].append(vus(fused_max, label, sw))
            vus_rows[src]["pa"].append(vus(fused_pa, label, sw))

    def mean_of(d, k):
        return float(np.mean([v for s in d for v in d[s][k]]))

    print("\n==== AUC ====")
    for s in sorted(auc_rows):
        print(f"  {s:14s} base={np.mean(auc_rows[s]['base']):.3f} "
              f"max={np.mean(auc_rows[s]['max']):.3f} pa={np.mean(auc_rows[s]['pa']):.3f}")
    print(f"  {'Overall':14s} base={mean_of(auc_rows,'base'):.3f} "
          f"max={mean_of(auc_rows,'max'):.3f} pa={mean_of(auc_rows,'pa'):.3f}")

    if vus_rows:
        print("\n==== VUS-PR ====")
        for s in sorted(vus_rows):
            print(f"  {s:14s} base={np.mean(vus_rows[s]['base'])*100:.2f} "
                  f"max={np.mean(vus_rows[s]['max'])*100:.2f} "
                  f"pa={np.mean(vus_rows[s]['pa'])*100:.2f}")
        print(f"  {'Overall':14s} base={mean_of(vus_rows,'base')*100:.2f} "
              f"max={mean_of(vus_rows,'max')*100:.2f} "
              f"pa={mean_of(vus_rows,'pa')*100:.2f}")


if __name__ == "__main__":
    main()
