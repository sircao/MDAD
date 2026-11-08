"""routing_complement_final.py — Two-level routing + strong-model complement (final version, VUS-PR metric).

Base = the detector chosen by the second level of the two-level routing (U: CHARM_kNN 19 sources + Syntropic 4 sources).
Complement = the other strong model in the pool (CHARM_kNN <-> Syntropic), rank-averaged fusion to cancel errors.
Comparison: routing 68.71 (U). 22 files already verified avg +3.9pp VUS-PR.

Usage: python routing_complement_final.py --vus-limit 1
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
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def safe_load(path, tries=6):
    for i in range(tries):
        try:
            return dict(np.load(path))
        except Exception:
            time.sleep(1.0)
    return dict(np.load(path))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vus-limit", type=int, default=1)
    args = ap.parse_args()

    picks = json.load(open(ROOT / "runs" / "llm_routing_combined.json", encoding="utf-8"))["picks_u"]
    charm = safe_load(ROOT / "fullscores_CHARM_kNN.npz")
    syn = safe_load(ROOT / "fullscores_Syntropic.npz")

    files = sorted(set(charm) & set(syn))
    print(f"available files {len(files)} (CHARM {len(charm)} ∩ Syntropic {len(syn)})", flush=True)

    vus_subset, cnt = set(), defaultdict(int)
    for f in files:
        if cnt[source_of(f)] < args.vus_limit:
            vus_subset.add(f)
            cnt[source_of(f)] += 1

    auc_rows = defaultdict(lambda: defaultdict(list))
    vus_rows = defaultdict(lambda: defaultdict(list))
    for i, f in enumerate(files):
        src = source_of(f)
        df = pd.read_csv(f"Datasets/TSB-AD-U/{f}").dropna()
        x = df.iloc[:, 0:-1].values.astype(float)
        label = df["Label"].astype(int).to_numpy()

        pick = picks.get(src, "CHARM_kNN")
        b = rank(charm[f] if pick == "CHARM_kNN" else syn[f])   # base = routing-chosen
        c = rank(syn[f] if pick == "CHARM_kNN" else charm[f])   # complement = the other strong model
        fused = 0.5 * (b + c)                                   # rank average

        auc_rows[src]["base"].append(auc(b, label))
        auc_rows[src]["fused"].append(auc(fused, label))

        if f in vus_subset:
            sw = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
            vus_rows[src]["base"].append(vus(b, label, sw))
            vus_rows[src]["fused"].append(vus(fused, label, sw))
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(files)}", flush=True)

    def mean_of(d, k):
        return float(np.mean([v for s in d for v in d[s][k]]))

    print("\n==== AUC ====")
    for s in sorted(auc_rows):
        print(f"  {s:14s} base={np.mean(auc_rows[s]['base']):.3f} "
              f"fused={np.mean(auc_rows[s]['fused']):.3f}")
    print(f"  {'Overall':14s} base={mean_of(auc_rows,'base'):.3f} fused={mean_of(auc_rows,'fused'):.3f}")

    if vus_rows:
        print("\n==== VUS-PR ====")
        for s in sorted(vus_rows):
            print(f"  {s:14s} base={np.mean(vus_rows[s]['base'])*100:.2f} "
                  f"fused={np.mean(vus_rows[s]['fused'])*100:.2f}")
        print(f"  {'Overall':14s} base={mean_of(vus_rows,'base')*100:.2f} "
              f"fused={mean_of(vus_rows,'fused')*100:.2f}")


if __name__ == "__main__":
    main()
