"""assist_memory.py — Complementary mechanism driven by the Memory_base morphology memory.

Base = Syntropic (or CHARM_kNN); complementary direction + weight come from morph_memory aggregation:
  classify determines the coarse morphology -> morph_memory.route() -> best assistant + data-driven assist_weight.
Fusion = base + w·relu(comp − base) (soft raise-only), plus a precision-aware (raise+suppress) comparison.

Compare against the hand-written version (routing[0] + ASSIST_WEIGHT from assist_strengthened.py) to see whether the memory bank yields a gain.
"""
from __future__ import annotations

import sys
import contextlib
import io
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Anomaly_algorithm.morphology import classify  # noqa: E402
from Anomaly_algorithm.registry import get  # noqa: E402
import morph_memory  # noqa: E402
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


def run_complement(x, morphology):
    """Run the best assistant via memory routing -> per-point rank scores."""
    tool, args, w = morph_memory.route(morphology)
    try:
        r = get(tool).run(x, **args)
        if not r.ok:
            raise RuntimeError('tool not ok')
        scores = np.asarray(r.data)
        if scores.ndim == 2:
            scores = scores.max(axis=1)
        return rank(scores.ravel()), w, tool
    except Exception:
        return None, w, tool


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Syntropic")
    ap.add_argument("--vus-limit", type=int, default=1)
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
        c = classify(x[:, 0].ravel(), source=src)
        morph = c["morphology"]
        comp, w, tool = run_complement(x, morph)
        if comp is None:
            for k in ["base", "mem", "mem_pa"]:
                auc_rows[src][k].append(auc(base, label))
            if f in vus_subset:
                sw = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
                b = vus(base, label, sw)
                for k in ["base", "mem", "mem_pa"]:
                    vus_rows[src][k].append(b)
            continue

        fused_mem = base + w * np.clip(comp - base, 0, None)
        # precision-aware: raise missed detections + suppress false alarms (reuse the operator from assist_strengthened)
        fused_pa = base.copy()
        rm = (comp > 0.9) & (base < 0.7)
        fused_pa[rm] = np.maximum(base, comp)[rm]
        sm = (comp < 0.1) & (base > 0.7)
        fused_pa[sm] = np.minimum(base, comp)[sm]

        auc_rows[src]["base"].append(auc(base, label))
        auc_rows[src]["mem"].append(auc(fused_mem, label))
        auc_rows[src]["mem_pa"].append(auc(fused_pa, label))

        if f in vus_subset:
            sw = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
            vus_rows[src]["base"].append(vus(base, label, sw))
            vus_rows[src]["mem"].append(vus(fused_mem, label, sw))
            vus_rows[src]["mem_pa"].append(vus(fused_pa, label, sw))

    def mean_of(d, k):
        return float(np.mean([v for s in d for v in d[s][k]]))

    print("\n==== AUC ====")
    for s in sorted(auc_rows):
        print(f"  {s:14s} base={np.mean(auc_rows[s]['base']):.3f} "
              f"mem={np.mean(auc_rows[s]['mem']):.3f} mem_pa={np.mean(auc_rows[s]['mem_pa']):.3f}")
    print(f"  {'Overall':14s} base={mean_of(auc_rows,'base'):.3f} "
          f"mem={mean_of(auc_rows,'mem'):.3f} mem_pa={mean_of(auc_rows,'mem_pa'):.3f}")

    if vus_rows:
        print("\n==== VUS-PR ====")
        for s in sorted(vus_rows):
            print(f"  {s:14s} base={np.mean(vus_rows[s]['base'])*100:.2f} "
                  f"mem={np.mean(vus_rows[s]['mem'])*100:.2f} "
                  f"mem_pa={np.mean(vus_rows[s]['mem_pa'])*100:.2f}")
        print(f"  {'Overall':14s} base={mean_of(vus_rows,'base')*100:.2f} "
              f"mem={mean_of(vus_rows,'mem')*100:.2f} "
              f"mem_pa={mean_of(vus_rows,'mem_pa')*100:.2f}")


if __name__ == "__main__":
    main()
