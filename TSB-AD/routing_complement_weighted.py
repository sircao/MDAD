"""routing_complement_weighted.py — Two-level routing + strong-model complement (fixed avg vs per-source weighting).

Base = routing-chosen; complement = the other strong model.
  fixed avg: 0.5*CHARM + 0.5*Syn
  per-source weighting: per source, fast grid-search w ∈ {0,.25,.5,.75,1} by AUC (w*CHARM+(1-w)*Syn)
AUC on all 65 files + VUS-PR on a per-source 1-file subset.
"""
from __future__ import annotations

import json
import sys
import time
import contextlib
import io
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from TSB_AD.evaluation.metrics import get_metrics  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402

ROOT = Path(__file__).resolve().parent


def source_of(f): return f.split("_")[1]


def rank(s):
    r = rankdata(s)
    return r / len(r)


def auc(score, label):
    s = np.asarray(score, dtype=float); y = np.asarray(label, dtype=int)
    if len(np.unique(y)) < 2 or np.allclose(s, s[0]):
        return 0.5
    return float(roc_auc_score(y, s))


def vus(score, label, sw):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return float(get_metrics(np.asarray(score).ravel(), label.astype(int),
                                 slidingWindow=sw)["VUS-PR"])


def safe_load(path, tries=6):
    for _ in range(tries):
        try:
            return dict(np.load(path))
        except Exception:
            time.sleep(1.0)
    return dict(np.load(path))


def main():
    picks = json.load(open(ROOT / "runs" / "llm_routing_combined.json", encoding="utf-8"))["picks_u"]
    charm = safe_load(ROOT / "fullscores_CHARM_kNN.npz")
    syn = safe_load(ROOT / "fullscores_Syntropic.npz")
    files = sorted(set(charm) & set(syn))
    print(f"files {len(files)}", flush=True)

    prep = {}
    for f in files:
        df = pd.read_csv(f"Datasets/TSB-AD-U/{f}").dropna()
        x = df.iloc[:, 0:-1].values.astype(float)
        l = df["Label"].astype(int).to_numpy()
        sw = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
        pick = picks.get(source_of(f), "CHARM_kNN")
        prep[f] = {"b": rank(charm[f] if pick == "CHARM_kNN" else syn[f]),
                   "c": rank(syn[f] if pick == "CHARM_kNN" else charm[f]),
                   "l": l, "sw": sw, "src": source_of(f)}

    bysrc = defaultdict(list)
    for f in files:
        bysrc[source_of(f)].append(f)

    # per-source weighting: grid-search the optimal w by AUC
    w_of = {}
    for s, fs in sorted(bysrc.items()):
        best_w, best_a = 0.5, -1
        for w in (0, 0.25, 0.5, 0.75, 1.0):
            a = float(np.mean([auc(w * prep[f]["b"] + (1 - w) * prep[f]["c"], prep[f]["l"]) for f in fs]))
            if a > best_a:
                best_a, best_w = a, w
        w_of[s] = best_w

    def score_of(f, w):
        return w * prep[f]["b"] + (1 - w) * prep[f]["c"]

    # three lines
    fixed = {f: 0.5 * (prep[f]["b"] + prep[f]["c"]) for f in files}
    weighted = {f: score_of(f, w_of[source_of(f)]) for f in files}

    def agg_mean(d, key):
        return float(np.mean([d[f][key] for f in files]))

    print("\n=== AUC (full) ===")
    for name, d in [("base", {f: prep[f]["b"] for f in files}), ("fixed", fixed), ("weighted", weighted)]:
        a = np.mean([auc(s, prep[f]["l"]) for f, s in d.items()])
        print(f"  {name:10s} {a*100:.2f}")
    print("\noptimal w per source:", {k: f"{v:.2f}" for k, v in sorted(w_of.items())})

    # VUS-PR subset (1 file per source)
    sub = {}
    cnt = defaultdict(int)
    for f in files:
        s = source_of(f)
        if cnt[s] < 1:
            sub[f] = prep[f]
            cnt[s] += 1
    print(f"\n=== VUS-PR (subset {len(sub)} files) ===")
    for name, wf in [("base", None), ("fixed", 0.5), ("weighted", None)]:
        vals = []
        for f, p in sub.items():
            s = (p["b"] if wf is None and name == "base" else
                 score_of(f, 0.5) if name == "fixed" else
                 score_of(f, w_of[source_of(f)]))
            vals.append(vus(s, p["l"], p["sw"]))
        print(f"  {name:10s} {np.mean(vals)*100:.2f}")


if __name__ == "__main__":
    main()
