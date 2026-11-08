"""best_selector_exp.py — Find a label-free "best detector" selection signal stronger than morphology.

Mark oracle best (argmax det_vus) on tuning (labeled), then test several label-free signals
computable in eva: source prior, detector-score/support consistency, and raw-score statistics.
Whichever has the highest hit rate becomes the best-selection rule to write into route_params.json next.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from Anomaly_algorithm.morphology_evidence import extract_evidence  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
SCORE_DIR = ROOT / "eval" / "score" / "uni" / "tuning"
DETS = ["Syntropic", "Time_RCD", "CHARM_stat_kNN"]
MAX_LEN = 25000


def tr_of(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def _consistency(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(rankdata(a), rankdata(b))[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def main() -> None:
    rows = json.loads((ROOT / "morph_validate_tuning.json").read_text(encoding="utf-8"))
    feat = {}
    for r in rows:
        name = r["file"]
        full = pd.read_csv(DATA_DIR / name).to_numpy(float)
        n = len(full)
        idx = np.arange(n)
        if n > MAX_LEN:
            step = int(np.ceil(n / MAX_LEN))
            idx = np.arange(0, n, step)
        x = full[idx]
        tr = int(np.searchsorted(idx, tr_of(name)))
        support = extract_evidence(x, reference=x[:tr]).support
        stem = Path(name).stem
        S_raw = {}
        for d in DETS:
            p = SCORE_DIR / d / f"{stem}.npy"
            if p.exists():
                S_raw[d] = np.load(p, allow_pickle=False).ravel()[idx]
        cons = {d: _consistency(S_raw[d], support) for d in S_raw}
        feat[name] = {"source": r["source"], "morph": r["top_kind"],
                      "true_best": r["true_best_det"], "cons": cons, "S": S_raw}

    def hit(pred_fn):
        hit = tot = 0
        for f in feat.values():
            p = pred_fn(f)
            if p is not None:
                tot += 1
                hit += int(p == f["true_best"])
        return f"{hit}/{tot} = {hit/max(tot,1):.1%}"

    morph_mode = {}
    g = defaultdict(list)
    for f in feat.values():
        g[f["morph"]].append(f["true_best"])
    for k, vs in g.items():
        morph_mode[k] = Counter(vs).most_common(1)[0][0]

    src_mode = {}
    g = defaultdict(list)
    for f in feat.values():
        g[f["source"]].append(f["true_best"])
    for k, vs in g.items():
        src_mode[k] = Counter(vs).most_common(1)[0][0]

    print("== label-free best selector hit rate (tuning; oracle best upper bound=100%) ==\n")
    print("morph->best (current baseline):", hit(lambda f: morph_mode.get(f["morph"])))
    print("source->best           :", hit(lambda f: src_mode.get(f["source"])))
    print("argmax consistency(S,support):", hit(lambda f: max(f["cons"], key=f["cons"].get) if f["cons"] else None))
    print("argmax raw-score mean     :", hit(lambda f: max(f["S"], key=lambda d: float(np.mean(f["S"][d])))))
    print("argmax raw-score max      :", hit(lambda f: max(f["S"], key=lambda d: float(np.max(f["S"][d])))))
    print("argmax raw-score std      :", hit(lambda f: max(f["S"], key=lambda d: float(np.std(f["S"][d])))))
    print("argmax raw-score q99      :", hit(lambda f: max(f["S"], key=lambda d: float(np.quantile(f["S"][d], 0.99)))))

    # source hit rate + who wins when source conflicts with morph
    print("\n== oracle best mode per source ==\n")
    for k, vs in sorted(g.items()):
        c = Counter(vs)
        print(f"  {k:12s} {dict(c)}")


if __name__ == "__main__":
    main()
