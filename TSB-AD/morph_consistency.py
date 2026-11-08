"""morph_consistency.py — Quantify intra-morphology detector consistency: which morphologies should be split and which merged.

For each Tuning file (≥2 detector scores):
  - determine morphology (extract_evidence, label-free)
  - compute each detector's AUC (fast ranking proxy, an order of magnitude faster than VUS-PR)
  - best = the detector with the highest AUC
Aggregate at two levels — "morphology" and "morphology+source" — to look at the voting consistency of the best detector.
High consistency → the morphology granularity is enough; low consistency → should split (by source or finer morphology).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from Anomaly_algorithm.morphology_evidence import extract_evidence  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
TUNING_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv"
SCORE_DIR = ROOT / "eval" / "score" / "uni" / "tuning"
DETS = ["Syntropic", "CHARM_kNN", "Time_RCD", "CHARM_ZS"]


def tr_of(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def source_of(f: str) -> str:
    return f.split("_")[1]


def auc(score, label) -> float:
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=int)
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, s))


def main() -> None:
    files = pd.read_csv(TUNING_LIST)["file_name"].tolist()
    rows = []
    for f in files:
        stem = Path(f).stem
        sc = {}
        for d in DETS:
            p = SCORE_DIR / d / f"{stem}.npy"
            if p.exists():
                sc[d] = np.load(p, allow_pickle=False).ravel()
        if len(sc) < 2:
            continue
        data = pd.read_csv(DATA_DIR / f).to_numpy(dtype=float)
        label = pd.read_csv(LABEL_DIR / f)["Label"].to_numpy(dtype=int)
        morph = str(extract_evidence(data, reference=data[:tr_of(f)]).summary()["morphology"])
        v = {d: auc(s, label) for d, s in sc.items()}
        best = max(v, key=v.get)
        rows.append((f, morph, source_of(f), best, v))
    print(f"available files (≥2 detector scores): {len(rows)}\n", flush=True)

    # aggregate by morphology
    print("==== by morphology: best-detector voting consistency ====")
    by_morph = defaultdict(list)
    for f, m, src, best, v in rows:
        by_morph[m].append(best)
    for m, bests in sorted(by_morph.items(), key=lambda kv: -len(kv[1])):
        c = Counter(bests)
        top, topn = c.most_common(1)[0]
        cons = topn / len(bests)
        votes = "  ".join(f"{d}×{n}" for d, n in c.most_common())
        flag = "" if cons >= 0.8 or len(bests) < 3 else "  ← inconsistent, suggest split"
        print(f"  {m:16s} n={len(bests):2d}  consistency={cons*100:4.0f}%  [{votes}]{flag}")

    # aggregate by morphology+source
    print("\n==== by morphology+source: consistency after splitting out source ====")
    by_ms = defaultdict(list)
    for f, m, src, best, v in rows:
        by_ms[(m, src)].append(best)
    for (m, src), bests in sorted(by_ms.items(), key=lambda kv: -len(kv[1])):
        c = Counter(bests)
        top, topn = c.most_common(1)[0]
        votes = "  ".join(f"{d}×{n}" for d, n in c.most_common())
        print(f"  {m:16s} {src:8s} n={len(bests):2d}  [{votes}]")

    # source distribution of the unknown bucket (known issue)
    print("\n==== source distribution of the unknown bucket ====")
    unk = Counter(src for f, m, src, best, v in rows if m == "unknown")
    for src, n in unk.most_common():
        print(f"  {src:8s} {n}")


if __name__ == "__main__":
    main()
