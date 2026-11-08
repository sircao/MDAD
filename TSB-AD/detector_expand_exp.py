"""detector_expand_exp.py — Does adding detectors help?

Compare 3 detectors vs 6 detectors (adding CHARM_ZS / CHARM_kNN / LOF):
  - how much the SOTA ceiling (oracle picking the best single detector per file) rises;
  - how much the label-free mean/max score ensemble rises;
  - how often each detector becomes the oracle best (to see whether the new detectors actually win).
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402
from fast_vus import fast_vus  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
SCORE_DIR = ROOT / "eval" / "score" / "uni" / "eva"
DETS3 = ["Syntropic", "Time_RCD", "CHARM_stat_kNN"]
DETS6 = ["Syntropic", "Time_RCD", "CHARM_stat_kNN", "CHARM_ZS", "CHARM_kNN", "LOF"]
MAX_LEN = 8000


def rank01(s):
    s = np.asarray(s, float).ravel()
    return s / max(len(s), 1)


def main() -> None:
    files = pd.read_csv(EVA_LIST)["file_name"].tolist()[::5]  # sample 70

    res = {"sota3": [], "sota6": [], "mean3": [], "max3": [], "mean6": [], "max6": []}
    best6 = Counter()
    for name in files:
        full = pd.read_csv(DATA_DIR / name).to_numpy(float)
        label = pd.read_csv(LABEL_DIR / name)["Label"].to_numpy(int)
        n = len(full)
        idx = np.arange(n)
        if n > MAX_LEN:
            step = int(np.ceil(n / MAX_LEN))
            idx = np.arange(0, n, step)
        y = label[idx]
        sw = int(find_length_rank(full[idx][:, 0].reshape(-1, 1), rank=1))
        stem = Path(name).stem
        S = {}
        for d in DETS6:
            p = SCORE_DIR / d / f"{stem}.npy"
            if p.exists():
                S[d] = rank01(np.load(p, allow_pickle=False).ravel()[idx])
        if len(S) < 6:
            continue
        vus = {d: fast_vus(s, y, sw) for d, s in S.items()}
        v3 = {d: vus[d] for d in DETS3}
        res["sota3"].append(max(v3.values()))
        res["sota6"].append(max(vus.values()))
        res["mean3"].append(fast_vus(np.mean([S[d] for d in DETS3], axis=0), y, sw))
        res["max3"].append(fast_vus(np.max([S[d] for d in DETS3], axis=0), y, sw))
        res["mean6"].append(fast_vus(np.mean([S[d] for d in S], axis=0), y, sw))
        res["max6"].append(fast_vus(np.max([S[d] for d in S], axis=0), y, sw))
        best6[max(vus, key=vus.get)] += 1

    print(f"sampled {len(res['sota3'])} eva files\n")
    print("== ceiling & label-free ensemble ==")
    for k, v in res.items():
        print(f"  {k:8s} mean VUS-PR = {np.mean(v):.4f}")
    print("\n== frequency of each of the 6 detectors as oracle best ==")
    for d, c in best6.most_common():
        print(f"  {d:16s} {c}")


if __name__ == "__main__":
    main()
