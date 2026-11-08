"""ensemble_exp.py — Test whether a "label-free score ensemble" can bypass best-selection and approach/beat SOTA.

On sampled eva files, compute the VUS-PR of the mean/max/sum of three detectors' rank01 scores,
comparing routing base (picks best) against SOTA (per-file label oracle picking the best single detector).
"""
from __future__ import annotations

import re
import sys
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
DETS = ["Syntropic", "Time_RCD", "CHARM_stat_kNN"]
MAX_LEN = 8000


def tr_of(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def rank01(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float).ravel()
    return s / max(len(s), 1)


def main() -> None:
    files = pd.read_csv(EVA_LIST)["file_name"].tolist()
    files = files[::5]   # sample 70 for a quick read of the conclusion

    agg = {"mean": [], "max": [], "sum": [], "best_single": []}
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
        for d in DETS:
            p = SCORE_DIR / d / f"{stem}.npy"
            if p.exists():
                S[d] = rank01(np.load(p, allow_pickle=False).ravel()[idx])
        if len(S) < 3:
            continue
        vals = list(S.values())
        agg["mean"].append(fast_vus(np.mean(vals, axis=0), y, sw))
        agg["max"].append(fast_vus(np.max(vals, axis=0), y, sw))
        agg["sum"].append(fast_vus(np.sum(vals, axis=0), y, sw))
        agg["best_single"].append(max(fast_vus(s, y, sw) for s in vals))

    print(f"eva sampled {len(agg['mean'])} files\n")
    for k, v in agg.items():
        print(f"  {k:14s} mean VUS-PR = {np.mean(v):.4f}")


if __name__ == "__main__":
    main()
