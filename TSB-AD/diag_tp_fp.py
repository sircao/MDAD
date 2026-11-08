"""diag_tp_fp.py —— Decompose the loop's TP / FP contribution ratio from VUS-PR.

For each file: base (routing base) → loop (complementary morphology prior). Split the loop's changes
relative to base into two paths by position:
  tp_only = base but with loop's scores at TP positions (only strengthen TP)
  fp_only = base but with loop's scores at FP positions (only change FP)
Compute VUS-PR for all four, giving TP contribution = VUSPR(tp_only)-VUSPR(base), FP contribution = VUSPR(fp_only)-VUSPR(base).
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from eval_loop import rank01, _vus, run_loop  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402

CACHE = ROOT / "eval" / "score" / "uni" / "freelabel"
CHARM_FULL_DIR = ROOT / "eval" / "score" / "uni" / "CHARM_kNN"
MEM_PATH = ROOT.parent / "Memory_base" / "morph_memory_freelabel.json"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
MODELS = ["CHARM_stat_kNN", "Syntropic", "Time_RCD", "LOF", "CHARM_stat_ZS"]
BASE_MODELS = {"CHARM_stat_kNN", "Syntropic"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    mem = json.loads(MEM_PATH.read_text(encoding="utf-8"))
    atlas, per_file = mem["atlas"], mem["per_file_morph"]
    files = pd.read_csv(EVA_LIST)["file_name"].tolist()[:args.limit]

    tp_gain, fp_gain = [], []
    base_v, loop_v = [], []
    for f in files:
        stem = Path(f).stem
        labels = pd.read_csv(LABEL_DIR / f)["Label"].to_numpy(dtype=float)
        x = pd.read_csv(DATA_DIR / f).to_numpy(dtype=float)
        sw = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
        scores = [rank01(np.load(CACHE / m / f"{stem}.npy", allow_pickle=False))
                  for m in MODELS]
        cfull = CHARM_FULL_DIR / f"{stem}.npy"
        if cfull.exists():
            scores.append(rank01(np.load(cfull, allow_pickle=False)))
        S = np.stack(scores).astype(np.float32)
        morph = per_file.get(f, "level_shift")
        best = atlas.get(morph, {}).get("best", "CHARM_stat_kNN")
        if best not in BASE_MODELS:
            best = "CHARM_stat_kNN"
        if cfull.exists():
            base = S[-1]
        else:
            bi = MODELS.index(best)
            base = S[bi]
        base_max_vus = _vus(base, labels, sw)  # simplification: baseline uses only base, approximate
        loop = run_loop(base, S, labels, sw, base_max_vus)

        y = labels.astype(bool)
        tp_only = base.copy(); tp_only[y] = loop[y]
        fp_only = base.copy(); fp_only[~y] = loop[~y]

        b = _vus(base, labels, sw)
        l = _vus(loop, labels, sw)
        t = _vus(tp_only, labels, sw)
        fp = _vus(fp_only, labels, sw)
        base_v.append(b); loop_v.append(l)
        tp_gain.append(t - b)
        fp_gain.append(fp - b)
        print(f"{f[:40]} base={b:.4f} loop={l:.4f} tp_only={t:.4f} fp_only={fp:.4f} "
              f"TP contrib={t-b:+.4f} FP contrib={fp-b:+.4f}", flush=True)

    tp = float(np.mean(tp_gain))
    fp = float(np.mean(fp_gain))
    total = tp + fp
    print("\n==== VUS-PR decomposition (mean) ====")
    print(f"base  VUS-PR          = {np.mean(base_v):.4f}")
    print(f"loop  VUS-PR          = {np.mean(loop_v):.4f}  (Δ {np.mean(loop_v)-np.mean(base_v):+.4f})")
    print(f"TP contribution (only TP positions changed) = {tp:+.4f}")
    print(f"FP contribution (only FP positions changed) = {fp:+.4f}")
    print(f"TP share = {tp/total*100:.1f}%  |  FP share = {fp/total*100:.1f}%")


if __name__ == "__main__":
    main()
