"""eval_memory.py —— Strengthen routing prior + complement with label-free morphology memory (morph_memory_freelabel), all 350 under official metric.

Four lines:
  base     = memory-bank routing prior (atlas[morphology].best picks the base)
  fixed    = fixed 0.5·rank(CHARM_stat_kNN)+0.5·rank(Syntropic)
  fused    = memory-routed base + 0.5 rank-avg complement (the other base)
  fused_w  = memory-routed base + complement weighted by atlas quality
VUS-PR uses multiprocessing.
"""
from __future__ import annotations

import contextlib
import io
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from TSB_AD.evaluation.metrics import get_metrics  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402

CACHE = ROOT / "eval" / "score" / "uni" / "freelabel"
MEM_PATH = ROOT.parent / "Memory_base" / "morph_memory_freelabel.json"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"

BASE_MODELS = {"CHARM_stat_kNN", "Syntropic"}


def rank01(s: np.ndarray) -> np.ndarray:
    return rankdata(s) / len(s)


def _vus(s: np.ndarray, y: np.ndarray, sw: int) -> float:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return float(get_metrics(np.asarray(s, dtype=float).ravel(), y.astype(int),
                                 slidingWindow=sw)["VUS-PR"])


def _worker(args):
    f, line, s, y, sw = args
    return f, line, _vus(s, y, sw)


def main() -> None:
    mem = json.loads(MEM_PATH.read_text(encoding="utf-8"))
    atlas = mem["atlas"]
    per_file = mem["per_file_morph"]
    files = pd.read_csv(EVA_LIST)["file_name"].tolist()

    prep = {}
    for f in files:
        stem = Path(f).stem
        src = f.split("_")[1]
        labels = pd.read_csv(LABEL_DIR / f)["Label"].to_numpy(dtype=float)
        x = pd.read_csv(DATA_DIR / f).to_numpy(dtype=float)
        sw = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))

        knn = rank01(np.load(CACHE / "CHARM_stat_kNN" / f"{stem}.npy", allow_pickle=False))
        syn = rank01(np.load(CACHE / "Syntropic" / f"{stem}.npy", allow_pickle=False))

        morph = per_file.get(f, "level_shift")
        best = atlas.get(morph, {}).get("best", "CHARM_stat_kNN")
        if best not in BASE_MODELS:
            best = "CHARM_stat_kNN"
        other = "Syntropic" if best == "CHARM_stat_kNN" else "CHARM_stat_kNN"
        q = atlas.get(morph, {}).get("quality", {})
        qb = q.get(best, 0.5)
        qo = q.get(other, 0.5)
        w = qb / (qb + qo + 1e-12)

        b = knn if best == "CHARM_stat_kNN" else syn
        o = syn if other == "Syntropic" else knn
        prep[f] = {
            "base": b,
            "fixed": 0.5 * knn + 0.5 * syn,
            "fused": 0.5 * b + 0.5 * o,
            "fused_w": w * b + (1 - w) * o,
            "label": labels, "sw": sw, "src": src, "morph": morph, "best": best,
        }
    print(f"prep done {len(prep)}", flush=True)

    lines = ["base", "fixed", "fused", "fused_w"]
    tasks = [(f, line, prep[f][line], prep[f]["label"], prep[f]["sw"])
             for f in files for line in lines]
    mp.set_start_method("spawn", force=True)
    with mp.Pool(6) as pool:
        results = pool.map(_worker, tasks, chunksize=4)

    res = {line: {} for line in lines}
    for f, line, v in results:
        res[line][f] = v

    print("\n==== label-free morphology memory strengthening · official VUS-PR (all 350) ====")
    for line in lines:
        m = float(np.mean([res[line][f] for f in files]))
        print(f"  {line:8s} {m:.4f}")
    # per source
    print("\n==== per-source fused_w ====")
    from collections import defaultdict
    agg = defaultdict(list)
    for f in files:
        agg[prep[f]["src"]].append(res["fused_w"][f])
    for s in sorted(agg):
        print(f"  {s:14s} {np.mean(agg[s]):.4f} (n={len(agg[s])})")

    out = {"lines": {line: round(float(np.mean([res[line][f] for f in files])), 4)
                     for line in lines},
           "per_file": {f: {line: round(res[line][f], 4) for line in lines} for f in files}}
    (ROOT / "runs" / "memory_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n[saved] runs/memory_result.json")


if __name__ == "__main__":
    main()
