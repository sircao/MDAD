"""gen_tuning_scores.py — Generate local detector scores for the Tuning 48 files (resume).

Detectors: Syntropic (local) + CHARM_stat_kNN (CHARM's local statistical part, no remote embedding).
Output eval/score/uni/tuning/{Syntropic,CHARM_stat_kNN}/<stem>.npy (full-length, higher = more anomalous).
"""
from __future__ import annotations

import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark_exp"))

from TSB_AD.model_wrapper import run_Syntropic  # noqa: E402
from Run_CHARM import HP, minmax, multiscale_max  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
TUNING_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv"
OUT_DIR = ROOT / "eval" / "score" / "uni" / "tuning"


def train_length(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def charm_stat_knn(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """CHARM_kNN's local statistical component (stats_scales, no remote embedding)."""
    return minmax(multiscale_max(train, test, HP["stats_scales"], use_stats=True))


def main() -> None:
    files = pd.read_csv(TUNING_LIST)["file_name"].tolist()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for det in ("Syntropic", "CHARM_stat_kNN"):
        (OUT_DIR / det).mkdir(exist_ok=True)

    for i, f in enumerate(files, 1):
        stem = Path(f).stem
        syn_out = OUT_DIR / "Syntropic" / f"{stem}.npy"
        csk_out = OUT_DIR / "CHARM_stat_kNN" / f"{stem}.npy"
        if syn_out.exists() and csk_out.exists():
            continue
        data = pd.read_csv(DATA_DIR / f).to_numpy(dtype=float)
        tr = train_length(f)
        t0 = time.time()
        if not syn_out.exists():
            s = np.asarray(run_Syntropic(data[:tr], data), dtype=float).ravel()
            np.save(syn_out, s)
        if not csk_out.exists():
            c = np.asarray(charm_stat_knn(data[:tr], data), dtype=float).ravel()
            np.save(csk_out, c)
        print(f"[{i}/{len(files)}] {f} ({time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
