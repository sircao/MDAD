"""gen_scores_350.py —— Locally re-run Syntropic's full scores (350 U-Eva files).

Label-free data + official training segment, run_Syntropic(WSK+DiffZ+Time-RCD). Resume.
Output eval/score/uni/Syntropic/<stem>.npy (full-length 1-D, higher = more anomalous).
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

from TSB_AD.model_wrapper import run_Syntropic  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
OUT_DIR = ROOT / "eval" / "score" / "uni" / "Syntropic"


def train_length(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def main() -> None:
    files = pd.read_csv(EVA_LIST)["file_name"].tolist()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(files)
    done = 0
    t_start = time.time()
    for i, name in enumerate(files, 1):
        out = OUT_DIR / f"{Path(name).stem}.npy"
        if out.exists():
            continue
        data = pd.read_csv(DATA_DIR / name).to_numpy(dtype=float)
        tr = train_length(name)
        t0 = time.time()
        try:
            s = np.asarray(run_Syntropic(data[:tr], data), dtype=float).ravel()
            if s.ndim != 1 or len(s) != len(data) or not np.isfinite(s).all():
                raise ValueError(f"bad score shape {s.shape}")
            np.save(out, s)
            done += 1
            print(f"[{i}/{total}] {name} done ({len(data)} points, {time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[{i}/{total}] {name} FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)
    print(f"\nnewly generated {done}/{total} files, total time {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
