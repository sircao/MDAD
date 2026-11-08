"""gen_charm_350.py —— Generate CHARM_kNN scores for 350 files (remote API, resume).

73 files reuse the old cache fullscores_CHARM_kNN.npz; the rest call run_CHARM_kNN (remote).
Output eval/score/uni/CHARM_kNN/<stem>.npy (full-length 1-D, higher = more anomalous).
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["CHARM_BASE_URL"] = "http://ab778f946c58843afa52a72d5af0657a-1381817648.us-west-2.elb.amazonaws.com:8080"
os.environ["CHARM_API_KEY"] = "token"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark_exp"))

from Run_CHARM import run_CHARM_kNN  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
CACHE = ROOT / "eval" / "score" / "uni" / "fullscores_CHARM_kNN.npz"
OUT_DIR = ROOT / "eval" / "score" / "uni" / "CHARM_kNN"


def train_length(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def main() -> None:
    files = pd.read_csv(EVA_LIST)["file_name"].tolist()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) reuse the old cache for 73 files
    cache = dict(np.load(CACHE)) if CACHE.exists() else {}
    reused = 0
    for name, sc in cache.items():
        out = OUT_DIR / f"{Path(name).stem}.npy"
        if not out.exists():
            np.save(out, np.asarray(sc, dtype=float).ravel())
            reused += 1
    print(f"reused old cache for {reused} files, {len(cache)} files in cache", flush=True)

    # 2) re-run the rest remotely
    total = len(files)
    t_start = time.time()
    for i, name in enumerate(files, 1):
        out = OUT_DIR / f"{Path(name).stem}.npy"
        if out.exists():
            continue
        data = pd.read_csv(DATA_DIR / name).to_numpy(dtype=float)
        tr = train_length(name)
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                s = np.asarray(run_CHARM_kNN(data[:tr], data), dtype=float).ravel()
            if s.ndim != 1 or len(s) != len(data) or not np.isfinite(s).all():
                raise ValueError(f"bad score shape {s.shape}")
            np.save(out, s)
            print(f"[{i}/{total}] {name} done ({len(data)} points, {time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{i}/{total}] {name} FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)

    print(f"\ntotal time {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
