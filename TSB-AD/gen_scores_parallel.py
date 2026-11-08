"""gen_scores_parallel.py —— Multiprocess parallel generation of CHARM_kNN + Syntropic scores.

Remote CHARM embedding is the bottleneck; use N processes to call the remote API concurrently for throughput; Syntropic is fast locally.
Each worker processes one file, resume (skip if the npy already exists).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

os.environ["CHARM_BASE_URL"] = "http://ab778f946c58843afa52a72d5af0657a-1381817648.us-west-2.elb.amazonaws.com:8080"
os.environ["CHARM_API_KEY"] = "token"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
SYN_DIR = ROOT / "eval" / "score" / "uni" / "Syntropic"
CHARM_DIR = ROOT / "eval" / "score" / "uni" / "CHARM_kNN"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "benchmark_exp"))

from TSB_AD.model_wrapper import run_Syntropic  # noqa: E402
from Run_CHARM import run_CHARM_kNN  # noqa: E402


def train_length(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def process_one(name: str) -> str:
    """Single file: Syntropic (local) + CHARM_kNN (remote), resume."""
    stem = Path(name).stem
    syn_out = SYN_DIR / f"{stem}.npy"
    charm_out = CHARM_DIR / f"{stem}.npy"
    if syn_out.exists() and charm_out.exists():
        return name

    data = pd.read_csv(DATA_DIR / name).to_numpy(dtype=float)
    tr = train_length(name)

    if not syn_out.exists():
        s = np.asarray(run_Syntropic(data[:tr], data), dtype=float).ravel()
        np.save(syn_out, s)

    if not charm_out.exists():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            c = np.asarray(run_CHARM_kNN(data[:tr], data), dtype=float).ravel()
        np.save(charm_out, c)

    return name


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    files = pd.read_csv(EVA_LIST)["file_name"].tolist()
    total = len(files)
    t0 = time.time()
    done = 0
    with Pool(processes=args.workers) as pool:
        for name in pool.imap_unordered(process_one, files, chunksize=1):
            done += 1
            syn_n = len(list(SYN_DIR.glob("*.npy")))
            charm_n = len(list(CHARM_DIR.glob("*.npy")))
            (ROOT / "score_progress.json").write_text(json.dumps({
                "status": "scoring", "total": total, "done": done,
                "syn_count": syn_n, "charm_count": charm_n,
                "started": t0, "updated": time.time(),
            }, ensure_ascii=False), encoding="utf-8")
            print(f"[{done}/{total}] {name} | Syn={syn_n} CHARM={charm_n} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    syn_n = len(list(SYN_DIR.glob("*.npy")))
    charm_n = len(list(CHARM_DIR.glob("*.npy")))
    (ROOT / "score_progress.json").write_text(json.dumps({
        "status": "scoring_done", "total": total, "done": total,
        "syn_count": syn_n, "charm_count": charm_n,
        "started": t0, "updated": time.time(),
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\ndone. Syn={syn_n} "
          f"CHARM={charm_n} total time {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
