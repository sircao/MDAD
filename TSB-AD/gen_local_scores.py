"""gen_local_scores.py —— Multiprocess parallel generation of 4 local detector scores (resume).

Detectors (all local, no remote): Syntropic / Time_RCD / CHARM_stat_kNN / OmniAnomaly (VAE reconstruction, strong on point anomalies/Stock).
Coverage: Tuning 48 first, then Eva 350.
Output eval/score/uni/{tuning,eva}/{det}/<stem>.npy (full-length, higher = more anomalous).
"""
from __future__ import annotations

import contextlib
import io
import json
import multiprocessing as mp
import os
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

from TSB_AD.model_wrapper import run_Syntropic, run_Time_RCD, run_OmniAnomaly  # noqa: E402
from Run_CHARM import HP, minmax, multiscale_max  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
TUNING_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv"
BASE = ROOT / "eval" / "score" / "uni"
DETS = ("Syntropic", "Time_RCD", "CHARM_stat_kNN", "OmniAnomaly")  # all local, no remote


def tr_of(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args)


def charm_stat_knn(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """CHARM_kNN's local statistical component (per-window [std,range,max,min,mean] L2 kNN, no remote embedding)."""
    return minmax(multiscale_max(train, test, HP["stats_scales"], use_stats=True))


def run_det(det: str, data: np.ndarray, tr: int) -> np.ndarray:
    if det == "Syntropic":
        return np.asarray(run_Syntropic(data[:tr], data), dtype=float).ravel()
    if det == "Time_RCD":
        return np.asarray(run_Time_RCD(data), dtype=float).ravel()
    if det == "CHARM_stat_kNN":
        return np.asarray(charm_stat_knn(data[:tr], data), dtype=float).ravel()
    if det == "OmniAnomaly":
        # OmniAnomaly.fit splits the training segment 80%/20% into train/valid, and valid needs >= win_size(100),
        # so the training segment needs at least ~500 points, otherwise ReconstructDataset crashes on an empty stack.
        # Short tr (especially when tr_ds shrinks after downsampling) is padded to 500. tqdm output is huge, so
        # suppress it with _quiet; progress is reported by the outer score_progress.
        min_train = 500
        train = data[:max(tr, min_train)]
        return np.asarray(_quiet(run_OmniAnomaly, train, data), dtype=float).ravel()
    raise ValueError(det)


def _worker(args):
    split, det, f, max_len = args
    stem = Path(f).stem
    out = BASE / split / det / f"{stem}.npy"
    if out.exists():
        return split, det, f, "skip"
    try:
        data = pd.read_csv(DATA_DIR / f).to_numpy(dtype=float)
        tr = tr_of(f)
        n = len(data)
        if max_len and n > max_len:
            # downsampling: sample one point every step, then linearly interpolate back to the original length (not skipped)
            step = int(np.ceil(n / max_len))
            idx = np.arange(0, n, step)
            data_ds = data[idx]
            tr_ds = int(np.searchsorted(idx, tr))
            s_ds = np.asarray(run_det(det, data_ds, tr_ds), dtype=float).ravel()
            s = np.interp(np.arange(n), idx, s_ds)
            np.save(out, s)
            return split, det, f, f"ds(x{step})"
        s = run_det(det, data, tr)
        np.save(out, s)
        return split, det, f, "done"
    except Exception as exc:  # noqa: BLE001
        # a single file failure does not abort the whole batch (e.g. OmniAnomaly can still crash on extremely short sequences), log FAILED and continue
        return split, det, f, f"FAILED:{type(exc).__name__}"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="multiprocessing count (Time_RCD is a large PyTorch model, too many workers OOMs; use 2 when memory is tight")
    ap.add_argument("--max-len", type=int, default=25000,
                    help="skip files exceeding this sample-point count (CHARM_stat_kNN's kNN distance is extremely slow on very large files")
    args = ap.parse_args()

    eva = pd.read_csv(EVA_LIST)["file_name"].tolist()
    tun = pd.read_csv(TUNING_LIST)["file_name"].tolist()
    splits = {"tuning": tun, "eva": eva}

    for split in splits:
        for det in DETS:
            (BASE / split / det).mkdir(parents=True, exist_ok=True)

    tasks = [(split, det, f, args.max_len) for split, files in splits.items()
             for det in DETS for f in files]  # detector-major: all local
    total = len(tasks)
    done = 0
    t_start = time.time()

    def _progress(split, det, f):
        counts = {f"{s}/{d}": len(list((BASE / s / d).glob("*.npy")))
                  for s in splits for d in DETS}
        state = {
            "done": done, "total": total,
            "started": t_start, "updated": time.time(),
            "current": {"split": split, "det": det, "file": f},
            "counts": counts,
        }
        (ROOT / "score_progress.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        # sync a JS data file for file://-opened HTML to read live via <script>
        (ROOT / "score_data.js").write_text(
            "window.__SCORE__ = " + json.dumps(state, ensure_ascii=False) + ";",
            encoding="utf-8")

    mp.set_start_method("spawn", force=True)
    with mp.Pool(args.workers) as pool:
        for split, det, f, status in pool.imap_unordered(_worker, tasks, chunksize=1):
            done += 1
            _progress(split, det, f)
            print(f"[{done}/{total}] {split}/{det}/{f} ({status})", flush=True)

    print(f"done {done}/{total}", flush=True)


if __name__ == "__main__":
    main()
