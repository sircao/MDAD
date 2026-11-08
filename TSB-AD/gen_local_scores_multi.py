"""gen_local_scores_multi.py —— Multivariate version: multiprocess parallel generation of 3 local detector scores (resume).

Fully decoupled from gen_local_scores.py (univariate): separate file, separate output dir eval/score/multi/,
separate progress files score_progress_multi.json / score_data_multi.js.

Data convention (this repo's local version):
  Dataset/TSB-AD-M/<name>.csv   only feature columns (header is channel indices 0,1,...,C-1), no Label
  Validation/TSB-AD-M/<name>.csv features + a final Label column
Only files present in Validation have labels; this script only generates scores for those (others are skipped).

Detectors (all local, multivariate-adapted):
  Syntropic      : per-channel WSK+DiffZ averaged + univariate Time-RCD (channel-mean compression, offline)
  Time_RCD       : multivariate checkpoint (pretrain_checkpoint_best_multi.pth, all channels input_c=C, offline)
  CHARM_stat_kNN : channel-collapsed window-statistics kNN (already handles multiple channels)
Output eval/score/multi/{tuning,eva}/{det}/<stem>.npy (full-length, higher = more anomalous).
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

from TSB_AD.model_wrapper import run_Time_RCD  # noqa: E402
from Run_CHARM import HP, minmax, multiscale_max  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-M"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-M"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-M-Eva.csv"
TUNING_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-M-Tuning.csv"
BASE = ROOT / "eval" / "score" / "multi"
DETS = ("Syntropic", "Time_RCD", "CHARM_stat_kNN")  # all local, no remote
PROG = ROOT / "score_progress_multi.json"
DATAJS = ROOT / "score_data_multi.js"


def tr_of(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args)


def charm_stat_knn(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """CHARM_kNN's local statistical component (per-window [std,range,max,min,mean] L2 kNN, no remote embedding).

    window_stats collapses over channels (axis=(1,2)), so it already handles multiple channels.
    """
    return minmax(multiscale_max(train, test, HP["stats_scales"], use_stats=True))


def _local_uni_checkpoint_dir() -> str:
    """Find the local Time-RCD univariate checkpoint directory (containing pretrain_checkpoint_best_uni.pth).

    Pass the local path directly to run_Time_RCD(checkpoint=...), bypassing hf_hub_download, fully offline.
    """
    fname = "pretrain_checkpoint_best_uni.pth"
    cands: list[Path] = []
    for d in (ROOT.parent / "Time-RCD" / "best_model",
              ROOT.parent / "checkpoints" / "time-rcd" / "best_model"):
        if (d / fname).exists():
            cands.append(d)
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--thu-sail-lab--Time-RCD"
    if cache.exists():
        for snap in sorted((cache / "snapshots").glob("*"), reverse=True):
            d = snap / "best_model"
            if (d / fname).exists():
                cands.append(d)
    if not cands:
        raise FileNotFoundError(
            f"cannot find the local Time-RCD uni checkpoint ({fname})."
            f"Please put it under Time-RCD/best_model/ or the HF cache directory and retry.")
    return str(cands[0])


def time_rcd_uni(data: np.ndarray) -> np.ndarray:
    """Time-RCD uses the locally cached univariate checkpoint (input_c=1), fully offline.

    Running Time-RCD directly on multivariate would download the multi checkpoint from HF (pretrain_checkpoint_best_multi.pth),
    which is absent locally and HF is unreachable. Here we first collapse multiple channels to a single channel by mean, then
    take the univariate path, and explicitly pass the local checkpoint path (no network).
    """
    if data.ndim == 1:
        x = data.reshape(-1, 1)
    else:
        x = data.mean(axis=1, keepdims=True)
    return np.asarray(run_Time_RCD(x, checkpoint=_local_uni_checkpoint_dir()), dtype=float).ravel()


PATCH_SIZE = 16
MAX_WIN_TOKENS = 2048   # per-window patch-sequence length cap (C * win/16), controls multi attention memory


def _local_multi_checkpoint() -> str:
    """Find the local Time-RCD multivariate checkpoint (pretrain_checkpoint_best_multi.pth).

    Pass the local path directly to run_Time_RCD(checkpoint=...), bypassing hf_hub_download, fully offline.
    """
    fname = "pretrain_checkpoint_best_multi.pth"
    cands: list[Path] = [
        ROOT.parent / fname,
        ROOT.parent / "Time-RCD" / "best_model" / fname,
        ROOT.parent / "checkpoints" / "time-rcd" / "best_model" / fname,
    ]
    for p in cands:
        if p.exists():
            return str(p)
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--thu-sail-lab--Time-RCD"
    if cache.exists():
        for snap in sorted((cache / "snapshots").glob("*"), reverse=True):
            p = snap / "best_model" / fname
            if p.exists():
                return str(p)
    raise FileNotFoundError(
        f"cannot find the local Time-RCD multi checkpoint ({fname})."
        f"Please put it in the project root / Time-RCD/best_model/ or the HF cache directory and retry.")


def _time_rcd_win_size(C: int) -> int:
    """Choose win_size by channel count so each window's token count (C * win/16) ≤ MAX_WIN_TOKENS.

    Time-RCD multi's attention sequence length ≈ C * win_size / patch_size; high-channel files (e.g. 248 channels)
    OOM with the default win_size=15000. Here we shrink the window by channel count; the checkpoint weights themselves
    are channel-independent (checkpoint config num_features=8 only affects the token count, not C).
    """
    win = int(MAX_WIN_TOKENS * PATCH_SIZE / max(int(C), 1))
    return int(np.clip(win, PATCH_SIZE, 15000))


def time_rcd_multi(data: np.ndarray) -> np.ndarray:
    """Time-RCD multivariate: all channels + local multi checkpoint (input_c=C), fully offline.

    Difference from time_rcd_uni: it does not collapse the channel mean into a single channel; it feeds C channels
    directly to the multi checkpoint, and adapts win_size by channel count to control attention memory.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    win = _time_rcd_win_size(data.shape[1])
    return np.asarray(run_Time_RCD(data, checkpoint=_local_multi_checkpoint(), win_size=win),
                      dtype=float).ravel()


def syntropic_multi(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Syntropic's multivariate adaptation.

    The original univariate Syntropic does test.ravel() to flatten all channels (length becomes L*C, and cross-channel windows are meaningless),
    so it cannot be used directly on multivariate. Here we change it to:
      - WSK, DiffZ computed per channel separately, averaged across channels (weight 2:1, matching the original Syntropic)
      - Time-RCD uses the univariate checkpoint + channel-mean compression (offline, see time_rcd_uni)
    Combination = mean_c(2*norm(WSK_c) + 1*norm(DiffZ_c)) + 0.5*norm(Time-RCD_uni).
    Scores are only for ranking (subsequent rank01 / VUS-PR), so overall scaling does not affect results.
    """
    from TSB_AD.models.Syntropic import _normalise, diffz, wsk  # noqa: E402

    test = np.asarray(test, dtype=float)
    train = np.asarray(train, dtype=float)
    if test.ndim == 1:
        test = test.reshape(-1, 1)
        train = train.reshape(-1, 1)
    n, C = test.shape
    train_len = len(train)

    base = np.zeros(n)
    for c in range(C):
        x = test[:, c]
        base += 2.0 * _normalise(wsk(x, train_len)) + 1.0 * _normalise(diffz(x, train_len))
    base /= max(C, 1)

    trcd = time_rcd_uni(test)
    return base + 0.5 * _normalise(trcd)


def run_det(det: str, data: np.ndarray, tr: int) -> np.ndarray:
    if det == "Syntropic":
        return np.asarray(syntropic_multi(data[:tr], data), dtype=float).ravel()
    if det == "Time_RCD":
        return np.asarray(time_rcd_multi(data), dtype=float).ravel()
    if det == "CHARM_stat_kNN":
        return np.asarray(charm_stat_knn(data[:tr], data), dtype=float).ravel()
    raise ValueError(det)


def _worker(args):
    split, det, f, max_len = args
    stem = Path(f).stem
    out = BASE / split / det / f"{stem}.npy"
    if out.exists():
        return split, det, f, "skip"
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


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="multiprocessing count (Time_RCD is a large PyTorch model, too many workers OOMs; use 2 when memory is tight")
    ap.add_argument("--max-len", type=int, default=25000,
                    help="downsample to no more than this sample-point count (CHARM_stat_kNN's kNN distance is extremely slow on very large files")
    args = ap.parse_args()

    # only keep files with labels (present as label in Validation/TSB-AD-M)
    labeled = set(os.listdir(LABEL_DIR))
    eva = [f for f in pd.read_csv(EVA_LIST)["file_name"].tolist() if f in labeled]
    tun = [f for f in pd.read_csv(TUNING_LIST)["file_name"].tolist() if f in labeled]
    splits = {"tuning": tun, "eva": eva}
    print(f"labeled files -> tuning {len(tun)} / eva {len(eva)}", flush=True)

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
        PROG.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        DATAJS.write_text("window.__SCORE_MULTI__ = " + json.dumps(state, ensure_ascii=False) + ";",
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
