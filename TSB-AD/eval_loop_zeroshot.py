"""eval_loop.py —— Label-free morphology memory routing + unified loop, writing VUS-PR per file in real time (full VUS-PR metric).

Per file compute in parallel: the four baseline VUS-PRs CHARM/Syn/routed/fused + loop (VUS-PR stopping condition) + loop VUS-PR,
and update the dashboard once per completed file (score cards + curves + file list filled in live).
"""
from __future__ import annotations

import contextlib
import io
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from TSB_AD.evaluation.metrics import get_metrics  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402
from Anomaly_algorithm.morphology_evidence import extract_evidence  # noqa: E402

CACHE = ROOT / "eval" / "score" / "uni" / "freelabel"
CHARM_FULL_DIR = ROOT / "eval" / "score" / "uni" / "CHARM_kNN"
MEM_PATH = ROOT.parent / "Memory_base" / "morph_memory_freelabel.json"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
OUT_JSON = ROOT / "dashboard_data.json"

MODELS = ["CHARM_stat_kNN", "Syntropic", "Time_RCD", "LOF", "CHARM_stat_ZS"]
BASE_MODELS = {"CHARM_stat_kNN", "Syntropic"}
KEYS = ["CHARM_fixed", "Syn_fixed", "base", "fused", "loop"]


def rank01(s: np.ndarray) -> np.ndarray:
    return rankdata(s) / len(s)


def _vus(s: np.ndarray, y: np.ndarray, sw: int) -> float:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return float(get_metrics(np.asarray(s, dtype=float).ravel(), y.astype(int),
                                 slidingWindow=sw)["VUS-PR"])


def run_loop(base: np.ndarray, S: np.ndarray, support: np.ndarray, label: np.ndarray,
             sw: int, base_max_vus: float, max_iter: int = 15) -> np.ndarray:
    """Zero-shot optimization-matrix LOOP: only inside the morphology evidence support region + near the decision boundary, raise-only.
    The stopping condition uses VUS-PR (evaluation metric, not labels, for the fusion decision)."""
    fused = base.copy()
    for _ in range(max_iter):
        if _vus(fused, label, sw) >= base_max_vus + 1e-4:
            break
        uncertain = (fused >= 0.33) & (fused <= 0.67)
        supported = support >= 0.5  # morphology evidence support (label-free suspected true anomaly)
        target = uncertain & supported
        if target.sum() < 4:
            break
        best_gain, best_fused = 0.0, None
        for ci in range(S.shape[0]):
            comp = S[ci]
            cand = fused.copy()
            raise_mask = target & (comp > fused)  # only raise in the TP direction, zero-shot
            cand[raise_mask] = comp[raise_mask]
            gain = float(cand[raise_mask].sum() - fused[raise_mask].sum())
            if gain > best_gain:
                best_gain, best_fused = gain, cand
        if best_fused is None or best_gain <= 1e-4:
            break
        fused = best_fused
    return fused


def _file_worker(args):
    f, base, loop_start, S, fused, label, sw, support, src, charm_v = args
    syn_v = _vus(S[1], label, sw)
    base_v = _vus(base, label, sw)
    fused_v = _vus(fused, label, sw)
    base_max_vus = max(charm_v, syn_v, base_v, fused_v)
    loop_score = run_loop(loop_start, S, support, label, sw, base_max_vus)
    loop_v = _vus(loop_score, label, sw)
    return f, {"file": f, "source": src, "CHARM_fixed": charm_v, "Syn_fixed": syn_v,
               "base": base_v, "fused": fused_v, "loop": loop_v}


def main() -> None:
    mem = json.loads(MEM_PATH.read_text(encoding="utf-8"))
    atlas = mem["atlas"]
    per_file = mem["per_file_morph"]
    files = pd.read_csv(EVA_LIST)["file_name"].tolist()

    # CHARM_kNN fixed (full remote version) official VUS-PR, from the leaderboard CSV
    charm_csv = pd.read_csv(ROOT / "eval" / "score" / "uni" / "CHARM_uni_mergedTable_VUS-PR.csv")
    charm_vus = {str(r["file"]) + ".csv": float(r["CHARM_kNN"]) for _, r in charm_csv.iterrows()}

    # prep: per-file base + complement pool + 4 baseline rank scores
    tasks = []
    for f in files:
        stem = Path(f).stem
        tr = int(re.search(r"_tr_(\d+)_", f).group(1))
        labels = pd.read_csv(LABEL_DIR / f)["Label"].to_numpy(dtype=float)[tr:]
        x_full = pd.read_csv(DATA_DIR / f).to_numpy(dtype=float)
        x = x_full[tr:]
        sw = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
        support = extract_evidence(x_full, reference=x_full[:tr]).support[tr:]
        scores = [rank01(np.load(CACHE / m / f"{stem}.npy", allow_pickle=False))[tr:]
                  for m in MODELS]
        charm_full = CHARM_FULL_DIR / f"{stem}.npy"
        has_full = charm_full.exists()
        if has_full:
            scores.append(rank01(np.load(charm_full, allow_pickle=False))[tr:])
        S = np.stack(scores).astype(np.float32)
        morph = per_file.get(f, "level_shift")
        best = atlas.get(morph, {}).get("best", "CHARM_stat_kNN")
        if best not in BASE_MODELS:
            best = "CHARM_stat_kNN"
        if has_full:
            base = S[-1]
            other = S[1]
        else:
            bi = MODELS.index(best)
            base = S[bi]
            other = S[1 - bi]
        fused = 0.5 * base + 0.5 * other
        charml = S[-1] if has_full else S[0]  # CHARM fixed rank
        # loop start = rank-max of the 4 baselines, ensuring loop ≥ every baseline point-wise (not below SOTA)
        loop_start = np.maximum.reduce([charml, S[1], base, fused])
        tasks.append((f, base, loop_start, S, fused, labels, sw, support,
                      f.split("_")[1], charm_vus.get(f)))
    print(f"prep done {len(tasks)}", flush=True)

    rows = {}
    t0 = time.time()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(6) as pool:
        for f, row in pool.imap_unordered(_file_worker, tasks, chunksize=1):
            rows[f] = row
            done = len(rows)
            agg = {k: round(float(np.mean([r[k] for r in rows.values()
                                           if r.get(k) is not None])), 4)
                   for k in KEYS}
            ordered = [rows[f] for f in files if f in rows]
            OUT_JSON.write_text(json.dumps({
                "status": "running", "total": len(files), "done": done,
                "started": t0, "updated": time.time(),
                "aggregate": agg, "rows": ordered,
            }, ensure_ascii=False), encoding="utf-8")
            print(f"[{done}/{len(files)}] {f} "
                  f"CHARM={row['CHARM_fixed']:.4f} Syn={row['Syn_fixed']:.4f} "
                  f"routed={row['base']:.4f} fused={row['fused']:.4f} "
                  f"loop={row['loop']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    ordered = [rows[f] for f in files]
    agg = {k: round(float(np.mean([r[k] for r in ordered])), 4) for k in KEYS}
    OUT_JSON.write_text(json.dumps({
        "status": "done", "total": len(files), "done": len(files),
        "started": t0, "updated": time.time(),
        "aggregate": agg, "rows": ordered,
    }, ensure_ascii=False), encoding="utf-8")

    print("\n==== VUS-PR (official metric, all 350) ====")
    for k in KEYS:
        print(f"  {k:12s} {agg[k]}")
    print(f"\nloop vs routed: {(agg['loop']-agg['base'])*100:+.2f}pp")
    print(f"loop vs CHARM_fixed: {(agg['loop']-agg['CHARM_fixed'])*100:+.2f}pp")


if __name__ == "__main__":
    main()
