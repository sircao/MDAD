"""eval_loop_tpfp.py —— Two-level routing + TP/FP-level model-selection-parameter complementary loop.

Architecture (aligned with the paper's two-level routing):
  Level 1  morphology features → direction routing (extract_evidence determines morphology, label-free; can attach LLM consensus fallback).
  Level 2  morphology prior → dispatch selects the optimal detector (the base "best" with the highest VUS-PR for that morphology).
  Loop     complements the chosen best base, and **starts from the SOTA baseline** (fused = best's per-point scores).

The loop's core (the part this file emphasizes) — learn one model-selection parameter on each of the two dimensions, misjudged TP / misjudged FP:
  tp_model[morphology]  = the detector that scores "TP the base misjudged (missed detection: true anomaly but low score)" most correctly → raise the score.
  fp_model[morphology]  = the detector that scores "FP the base misjudged (false alarm: normal but high score)" most correctly → lower the score.
  Each round the loop re-measures the TP/FP still misjudged on the current fused, re-selects tp_model/fp_model by correction gain,
  until that morphology is "judged correct" (the in-morphology mean VUS-PR reaches the SOTA single-detector level, or stops improving).

Protocol: Tuning (48, labeled) learns per-morphology parameters and freezes them; Eva (350) labels are only for reporting (this file is like
eval_loop_fullshot, the fullshot upper-bound line — judging misclassified TP/FP requires labels).
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import time
from collections import defaultdict
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

TUNING_DIR = ROOT / "eval" / "score" / "uni" / "tuning"
EVA_DIR = ROOT / "eval" / "score" / "uni" / "eva"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
TUNING_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"

DETS = ["Syntropic", "CHARM_kNN", "Time_RCD", "CHARM_ZS"]
MAX_OUTER_ITER = 20
PROG = ROOT / "tpfp_progress.json"
DATAJS = ROOT / "tpfp_data.js"


def _write_progress(state: dict) -> None:
    state["updated"] = time.time()
    state["elapsed"] = round(state["updated"] - state.get("started", state["updated"]), 1)
    done = state.get("done", 0)
    total = state.get("total", 0)
    if done > 0 and total > 0:
        rate = state["elapsed"] / done
        state["eta"] = round((total - done) * rate, 1)
    else:
        state["eta"] = None
    PROG.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    # sync a JS data file for file://-opened HTML to read live via <script src>
    js = "window.__TPFP__ = " + json.dumps(state, ensure_ascii=False) + ";"
    DATAJS.write_text(js, encoding="utf-8")


def rank01(s: np.ndarray) -> np.ndarray:
    return rankdata(s) / len(s)


def _vus(s: np.ndarray, y: np.ndarray, sw: int) -> float:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return float(get_metrics(np.asarray(s, dtype=float).ravel(), y.astype(int),
                                 slidingWindow=sw)["VUS-PR"])


def tr_of(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def ev_morph_support(data_full: np.ndarray, tr: int):
    ev = extract_evidence(data_full, reference=data_full[:tr])
    return str(ev.summary()["morphology"]), ev.support


def load_scores(name: str, det: str, base_dir: Path) -> np.ndarray:
    stem = Path(name).stem
    return np.load(base_dir / det / f"{stem}.npy", allow_pickle=False)


def best_f1_threshold(s: np.ndarray, y: np.ndarray) -> float:
    """F1-optimal decision threshold on the fused scores, used to split "misjudged TP (miss) / misjudged FP (false alarm)"."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=int)
    pos = s[y == 1]
    if len(pos) == 0:
        return float(np.percentile(s, 50))
    best, best_tau = -1.0, float(np.median(pos))
    for t in np.percentile(s, np.linspace(5, 95, 37)):
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        if f1 > best:
            best, best_tau = f1, float(t)
    return best_tau


# ---------------------------------------------------------------------------
# TP/FP-level correction-gain measurement
# ---------------------------------------------------------------------------
def measure_tpfp_gain(fused: np.ndarray, S: dict[str, np.ndarray], y: np.ndarray) -> dict:
    """Measure each detector's correction gain at the TP/FP positions the fused currently misjudges.

    Misjudged TP (miss) = true anomaly but low score → raise gain = sum(clip(s - fused, 0, None)).
    Misjudged FP (false alarm) = normal but high score → lower gain = sum(clip(fused - s, 0, None)).
    """
    tau = best_f1_threshold(fused, y)
    miss = (y == 1) & (fused < tau)
    fa = (y == 0) & (fused > tau)
    tp_gain, fp_gain, dets = {}, {}, list(S.keys())
    for d in dets:
        s = S[d]
        tp_gain[d] = float(np.clip(s[miss] - fused[miss], 0.0, None).sum()) if miss.any() else 0.0
        fp_gain[d] = float(np.clip(fused[fa] - s[fa], 0.0, None).sum()) if fa.any() else 0.0
    return {"miss": miss, "fa": fa, "tp_gain": tp_gain, "fp_gain": fp_gain,
            "tp_n": int(miss.sum()), "fp_n": int(fa.sum())}


def apply_tpfp(fused: np.ndarray, S: dict[str, np.ndarray], y: np.ndarray,
               tp_model: str | None, fp_model: str | None) -> np.ndarray:
    """By the TP/FP model-selection parameters, only raise misjudged TP and only lower misjudged FP."""
    cur = fused.copy()
    tau = best_f1_threshold(cur, y)
    miss = (y == 1) & (cur < tau)
    fa = (y == 0) & (cur > tau)
    if tp_model is not None and miss.any():
        s = S[tp_model]
        m = miss & (s > cur)
        cur[m] = s[m]
    if fp_model is not None and fa.any():
        s = S[fp_model]
        m = fa & (s < cur)
        cur[m] = s[m]
    return cur


# ---------------------------------------------------------------------------
# per-morphology parameter-learning loop: keep updating tp_model / fp_model until that morphology is judged correct
# ---------------------------------------------------------------------------
def learn_morph_params(files: list[str], data: dict) -> dict:
    """Learn parameters for all Tuning files of one morphology.

    data[f] = {"S": {det: rank01 per-point scores}, "label": label, "sw": window, "vus": {det: VUS-PR}}
    Returns {best, tp_model, fp_model, sota_vus, final_vus, converged, iterations, trace}.
    """
    avail = sorted({d for f in files for d in data[f]["S"]})
    # —— level 2: morphology prior picks the optimal detector (highest mean VUS-PR for that morphology) ——
    mean_vus = {d: float(np.mean([data[f]["vus"][d] for f in files])) for d in avail}
    best = max(mean_vus, key=mean_vus.get)
    sota_vus = float(np.mean([max(data[f]["vus"].values()) for f in files]))

    # —— weight parameter w: weighted fusion of best and second-best, grid-searched optimum ——
    other = sorted(avail, key=lambda d: mean_vus[d])[-2] if len(avail) >= 2 else best
    best_w, fused_w_vus = 0.5, -1.0
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        wv = float(np.mean([_vus(w * data[f]["S"][best] + (1 - w) * data[f]["S"][other],
                                data[f]["label"], data[f]["sw"]) for f in files]))
        if wv > fused_w_vus:
            fused_w_vus, best_w = wv, w

    # —— start from the SOTA baseline: fused = best base's per-point scores ——
    fused = {f: data[f]["S"][best].copy() for f in files}
    cur_v = {f: data[f]["vus"][best] for f in files}   # reuse the already-computed base VUS-PR
    prev_v = float(np.mean(list(cur_v.values())))
    tp_model = fp_model = None
    converged = False
    trace = []

    for it in range(MAX_OUTER_ITER):
        # 1. aggregate the TP/FP gains currently misjudged across all files
        tp_gain = defaultdict(float)
        fp_gain = defaultdict(float)
        tp_n = fp_n = 0
        for f in files:
            g = measure_tpfp_gain(fused[f], data[f]["S"], data[f]["label"])
            tp_n += g["tp_n"]; fp_n += g["fp_n"]
            for d in avail:
                tp_gain[d] += g["tp_gain"][d]
                fp_gain[d] += g["fp_gain"][d]

        # 2. re-select the TP/FP model-selection parameters (detectors with the highest gain)
        new_tp = max(avail, key=lambda d: tp_gain[d]) if tp_n > 0 else None
        new_fp = max(avail, key=lambda d: fp_gain[d]) if fp_n > 0 else None

        # 3. apply the parameters and compute the new VUS-PR (once per file per round)
        new_fused = {f: apply_tpfp(fused[f], data[f]["S"], data[f]["label"], new_tp, new_fp)
                     for f in files}
        new_v = {f: _vus(new_fused[f], data[f]["label"], data[f]["sw"]) for f in files}
        mean_v = float(np.mean(list(new_v.values())))

        tp_model, fp_model = new_tp, new_fp
        trace.append({"iter": it + 1, "tp_model": new_tp, "fp_model": new_fp,
                      "tp_n": tp_n, "fp_n": fp_n, "mean_vus": round(mean_v, 4),
                      "best_gain_tp": round(tp_gain.get(new_tp, 0.0), 3) if new_tp else 0.0,
                      "best_gain_fp": round(fp_gain.get(new_fp, 0.0), 3) if new_fp else 0.0})

        # 4. judged correct? (reached the SOTA single-detector level)
        if mean_v >= sota_vus + 1e-4:
            fused, cur_v = new_fused, new_v
            converged = True
            break
        # no improvement → stop (keep the previous round, do not use the worse result)
        if mean_v <= prev_v + 1e-6:
            break
        fused, cur_v, prev_v = new_fused, new_v, mean_v

    final_vus = float(np.mean(list(cur_v.values())))
    if converged:
        status = "converged"
    elif final_vus >= sota_vus - 1e-4:
        status = "at_sota"   # the base itself is already at SOTA, the loop cannot find better, correctly keep the base
    else:
        status = "no_improve"  # loop stopped but did not reach SOTA (complement pool has no better correction)
    return {"best": best, "other": other, "tp_model": tp_model, "fp_model": fp_model,
            "w": best_w, "sota_vus": round(sota_vus, 4), "final_vus": round(final_vus, 4),
            "base_vus": round(mean_vus[best], 4),
            "fused_w_vus": round(fused_w_vus, 4),
            "status": status, "iterations": len(trace), "n_files": len(files),
            "trace": trace, "loop_vus_per_file": {f: round(cur_v[f], 4) for f in files}}


def load_file_data(name: str, base_dir: Path, max_len: int | None = None) -> dict | None:
    """Read a file's available detector scores + label + window. At least 2 detectors required to be usable."""
    data_full = pd.read_csv(DATA_DIR / name).to_numpy(dtype=float)
    if max_len is not None and len(data_full) > max_len:
        return None
    tr = tr_of(name)
    labels = pd.read_csv(LABEL_DIR / name)["Label"].to_numpy(dtype=float)
    sw = int(find_length_rank(data_full[:, 0].reshape(-1, 1), rank=1))

    S = {}
    for d in DETS:
        p = base_dir / d / f"{Path(name).stem}.npy"
        if p.exists():
            S[d] = rank01(np.load(p, allow_pickle=False))
    if len(S) < 2:
        return None

    vus = {d: _vus(S[d], labels, sw) for d in S}
    return {"S": S, "label": labels, "sw": sw, "vus": vus, "morph": None}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tuning", "eva"], default="tuning")
    ap.add_argument("--apply-eva", action="store_true",
                    help="after freezing Tuning parameters, run the fullshot upper bound on Eva")
    ap.add_argument("--max-len", type=int, default=15000,
                    help="skip files exceeding this sample-point count (the official VUS-PR metric is very slow on large files)")
    args = ap.parse_args()

    base_dir = TUNING_DIR if args.split == "tuning" else EVA_DIR
    files = pd.read_csv(TUNING_LIST if args.split == "tuning" else EVA_LIST)["file_name"].tolist()

    print(f"eval_loop_tpfp.py started: split={args.split} max_len={args.max_len} "
          f"total files={len(files)} detectors={DETS}", flush=True)

    state = {"phase": "loading", "split": args.split, "max_len": args.max_len,
             "detectors": DETS, "total": len(files), "done": 0, "usable": 0,
             "started": time.time(), "files": [], "rule": {}, "aggregate": {},
             "current": None, "skipped": []}
    _write_progress(state)
    print("  [progress] wrote tpfp_progress.json (refresh the dashboard to see progress)", flush=True)

    # —— read usable files + level-1 morphology classification ——
    data = {}
    morph_of = {}
    for i, f in enumerate(files, 1):
        t0 = time.time()
        d = load_file_data(f, base_dir, args.max_len)
        if d is None:
            state["skipped"].append(f)
            state["done"] = i
            _write_progress(state)
            print(f"  [{i}/{len(files)}] skip {f} (<2 detector scores or len>{args.max_len})", flush=True)
            continue
        data_full = pd.read_csv(DATA_DIR / f).to_numpy(dtype=float)
        morph, _ = ev_morph_support(data_full, tr_of(f))
        d["morph"] = morph
        data[f] = d
        morph_of[f] = morph
        state["done"] = i
        state["usable"] = len(data)
        state["files"].append({"file": f, "morph": morph, "len": int(len(data_full)),
                               "vus": {k: round(v, 4) for k, v in d["vus"].items()}})
        state["current"] = {"file": f, "morph": morph, "step": "level-1 morphology classification"}
        _write_progress(state)
        print(f"  [{i}/{len(files)}] {f} morph={morph} "
              f"({time.time()-t0:.1f}s, usable {len(data)})", flush=True)

    print(f"\nusable files {len(data)}/{len(files)} (at least 2 detectors have scores)", flush=True)

    # —— group by morphology ——
    by_morph = defaultdict(list)
    for f, d in data.items():
        by_morph[d["morph"]].append(f)

    print("\n==== morphology distribution ====")
    for m in sorted(by_morph):
        print(f"  {m:18s} {len(by_morph[m])} files")

    # —— learn TP/FP model-selection parameters per morphology ——
    rule = {}
    print("\n==== per-morphology TP/FP model-selection parameter learning ====")
    for m, fs in sorted(by_morph.items(), key=lambda kv: -len(kv[1])):
        if len(fs) < 1:
            continue
        state["phase"] = "learning"
        state["current"] = {"morph": m, "step": "level-2 routing + loop parameter learning"}
        _write_progress(state)
        r = learn_morph_params(fs, data)
        rule[m] = r
        # backfill each file's loop VUS-PR into the live table
        for row in state["files"]:
            if row["file"] in r["loop_vus_per_file"]:
                row["loop_vus"] = r["loop_vus_per_file"][row["file"]]
                row["best"] = r["best"]
        state["rule"][m] = {k: r[k] for k in ("best", "other", "tp_model", "fp_model",
                                              "w", "sota_vus", "base_vus", "fused_w_vus",
                                              "final_vus", "status", "iterations",
                                              "n_files", "trace")}
        mark = {"converged": "judged correct (reached SOTA)", "at_sota": "base already at SOTA",
                "no_improve": "complement has no better"}.get(r["status"], r["status"])
        print(f"  {m:18s} best={r['best']:14s} w={r['w']:.2f} "
              f"tp_model={str(r['tp_model']):14s} fp_model={str(r['fp_model']):14s} "
              f"base={r['base_vus']:.3f} -> loop={r['final_vus']:.3f} "
              f"(SOTA={r['sota_vus']:.3f}) iters={r['iterations']} [{mark}]", flush=True)
        for t in r["trace"]:
            print(f"     round {t['iter']} tp_model={t['tp_model']} fp_model={t['fp_model']} "
                  f"tp_n={t['tp_n']} fp_n={t['fp_n']} VUS-PR={t['mean_vus']:.4f}", flush=True)
        _write_progress(state)

    # —— summary: base vs loop ——
    print("\n==== summary (in-morphology mean VUS-PR) ====")
    agg_base, agg_loop, agg_sota = [], [], []
    for m, r in rule.items():
        agg_base.append(r["base_vus"])
        agg_loop.append(r["final_vus"])
        agg_sota.append(r["sota_vus"])
        print(f"  {m:18s} best={r['best']:14s} w={r['w']:.2f} tp={str(r['tp_model']):14s} "
              f"fp={str(r['fp_model']):14s} base={r['base_vus']:.3f} loop={r['final_vus']:.3f} "
              f"SOTA={r['sota_vus']:.3f}")
    if agg_base:
        state["aggregate"] = {
            "base": round(float(np.mean(agg_base)), 4),
            "loop": round(float(np.mean(agg_loop)), 4),
            "sota": round(float(np.mean(agg_sota)), 4),
            "delta": round(float(np.mean(agg_loop) - np.mean(agg_base)), 4),
        }
        print(f"\n  base(level-2 routing picks best) = {state['aggregate']['base']:.4f}")
        print(f"  loop(TP/FP parameter complement) = {state['aggregate']['loop']:.4f}  "
              f"(Δ {state['aggregate']['delta']:+.4f})")
        print(f"  SOTA(single-detector upper bound)   = {state['aggregate']['sota']:.4f}")

    state["phase"] = "done"
    _write_progress(state)

    (ROOT / "runs" / "tpfp_params.json").write_text(
        json.dumps({"split": args.split, "rule": rule, "morph_of": morph_of},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("\n[saved] runs/tpfp_params.json", flush=True)
    print("[saved] tpfp_progress.json", flush=True)


if __name__ == "__main__":
    main()
