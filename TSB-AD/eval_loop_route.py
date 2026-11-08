"""eval_loop_route.py — Morphology+source multi-level routing (vectorized VUS-PR + multiprocessing).

Routing levels (fall back level by level; learn parameters only when a bucket has ≥3 files):
  (fine_morph, source) ≥3 → learn that bucket's parameters            # fine_morph = primary morphology + secondary morphology (≤2)
  otherwise fine_morph ≥3       → learn fine-morphology-level parameters
  otherwise (coarse_morph, source) ≥3 → learn coarse-morphology+source level (keeping the source prior)
  otherwise coarse_morph ≥3     → learn coarse-morphology-level parameters (the original single morphology)
  otherwise                     → global best

Tuning learns parameters (labels are legitimate): the loop uses label-free quantile positions (positions_zeroshot_q, identical to Eva)
to jointly learn (best, tp_model, fp_model, K); labels are only used for fast_vus scoring/selecting/stopping —
train/test positions are consistent, so parameters generalize directly to Eva.

The Eva phase has frozen, label-free parameters with no label leakage: decisions use only S/support/morph/source (never touching
label/vus), the zero-shot loop raises missed detections / suppresses false alarms, accepted by a label-free gain criterion
(gain = rank-consistency gain + support-weighted raise/lower magnitude); the threshold gain_tau is calibrated and frozen with labels
in tuning, falling back to best when the criterion is not met; labels are used only once, at the very end, to report VUS-PR.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402
from Anomaly_algorithm.morphology_evidence import extract_evidence  # noqa: E402
from fast_vus import fast_vus  # noqa: E402

try:
    from llm_verify_best import verify_rules, apply_corrections  # noqa: E402
except Exception:  # noqa: BLE001
    verify_rules = None
    apply_corrections = None

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
TUNING_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
SCORE_DIR = ROOT / "eval" / "score" / "uni" / "tuning"
EVA_SCORE_DIR = ROOT / "eval" / "score" / "uni" / "eva"
PARAMS = ROOT / "runs" / "route_params.json"
DETS = ["Syntropic", "Time_RCD", "CHARM_stat_kNN", "OmniAnomaly"]  # all local, no remote
MAX_ITER = 50   # strengthen: the loop keeps iterating while there is gain, until VUS-PR=1 or no more gain (negative/no gain stops)
PROG = ROOT / "route_progress.json"
DATAJS = ROOT / "route_data.js"


def _write_route(state: dict) -> None:
    state["updated"] = time.time()
    state["elapsed"] = round(state["updated"] - state.get("started", state["updated"]), 1)
    done, total = state.get("done", 0), state.get("total", 0)
    state["eta"] = round((total - done) * state["elapsed"] / done, 1) if done > 0 else None
    PROG.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    DATAJS.write_text("window.__ROUTE__ = " + json.dumps(state, ensure_ascii=False) + ";",
                      encoding="utf-8")
    cur = state.get("current")
    cur_name = cur.get("file", "") if isinstance(cur, dict) else ""
    print(f"[save] {state.get('phase', '?'):8s} {done}/{total}  {cur_name}  "
          f"-> route_progress.json + route_data.js", flush=True)


def tr_of(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def source_of(f: str) -> str:
    return f.split("_")[1]


def rank01(s: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(s, dtype=float).ravel()) / len(s)


DROP_KINDS = {"relationship_change"}   # fires 0/48 on TSB-AD-U tuning, pure noise, dropped when routing


def morph_signature(summary: dict, sec_frac: float = 0.05, sec_strength: float = 0.6,
                    max_secondary: int = 2) -> tuple[str, str]:
    """Split Evidence.summary()'s ranked evidence into a finer morphology signature.

    Returns (fine, coarse). coarse = primary morphology (equivalent to the original single morphology); fine = primary + meaningful secondary
    morphologies (at most 2, capturing top-3 kind mixed anomalies), so tuning can split buckets finer → eva routes best more accurately.
    Drop DROP_KINDS (non-discriminative categories) first, then take the primary/secondary morphologies; if all are dropped, fall back to unknown.
    """
    ranked = [r for r in (summary.get("ranked") or [])
              if r.get("morphology") not in DROP_KINDS]
    if not ranked:
        return "unknown", "unknown"
    primary = str(ranked[0].get("morphology", "unknown"))
    secondaries = [
        str(r.get("morphology", "")) for r in ranked[1:]
        if r.get("supported_fraction", 0.0) >= sec_frac
        and r.get("strength", 0.0) >= sec_strength
        and str(r.get("morphology", "")) != primary
        and str(r.get("morphology", "")) not in DROP_KINDS
    ]
    fine = "+".join([primary] + secondaries[:max_secondary])
    return fine, primary


def best_f1_threshold(s: np.ndarray, y: np.ndarray) -> float:
    s = np.asarray(s, dtype=float); y = np.asarray(y, dtype=int)
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


def positions_labeled(fused, label):
    tau = best_f1_threshold(fused, label)
    miss = (label == 1) & (fused < tau)
    fa = (label == 0) & (fused > tau)
    return miss, fa


def positions_zeroshot(fused, support):
    """label-free: high evidence support = suspected anomaly (raise), low evidence support = suspected normal (lower). Eva uses this, no label leakage."""
    miss = (support >= 0.5) & (fused < 0.5)
    fa = (support < 0.5) & (fused > 0.5)
    return miss, fa


def positions_zeroshot_q(fused, support, support_hi_q=0.9, fused_lo_q=0.3,
                         support_lo_q=0.1, fused_hi_q=0.7):
    """Find missed detections (raise) and false alarms (lower) by quantile thresholds.

    miss: support in the top support_hi_q quantile (default 0.9 = top 10%) and fused in the lowest fused_lo_q quantile (default 0.3).
    fa:   support in the bottom support_lo_q quantile (default 0.1) and fused in the highest fused_hi_q quantile (default 0.7).
    """
    s_hi = float(np.quantile(support, support_hi_q))
    f_lo = float(np.quantile(fused, fused_lo_q))
    miss = (support >= s_hi) & (fused < f_lo)
    s_lo = float(np.quantile(support, support_lo_q))
    f_hi = float(np.quantile(fused, fused_hi_q))
    fa = (support <= s_lo) & (fused > f_hi)
    return miss, fa


def _consistency(a, b) -> float:
    """Rank consistency of two signals (Spearman approximation), used by the label-free gain criterion."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(rankdata(a), rankdata(b))[0, 1]
    return float(r) if np.isfinite(r) else 0.0


# fixed lower bounds of the label-free gain criterion (structural fallback); the core threshold gain_tau is calibrated with labels in tuning
MIN_CHANGE_FRAC = 0.01   # raise/lower points must be at least 1%, otherwise the loop has no effect
MIN_SEP = 0.05           # median(support) of raised points and median(support) of lowered points must be at least 0.05 apart
DEFAULT_GAIN_TAU = 1e-3  # conservative default when there is no calibration data


def label_free_gain(d, base, cand) -> dict:
    """label-free gain signal: how trustworthy the loop's raise/lower is relative to best (no labels).

    gain = rank-consistency gain + support-weighted raise/lower magnitude (score-increase magnitude).
    The consistency gain measures whether fused's rank consistency with the evidence support improves over base;
    the magnitude measures the actual movement of "raise (high support) / lower (low support)". The larger both are, the more trustworthy.
    """
    eps = 1e-4
    n = len(base)
    movement = cand - base
    raised = movement > eps
    lowered = movement < -eps
    n_raised, n_lowered = int(raised.sum()), int(lowered.sum())
    change_frac = (n_raised + n_lowered) / max(n, 1)

    support = d["support"]
    s_raised = support[raised]
    s_lowered = support[lowered]
    sep = float(np.median(s_raised) - np.median(s_lowered)) if n_raised and n_lowered else 0.0

    consistency_gain = _consistency(cand, support) - _consistency(base, support)
    mag_raise = float(np.sum(support[raised] * movement[raised])) if n_raised else 0.0
    mag_lower = float(np.sum((1.0 - support[lowered]) * (-movement[lowered]))) if n_lowered else 0.0
    magnitude = (mag_raise + mag_lower) / max(n, 1)
    return {"change_frac": change_frac, "sep": sep,
            "consistency_gain": consistency_gain, "magnitude": magnitude,
            "gain": consistency_gain + magnitude}


def apply_q(fused, S, miss, fa, tp_model, fp_model):
    """Raise missed detections + suppress false alarms. Before suppressing a false alarm, verify with the complementary model: only lower if fp_model's score is lower (confirms it is a false alarm)."""
    cur = fused.copy()
    if tp_model is not None and tp_model in S and miss.any():
        c = S[tp_model]
        m = miss & (c > cur)
        cur[m] = c[m]
    if fp_model is not None and fp_model in S and fa.any():
        c = S[fp_model]
        m = fa & (c < cur)   # verify: only lower if the complementary model confirms it is a false alarm (score lower)
        cur[m] = c[m]
    return cur


def apply(fused, S, miss, fa, tp_model, fp_model):
    cur = fused.copy()
    if tp_model is not None and tp_model in S and miss.any():
        comp = S[tp_model]
        m = miss & (comp > cur)
        cur[m] = comp[m]
    if fp_model is not None and fp_model in S and fa.any():
        comp = S[fp_model]
        m = fa & (comp < cur)
        cur[m] = comp[m]
    return cur


def resolve_for_file(d, r):
    """Map the frozen parameters r onto the detectors actually available for file d (Eva files may lack some detector scores).

    The fallback must be label-free: when best is missing, use the first available detector, never select with Eva labels (d['vus']).
    """
    best = r["best"] if r["best"] in d["S"] else next(iter(d["S"]))
    tp = r.get("tp_model") if r.get("tp_model") in d["S"] else None
    fp = r.get("fp_model") if r.get("fp_model") in d["S"] else None
    return {"best": best, "tp_model": tp, "fp_model": fp,
            "K": r.get("K", r.get("iters", 0)),
            "support_q": r.get("support_q", 0.9),
            "fused_q": r.get("fused_q", 0.3)}


def _status(loop, base, sota):
    """Per-file status (corresponds to the STATUS mapping in report): reached SOTA / improved / fall back to base."""
    if loop is None:
        return None
    if loop >= sota - 1e-4:
        return "converged"
    if loop > base + 1e-6:
        return "improved"
    return "base_only"


def _file_row(f, d, best, r, raised=0, lowered=0, loop_vus=None):
    """One per-file compact table row: best + frozen parameters (support_q/fused_q/K) + base/loop/Δ/status."""
    base = d["vus"][best]
    sota = max(d["vus"].values())
    delta = (loop_vus - base) if loop_vus is not None else None
    return {"file": f, "morph": d["morph"], "morph_c": d.get("morph_c", d["morph"]),
            "source": d["source"],
            "vus": {k: round(v, 4) for k, v in d["vus"].items()},
            "best": best,
            "support_q": r.get("support_q"), "fused_q": r.get("fused_q"),
            "K": r.get("K", r.get("iters", 0)),
            "base_vus": round(base, 4),
            "tp_model": r.get("tp_model"), "fp_model": r.get("fp_model"),
            "raised": raised, "lowered": lowered,
            "loop_vus": round(loop_vus, 4) if loop_vus is not None else None,
            "delta": round(delta, 4) if delta is not None else None,
            "status": _status(loop_vus, base, sota)}


def _zeroshot_loop(d, r):
    """Raw zero-shot loop (no gain gate): raise missed detections + suppress false alarms after verification, run the full K rounds."""
    r = resolve_for_file(d, r)
    fused = d["S"][r["best"]].copy()
    raised = lowered = 0
    for _ in range(r.get("K", 0)):
        miss, fa = positions_zeroshot_q(fused, d["support"], r.get("support_q", 0.9),
                                        r.get("fused_q", 0.3))
        before = fused.copy()
        cand = apply_q(fused, d["S"], miss, fa, r.get("tp_model"), r.get("fp_model"))
        if np.array_equal(cand, before):
            break
        fused = cand
        raised += int((fused > before).sum())
        lowered += int((fused < before).sum())
    return fused, raised, lowered


def apply_zeroshot_params(d, r, tau=None):
    """Run the zero-shot loop on file d with frozen parameters, then accept it with the label-free gain criterion.

    If the criterion is not met (gain < tau, or the magnitude/support-consistency lower bounds fail), fall back to best (original model scores).
    Returns (fused, raised, lowered, diag).
    """
    r = resolve_for_file(d, r)
    base = d["S"][r["best"]].copy()
    fused, raised, lowered = _zeroshot_loop(d, r)
    diag = label_free_gain(d, base, fused)
    diag["reason"] = "ok"
    if tau is None:
        tau = DEFAULT_GAIN_TAU
    diag["tau"] = tau
    if diag["change_frac"] < MIN_CHANGE_FRAC:
        diag["reason"] = "no_change"
    elif diag["sep"] < MIN_SEP:
        diag["reason"] = "support_inseparable"
    elif diag["gain"] < tau:
        diag["reason"] = "no_gain"
    if diag["reason"] != "ok":
        return base.copy(), 0, 0, diag   # criterion not met → fall back to best
    return fused, raised, lowered, diag


def calibrate_gain_tau(data, rows) -> tuple[float, dict]:
    """tuning (labeled) calibrates the label-free gain threshold gain_tau, frozen for eva.

    Run the raw zero-shot loop on each tuning file, compute label_free_gain's gain and the real VUS-PR Δ;
    use the midpoint of the median gains of "gain files / non-gain files" as the threshold (with no gain files, set a high threshold to fall back everywhere).
    """
    pos, neg = [], []
    for row in rows:
        f = row["file"]
        d = data[f]
        r = {"best": row["best"], "tp_model": row.get("tp_model"),
             "fp_model": row.get("fp_model"), "K": row.get("K", 0),
             "support_q": 0.9, "fused_q": 0.3}
        base = d["S"][row["best"]]
        fused, _, _ = _zeroshot_loop(d, r)
        g = label_free_gain(d, base, fused)["gain"]
        delta = fast_vus(fused, d["label"], d["sw"]) - fast_vus(base, d["label"], d["sw"])
        (pos if delta > 1e-6 else neg).append(g)
    if pos and neg:
        tau = float((np.median(pos) + np.median(neg)) / 2.0)
    elif pos:
        tau = float(np.median(pos))
    else:
        tau = 1e9   # zero-shot loop gives no file a real gain → raise the threshold, eva falls back to best everywhere
    return tau, {"n_pos": len(pos), "n_neg": len(neg),
                 "gain_pos_median": float(np.median(pos)) if pos else None,
                 "gain_neg_median": float(np.median(neg)) if neg else None}


# ---------------------------------------------------------------------------
# multiprocessing worker: per-file compute the 4 detectors' VUS-PR (vectorized fast_vus)
# ---------------------------------------------------------------------------
def _file_worker(args):
    name, max_len, score_dir, min_dets, strip_tr = args
    stem = Path(name).stem
    S_raw = {}
    for d in DETS:
        p = score_dir / d / f"{stem}.npy"
        if p.exists():
            S_raw[d] = np.load(p, allow_pickle=False).ravel()
    if len(S_raw) < min_dets:
        return name, None, "too_few_detectors"
    full = pd.read_csv(DATA_DIR / name).to_numpy(dtype=float)
    label = pd.read_csv(LABEL_DIR / name)["Label"].to_numpy(dtype=int)
    n = len(full)
    # downsample oversized files (original data unchanged, only thinned in memory; scores/labels/data aligned at the same step)
    idx = np.arange(n)
    if max_len and n > max_len:
        step = int(np.ceil(n / max_len))
        idx = np.arange(0, n, step)
    S = {d: rank01(s[idx]) for d, s in S_raw.items()}
    y = label[idx]
    x = full[idx]
    tr = int(np.searchsorted(idx, tr_of(name)))
    sw = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
    ev = extract_evidence(x, reference=x[:tr])
    summary = ev.summary()
    morph, morph_c = morph_signature(summary)
    support = ev.support
    # Eva: drop the part before the training prefix tr, evaluate only on the detection segment (from tr onward), avoiding the normal training segment being counted into metrics
    if strip_tr:
        S = {d: s[tr:] for d, s in S.items()}
        y = y[tr:]
        support = support[tr:]
    data = {"S": S, "label": y, "sw": sw,
            "morph": morph, "morph_c": morph_c,
            "source": source_of(name), "support": support,
            "morph_ranked": summary.get("ranked") or [],
            "vus": {d: fast_vus(s, y, sw) for d, s in S.items()}}
    return name, data, "ok"


def _bucket_search(files: list[str], data: dict, sq_grid, fq_grid):
    """Core: learn (best, tp, fp, K, support_q, fused_q) for a set of files, maximizing the in-group mean VUS-PR.

    Labels are used only for fast_vus scoring; raise/lower positions use label-free positions_zeroshot_q (consistent with Eva).
    The sq/fq quantile thresholds are learned per bucket instead of hard-coded 0.9/0.3. Returns (best, base_vus, sota_vus, best_score, combo).
    """
    avail = [d for d in DETS if all(d in data[f]["S"] for f in files)]
    if len(avail) < 2:
        return None
    mean_vus = {d: float(np.mean([data[f]["vus"][d] for f in files])) for d in avail}
    best = max(mean_vus, key=mean_vus.get)
    base_vus = mean_vus[best]
    sota_vus = float(np.mean([max(data[f]["vus"].values()) for f in files]))
    comps = [d for d in avail if d != best]

    best_combo = None
    best_score = base_vus
    for sq in sq_grid:
        for fq in fq_grid:
            for tp in comps:
                for fp in comps:
                    fused = {f: data[f]["S"][best].copy() for f in files}
                    score = base_vus
                    K = 0
                    for it in range(MAX_ITER):
                        new = {}
                        for f in files:
                            miss, fa = positions_zeroshot_q(fused[f], data[f]["support"], sq, fq)
                            new[f] = apply_q(fused[f], data[f]["S"], miss, fa, tp, fp)
                        v = float(np.mean([fast_vus(new[f], data[f]["label"], data[f]["sw"]) for f in files]))
                        if v > score + 1e-6:
                            score = v
                            fused = new
                            K = it + 1
                            if v >= 1.0 - 1e-9:
                                break
                        else:
                            break
                    if score > best_score + 1e-6:
                        best_score = score
                        best_combo = {"tp_model": tp, "fp_model": fp, "K": K,
                                      "support_q": sq, "fused_q": fq}
    return best, base_vus, sota_vus, best_score, best_combo


def _heldout_gain(train_files: list[str], test_files: list[str], data: dict) -> list[float]:
    """Learn a rule on train (fixed 0.9/0.3 to save time), measure the loop's held-out VUS-PR gain on test."""
    r = _bucket_search(train_files, data, (0.9,), (0.3,))
    if r is None:
        return []
    best, base_vus, sota_vus, best_score, combo = r
    if combo is None or combo["K"] == 0:
        return []
    rule = {"best": best, "tp_model": combo["tp_model"], "fp_model": combo["fp_model"],
            "K": combo["K"], "support_q": combo["support_q"], "fused_q": combo["fused_q"]}
    gains = []
    for f in test_files:
        d = data[f]
        base = d["S"][best]
        fused, _, _ = _zeroshot_loop(d, rule)   # label-free loop, consistent with Eva
        gains.append(fast_vus(fused, d["label"], d["sw"]) - fast_vus(base, d["label"], d["sw"]))
    return gains


def learn_bucket(files: list[str], data: dict) -> dict | None:
    """Per-morphology / per-bucket complement + strengthening (sq/fq learning) + label held-out validation (generalization).

    Core: jointly learn (best, tp, fp, K, support_q, fused_q) in the group, maximizing the group mean VUS-PR;
    validation: 2-fold held-out; if the loop gain does not generalize (held-out gain ≤0), fall back to base (K=0), preserving accuracy + generalization.
    """
    t0 = time.time()
    print(f"  [bucket-learning] n={len(files)} files, grid search (best,tp,fp,K,sq,fq)…", flush=True)
    r = _bucket_search(files, data, (0.85, 0.9, 0.95), (0.25, 0.3, 0.4))
    if r is None:
        print(f"  [bucket-learning] n={len(files)} skip (<2 detectors, {time.time()-t0:.0f}s)", flush=True)
        return None
    best, base_vus, sota_vus, best_score, combo = r

    if combo is not None and combo["K"] > 0 and len(files) >= 4:
        mid = len(files) // 2
        gains = _heldout_gain(files[:mid], files[mid:], data) \
            + _heldout_gain(files[mid:], files[:mid], data)
        if gains and np.mean(gains) <= 1e-6:
            combo = None
            best_score = base_vus

    if combo is None:
        print(f"  [bucket-learning] n={len(files)} best={best} no complement→fall back to base ({time.time()-t0:.0f}s)", flush=True)
        return {"best": best, "tp_model": None, "fp_model": None,
                "support_q": 0.9, "fused_q": 0.3, "K": 0,
                "base_vus": round(base_vus, 4), "loop_vus": round(base_vus, 4),
                "sota_vus": round(sota_vus, 4), "n_files": len(files), "status": "base_only"}

    print(f"  [bucket-learning] n={len(files)} best={best} tp={combo['tp_model']} "
          f"fp={combo['fp_model']} K={combo['K']} loop={best_score:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return {"best": best, "tp_model": combo["tp_model"], "fp_model": combo["fp_model"],
            "support_q": combo["support_q"], "fused_q": combo["fused_q"], "K": combo["K"],
            "base_vus": round(base_vus, 4), "loop_vus": round(best_score, 4),
            "sota_vus": round(sota_vus, 4), "n_files": len(files),
            "status": "converged" if best_score >= sota_vus - 1e-4 else "improved"}


def learn_file(d: dict) -> dict:
    """Tuning (labeled) learns parameters: jointly select (best, tp_model, fp_model, K).

    Key: the loop uses positions_zeroshot_q (label-free, fully consistent with Eva) to find raise/lower positions,
    labels only for fast_vus scoring/selecting/stopping. Thus the learned best/tp/fp/K generalize directly to Eva,
    eliminating the train/test mismatch of "tuning uses the label oracle to find positions, eva uses quantiles".
    """
    S = d["S"]
    support = d["support"]
    avail = [x for x in DETS if x in S]
    if len(avail) < 2:
        b = avail[0]
        return {"best": b, "tp_model": None, "fp_model": None, "K": 0,
                "support_q": 0.9, "fused_q": 0.3,
                "loop_vus": float(d["vus"][b]), "raised": 0, "lowered": 0}

    best_combo = None
    best_score = -1.0
    for base_det in avail:
        base = S[base_det].copy()
        base_vus = float(d["vus"][base_det])
        comps = [x for x in avail if x != base_det]
        for tp in comps:
            for fp in comps:
                fused = base.copy()
                score = base_vus
                K = 0
                raised = lowered = 0
                for it in range(MAX_ITER):
                    miss, fa = positions_zeroshot_q(fused, support, 0.9, 0.3)  # consistent with Eva
                    before = fused.copy()
                    cand = apply_q(fused, S, miss, fa, tp, fp)
                    if np.array_equal(cand, before):
                        break
                    v = fast_vus(cand, d["label"], d["sw"])   # labels only for scoring/selecting/stopping
                    if v > score + 1e-6:
                        score = v
                        fused = cand
                        K = it + 1
                        raised += int((fused > before).sum())
                        lowered += int((fused < before).sum())
                        if v >= 1.0 - 1e-9:
                            break
                    else:
                        break
                if score > best_score + 1e-6:
                    best_score = score
                    best_combo = {"best": base_det, "tp_model": tp, "fp_model": fp, "K": K,
                                  "raised": raised, "lowered": lowered}
    if best_combo is None:
        b = avail[0]
        return {"best": b, "tp_model": None, "fp_model": None, "K": 0,
                "support_q": 0.9, "fused_q": 0.3,
                "loop_vus": float(d["vus"][b]), "raised": 0, "lowered": 0}
    best_combo["support_q"] = 0.9
    best_combo["fused_q"] = 0.3
    best_combo["loop_vus"] = best_score
    return best_combo


def _agg_rule(rows: list) -> dict:
    """Aggregate per-file oracle results (best/tp/fp/K) into one bucket rule (used by Eva label-free).

    best = mode, tp/fp = most common complement pair, K = median.
    """
    if not rows:
        return None
    best = Counter(r["best"] for r in rows).most_common(1)[0][0]
    pairs = Counter((r.get("tp_model"), r.get("fp_model")) for r in rows if r.get("tp_model"))
    tp, fp = pairs.most_common(1)[0][0] if pairs else (None, None)
    ks = [r["K"] for r in rows if r.get("K")]
    K = int(np.median(ks)) if ks else 1
    return {"best": best, "tp_model": tp, "fp_model": fp, "K": K,
            "support_q": 0.9, "fused_q": 0.3, "n_files": len(rows),
            "base_vus": round(float(np.mean([r["base_vus"] for r in rows])), 4),
            "loop_vus": round(float(np.mean([r["loop_vus"] for r in rows])), 4)}


def _rule_key(morph: str, morph_c: str, source: str, params: dict) -> str:
    """Return the bucket key the Eva lookup hits (consistent with the lookup order of run_eva/run_eva_watch),
    used to map the frozen LLM verification conclusions from tuning back to each Eva file."""
    if source in params.get("rule_src", {}):
        return f"src:{source}"
    if f"{morph}/{source}" in params.get("rule_ms", {}):
        return f"ms:{morph}/{source}"
    if morph in params.get("rule_m", {}):
        return f"m:{morph}"
    if f"{morph_c}/{source}" in params.get("rule_cs", {}):
        return f"cs:{morph_c}/{source}"
    if morph_c in params.get("rule_c", {}):
        return f"c:{morph_c}"
    return "g:global"


def run_eva(params: dict, max_len: int, workers: int, keep_tr: bool = False) -> None:
    """Eva phase: label-free (no labels for any decision).

    Frozen rules (best/tp/fp/K from Tuning) + zero-shot quantile loop; labels are used only once, at the very end, to report VUS-PR.
    """
    files = pd.read_csv(EVA_LIST)["file_name"].tolist()
    rule_ms = params.get("rule_ms", {})
    rule_m = params.get("rule_m", {})
    rule_cs = params.get("rule_cs", {})
    rule_c = params.get("rule_c", params.get("rule_mc", {}))   # compatible with the old rule_mc naming
    rule_src = params.get("rule_src", {})
    rule_g = params.get("rule_g") or {"best": DETS[0], "tp_model": None,
                                      "fp_model": None, "K": 1, "support_q": 0.9,
                                      "fused_q": 0.3}
    gain_tau = params.get("gain_tau", DEFAULT_GAIN_TAU)

    def lookup(morph, morph_c, source):
        if source in rule_src:            # source prior is strongest (70.8% vs 62.5%), preferred
            return rule_src[source]
        if f"{morph}/{source}" in rule_ms:
            return rule_ms[f"{morph}/{source}"]
        if morph in rule_m:
            return rule_m[morph]
        if f"{morph_c}/{source}" in rule_cs:
            return rule_cs[f"{morph_c}/{source}"]
        if morph_c in rule_c:
            return rule_c[morph_c]
        return rule_g

    state = {"phase": "eva", "split": "eva", "max_len": max_len,
             "total": len(files), "done": 0, "usable": 0, "trained": 0,
             "started": time.time(), "current": None, "detectors": DETS,
             "files": [], "aggregate": None, "result": {}}
    _write_route(state)

    rows = []
    t0 = time.time()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(workers) as pool:
        for name, d, status in pool.imap_unordered(
                _file_worker, [(f, max_len, EVA_SCORE_DIR, len(DETS), not keep_tr) for f in files], chunksize=1):
            state["done"] += 1
            if d is not None:
                # anti label leakage: decisions use only label-free fields; label/vus only for the final fast_vus and report
                free = {"S": d["S"], "support": d["support"], "sw": d["sw"],
                        "morph": d["morph"], "morph_c": d["morph_c"], "source": d["source"]}
                r = resolve_for_file(free, lookup(free["morph"], free["morph_c"], free["source"]))
                fused, raised, lowered, diag = apply_zeroshot_params(free, r, gain_tau)   # label-free
                l = fast_vus(fused, d["label"], d["sw"])   # labels only for the final evaluation
                row = _file_row(name, d, r["best"], r, raised, lowered, l)
                row["fallback"] = None if diag["reason"] == "ok" else diag["reason"]
                row["gain"] = round(diag.get("gain", 0.0), 5)
                row["llm_verify"] = params.get("llm_verify", {}).get(
                    _rule_key(free["morph"], free["morph_c"], free["source"], params))
                rows.append(row)
                state["files"] = rows
                state["usable"] = len(rows)
                state["trained"] = len(rows)
                state["current"] = {"file": name, "morph": d["morph"], "source": d["source"]}
                state["aggregate"] = {
                    "base": round(float(np.mean([x["base_vus"] for x in rows])), 4),
                    "loop": round(float(np.mean([x["loop_vus"] for x in rows])), 4),
                    "sota": round(float(np.mean([max(x["vus"].values()) for x in rows])), 4),
                    "delta": round(float(np.mean([x["delta"] for x in rows])), 4),
                }
                _write_route(state)
                print(f"  [eva {len(rows)}] {name} best={r['best']} "
                      f"tp={r.get('tp_model')} fp={r.get('fp_model')} "
                      f"base={row['base_vus']} loop={row['loop_vus']} Δ={row['delta']}", flush=True)

    if not rows:
        state["phase"] = "done"
        state["aggregate"] = {"base": None, "loop": None, "sota": None, "delta": None}
        _write_route(state)
        print(f"\n==== EVA label-free (VUS-PR) ====")
        print(f"  no usable Eva files")
        return

    state["phase"] = "done"
    state["result"] = {"base": state["aggregate"]["base"], "loop": state["aggregate"]["loop"],
                       "sota": state["aggregate"]["sota"], "delta": state["aggregate"]["delta"]}
    _write_route(state)

    print(f"\n==== EVA label-free (VUS-PR mean, {len(rows)} files, {time.time()-t0:.0f}s) ====")
    print(f"  base {state['aggregate']['base']:.4f} -> loop {state['aggregate']['loop']:.4f} "
          f"(Δ {state['aggregate']['delta']:+.4f})")
    print(f"  SOTA {state['aggregate']['sota']:.4f}")
    llm_verdicts = [x.get("llm_verify") for x in rows if x.get("llm_verify")]
    if llm_verdicts:
        c = Counter(v.get("verdict") for v in llm_verdicts)
        print(f"  LLM verification ({len(llm_verdicts)}/{len(rows)} files hit the frozen bucket conclusions): "
              f"confirm={c.get('confirm', 0)} conflict={c.get('conflict', 0)} "
              f"uncertain={c.get('uncertain', 0)}", flush=True)
    print("[saved] route_progress.json + route_data.js", flush=True)


def run_eva_watch(params: dict, max_len: int, workers: int, interval: int = 30,
                  keep_tr: bool = False) -> None:
    """Continuously watch Eva scores: label-free (frozen rules + zero-shot loop), update the dashboard as scores are generated, until Ctrl+C."""
    files = pd.read_csv(EVA_LIST)["file_name"].tolist()
    rule_ms = params.get("rule_ms", {})
    rule_m = params.get("rule_m", {})
    rule_cs = params.get("rule_cs", {})
    rule_c = params.get("rule_c", params.get("rule_mc", {}))   # compatible with the old rule_mc naming
    rule_src = params.get("rule_src", {})
    rule_g = params.get("rule_g") or {"best": DETS[0], "tp_model": None,
                                      "fp_model": None, "K": 1, "support_q": 0.9,
                                      "fused_q": 0.3}
    gain_tau = params.get("gain_tau", DEFAULT_GAIN_TAU)

    def lookup(morph, morph_c, source):
        if source in rule_src:            # source prior is strongest, preferred
            return rule_src[source]
        if f"{morph}/{source}" in rule_ms:
            return rule_ms[f"{morph}/{source}"]
        if morph in rule_m:
            return rule_m[morph]
        if f"{morph_c}/{source}" in rule_cs:
            return rule_cs[f"{morph_c}/{source}"]
        if morph_c in rule_c:
            return rule_c[morph_c]
        return rule_g

    def has_all(f):
        stem = Path(f).stem
        return all((EVA_SCORE_DIR / d / f"{stem}.npy").exists() for d in DETS)

    evaluated = {}
    state = {"phase": "eva_watch", "split": "eva", "max_len": max_len,
             "total": len(files), "done": 0, "usable": 0,
             "started": time.time(), "current": None, "detectors": DETS,
             "files": [], "aggregate": None, "result": {}}

    while True:
        # only process newly-appeared files with all three detectors: load one compute one, and update summary + write the dashboard immediately after each
        new_files = [f for f in files if f not in evaluated and has_all(f)]
        for f in new_files:
            _, d, _ = _file_worker((f, max_len, EVA_SCORE_DIR, len(DETS), not keep_tr))
            if d is None:
                continue
            # anti label leakage: decisions use only label-free fields
            free = {"S": d["S"], "support": d["support"], "sw": d["sw"],
                    "morph": d["morph"], "morph_c": d["morph_c"], "source": d["source"]}
            r = resolve_for_file(free, lookup(free["morph"], free["morph_c"], free["source"]))
            fused, raised, lowered, diag = apply_zeroshot_params(free, r, gain_tau)   # label-free
            l = fast_vus(fused, d["label"], d["sw"])   # labels only for the final evaluation
            evaluated[f] = _file_row(f, d, r["best"], r, raised, lowered, l)
            evaluated[f]["fallback"] = None if diag["reason"] == "ok" else diag["reason"]
            evaluated[f]["llm_verify"] = params.get("llm_verify", {}).get(
                _rule_key(free["morph"], free["morph_c"], free["source"], params))
            # immediately recompute the summary (incremental, live)
            rows = list(evaluated.values())
            state["files"] = rows
            state["usable"] = len(evaluated)
            state["done"] = len(evaluated)
            state["aggregate"] = {
                "base": round(float(np.mean([x["base_vus"] for x in rows])), 4),
                "loop": round(float(np.mean([x["loop_vus"] for x in rows])), 4),
                "sota": round(float(np.mean([max(x["vus"].values()) for x in rows])), 4),
                "delta": round(float(np.mean([x["delta"] for x in rows])), 4),
            }
            _write_route(state)
            print(f"  [eva {len(evaluated)}] {f} best={r['best']} "
                  f"tp={r.get('tp_model')} fp={r.get('fp_model')} "
                  f"base={evaluated[f]['base_vus']} loop={evaluated[f]['loop_vus']}", flush=True)

        state["updated"] = time.time()
        state["elapsed"] = round(state["updated"] - state["started"], 1)
        _write_route(state)
        print(f"  [watch] evaluated {len(evaluated)} Eva files, "
              f"{state['aggregate']['loop'] if evaluated else '—'} (rescan in {interval}s)", flush=True)
        time.sleep(interval)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=25000,
                    help="skip files exceeding this sample-point count")
    ap.add_argument("--workers", type=int, default=2,
                    help="multiprocessing count (memory is tight when running concurrently with gen_local_scores, use 2)")
    ap.add_argument("--split", choices=["tuning", "eva", "both"], default="tuning",
                    help="tuning=learn parameters / eva=evaluate zero-shot with frozen parameters / both=tuning then eva")
    ap.add_argument("--watch", action="store_true",
                    help="Eva continuous watch: compute VUS-PR and write the dashboard while scores are generated (Ctrl+C stops)")
    ap.add_argument("--interval", type=int, default=30, help="scan interval of watch mode (seconds)")
    ap.add_argument("--llm-verify", dest="llm_verify", action="store_true", default=True,
                    help="use the local LLM to verify the best selection (highest score + morphology/source-prior consistency; on by default, "
                         "falls back to rule-based verdict when Ollama is unavailable)")
    ap.add_argument("--no-llm-verify", dest="llm_verify", action="store_false",
                    help="disable LLM verification")
    ap.add_argument("--llm-model", default=None,
                    help="LLM verification model name (defaults to config.OLLAMA_MODEL)")
    ap.add_argument("--llm-timeout", type=float, default=60.0,
                    help="LLM verification per-request timeout (seconds)")
    ap.add_argument("--llm-correct", action="store_true", default=False,
                    help="when the LLM judges conflict, override best with expected_model (prior correction, off by default)")
    ap.add_argument("--keep-tr", action="store_true", default=False,
                    help="Eva keeps the training-prefix tr segment and evaluates it too (with tr segment); by default the tr segment is dropped, "
                         "evaluating only the detection segment after tr (without tr segment)")
    args = ap.parse_args()

    if args.split == "eva":
        params = json.loads(PARAMS.read_text(encoding="utf-8"))
        if args.watch:
            run_eva_watch(params, args.max_len, args.workers, args.interval, args.keep_tr)
        else:
            run_eva(params, args.max_len, args.workers, args.keep_tr)
        return

    files = pd.read_csv(TUNING_LIST)["file_name"].tolist()
    state = {"phase": "loading", "split": "tuning", "max_len": args.max_len,
             "total": len(files), "done": 0, "usable": 0, "trained": 0,
             "started": time.time(), "current": None, "detectors": DETS,
             "files": [], "aggregate": None, "result": {}}
    _write_route(state)
    data = {}
    t0 = time.time()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(args.workers) as pool:
        for name, d, status in pool.imap_unordered(
                _file_worker, [(f, args.max_len, SCORE_DIR, 2, False) for f in files], chunksize=1):
            state["done"] += 1
            if d is not None:
                data[name] = d
                state["usable"] = len(data)
                state["current"] = {"file": name, "morph": d["morph"], "source": d["source"]}
                state["files"].append({"file": name, "morph": d["morph"], "morph_c": d["morph_c"],
                                       "source": d["source"],
                                       "vus": {k: round(v, 4) for k, v in d["vus"].items()},
                                       "best": None, "loop_vus": None})
                _write_route(state)
                print(f"  [{time.time()-t0:5.0f}s] {name} morph={d['morph']} "
                      f"source={d['source']} "
                      f"VUS-PR={ {k: round(v, 3) for k, v in d['vus'].items()} }", flush=True)
    print(f"\nusable {len(data)} files (≥2 detector scores, len≤{args.max_len}, {time.time()-t0:.0f}s)\n", flush=True)

    # Tuning: per-file best=argmax + label complement loop (use labels to find real miss/fa)
    print("\n==== Tuning per-file best(argmax) + label complement loop ====")
    state["phase"] = "perfile"
    t_learn = time.time()
    rows = []
    for i, (f, d) in enumerate(data.items(), 1):
        r = learn_file(d)
        row = _file_row(f, d, r["best"], r, r["raised"], r["lowered"], r["loop_vus"])
        rows.append(row)
        state["files"] = rows
        state["trained"] = i
        state["current"] = {"file": f, "morph": d["morph"], "source": d["source"]}
        _write_route(state)
        print(f"  [per-file {i}/{len(data)}] {f} best={r['best']} tp={r.get('tp_model')} "
              f"fp={r.get('fp_model')} K={r.get('K', 0)} base={row['base_vus']} "
              f"loop={row['loop_vus']} Δ={row['delta']}", flush=True)

    state["aggregate"] = {
        "base": round(float(np.mean([r["base_vus"] for r in rows])), 4),
        "loop": round(float(np.mean([r["loop_vus"] for r in rows])), 4),
        "sota": round(float(np.mean([max(r["vus"].values()) for r in rows])), 4),
        "delta": round(float(np.mean([r["delta"] for r in rows])), 4),
    }
    state["phase"] = "done"
    state["result"] = {"base": state["aggregate"]["base"], "loop": state["aggregate"]["loop"],
                       "sota": state["aggregate"]["sota"], "delta": state["aggregate"]["delta"]}
    _write_route(state)

    print(f"\n==== Tuning per-file best + label loop (VUS-PR mean, {len(rows)} files, {time.time()-t_learn:.0f}s) ====")
    print(f"  base {state['aggregate']['base']:.4f} → loop {state['aggregate']['loop']:.4f} "
          f"(Δ {state['aggregate']['delta']:+.4f})")
    print(f"  SOTA {state['aggregate']['sota']:.4f}")
    print("[saved] route_progress.json + route_data.js", flush=True)

    # learn per-bucket complement parameters (for Eva label-free): (fine,src)/fine/(coarse,src)/coarse/src/global
    buckets_ms = defaultdict(list)   # (fine morph, source) -> [file names]
    buckets_m = defaultdict(list)    # fine morph
    buckets_cs = defaultdict(list)   # (coarse morph, source)
    buckets_c = defaultdict(list)    # coarse morph
    buckets_src = defaultdict(list)  # source
    for f, d in data.items():
        buckets_ms[(d["morph"], d["source"])].append(f)
        buckets_m[d["morph"]].append(f)
        buckets_cs[(d["morph_c"], d["source"])].append(f)
        buckets_c[d["morph_c"]].append(f)
        buckets_src[d["source"]].append(f)

    rule_ms = {f"{m}/{s}": r for (m, s), fs in buckets_ms.items()
               if len(fs) >= 3 and (r := learn_bucket(fs, data))}
    rule_m = {m: r for m, fs in buckets_m.items()
              if len(fs) >= 3 and (r := learn_bucket(fs, data))}
    rule_cs = {f"{m}/{s}": r for (m, s), fs in buckets_cs.items()
               if len(fs) >= 3 and (r := learn_bucket(fs, data))}
    rule_c = {m: r for m, fs in buckets_c.items()
              if len(fs) >= 3 and (r := learn_bucket(fs, data))}
    rule_src = {s: r for s, fs in buckets_src.items()
                if len(fs) >= 2 and (r := learn_bucket(fs, data))}
    rule_g = learn_bucket(list(data.keys()), data)

    gain_tau, gain_cal = calibrate_gain_tau(data, rows)

    # LLM verifies the best selection (highest score + morphology/source-prior consistency); audit only, does not change the routing.
    llm_verify = {}
    if args.llm_verify and verify_rules is not None:
        buckets_ms_str = {f"{m}/{s}": fs for (m, s), fs in buckets_ms.items()}
        buckets_cs_str = {f"{m}/{s}": fs for (m, s), fs in buckets_cs.items()}
        g_rules = {"global": rule_g} if rule_g else {}
        specs = [("ms", rule_ms, buckets_ms_str), ("m", rule_m, buckets_m),
                 ("cs", rule_cs, buckets_cs_str), ("c", rule_c, buckets_c),
                 ("src", rule_src, buckets_src),
                 ("g", g_rules, {"global": list(data.keys())})]
        llm_verify = verify_rules(specs, data, model=args.llm_model, timeout=args.llm_timeout)
        if args.llm_correct and apply_corrections is not None:
            n_corrected = apply_corrections(specs, llm_verify)
            print(f"[corrected] {n_corrected} buckets' best overridden by the prior (best→expected_model, cleared loop)", flush=True)
        (ROOT / "runs").mkdir(parents=True, exist_ok=True)
        (ROOT / "runs" / "route_llm_verify.json").write_text(
            json.dumps(llm_verify, ensure_ascii=False, indent=1), encoding="utf-8")
        c = Counter(v.get("verdict") for v in llm_verify.values())
        print(f"[saved] runs/route_llm_verify.json (LLM verified {len(llm_verify)} buckets: "
              f"confirm={c.get('confirm', 0)} conflict={c.get('conflict', 0)} "
              f"uncertain={c.get('uncertain', 0)})", flush=True)
        for k in [k for k, v in llm_verify.items() if v.get("verdict") == "conflict"]:
            print(f"  [conflict] {k}: {llm_verify[k].get('reasoning', '')[:100]}", flush=True)

    params_save = {"rule_ms": rule_ms, "rule_m": rule_m, "rule_cs": rule_cs,
                   "rule_c": rule_c, "rule_src": rule_src, "rule_g": rule_g,
                   "gain_tau": gain_tau, "gain_cal": gain_cal,
                   "llm_verify": llm_verify}
    PARAMS.write_text(json.dumps(params_save, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] runs/route_params.json (ms {len(rule_ms)} / m {len(rule_m)} / "
          f"cs {len(rule_cs)} / c {len(rule_c)} / src {len(rule_src)} / "
          f"global best={rule_g['best'] if rule_g else DETS[0]} / gain_tau={gain_tau:.4g})", flush=True)


if __name__ == "__main__":
    main()
