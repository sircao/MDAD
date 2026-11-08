"""eval_fullshot_standard.py —— Standard fullshot: Tuning train / Eva test (strict train/test separation + official scoring).

Protocol discipline:
  - VUS-PR is computed only on the "whole sequence": get_metrics(score_full, label_full, slidingWindow=find_length_rank(data[:,0], rank=1)).
    It is not computed on the tr_N prefix alone (no positives there, VUS-PR is meaningless); do not slice [tr:].
  - Tuning (48, labeled) → learn "morphology→optimal base" + "fusion weight", reusable (for tuning).
  - Eva (350, labeled) → compute VUS-PR once only after the rules are frozen, for reporting; never used for model selection/early stopping.
  - loop fusion uses label-free evidence support (support), never touching labels (zero-shot).
Monitoring: per-morphology VUS-PR/optimal weight during training; per-file VUS-PR + label-free evidence-quality proxy during testing.
"""
from __future__ import annotations

import contextlib
import io
import json
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

TUNING_DIR = ROOT / "eval" / "score" / "uni" / "tuning"
EVA_DIR = ROOT / "eval" / "score" / "uni" / "eva"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
TUNING_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
OUT_JSON = ROOT / "dashboard_data.json"

DETS = ["Syntropic", "CHARM_kNN", "Time_RCD", "CHARM_ZS"]
KEYS = ["CHARM_fixed", "Syn_fixed", "base", "fused", "loop"]


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


def run_loop_zero(S: np.ndarray, base_idx: int, support: np.ndarray,
                  label=None, sw=None, sota_vus=None, max_iter: int = 15, write_cb=None):
    """loop: start from the base (SOTA); complement = per-point max (rank-max) of all non-base detectors, raise-only.
    Tuning passes label/sw/sota_vus: stop once each round's VUS-PR exceeds SOTA (learn K). Eva passes nothing, runs with the frozen K."""
    fused = S[base_idx].copy()
    comp = np.max([S[i] for i in range(S.shape[0]) if i != base_idx], axis=0)  # non-base rank-max
    trace = []
    for it in range(max_iter):
        if label is not None and sw is not None and sota_vus is not None:
            if _vus(fused, label, sw) >= sota_vus + 1e-4:
                break  # exceeds SOTA, stop (only for Tuning decisions)
        uncertain = (fused >= 0.33) & (fused <= 0.67)
        target = uncertain & (support >= 0.5)
        if target.sum() < 4:
            break
        raise_mask = target & (comp > fused)
        if raise_mask.sum() == 0:
            break
        gain = float((comp[raise_mask] - fused[raise_mask]).sum())
        fused[raise_mask] = comp[raise_mask]
        t = {"iter": it + 1, "raised": int(raise_mask.sum()), "gain": round(gain, 4)}
        if label is not None and sw is not None:
            t["vus"] = round(_vus(fused, label, sw), 4)
        trace.append(t)
        if write_cb is not None:
            write_cb(trace)
    return fused, trace


def main() -> None:
    tun_files = pd.read_csv(TUNING_LIST)["file_name"].tolist()
    eva_files = pd.read_csv(EVA_LIST)["file_name"].tolist()

    # —— training (Tuning, learn rules with labels) ——
    tun_morph = {}
    tun_scores = {}
    for i, f in enumerate(tun_files, 1):
        data = pd.read_csv(DATA_DIR / f).to_numpy(dtype=float)
        tr = tr_of(f)
        labels = pd.read_csv(LABEL_DIR / f)["Label"].to_numpy(dtype=float)
        sw = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
        morph, _ = ev_morph_support(data, tr)
        tun_morph[f] = morph
        sc = {d: rank01(load_scores(f, d, TUNING_DIR)) for d in DETS}
        v = {d: _vus(sc[d], labels, sw) for d in DETS}
        tun_scores[f] = {"sc": sc, "vus": v, "label": labels, "sw": sw, "morph": morph}
        # write training intermediate variables (per file)
        training_rows = [{"file": tf, "morph": tun_scores[tf]["morph"],
                          "Syn": round(tun_scores[tf]["vus"]["Syntropic"], 4),
                          "CHARM": round(tun_scores[tf]["vus"]["CHARM_kNN"], 4)}
                         for tf in tun_files if tf in tun_scores]
        (ROOT / "training_progress.json").write_text(json.dumps({
            "phase": "training", "done": len(tun_scores), "total": len(tun_files),
            "updated": time.time(), "rows": training_rows, "rule": {},
        }, ensure_ascii=False), encoding="utf-8")
        print(f"[train {i}/{len(tun_files)}] {f} morph={morph} "
              f"Syn={v['Syntropic']:.3f} CHARM={v['CHARM_kNN']:.3f}", flush=True)
    print(f"Tuning training samples {len(tun_files)}", flush=True)

    from collections import defaultdict
    by_morph = defaultdict(list)
    for f in tun_files:
        by_morph[tun_morph[f]].append(f)

    rule = {}
    print("\n==== rules learned on Tuning ====")
    for morph, fs in sorted(by_morph.items()):
        mean_v = {d: float(np.mean([tun_scores[f]["vus"][d] for f in fs])) for d in DETS}
        best = max(DETS, key=lambda d: mean_v[d])
        other = max([d for d in DETS if d != best], key=lambda d: mean_v[d])  # second-best as complement
        best_w, best_a = 0.5, -1.0
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            a = float(np.mean([_vus(w * tun_scores[f]["sc"][best] + (1 - w) * tun_scores[f]["sc"][other],
                                   tun_scores[f]["label"], tun_scores[f]["sw"]) for f in fs]))
            if a > best_a:
                best_a, best_w = a, w
        rule[morph] = {"best": best, "other": other, "w": best_w}
        print(f"  {morph:16s} best={best:16s} w={best_w:.2f} "
              f"(Syn={mean_v['Syntropic']:.3f} CHARM={mean_v['CHARM_kNN']:.3f})", flush=True)

    # write full training rules + Tuning-phase loop intermediate variables
    training_rows = [{"file": tf, "morph": tun_scores[tf]["morph"],
                      "Syn": round(tun_scores[tf]["vus"]["Syntropic"], 4),
                      "CHARM": round(tun_scores[tf]["vus"]["CHARM_kNN"], 4)}
                     for tf in tun_files]

    loop_rows = []
    for f in tun_files:
        data = pd.read_csv(DATA_DIR / f).to_numpy(dtype=float)
        tr = tr_of(f)
        labels = pd.read_csv(LABEL_DIR / f)["Label"].to_numpy(dtype=float)
        sw = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
        morph, support = ev_morph_support(data, tr)
        sc = {d: rank01(load_scores(f, d, TUNING_DIR)) for d in DETS}
        r = rule.get(morph, {"best": "Syntropic", "other": "CHARM_kNN", "w": 0.5})
        base = sc[r["best"]]
        other = sc[r["other"]]
        fused_w = r["w"] * base + (1 - r["w"]) * other
        def _wloop(trace):
            (ROOT / "loop_progress.json").write_text(json.dumps({
                "phase": "tuning", "file": f, "trace": trace, "updated": time.time(),
            }, ensure_ascii=False), encoding="utf-8")
        S = np.stack([sc[d] for d in DETS])
        base_idx = DETS.index(r["best"])
        sota_vus = max(tun_scores[f]["vus"].values())  # SOTA = best single-detector VUS-PR
        loop, trace = run_loop_zero(S, base_idx, support, labels, sw,
                                    sota_vus=sota_vus, write_cb=_wloop)
        loop_rows.append({"file": f, "morph": morph, "best": r["best"], "w": r["w"],
                          "base_vus": round(tun_scores[f]["vus"][r["best"]], 4),
                          "fused_vus": round(_vus(fused_w, labels, sw), 4),
                          "loop_vus": round(_vus(loop, labels, sw), 4),
                          "iters": len(trace), "trace": trace})

    # learn per-morphology "rounds K needed to beat SOTA" (median), frozen for Eva use
    from collections import defaultdict
    k_by_morph = defaultdict(list)
    for lr in loop_rows:
        k_by_morph[lr["morph"]].append(lr["iters"])
    for m in rule:
        ks = k_by_morph.get(m, [0])
        rule[m]["K"] = int(round(float(np.median(ks))))
    print("\n==== per-morphology frozen loop round count K ====")
    for m in sorted(rule):
        print(f"  {m:16s} K={rule[m]['K']}", flush=True)

    (ROOT / "training_progress.json").write_text(json.dumps({
        "phase": "trained", "done": len(tun_files), "total": len(tun_files),
        "updated": time.time(), "rows": training_rows, "rule": rule, "loop_rows": loop_rows,
    }, ensure_ascii=False), encoding="utf-8")

    print("\n==== Tuning-phase loop (using learned rules) ====")
    for lr in loop_rows:
        print(f"  {lr['file'][:40]} morph={lr['morph']} best={lr['best']} w={lr['w']} "
              f"base={lr['base_vus']:.4f} fused={lr['fused_vus']:.4f} loop={lr['loop_vus']:.4f} "
              f"iters={lr['iters']}", flush=True)
        for t in lr["trace"]:
            print(f"    round {t['iter']} raised {t['raised']} points gain={t['gain']:.4f} VUS-PR={t['vus']:.4f}", flush=True)

    # —— testing (Eva, labels only for VUS-PR) ——
    rows = {}
    t0 = time.time()

    def _write_dash():
        ordered = [rows[f] for f in eva_files if f in rows]
        agg = {k: round(float(np.mean([r[k] for r in ordered])), 4)
               for k in KEYS} if ordered else {}
        OUT_JSON.write_text(json.dumps({
            "status": "running" if len(rows) < len(eva_files) else "done",
            "total": len(eva_files), "done": len(rows),
            "started": t0, "updated": time.time(),
            "aggregate": agg, "rows": ordered, "rule": rule,
        }, ensure_ascii=False), encoding="utf-8")

    for f in eva_files:
        data = pd.read_csv(DATA_DIR / f).to_numpy(dtype=float)
        tr = tr_of(f)
        labels = pd.read_csv(LABEL_DIR / f)["Label"].to_numpy(dtype=float)
        sw = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
        morph, support = ev_morph_support(data, tr)
        sc = {d: rank01(load_scores(f, d, EVA_DIR)) for d in DETS}
        r = rule.get(morph, {"best": "Syntropic", "other": "CHARM_kNN", "w": 0.5})
        base = sc[r["best"]]
        other = sc[r["other"]]
        fused = 0.5 * base + 0.5 * other
        fused_w = r["w"] * base + (1 - r["w"]) * other
        def _wloop_eva(trace):
            (ROOT / "loop_progress.json").write_text(json.dumps({
                "phase": "eva", "file": f, "trace": trace, "updated": time.time(),
            }, ensure_ascii=False), encoding="utf-8")
        S = np.stack([sc[d] for d in DETS])
        base_idx = DETS.index(r["best"])
        K = r.get("K", 15)  # frozen loop round count (learned on Tuning)
        loop, loop_trace = run_loop_zero(S, base_idx, support, max_iter=K,
                                         write_cb=_wloop_eva)  # Eva: label-free, run the frozen K rounds

        rows[f] = {
            "file": f, "source": f.split("_")[1], "morph": morph, "best": r["best"],
            "CHARM_fixed": _vus(sc["CHARM_kNN"], labels, sw),
            "Syn_fixed": _vus(sc["Syntropic"], labels, sw),
            "base": _vus(base, labels, sw),
            "fused": _vus(fused, labels, sw),
            "loop": _vus(loop, labels, sw),
            "loop_iters": len(loop_trace),
            "loop_trace": loop_trace,
            "ev_quality": round(float(np.mean(support)), 4),  # label-free evidence-quality proxy
        }
        _write_dash()
        print(f"[{len(rows)}/{len(eva_files)}] {f} CHARM={rows[f]['CHARM_fixed']:.4f} "
              f"Syn={rows[f]['Syn_fixed']:.4f} routed={rows[f]['base']:.4f} "
              f"fused={rows[f]['fused']:.4f} loop={rows[f]['loop']:.4f} "
              f"iters={rows[f]['loop_iters']} ev_q={rows[f]['ev_quality']:.3f}", flush=True)
        for t in loop_trace:
            print(f"    loop round {t['iter']}: raised { t['raised']} points gain={t['gain']:.4f} VUS-PR={t['vus']:.4f}", flush=True)

    ordered = [rows[f] for f in eva_files]
    agg = {k: round(float(np.mean([r[k] for r in ordered])), 4) for k in KEYS}
    OUT_JSON.write_text(json.dumps({
        "status": "done", "total": len(eva_files), "done": len(eva_files),
        "started": t0, "updated": time.time(),
        "aggregate": agg, "rows": ordered,
    }, ensure_ascii=False), encoding="utf-8")

    print("\n==== FULLSHOT (Tuning train / Eva test) VUS-PR ====")
    for k in KEYS:
        print(f"  {k:12s} {agg[k]}")
    (ROOT / "runs" / "fullshot_standard.json").write_text(
        json.dumps({"rule": rule, "aggregate": agg}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("\n[saved] runs/fullshot_standard.json")


if __name__ == "__main__":
    main()
