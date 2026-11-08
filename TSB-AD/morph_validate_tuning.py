"""morph_validate_tuning.py — Validate morphology categories and features using the TSB-AD tuning labels, and output strengthening suggestions.

For the 48 tuning files:
  1) extract_evidence → per-point score + support + ranked top kind for each KINDS;
  2) AUC of each kind's score vs label (fast discrimination proxy; VUS-PR with its window loop is too slow);
  3) VUS-PR of the 3 detector scores vs label (the true best detector) + the support's VUS-PR;
  4) global morphology features from extract_features.

Aggregated output:
  - each kind: fire count / top count / "most discriminative kind" count / mean AUC
  - routing-signal consistency: whether top kind == most discriminative kind; which better predicts the true best detector
  - features: rank correlation against support_vus / best_det_vus / top_ok
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402
from Anomaly_algorithm.morphology_evidence import extract_evidence, KINDS  # noqa: E402
from Anomaly_algorithm.morphology import extract_features  # noqa: E402
from fast_vus import fast_vus  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
TUNING_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv"
SCORE_DIR = ROOT / "eval" / "score" / "uni" / "tuning"
DETS = ["Syntropic", "Time_RCD", "CHARM_stat_kNN"]
MAX_LEN = 25000   # consistent with eval_loop_route: downsample overly long files; the validation target is exactly what the routing actually sees


def tr_of(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def source_of(f: str) -> str:
    return f.split("_")[1]


def rank01(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float).ravel()
    return s / max(len(s), 1)


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """Fast rank-based AUC (Mann-Whitney), as a proxy for kind discriminative power."""
    from scipy.stats import rankdata
    score = np.asarray(score, dtype=float).ravel()
    label = np.asarray(label, dtype=int)
    pos = label == 1
    p = int(pos.sum())
    n = int((~pos).sum())
    if p == 0 or n == 0:
        return 0.5
    ranks = rankdata(score)
    return (float(ranks[pos].sum()) - p * (p + 1) / 2.0) / (p * n)


def _corr(a, b):
    from scipy.stats import rankdata
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(rankdata(a), rankdata(b))[0, 1]
    return float(r) if np.isfinite(r) else float("nan")


def main() -> None:
    files = pd.read_csv(TUNING_LIST)["file_name"].tolist()
    rows = []
    per_kind = {k: {"fire": 0, "top": 0, "best": 0, "auc": []} for k in KINDS}
    top_best_det = defaultdict(list)    # top kind -> true best detector
    kind_best_det = defaultdict(list)   # most discriminative kind -> true best detector

    t0 = time.time()
    for i, name in enumerate(files, 1):
        t_file = time.time()
        try:
            full = pd.read_csv(DATA_DIR / name).to_numpy(dtype=float)
            label = pd.read_csv(LABEL_DIR / name)["Label"].to_numpy(dtype=int)
            n = len(full)
            idx = np.arange(n)
            if n > MAX_LEN:
                step = int(np.ceil(n / MAX_LEN))
                idx = np.arange(0, n, step)
            x = full[idx]
            y = label[idx]
            tr = int(np.searchsorted(idx, tr_of(name)))
            sw = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
            ev = extract_evidence(x, reference=x[:tr])
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(files)}] SKIP {name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        summary = ev.summary()
        top_kind = summary["morphology"] if summary["ranked"] else "unknown"
        support = ev.support
        support_vus = fast_vus(support, y, sw)

        kind_auc = {}
        for k in KINDS:
            s = ev.scores[k]
            fired = bool(np.any(s >= 0.5))
            a = _auc(s, y) if fired else 0.0
            kind_auc[k] = a
            per_kind[k]["auc"].append(a)
            if fired:
                per_kind[k]["fire"] += 1
        best_kind = max(kind_auc, key=kind_auc.get) if summary["ranked"] else "unknown"
        if summary["ranked"]:
            per_kind[top_kind]["top"] += 1
            per_kind[best_kind]["best"] += 1

        det_vus = {}
        stem = Path(name).stem
        for d in DETS:
            p = SCORE_DIR / d / f"{stem}.npy"
            if p.exists():
                s = rank01(np.load(p, allow_pickle=False).ravel()[idx])
                det_vus[d] = fast_vus(s, y, sw)
        true_best_det = max(det_vus, key=det_vus.get) if det_vus else None
        if true_best_det:
            top_best_det[top_kind].append(true_best_det)
            kind_best_det[best_kind].append(true_best_det)

        feat = extract_features(x[:, 0].ravel())

        rows.append({
            "file": name, "source": source_of(name), "sw": sw,
            "top_kind": top_kind, "best_kind": best_kind, "top_ok": top_kind == best_kind,
            "support_vus": round(support_vus, 4),
            "true_best_det": true_best_det,
            "det_vus": {k: round(v, 4) for k, v in det_vus.items()},
            "kind_auc": {k: round(v, 4) for k, v in kind_auc.items()},
            "features": {k: round(float(v), 4) for k, v in feat.items()},
        })
        print(f"[{i}/{len(files)}] {source_of(name):8s} sw={sw:3d} top={top_kind:18s} "
              f"best_kind={best_kind:18s} ok={top_kind == best_kind} "
              f"sup_vus={support_vus:.3f} best_det={true_best_det} "
              f"({time.time()-t_file:.1f}s)", flush=True)

    n = len(rows)
    print(f"\n==== morphology validation (tuning {n}/{len(files)} files, {time.time()-t0:.0f}s) ====\n")

    print("== per-kind discriminative power (AUC vs label; fire=triggered with score≥0.5) ==")
    print(f"{'kind':20s} {'fire':>5s} {'top':>5s} {'best':>5s} {'mean_auc':>9s} {'median_auc':>11s}")
    for k in KINDS:
        v = per_kind[k]["auc"]
        mean = float(np.mean(v)) if v else 0.0
        med = float(np.median(v)) if v else 0.0
        print(f"{k:20s} {per_kind[k]['fire']:5d} {per_kind[k]['top']:5d} "
              f"{per_kind[k]['best']:5d} {mean:9.4f} {med:11.4f}")

    top_ok = sum(1 for r in rows if r["top_ok"])
    print(f"\n== routing-signal consistency ==")
    print(f"top kind == most discriminative kind: {top_ok}/{n} = {top_ok/n:.1%}")

    print("\n== morphology signal → true best detector (mode hit rate) ==")
    for label_name, mapping in [("top kind", top_best_det), ("best kind (most discriminative)", kind_best_det)]:
        hit = total = 0
        detail = []
        for k, dets in sorted(mapping.items()):
            mode = Counter(dets).most_common(1)[0][0]
            h = sum(1 for d in dets if d == mode)
            hit += h; total += len(dets)
            detail.append(f"{k}->{mode}({h}/{len(dets)})")
        print(f"[{label_name}] hit {hit}/{total} = {hit/max(total,1):.1%}")
        print("   " + "  ".join(detail))

    sv = [r["support_vus"] for r in rows]
    bv = [r["det_vus"][r["true_best_det"]] for r in rows if r["true_best_det"]]
    print(f"\n== morphology evidence support as a detector vs true best detector ==")
    print(f"support mean VUS-PR={np.mean(sv):.4f}  vs  true best mean={np.mean(bv):.4f}  "
          f"(gap {np.mean(bv)-np.mean(sv):+.4f})")

    feat_names = sorted({k for r in rows for k in r["features"]})
    print("\n== feature rank correlation (Spearman) ==")
    print(f"{'feature':16s} {'~support_vus':>13s} {'~best_det_vus':>15s} {'~top_ok':>9s}")
    for f in feat_names:
        vals = np.array([r["features"].get(f, np.nan) for r in rows], dtype=float)
        if np.all(np.isnan(vals)) or np.std(vals) < 1e-12:
            continue
        sv_arr = np.array([r["support_vus"] for r in rows])
        bv_arr = np.array([r["det_vus"][r["true_best_det"]] if r["true_best_det"] else np.nan
                           for r in rows], dtype=float)
        top_ok_arr = np.array([1.0 if r["top_ok"] else 0.0 for r in rows])
        print(f"{f:16s} {_corr(vals, sv_arr):13.3f} {_corr(vals, bv_arr):15.3f} "
              f"{_corr(vals, top_ok_arr):9.3f}")

    out = ROOT / "morph_validate_tuning.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[saved] {out.name}")


if __name__ == "__main__":
    main()
