"""validate_clean.py —— End-to-end validation under the anti-leakage separation protocol.

Routing/scoring reads only label-free detection data; evaluation reads only Validation's Label.
detectors is a registry of locally runnable scorers; charm/syntropic_full need network and are off by default.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from Architecture_design.evidence_routing import route_scores  # noqa: E402
from TSB_AD.evaluation.metrics import get_metrics  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402
from TSB_AD.models.Syntropic import wsk, diffz, robust_z  # noqa: E402


def train_length(name: str) -> int:
    m = re.search(r"_tr_(\d+)_", name)
    if m is None:
        raise ValueError(f"missing training-prefix length in {name}")
    return int(m.group(1))


def load_unlabeled(path: Path) -> np.ndarray:
    """Read label-free detection data: all columns are features, no Label column."""
    return pd.read_csv(path).to_numpy(dtype=float)


def load_label(path: Path) -> np.ndarray:
    """Read Label only from Validation, never participating in routing."""
    df = pd.read_csv(path)
    if "Label" not in df.columns:
        raise ValueError(f"{path}: no Label column")
    return df["Label"].to_numpy(dtype=float)


def score_syntropic_stat(data: np.ndarray, tr: int) -> np.ndarray:
    """Syntropic's local statistical component (WSK : DiffZ = 2 : 1), without Time-RCD."""
    x = np.asarray(data, dtype=float).ravel()
    a = wsk(x, tr)
    b = diffz(x, tr)
    return (2.0 * robust_z(a) + 1.0 * robust_z(b)) / 3.0


def score_iforest(data: np.ndarray, tr: int) -> np.ndarray:
    """sklearn IsolationForest's per-point anomaly scores (higher = more anomalous)."""
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    clf = IsolationForest(random_state=0, n_estimators=100).fit(x[:tr])
    s = -clf.decision_function(x)
    return np.asarray(s, dtype=float).ravel()


def score_knn(data: np.ndarray, tr: int) -> np.ndarray:
    """Local window-statistics kNN (same idea as CHARM's statistical part, purely local)."""
    x = np.asarray(data, dtype=float).ravel()
    w = max(8, tr // 4)
    from TSB_AD.models.Syntropic import window_stats, windows_to_points
    feats = window_stats(x, w)
    n_tr = max(tr - w + 1, 1)
    tr_feats = feats[:n_tr]
    mu, sd = tr_feats.mean(0), tr_feats.std(0) + 1e-8
    feats = (feats - mu) / sd
    tr_feats = feats[:n_tr]
    nn = NearestNeighbors(n_neighbors=min(3, len(tr_feats))).fit(tr_feats)
    dist, _ = nn.kneighbors(feats)
    return robust_z(windows_to_points(dist.mean(1), w, len(x)))


SCORERS = {
    "syntropic_stat": score_syntropic_stat,
    "iforest": score_iforest,
    "knn": score_knn,
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["U", "M"], required=True)
    ap.add_argument("--file-list", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--label-dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    for m in [args.baseline, *args.models]:
        if m not in SCORERS:
            ap.error(f"unknown model {m}; available: {sorted(SCORERS)}")
    models = list(dict.fromkeys([args.baseline, *args.models]))

    names = pd.read_csv(args.file_list)["file_name"].tolist()
    if args.limit:
        names = names[: args.limit]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in names:
        det_path = args.data_dir / name
        lab_path = args.label_dir / name
        data = load_unlabeled(det_path)
        labels = load_label(lab_path)
        if len(data) != len(labels):
            raise ValueError(f"{name}: data/label length mismatch {len(data)} vs {len(labels)}")
        tr = train_length(name)
        candidates = {}
        for m in models:
            s = SCORERS[m](data, tr)
            s = np.asarray(s, dtype=float)
            if s.ndim != 1 or len(s) != len(data) or not np.isfinite(s).all():
                raise ValueError(f"{name}/{m}: invalid score shape {s.shape} finite={np.isfinite(s).all()}")
            candidates[m] = s

        # label-free routing
        res = route_scores(data, candidates, baseline=args.baseline, train_length=tr)
        base = candidates[args.baseline]
        routed = res.scores
        sel = res.decision["selected"]

        # complement fusion: base = routing-chosen, complement = another model in the pool, rank average
        other = next(m for m in models if m != sel)
        r_b = np.argsort(np.argsort(routed))
        r_o = np.argsort(np.argsort(candidates[other]))
        fused = (r_b / len(data) + r_o / len(data)) / 2.0

        win = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
        v = {}
        for tag, sc in [("baseline", base), ("routed", routed), ("fused", fused)]:
            m = get_metrics(sc, labels.astype(int), slidingWindow=win)
            v[tag] = float(m["VUS-PR"])
        rows.append({"file": name, "selected": sel, **v})
        print(f"{name}: -> {sel} | base={v['baseline']:.4f} "
              f"routed={v['routed']:.4f} fused={v['fused']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    print("\n==== mean VUS-PR ====")
    for c in ["baseline", "routed", "fused"]:
        print(f"  {c:8s} {df[c].mean():.4f}")


if __name__ == "__main__":
    main()
