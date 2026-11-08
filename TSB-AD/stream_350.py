"""stream_350.py —— 350-file real-time streaming: scoring → routing → official VUS-PR → dashboard.

Once each file is processed (Syntropic local + CHARM remote, both resume) it is immediately routed + evaluated under the official metric,
writing dashboard_data.json for dashboard.html to poll. Routing includes the source-prior fallback.
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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

os.environ["CHARM_BASE_URL"] = "http://ab778f946c58843afa52a72d5af0657a-1381817648.us-west-2.elb.amazonaws.com:8080"
os.environ["CHARM_API_KEY"] = "token"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "benchmark_exp"))

from Architecture_design.evidence_routing import route_scores  # noqa: E402
from Anomaly_algorithm.morphology import SOURCE_PRIOR  # noqa: E402
from TSB_AD.evaluation.metrics import get_metrics  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402
from TSB_AD.model_wrapper import run_Syntropic  # noqa: E402
from Run_CHARM import run_CHARM_kNN  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
SYN_DIR = ROOT / "eval" / "score" / "uni" / "Syntropic"
CHARM_DIR = ROOT / "eval" / "score" / "uni" / "CHARM_kNN"
CHARM_CACHE = ROOT / "eval" / "score" / "uni" / "fullscores_CHARM_kNN.npz"
OUT_JSON = ROOT / "dashboard_data.json"

MORPH_TO_DETECTOR = {
    "spike": "CHARM_kNN", "dense_cluster": "CHARM_kNN", "flatline": "CHARM_kNN",
    "level_shift": "Syntropic", "trend": "Syntropic", "variance_change": "Syntropic",
    "seasonal": "Syntropic", "noise": "Syntropic", "persistent": "Syntropic",
}


def prior_weight(src: str) -> float:
    """morphology prior → fusion weight on CHARM (label-free, 0.75/0.5/0.25)."""
    morph = SOURCE_PRIOR.get(src, "")
    pref = MORPH_TO_DETECTOR.get(morph, "")
    if pref == "CHARM_kNN":
        return 0.75
    if pref == "Syntropic":
        return 0.25
    return 0.5


def train_length(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def rank01(s: np.ndarray) -> np.ndarray:
    return rankdata(s) / len(s)


def vus(score: np.ndarray, labels: np.ndarray, win: int) -> float:
    return float(get_metrics(np.asarray(score, dtype=float), labels.astype(int),
                             slidingWindow=win)["VUS-PR"])


def get_syntropic(name: str, data: np.ndarray, tr: int) -> np.ndarray:
    out = SYN_DIR / f"{Path(name).stem}.npy"
    if out.exists():
        return np.load(out, allow_pickle=False)
    s = np.asarray(run_Syntropic(data[:tr], data), dtype=float).ravel()
    np.save(out, s)
    return s


def get_charm(name: str, data: np.ndarray, tr: int) -> np.ndarray:
    out = CHARM_DIR / f"{Path(name).stem}.npy"
    if out.exists():
        return np.load(out, allow_pickle=False)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        s = np.asarray(run_CHARM_kNN(data[:tr], data), dtype=float).ravel()
    np.save(out, s)
    return s


def write_json(payload: dict) -> None:
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    files = pd.read_csv(EVA_LIST)["file_name"].tolist()
    # flush the old CHARM cache to disk first
    if CHARM_CACHE.exists():
        cache = dict(np.load(CHARM_CACHE))
        for name, sc in cache.items():
            out = CHARM_DIR / f"{Path(name).stem}.npy"
            if not out.exists():
                np.save(out, np.asarray(sc, dtype=float).ravel())
    total = len(files)
    rows: list[dict] = []
    t_start = time.time()
    write_json({"status": "running", "total": total, "done": 0,
                "started": t_start, "updated": time.time(), "aggregate": {}, "rows": []})

    for i, name in enumerate(files, 1):
        src = name.split("_")[1]
        data = pd.read_csv(DATA_DIR / name).to_numpy(dtype=float)
        labels = pd.read_csv(LABEL_DIR / name)["Label"].to_numpy(dtype=float)
        tr = train_length(name)
        win = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))

        t0 = time.time()
        syn_s = get_syntropic(name, data, tr)
        charm_s = get_charm(name, data, tr)
        c = {"CHARM_kNN": charm_s, "Syntropic": syn_s}
        for m, s in c.items():
            if s.ndim != 1 or len(s) != len(data) or not np.isfinite(s).all():
                raise ValueError(f"{name}/{m}: bad score {s.shape}")

        res = route_scores(data, c, baseline="CHARM_kNN", train_length=tr)
        sel = res.decision["selected"]
        reason = res.decision["reason"]
        prior_morph = SOURCE_PRIOR.get(src, "")
        prior_sel = MORPH_TO_DETECTOR.get(prior_morph, "")
        # morphology prior + source-prior fallback, but "no dragging down": it only applies when the prior direction's label-free evidence quality is not lower than the other direction
        q_charm = res.decision["detectors"]["CHARM_kNN"]["quality"]
        q_syn = res.decision["detectors"]["Syntropic"]["quality"]
        prior_ok = (prior_sel == "CHARM_kNN" and q_charm >= q_syn) or \
                   (prior_sel == "Syntropic" and q_syn >= q_charm)
        if (prior_ok and prior_sel != sel
                and reason in ("insufficient_evidence", "baseline_retained")):
            sel = prior_sel
            reason = "source_prior_fallback"
        routed = c[sel]
        other = "Syntropic" if sel == "CHARM_kNN" else "CHARM_kNN"
        fused = 0.5 * (rank01(routed) + rank01(c[other]))
        # morphology-prior weighted complement (label-free; bias only when the prior is valid, otherwise neutral 0.5)
        if prior_ok:
            w_charm = 0.75 if prior_sel == "CHARM_kNN" else 0.25
        else:
            w_charm = 0.5
        fused_prior = w_charm * rank01(c["CHARM_kNN"]) + (1 - w_charm) * rank01(c["Syntropic"])

        charm_v = vus(c["CHARM_kNN"], labels, win)
        syn_v = vus(c["Syntropic"], labels, win)
        routed_v = charm_v if sel == "CHARM_kNN" else syn_v
        fused_v = vus(fused, labels, win)
        fused_prior_v = vus(fused_prior, labels, win)

        rows.append({"file": name, "source": src, "selected": sel, "reason": reason,
                     "CHARM_fixed": charm_v, "Syn_fixed": syn_v,
                     "routed": routed_v, "fused": fused_v, "fused_prior": fused_prior_v})
        agg = {k: float(np.mean([r[k] for r in rows])) for k in
               ["CHARM_fixed", "Syn_fixed", "routed", "fused", "fused_prior"]}
        write_json({"status": "running", "total": total, "done": i,
                    "started": t_start, "updated": time.time(), "aggregate": agg, "rows": rows})
        print(f"[{i}/{total}] {name} -> {sel}({reason}) "
              f"CHARM={charm_v:.4f} Syn={syn_v:.4f} fused={fused_v:.4f} "
              f"fused_prior={fused_prior_v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    final_agg = {k: float(np.mean([r[k] for r in rows])) for k in
                 ["CHARM_fixed", "Syn_fixed", "routed", "fused", "fused_prior"]}
    write_json({"status": "done", "total": total, "done": total,
                "started": t_start, "updated": time.time(), "aggregate": final_agg, "rows": rows})
    print("\n==== official-metric 350-file mean VUS-PR ====")
    for k in ["CHARM_fixed", "Syn_fixed", "routed", "fused", "fused_prior"]:
        print(f"  {k:12s} {final_agg[k]:.4f}")


if __name__ == "__main__":
    main()
