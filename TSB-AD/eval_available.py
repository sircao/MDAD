"""eval_available.py —— Real-time evaluation for files that already have both scores, writing dashboard_data.json.

Loop continuously: scan the CHARM/Syntropic score directories, and for files where both scores are ready run
routing (evidence-v2 + morphology prior that does not drag performance down) + official VUS-PR, merging scoring progress into the dashboard.
"""
from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from Architecture_design.evidence_routing import route_scores  # noqa: E402
from Anomaly_algorithm.morphology import SOURCE_PRIOR  # noqa: E402
from TSB_AD.evaluation.metrics import get_metrics  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
SYN_DIR = ROOT / "eval" / "score" / "uni" / "Syntropic"
CHARM_DIR = ROOT / "eval" / "score" / "uni" / "CHARM_kNN"
OUT_JSON = ROOT / "dashboard_data.json"
PROG_JSON = ROOT / "score_progress.json"

MORPH_TO_DETECTOR = {
    "spike": "CHARM_kNN", "dense_cluster": "CHARM_kNN", "flatline": "CHARM_kNN",
    "level_shift": "Syntropic", "trend": "Syntropic", "variance_change": "Syntropic",
    "seasonal": "Syntropic", "noise": "Syntropic", "persistent": "Syntropic",
}


def train_length(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def rank01(s: np.ndarray) -> np.ndarray:
    return rankdata(s) / len(s)


def vus(score: np.ndarray, labels: np.ndarray, win: int) -> float:
    return float(get_metrics(np.asarray(score, dtype=float), labels.astype(int),
                             slidingWindow=win)["VUS-PR"])


def eval_one(name: str) -> dict:
    data = pd.read_csv(DATA_DIR / name).to_numpy(dtype=float)
    labels = pd.read_csv(LABEL_DIR / name)["Label"].to_numpy(dtype=float)
    tr = train_length(name)
    win = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
    src = name.split("_")[1]

    charm = np.load(CHARM_DIR / f"{Path(name).stem}.npy", allow_pickle=False)
    syn = np.load(SYN_DIR / f"{Path(name).stem}.npy", allow_pickle=False)
    c = {"CHARM_kNN": np.asarray(charm, dtype=float), "Syntropic": np.asarray(syn, dtype=float)}

    res = route_scores(data, c, baseline="CHARM_kNN", train_length=tr)
    sel = res.decision["selected"]
    reason = res.decision["reason"]
    prior_morph = SOURCE_PRIOR.get(src, "")
    prior_sel = MORPH_TO_DETECTOR.get(prior_morph, "")
    q_charm = res.decision["detectors"]["CHARM_kNN"]["quality"]
    q_syn = res.decision["detectors"]["Syntropic"]["quality"]
    prior_ok = (prior_sel == "CHARM_kNN" and q_charm >= q_syn) or \
               (prior_sel == "Syntropic" and q_syn >= q_charm)
    if prior_ok and prior_sel != sel and reason in ("insufficient_evidence", "baseline_retained"):
        sel = prior_sel
        reason = "source_prior_fallback"

    routed = c[sel]
    other = "Syntropic" if sel == "CHARM_kNN" else "CHARM_kNN"
    fused = 0.5 * (rank01(routed) + rank01(c[other]))
    w_charm = (0.75 if prior_sel == "CHARM_kNN" else 0.25) if prior_ok else 0.5
    fused_prior = w_charm * rank01(c["CHARM_kNN"]) + (1 - w_charm) * rank01(c["Syntropic"])

    charm_v = vus(c["CHARM_kNN"], labels, win)
    syn_v = vus(c["Syntropic"], labels, win)
    routed_v = charm_v if sel == "CHARM_kNN" else syn_v
    fused_v = vus(fused, labels, win)
    fused_prior_v = vus(fused_prior, labels, win)

    return {"file": name, "source": src, "selected": sel, "reason": reason,
            "CHARM_fixed": charm_v, "Syn_fixed": syn_v, "routed": routed_v,
            "fused": fused_v, "fused_prior": fused_prior_v}


def main() -> None:
    files = pd.read_csv(EVA_LIST)["file_name"].tolist()
    total = len(files)
    rows: list[dict] = []
    evaluated: set[str] = set()
    t_start = time.time()

    while True:
        syn_files = {p.stem + ".csv" for p in SYN_DIR.glob("*.npy")}
        charm_files = {p.stem + ".csv" for p in CHARM_DIR.glob("*.npy")}
        ready = sorted((syn_files & charm_files) & set(files))

        def _flush():
            agg = {k: float(np.mean([r[k] for r in rows])) for k in
                   ["CHARM_fixed", "Syn_fixed", "routed", "fused", "fused_prior"]} if rows else {}
            status = "done" if len(evaluated) >= total else ("running" if rows else "scoring")
            OUT_JSON.write_text(json.dumps({
                "status": status, "total": total, "done": len(evaluated),
                "syn_count": len(syn_files & set(files)), "charm_count": len(charm_files & set(files)),
                "started": t_start, "updated": time.time(), "aggregate": agg, "rows": rows,
            }, ensure_ascii=False), encoding="utf-8")
            print(f"[eval] {len(evaluated)}/{total} evaluated | Syn={len(syn_files & set(files))} "
                  f"CHARM={len(charm_files & set(files))}", flush=True)

        progressed = False
        for name in ready:
            if name in evaluated:
                continue
            try:
                rows.append(eval_one(name))
            except Exception as e:
                print(f"FAIL {name}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            evaluated.add(name)
            progressed = True
            _flush()

        if not progressed:
            _flush()

        scoring_done = len(charm_files & set(files)) >= total
        if len(evaluated) >= total and scoring_done:
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
