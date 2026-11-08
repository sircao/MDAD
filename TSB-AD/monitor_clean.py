"""monitor_clean.py —— Real-time validation monitor under the official metric (find_length_rank + get_metrics).

Routing reads only the label-free Dataset; evaluation reads only Validation's Label.
All VUS-PR are computed live with the official metric implementation (no old cache reuse, avoiding window-metric mismatch).
After each file is processed, write dashboard_data.json once for dashboard.html to poll.
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

SCORE_DIR = ROOT / "eval" / "score" / "uni"
DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
OUT_JSON = ROOT / "dashboard_data.json"

# morphology → detector direction (derived from the DETECTOR_ROUTING tool pool + the paper's detector profiles):
# point/dense/flatline → CHARM_kNN (kNN distance, point-level); segment-level/drift/fluctuation/periodic → Syntropic (multi-scale statistics)
MORPH_TO_DETECTOR = {
    "spike": "CHARM_kNN",
    "dense_cluster": "CHARM_kNN",
    "flatline": "CHARM_kNN",
    "level_shift": "Syntropic",
    "trend": "Syntropic",
    "variance_change": "Syntropic",
    "seasonal": "Syntropic",
    "noise": "Syntropic",
    "persistent": "Syntropic",
}


def train_length(name: str) -> int:
    m = re.search(r"_tr_(\d+)_", name)
    if m is None:
        raise ValueError(f"missing training-prefix length in {name}")
    return int(m.group(1))


def rank01(s: np.ndarray) -> np.ndarray:
    return rankdata(s) / len(s)


def vus(score: np.ndarray, labels: np.ndarray, win: int) -> float:
    # official metric: get_metrics(version='opt') + official sliding window
    return float(get_metrics(np.asarray(score, dtype=float), labels.astype(int),
                             slidingWindow=win)["VUS-PR"])


def write_json(payload: dict) -> None:
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    charm_scores = dict(np.load(SCORE_DIR / "fullscores_CHARM_kNN.npz"))
    syn_scores = dict(np.load(SCORE_DIR / "fullscores_Syntropic.npz"))

    files = sorted(set(charm_scores) & set(syn_scores))
    eva = set(pd.read_csv(EVA_LIST)["file_name"].tolist())
    files = [f for f in files if f in eva]
    total = len(files)
    print(f"total files {total}", flush=True)

    rows: list[dict] = []
    write_json({"status": "running", "total": total, "done": 0,
                "updated": time.time(), "aggregate": {}, "rows": []})

    for i, name in enumerate(files, 1):
        src = name.split("_")[1]
        data = pd.read_csv(DATA_DIR / name).to_numpy(dtype=float)
        labels = pd.read_csv(LABEL_DIR / name)["Label"].to_numpy(dtype=float)
        tr = train_length(name)
        # official sliding window
        win = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))

        c = {"CHARM_kNN": np.asarray(charm_scores[name], dtype=float),
             "Syntropic": np.asarray(syn_scores[name], dtype=float)}
        res = route_scores(data, c, baseline="CHARM_kNN", train_length=tr)
        sel = res.decision["selected"]
        reason = res.decision["reason"]
        # source-prior fallback: when evidence is insufficient to switch models, pick direction by the source morphology prior (does not override a switch already decided by evidence)
        prior_morph = SOURCE_PRIOR.get(src, "")
        prior_sel = MORPH_TO_DETECTOR.get(prior_morph, "")
        if (prior_sel and reason in ("insufficient_evidence", "baseline_retained")
                and prior_sel != sel):
            sel = prior_sel
            reason = "source_prior_fallback"
        routed = c[sel]
        other = "Syntropic" if sel == "CHARM_kNN" else "CHARM_kNN"
        fused = 0.5 * (rank01(routed) + rank01(c[other]))

        # official metric: compute three lines live; routed comes from the selected detector's official VUS-PR
        charm_v = vus(c["CHARM_kNN"], labels, win)
        syn_v = vus(c["Syntropic"], labels, win)
        routed_v = charm_v if sel == "CHARM_kNN" else syn_v
        fused_v = vus(fused, labels, win)

        rows.append({"file": name, "source": src, "selected": sel, "reason": reason,
                     "CHARM_fixed": charm_v, "Syn_fixed": syn_v,
                     "routed": routed_v, "fused": fused_v})
        agg = {k: float(np.mean([r[k] for r in rows])) for k in
               ["CHARM_fixed", "Syn_fixed", "routed", "fused"]}
        write_json({"status": "running", "total": total, "done": i,
                    "updated": time.time(), "aggregate": agg, "rows": rows})
        print(f"[{i}/{total}] {name} -> {sel}({reason}) "
              f"CHARM={charm_v:.4f} Syn={syn_v:.4f} fused={fused_v:.4f}", flush=True)

    final_agg = {k: float(np.mean([r[k] for r in rows])) for k in
                 ["CHARM_fixed", "Syn_fixed", "routed", "fused"]}
    write_json({"status": "done", "total": total, "done": total,
                "updated": time.time(), "aggregate": final_agg, "rows": rows})
    print("\n==== official-metric mean VUS-PR ====")
    for k in ["CHARM_fixed", "Syn_fixed", "routed", "fused"]:
        print(f"  {k:12s} {final_agg[k]:.4f}")


if __name__ == "__main__":
    main()
