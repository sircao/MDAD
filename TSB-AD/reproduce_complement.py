"""reproduce_complement.py —— Reuse the unsupervised score cache, clean routing + complement + evaluation.

Routing reads only the label-free Dataset; evaluation reads only Validation's Label.
Scores come from the old unsupervised cache (CHARM_kNN / Syntropic), whose per-file lengths are already verified aligned.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from Architecture_design.evidence_routing import route_scores  # noqa: E402
from TSB_AD.evaluation.metrics import get_metrics  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402

CACHE_DIR = ROOT / "eval" / "score" / "uni"
DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
EVA_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"


def train_length(name: str) -> int:
    m = re.search(r"_tr_(\d+)_", name)
    if m is None:
        raise ValueError(f"missing training-prefix length in {name}")
    return int(m.group(1))


def rank01(s: np.ndarray) -> np.ndarray:
    return rankdata(s) / len(s)


def main() -> None:
    charm = dict(np.load(CACHE_DIR / "fullscores_CHARM_kNN.npz"))
    syn = dict(np.load(CACHE_DIR / "fullscores_Syntropic.npz"))
    files = sorted(set(charm) & set(syn))
    eva = set(pd.read_csv(EVA_LIST)["file_name"].tolist())
    files = [f for f in files if f in eva]
    print(f"reused cache intersection (∩ U-Eva): {len(files)} files", flush=True)

    rows = []
    for name in files:
        data = pd.read_csv(DATA_DIR / name).to_numpy(dtype=float)
        labels = pd.read_csv(LABEL_DIR / name)["Label"].to_numpy(dtype=float)
        tr = train_length(name)

        c = {"CHARM_kNN": np.asarray(charm[name], dtype=float),
             "Syntropic": np.asarray(syn[name], dtype=float)}
        for m, s in c.items():
            if s.ndim != 1 or len(s) != len(data) or not np.isfinite(s).all():
                raise ValueError(f"{name}/{m}: bad cache score {s.shape}")

        # label-free routing (evidence-gated, conservative by default, only switches when evidence is strong)
        res = route_scores(data, c, baseline="CHARM_kNN", train_length=tr)
        sel = res.decision["selected"]
        routed = res.scores
        other = "Syntropic" if sel == "CHARM_kNN" else "CHARM_kNN"
        fused = 0.5 * (rank01(routed) + rank01(c[other]))

        win = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
        scores = {
            "CHARM_fixed": c["CHARM_kNN"],
            "Syn_fixed": c["Syntropic"],
            "routed": routed,
            "fused": fused,
        }
        row = {"file": name, "selected": sel}
        for tag, sc in scores.items():
            row[tag] = float(get_metrics(sc, labels.astype(int), slidingWindow=win)["VUS-PR"])
        rows.append(row)
        print(f"{name}: -> {sel} | CHARM={row['CHARM_fixed']:.4f} "
              f"Syn={row['Syn_fixed']:.4f} routed={row['routed']:.4f} "
              f"fused={row['fused']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "runs" / "reproduce_complement.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n==== mean VUS-PR ====")
    for col in ["CHARM_fixed", "Syn_fixed", "routed", "fused"]:
        print(f"  {col:12s} {df[col].mean():.4f}")
    print(f"\nresults saved to {out}")


if __name__ == "__main__":
    main()
