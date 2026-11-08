"""morph_diag_unknown.py — Dig into P2: why extract_evidence cannot determine a morphology for 46% of tuning files.

Reuse the top_kind results in morph_validate_tuning.json, re-run extract_evidence only on unknown files + a few files with morphology,
print each kind's max score, support max, and label distribution (total/test segment), to see whether it is "just below the trigger"
or "completely silent", and whether anomalies fall in the reference segment (before tr) so the threshold is polluted.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from Anomaly_algorithm.morphology_evidence import extract_evidence, KINDS  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
MAX_LEN = 25000


def tr_of(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def diagnose(name: str) -> dict:
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
    ev = extract_evidence(x, reference=x[:tr])
    kinds = {k: float(ev.scores[k].max()) for k in KINDS}
    top3 = sorted(kinds.items(), key=lambda kv: -kv[1])[:3]
    return {
        "support_max": float(ev.support.max()),
        "top3": " ".join(f"{k}={v:.2f}" for k, v in top3),
        "kinds": {k: round(v, 3) for k, v in kinds.items()},
        "lbl_frac": float(y.mean()),
        "lbl_after_tr": float(y[tr:].mean()) if tr < len(y) else float("nan"),
        "lbl_before_tr": float(y[:tr].mean()) if tr > 0 else float("nan"),
        "n": int(len(y)),
    }


def main() -> None:
    rows = json.loads((ROOT / "morph_validate_tuning.json").read_text(encoding="utf-8"))
    unknown = [r for r in rows if r["top_kind"] == "unknown"]
    known = [r for r in rows if r["top_kind"] != "unknown"][:6]

    cache = {}
    for r in unknown + known:
        cache[r["file"]] = diagnose(r["file"])

    print(f"unknown {len(unknown)} files; with-morphology sample {len(known)} files\n")
    hdr = f"{'file':44s} {'src':8s} {'sup':>5s} {'lbl':>5s} {'aft_tr':>7s} {'bef_tr':>7s}  top3"
    print("== UNKNOWN files ==")
    print(hdr)
    for r in unknown:
        d = cache[r["file"]]
        print(f"{r['file']:44s} {r['source']:8s} {d['support_max']:5.2f} "
              f"{d['lbl_frac']:5.3f} {d['lbl_after_tr']:7.3f} {d['lbl_before_tr']:7.3f}  {d['top3']}")

    print("\n== files with morphology (comparison) ==")
    print(hdr)
    for r in known:
        d = cache[r["file"]]
        print(f"{r['file']:44s} {r['source']:8s} {d['support_max']:5.2f} "
              f"{d['lbl_frac']:5.3f} {d['lbl_after_tr']:7.3f} {d['lbl_before_tr']:7.3f}  {d['top3']}")

    # summarize: per-kind mean max of unknown (see which is closest to 0.5)
    print("\n== per-kind mean max of UNKNOWN files (which is closest to the 0.5 threshold) ==")
    agg = {k: [] for k in KINDS}
    for r in unknown:
        for k in KINDS:
            agg[k].append(cache[r["file"]]["kinds"][k])
    for k in KINDS:
        vals = agg[k]
        print(f"  {k:20s} mean_max={np.mean(vals):.3f}  max_max={np.max(vals):.3f}  "
              f"n>=0.3={sum(1 for v in vals if v >= 0.3)}/{len(vals)}")


if __name__ == "__main__":
    main()
