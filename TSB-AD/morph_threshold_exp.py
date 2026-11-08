"""morph_threshold_exp.py — Experiment B: lower extract_evidence's threshold and see whether the extra morphologies can
push the "morphology→best detector" hit rate above the 68% baseline of unknown→Syntropic.

Experiment only on the 22 original unknown files: kind_prior is learned from the 26 files with morphology (top_kind→best mode),
re-run extract_evidence for each threshold config, route kind_prior[top_kind] (unknown falls back to Syntropic).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from Anomaly_algorithm.morphology_evidence import extract_evidence  # noqa: E402

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
MAX_LEN = 25000
DROP = {"relationship_change"}

CONFIGS = {
    "default (current)": {},
    "B (moderate)": {"fire": 0.25, "spike_floor": 3.0, "spike_q": 0.99, "spike_mult": 1.2},
    "B+ (aggressive)": {"fire": 0.1,
                "spike_floor": 2.5, "spike_q": 0.98, "spike_mult": 1.0,
                "level_floor": 0.5, "level_q": 0.99, "level_add": 0.3,
                "var_floor": 1.2, "var_q": 0.99, "var_mult": 1.0,
                "trend_floor": 0.8, "trend_q": 0.99, "trend_mult": 1.5,
                "seasonal_floor": 0.8, "seasonal_q": 0.99, "seasonal_mult": 1.5},
}


def tr_of(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def top_kind_of(name: str, thresholds: dict) -> str:
    full = pd.read_csv(DATA_DIR / name).to_numpy(dtype=float)
    n = len(full)
    idx = np.arange(n)
    if n > MAX_LEN:
        step = int(np.ceil(n / MAX_LEN))
        idx = np.arange(0, n, step)
    x = full[idx]
    tr = int(np.searchsorted(idx, tr_of(name)))
    ev = extract_evidence(x, reference=x[:tr], thresholds=thresholds)
    m = ev.summary()["morphology"]
    return "unknown" if m in DROP else m


def main() -> None:
    rows = json.loads((ROOT / "morph_validate_tuning.json").read_text(encoding="utf-8"))
    unknown = [r for r in rows if r["top_kind"] == "unknown"]
    known = [r for r in rows if r["top_kind"] != "unknown"]

    # kind_prior: learn top_kind -> best-detector mode from the files with morphology (excluding unknown)
    kind_prior = {}
    grouped = {}
    for r in known:
        grouped.setdefault(r["top_kind"], []).append(r["true_best_det"])
    for k, dets in grouped.items():
        kind_prior[k] = Counter(dets).most_common(1)[0][0]
    print("kind_prior (learned from the 26 files with morphology):")
    for k, d in sorted(kind_prior.items()):
        print(f"  {k:18s} -> {d}")

    # baseline: all 22 unknown → Syntropic
    base_hit = sum(1 for r in unknown if r["true_best_det"] == "Syntropic")
    print(f"\nbaseline unknown→Syntropic: {base_hit}/{len(unknown)} = {base_hit/len(unknown):.1%}\n")

    for cfg_name, thresholds in CONFIGS.items():
        hits = 0
        surfaced = 0
        lines = []
        for r in unknown:
            tk = top_kind_of(r["file"], thresholds)
            route = kind_prior.get(tk, "Syntropic") if tk != "unknown" else "Syntropic"
            ok = route == r["true_best_det"]
            hits += int(ok)
            if tk != "unknown":
                surfaced += 1
            lines.append(f"    {r['source']:8s} {r['file'][:40]:40s} "
                         f"{r['top_kind']:18s} -> {tk:18s} route={route:12s} true={r['true_best_det']:12s} {'OK' if ok else 'X'}")
        print(f"== {cfg_name} ==")
        print(f"  surfaced {surfaced}/{len(unknown)}, hit {hits}/{len(unknown)} = {hits/len(unknown):.1%}")
        for ln in lines:
            print(ln)
        print()


if __name__ == "__main__":
    main()
