"""loop_routing_headroom.py — Measure the "per-file complement" headroom on top of two-level routing (per-file VUS-PR metric).

Answers: above routing (U 68.71 / M 53.94), how far "swapping files where the routing-chosen best algorithm errs for a complementary
algorithm that is fine on that file" can go. This is the **per-file lower bound** of the per-timestamp LOOP:
if per-file complement has no headroom, per-timestamp even more so cannot.

Currently available clean data (official/correct per-file VUS-PR):
  U: Syntropic (Uni_Syntropic.csv all 23 sources) + CHARM_kNN/ZS (CHARM_uni CSV all 23 sources)
  M: CHARM_kNN/ZS (CHARM_multi CSV all 17 sources)   ← for Time_RCD/OmniAnomaly/StreamVAE there is
                                                       no clean local M per-file VUS-PR table, so M headroom is underestimated

Three lines (per-file equal-weight mean, same metric as routing u_mean/m_mean):
  routing      = per file take the model the routing picks selected
  per-file oracle = per file take the max in the pool (per-file ceiling of the complement pool)
  loop (per-file greedy patch) = from the routing base, swap if a complement is stronger for that file
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LDR = ROOT / "benchmark_exp" / "leaderboard_results"


def norm_file(name: str) -> str:
    return name[:-4] if name.endswith(".csv") else name


def load_syntropic_uni():
    """Uni_Syntropic.csv -> {file: VUS-PR} (official, all 23 sources)."""
    df = pd.read_csv(LDR / "Uni_Syntropic.csv")
    out = {}
    for _, r in df.iterrows():
        out[norm_file(str(r["file"]))] = float(r["VUS-PR"])
    return out


def load_charm(csv: str):
    """CHARM_uni/multi merged CSV -> {file: (kNN, ZS)}."""
    df = pd.read_csv(LDR / csv)
    return {norm_file(str(r["file"])): (float(r["CHARM_kNN"]), float(r["CHARM_ZS"]))
            for _, r in df.iterrows()}


def source_of(f: str) -> str:
    return f.split("_")[1]


def report(files, picks, pool_of, name):
    """pool_of(file) -> {model: vus}; routing uses picks[src], the rest of the pool members are complements."""
    rows = defaultdict(lambda: {"routing": [], "oracle": []})
    all_route, all_oracle = [], []
    for f in files:
        src = source_of(f)
        pool = pool_of(f)
        if not pool or src not in picks:
            continue
        pick = picks[src]
        rv = pool.get(pick)
        if rv is None:
            continue
        ov = max(pool.values())
        rows[src]["routing"].append(rv)
        rows[src]["oracle"].append(ov)
        all_route.append(rv)
        all_oracle.append(ov)
    print(f"\n=== {name} ===")
    print(f"{'source':14s} {'n':>4s} {'routing':>8s} {'oracle':>8s} {'Δ':>7s}")
    for s in sorted(rows):
        r = float(np.mean(rows[s]["routing"])) * 100
        o = float(np.mean(rows[s]["oracle"])) * 100
        print(f"{s:14s} {len(rows[s]['routing']):4d} {r:8.2f} {o:8.2f} {o-r:+7.2f}")
    r = float(np.mean(all_route)) * 100
    o = float(np.mean(all_oracle)) * 100
    print(f"{'Overall':14s} {len(all_route):4d} {r:8.2f} {o:8.2f} {o-r:+7.2f}")
    return r, o


def main():
    picks = json.load(open(ROOT / "runs" / "llm_routing_combined.json", encoding="utf-8"))
    picks_u, picks_m = picks["picks_u"], picks["picks_m"]

    syn = load_syntropic_uni()
    charm_uni = load_charm("CHARM_uni_mergedTable_VUS-PR.csv")
    charm_multi = load_charm("CHARM_multi_mergedTable_VUS-PR.csv")

    # U: pool = {Syntropic, CHARM_kNN, CHARM_ZS} (three models, all 23 sources clean values)
    uni_files = sorted(charm_uni.keys())
    def pool_u(f):
        p = {"CHARM_kNN": charm_uni[f][0], "CHARM_ZS": charm_uni[f][1]}
        if f in syn:
            p["Syntropic"] = syn[f]
        return p

    # M: pool = {CHARM_kNN, CHARM_ZS} (no clean local M per-file table for Time_RCD/Omni/StreamVAE)
    multi_files = sorted(charm_multi.keys())
    def pool_m(f):
        return {"CHARM_kNN": charm_multi[f][0], "CHARM_ZS": charm_multi[f][1]}

    print("reproducing routing picks (compare against llm_routing_combined.json's u_mean/m_mean)")
    ru, ou = report(uni_files, picks_u, pool_u, "U (23 sources, pool=Syntropic+CHARM_kNN+CHARM_ZS)")
    rm, om = report(multi_files, picks_m, pool_m, "M (17 sources, pool=CHARM_kNN+CHARM_ZS, missing 3 SOTA)")
    print("\ncomparison target: routing U=68.71 / M=53.94")
    print(f"reproduced routing: U={ru:.2f} / M={rm:.2f}")
    print(f"per-file complement ceiling: U={ou:.2f} (Δ{ou-ru:+.2f}) / M={om:.2f} (Δ{om-rm:+.2f})")


if __name__ == "__main__":
    main()
