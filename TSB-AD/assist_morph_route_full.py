"""assist_morph_route_full.py — Morphology-prior routing assist (all 23 sources, base Syntropic + direction-matched morphology detector).

For each file that already has dumped per-point scores:
  Base    = Syntropic rank scores
  Assistant = morphology_routed_scores(x, source) (SOURCE_PRIOR→DETECTOR_ROUTING, label-free)
  Fusion    = max-union (union that raises recall to cover misses)

Reports two layers:
  AUC   —— full per-file (fast ranking proxy)
  VUS-PR —— subset (--vus-limit N files per source), per-file mean compared against routing's 68.71 under the same metric.

routing comparison = picks from llm_routing_combined.json × {Syntropic, CHARM_kNN, CHARM_ZS} per-file VUS-PR table.
"""
from __future__ import annotations

import json
import sys
import contextlib
import io
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Anomaly_algorithm.morphology import morphology_routed_scores  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from TSB_AD.evaluation.metrics import get_metrics  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402

ROOT = Path(__file__).resolve().parent
LDR = ROOT / "benchmark_exp" / "leaderboard_results"

# Synthetic switch: morphology → base(Syntropic) strength prior → assistant trust weight w.
# Syntropic is a statistical/segment-level detector: point anomalies (spike/dense/flatline) are weak → fully trust the assistant;
# segment-level (level_shift/variance_change/trend) are strong → keep the base.
ASSIST_WEIGHT = {
    "spike": 1.0, "dense_cluster": 0.8, "flatline": 0.8,
    "level_shift": 0.0, "variance_change": 0.0, "trend": 0.0,
    "noise": 0.3, "seasonal": 0.3, "persistent": 0.5,
}


def source_of(f: str) -> str:
    return f.split("_")[1]


def rank(s: np.ndarray) -> np.ndarray:
    r = rankdata(s)
    return r / len(r)


def auc(score, label) -> float:
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=int)
    if len(np.unique(y)) < 2 or np.allclose(s, s[0]):
        return 0.5
    return float(roc_auc_score(y, s))


def vus(score, label, sw) -> float:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return float(get_metrics(np.asarray(score).ravel(), label.astype(int),
                                 slidingWindow=sw)["VUS-PR"])


def load_routing_vus():
    """Per-file VUS-PR table: Syntropic + CHARM_kNN + CHARM_ZS."""
    syn = {}
    df = pd.read_csv(LDR / "Uni_Syntropic.csv")
    for _, r in df.iterrows():
        syn[r["file"][:-4] if r["file"].endswith(".csv") else r["file"]] = float(r["VUS-PR"])
    charm = {}
    df = pd.read_csv(LDR / "CHARM_uni_mergedTable_VUS-PR.csv")
    for _, r in df.iterrows():
        f = r["file"][:-4] if r["file"].endswith(".csv") else r["file"]
        charm[f] = (float(r["CHARM_kNN"]), float(r["CHARM_ZS"]))
    return syn, charm


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vus-limit", type=int, default=2, help="max files per source for VUS-PR")
    ap.add_argument("--metric", choices=["auc", "vus"], default="auc")
    args = ap.parse_args()

    syn_scores = dict(np.load(ROOT / "fullscores_Syntropic.npz"))
    syn_vus, charm = load_routing_vus()
    picks = json.load(open(ROOT / "runs" / "llm_routing_combined.json", encoding="utf-8"))["picks_u"]

    files = sorted(syn_scores.keys())
    print(f"dumped Syntropic per-point scores for {len(files)} files", flush=True)

    # per-source VUS-PR subset (shortest first, but already-dumped files taken in existing order)
    vus_subset = set()
    cnt = defaultdict(int)
    for f in files:
        s = source_of(f)
        if cnt[s] < args.vus_limit:
            vus_subset.add(f)
            cnt[s] += 1

    auc_rows = defaultdict(lambda: defaultdict(list))
    vus_rows = defaultdict(lambda: defaultdict(list))
    morphs = defaultdict(list)
    for f in files:
        src = source_of(f)
        df = pd.read_csv(f"Datasets/TSB-AD-U/{f}").dropna()
        x = df.iloc[:, 0:-1].values.astype(float)
        label = df["Label"].astype(int).to_numpy()

        base = rank(syn_scores[f])
        r = morphology_routed_scores(x, source=src)
        comp = rank(r["scores"])
        fused = np.maximum(base, comp)
        # Synthetic: morphology→weight w, soft raise-only fusion
        w = ASSIST_WEIGHT.get(r["morphology"], 0.5)
        fused_synth = base + w * np.clip(comp - base, 0, None)
        # confidence-gated max: trust the assistant to raise only when its rank score reaches the top q quantile
        fused_g90 = np.where(comp > 0.90, np.maximum(base, comp), base)
        fused_g95 = np.where(comp > 0.95, np.maximum(base, comp), base)

        for k, v in [("base", auc(base, label)), ("comp", auc(comp, label)),
                     ("fused", auc(fused, label)), ("synth", auc(fused_synth, label)),
                     ("gate90", auc(fused_g90, label)), ("gate95", auc(fused_g95, label))]:
            auc_rows[src][k].append(v)
        morphs[src].append((r["morphology"], w))

        if f in vus_subset:
            sw = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
            bv = vus(base, label, sw)
            fv = vus(fused, label, sw)
            sv = vus(fused_synth, label, sw)
            # routing comparison (per-file VUS-PR of the model picked for this source)
            fn = f[:-4] if f.endswith(".csv") else f
            pick = picks.get(src)
            rv = (syn_vus.get(fn, 0.0) if pick == "Syntropic"
                  else charm[fn][0] if pick == "CHARM_kNN" else charm[fn][1])
            vus_rows[src]["base"].append(bv)
            vus_rows[src]["fused"].append(fv)
            vus_rows[src]["synth"].append(sv)
            vus_rows[src]["routing"].append(rv)
        if len(auc_rows) % 20 == 0:
            print(f"  processed {len([1 for s in auc_rows for _ in auc_rows[s]['base']])} files", flush=True)

    def mean_of(d, k):
        return float(np.mean([v for s in d for v in d[s][k]]))

    print("\n==== AUC (full per-file) ====")
    for s in sorted(auc_rows):
        m, w = morphs[s][0] if morphs[s] else ("?", 0.5)
        print(f"  {s:14s} [{m:15s}w={w:.1f}] base={np.mean(auc_rows[s]['base']):.3f} "
              f"fused={np.mean(auc_rows[s]['fused']):.3f} "
              f"synth={np.mean(auc_rows[s]['synth']):.3f}")
    print(f"  {'Overall':14s} base={mean_of(auc_rows,'base'):.3f} "
          f"comp={mean_of(auc_rows,'comp'):.3f} fused={mean_of(auc_rows,'fused'):.3f} "
          f"synth={mean_of(auc_rows,'synth'):.3f} "
          f"gate90={mean_of(auc_rows,'gate90'):.3f} gate95={mean_of(auc_rows,'gate95'):.3f}")

    if vus_rows:
        print("\n==== VUS-PR (per-file subset, same metric as routing) ====")
        for s in sorted(vus_rows):
            print(f"  {s:14s} routing={np.mean(vus_rows[s]['routing'])*100:.2f} "
                  f"base={np.mean(vus_rows[s]['base'])*100:.2f} "
                  f"fused={np.mean(vus_rows[s]['fused'])*100:.2f} "
                  f"synth={np.mean(vus_rows[s]['synth'])*100:.2f}")
        print(f"  {'Overall':14s} routing={mean_of(vus_rows,'routing')*100:.2f} "
              f"base={mean_of(vus_rows,'base')*100:.2f} "
              f"fused={mean_of(vus_rows,'fused')*100:.2f} "
              f"synth={mean_of(vus_rows,'synth')*100:.2f}")


if __name__ == "__main__":
    main()
