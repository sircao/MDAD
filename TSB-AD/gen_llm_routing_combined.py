"""Two-level routing entry point: evidence-v2 by default, --method legacy runs the old metric-table routing.

The new version uses per-file, per-channel morphology evidence and real per-point scores, and does not read the leaderboard to pick models.
Parameters and protocol are in Architecture_design/README.md. The following notes apply only to legacy:

Two-level LLM routing (morphology → pick LLM → LLM picks detector) + official VUS-PR.

Morphology → LLM mapping = data-driven (A) as primary + prior weight (B) to break ties:
  A: per-morphology Aff-F scores of the 5 local LLMs' selections over the 11 sources in llm_select_progress.json;
  B: LLM_WEIGHTS from architecture.py.
  level_shift/flatline three-way tie → B's highest qwen3-30b; spike/variance → llama3.1 (A's highest);
  dense_cluster → qwen2.5 (A's highest); no observed morphology → llama3.1 (A best overall).

Detector pool: U = {Syntropic, Time_RCD, OmniAnomaly, StreamVAE}; M = {Time_RCD, OmniAnomaly, StreamVAE}.
Scoring: official per-file VUS-PR, equal-weight mean over all files (same metric as the official leaderboard 0.52/0.32).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from Anomaly_algorithm.morphology import extract_features, classify  # noqa: E402
from Architecture_design.client import OllamaClient  # noqa: E402
from gen_llm_routing_official import (  # noqa: E402
    build_vus, src_features, parse_choice, PROFILES,
)

LDR = ROOT / "benchmark_exp" / "leaderboard_results"
DATASETS = ROOT / "Datasets"

# morphology -> LLM (A+B combined)
MORPH_TO_LLM = {
    "level_shift": "qwen3-coder-30b-a3b:latest",
    "flatline": "qwen3-coder-30b-a3b:latest",
    "spike": "llama3.1:8b",
    "variance_change": "llama3.1:8b",
    "dense_cluster": "qwen2.5-coder:7b",
    "trend": "llama3.1:8b",
    "noise": "llama3.1:8b",
    "seasonal": "llama3.1:8b",
    "persistent": "llama3.1:8b",
}

POOL_U = ["Syntropic", "Time_RCD", "OmniAnomaly", "StreamVAE", "CHARM_kNN", "CHARM_ZS"]
POOL_M = ["Time_RCD", "OmniAnomaly", "StreamVAE", "CHARM_kNN", "CHARM_ZS"]

# model strength prior (this benchmark's overall VUS-PR, empirical; same family as architecture.py's LLM_WEIGHTS)
STRENGTH = {
    "CHARM_kNN": "U 0.68 / M 0.54",
    "Syntropic": "U 0.67 / M n/a",
    "CHARM_ZS": "U 0.61 / M 0.46",
    "Time_RCD": "U 0.52 / M 0.32",
    "StreamVAE": "U 0.45 / M 0.44",
    "OmniAnomaly": "U 0.29 / M 0.31",
}


def source_of(fname):
    return fname.split("_")[1]


def prompt_pick(src, f, morph, pool):
    lines = [
        "You are an anomaly-detection model-selection assistant. Given the label-free features of a data source, pick the model"
        " most likely to be strongest for this source from the candidate models. Output only one model name; do not explain.",
        "",
        f"Data source: {src}  morphology: {morph}",
        f"Features: kurtosis={f['kurtosis']:.2f} skew={f['skewness']:.2f} acf1={f['acf1']:.2f} "
        f"slope={f['slope']:.2f} cusum={f['cusum']:.2f}",
        f"flat_ratio={f['flat_ratio']:.3f} hfr={f['hfr']:.2f} spec_peak={f['spec_peak']:.1f} "
        f"roll_std_cv={f['roll_std_cv']:.2f} max/σ={f['max_over_std']:.2f}",
        "",
        "Overall model strength (VUS-PR performance on this benchmark, for reference alongside morphology; not the sole basis):",
    ]
    for m in pool:
        lines.append(f"  - {m}: {PROFILES[m]} (overall VUS-PR {STRENGTH[m]})")
    lines += ["", "Prefer the model that is both strong overall and morphology-matched; output only the model name."]
    return "\n".join(lines)


def pick_with_llm(client, src, files, subdir, pool):
    f, morph = src_features(src, files, subdir)
    if f is None:
        return pool[0], "default"
    try:
        rep = client.chat([{"role": "user", "content": prompt_pick(src, f, morph, pool)}])
        p = parse_choice(rep["content"], pool)
    except Exception as e:
        print(f"    [{src}] LLM call failed {str(e)[:40]}", flush=True)
        p = None
    return (p or pool[0]), morph


def legacy_main():
    vus = build_vus()
    u_files = pd.read_csv(DATASETS / "File_List" / "TSB-AD-U-Eva.csv")["file_name"].tolist()
    m_files = pd.read_csv(DATASETS / "File_List" / "TSB-AD-M-Eva.csv")["file_name"].tolist()

    u_by_src, m_by_src = {}, {}
    for f in u_files:
        u_by_src.setdefault(source_of(f), []).append(f)
    for f in m_files:
        m_by_src.setdefault(source_of(f), []).append(f)

    # the 11 sources already have each LLM's selection (reuse directly, do not re-run)
    existing = {}
    p = ROOT / "llm_select_progress.json"
    if p.exists():
        s = json.load(open(p))
        for r in s.get("runs", []):
            if r.get("n", 0) > 0:
                existing[r["name"]] = r.get("picks", {})
    morph_of = s.get("morph", {})

    # final detector per source: picks[src] -> detection model
    picks_u, picks_m = {}, {}
    clients = {}

    def get_client(llm):
        if llm not in clients:
            clients[llm] = OllamaClient(model=llm, timeout=240)
        return clients[llm]

    print("=== U (23 sources) two-level routing (pool includes CHARM, all run now) ===", flush=True)
    for src, files in u_by_src.items():
        f, morph = src_features(src, files, "TSB-AD-U")
        g = morph if morph else "persistent"
        llm = MORPH_TO_LLM.get(g, "llama3.1:8b")
        det, _ = pick_with_llm(get_client(llm), src, files, "TSB-AD-U", POOL_U)
        picks_u[src] = det
        print(f"  U {src:14s} morph={g:16s} LLM={llm.split(':')[0]:12s} -> {det}", flush=True)

    print("=== M (17 sources) two-level routing ===", flush=True)
    for src, files in m_by_src.items():
        f, morph = src_features(src, files, "TSB-AD-M")
        g = morph if morph else "persistent"
        llm = MORPH_TO_LLM.get(g, "llama3.1:8b")
        det, _ = pick_with_llm(get_client(llm), src, files, "TSB-AD-M", POOL_M)
        picks_m[src] = det
        print(f"  M {src:14s} morph={g:16s} LLM={llm.split(':')[0]:12s} -> {det}", flush=True)

    def mean_vus(files, picks, pool):
        vals, miss = [], 0
        for f in files:
            m = picks.get(source_of(f))
            if m is None or m not in pool:
                miss += 1
                continue
            v = vus[m].get(f)
            if v is None:
                miss += 1
            else:
                vals.append(v)
        return (np.mean(vals) * 100 if vals else float("nan")), len(vals), miss

    u_mean, u_n, u_miss = mean_vus(u_files, picks_u, POOL_U)
    m_mean, m_n, m_miss = mean_vus(m_files, picks_m, POOL_M)

    print("\n" + "=" * 60)
    print(f"TSB-AD-U  LLM routing(two-level) mean VUS-PR = {u_mean:.2f}%  (coverage {u_n}/{len(u_files)})")
    print(f"TSB-AD-M  LLM routing(two-level) mean VUS-PR = {m_mean:.2f}%  (coverage {m_n}/{len(m_files)})")

    out = {"picks_u": picks_u, "picks_m": picks_m, "morph_to_llm": MORPH_TO_LLM,
           "u_mean_vus_pr": round(u_mean, 4), "m_mean_vus_pr": round(m_mean, 4)}
    (ROOT / "runs" / "llm_routing_combined.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n[saved] runs/llm_routing_combined.json")


def main(argv=None):
    """Default to score-backed v2; historical table lookup is explicit opt-in."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", choices=["evidence", "legacy"], default="evidence")
    args, remaining = parser.parse_known_args(argv)
    if args.method == "legacy":
        if remaining:
            parser.error("legacy mode does not accept additional arguments")
        legacy_main()
    else:
        from evidence_route_bench import main as evidence_main
        evidence_main(remaining)


if __name__ == "__main__":
    main()
