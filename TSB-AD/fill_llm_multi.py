"""Fill in the multivariate LLM routing + Oracle cells of full_leaderboard.

Routing pool = the 9 models in the full-shot block that have official multivariate values (Syntropic has no multivariate, so it is excluded):
TimeRCD / TranAD / USAD / OmniAnomaly / LOF / IForest / Sub-PCA / DCdetector / TFMAE.

- Oracle: per source take the max over the 9 models (label upper bound, directly from the official leaderboard DATA).
- LLM routing: llama3.1-8B picks one of the 9 models per source using **label-free** features (channel-mean morphology + statistics),
  taking that model's official leaderboard value. Labels are used only for evaluation; decisions are label-free throughout.
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

from gen_full_leaderboard import DATA, SRC16, METRICS  # noqa: E402
from Anomaly_algorithm.morphology import extract_features, classify  # noqa: E402
from Architecture_design.client import OllamaClient  # noqa: E402

MUL = ["MSL", "PSM", "SMAP", "SMD", "SWaT"]
POOL = ["TimeRCD", "TranAD", "USAD", "OmniAnomaly", "LOF", "IForest", "Sub-PCA", "DCdetector", "TFMAE"]

REP = {
    "MSL": "Datasets/TSB-AD-M/002_MSL_id_1_Sensor_tr_500_1st_900.csv",
    "PSM": "Datasets/TSB-AD-M/115_PSM_id_1_Facility_tr_50000_1st_129872.csv",
    "SMAP": "Datasets/TSB-AD-M/144_SMAP_id_1_Sensor_tr_2052_1st_5300.csv",
    "SMD": "Datasets/TSB-AD-M/057_SMD_id_1_Facility_tr_4529_1st_4629.csv",
    "SWaT": "Datasets/TSB-AD-M/171_SWaT_id_1_Sensor_tr_3749_1st_9522.csv",
}

PROFILES = {
    "TimeRCD": "Zero-shot time-series root-cause foundation model; general-purpose, no clear morphology preference",
    "TranAD": "Adversarial transformer reconstruction; captures multivariate temporal dependencies",
    "USAD": "Dual autoencoder adversarial training; multivariate reconstruction",
    "OmniAnomaly": "VAE reconstruction family; strong on point anomalies / dense clusters",
    "LOF": "Local density outlier; sparse / isolated anomalies",
    "IForest": "Tree-ensemble isolation; general-purpose outlier detection",
    "Sub-PCA": "Subspace PCA reconstruction error; multivariate",
    "DCdetector": "Dual contrastive-learning representation; general-purpose",
    "TFMAE": "Masked autoencoder reconstruction; multivariate",
}

FEAT_KEYS = ["kurtosis", "skewness", "acf1", "slope", "cusum",
             "flat_ratio", "hfr", "spec_peak", "roll_std_cv", "max_over_std"]


def src_features(src):
    df = pd.read_csv(ROOT / REP[src]).dropna()
    x = df.iloc[:, 0:-1].values.astype(float)
    xm = x.mean(axis=1)
    feats = {k: extract_features(xm, window=0).get(k, 0.0) for k in FEAT_KEYS}
    morph = classify(xm, window=0, source=src)["morphology"]
    return feats, morph, int(x.shape[1])


def build_prompt(src, feats, morph, n_ch):
    f = feats
    lines = [
        "You are a multivariate anomaly-detection model-selection assistant. Given the label-free features of a multivariate data source, pick the model"
        " most likely to be strongest for this source from the 9 candidate models. Output only one model name; do not explain.",
        "",
        f"Data source: {src} ({n_ch} channels)",
        "Label-free statistical features of the channel-mean series:",
        f"  kurtosis={f['kurtosis']:.2f} skew={f['skewness']:.2f} acf1={f['acf1']:.2f} "
        f"slope={f['slope']:.2f} cusum={f['cusum']:.2f}",
        f"  flat_ratio={f['flat_ratio']:.3f} high-freq-energy-ratio={f['hfr']:.2f} "
        f"spectral-peak={f['spec_peak']:.1f} roll_std_cv={f['roll_std_cv']:.2f} max/σ={f['max_over_std']:.2f}",
        f"Local-rule morphology: {morph} (level_shift=drift, spike=point, dense_cluster=dense, "
        f"variance_change=variance, flatline=stuck, trend=trend, noise=noise)",
        "",
        "Candidate model profiles:",
    ]
    for m in POOL:
        lines.append(f"  - {m}: {PROFILES[m]}")
    lines += ["", "Combine source-name domain knowledge, statistical features and morphology to select the best model. Output only the model name."]
    return "\n".join(lines)


def parse_choice(text):
    t = (text or "").lower()
    for m in POOL:
        if m.lower() in t:
            return m
    return None


def main():
    # Oracle: per source take the max over the 9 models
    oracle = {mk: {} for mk, _ in METRICS}
    for mk, _ in METRICS:
        for src in MUL:
            j = SRC16.index(src)
            vals = [DATA[m][mk][j] for m in POOL]
            oracle[mk][src] = round(max(vals), 2)

    # LLM routing: llama3.1 per-source model selection
    client = OllamaClient(model="llama3.1:8b", timeout=180)
    picks = {}
    for src in MUL:
        feats, morph, n_ch = src_features(src)
        try:
            rep = client.chat([{"role": "user", "content": build_prompt(src, feats, morph, n_ch)}])
            p = parse_choice(rep["content"])
        except Exception as e:
            print(f"  [{src}] call failed {e}", flush=True)
            p = None
        picks[src] = p or "OmniAnomaly"
        print(f"  {src:5s} morph={morph:16s} -> {picks[src]}", flush=True)

    routing = {mk: {} for mk, _ in METRICS}
    for mk, _ in METRICS:
        for src in MUL:
            j = SRC16.index(src)
            routing[mk][src] = round(DATA[picks[src]][mk][j], 2)

    out = {"pool": POOL, "picks": picks, "oracle": oracle, "routing": routing}
    (ROOT / "runs" / "llm_routing_multi.json").parent.mkdir(exist_ok=True)
    (ROOT / "runs" / "llm_routing_multi.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print("\n=== Oracle (multivariate) ===")
    for mk, _ in METRICS:
        print(f"  {mk:14s}", {s: oracle[mk][s] for s in MUL})
    print("\n=== LLM routing (multivariate) ===")
    for mk, _ in METRICS:
        print(f"  {mk:14s}", {s: routing[mk][s] for s in MUL})
    print("\n[saved] runs/llm_routing_multi.json")


if __name__ == "__main__":
    main()
