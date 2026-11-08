"""gen_full_leaderboard.py — Full leaderboard (zero-shot/full-shot) + Syntropic + LLM routing + Oracle.

16 sources (11 univariate + 5 multivariate) × 4 metrics. External model values come from the official leaderboard (DATA hard-coded).
- Syntropic: univariate only, multivariate columns are ---.
- LLM routing: llama3.1-8B label-free per-source selection; univariate reuses llm_select_progress's selection,
  multivariate selects within the block's model pool (zero-shot block 7 models / full-shot block 9 models).
- Oracle: per source take the max in that block's model pool (label upper bound).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC16 = ['IOPS', 'MGAB', 'NAB', 'NEK', 'Power', 'SED', 'Stock', 'TODS', 'UCR', 'WSD', 'YAHOO',
         'MSL', 'PSM', 'SMAP', 'SMD', 'SWaT']
MUL = ['MSL', 'PSM', 'SMAP', 'SMD', 'SWaT']
METRICS = [('Aff-F', 'Affiliation-F'), ('F1-T', 'F1$_T$'), ('StdF1', 'Standard-F1'), ('VUS-PR', 'VUS-PR')]

# ---- official leaderboard values (zero-shot + full-shot) ----
DATA = {
 'TimeRCD': {
  'Aff-F': [86.98,74.54,93.81,83.95,85.02,88.99,41.31,84.86,84.34,91.00,95.94,84.66,82.31,86.01,91.75,69.43],
  'F1-T': [45.39,9.05,50.49,76.91,23.05,60.59,18.38,68.46,41.18,44.13,79.14,43.67,43.86,35.82,55.96,35.47],
  'StdF1':[47.28,8.63,42.66,68.64,23.12,60.85,19.56,70.02,33.47,41.04,89.05,34.48,28.25,25.36,53.82,34.44],
  'VUS-PR':[42.98,3.09,44.65,71.13,14.63,69.86,72.73,91.67,27.48,29.95,87.88,23.41,21.69,19.68,46.70,18.47],
 },
 'DADA': {
  'Aff-F': [89.37,67.66,86.56,95.40,69.79,65.18,98.77,76.89,72.21,93.92,92.20,76.57,81.27,76.92,83.74,76.18],
  'F1-T': [42.50,0.91,37.24,47.98,19.80,9.56,95.49,35.18,7.22,48.46,79.52,34.58,31.84,30.42,40.80,35.13],
  'StdF1':[32.76,0.80,26.91,48.24,15.99,2.69,95.59,28.18,3.36,45.06,79.30,22.13,24.07,26.75,34.98,34.78],
  'VUS-PR':[24.97,0.57,24.73,46.85,10.61,6.42,99.51,64.83,2.94,33.42,70.74,12.74,17.17,20.02,25.98,21.13],
 },
 'MOMENT': {
  'Aff-F': [87.54,66.76,90.45,92.26,75.97,59.13,45.26,59.76,75.77,95.39,79.99,74.55,65.79,77.42,74.00,70.17],
  'F1-T': [33.15,0.80,52.27,63.66,19.91,9.54,18.04,17.47,13.02,41.98,11.69,25.97,27.77,17.93,28.68,28.76],
  'StdF1':[30.69,0.67,44.75,63.85,16.39,3.36,19.38,14.64,9.00,41.42,10.54,14.43,23.83,12.92,29.78,21.30],
  'VUS-PR':[37.35,0.56,45.38,67.74,10.50,4.31,76.97,56.45,6.17,55.26,30.81,9.32,16.48,8.97,15.96,14.90],
 },
 'TimesFM': {
  'Aff-F': [81.88,66.95,79.73,90.49,69.88,67.14,97.53,89.08,70.03,78.97,91.28,20.35,71.24,45.44,62.85,44.37],
  'F1-T': [48.95,0.93,36.74,36.63,19.80,9.58,88.94,51.13,10.78,41.38,83.46,7.83,25.42,11.64,18.65,21.39],
  'StdF1':[34.28,0.83,26.46,38.15,16.73,2.96,89.13,40.08,7.86,38.50,84.44,5.75,22.18,10.46,18.65,22.84],
  'VUS-PR':[19.56,0.58,24.01,35.02,10.44,6.13,98.39,72.89,6.03,21.57,86.78,11.84,14.76,16.95,13.02,19.43],
 },
 'Chronos': {
  'Aff-F': [90.12,67.89,86.66,93.63,69.72,67.89,96.85,91.96,74.35,90.98,96.34,75.52,70.88,72.22,75.31,70.43],
  'F1-T': [45.45,1.10,36.10,33.16,19.90,13.18,89.30,53.90,10.88,39.82,79.00,15.59,25.42,11.72,17.32,28.88],
  'StdF1':[32.69,0.99,26.22,33.54,17.47,8.74,89.41,40.52,8.21,34.58,78.89,11.63,22.27,9.62,17.50,24.03],
  'VUS-PR':[19.00,0.60,23.76,31.80,10.95,8.65,97.49,70.66,6.56,18.81,83.54,8.25,14.61,5.18,10.22,16.44],
 },
 'TimeMOE': {
  'Aff-F': [76.34,67.23,80.51,80.50,71.19,60.98,63.28,54.68,73.56,80.25,69.70,69.85,54.74,74.38,69.97,64.37],
  'F1-T': [25.95,0.63,38.70,15.78,19.85,17.73,34.13,20.91,8.29,22.60,37.11,23.92,26.82,14.22,19.90,30.11],
  'StdF1':[26.52,0.45,26.20,11.47,12.16,17.73,34.32,16.38,4.09,20.09,27.50,12.85,24.80,9.01,21.62,23.58],
  'VUS-PR':[16.63,0.52,22.62,19.76,9.34,10.87,74.78,48.78,2.10,10.93,20.90,7.82,15.68,4.98,11.12,16.20],
 },
 'MovingVar': {
  'Aff-F': [83.90,67.56,91.50,77.81,74.77,67.14,67.96,71.57,78.44,89.13,78.00,89.43,73.91,88.14,81.33,75.55],
  'F1-T': [22.53,0.85,56.66,55.72,19.93,9.59,17.53,16.43,12.85,22.01,13.89,41.08,26.13,24.13,32.67,27.11],
  'StdF1':[22.74,0.74,51.37,49.19,19.94,9.60,19.01,16.10,9.51,22.57,11.62,35.43,26.10,23.66,38.43,26.86],
  'VUS-PR':[22.47,0.44,48.75,38.81,10.70,5.17,73.99,51.58,6.90,22.83,19.85,25.02,16.45,12.73,21.43,14.49],
 },
 'TranAD': {
  'Aff-F': [83.19,67.28,90.28,85.02,71.56,61.03,57.94,52.76,73.31,84.34,76.08,79.91,73.83,87.39,92.20,75.37],
  'F1-T': [22.63,1.65,37.28,69.97,22.36,9.57,16.73,13.51,7.75,20.94,8.41,39.42,25.49,29.12,37.98,49.58],
  'StdF1':[34.85,1.46,27.33,60.36,22.36,2.63,16.23,11.94,4.40,20.23,5.70,29.60,25.63,25.11,43.99,61.86],
  'VUS-PR':[21.61,0.64,24.82,61.63,13.04,5.75,78.08,47.33,2.25,12.20,25.78,14.78,16.49,13.37,28.34,47.37],
 },
 'USAD': {
  'Aff-F': [71.08,67.81,91.54,71.13,76.48,55.60,35.92,47.90,76.00,65.10,53.05,81.86,57.86,87.25,85.09,75.06],
  'F1-T': [20.99,4.07,61.46,70.64,28.23,9.54,16.86,20.85,14.63,14.18,9.35,48.71,28.96,43.94,50.41,50.41],
  'StdF1':[30.66,3.89,56.15,62.91,28.24,3.41,17.99,23.87,10.74,13.20,7.21,38.71,28.41,38.66,53.06,62.82],
  'VUS-PR':[16.58,0.75,55.03,58.53,18.68,4.37,74.53,56.36,8.85,10.00,14.15,29.95,17.59,26.37,34.53,44.73],
 },
 'OmniAnomaly': {
  'Aff-F': [80.32,67.35,92.35,86.30,78.16,61.26,75.24,50.73,73.53,78.02,71.31,83.15,58.17,91.38,85.82,73.39],
  'F1-T': [51.17,1.61,40.09,82.20,23.48,9.68,36.22,14.33,8.47,34.79,24.16,49.36,30.42,46.63,51.84,46.64],
  'StdF1':[47.05,1.44,28.81,74.03,23.50,0.43,38.59,12.65,5.11,29.57,21.40,39.10,30.43,40.50,57.06,55.93],
  'VUS-PR':[25.35,0.64,27.17,74.51,14.32,6.20,91.29,45.55,2.40,16.37,29.26,31.57,18.58,28.07,37.44,42.97],
 },
 'LOF': {
  'Aff-F': [81.06,68.44,75.75,84.74,66.76,63.85,69.74,60.58,73.53,81.29,75.63,84.35,61.98,63.32,64.13,56.34],
  'F1-T': [27.97,1.15,35.76,63.57,19.80,9.60,66.14,31.63,8.31,24.38,55.93,38.97,25.58,21.81,10.13,30.62],
  'StdF1':[30.28,1.05,24.04,56.92,12.18,4.11,66.20,25.77,4.70,22.62,48.95,30.65,18.80,18.70,8.41,29.08],
  'VUS-PR':[19.43,0.57,21.18,58.52,9.31,6.81,83.07,49.14,2.39,12.85,41.37,24.67,13.58,10.59,4.40,14.50],
 },
 'IForest': {
  'Aff-F': [52.81,68.82,39.84,71.15,0.00,70.09,0.06,44.17,50.56,41.24,33.30,63.36,63.78,59.96,69.71,0.00],
  'F1-T': [7.64,0.84,21.44,65.56,0.00,9.54,1.10,11.06,6.36,4.28,4.90,20.73,25.39,14.32,16.20,0.00],
  'StdF1':[8.37,0.73,29.41,58.10,19.77,3.81,16.91,13.35,4.09,2.07,3.20,14.68,24.15,13.61,16.89,26.76],
  'VUS-PR':[8.59,0.62,23.57,56.50,11.56,7.71,70.99,46.62,2.88,2.06,10.47,11.29,15.85,7.55,8.88,15.49],
 },
 'Sub-PCA': {
  'Aff-F': [75.39,66.90,89.29,97.10,71.37,67.14,70.63,72.75,76.66,76.45,75.85,84.25,71.49,90.08,85.80,76.29],
  'F1-T': [32.75,0.98,54.15,84.18,20.30,9.54,20.10,18.40,18.90,24.62,11.57,49.02,30.38,44.31,51.87,46.65],
  'StdF1':[33.96,0.83,46.71,85.43,16.05,9.56,21.77,18.72,15.12,24.74,11.06,38.29,30.26,38.90,57.22,55.98],
  'VUS-PR':[23.02,0.60,46.08,88.91,10.49,3.72,80.86,54.16,12.92,16.41,21.57,31.43,18.52,26.42,37.50,43.02],
 },
 'DCdetector': {
  'Aff-F': [71.83,67.91,72.21,62.31,69.75,72.20,55.79,57.81,70.18,72.79,67.77,67.74,67.32,67.10,69.55,71.07],
  'F1-T': [6.61,1.32,32.72,29.21,21.13,10.53,16.07,16.40,6.62,7.32,6.81,23.24,25.34,15.73,9.47,28.64],
  'StdF1':[5.19,1.21,24.02,17.37,21.10,10.54,16.97,17.85,3.18,4.64,4.14,14.08,25.33,10.67,8.99,27.02],
  'VUS-PR':[5.83,0.59,16.60,14.03,12.32,9.37,74.16,46.66,1.53,3.23,10.17,7.01,14.49,4.21,4.66,15.04],
 },
 'TFMAE': {
  'Aff-F': [78.25,67.50,75.99,76.91,70.30,68.17,56.39,62.83,70.60,80.25,76.87,75.70,70.07,75.36,70.85,75.72],
  'F1-T': [19.41,1.07,33.04,31.11,20.18,11.82,22.15,16.48,5.90,19.11,23.85,25.28,25.36,19.39,10.13,28.46],
  'StdF1':[9.48,0.97,23.78,19.74,20.14,11.87,23.32,14.88,2.83,15.53,20.50,15.68,25.39,12.58,9.16,27.08],
  'VUS-PR':[5.32,0.64,15.68,17.81,11.90,9.55,73.54,48.79,2.57,5.36,25.93,8.25,14.22,5.76,4.77,15.38],
 },
}

ZERO_SHOT = ['TimeRCD', 'DADA', 'MOMENT', 'TimesFM', 'Chronos', 'TimeMOE', 'MovingVar']
FULL_SHOT = ['TimeRCD', 'TranAD', 'USAD', 'OmniAnomaly', 'LOF', 'IForest', 'Sub-PCA', 'DCdetector', 'TFMAE']

PROFILES = {
    'TimeRCD': 'Zero-shot time-series root-cause foundation model; general-purpose',
    'DADA': 'Anomaly-aware representation model; multivariate',
    'MOMENT': 'General-purpose time-series foundation model (zero-shot)',
    'TimesFM': 'Time-series foundation model (zero-shot forecasting)',
    'Chronos': 'Probabilistic time-series foundation model (zero-shot forecasting)',
    'TimeMOE': 'Mixture-of-experts time-series foundation model',
    'MovingVar': 'Moving-variance statistical detector; strong on segment-level drift',
    'TranAD': 'Adversarial transformer reconstruction; multivariate',
    'USAD': 'Dual autoencoder adversarial; multivariate reconstruction',
    'OmniAnomaly': 'VAE reconstruction; strong on point anomalies / dense clusters',
    'LOF': 'Local density outlier; sparse / isolated anomalies',
    'IForest': 'Tree-ensemble isolation; general-purpose outlier detection',
    'Sub-PCA': 'Subspace PCA reconstruction; multivariate',
    'DCdetector': 'Dual contrastive-learning representation; general-purpose',
    'TFMAE': 'Masked autoencoder reconstruction; multivariate',
}


def load_syntropic():
    import pandas as pd
    df = pd.read_csv(ROOT / 'benchmark_exp/leaderboard_results/Uni_Syntropic.csv')
    uni = ['IOPS', 'MGAB', 'NAB', 'NEK', 'Power', 'SED', 'Stock', 'TODS', 'UCR', 'WSD', 'YAHOO']
    cols = {'Aff-F': 10, 'F1-T': 7, 'StdF1': 6, 'VUS-PR': 4}
    agg = {mk: defaultdict(list) for mk, _ in METRICS}
    for i in range(len(df)):
        s = str(df.iloc[i, 0]).split('_')[1]
        if s in uni:
            for mk, c in cols.items():
                agg[mk][s].append(float(df.iloc[i, c]))
    out = {mk: [None] * 16 for mk, _ in METRICS}
    for mk, _ in METRICS:
        for j, s in enumerate(SRC16):
            if s in agg[mk] and agg[mk][s]:
                out[mk][j] = sum(agg[mk][s]) / len(agg[mk][s]) * 100
    return out


def load_local_fullcache(fullcache_name):
    """Local fullcache_*.json (11 univariate sources) + multicache_*.json (5 multivariate sources) → {metric: [16 source values]},
    missing sources/metrics are None. F1-T is the corrected 3σ F1_T."""
    d = json.load(open(ROOT / f'fullcache_{fullcache_name}.json'))
    agg = {mk: defaultdict(list) for mk, _ in METRICS}
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        s = k.split('/')[0]
        if s not in SRC16:
            continue
        for mk, _ in METRICS:
            if mk in v and isinstance(v[mk], (int, float)):
                agg[mk][s].append(v[mk] * 100)
    # multivariate: fill from multicache_*.json (Syntropic has no multicache, so multivariate is always None)
    mp = ROOT / f'multicache_{fullcache_name}.json'
    if mp.exists():
        md = json.load(open(mp))
        for k, v in md.items():
            if not isinstance(v, dict):
                continue
            s = k.split('_')[1]
            if s not in MUL:
                continue
            for mk, _ in METRICS:
                if mk in v and isinstance(v[mk], (int, float)):
                    agg[mk][s].append(v[mk] * 100)
    out = {mk: [None] * 16 for mk, _ in METRICS}
    for mk, _ in METRICS:
        for j, s in enumerate(SRC16):
            if agg[mk].get(s):
                out[mk][j] = float(np.mean(agg[mk][s]))
    return out


def load_llm_picks():
    p = ROOT / 'llm_select_progress.json'
    if not p.exists():
        return {}
    s = json.load(open(p))
    for r in s.get('runs', []):
        if r['name'] == 'llama3.1:8b':
            return r.get('picks', {})
    return {}


def load_multivar_routing(name):
    p = ROOT / 'runs' / name
    if not p.exists():
        return {}
    s = json.load(open(p))
    return s.get('routing', {})


def fmt(v, rank):
    s = f"{v:.2f}"
    if rank == 1:
        return r"\textcolor{red}{\textbf{" + s + r"}}"
    if rank == 2:
        return r"\textcolor{blue}{" + s + r"}"
    if rank == 3:
        return s + r"${}^{*}$"
    return s


def build_entity_values(block_pool, syntropic, picks, multi_routing, local):
    data = {m: {mk: list(DATA[m][mk]) for mk, _ in METRICS} for m in block_pool}
    # local inference overrides TimeRCD / OmniAnomaly (univariate; multivariate None)
    for name in ('TimeRCD', 'OmniAnomaly'):
        if name in data and name in local:
            data[name] = {mk: list(local[name][mk]) for mk, _ in METRICS}
    data['Syntropic'] = syntropic

    oracle = {mk: [None] * 16 for mk, _ in METRICS}
    for mk, _ in METRICS:
        for j in range(16):
            vals = [data[m][mk][j] for m in block_pool if data[m][mk][j] is not None]
            oracle[mk][j] = max(vals) if vals else None
    data['Oracle'] = oracle

    pick_map = {'Syntropic': 'Syntropic', 'Time_RCD': 'TimeRCD', 'OmniAnomaly': 'OmniAnomaly'}
    routing = {mk: [None] * 16 for mk, _ in METRICS}
    for mk, _ in METRICS:
        for j in range(11):
            src = SRC16[j]
            p = picks.get(src)
            if p in pick_map:
                m = pick_map[p]
                routing[mk][j] = local[m][mk][j]
        for j in range(11, 16):
            src = SRC16[j]
            if src in multi_routing.get(mk, {}):
                routing[mk][j] = multi_routing[mk][src]
    data['LLM routing'] = routing
    return data


def render_block(entities, data, block_title):
    lines = []
    n = len(entities)
    lines.append(r"\multicolumn{20}{c}{\textbf{" + block_title + r"}} \\")
    lines.append(r"\midrule")
    for mi, (mk, disp) in enumerate(METRICS):
        ranks = {}
        for j in range(16):
            scored = [(m, data[m][mk][j]) for m in entities if data[m][mk][j] is not None]
            scored.sort(key=lambda x: -x[1])
            ranks[j] = {m: i + 1 for i, (m, _) in enumerate(scored)}
        for i, e in enumerate(entities):
            first = (r"\multirow{" + str(n) + r"}{*}{" + disp + "}") if i == 0 else ""
            cells = []
            for j in range(16):
                v = data[e][mk][j]
                cells.append(fmt(v, ranks[j].get(e, 99)) if v is not None else "---")
            c1 = sum(1 for j in range(16) if ranks[j].get(e) == 1)
            c2 = sum(1 for j in range(16) if ranks[j].get(e) == 2)
            lines.append(first + " & " + e + " & " + " & ".join(cells) + f" & {c1:02d} & {c2:02d} \\\\")
        lines.append(r"\midrule")
    return lines


def main():
    local = {
        'TimeRCD': load_local_fullcache('Time_RCD'),
        'OmniAnomaly': load_local_fullcache('OmniAnomaly'),
        'Syntropic': load_local_fullcache('Syntropic'),
    }
    syntropic = local['Syntropic']
    picks = load_llm_picks()
    zs_routing = load_multivar_routing('llm_routing_multi_zs.json')
    fs_routing = load_multivar_routing('llm_routing_multi.json')

    if not zs_routing:
        zs_routing = {mk: {} for mk, _ in METRICS}
    if not fs_routing:
        fs_routing = {mk: {} for mk, _ in METRICS}

    zs_data = build_entity_values(ZERO_SHOT, syntropic, picks, zs_routing, local)
    fs_data = build_entity_values(FULL_SHOT, syntropic, picks, fs_routing, local)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\resizebox{\textwidth}{0.9\textheight}{%")
    lines.append(r"\begin{tabular}{@{}cc " + "c" * 16 + r" cc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Metric}} & \multirow{2}{*}{\textbf{Model}} & \multicolumn{11}{c}{\textbf{Univariate Datasets}} & \multicolumn{5}{c}{\textbf{Multivariate Datasets}} & \multirow{2}{*}{\makecell{\textbf{Total}\\\textbf{1st}}} & \multirow{2}{*}{\makecell{\textbf{Total}\\\textbf{2nd}}} \\")
    lines.append(r"\cmidrule(lr){3-13} \cmidrule(lr){14-18}")
    lines.append("& & " + " & ".join(r"\textbf{" + s + "}" for s in SRC16) + r" & & \\")
    lines.append(r"\midrule")

    lines += render_block(ZERO_SHOT + ['LLM routing', 'Oracle'], zs_data, "Zero-Shot Models")
    lines += render_block(FULL_SHOT + ['Syntropic', 'LLM routing', 'Oracle'], fs_data, "Full-Shot Models")

    lines.append(r"\end{tabular}%")
    lines.append("}")
    lines.append(r"\caption{Full TSB-AD leaderboard (zero-shot and full-shot blocks). The three locally reproduced detectors (Syntropic, Time-RCD, OmniAnomaly) use local inference results (fullcache) with the official F1$_T$ metric; all other models are the official leaderboard values. Each block reports the label-free LLM routing (llama3.1-8B) and its oracle upper bound (per-source best within the block) as the last two rows. Best per source in bold red, second blue, third marked with an asterisk; --- marks a missing cell (no local multivariate run, or a source/metric not yet run).}")
    lines.append(r"\label{tab:result}")
    lines.append(r"\end{table*}")

    tex = "\n".join(lines)
    out = ROOT.parent / 'paper' / 'result.tex'
    out.write_text(tex, encoding='utf-8')
    print(tex)
    print(f"\n[saved to {out}]")


if __name__ == '__main__':
    main()
