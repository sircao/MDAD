"""gen_paper2_sota_table.py — Generate a leaderboard-style LaTeX table for SOTA models + LLM routing.

Output paper/paper2_sota_table.tex: 4 metrics × 5 methods × 11 univariate sources + Mean + 1st + 2nd.
Methods: Syntropic / Time-RCD / OmniAnomaly / LLM routing / Oracle. The three detector values come from local
fullcache_*.json (real inference), LLM routing attaches the local score after llama3.1 selection, Oracle is the per-source max.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SRC11 = ['IOPS', 'MGAB', 'NAB', 'NEK', 'Power', 'SED', 'Stock', 'TODS', 'UCR', 'WSD', 'YAHOO']
MODELS = ['Syntropic', 'Time_RCD', 'OmniAnomaly']
METRICS = [('Aff-F', 'Affiliation-F'), ('F1-T', 'F1$_T$'), ('StdF1', 'Standard-F1'), ('VUS-PR', 'VUS-PR')]
ENTITIES = ['Syntropic', 'Time_RCD', 'OmniAnomaly', 'LLM routing', 'Oracle']
DISP = {'Syntropic': 'Syntropic', 'Time_RCD': 'Time-RCD', 'OmniAnomaly': 'OmniAnomaly',
        'LLM routing': 'LLM routing', 'Oracle': 'Oracle'}

# official TSB-AD leaderboard published values (Time-RCD = Zero-Shot TimeRCD row; OmniAnomaly = Full-Shot row).
# Syntropic is not on the leaderboard; use the official submission Uni_Syntropic.csv (= local fullcache, already verified identical).
OFFICIAL = {
    'Time_RCD': {
        'Aff-F': {'IOPS': 86.98, 'MGAB': 74.54, 'NAB': 93.81, 'NEK': 83.95, 'Power': 85.02, 'SED': 88.99, 'Stock': 41.31, 'TODS': 84.86, 'UCR': 84.34, 'WSD': 91.00, 'YAHOO': 95.94},
        'F1-T': {'IOPS': 45.39, 'MGAB': 9.05, 'NAB': 50.49, 'NEK': 76.91, 'Power': 23.05, 'SED': 60.59, 'Stock': 18.38, 'TODS': 68.46, 'UCR': 41.18, 'WSD': 44.13, 'YAHOO': 79.14},
        'StdF1': {'IOPS': 47.28, 'MGAB': 8.63, 'NAB': 42.66, 'NEK': 68.64, 'Power': 23.12, 'SED': 60.85, 'Stock': 19.56, 'TODS': 70.02, 'UCR': 33.47, 'WSD': 41.04, 'YAHOO': 89.05},
        'VUS-PR': {'IOPS': 42.98, 'MGAB': 3.09, 'NAB': 44.65, 'NEK': 71.13, 'Power': 14.63, 'SED': 69.86, 'Stock': 72.73, 'TODS': 91.67, 'UCR': 27.48, 'WSD': 29.95, 'YAHOO': 87.88},
    },
    'OmniAnomaly': {
        'Aff-F': {'IOPS': 80.32, 'MGAB': 67.35, 'NAB': 92.35, 'NEK': 86.30, 'Power': 78.16, 'SED': 61.26, 'Stock': 75.24, 'TODS': 50.73, 'UCR': 73.53, 'WSD': 78.02, 'YAHOO': 71.31},
        'F1-T': {'IOPS': 51.17, 'MGAB': 1.61, 'NAB': 40.09, 'NEK': 82.20, 'Power': 23.48, 'SED': 9.68, 'Stock': 36.22, 'TODS': 14.33, 'UCR': 8.47, 'WSD': 34.79, 'YAHOO': 24.16},
        'StdF1': {'IOPS': 47.05, 'MGAB': 1.44, 'NAB': 28.81, 'NEK': 74.03, 'Power': 23.50, 'SED': 0.43, 'Stock': 38.59, 'TODS': 12.65, 'UCR': 5.11, 'WSD': 29.57, 'YAHOO': 21.40},
        'VUS-PR': {'IOPS': 25.35, 'MGAB': 0.64, 'NAB': 27.17, 'NEK': 74.51, 'Power': 14.32, 'SED': 6.20, 'Stock': 91.29, 'TODS': 45.55, 'UCR': 2.40, 'WSD': 16.37, 'YAHOO': 29.26},
    },
}


def source_of(k):
    return k.split('/')[0]


def load():
    # the three detectors all use local fullcache (real inference results), no longer hard-coded official values.
    out = {m: {mk: {} for mk, _ in METRICS} for m in MODELS}
    for m in MODELS:
        c = json.load(open(ROOT / f'fullcache_{m}.json'))
        agg = {mk: defaultdict(list) for mk, _ in METRICS}
        for k, v in c.items():
            s = source_of(k)
            if s not in SRC11 or not isinstance(v, dict):
                continue
            for mk, _ in METRICS:
                if mk in v and isinstance(v[mk], (int, float)):
                    agg[mk][s].append(v[mk] * 100)
        for mk, _ in METRICS:
            for s, vs in agg[mk].items():
                out[m][mk][s] = float(np.mean(vs))
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


def build(data, picks):
    # entity -> metric -> source -> value (percentage)
    ent = {e: {mk: {} for mk, _ in METRICS} for e in ENTITIES}
    for m in MODELS:
        for mk, _ in METRICS:
            ent[m][mk] = dict(data[m][mk])
    # LLM routing: per source take llama3.1's selected model's value for that source and metric
    for mk, _ in METRICS:
        for s in SRC11:
            p = picks.get(s)
            if p in MODELS:
                ent['LLM routing'][mk][s] = data[p][mk].get(s)
    # Oracle: per source take the max over the models
    for mk, _ in METRICS:
        for s in SRC11:
            vals = [ent[m][mk][s] for m in MODELS if ent[m][mk].get(s) is not None]
            ent['Oracle'][mk][s] = max(vals) if vals else None
    return ent


def fmt(v, rank):
    if rank == 1:
        return r"\textcolor{red}{\textbf{" + f"{v:.1f}" + r"}}"
    if rank == 2:
        return r"\textcolor{blue}{" + f"{v:.1f}" + r"}"
    if rank == 3:
        return f"{v:.1f}" + r"${}^{*}$"
    return f"{v:.1f}"


def gen(ent):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    cols = "c" * (2 + len(SRC11) + 3)  # Metric + Method + 11 sources + Mean + 1st + 2nd
    lines.append(r"\begin{tabular}{@{}cc " + "c" * len(SRC11) + r" c cc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{\textbf{Metric}} & \multirow{2}{*}{\textbf{Method}} & "
        + r"\multicolumn{" + str(len(SRC11)) + r"}{c}{\textbf{Univariate Datasets}} & "
        + r"\multirow{2}{*}{\textbf{Mean}} & \multirow{2}{*}{\makecell{\textbf{Total}\\\textbf{1st}}} & \multirow{2}{*}{\makecell{\textbf{Total}\\\textbf{2nd}}} \\"
    )
    lines.append(r"\cmidrule(lr){3-" + str(2 + len(SRC11)) + "}")
    lines.append("& & " + " & ".join(r"\textbf{" + s + "}" for s in SRC11) + r" & & & \\")
    lines.append(r"\midrule")
    n = len(ENTITIES)
    for mi, (mk, disp) in enumerate(METRICS):
        # rank among 5 entities per source
        ranks = {}
        for s in SRC11:
            scored = [(e, ent[e][mk].get(s)) for e in ENTITIES if ent[e][mk].get(s) is not None]
            scored.sort(key=lambda x: -x[1])
            ranks[s] = {e: i + 1 for i, (e, _) in enumerate(scored)}
        for i, e in enumerate(ENTITIES):
            first = r"\multirow{" + str(n) + r"}{*}{" + disp + "}" if i == 0 else ""
            cells = " & ".join(fmt(ent[e][mk][s], ranks[s].get(e, 9)) if ent[e][mk].get(s) is not None else "---" for s in SRC11)
            vals = [ent[e][mk][s] for s in SRC11 if ent[e][mk].get(s) is not None]
            mean = f"{sum(vals)/len(vals):.1f}" if vals else "---"
            c1 = sum(1 for s in SRC11 if ranks[s].get(e) == 1)
            c2 = sum(1 for s in SRC11 if ranks[s].get(e) == 2)
            lines.append(first + " & " + DISP[e] + " & " + cells + f" & {mean} & {c1} & {c2} \\\\")
        lines.append(r"\bottomrule" if mi == len(METRICS) - 1 else r"\midrule")
    lines.append(r"\end{tabular}%")
    lines.append("}")
    lines.append(
        r"\caption{Three SOTA detectors and the LLM source routing on the eleven univariate TSB-AD "
        r"sources, Affiliation-F (\%) and companion metrics, all three detectors reproduced locally "
        r"(fullcache). LLM routing is llama3.1-8B selecting "
        r"one detector per source from label-free features and model profiles; Oracle is the per-source best "
        r"single detector. Best per source in bold red, second blue, third marked with an asterisk.}"
    )
    lines.append(r"\label{tab:sota}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    data = load()
    picks = load_llm_picks()
    ent = build(data, picks)
    tex = gen(ent)
    out = ROOT.parent / 'paper' / 'paper2_sota_table.tex'
    out.write_text(tex, encoding='utf-8')
    print(tex)
    print(f"\n[saved to {out}]")


if __name__ == '__main__':
    main()
