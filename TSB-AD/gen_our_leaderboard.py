"""gen_our_leaderboard.py — Format the 4 univariate models + LLM routing as a table in the official leaderboard format.

Per metric (Affiliation-F / F1-T / Standard-F1 / VUS-PR) output: model × 11 univariate sources + Total + 1st + 2nd.
LLM routing row = the score of the model llama3.1:8b picks per source, for that source and metric. The oracle row is for reference only and does not join the 1st/2nd ranking.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SRCS = ['IOPS', 'MGAB', 'NAB', 'NEK', 'Power', 'SED', 'Stock', 'TODS', 'UCR', 'WSD', 'YAHOO']
MODELS = ['Syntropic', 'Time_RCD', 'OmniAnomaly', 'StreamVAE']
METRICS = [('Aff-F', 'Affiliation-F'), ('F1-T', 'F1-T'), ('StdF1', 'Standard-F1'), ('VUS-PR', 'VUS-PR')]


def source_of(k):
    return k.split('/')[0]


def load():
    """model -> {metric -> {source -> mean}}"""
    out = {m: {mk: {} for mk, _ in METRICS} for m in MODELS}
    for m in MODELS:
        p = ROOT / f'fullcache_{m}.json'
        if not p.exists():
            continue
        c = json.load(open(p))
        agg = {mk: defaultdict(list) for mk, _ in METRICS}
        for k, v in c.items():
            s = source_of(k)
            if s not in SRCS or not isinstance(v, dict):
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


def row_scores(mk, data, llm_picks):
    """Per-row {source: score}: 4 models + LLM routing; oracle computed separately."""
    rows = {}
    for m in MODELS:
        rows[m] = data[m][mk]
    # LLM routing row
    rows['LLM routing'] = {}
    for s in SRCS:
        p = llm_picks.get(s)
        if p in data:
            rows['LLM routing'][s] = data[p][mk].get(s)
    return rows


def mean_available(d):
    vs = [v for v in d.values() if v is not None]
    return sum(vs) / len(vs) if vs else None


def main():
    data = load()
    llm = load_llm_picks()

    lines = []
    lines.append('Our Models (univariate 11 sources, official TSB-AD metric; StreamVAE/OmniAnomaly not fully run, missing values shown as ·)')
    lines.append('')

    header = 'Model        ' + ' '.join(f'{s:>7s}' for s in SRCS) + '   Total  1st 2nd'
    for mk, disp in METRICS:
        lines.append(disp)
        lines.append(header)
        rows = row_scores(mk, data, llm)  # dict row_name -> {source: score}
        oracle = {}
        for s in SRCS:
            vals = [rows[r][s] for r in rows if rows[r].get(s) is not None]
            oracle[s] = max(vals) if vals else None

        # 1st/2nd ranking (among the 4 models + LLM routing, excluding oracle)
        rank_cnt = {r: [0, 0] for r in rows}
        for s in SRCS:
            scored = [(r, rows[r][s]) for r in rows if rows[r].get(s) is not None]
            scored.sort(key=lambda x: -x[1])
            for i, (r, _) in enumerate(scored[:2]):
                rank_cnt[r][i] += 1

        for r in ['Syntropic', 'Time_RCD', 'OmniAnomaly', 'StreamVAE', 'LLM routing']:
            d = rows[r]
            cells = ' '.join(f'{d.get(s, float("nan")):7.2f}' if d.get(s) is not None else f'{"·":>7s}' for s in SRCS)
            tot = mean_available(d)
            tot_s = f'{tot:6.2f}' if tot is not None else '     ·'
            lines.append(f'{r:12s} {cells}  {tot_s}  {rank_cnt[r][0]:02d} {rank_cnt[r][1]:02d}')

        # oracle reference row
        cells = ' '.join(f'{oracle.get(s, float("nan")):7.2f}' if oracle.get(s) is not None else f'{"·":>7s}' for s in SRCS)
        tot = mean_available(oracle)
        lines.append(f'{"oracle(max)":12s} {cells}  {tot:6.2f}')
        lines.append('')

    text = '\n'.join(lines)
    print(text)
    out = ROOT / 'our_leaderboard.txt'
    out.write_text(text, encoding='utf-8')
    print(f'\n[saved to {out}]')


if __name__ == '__main__':
    main()
