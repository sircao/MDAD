"""llm_source_select.py — Multiple local LLMs per-source model selection vs oracle (with live progress JSON).

For the 11 univariate sources, feed each source's **label-free features** (raw series statistics + morphology) and the profiles of the 4 candidate
models to several local Ollama models, letting each pick the strongest model per source; then score "picked correctly or not" using that model's
real Affiliation-F from fullcache. Labels are used only for evaluation; LLM decisions are label-free throughout.

After each source finishes, write the status into llm_select_progress.json for llm_select.html to read live.
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np, pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))  # Anomaly_algorithm / Architecture_design

from Anomaly_algorithm.morphology import extract_features, classify, SOURCE_PRIOR
from Architecture_design.client import OllamaClient

SRCS = ['NAB', 'WSD', 'Stock', 'MGAB', 'SED', 'IOPS', 'NEK', 'Power', 'TODS', 'UCR', 'YAHOO']
MODELS = ['Syntropic', 'Time_RCD', 'OmniAnomaly', 'StreamVAE']
PROGRESS = ROOT / 'llm_select_progress.json'

MODEL_PROFILES = {
    'Syntropic': 'Statistical synthesizer: weighted average of multiple robust-z normalized scores; strong on segment-level / level-drift / smooth collective anomalies',
    'Time_RCD': 'Zero-shot foundation model for time-series root-cause detection; general-purpose, no clear morphology preference',
    'OmniAnomaly': 'VAE reconstruction family: strong on point anomalies / dense clusters (isolated extremes, dense range deviation)',
    'StreamVAE': 'Dual-path (slow/fast dynamics) VAE, designed for vehicle telemetry; general-purpose reconstruction family',
}

MORPH_TO_MODEL = {
    'level_shift': 'Syntropic', 'trend': 'Syntropic', 'flatline': 'Syntropic',
    'persistent': 'Syntropic', 'seasonal': 'Syntropic',
    'spike': 'OmniAnomaly', 'dense_cluster': 'OmniAnomaly', 'variance_change': 'OmniAnomaly',
    'noise': 'Time_RCD',
}
SOURCE_PRIOR_EXTRA = {**SOURCE_PRIOR, 'WSD': 'spike'}

FEAT_KEYS = ['kurtosis', 'skewness', 'acf1', 'slope', 'cusum',
             'flat_ratio', 'hfr', 'spec_peak', 'roll_std_cv', 'max_over_std']

# fast models first, so results come out sooner; the heavy 30B / llama4 come last
FAST_FIRST = ['llama3.1:8b', 'qwen2.5-coder:7b', 'ibm/granite3.3:8b', 'IMSLAB:latest', 'qwen-gpu:latest']


def source_of(f):
    return f.split('_')[1]


def get_available_models():
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=5)
        return [m['name'] for m in r.json().get('models', [])]
    except Exception:
        return []


def order_models(available):
    ordered = [m for m in FAST_FIRST if m in available]
    ordered += [m for m in available if m not in FAST_FIRST]
    return ordered


def load_aff():
    out = {}
    for m in MODELS:
        p = ROOT / f'fullcache_{m}.json'
        if not p.exists():
            continue
        c = json.load(open(p))
        agg = defaultdict(list)
        for k, v in c.items():
            s = k.split('/')[0]
            if s in SRCS and isinstance(v, dict) and 'Aff-F' in v:
                agg[s].append(v['Aff-F'] * 100)
        for s, vs in agg.items():
            out.setdefault(s, {})[m] = float(np.mean(vs))
    return out


def source_features(src, n_sample=8):
    fl = pd.read_csv('Datasets/File_List/TSB-AD-U-Eva.csv')['file_name'].values
    files = [f for f in fl if source_of(f) == src][:n_sample]
    feats, morphs = [], []
    for f in files:
        df = pd.read_csv(f'Datasets/TSB-AD-U/{f}').dropna()
        x = df.iloc[:, 0].values.astype(float)
        try:
            feats.append([extract_features(x, window=0).get(k, 0.0) for k in FEAT_KEYS])
            morphs.append(classify(x, window=0, source=src)['morphology'])
        except Exception:
            pass
    if not feats:
        return None, None
    mean_f = dict(zip(FEAT_KEYS, np.mean(feats, axis=0).tolist()))
    mode_m = max(set(morphs), key=morphs.count)
    return mean_f, mode_m


def parse_choice(content):
    text = (content or '').replace('-', '_').lower()
    for m in MODELS:
        if m.lower() in text or m.lower().replace('_', '') in text.replace('_', ''):
            return m
    return None


def build_prompt(src, feats, morph):
    f = feats
    lines = [
        'You are an anomaly-detection tool-selection assistant. Given the raw statistical features of a data source, pick the detection model'
        ' most likely to be strongest for this source from the 4 candidate models. Output only one model name (Syntropic / Time_RCD / '
        'OmniAnomaly / StreamVAE); do not explain.',
        '',
        f'Data source: {src}',
        'Label-free features of the raw series (mean over multiple files within the source):',
        f'  kurtosis={f["kurtosis"]:.2f} skew={f["skewness"]:.2f} acf1={f["acf1"]:.2f} '
        f'slope={f["slope"]:.2f} cusum={f["cusum"]:.2f}',
        f'  flat_ratio={f["flat_ratio"]:.3f} high-freq-energy-ratio={f["hfr"]:.2f} '
        f'spectral-peak={f["spec_peak"]:.1f} roll_std_cv={f["roll_std_cv"]:.2f} max/σ={f["max_over_std"]:.2f}',
        f'Local-rule morphology: {morph} (level_shift=drift, spike=point, dense_cluster=dense, '
        f'variance_change=variance, flatline=stuck, trend=trend, noise=noise)',
        '',
        'Candidate model profiles:',
    ]
    for m in MODELS:
        lines.append(f'  - {m}: {MODEL_PROFILES[m]}')
    lines += ['', 'Combine source-name domain knowledge, statistical features and morphology to select the best model. Output only the model name.']
    return '\n'.join(lines)


def _mean(d):
    vs = [v for v in d.values() if v is not None]
    return round(sum(vs) / len(vs), 2) if vs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--llms', default='', help='comma-separated model names; empty=auto-detect all local models')
    ap.add_argument('--n-sample', type=int, default=8)
    args = ap.parse_args()

    available = get_available_models()
    llms = args.llms.split(',') if args.llms else order_models(available)
    print(f'available local models: {available}', flush=True)
    print(f'this run (fast first): {llms}\n', flush=True)

    aff = load_aff()
    feat_cache = {s: source_features(s, args.n_sample) for s in SRCS if s in aff}

    single, prior, oracle, morph_of = {}, {}, {}, {}
    # best single model = the fixed model with the highest cross-source mean (take its per-source score), not the per-source max
    model_mean = {}
    for m in MODELS:
        vs = [aff[s][m] for s in aff if m in aff[s]]
        model_mean[m] = sum(vs) / len(vs) if vs else -1.0
    best_model = max(model_mean, key=model_mean.get)
    for s in SRCS:
        if s not in aff:
            continue
        single[s] = round(aff[s][best_model], 2) if best_model in aff[s] else None
        oracle[s] = round(max(aff[s].values()), 2)
        pm = MORPH_TO_MODEL.get(SOURCE_PRIOR_EXTRA.get(s, ''), 'Syntropic')
        prior[s] = round(aff[s].get(pm, 0.0), 2) if aff[s].get(pm) is not None else None
        morph_of[s] = feat_cache[s][1] if feat_cache.get(s) else ''

    state = {
        'updated': time.strftime('%H:%M:%S'),
        'llms': llms,
        'baselines': {'single': single, 'prior': prior, 'oracle': oracle,
                      'single_model': best_model,
                      'single_mean': _mean(single), 'prior_mean': _mean(prior), 'oracle_mean': _mean(oracle)},
        'aff': {s: {m: round(v, 2) for m, v in aff[s].items()} for s in aff},
        'morph': morph_of,
        'runs': [{'name': m, 'status': 'pending', 'picks': {}, 'score': None, 'n': 0} for m in llms],
    }

    def save():
        state['updated'] = time.strftime('%H:%M:%S')
        with open(PROGRESS, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False)
    save()

    results = {}
    for ri, llm in enumerate(llms):
        client = OllamaClient(model=llm, timeout=180)
        run = state['runs'][ri]
        run['status'] = 'running'
        save()
        pick, score = {}, {}
        print(f'--- {llm} ---', flush=True)
        for s in SRCS:
            if s not in aff:
                continue
            feats, morph = feat_cache[s]
            if feats is None:
                continue
            try:
                r = client.chat([{'role': 'user', 'content': build_prompt(s, feats, morph)}])
                p = parse_choice(r['content'])
            except Exception as e:
                print(f'  [{s}] call failed: {str(e)[:60]}', flush=True)
                p = None
            pick[s] = p
            score[s] = aff[s].get(p) if p in aff.get(s, {}) else None
            run['picks'][s] = p
            run['n'] = sum(1 for v in score.values() if v is not None)
            run['score'] = _mean(score)
            save()
            ok = '✓' if (p and aff[s].get(p) == oracle[s]) else '·'
            sc = f'{score[s]:6.2f}' if score[s] is not None else '  —  '
            print(f'  {s:7s} {str(morph):14s} -> {str(p):12s} {sc} (oracle {oracle[s]:6.2f}) {ok}', flush=True)
        run['status'] = 'done' if run['n'] > 0 else 'failed'
        results[llm] = (_mean(score), run['n'], pick)
        save()
        print(f'  => {llm}: {_mean(score)} ({run["n"]} sources)\n', flush=True)

    print('=' * 60, flush=True)
    print('==== summary (Affiliation-F per-source mean) ====', flush=True)
    for name, d in [('best single model', single), ('source-prior routing', prior)]:
        print(f'  {name:16s} {_mean(d):6.2f}  ({len(d)} sources)', flush=True)
    for llm in llms:
        m, n, _ = results.get(llm, (None, 0, {}))
        print(f'  LLM {llm:16s} {str(m):>6s}  ({n} sources)', flush=True)
    print(f'  {"oracle(per-source max)":16s} {_mean(oracle):6.2f}  ({len(oracle)} sources)', flush=True)


if __name__ == '__main__':
    main()
