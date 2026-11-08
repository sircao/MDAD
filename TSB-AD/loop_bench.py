"""loop_bench.py — Residual-completion LOOP: best detector + complementary algorithms correcting detection errors per timestamp.

The starting point is the "best detector" (the strongest single model chosen by the two-level routing). A LOOP iteratively patches
the scores of complementary algorithms onto the timestamps where the best detector errs:

  1. Locate error timestamps:
       - oracle: use labels to split the miss (missed detection=FN) region and the fa (false alarm=FP) region;
       - label-free: use the label-free proxy of "best-detector score near the decision boundary + complementary detectors disagree strongly".
  2. Error morphology features: take the local morphology (classify) of the raw series on the error region, recording the morphology of miss/fa respectively.
  3. Pick the complement:
       - oracle: try "miss-region max / fa-region min / both" one by one, pick the one with the largest AUC gain;
       - label-free: pick the complement "most active in the uncertain region" (highest rank-score mean).
  4. Gated residual fusion (monotonically safe: miss only raises, fa only suppresses), until no gain.

Reports five lines (AUC-ROC per-file mean, a fast ranking proxy, same source as stacking_bench --metric auc):
  best single model / rank_mean / oracle (per-file max) / LOOP-oracle / LOOP-label-free.

Usage:
  python loop_bench.py --models IForest,LOF,Sub_PCA,FFT --max-iter 5 --metric auc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Anomaly_algorithm importable

from Anomaly_algorithm.morphology import classify  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent
ALL_UNI = ['NAB', 'WSD', 'Stock', 'MGAB', 'SED', 'IOPS', 'NEK', 'Power',
           'TODS', 'UCR', 'YAHOO']


def source_of(f: str) -> str:
    return f.split('_')[1]


def rank_norm(s: np.ndarray) -> np.ndarray:
    r = rankdata(s)
    return r / len(r)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=int)
    if len(np.unique(y)) < 2 or np.allclose(s, s[0]):
        return 0.5
    return float(roc_auc_score(y, s))


def best_f1_threshold(score: np.ndarray, label: np.ndarray) -> float:
    """Scan thresholds for the best-F1 decision boundary, used to split scores into miss/fa regions."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=int)
    pos = s[y == 1]
    if len(pos) == 0:
        return float(np.percentile(s, 50))
    tau = float(np.median(pos))
    best, best_tau = -1.0, tau
    for t in np.percentile(s, np.linspace(5, 95, 37)):
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        if f1 > best:
            best, best_tau = f1, float(t)
    return best_tau


def error_mask_oracle(fused: np.ndarray, label: np.ndarray):
    """Use labels to split error regions: miss = FN (true anomaly but score below the boundary), fa = FP (true normal but score above the boundary)."""
    tau = best_f1_threshold(fused, label)
    miss = (label == 1) & (fused < tau)
    fa = (label == 0) & (fused > tau)
    return miss, fa


def error_mask_labelfree(fused: np.ndarray, complements: np.ndarray):
    """Label-free proxy: best-detector score in the middle of rank space (near the decision boundary) and complements disagree strongly."""
    mid = (fused >= 0.33) & (fused <= 0.67)          # near the boundary
    disagree = complements.std(0) > np.median(complements.std(0))  # strong disagreement
    return mid & disagree


def apply_op(fused: np.ndarray, sc: np.ndarray, miss: np.ndarray, fa: np.ndarray,
             op: str) -> np.ndarray:
    """Monotonically-safe gated fusion: miss region only max (raise recall), fa region only min (lower precision)."""
    out = fused.copy()
    if op in ('miss_max', 'both'):
        m = miss & (sc > out)
        out[m] = sc[m]
    if op in ('fa_min', 'both'):
        f = fa & (sc < out)
        out[f] = sc[f]
    return out


def morph_of_region(x: np.ndarray, mask: np.ndarray, source: str) -> str:
    """Local morphology of the error region (error-morphology feature axis)."""
    xs = np.asarray(x).ravel()[mask]
    if len(xs) < 8:
        return 'too_small'
    try:
        return str(classify(xs, source=source)['morphology'])
    except Exception:
        return 'na'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True, help='comma-separated model names (corresponding to fullscores_*.npz)')
    ap.add_argument('--sources', default='', help='comma-separated sources; empty=all sources covered by the dump')
    ap.add_argument('--max-iter', type=int, default=5)
    ap.add_argument('--tol', type=float, default=1e-4)
    ap.add_argument('--metric', default='auc', choices=['auc'],
                    help='evaluation metric (currently uses auc as a fast proxy for smoke tests)')
    args = ap.parse_args()

    models = args.models.split(',')
    score_cache = {}
    for m in models:
        p = ROOT / f'fullscores_{m}.npz'
        score_cache[m] = dict(np.load(p)) if p.exists() else {}
        print(f'load {m}: {len(score_cache[m])} files', flush=True)

    common = set(score_cache[models[0]].keys())
    for m in models[1:]:
        common &= set(score_cache[m].keys())
    files = sorted(f for f in common
                   if (not args.sources or source_of(f) in args.sources.split(',')))
    print(f'after alignment {len(files)} files: {[source_of(f) for f in files]}', flush=True)
    if not files:
        print('no common files; run dump_scores.py first to fill in scores.')
        return

    # precompute per-file label + rank-normalized scores
    prep = {}
    for f in files:
        df = pd.read_csv(f'Datasets/TSB-AD-U/{f}').dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        sc = np.stack([rank_norm(score_cache[m][f]) for m in models]).astype(np.float32)
        prep[f] = {'x': data[:, 0].ravel(), 'label': label, 'scores': sc,
                   'src': source_of(f)}
        assert sc.shape[1] == len(label), f'{f} score length {sc.shape[1]} != label {len(label)}'

    def per_src_mean(file_vals):
        agg = {}
        for f, v in file_vals.items():
            agg.setdefault(source_of(f), []).append(v)
        return {s: float(np.mean(vs)) for s, vs in agg.items()}, \
               float(np.mean(list(file_vals.values())))

    # global best single model (highest per-file AUC mean), as the LOOP start
    model_mean = {m: float(np.mean([auc(prep[f]['scores'][i], prep[f]['label'])
                                    for f in files]))
                  for i, m in enumerate(models)}
    base = max(model_mean, key=model_mean.get)
    bi = models.index(base)
    print(f'\nbest single model = {base} (AUC {model_mean[base]:.4f})', flush=True)

    best_single, oracle, rankmean, loop_oracle, loop_lf = {}, {}, {}, {}, {}
    for f in files:
        label, x, src = prep[f]['label'], prep[f]['x'], prep[f]['src']
        S = prep[f]['scores']
        fused = S[bi].copy()  # start = best detector's rank scores
        base_auc = auc(fused, label)
        best_single[f] = base_auc
        oracle[f] = max(auc(S[mi], label) for mi in range(len(models)))
        rankmean[f] = auc(S.mean(0), label)

        # ---- LOOP-oracle (ceiling) ----
        fo = fused.copy()
        steps = []
        for _ in range(args.max_iter):
            miss, fa = error_mask_oracle(fo, label)
            if miss.sum() + fa.sum() == 0:
                break
            best_v, best_cand, best_desc = auc(fo, label), None, None
            for ci in range(len(models)):
                for op in ('miss_max', 'fa_min', 'both'):
                    cand = apply_op(fo, S[ci], miss, fa, op)
                    v = auc(cand, label)
                    if v > best_v + args.tol:
                        best_v, best_cand, best_desc = v, cand, (models[ci], op)
            if best_cand is None:
                break
            fo = best_cand
            steps.append(best_desc)
            if best_desc[1] == 'miss_max':
                morph = morph_of_region(x, miss, src)
            elif best_desc[1] == 'fa_min':
                morph = morph_of_region(x, fa, src)
            else:
                morph = morph_of_region(x, miss | fa, src)
            print(f'  [o]{src}/{f} step{len(steps)}: {best_desc[0]}.{best_desc[1]} '
                  f'(err morph={morph}) AUC {best_v:.4f}', flush=True)
        loop_oracle[f] = auc(fo, label)

        # ---- LOOP-label-free (deployable without labels) ----
        fl = fused.copy()
        lf_steps = []
        for _ in range(args.max_iter):
            unc = error_mask_labelfree(fl, S)
            if unc.sum() < 4:
                break
            # pick the complement most active in the uncertain region (highest rank-score mean), gated in via max-union
            act = [float(S[ci][unc].mean()) for ci in range(len(models))]
            ci = int(np.argmax(act))
            gain = np.maximum(fl[unc], S[ci][unc]).sum() - fl[unc].sum()
            if gain <= args.tol:
                break
            fl[unc] = np.maximum(fl[unc], S[ci][unc])
            lf_steps.append(models[ci])
            print(f'  [l]{src}/{f} step{len(lf_steps)}: patch {models[ci]} '
                  f'(unc={int(unc.sum())} points) AUC {auc(fl, label):.4f}', flush=True)
        loop_lf[f] = auc(fl, label)

    print('\n==== LOOP results (AUC-ROC per-file mean) ====')
    for name, d in [('best single model', best_single), ('rank_mean', rankmean),
                    ('oracle(per-file max)', oracle),
                    ('LOOP-oracle', loop_oracle), ('LOOP-label-free', loop_lf)]:
        per, mean = per_src_mean(d)
        print(f'{name:18s} overall {mean*100:6.2f}   '
              + '  '.join(f'{s}={v*100:.1f}' for s, v in sorted(per.items())))

    # label-free ceiling: the gap between the oracle ceiling and the label-free deployable
    _, mo = per_src_mean(loop_oracle)
    _, ml = per_src_mean(loop_lf)
    print(f'\nlabel-free ceiling (LOOP-oracle − LOOP-label-free) = {(mo - ml)*100:+.2f}pp')


if __name__ == '__main__':
    main()
