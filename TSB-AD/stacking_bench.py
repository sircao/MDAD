"""stacking_bench.py — Score-level conditional-weight fusion + leave-one-source cross-validation.

Load the fullscores_*.npz produced by dump_scores.py, rank-normalize per file,
fuse the scores with a conditional-weight model w = softmax(W·φ(x)) (φ = source embedding + local morphology features),
soft-AUC ranking loss + Adam training, leave-one-source CV.

Reports four lines (all per-file mean Affiliation-F):
  best single model / rank_mean equal-weight fusion / trained conditional fusion / oracle (per-file model max)
"""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make Anomaly_algorithm importable
from TSB_AD.evaluation.basic_metrics import basic_metricor
from TSB_AD.utils.slidingWindows import find_length_rank
from Anomaly_algorithm.morphology import extract_features, classify
from scipy.stats import rankdata

_grader = basic_metricor()
METRIC = 'auc'

ROOT = Path(__file__).resolve().parent
ALL_UNI = ['NAB', 'WSD', 'Stock', 'MGAB', 'SED', 'IOPS', 'NEK', 'Power', 'TODS', 'UCR', 'YAHOO']
MORPH_KEYS = ['kurtosis', 'skewness', 'acf1', 'slope', 'cusum', 'roll_std_mean',
              'roll_std_cv', 'flat_ratio', 'hfr', 'spec_peak', 'zero_cross_rate', 'max_over_std']
SIG_KEYS = ['res_run', 'res_ratio', 'dense_strength', 'dense_val_shift']


def source_of(f):
    return f.split('_')[1]


def load_data_and_label(f):
    df = pd.read_csv(f'Datasets/TSB-AD-U/{f}').dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    return data, label


def rank_norm(s):
    r = rankdata(s)
    return r / len(r)


def features_of(data, src):
    """source + local morphology features -> fixed-length vector."""
    x = data[:, 0].ravel() if data.ndim == 2 else np.asarray(data).ravel()
    try:
        f = extract_features(x, window=0)
        c = classify(x, window=0, source=src)
        sig = c.get('signals', {})
    except Exception:
        f, sig = {}, {}
    vec = [float(f.get(k, 0.0)) for k in MORPH_KEYS] + [float(sig.get(k, 0.0)) for k in SIG_KEYS]
    return np.array(vec, dtype=np.float32)


class Fusion(nn.Module):
    """w = softmax(W · [src_emb, feats]), per-file conditional weight."""

    def __init__(self, feat_dim, n_models, n_src):
        super().__init__()
        self.src_emb = nn.Embedding(n_src, 8)
        self.W = nn.Linear(8 + feat_dim, n_models)

    def forward(self, src_idx, feats):
        z = self.src_emb(src_idx)
        h = torch.cat([z, feats], dim=-1)
        return F.softmax(self.W(h), dim=-1)


def soft_auc_loss(fused, label, k=300):
    """Differentiable pairwise ranking proxy (soft-AUC)."""
    anom = fused[label > 0.5]
    norm = fused[label < 0.5]
    if len(anom) == 0 or len(norm) == 0:
        return torch.tensor(0.0, device=fused.device, requires_grad=True)
    a = anom[torch.randperm(len(anom), device=fused.device)[:k]] if len(anom) > k else anom
    n = norm[torch.randperm(len(norm), device=fused.device)[:k]] if len(norm) > k else norm
    diff = a.unsqueeze(1) - n.unsqueeze(0)  # (ka, kn)  want a > n
    return torch.log1p(torch.exp(-diff)).mean()


def aff(score, label, sw):
    if METRIC == 'aff':
        # official Affiliation-F (slow: 100 thresholds × affiliation event computation, expensive on long series)
        return _grader.metric_Affiliation(np.asarray(label, dtype=int), np.asarray(score))
    # fast ranking proxy AUC-ROC (same source as the soft-AUC loss), for smoke tests; use --metric aff on the server
    from sklearn.metrics import roc_auc_score
    s = np.asarray(score, dtype=float)
    if len(np.unique(np.asarray(label))) < 2 or np.allclose(s, s[0]):
        return 0.5
    return float(roc_auc_score(label, s))


def standardize(feat_mat, mu=None, sd=None):
    mat = np.asarray(feat_mat, dtype=np.float32)
    mat = np.clip(mat, -50, 50)
    if mu is None:
        mu, sd = mat.mean(0), mat.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
    return ((mat - mu) / sd).clip(-5, 5), mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True, help='comma-separated model names')
    ap.add_argument('--sources', default='', help='comma-separated sources; empty=all')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--metric', default='auc', choices=['auc', 'aff'],
                    help='evaluation metric: auc=fast ranking proxy (smoke), aff=official Affiliation-F (slow)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    global METRIC
    METRIC = args.metric
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    models = args.models.split(',')
    srcs = args.sources.split(',') if args.sources else ALL_UNI
    src2idx = {s: i for i, s in enumerate(ALL_UNI)}

    # load scores: score_cache[model][file] = 1d array
    score_cache = {}
    for m in models:
        p = ROOT / f'fullscores_{m}.npz'
        score_cache[m] = dict(np.load(p)) if p.exists() else {}
        print(f'load {m}: {len(score_cache[m])} files', flush=True)

    # only use files where all models have scores (aligned)
    common = set(score_cache[models[0]].keys())
    for m in models[1:]:
        common &= set(score_cache[m].keys())
    files = sorted(f for f in common if source_of(f) in srcs)
    print(f'after alignment {len(files)} files, sources: {sorted(set(source_of(f) for f in files))}', flush=True)
    if len(files) < 4:
        print('too few files for leave-one-source CV; dump more files first.')
        return

    # precompute per file: label / sw / features / rank-normalized scores (torch)
    prep = {}
    for f in files:
        data, label = load_data_and_label(f)
        sw = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
        feat = features_of(data, source_of(f))
        sc = np.stack([rank_norm(score_cache[m][f]) for m in models]).astype(np.float32)  # (n_models, T)
        prep[f] = {'label': label, 'sw': sw, 'feat': feat, 'scores': sc,
                   'src': src2idx[source_of(f)]}

    # aggregate each line's results by source
    def per_src_mean(file_vals):
        agg = {}
        for f, v in file_vals.items():
            agg.setdefault(source_of(f), []).append(v)
        return {s: float(np.mean(vs)) for s, vs in agg.items()}, float(np.mean(list(file_vals.values())))

    # baseline 1: best single model (chosen on training files, applied to test files)
    # baseline 2: oracle (per-file model max)
    # baseline 3: rank_mean equal-weight
    all_best_single, all_oracle, all_rankmean, all_fusion = {}, {}, {}, {}

    for held in sorted(set(source_of(f) for f in files)):
        tr = [f for f in files if source_of(f) != held]
        te = [f for f in files if source_of(f) == held]
        if not tr or not te:
            continue

        # pick the best single model (highest per-model Aff-F mean on training files)
        model_mean = {}
        for mi, m in enumerate(models):
            vs = [aff(prep[f]['scores'][mi], prep[f]['label'], prep[f]['sw']) for f in tr]
            model_mean[m] = float(np.mean(vs))
        best_m = max(model_mean, key=model_mean.get)
        bi = models.index(best_m)

        # train conditional fusion
        feat_mat = np.stack([prep[f]['feat'] for f in tr])
        feat_mat, mu, sd = standardize(feat_mat)
        feat_dict = {f: torch.tensor(feat_mat[i]) for i, f in enumerate(tr)}
        model = Fusion(len(MORPH_KEYS) + len(SIG_KEYS), len(models), len(ALL_UNI))
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            model.train(); opt.zero_grad()
            total = 0.0; cnt = 0
            for f in tr:
                w = model(torch.tensor([prep[f]['src']]), feat_dict[f].unsqueeze(0))[0]  # (n_models,)
                fused = (w.unsqueeze(-1) * torch.tensor(prep[f]['scores'])).sum(0)
                loss = soft_auc_loss(fused, torch.tensor(prep[f]['label']))
                loss.backward(); total += loss.item(); cnt += 1
            opt.step()
            if (ep + 1) % 20 == 0:
                print(f'  [held={held}] epoch {ep+1}: loss={total/cnt:.4f}', flush=True)

        # evaluate
        model.eval()
        for f in te:
            feats = torch.tensor(standardize([prep[f]['feat']], mu, sd)[0])  # (1, F)
            w = model(torch.tensor([prep[f]['src']]), feats)[0]
            fused = (w.detach().unsqueeze(-1) * torch.tensor(prep[f]['scores'])).sum(0).numpy()
            label, sw = prep[f]['label'], prep[f]['sw']
            all_fusion[f] = aff(fused, label, sw)
            all_best_single[f] = aff(prep[f]['scores'][bi], label, sw)
            orc = max(aff(prep[f]['scores'][mi], label, sw) for mi in range(len(models)))
            all_oracle[f] = orc
            rm = prep[f]['scores'].mean(0)
            all_rankmean[f] = aff(rm, label, sw)

    metric_name = 'Affiliation-F' if METRIC == 'aff' else 'AUC-ROC (ranking proxy)'
    print(f'\n==== leave-one-source CV results ({metric_name} per-file mean) ====')
    for name, d in [('best single model', all_best_single), ('rank_mean', all_rankmean),
                    ('trained conditional fusion', all_fusion), ('oracle (per-file max)', all_oracle)]:
        per, mean = per_src_mean(d)
        print(f'{name:16s} overall {mean*100:6.2f}   ' + '  '.join(f'{s}={v*100:.1f}' for s, v in sorted(per.items())))


if __name__ == '__main__':
    main()
