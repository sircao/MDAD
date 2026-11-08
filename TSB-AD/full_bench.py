"""Full benchmark: model × all 16-source files, per-source mean Affiliation-F, cached scores/metrics."""
import sys, os, io, contextlib, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'stubs'))
import numpy as np, pandas as pd, torch, random
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD, Unsupervise_AD_Pool, Semisupervise_AD_Pool
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict

seed = 2024
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
CHECKPOINTS = {'Time_RCD': 'Time-RCD/best_model'}
ROOT = Path(__file__).resolve().parent

def run_one(model_name, filename):
    df = pd.read_csv(filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    sw = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
    train_index = Path(filename).stem.split('_')[-3]
    data_train = data[:int(train_index), :]
    multi = data.shape[1] > 1
    HP = Optimal_Multi_algo_HP_dict[model_name] if multi else Optimal_Uni_algo_HP_dict[model_name]
    HP = dict(HP)
    if model_name in CHECKPOINTS:
        HP['checkpoint'] = str(ROOT.parent / CHECKPOINTS[model_name])
    if model_name in Semisupervise_AD_Pool:
        out = run_Semisupervise_AD(model_name, data_train, data, **HP)
    else:
        out = run_Unsupervise_AD(model_name, data, **HP)
    m = get_metrics(np.asarray(out).ravel(), label, slidingWindow=sw)
    return {'Aff-F': float(m['Affiliation-F']), 'F1-T': float(m['F1-T']), 'VUS-PR': float(m['VUS-PR']), 'AUC': float(m['AUC-ROC']), 'StdF1': float(m['Standard-F1']), 'sw': int(sw)}

def source_of(fname):
    return fname.split('_')[1]

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--sources', default='')  # comma-separated; empty=all 11 univariate sources
    ap.add_argument('--limit', type=int, default=0)  # max files per source (0=all)
    ap.add_argument('--refresh', action='store_true')  # recompute cached files (used to refresh after the F1-T metric changed)
    args = ap.parse_args()

    # the 11 univariate sources (user table)
    ALL_UNI = ['NAB','WSD','Stock','MGAB','SED','IOPS','NEK','Power','TODS','UCR','YAHOO']
    srcs = args.sources.split(',') if args.sources else ALL_UNI
    fl = pd.read_csv('Datasets/File_List/TSB-AD-U-Eva.csv')['file_name'].values
    files_by_src = {s: [] for s in srcs}
    for f in fl:
        s = source_of(f)
        if s in files_by_src:
            files_by_src[s].append(f)

    cache = ROOT / f'fullcache_{args.model}.json'
    res = json.load(open(cache)) if cache.exists() else {}

    for s in srcs:
        fs = files_by_src[s][:args.limit] if args.limit else files_by_src[s]
        vals = []
        for f in fs:
            key = f'{s}/{f}'
            if key in res and not args.refresh and isinstance(res[key], dict) and 'Aff-F' in res[key] and 'F1-T' in res[key]:
                vals.append(res[key]['Aff-F']); continue
            fp = f'Datasets/TSB-AD-U/{f}'
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    r = run_one(args.model, fp)
                res[key] = r; vals.append(r['Aff-F'])
                print(f'[{args.model}] {s}/{f}: Aff-F={r["Aff-F"]*100:.2f}', flush=True)
            except Exception as e:
                res[key] = {'err': str(e)[:60]}
                print(f'[{args.model}] {s}/{f}: FAIL {str(e)[:50]}', flush=True)
            json.dump(res, open(cache, 'w'))
        ok = [v for v in vals if isinstance(v, float)]
        print(f'>> {args.model} {s}: mean Aff-F = {np.mean(ok)*100:.2f} ({len(ok)}/{len(fs)} files)', flush=True)
