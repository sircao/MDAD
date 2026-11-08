"""Multivariate 5-source benchmark: model × file, per-source mean Affiliation-F, cached."""
import sys, io, contextlib, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'stubs'))
import numpy as np, pandas as pd, torch, random
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD, Unsupervise_AD_Pool, Semisupervise_AD_Pool
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict

seed = 2024
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
CHECKPOINTS = {'Time_RCD': 'Time-RCD/best_model'}
ROOT = Path(__file__).resolve().parent
SRCS = ['MSL','PSM','SMAP','SMD','SWaT']

def run_one(model, filename):
    df = pd.read_csv(filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    sw = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
    train_index = Path(filename).stem.split('_')[-3]
    data_train = data[:int(train_index), :]
    HP = dict(Optimal_Multi_algo_HP_dict[model])
    if model in CHECKPOINTS:
        HP['checkpoint'] = str(ROOT.parent / CHECKPOINTS[model])
    if model in Semisupervise_AD_Pool:
        out = run_Semisupervise_AD(model, data_train, data, **HP)
    else:
        out = run_Unsupervise_AD(model, data, **HP)
    m = get_metrics(np.asarray(out).ravel(), label, slidingWindow=sw)
    return {'Aff-F': float(m['Affiliation-F']), 'F1-T': float(m['F1-T']), 'StdF1': float(m['Standard-F1']), 'VUS-PR': float(m['VUS-PR'])}

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
    args = ap.parse_args()
    fl = pd.read_csv('Datasets/File_List/TSB-AD-M-Eva.csv')['file_name'].values
    files = [f for f in fl if f.split('_')[1] in SRCS]
    cache = ROOT / f'multicache_{args.model}.json'
    res = json.load(open(cache)) if cache.exists() else {}
    for f in files:
        s = f.split('_')[1]
        if f in res and isinstance(res[f], dict) and 'Aff-F' in res[f]: continue
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                v = run_one(args.model, f'Datasets/TSB-AD-M/{f}')
            res[f] = v
            print(f'[{args.model}] {s}/{f}: Aff-F={v["Aff-F"]*100:.2f}', flush=True)
        except Exception as e:
            res[f] = {'err': str(e)[:60]}
            print(f'[{args.model}] {s}/{f}: FAIL {str(e)[:50]}', flush=True)
        json.dump(res, open(cache, 'w'))
    print('\n=== per-source mean ===')
    for s in SRCS:
        vs = [v['Aff-F'] for f, v in res.items() if f.split('_')[1] == s and isinstance(v, dict)]
        if vs: print(f'{args.model} {s}: mean Aff-F = {np.mean(vs)*100:.2f} ({len(vs)} files)')
