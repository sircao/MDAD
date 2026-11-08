"""dump_scores.py — Dump raw per-point anomaly scores file by file (the prerequisite for score-level fusion).

Difference from full_bench.py: not just metrics, but also each model's **raw per-point scores** on every file.
Stored as a per-model npz (key=file name, value=float32 score array), resumable.
"""
import sys, io, contextlib, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, random

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'stubs'))
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD, Unsupervise_AD_Pool, Semisupervise_AD_Pool
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict

seed = 2024
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {'Time_RCD': 'Time-RCD/best_model'}

ALL_UNI = ['NAB', 'WSD', 'Stock', 'MGAB', 'SED', 'IOPS', 'NEK', 'Power', 'TODS', 'UCR', 'YAHOO']


def source_of(fname):
    return fname.split('_')[1]


def run_score(model_name, filename):
    df = pd.read_csv(filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    multi = data.shape[1] > 1
    train_index = Path(filename).stem.split('_')[-3]
    data_train = data[:int(train_index), :]
    HP = dict(Optimal_Multi_algo_HP_dict[model_name] if multi else Optimal_Uni_algo_HP_dict[model_name])
    if model_name in CHECKPOINTS:
        HP['checkpoint'] = str(ROOT.parent / CHECKPOINTS[model_name])
    if model_name in Semisupervise_AD_Pool:
        out = run_Semisupervise_AD(model_name, data_train, data, **HP)
    else:
        out = run_Unsupervise_AD(model_name, data, **HP)
    if isinstance(out, str):  # model_wrapper returns str on error
        raise RuntimeError(out)
    return np.asarray(out, dtype=np.float32).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True, help='comma-separated model names')
    ap.add_argument('--sources', default='', help='comma-separated sources; empty=all 11 univariate sources')
    ap.add_argument('--limit', type=int, default=0, help='max files per source (0=all)')
    args = ap.parse_args()

    models = args.models.split(',')
    srcs = args.sources.split(',') if args.sources else ALL_UNI
    fl = pd.read_csv('Datasets/File_List/TSB-AD-U-Eva.csv')['file_name'].values
    files_by_src = {s: [] for s in srcs}
    for f in fl:
        s = source_of(f)
        if s in files_by_src:
            files_by_src[s].append(f)

    for m in models:
        cache = ROOT / f'fullscores_{m}.npz'
        res = dict(np.load(cache)) if cache.exists() else {}
        for s in srcs:
            fs = files_by_src[s][:args.limit] if args.limit else files_by_src[s]
            for f in fs:
                if f in res:
                    print(f'[{m}] skip {s}/{f}', flush=True)
                    continue
                fp = f'Datasets/TSB-AD-U/{f}'
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        sc = run_score(m, fp)
                    res[f] = sc
                    print(f'[{m}] {s}/{f}: score len={len(sc)}', flush=True)
                except Exception as e:
                    print(f'[{m}] {s}/{f}: FAIL {str(e)[:70]}', flush=True)
                np.savez_compressed(cache, **res)
        np.savez_compressed(cache, **res)
        print(f'>> {m}: {len(res)} files dumped -> {cache.name}')


if __name__ == '__main__':
    main()
