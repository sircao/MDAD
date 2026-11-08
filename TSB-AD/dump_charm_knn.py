"""dump_charm_knn.py — Use Run_CHARM.py's run_CHARM_kNN (multi-scale + OOM retry) to dump CHARM_kNN per-point scores.

Same implementation as the previous CHARM_uni_mergedTable_VUS-PR.csv (run_CHARM_kNN), with request_with_retry
(halve the batch and retry on GPU OOM). Resume: already-dumped files are skipped. Base = the routing-chosen CHARM_kNN.
"""
import os
import sys
import contextlib
import io
import time
from pathlib import Path

os.environ['CHARM_BASE_URL'] = 'http://ab778f946c58843afa52a72d5af0657a-1381817648.us-west-2.elb.amazonaws.com:8080'
os.environ['CHARM_API_KEY'] = 'token'

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmark_exp'))

import numpy as np
import pandas as pd

from Run_CHARM import run_CHARM_kNN  # noqa: E402

CACHE = ROOT / 'fullscores_CHARM_kNN.npz'


def main():
    syn = dict(np.load(ROOT / 'fullscores_Syntropic.npz'))
    files = sorted(syn.keys())  # same 73 files as Syntropic
    res = dict(np.load(CACHE)) if CACHE.exists() else {}
    print(f'{len(files)} files total, {len(res)} dumped', flush=True)
    for i, f in enumerate(files):
        if f in res:
            print(f'[{i+1}/{len(files)}] skip {f}', flush=True)
            continue
        df = pd.read_csv(f'Datasets/TSB-AD-U/{f}').dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        train_index = int(f.split('_')[-3])
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                sc = run_CHARM_kNN(data[:train_index], data)
            res[f] = np.asarray(sc, dtype=np.float32).ravel()
            np.savez_compressed(CACHE, **res)
            print(f'[{i+1}/{len(files)}] dumped {f} len={len(res[f])} '
                  f'took {time.time()-t0:.0f}s', flush=True)
        except Exception as e:
            print(f'[{i+1}/{len(files)}] FAIL {f} {str(e)[:80]}', flush=True)
    print(f'done {len(res)} files', flush=True)


if __name__ == '__main__':
    main()
