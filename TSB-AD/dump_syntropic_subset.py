"""dump_syntropic_subset.py — Background dump of Syntropic per-point scores (per source the shortest N files ≤MAXLEN, resumable)."""
import sys, io, contextlib
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'stubs'))
from dump_scores import run_score

ROOT = Path(__file__).resolve().parent
MAXLEN = 50000
PER_SRC = 5

fl = pd.read_csv('Datasets/File_List/TSB-AD-U-Eva.csv')['file_name'].tolist()
bysrc = {}
for f in fl:
    src = f.split('_')[1]
    n = pd.read_csv(f'Datasets/TSB-AD-U/{f}', usecols=[0]).shape[0]
    if n <= MAXLEN:
        bysrc.setdefault(src, []).append((n, f))
files = [f for src in sorted(bysrc) for _, f in sorted(bysrc[src])[:PER_SRC]]
print(f'selected {len(files)} files / {len(bysrc)} sources', flush=True)

cache = ROOT / 'fullscores_Syntropic.npz'
res = dict(np.load(cache)) if cache.exists() else {}
for f in files:
    if f in res:
        print('skip', f, flush=True)
        continue
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sc = run_score('Syntropic', f'Datasets/TSB-AD-U/{f}')
        res[f] = np.asarray(sc, dtype=np.float32).ravel()
        np.savez_compressed(cache, **res)
        print('dumped', f, len(res[f]), flush=True)
    except Exception as e:
        print('FAIL', f, str(e)[:60], flush=True)
print('done', len(res), 'files')
