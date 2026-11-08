"""monitor_progress.py — Monitor CHARM dump progress in real time, compute AUC/VUS-PR, write progress.json + start an HTTP server.

Open http://localhost:8123/dashboard_progress.html in a browser to see live progress and scores.
The monitor process reads fullscores_CHARM_kNN.npz (written incrementally by the dump) while computing AUC (fast) + VUS-PR (slow) for each new file.
"""
import json
import os
import sys
import threading
import time
import contextlib
import io
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

ROOT = Path(__file__).resolve().parent
NPZ = ROOT / 'fullscores_CHARM_kNN.npz'
LOG = Path(os.environ.get('DUMP_LOG', str(ROOT / 'dump_charm_knn.log')))
OUT = ROOT / 'progress.json'
PORT = 8123


def source_of(f):
    return f.split('_')[1]


def auc(score, label):
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=int)
    if len(np.unique(y)) < 2 or np.allclose(s, s[0]):
        return 0.5
    return float(roc_auc_score(y, s))


def vus(score, label, sw):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return float(get_metrics(np.asarray(score).ravel(), label.astype(int),
                                 slidingWindow=sw)['VUS-PR'])


def serve_http():
    os.chdir(ROOT)
    handler = SimpleHTTPRequestHandler
    ThreadingHTTPServer(('127.0.0.1', PORT), handler).serve_forever()


def main():
    threading.Thread(target=serve_http, daemon=True).start()
    print(f'HTTP server: http://localhost:{PORT}/dashboard_progress.html', flush=True)

    all_syn = dict(np.load(ROOT / 'fullscores_Syntropic.npz'))  # total file list (73 files)
    total = len(all_syn)
    scored = {}   # f -> {auc, vus, src, len, secs}
    pending = []  # queue of files awaiting VUS-PR
    while True:
        # read currently-dumped CHARM scores
        try:
            charm = dict(np.load(NPZ))
        except Exception:
            charm = {}
        # newly-dumped files get AUC first (fast), VUS-PR is queued
        for f in sorted(charm.keys()):
            if f not in scored:
                df = pd.read_csv(f'Datasets/TSB-AD-U/{f}').dropna()
                label = df['Label'].astype(int).to_numpy()
                x = df.iloc[:, 0:-1].values.astype(float)
                sw = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
                scored[f] = {'auc': auc(charm[f], label), 'vus': None,
                             'src': source_of(f), 'len': int(len(charm[f]))}
                pending.append((f, label, sw))
        # compute VUS-PR for one file per loop (slow, catches up gradually)
        while pending:
            f, label, sw = pending.pop(0)
            try:
                scored[f]['vus'] = vus(charm[f], label, sw)
            except Exception:
                scored[f]['vus'] = None
        # read the dump log to get the current file + elapsed time
        current = None
        try:
            lines = LOG.read_text(encoding='utf-8', errors='ignore').strip().splitlines()
            for ln in reversed(lines):
                if ln.startswith('dumped') or ln.startswith('[') and 'dumped' in ln:
                    current = ln.strip()
                    break
        except Exception:
            pass
        # summarize
        done = [scored[f] for f in sorted(scored)]
        vus_vals = [d['vus'] for d in done if d['vus'] is not None]
        auc_vals = [d['auc'] for d in done]
        per_src = {}
        for d in done:
            s = per_src.setdefault(d['src'], {'n': 0, 'auc': [], 'vus': []})
            s['n'] += 1
            s['auc'].append(d['auc'])
            if d['vus'] is not None:
                s['vus'].append(d['vus'])
        payload = {
            'total': total, 'done': len(done),
            'current': current, 'updated': time.strftime('%H:%M:%S'),
            'mean_auc': float(np.mean(auc_vals)) if auc_vals else None,
            'mean_vus': float(np.mean(vus_vals)) if vus_vals else None,
            'per_src': {k: {'n': v['n'], 'auc': float(np.mean(v['auc'])),
                            'vus': float(np.mean(v['vus'])) if v['vus'] else None}
                        for k, v in sorted(per_src.items())},
            'files': [{'f': f, 'src': d['src'], 'len': d['len'],
                       'auc': d['auc'], 'vus': d['vus']}
                      for f, d in sorted(scored.items(), key=lambda kv: kv[0])][-20:],
        }
        OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        time.sleep(3)


if __name__ == '__main__':
    main()
