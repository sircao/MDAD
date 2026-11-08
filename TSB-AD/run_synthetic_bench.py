"""run_synthetic_bench.py — Run Syntropic + CHARM_kNN on synthetic_benchmark validation c01.

Compute AUC + VUS-PR per sample, store JSON. train split = first-10% heuristic (note: some samples have anomalies at the start, which pollutes train).
Usage: python run_synthetic_bench.py --limit 54
"""
import os
import sys
import json
import time
import contextlib
import io
from pathlib import Path

os.environ['CHARM_BASE_URL'] = 'http://ab778f946c58843afa52a72d5af0657a-1381817648.us-west-2.elb.amazonaws.com:8080'
os.environ['CHARM_API_KEY'] = 'token'

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'stubs'))
sys.path.insert(0, str(ROOT / 'benchmark_exp'))

import numpy as np
import pickle

from TSB_AD.model_wrapper import run_Semisupervise_AD
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from Run_CHARM import run_CHARM_kNN
from sklearn.metrics import roc_auc_score

VAL = ROOT.parent / "synthetic_benchmark_v1_r1" / "data" / "id" / "validation" / "c01_00015.pkl"
OUT = ROOT / "synthetic_c01_scores.json"


def auc(s, l):
    if len(np.unique(l)) < 2 or np.allclose(s, s[0]):
        return 0.5
    return float(roc_auc_score(l, s))


def vus(s, l, sw):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return float(get_metrics(np.asarray(s).ravel(), l.astype(int), slidingWindow=sw)["VUS-PR"])


def main():
    with open(VAL, "rb") as f:
        samples = pickle.load(f)
    print(f"validation c01 samples {len(samples)}", flush=True)

    res = []
    for i, s in enumerate(samples):
        data = s["time_series"]  # (T, 1)
        label = s["labels"]      # (T,)
        T = data.shape[0]
        tr = int(0.1 * T)
        data_train = data[:tr]
        sw = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
        # morphology (event type)
        morphs = [str(e["type"]) for e in s.get("events", [])]

        row = {"sample": s["sample_id"], "T": T, "morphs": morphs}

        # Syntropic (semi-supervised)
        try:
            hp = dict(Optimal_Uni_algo_HP_dict["Syntropic"])
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                sc = run_Semisupervise_AD("Syntropic", data_train, data, **hp)
            sc = np.asarray(sc).ravel()
            row["syn_auc"] = auc(sc, label)
            row["syn_vus"] = vus(sc, label, sw)
        except Exception as e:
            row["syn_err"] = str(e)[:80]

        # CHARM_kNN (semi-supervised)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                cc = run_CHARM_kNN(data_train, data)
            cc = np.asarray(cc).ravel()
            row["charm_auc"] = auc(cc, label)
            row["charm_vus"] = vus(cc, label, sw)
        except Exception as e:
            row["charm_err"] = str(e)[:80]

        res.append(row)
        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            OUT.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            print(f"  ran {i+1}/{len(samples)}: {row.get('sample')} "
                  f"syn_auc={row.get('syn_auc', 0):.3f} charm_auc={row.get('charm_auc', 0):.3f}", flush=True)

    # summarize
    syn_a = [r["syn_auc"] for r in res if "syn_auc" in r]
    charm_a = [r["charm_auc"] for r in res if "charm_auc" in r]
    print(f"\nSyntropic: n={len(syn_a)} AUC={np.mean(syn_a):.3f}")
    print(f"CHARM_kNN: n={len(charm_a)} AUC={np.mean(charm_a):.3f}")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
