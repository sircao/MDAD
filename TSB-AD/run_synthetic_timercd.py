"""run_synthetic_timercd.py — Run Time-RCD on validation c01 (self-supervised, no train split needed)."""
import sys
import json
import time
import contextlib
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'stubs'))

import numpy as np
import pickle

from TSB_AD.model_wrapper import run_Unsupervise_AD
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from sklearn.metrics import roc_auc_score

VAL = ROOT.parent / "synthetic_benchmark_v1_r1" / "data" / "id" / "validation" / "c01_00015.pkl"
OUT = ROOT / "synthetic_c01_timercd.json"


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

    hp = dict(Optimal_Uni_algo_HP_dict["Time_RCD"])
    hp["checkpoint"] = str(ROOT.parent / "Time-RCD" / "best_model")

    res = []
    for i, s in enumerate(samples):
        data = s["time_series"]
        label = s["labels"]
        sw = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
        row = {"sample": s["sample_id"], "T": int(data.shape[0]),
               "morphs": [str(e["type"]) for e in s.get("events", [])]}
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                sc = run_Unsupervise_AD("Time_RCD", data, **hp)
            sc = np.asarray(sc).ravel()
            row["rcd_auc"] = auc(sc, label)
            row["rcd_vus"] = vus(sc, label, sw)
        except Exception as e:
            row["rcd_err"] = str(e)[:100]
        res.append(row)
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            OUT.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            print(f"  {i+1}/{len(samples)} {row.get('sample')} rcd_auc={row.get('rcd_auc',0):.3f}", flush=True)

    a = [r["rcd_auc"] for r in res if "rcd_auc" in r]
    print(f"\nTime-RCD: n={len(a)} AUC={np.mean(a):.3f}")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
