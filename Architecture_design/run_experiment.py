"""Direction-agnostic detector experiment: 12 detectors x 14 datasets, computing
ROC-AUC / AUC-PR / best-F1 / four metrics, and producing the direction-recovery
table, ensemble table, oracle and inversion counts. Results saved to
runs/direction_agnostic_experiment.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Architecture_design"))

from Architecture_design.metrics import evaluate
from gen_metric_tables import (ALL, DETECTORS, FILES, MUL, UNI,
                               affiliation_f, f1_t, run_detector, standard_f1, vus_pr)

STOCK = "TSB-AD-U/149_Stock_id_1_Finance_tr_500_1st_7.csv"

def load(src, f):
    d = (__import__("Architecture_design.dataset", fromlist=["split_standardize_multi"]).split_standardize_multi(f)
         if src in MUL else
         __import__("Architecture_design.dataset", fromlist=["split_standardize"]).split_standardize(f))
    return d["test_std"], d["y_test"]

def roc(s, y):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s)) if 0 < (y > 0).sum() < len(y) else float('nan')
    except Exception:
        return float('nan')

# 1) 12 detectors x 14 datasets, all metrics
scores = {}
for src, f in FILES.items():
    x, y = load(src, f)
    scores[src] = {"y": y}
    for name in DETECTORS:
        scores[src][name] = run_detector(x, name)[: len(y)]

mean = {det: {} for det in DETECTORS}
per = {}
for det in DETECTORS:
    per[det] = {}
    aucs, aprs, f1s, affs, f1ts, sf1s, vus = [], [], [], [], [], [], []
    for src in ALL:
        s, y = scores[src][det], scores[src]["y"]
        ev = evaluate(s, y)
        aucs.append(ev["auc_roc"]); aprs.append(ev["auc_pr"]); f1s.append(ev["best_f1"])
        affs.append(affiliation_f(s, y)); f1ts.append(f1_t(s, y)); sf1s.append(standard_f1(s, y)); vus.append(vus_pr(s, y))
    mean[det] = {
        "roc_auc": round(float(np.nanmean(aucs)), 3),
        "auc_pr": round(float(np.nanmean(aprs)), 3),
        "best_f1": round(float(np.nanmean(f1s)), 3),
        "affiliation_f": round(float(np.mean(affs)), 3),
        "f1_t": round(float(np.mean(f1ts)), 3),
        "standard_f1": round(float(np.mean(sf1s)), 3),
        "vus_pr": round(float(np.mean(vus)), 3),
    }
    per[det]["auc_by_src"] = {src: round(a, 3) for src, a in zip(ALL, aucs)}

# 2) inversion count + oracle
invert = {det: sum(1 for v in per[det]["auc_by_src"].values() if v < 0.5) for det in DETECTORS}
oracle_mean = round(float(np.mean([max(per[det]["auc_by_src"][src] for det in DETECTORS) for src in ALL])), 3)
dir_oracle_mean = round(float(np.mean([max(max(per[det]["auc_by_src"][src], 1 - per[det]["auc_by_src"][src]) for det in DETECTORS) for src in ALL])), 3)

# 3) direction-recovery table (4 cases)
def auc_of(src, det, f):
    x, y = load(src, f)
    return roc(run_detector(x, det)[: len(y)], y)
direction = {
    "UCR": {"single": per["roll_std100"]["auc_by_src"]["UCR"], "two_sided_vol": per["two_sided_volatility"]["auc_by_src"]["UCR"], "two_sided_knn": per["two_sided_knn"]["auc_by_src"]["UCR"]},
    "TODS": {"knn": per["knn"]["auc_by_src"]["TODS"], "two_sided_knn": per["two_sided_knn"]["auc_by_src"]["TODS"]},
    "YAHOO": {"single": per["roll_std100"]["auc_by_src"]["YAHOO"], "two_sided_vol": per["two_sided_volatility"]["auc_by_src"]["YAHOO"]},
    "NAB": {"single": per["roll_std100"]["auc_by_src"]["NAB"], "two_sided_vol": per["two_sided_volatility"]["auc_by_src"]["NAB"]},
}

# 4) ensemble table (5 series: NAB-001, NAB-008, Stock, TODS, UCR)
ens_series = {
    "NAB-001": ("NAB", FILES["NAB"]),
    "NAB-008": ("NAB-008", "TSB-AD-U/008_NAB_id_8_Synthetic_tr_1007_1st_2734.csv"),
    "Stock": ("Stock", STOCK),
    "TODS": ("TODS", FILES["TODS"]),
    "UCR": ("UCR", FILES["UCR"]),
}
from scipy.stats import rankdata
ensemble = {}
for key, (srcname, f) in ens_series.items():
    x, y = load(srcname, f)
    ranks = []
    for det in DETECTORS:
        s = run_detector(x, det)[: len(y)]
        s = np.nan_to_num(np.asarray(s, float), nan=0.0)
        ranks.append(rankdata(s) / len(s))
    rmax = np.max(ranks, axis=0)
    rmed = np.mean(ranks, axis=0)
    oracle = max(roc(run_detector(x, det)[: len(y)], y) for det in DETECTORS)
    ensemble[key] = {"rank_max": round(roc(rmax, y), 3), "rank_median": round(roc(rmed, y), 3), "oracle": round(oracle, 3)}

out = {"mean": mean, "invert_count": invert, "oracle_mean": oracle_mean, "dir_oracle_mean": dir_oracle_mean,
       "direction": direction, "ensemble": ensemble, "n_datasets": len(ALL)}
outpath = ROOT / "Architecture_design/runs/direction_agnostic_experiment.json"
outpath.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("=== mean (12 detectors x 14 datasets) ===")
print(json.dumps(mean, ensure_ascii=False, indent=1))
print("=== invert_count (AUC<0.5) ===", json.dumps(invert, ensure_ascii=False))
print("oracle_mean=", oracle_mean, " dir_oracle_mean(max(AUC,1-AUC))=", dir_oracle_mean)
print("=== direction ===", json.dumps(direction, ensure_ascii=False, indent=1))
print("=== ensemble ===", json.dumps(ensemble, ensure_ascii=False, indent=1))
print("SAVED ->", outpath)
