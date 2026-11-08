# Generate the wide case table: 8 cases as columns, 4 metrics as rows, top1 red / top2 blue / top3 starred per metric.
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "Architecture_design")
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from gen_metric_tables import run_detector, affiliation_f, f1_t, standard_f1, vus_pr
from Architecture_design.dataset import split_standardize
from Architecture_design.architecture import MORPHOLOGY_WEIGHTS, TOP6

GT_KEY = {"iforest": "iforest", "knn": "knn", "two_sided_volatility": "two_sided_volatility",
          "two_sided_knn": "two_sided_knn", "rolling_volatility": "roll_std100", "tods_ma": "tods_ma"}
BS = chr(92)


def compute(f, morph):
    d = split_standardize(f)
    x, y = d["test_std"], d["y_test"]
    keys = [GT_KEY[d_] for d_ in TOP6]
    w = {GT_KEY[d_]: MORPHOLOGY_WEIGHTS[morph][d_] for d_ in TOP6}
    ranks = [rankdata(np.nan_to_num(np.asarray(run_detector(x, k), float), nan=0.0)) / len(y) for k in keys]
    wl = np.array([w[k] for k in keys]); wl = wl / wl.sum()
    s = sum(a * b for a, b in zip(wl, ranks))

    def auc(yy, ss):
        ss = np.asarray(ss, float).ravel()[:len(yy)]
        return float(roc_auc_score(yy, ss)) if 0 < (yy > 0).sum() < len(yy) else float("nan")

    roll = run_detector(x, "roll_std100")
    return {"morphology": morph, "auc_before": auc(y, roll), "auc_ours": auc(y, s),
            "Affiliation-F": affiliation_f(s, y), "F1_T": f1_t(s, y),
            "Standard-F1": standard_f1(s, y), "VUS-PR": vus_pr(s, y)}


CASES = {
    "NAB-001": ("TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv", "level_shift"),
    "NAB-008": ("TSB-AD-U/008_NAB_id_8_Synthetic_tr_1007_1st_2734.csv", "level_shift"),
    "TODS": ("TSB-AD-U/287_TODS_id_1_Synthetic_tr_500_1st_11.csv", "dense_cluster"),
    "YAHOO": ("TSB-AD-U/551_YAHOO_id_1_Synthetic_tr_500_1st_893.csv", "spike"),
    "Stock": ("TSB-AD-U/149_Stock_id_1_Finance_tr_500_1st_7.csv", "spike"),
    "UCR": ("TSB-AD-U/303_UCR_id_1_Medical_tr_2499_1st_8000.csv", "flatline"),
    "SMAP": ("TSB-AD-M/144_SMAP_id_1_Sensor_tr_2052_1st_5300.csv", "level_shift"),
    "IOPS": ("TSB-AD-U/260_IOPS_id_1_WebService_tr_5000_1st_11804.csv", "level_shift"),
}
data = {name: compute(f, m) for name, (f, m) in CASES.items()}
names = list(CASES.keys())
METRICS = ["Affiliation-F", "F1_T", "Standard-F1", "VUS-PR"]


def fmt(v, rank):
    if rank == 1:
        return BS + r"textcolor{red}{" + BS + r"textbf{" + f"{v:.3f}" + r"}}"
    if rank == 2:
        return BS + r"textcolor{blue}{" + f"{v:.3f}" + r"}"
    if rank == 3:
        return f"{v:.3f}" + "${}^{*}$"
    return f"{v:.3f}"


rows = []
rows.append(BS + r"begin{table*}[t]")
rows.append(BS + "centering")
rows.append(BS + "footnotesize")
rows.append(BS + r"setlength{" + BS + "tabcolsep}{3.5pt}")
rows.append(BS + r"resizebox{" + BS + r"textwidth}{!}{%")
rows.append(BS + r"begin{tabular}{@{}l c c c c c c c c@{}}")
rows.append(BS + "toprule")
rows.append("Metric & " + " & ".join(BS + "textbf{" + n + "}" for n in names) + " " + BS + BS)
rows.append(BS + "midrule")
# morphology row
rows.append("Morphology & " + " & ".join(data[n]["morphology"].replace("_", BS + "_") for n in names) + " " + BS + BS)
# AUC row
rows.append("AUC (before) & " + " & ".join(f"{data[n]['auc_before']:.3f}" for n in names) + " " + BS + BS)
rows.append("AUC (Ours) & " + " & ".join(BS + r"textbf{" + f"{data[n]['auc_ours']:.3f}" + "}" for n in names) + " " + BS + BS)
rows.append(BS + "midrule")
# 4 metric rows (each row highlights the ranking over the 8 cases)
for mt in METRICS:
    vals = sorted(names, key=lambda n: -data[n][mt])
    rank = {n: i + 1 for i, n in enumerate(vals)}
    cells = " & ".join(fmt(data[n][mt], rank[n]) for n in names)
    rows.append(mt + " & " + cells + " " + BS + BS)
rows.append(BS + "bottomrule")
rows.append(BS + r"end{tabular}%")
rows.append("}")
rows.append(BS + "caption{Case study of the eight representative series under the detection mechanism. "
           "AUC(before) is the single-direction rolling volatility baseline, AUC(Ours) the mechanism; the "
           "inverted cases (TODS 0.614 to 0.832, UCR 0.090 to 0.960) and the weak point case (Stock 0.598 to "
           "0.994) are recovered. For each metric, the best case is bold red, the second blue, and the third "
           "marked with an asterisk.}")
rows.append(BS + "label{tab:case}")
rows.append(BS + "end{table*}")
open(r"paper/paper2_case_table.tex", "w", encoding="utf-8").write("\n".join(rows))
print("Wide case table generated.")
for n in names:
    d = data[n]
    print(f"  {n:8s} {d['morphology']:14s} before={d['auc_before']:.3f} Ours={d['auc_ours']:.3f}")
