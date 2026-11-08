"""Generate the cross-dataset comparison table from cross_dataset.json (Metric x Model x 14 datasets + Total 1st/2nd)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

d = json.load(open(ROOT / "Architecture_design/runs/cross_dataset.json", encoding="utf-8"))
res, tot1, tot2 = d["res"], d["tot1"], d["tot2"]
UNI = ["IOPS", "MGAB", "NAB", "NEK", "Power", "SED", "TODS", "UCR", "YAHOO"]
MUL = ["MSL", "PSM", "SMAP", "SMD", "SWaT"]
ALL = UNI + MUL
METRICS = ["auc_roc", "auc_pr", "best_f1"]
MLAB = {"auc_roc": "AUC-ROC", "auc_pr": "AUC-PR", "best_f1": "F1"}
MODELS = ["iforest", "knn", "lof", "tods_ma", "tods_z",
          "roll_std50", "roll_std100", "roll_range100", "flatline"]


def fmt(v) -> str:
    return f"{v:.3f}"


rows = []
for m in METRICS:
    for name in MODELS:
        cells = [MLAB[m], name] + [fmt(res[m][name][s]) for s in ALL] + [
            str(tot1[m][name]), str(tot2[m][name])
        ]
        rows.append(cells)

NL = "\n"
L = []
L.append("\\begin{table*}[t]")
L.append("\\centering")
L.append("\\footnotesize")
L.append("\\setlength{\\tabcolsep}{4pt}")
L.append("\\resizebox{\\textwidth}{!}{%")
L.append("\\begin{tabular}{@{}cc ccccccccc ccccc cc@{}}")
L.append("\\toprule")
L.append(
    "\\multirow{2}{*}{\\textbf{Metric}} & \\multirow{2}{*}{\\textbf{Model}} & "
    "\\multicolumn{9}{c}{\\textbf{Univariate Datasets}} & "
    "\\multicolumn{5}{c}{\\textbf{Multivariate Datasets}} & "
    "\\multirow{2}{*}{\\makecell{\\textbf{Total}\\\\\\textbf{1st}}} & "
    "\\multirow{2}{*}{\\makecell{\\textbf{Total}\\\\\\textbf{2nd}}} \\\\"
)
L.append("\\cmidrule(lr){3-11} \\cmidrule(lr){12-16}")
L.append("& & " + " & ".join(f"\\textbf{{{s}}}" for s in UNI) + " & " +
         " & ".join(f"\\textbf{{{s}}}" for s in MUL) + " & & \\\\")
L.append("\\midrule")
L.append("\\multicolumn{18}{c}{\\textbf{Unsupervised Detectors}} \\\\")
L.append("\\midrule")
for r in rows:
    L.append(" & ".join(r) + " \\\\")
L.append("\\bottomrule")
L.append("\\end{tabular}%")
L.append("}")
L.append(
    "\\caption{Performance comparison of unsupervised detectors across nine univariate and "
    "five multivariate TSB-AD datasets (one representative series per source). ROC-AUC, "
    "average precision, and best-F1 are threshold-free or threshold-swept; Total 1st/2nd "
    "count the datasets on which each detector ranks first/second for that metric.}"
)
L.append("\\label{tab:cross_dataset}")
L.append("\\end{table*}")

out = NL.join(L) + NL
(ROOT / "paper/cross_dataset_table.tex").write_text(out, encoding="utf-8")
print(out)
