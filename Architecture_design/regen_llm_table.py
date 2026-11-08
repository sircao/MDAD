"""Regenerate the LLM cross-dataset table from llm_cross_dataset.json (Total 1st/2nd using the corrected values)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
d = json.load(open(ROOT / "Architecture_design/runs/llm_cross_dataset.json", encoding="utf-8"))
res, tot1, tot2 = d["res"], d["tot1"], d["tot2"]
UNI = ["IOPS", "MGAB", "NAB", "NEK", "Power", "SED", "TODS", "UCR", "YAHOO"]
MUL = ["MSL", "PSM", "SMAP", "SMD", "SWaT"]
ALL = UNI + MUL
MODELS = ["llama3.1:8b", "qwen2.5-coder:7b", "qwen3-coder-30b-a3b:latest", "ibm/granite3.3:8b"]
METRICS = ["auc_roc", "auc_pr", "best_f1"]
MLAB = {"auc_roc": "AUC-ROC", "auc_pr": "AUC-PR", "best_f1": "F1"}

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
L.append("\\multicolumn{18}{c}{\\textbf{LLM Tool Selectors}} \\\\")
L.append("\\midrule")
for m in METRICS:
    for model in MODELS:
        cells = [MLAB[m], model] + [f"{res[m][model][s]:.3f}" for s in ALL] + \
                [str(tot1[m][model]), str(tot2[m][model])]
        L.append(" & ".join(cells) + " \\\\")
L.append("\\bottomrule")
L.append("\\end{tabular}%")
L.append("}")
L.append("\\caption{Performance of four LLM tool selectors across nine univariate and five "
         "multivariate TSB-AD datasets. Each cell reports the metric of the detector that the "
         "LLM selected on that dataset using a label-free Pareto frontier; no labels are used "
         "in selection. Total 1st/2nd count datasets (with ties shared) on which each selector "
         "ranks first/second for that metric.}")
L.append("\\label{tab:llm_cross_dataset}")
L.append("\\end{table*}")
(ROOT / "paper/llm_cross_table.tex").write_text(NL.join(L) + NL, encoding="utf-8")
print(NL.join(L))
