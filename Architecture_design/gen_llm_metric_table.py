"""Generate the LLM-selector four-metric table: 4 metrics x 4 models, with top1 red / top2 blue / top3 * within each dataset column."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gen_metric_tables import (  # noqa: E402
    ALL, DETECTORS, FILES, MUL, affiliation_f, f1_t, run_detector, standard_f1, vus_pr,
)

UNI = ["IOPS", "MGAB", "NAB", "NEK", "Power", "SED", "TODS", "UCR", "YAHOO"]
MODELS = ["llama3.1:8b", "qwen2.5-coder:7b", "qwen3-coder-30b-a3b:latest", "ibm/granite3.3:8b"]
METRICS = {
    "Affiliation-F": affiliation_f,
    "F1$_T$": f1_t,
    "Standard-F1": standard_f1,
    "VUS-PR": vus_pr,
}


def fmt(v, rank):
    if rank == 1:
        return f"\\textcolor{{red}}{{\\textbf{{{v:.3f}}}}}"
    if rank == 2:
        return f"\\textcolor{{blue}}{{{v:.3f}}}"
    if rank == 3:
        return f"{v:.3f}" + "${}^{*}$"
    return f"{v:.3f}"


def main():
    # load LLM selections
    d = json.load(open(ROOT / "Architecture_design/runs/llm_cross_dataset.json", encoding="utf-8"))
    selection = d["selection"]  # model -> {src -> chosen det}

    # recompute raw scores
    scores = {}
    for src, f in FILES.items():
        if src in MUL:
            from Architecture_design.dataset import split_standardize_multi
            dd = split_standardize_multi(f)
        else:
            from Architecture_design.dataset import split_standardize
            dd = split_standardize(f)
        x, y = dd["test_std"], dd["y_test"]
        scores[src] = {"y": y}
        for name in DETECTORS:
            scores[src][name] = run_detector(x, name)[: len(y)]

    # per LLM x dataset x metric
    res = {m: {} for m in METRICS}
    for m, fn in METRICS.items():
        for model in MODELS:
            res[m][model] = {}
            for src in ALL:
                det = selection[model][src]
                res[m][model][src] = fn(scores[src][det], scores[src]["y"])

    # Total 1st/2nd (ties shared)
    tot1 = {m: {} for m in METRICS}
    tot2 = {m: {} for m in METRICS}
    for m in METRICS:
        for model in MODELS:
            c1 = c2 = 0
            for src in ALL:
                vals = sorted({res[m][mm][src] for mm in MODELS}, reverse=True)
                v = res[m][model][src]
                if vals and abs(v - vals[0]) < 1e-9:
                    c1 += 1
                elif len(vals) > 1 and abs(v - vals[1]) < 1e-9:
                    c2 += 1
            tot1[m][model] = c1
            tot2[m][model] = c2

    # generate the table
    L = []
    L.append("\\begin{table*}[t]")
    L.append("\\centering")
    L.append("\\footnotesize")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    L.append("\\resizebox{\\textwidth}{!}{%")
    L.append("\\begin{tabular}{@{}cc ccccccccc ccccc cc@{}}")
    L.append("\\toprule")
    L.append("\\multirow{2}{*}{\\textbf{Metric}} & \\multirow{2}{*}{\\textbf{Model}} & "
             "\\multicolumn{9}{c}{\\textbf{Univariate Datasets}} & "
             "\\multicolumn{5}{c}{\\textbf{Multivariate Datasets}} & "
             "\\multirow{2}{*}{\\makecell{\\textbf{Total}\\\\\\textbf{1st}}} & "
             "\\multirow{2}{*}{\\makecell{\\textbf{Total}\\\\\\textbf{2nd}}} \\\\")
    L.append("\\cmidrule(lr){3-11} \\cmidrule(lr){12-16}")
    L.append("& & " + " & ".join(f"\\textbf{{{s}}}" for s in UNI) + " & " +
             " & ".join(f"\\textbf{{{s}}}" for s in ["MSL", "PSM", "SMAP", "SMD", "SWaT"]) + " & & \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{18}{c}{\\textbf{LLM Tool Selectors}} \\\\")
    L.append("\\midrule")
    for mi, m in enumerate(METRICS):
        for i, model in enumerate(MODELS):
            cells = []
            for src in ALL:
                order = sorted(MODELS, key=lambda mm: -res[m][mm][src])
                rank = order.index(model) + 1
                cells.append(fmt(res[m][model][src], rank))
            metric_cell = f"\\multirow{{{len(MODELS)}}}{{*}}{{{m}}}" if i == 0 else ""
            L.append(f"{metric_cell} & {model} & " + " & ".join(cells) +
                     f" & {tot1[m][model]} & {tot2[m][model]} \\\\")
        if mi < len(METRICS) - 1:
            L.append("\\midrule")
    L.append("\\bottomrule")
    L.append("\\end{tabular}%")
    L.append("}")
    L.append("\\caption{Performance of four LLM tool selectors across nine univariate and five "
             "multivariate TSB-AD datasets. Each cell reports the metric of the detector that the "
             "LLM selected on that dataset using a label-free Pareto frontier; no labels are used "
             "in selection. For each metric and dataset, the best value is bold red, the second "
             "best blue, and the third best carries an asterisk. Total 1st/2nd count datasets "
             "(with ties shared) on which each selector ranks first/second for that metric.}")
    L.append("\\label{tab:llm_cross_dataset}")
    L.append("\\end{table*}")
    out = "\n".join(L) + "\n"
    (ROOT / "paper/llm_metric_table.tex").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
