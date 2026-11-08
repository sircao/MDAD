"""Generate a single summary table: 9 detectors x 5 metrics (mean over 14 datasets), with top1 red / top2 blue / top3 * within each column."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402

from gen_metric_tables import (  # noqa: E402
    ALL, DETECTORS, FILES, MUL, affiliation_f, f1_t, run_detector, standard_f1, vus_pr,
)

UNI = ["IOPS", "MGAB", "NAB", "NEK", "Power", "SED", "TODS", "UCR", "YAHOO"]
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
    # compute each detector's scores
    scores = {}
    for src, f in FILES.items():
        d = (__import__("Architecture_design.dataset", fromlist=["split_standardize_multi"])
             .split_standardize_multi(f) if src in MUL else
             __import__("Architecture_design.dataset", fromlist=["split_standardize"])
             .split_standardize(f))
        x, y = d["test_std"], d["y_test"]
        scores[src] = {"y": y}
        for name in DETECTORS:
            scores[src][name] = run_detector(x, name)[: len(y)]

    # per detector x metric: mean over 14 datasets
    mean = {det: {} for det in DETECTORS}
    for m, fn in METRICS.items():
        for det in DETECTORS:
            vals = [fn(scores[src][det], scores[src]["y"]) for src in ALL]
            mean[det][m] = float(np.mean(vals))

    # ranking per column
    ranks = {m: {} for m in METRICS}
    for m in METRICS:
        order = sorted(DETECTORS, key=lambda d: -mean[d][m])
        for i, d in enumerate(order):
            ranks[m][d] = i + 1

    # generate the single table
    L = []
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\footnotesize")
    L.append("\\setlength{\\tabcolsep}{3pt}")
    L.append("\\resizebox{\\columnwidth}{!}{%")
    L.append("\\begin{tabular}{@{}lcccc@{}}")
    L.append("\\toprule")
    L.append("Detector & Affiliation-F & F1$_T$ & Standard-F1 & VUS-PR \\\\")
    L.append("\\midrule")
    for det in DETECTORS:
        cells = [det] + [fmt(mean[det][m], ranks[m][det]) for m in METRICS]
        L.append(" & ".join(cells) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}%")
    L.append("}")
    L.append("\\caption{Mean detector performance over fourteen TSB-AD datasets (nine univariate "
             "and five multivariate, one representative series per source). Affiliation-F is the "
             "affiliation F-score; F1$_T$ is the point-adjusted F1 with tolerance $T=100$; "
             "Standard-F1 is the point-wise F1 at its best threshold; VUS-PR is the volume under "
             "the precision--recall surface. For each metric, the best value is bold red, the "
             "second best blue, and the third best carries an asterisk.}")
    L.append("\\label{tab:summary}")
    L.append("\\end{table}")
    out = "\n".join(L) + "\n"
    (ROOT / "paper/summary_table.tex").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
