"""Merge the redundant metric tables in paper.tex into one summary table: delete
the cross-dataset detector table + 4 TSB-AD metric tables, keep the LLM selector
table, and insert summary_table."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
paper = ROOT / "paper" / "paper.tex"
summary = ROOT / "paper" / "summary_table.tex"

t = paper.read_text(encoding="utf-8")
s = summary.read_text(encoding="utf-8").strip()

# extract the LLM paragraph + table (from "Table~\ref{tab:llm_cross_dataset}" to the end of its table*)
llm_start = t.index(r"Table~\ref{tab:llm_cross_dataset}")
llm_end = t.index(r"\label{tab:llm_cross_dataset}") + len(r"\label{tab:llm_cross_dataset}")
llm_end = t.index(r"\end{table*}", llm_end) + len(r"\end{table*}")
llm_block = t[llm_start:llm_end].strip()

# locate the whole block to replace: from Cross-Dataset to before Label-Free Ceiling
block_start = t.index(r"\subsection{Cross-Dataset Comparison}")
block_end = t.index(r"\subsection{The Label-Free Ceiling}")

new_intro = (
    "To assess detectors across a broader source distribution than the ten-series batch, we "
    "evaluate all nine detectors on fourteen datasets---nine univariate and five multivariate "
    "sources, one representative series per source---and report five metrics: ROC-AUC and VUS-PR "
    "(threshold-free), the affiliation F-score, Standard-F1 (the point-wise F1 at its best "
    "threshold), and F1$_T$ (the point-adjusted F1 with tolerance $T=100$). "
    "Table~\\ref{tab:summary} reports each detector's mean over the fourteen datasets; within each "
    "metric the best value is bold red, the second best blue, and the third best carries an "
    "asterisk. No detector dominates---the best value is spread across at least three detectors on "
    "every metric---and the point-adjusted ranking disagrees with the point-wise ranking, because "
    "F1$_T$ is lenient by construction on segment-level labels.\n\n"
)

new_block = (
    "\\subsection{Cross-Dataset Comparison}\n\n"
    + new_intro
    + s
    + "\n\n"
    + llm_block
    + "\n\n"
)

t_new = t[:block_start] + new_block + t[block_end:]
paper.write_text(t_new, encoding="utf-8")
print("Rewrote paper.tex")
print("New block length:", len(new_block))
print("summary table inserted:", "tab:summary" in t_new)
print("LLM table kept:", "tab:llm_cross_dataset" in t_new)
print("Redundant tables deleted:", all(x not in t_new for x in
      ["tab:cross_dataset", "tab:affiliation_f", "tab:f1_t", "tab:standard_f1", "tab:vus_pr"]))
