# Regenerate the three paper2 tables, all using raw strings to avoid backslash escaping issues.
import json

methods = json.load(open(r"Architecture_design/runs/_paper2_methods.json", encoding="utf-8"))
ablation = json.load(open(r"Architecture_design/runs/_paper2_ablation.json", encoding="utf-8"))
cases = json.load(open(r"Architecture_design/runs/_paper2_cases.json", encoding="utf-8"))

METRICS = ["Affiliation-F", "F1_T", "Standard-F1", "VUS-PR"]
UNI = ["IOPS", "MGAB", "NAB", "NEK", "Power", "SED", "TODS", "UCR", "YAHOO"]
MUL = ["MSL", "PSM", "SMAP", "SMD", "SWaT"]
ALL = UNI + MUL

BS = chr(92)  # a single backslash, avoiding escapes entirely


def esc(t):
    return t


def fmt(v, rank):
    if rank == 1:
        return BS + r"textcolor{red}{" + BS + r"textbf{" + f"{v:.3f}" + r"}}"
    if rank == 2:
        return BS + r"textcolor{blue}{" + f"{v:.3f}" + r"}"
    if rank == 3:
        return f"{v:.3f}" + "${}^{*}$"
    return f"{v:.3f}"


def cross_table(entities, data, caption, label):
    n = len(entities)
    rows = []
    rows.append(BS + r"begin{table*}[t]")
    rows.append(BS + "centering")
    rows.append(BS + "footnotesize")
    rows.append(BS + r"setlength{" + BS + "tabcolsep}{3.5pt}")
    rows.append(BS + r"resizebox{" + BS + r"textwidth}{!}{%")
    rows.append(BS + r"begin{tabular}{@{}cc ccccccccc ccccc cc@{}}")
    rows.append(BS + "toprule")
    rows.append(
        BS + r"multirow{2}{*}{" + BS + r"textbf{Metric}} & " + BS + r"multirow{2}{*}{" + BS + r"textbf{Method}} & "
        + BS + r"multicolumn{9}{c}{" + BS + r"textbf{Univariate Datasets}} & "
        + BS + r"multicolumn{5}{c}{" + BS + r"textbf{Multivariate Datasets}} & "
        + BS + r"multirow{2}{*}{" + BS + r"makecell{" + BS + r"textbf{Total}" + BS + BS + BS + r"textbf{1st}}} & "
        + BS + r"multirow{2}{*}{" + BS + r"makecell{" + BS + r"textbf{Total}" + BS + BS + BS + r"textbf{2nd}}} " + BS + BS
    )
    rows.append(BS + r"cmidrule(lr){3-11} " + BS + r"cmidrule(lr){12-16}")
    rows.append("& & " + " & ".join(BS + "textbf{" + s + "}" for s in ALL) + " & & " + BS + BS)
    rows.append(BS + "midrule")
    for mt in METRICS:
        ranks = {}
        for s in ALL:
            vals = sorted(entities, key=lambda e: -data[e][mt][s])
            ranks[s] = {e: i + 1 for i, e in enumerate(vals)}
        for i, e in enumerate(entities):
            first = (BS + r"multirow{" + str(n) + "}{*}{" + mt + "}") if i == 0 else ""
            cells = " & ".join(fmt(data[e][mt][s], ranks[s][e]) for s in ALL)
            c1 = sum(1 for s in ALL if ranks[s][e] == 1)
            c2 = sum(1 for s in ALL if ranks[s][e] == 2)
            rows.append(first + " & " + e + " & " + cells + f" & {c1} & {c2} " + BS + BS)
        rows.append(BS + ("bottomrule" if mt == METRICS[-1] else "midrule"))
    rows.append(BS + r"end{tabular}%")
    rows.append("}")
    rows.append(BS + "caption{" + caption + "}")
    rows.append(BS + "label{" + label + "}")
    rows.append(BS + "end{table*}")
    return "\n".join(rows)


methods_tbl = cross_table(
    list(methods.keys()), methods,
    "Methods comparison across fourteen TSB-AD datasets. Ours is the morphology-first adaptive "
    "ensemble; Oracle is the best single detector chosen with labels. Best per dataset in bold red, "
    "second blue, third marked with an asterisk.",
    "tab:methods",
)
ablation_tbl = cross_table(
    list(ablation.keys()), ablation,
    "Ablation of the architecture over fourteen TSB-AD datasets. Full is the morphology-first "
    "adaptive ensemble; Uniform drops the morphology; Single routing drops the adaptive weighting; "
    "Label-free drops the source prior; Rank-max is the no-morphology baseline.",
    "tab:ablation",
)
open(r"paper/paper2_methods_table.tex", "w", encoding="utf-8").write(methods_tbl)
open(r"paper/paper2_ablation_table.tex", "w", encoding="utf-8").write(ablation_tbl)
print("methods/ablation tables rewritten")

# case table (8 cases x metrics, showing inversion correction)
order = ["NAB-001", "NAB-008", "TODS", "YAHOO", "Stock", "UCR", "SMAP", "IOPS"]
rows = []
rows.append(BS + r"begin{table}[t]")
rows.append(BS + "centering")
rows.append(BS + "footnotesize")
rows.append(BS + r"setlength{" + BS + "tabcolsep}{3pt}")
rows.append(BS + r"resizebox{" + BS + r"columnwidth}{!}{%")
rows.append(BS + r"begin{tabular}{@{}l c c c c c c c@{}}")
rows.append(BS + "toprule")
rows.append("Case & Morphology & " + BS + r"makecell{AUC" + BS + "(before)} & " + BS + r"makecell{AUC" + BS + "(Ours)} & Affiliation-F & F1$_T$ & Standard-F1 & VUS-PR " + BS + BS)
rows.append(BS + "midrule")
for name in order:
    d = cases[name]
    rows.append(f"{name} & {d['morphology']} & {d['auc_before']:.3f} & " + BS + r"textbf{" + f"{d['auc_ours']:.3f}" + "} & "
                f"{d['Affiliation-F']:.3f} & {d['F1_T']:.3f} & {d['Standard-F1']:.3f} & {d['VUS-PR']:.3f} " + BS + BS)
rows.append(BS + "bottomrule")
rows.append(BS + r"end{tabular}%")
rows.append("}")
rows.append(BS + "caption{Case study of the eight representative series under the morphology-first "
           "architecture. AUC(before) is the single-direction rolling volatility baseline; AUC(Ours) is the "
           "architecture. The inverted cases (TODS, UCR) and the weak point cases (Stock) are recovered.}")
rows.append(BS + "label{tab:case}")
rows.append(BS + "end{table}")
open(r"paper/paper2_case_table.tex", "w", encoding="utf-8").write("\n".join(rows))
print("case table rewritten")
