"""Final full_leaderboard: Syntropic + CHARM + LLM routing (two-level routing).

- Model rows: official leaderboard 15 models + Syntropic + CHARM_ZS (zero-shot block)/CHARM_kNN (full-shot block).
- CHARM only has VUS-PR (local CSV); the other metrics are ---.
- LLM routing = legacy two-level routing (morphology→LLM→detector, official VUS-PR table lookup),
  per-source selection (including CHARM); VUS-PR has values on all 16 sources, other metrics only for Syntropic sources, CHARM sources ---.
- Oracle = per source take the max within the block.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from gen_full_leaderboard import DATA, SRC16, MUL, METRICS, ZERO_SHOT, FULL_SHOT, load_syntropic  # noqa: E402

LDR = ROOT / "benchmark_exp" / "leaderboard_results"


def load_charm_vus():
    """CHARM_kNN / CHARM_ZS per-source VUS-PR (local CSV, VUS-PR only; per-file→per-source mean).

    Univariate/multivariate separated metrics: univariate sources (first 11) read only the uni CSV, multivariate sources (MSL/PSM/SMAP/SMD/SWaT) read only
    the multi CSV. Otherwise MSL/SMAP/SMD/SWaT/PSM share names between TSB-AD-U and TSB-AD-M and would be mixed-averaged.
    """
    agg = {m: {src: [] for src in SRC16} for m in ["CHARM_kNN", "CHARM_ZS"]}
    for f, is_multi in [("CHARM_uni_mergedTable_VUS-PR.csv", False),
                        ("CHARM_multi_mergedTable_VUS-PR.csv", True)]:
        df = pd.read_csv(LDR / f)
        for _, r in df.iterrows():
            key = r["file"] + ("" if str(r["file"]).endswith(".csv") else ".csv")
            src = key.split("_")[1]
            if src not in SRC16 or (src in MUL) != is_multi:
                continue
            agg["CHARM_kNN"][src].append(float(r["CHARM_kNN"]))
            agg["CHARM_ZS"][src].append(float(r["CHARM_ZS"]))
    out = {m: {mk: [None] * 16 for mk, _ in METRICS} for m in ["CHARM_kNN", "CHARM_ZS"]}
    for m in ["CHARM_kNN", "CHARM_ZS"]:
        for src, vs in agg[m].items():
            if not vs:
                continue
            j = SRC16.index(src)
            out[m]["VUS-PR"][j] = round(float(np.mean(vs)) * 100, 2)
    return out


def load_legacy_picks():
    d = json.load(open(ROOT / "runs" / "llm_routing_combined.json", encoding="utf-8"))
    return d.get("picks_u", {}), d.get("picks_m", {})


def main():
    syntropic = load_syntropic()
    charm = load_charm_vus()

    data = {m: {mk: list(DATA[m][mk]) for mk, _ in METRICS} for m in set(ZERO_SHOT) | set(FULL_SHOT)}
    data["Syntropic"] = syntropic
    data["CHARM_kNN"] = charm["CHARM_kNN"]
    data["CHARM_ZS"] = charm["CHARM_ZS"]

    picks_u, picks_m = load_legacy_picks()

    # LLM routing: per source = the selected model's value for that source and metric
    routing = {mk: [None] * 16 for mk, _ in METRICS}
    for j, s in enumerate(SRC16):
        p = picks_u.get(s) if j < 11 else picks_m.get(s)
        if p is None:
            continue
        for mk, _ in METRICS:
            routing[mk][j] = data[p][mk][j]
    data["LLM routing"] = routing
    data["Oracle"] = {mk: [None] * 16 for mk, _ in METRICS}

    def fmt(v, rank):
        if v is None:
            return "---"
        s = f"{v:.2f}"
        if rank == 1:
            return r"\textcolor{red}{\textbf{" + s + r"}}"
        if rank == 2:
            return r"\textcolor{blue}{" + s + r"}"
        if rank == 3:
            return s + r"${}^{*}$"
        return s

    def render(entities, block_title):
        models = [e for e in entities if e not in ("LLM routing", "Oracle")]
        n = len(entities)
        L = [r"\multicolumn{20}{c}{\textbf{" + block_title + r"}} \\", r"\midrule"]
        for mi, (mk, disp) in enumerate(METRICS):
            for j in range(16):
                vals = [data[e][mk][j] for e in models if data[e][mk][j] is not None]
                data["Oracle"][mk][j] = max(vals) if vals else None
            ranks = {}
            for j in range(16):
                scored = [(e, data[e][mk][j]) for e in entities if data[e][mk][j] is not None]
                scored.sort(key=lambda x: -x[1])
                ranks[j] = {e: i + 1 for i, (e, _) in enumerate(scored)}
            for i, e in enumerate(entities):
                first = (r"\multirow{" + str(n) + r"}{*}{" + disp + "}") if i == 0 else ""
                cells = [fmt(data[e][mk][j], ranks[j].get(e, 99)) for j in range(16)]
                c1 = sum(1 for j in range(16) if ranks[j].get(e) == 1)
                c2 = sum(1 for j in range(16) if ranks[j].get(e) == 2)
                L.append(first + " & " + e + " & " + " & ".join(cells) + f" & {c1:02d} & {c2:02d} \\\\")
            L.append(r"\midrule")
        return L

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}cc " + "c" * 16 + r" cc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Metric}} & \multirow{2}{*}{\textbf{Model}} & \multicolumn{11}{c}{\textbf{Univariate Datasets}} & \multicolumn{5}{c}{\textbf{Multivariate Datasets}} & \multirow{2}{*}{\makecell{\textbf{Total}\\\textbf{1st}}} & \multirow{2}{*}{\makecell{\textbf{Total}\\\textbf{2nd}}} \\")
    lines.append(r"\cmidrule(lr){3-13} \cmidrule(lr){14-18}")
    lines.append("& & " + " & ".join(r"\textbf{" + s + "}" for s in SRC16) + r" & & \\")
    lines.append(r"\midrule")

    lines += render(ZERO_SHOT + ["CHARM_ZS", "LLM routing", "Oracle"], "Zero-Shot Models")
    lines += render(FULL_SHOT + ["Syntropic", "CHARM_kNN", "LLM routing", "Oracle"], "Full-Shot Models")

    lines.append(r"\end{tabular}%")
    lines.append("}")
    lines.append(r"\caption{Full TSB-AD leaderboard (zero-shot and full-shot blocks) with our Syntropic detector and CHARM added, on the official Affiliation-F (\%) and companion metrics. LLM routing is our two-level morphology-gated routing (llama3.1/qwen3/qwen2.5) selecting one detector per source from the strong candidate panel including CHARM; its VUS-PR is reported on all sixteen sources, while CHARM's other metrics are unavailable (---). Best per source in bold red, second blue, third marked with an asterisk; --- marks a missing submission or metric.}")
    lines.append(r"\label{tab:full_leaderboard}")
    lines.append(r"\end{table*}")

    tex = "\n".join(lines)
    out = ROOT.parent / "paper" / "full_leaderboard.tex"
    out.write_text(tex, encoding="utf-8")
    print("[saved]", out)
    none = sum(1 for j in range(16) if routing["VUS-PR"][j] is None)
    print(f"LLM routing VUS-PR missing {none}/16")


if __name__ == "__main__":
    main()
