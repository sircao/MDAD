"""Multivariate SOTA table: 12 detectors + LLM routing + Oracle, four metrics × 5 multivariate sources.

The external SOTA three baselines (Syntropic/Time-RCD/OmniAnomaly) are not feasible on multivariate:
- Syntropic is univariate-only (not in Optimal_Multi_algo_HP_dict)
- Time-RCD with win_size=15000 OOMs on CPU (needs 81GB)
- OmniAnomaly is a VAE, slow to train on CPU and its cache is incomplete

Therefore this table applies the SOTA table's "LLM routing + Oracle" idea to the paper's own 12-detector panel:
- reference rows: rolling_volatility(std100) / two_sided_volatility / iforest (representatives of single-direction / direction-agnostic / multivariate baseline)
- LLM routing: llama3.1-8B picks one detector per source by the label-free Pareto frontier
- Oracle: per source take the best of the 12 detectors (label-based, upper bound)

Metrics use TSB-AD's official four: Affiliation-F / F1_T / Standard-F1 / VUS-PR (%).
Each source uses one representative TSB-AD-M sequence (consistent with the methods table).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))  # Anomaly_algorithm / Architecture_design
sys.path.insert(0, str(ROOT.parent / "Architecture_design"))

from Architecture_design.client import OllamaClient  # noqa: E402
from Architecture_design.dataset import split_standardize_multi  # noqa: E402
from gen_metric_tables import (  # noqa: E402
    DETECTORS, FILES, MUL, affiliation_f, f1_t, run_detector, standard_f1, vus_pr,
)

METRICS = [("Affiliation-F", affiliation_f), ("F1$_T$", f1_t),
           ("Standard-F1", standard_f1), ("VUS-PR", vus_pr)]
MUL_ORDER = ["MSL", "PSM", "SMAP", "SMD", "SWaT"]

# reference rows (detector name -> display name)
REFS = [
    ("roll_std100", "rolling\\_volatility"),
    ("two_sided_volatility", "two\\_sided\\_volatility"),
    ("iforest", "iForest"),
]
ENTITIES = [r[1] for r in REFS] + ["LLM routing", "Oracle"]


def intrinsic(s):
    from scipy.stats import kurtosis, rankdata, skew

    s = np.nan_to_num(np.asarray(s, float), nan=0.0)
    r = rankdata(s) / len(s)
    acf = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if np.std(r) > 1e-12 else 0.0
    m = r >= np.percentile(r, 99)
    run = cur = 0
    for v in m:
        cur = cur + 1 if v else 0
        run = max(run, cur)
    return {"name": "", "acf1": round(acf, 3), "runlen": int(run),
            "kurt_rank": round(float(kurtosis(r)), 2),
            "tailgap_rank": round(float(np.percentile(r, 99.9) - np.percentile(r, 99)), 4),
            "skew_rank": round(float(skew(r)), 2)}


def pareto_flags(rows):
    flags = {r["name"]: [] for r in rows}
    for ka, kb in (("acf1", "runlen"), ("kurt_rank", "tailgap_rank")):
        for r in rows:
            dom = any(q[ka] >= r[ka] and q[kb] >= r[kb] and (q[ka] > r[ka] or q[kb] > r[kb]) for q in rows)
            if not dom:
                flags[r["name"]].append(ka)
    return flags


def raw_stats(x):
    from scipy.stats import kurtosis, skew

    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = np.nan_to_num(x, nan=0.0)
    acf = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if np.std(x) > 1e-12 else 0.0
    return {"kurt": round(float(kurtosis(x)), 1), "skew": round(float(skew(x)), 1),
            "acf1": round(float(acf), 2)}


def build_prompt(rows, flags, raw):
    tbl = "| Tool | acf1 | runlen | kurt_rank | tailgap_rank | skew | Pareto frontier |\n|---|---|---|---|---|---|---|\n"
    for r in rows:
        tbl += (f"| {r['name']} | {r['acf1']} | {r['runlen']} | {r['kurt_rank']} | "
                f"{r['tailgap_rank']} | {r['skew_rank']} | {'+'.join(flags[r['name']]) or '-'} |\n")
    return (
        "You are an anomaly-detection tool-selection assistant. Given the raw statistics of a multivariate series: "
        f"kurtosis={raw['kurt']}, skew={raw['skew']}, acf1={raw['acf1']}.\n"
        "Below are the label-free intrinsic metrics of each detector: acf1=score smoothness; runlen=longest contiguous top-1% run; "
        "kurt_rank=score peakedness; tailgap_rank=extreme-tail gap. The \"Pareto frontier\" column marks non-dominated tools.\n"
        "Rules: two_sided detectors are direction-agnostic (no phase inversion); roll_* are single-direction (inversion if the direction is wrong). "
        "Prefer direction-agnostic detectors unless the direction can be determined exactly.\n\n"
        f"{tbl}\nCombine the raw statistics and intrinsic metrics to pick the detector with the correct direction and highest precision. Output only the tool name."
    )


def main():
    # 1) run 12 detectors × 5 multivariate sources, compute four metrics
    scores = {}  # src -> {det -> scores}
    for src in MUL_ORDER:
        d = split_standardize_multi(str(ROOT.parent / FILES[src]))
        x, y = d["test_std"], d["y_test"]
        scores[src] = {"y": y, "x": x}
        for name in DETECTORS:
            scores[src][name] = run_detector(x, name)[: len(y)]
        print(f"[{src}] {x.shape[1]} channels / test {len(y)} points / anomalies {int((y > 0).sum())}", flush=True)

    metric_vals = {mname: {src: {det: fn(scores[src][det], scores[src]["y"]) for det in DETECTORS}
                           for src in MUL_ORDER}
                   for mname, fn in METRICS}

    # 2) LLM routing: llama3.1 per-source Pareto detector selection
    # guardrail: drop degenerate detectors (near-zero score variance makes acf1 spuriously high, see the same std guardrail in architecture.py)
    client = OllamaClient(model="llama3.1:8b")
    picks = {}
    for src in MUL_ORDER:
        rows = []
        for name in DETECTORS:
            s = np.nan_to_num(np.asarray(scores[src][name], float), nan=0.0)
            if np.std(s) < 1e-6 or np.unique(np.round(s, 6)).size <= 2:
                continue  # degenerate score, do not enter routing candidates
            m = intrinsic(s)
            m["name"] = name
            rows.append(m)
        flags = pareto_flags(rows)
        routable = [r["name"] for r in rows]
        try:
            rep = client.chat([
                {"role": "system", "content": "You are an anomaly-detection tool-selection assistant; output only one tool name."},
                {"role": "user", "content": build_prompt(rows, flags, raw_stats(scores[src]["x"]))},
            ])["content"]
            choice = next((n for n in routable if n in rep.lower()), None)
        except Exception as e:
            print(f"  [{src}] LLM call failed {e}", flush=True)
            choice = None
        picks[src] = choice or routable[0]
        print(f"  LLM {src:5s} -> {picks[src]} (candidates {len(routable)}/12)", flush=True)

    # 3) assemble entity -> metric -> source -> value (%)
    ent = {e: {mname: {} for mname, _ in METRICS} for e in ENTITIES}
    for det_key, disp in REFS:
        for mname, _ in METRICS:
            for src in MUL_ORDER:
                ent[disp][mname][src] = metric_vals[mname][src][det_key] * 100
    for mname, _ in METRICS:
        for src in MUL_ORDER:
            ent["LLM routing"][mname][src] = metric_vals[mname][src][picks[src]] * 100
            ent["Oracle"][mname][src] = max(metric_vals[mname][src][d] for d in DETECTORS) * 100

    # 4) generate LaTeX (aligned with paper2_sota_table.tex format)
    BS = "\\"
    n = len(ENTITIES)
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{@{}cc " + "c" * len(MUL_ORDER) + r" c cc@{}}")
    L.append(r"\toprule")
    L.append(r"\multirow{2}{*}{\textbf{Metric}} & \multirow{2}{*}{\textbf{Method}} & "
             + r"\multicolumn{" + str(len(MUL_ORDER)) + r"}{c}{\textbf{Multivariate Datasets}} & "
             + r"\multirow{2}{*}{\textbf{Mean}} & \multirow{2}{*}{\makecell{\textbf{Total}\\\textbf{1st}}} & "
             + r"\multirow{2}{*}{\makecell{\textbf{Total}\\\textbf{2nd}}} \\")
    L.append(r"\cmidrule(lr){3-" + str(2 + len(MUL_ORDER)) + r"}")
    L.append("& & " + " & ".join(r"\textbf{" + s + "}" for s in MUL_ORDER) + r" & & & \\")
    L.append(r"\midrule")

    def fmt(v, rank):
        if rank == 1:
            return r"\textcolor{red}{\textbf{" + f"{v:.1f}" + r"}}"
        if rank == 2:
            return r"\textcolor{blue}{" + f"{v:.1f}" + r"}"
        if rank == 3:
            return f"{v:.1f}" + r"${}^{*}$"
        return f"{v:.1f}"

    for mi, (mname, _) in enumerate(METRICS):
        ranks = {}
        for s in MUL_ORDER:
            scored = sorted(ENTITIES, key=lambda e: -ent[e][mname][s])
            ranks[s] = {e: i + 1 for i, e in enumerate(scored)}
        for i, e in enumerate(ENTITIES):
            first = r"\multirow{" + str(n) + r"}{*}{" + mname + "}" if i == 0 else ""
            cells = " & ".join(fmt(ent[e][mname][s], ranks[s][e]) for s in MUL_ORDER)
            vals = [ent[e][mname][s] for s in MUL_ORDER]
            mean = f"{sum(vals)/len(vals):.1f}"
            c1 = sum(1 for s in MUL_ORDER if ranks[s][e] == 1)
            c2 = sum(1 for s in MUL_ORDER if ranks[s][e] == 2)
            L.append(first + " & " + e + " & " + cells + f" & {mean} & {c1} & {c2} \\\\")
        L.append(r"\bottomrule" if mi == len(METRICS) - 1 else r"\midrule")
    L.append(r"\end{tabular}%")
    L.append("}")
    L.append(r"\caption{Detector panel, LLM source routing and Oracle on the five multivariate TSB-AD "
             r"sources, official Affiliation-F (\%) and companion metrics. LLM routing is llama3.1-8B "
             r"selecting one detector per source from label-free Pareto-frontier features; Oracle is the "
             r"per-source best single detector (label-based upper bound). Best per source in bold red, "
             r"second blue, third marked with an asterisk.}")
    L.append(r"\label{tab:sota_multi}")
    L.append(r"\end{table*}")
    tex = "\n".join(L)

    out = ROOT.parent / "paper" / "paper2_sota_multivar_table.tex"
    out.write_text(tex, encoding="utf-8")

    # store JSON for review
    dump = {"picks": picks, "metric_vals": {m: {s: {d: round(v * 100, 2) for d, v in dd.items()}
                                                  for s, dd in mm.items()} for m, mm in metric_vals.items()},
            "entity": {e: {m: {s: round(v, 2) for s, v in mm.items()} for m, mm in ee.items()}
                       for e, ee in ent.items()}}
    (ROOT / "runs" / "sota_multivar.json").parent.mkdir(exist_ok=True)
    (ROOT / "runs" / "sota_multivar.json").write_text(json.dumps(dump, ensure_ascii=False, indent=1))

    print(tex)
    print("\n[saved]", out)
    print("[JSON]", ROOT / "runs" / "sota_multivar.json")


if __name__ == "__main__":
    main()
