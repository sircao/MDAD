"""Generate the cross-dataset comparison table with Model=LLM.

For each LLM x each dataset: use the Pareto frontier (label-free intrinsic
metrics) to let the LLM autonomously pick a detector, and record that detector's
AUC-ROC/AUC-PR/F1 as the LLM's score on that dataset. Label-free throughout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, rankdata, skew

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Anomaly_algorithm.registry import get as aget  # noqa: E402
from Architecture_design.client import OllamaClient  # noqa: E402
from Architecture_design.dataset import split_standardize, split_standardize_multi  # noqa: E402
from Architecture_design.metrics import evaluate  # noqa: E402

FILES = {
    "IOPS": "TSB-AD-U/260_IOPS_id_1_WebService_tr_5000_1st_11804.csv",
    "MGAB": "TSB-AD-U/225_MGAB_id_1_Synthetic_tr_25000_1st_38478.csv",
    "NAB": "TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv",
    "NEK": "TSB-AD-U/278_NEK_id_2_WebService_tr_500_1st_64.csv",
    "Power": "TSB-AD-U/302_Power_id_1_Facility_tr_7685_1st_7785.csv",
    "SED": "TSB-AD-U/234_SED_id_1_Medical_tr_3740_1st_3840.csv",
    "TODS": "TSB-AD-U/287_TODS_id_1_Synthetic_tr_500_1st_11.csv",
    "UCR": "TSB-AD-U/303_UCR_id_1_Medical_tr_2499_1st_8000.csv",
    "YAHOO": "TSB-AD-U/551_YAHOO_id_1_Synthetic_tr_500_1st_893.csv",
    "MSL": "TSB-AD-M/002_MSL_id_1_Sensor_tr_500_1st_900.csv",
    "PSM": "TSB-AD-M/115_PSM_id_1_Facility_tr_50000_1st_129872.csv",
    "SMAP": "TSB-AD-M/144_SMAP_id_1_Sensor_tr_2052_1st_5300.csv",
    "SMD": "TSB-AD-M/057_SMD_id_1_Facility_tr_4529_1st_4629.csv",
    "SWaT": "TSB-AD-M/171_SWaT_id_1_Sensor_tr_3749_1st_9522.csv",
}
UNI = ["IOPS", "MGAB", "NAB", "NEK", "Power", "SED", "TODS", "UCR", "YAHOO"]
MUL = ["MSL", "PSM", "SMAP", "SMD", "SWaT"]
ALL = UNI + MUL
MODELS = ["llama3.1:8b", "qwen2.5-coder:7b", "qwen3-coder-30b-a3b:latest", "ibm/granite3.3:8b"]
METRICS = ["auc_roc", "auc_pr", "best_f1"]
MLAB = {"auc_roc": "AUC-ROC", "auc_pr": "AUC-PR", "best_f1": "F1"}

DETECTORS = {
    "iforest": ("pyod", {"algorithm": "iforest"}),
    "knn": ("pyod", {"algorithm": "knn"}),
    "lof": ("pyod", {"algorithm": "lof"}),
    "tods_ma": ("tods", {"detector": "moving_average", "return_scores": True}),
    "tods_z": ("tods", {"detector": "zscore", "return_scores": True}),
    "roll_std50": ("rolling_volatility", {"stat": "std", "window": 50}),
    "roll_std100": ("rolling_volatility", {"stat": "std", "window": 100}),
    "roll_range100": ("rolling_volatility", {"stat": "range", "window": 100}),
    "flatline": ("__flat__", {"window": 100}),
    "two_sided_volatility": ("two_sided_volatility", {"window": 100}),
    "two_sided_knn": ("two_sided_knn", {"n_neighbors": 5}),
    "stuck": ("stuck", {"eps": 1e-3}),
}


def run_detector(x, name):
    tool, params = DETECTORS[name]
    if tool == "__flat__":
        r = aget("rolling_volatility").run(x, stat="std", window=params["window"])
        return -np.asarray(r.data).ravel()
    r = aget(tool).run(x, **params)
    a = np.asarray(r.data)
    if hasattr(r.data, "iloc"):
        a = r.data.iloc[:, 0].to_numpy()
    a = np.asarray(a, float).ravel()
    return a.max(axis=1) if a.ndim == 2 else a


def intrinsic(s):
    s = np.nan_to_num(np.asarray(s, float), nan=0.0)
    r = rankdata(s) / len(s)
    acf = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if np.std(r) > 1e-12 else 0.0
    m = r >= np.percentile(r, 99)
    run = cur = 0
    for v in m:
        cur = cur + 1 if v else 0
        run = max(run, cur)
    return {"acf1": round(acf, 3), "runlen": int(run),
            "kurt_rank": round(float(kurtosis(r)), 2),
            "tailgap_rank": round(float(np.percentile(r, 99.9) - np.percentile(r, 99)), 4),
            "skew_rank": round(float(skew(r)), 2)}


def raw_stats(x):
    """Raw series statistics: kurtosis / skew / acf1, used to infer anomaly morphology (point vs segment vs stuck/flatline)."""
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = np.nan_to_num(x, nan=0.0)
    acf = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if np.std(x) > 1e-12 else 0.0
    return {"kurt": round(float(kurtosis(x)), 1), "skew": round(float(skew(x)), 1),
            "acf1": round(float(acf), 2)}


def pareto_flags(rows):
    flags = {r["name"]: [] for r in rows}
    for ka, kb in (("acf1", "runlen"), ("kurt_rank", "tailgap_rank")):
        for r in rows:
            dom = any(q[ka] >= r[ka] and q[kb] >= r[kb] and (q[ka] > r[ka] or q[kb] > r[kb]) for q in rows)
            if not dom:
                flags[r["name"]].append(ka)
    return flags


def build_prompt(rows, flags, raw):
    tbl = "| tool | acf1 | runlen | kurt_rank | tailgap_rank | skew | Pareto front |\n|---|---|---|---|---|---|---|\n"
    for r in rows:
        tbl += f"| {r['name']} | {r['acf1']} | {r['runlen']} | {r['kurt_rank']} | {r['tailgap_rank']} | {r['skew_rank']} | {'+'.join(flags[r['name']]) or '-'} |\n"
    return (
        "You are an anomaly detection tool-selection assistant. Given the raw statistics of the series: "
        f"kurtosis={raw['kurt']}, skew={raw['skew']}, acf1={raw['acf1']}.\n"
        "Below are each detector's label-free intrinsic metrics: acf1=score smoothness; runlen=longest contiguous top-1% segment; "
        "kurt_rank=score peakedness; tailgap_rank=extreme-tail gap; skew=score skewness. "
        "The \"Pareto front\" column marks tools that are non-dominated on (acf1,runlen) or (kurt_rank,tailgap_rank).\n"
        "Rules: two_sided_volatility / two_sided_knn are direction-agnostic (deviation from the median is anomalous, so no inversion); "
        "roll_std*/roll_range/flatline are one-directional (they assume high or low volatility is anomalous, and will be inverted when the direction is opposite). "
        "For robustness, prefer direction-agnostic two_sided detectors; choose a one-directional detector only when the anomaly direction can be determined exactly.\n\n"
        f"{tbl}\nCombine the raw statistics and intrinsic metrics to pick the detector with the correct direction and highest accuracy. Output only the tool name."
    )


def main():
    # precompute each dataset's detector scores + intrinsic metrics
    det_scores = {}   # src -> {det -> scores}
    for src, f in FILES.items():
        multi = src in MUL
        d = split_standardize_multi(f) if multi else split_standardize(f)
        x, y = d["test_std"], d["y_test"]
        det_scores[src] = {"y": y, "x": x}
        for name in DETECTORS:
            det_scores[src][name] = run_detector(x, name)[: len(y)]

    # per LLM x dataset: Pareto selection
    selection = {}  # model -> {src -> chosen det}
    for model in MODELS:
        client = OllamaClient(model=model)
        selection[model] = {}
        print(f"=== {model} ===", flush=True)
        for src in ALL:
            rows = []
            for name in DETECTORS:
                m = intrinsic(det_scores[src][name])
                m["name"] = name
                rows.append(m)
            flags = pareto_flags(rows)
            try:
                rep = client.chat([
                    {"role": "system", "content": "You are an anomaly detection tool-selection assistant; output only one tool name."},
                    {"role": "user", "content": build_prompt(rows, flags, raw_stats(det_scores[src]["x"]))},
                ])["content"]
                choice = next((n for n in DETECTORS if n in rep.lower()), None)
            except Exception as e:
                print(f"  {src} failed: {e}", flush=True)
                choice = None
            selection[model][src] = choice or "iforest"  # fallback
            print(f"  {src:6s} -> {selection[model][src]}", flush=True)

    # compute each LLM x dataset metric (the score of its selected detector)
    res = {m: {} for m in METRICS}
    for m in METRICS:
        for model in MODELS:
            res[m][model] = {}
            for src in ALL:
                det = selection[model][src]
                res[m][model][src] = evaluate(det_scores[src][det], det_scores[src]["y"])[m]

    # Total 1st/2nd
    tot1 = {m: {} for m in METRICS}
    tot2 = {m: {} for m in METRICS}
    for m in METRICS:
        for model in MODELS:
            c1 = c2 = 0
            for src in ALL:
                vals = sorted([(mm, res[m][mm][src]) for mm in MODELS], key=lambda t: -t[1])
                rank = [i for i, (mm, v) in enumerate(vals) if mm == model][0] + 1
                if rank == 1:
                    c1 += 1
                elif rank == 2:
                    c2 += 1
            tot1[m][model] = c1
            tot2[m][model] = c2

    json.dump({"selection": selection, "res": res, "tot1": tot1, "tot2": tot2},
              open(ROOT / "Architecture_design/runs/llm_cross_dataset.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("Saved llm_cross_dataset.json")

    # generate LaTeX table
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
             "in selection. Total 1st/2nd count datasets on which each selector ranks first/second "
             "for that metric.}")
    L.append("\\label{tab:llm_cross_dataset}")
    L.append("\\end{table*}")
    out = "\n".join(L) + "\n"
    (ROOT / "paper/llm_cross_table.tex").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
