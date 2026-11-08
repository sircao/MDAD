"""LLM tool-selection benchmark (label-free decision): multi-model x multi-series,
the model picks a detector based on a Pareto frontier.

Flow:
1. For each series, compute **label-free intrinsic metrics** on the orthogonal
   "detector x window" candidate grid;
2. Hand the metric table + Pareto-frontier annotation to the LLM, so it picks the
   optimal detector using only these label-free signals;
3. The selected detector is scored with **held-out labels** for AUC -- labels are
   used only for evaluation, while the model's decision is unsupervised throughout.

This compares the selection ability of "different LLMs" horizontally, as well as
"which candidate tool of the same model" gets chosen and how it performs.

Usage::

    python Architecture_design/llm_selection_bench.py --seqs 5 --models llama3.1:8b,qwen2.5-coder:7b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, rankdata, skew

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Anomaly_algorithm.registry import get as aget  # noqa: E402
from Architecture_design.client import OllamaClient  # noqa: E402
from Architecture_design.dataset import split_standardize  # noqa: E402
from Architecture_design.metrics import evaluate  # noqa: E402

# (name, tool name, params, whether flatline special-case)
_CANDIDATES = [
    ("iforest", "pyod", {"algorithm": "iforest"}, False),
    ("knn", "pyod", {"algorithm": "knn"}, False),
    ("lof", "pyod", {"algorithm": "lof"}, False),
    ("tods_ma", "tods", {"detector": "moving_average", "return_scores": True}, False),
    ("tods_z", "tods", {"detector": "zscore", "return_scores": True}, False),
    ("roll_std50", "rolling_volatility", {"stat": "std", "window": 50}, False),
    ("roll_std100", "rolling_volatility", {"stat": "std", "window": 100}, False),
    ("roll_range100", "rolling_volatility", {"stat": "range", "window": 100}, False),
    ("flatline", None, {"window": 100}, True),
]

_DEFAULT_SEQS = [
    "TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv",   # segment-level (level shift)
    "TSB-AD-U/008_NAB_id_8_Synthetic_tr_1007_1st_2734.csv",  # segment-level
    "TSB-AD-U/149_Stock_id_1_Finance_tr_500_1st_7.csv",      # point anomaly
    "TSB-AD-U/287_TODS_id_1_Synthetic_tr_500_1st_11.csv",    # point anomaly
    "TSB-AD-U/303_UCR_id_1_Medical_tr_2499_1st_8000.csv",    # stuck/flatline (low volatility)
]


def _flatline(x: np.ndarray, window: int) -> np.ndarray:
    import pandas as pd

    vol = pd.Series(np.asarray(x, float)).rolling(window, min_periods=window // 2).std().to_numpy()
    return -np.nan_to_num(vol, nan=0.0)


def _run_candidate(x: np.ndarray, name: str, tool: str, params: dict) -> np.ndarray:
    if tool is None:  # flatline
        return _flatline(x, params["window"])
    r = aget(tool).run(x, **params)
    if not r.ok:
        raise RuntimeError(f"{name}: {r.error}")
    a = np.asarray(r.data)
    if hasattr(r.data, "iloc"):
        a = r.data.iloc[:, 0].to_numpy()
    a = np.asarray(a, float).ravel()
    if a.ndim == 2:
        a = a.max(axis=1)
    return a


def _intrinsic(s: np.ndarray) -> dict:
    s = np.nan_to_num(np.asarray(s, float), nan=0.0)
    r = rankdata(s) / len(s)
    acf = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if np.std(r) > 1e-12 else 0.0
    mask = r >= np.percentile(r, 99)
    run = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        run = max(run, cur)
    return {
        "acf1": round(acf, 3),
        "runlen": int(run),
        "kurtosis": round(float(kurtosis(s)), 2),
        "skew": round(float(skew(s)), 2),
        "tail_gap": round(float(np.percentile(s, 99.9) - np.percentile(s, 99)), 4),
    }


def _pareto_flags(rows: list) -> dict:
    """Mark the non-dominated frontier on the two objective pairs (acf1, runlen) and (kurtosis, tail_gap)."""
    flags = {r["name"]: [] for r in rows}
    for key_a, key_b in (("acf1", "runlen"), ("kurtosis", "tail_gap")):
        for r in rows:
            dominated = any(
                q[key_a] >= r[key_a] and q[key_b] >= r[key_b]
                and (q[key_a] > r[key_a] or q[key_b] > r[key_b])
                for q in rows
            )
            if not dominated:
                flags[r["name"]].append(key_a)
    return flags


def _prompt(rows: list, flags: dict) -> str:
    header = "| tool | acf1 | runlen | kurtosis | skew | tail_gap | Pareto front |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        f = "+".join(flags[r["name"]]) or "-"
        lines.append(
            f"| {r['name']} | {r['acf1']} | {r['runlen']} | {r['kurtosis']} | "
            f"{r['skew']} | {r['tail_gap']} | {f} |"
        )
    table = "\n".join(lines)
    return (
        "You are a tool-selection expert for unsupervised anomaly detection. Below are the "
        "**label-free intrinsic metrics** of 9 detectors on the same time series (the model cannot see labels):\n\n"
        f"{table}\n\n"
        "Metric meanings: high acf1 = smooth scores (tends toward segment-level/level-shift anomalies); high runlen = anomalies form segments; "
        "high kurtosis = peaked scores (tends toward point anomalies); large tail_gap = extreme-tail anomalies stand out more.\n"
        "The \"Pareto front\" column marks tools that are non-dominated on (acf1,runlen) or (kurtosis,tail_gap).\n\n"
        "Please use the Pareto-front idea to pick the one detector **most suitable for this series**. "
        "Output only the tool name (e.g. roll_std100), no explanation and no other text."
    )


def _parse_choice(text: str, names: list) -> str | None:
    low = text.lower()
    for n in names:
        if n in low:
            return n
    return None


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default=None, help="comma-separated series filenames/paths; default 5 covering three anomaly types")
    ap.add_argument("--models", default="llama3.1:8b,qwen2.5-coder:7b,qwen3-coder-30b-a3b:latest,ibm/granite3.3:8b")
    ap.add_argument("--n", type=int, default=None, help="use only the first N series")
    args = ap.parse_args(argv)

    seqs = _DEFAULT_SEQS if args.seqs is None else args.seqs.split(",")
    if args.n:
        seqs = seqs[: args.n]
    models = [m for m in args.models.split(",") if m]
    names = [c[0] for c in _CANDIDATES]

    for seq in seqs:
        data = split_standardize(seq)
        x, y = data["test_std"], data["y_test"]
        rows = []
        for name, tool, params, _ in _CANDIDATES:
            s = _run_candidate(x, name, tool, params)[: len(y)]
            m = _intrinsic(s)
            m["name"] = name
            m["auc"] = evaluate(s, y)["auc_roc"]  # only used on the evaluation side, not fed to the model
            rows.append(m)
        flags = _pareto_flags(rows)
        oracle = max(rows, key=lambda r: r["auc"])
        print(f"\n===== {Path(seq).name} (oracle best={oracle['name']} AUC={oracle['auc']:.3f}) =====", flush=True)

        for model in models:
            try:
                client = OllamaClient(model=model)
                reply = client.chat(
                    [
                        {"role": "system", "content": "You are a tool-selection expert for unsupervised anomaly detection; pick tools based only on label-free metrics."},
                        {"role": "user", "content": _prompt(rows, flags)},
                    ]
                )
                choice = _parse_choice(reply["content"], names)
                if choice is None:
                    print(f"  [{model:28s}] no tool name parsed -> {reply['content'][:60]!r}", flush=True)
                    continue
                auc = next(r["auc"] for r in rows if r["name"] == choice)
                best = max(rows, key=lambda r: r["auc"])
                print(
                    f"  [{model:28s}] picked {choice:14s} AUC={auc:.3f} "
                    f"(oracle={best['name']}:{best['auc']:.3f})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [{model:28s}] call failed: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
