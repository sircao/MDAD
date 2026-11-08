"""Fully model-autonomous selection: the LLM freely calls detector tools, compares
score features, and autonomously picks the optimal detector.

The model is not given a precomputed Pareto table or ranking -- it decides for
itself which detectors to run, how to compare score morphology, how to incorporate
domain knowledge from the series source, and finally makes the call on its own.
Label-free throughout.

Usage::

    python Architecture_design/autonomous_select.py --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Anomaly_algorithm.registry import get as aget  # noqa: E402
from Architecture_design.client import OllamaClient  # noqa: E402
from Architecture_design.dataset import series_stats, split_standardize  # noqa: E402
from Architecture_design.metrics import evaluate  # noqa: E402

_DETECTORS = {
    "iforest": ("pyod", {"algorithm": "iforest"}),
    "knn": ("pyod", {"algorithm": "knn"}),
    "lof": ("pyod", {"algorithm": "lof"}),
    "tods_ma": ("tods", {"detector": "moving_average", "return_scores": True}),
    "tods_z": ("tods", {"detector": "zscore", "return_scores": True}),
    "roll_std50": ("rolling_volatility", {"stat": "std", "window": 50}),
    "roll_std100": ("rolling_volatility", {"stat": "std", "window": 100}),
    "roll_range100": ("rolling_volatility", {"stat": "range", "window": 100}),
    "flatline": ("__flatline__", {"window": 100}),
}

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_detector",
            "description": "Run an anomaly detector and return a summary of its score features (mean/std/extremes/top10 indices/extreme-point count), for comparison and selection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": sorted(_DETECTORS),
                        "description": "Detector name.",
                    }
                },
                "required": ["name"],
            },
        },
    }
]

_SYSTEM = """You are an industrial anomaly detection expert. You may repeatedly call the run_detector tool to run different detectors.

Your task: **autonomously** explore multiple detectors, compare the score features they return, and, combined with domain knowledge of the series source,
pick the detector most suitable for this series.

Comparison points:
- large max/std with a very small n_over_3std → scores are "spiky" (a few extreme points), suitable for point anomalies;
- top10 indices forming contiguous segments with a larger n_over_3std → scores are "smooth segments", suitable for segment-level/drift anomalies;
- consider the source: Finance/Stock are mostly point-anomaly spikes; Sensor/Facility are mostly drift or spikes; Medical may be stuck (flatline).

When you decide the choice, output one line: Choice:<tool name> (e.g. Choice:tods_z), with no other content."""


def _flatline(x, window):
    import pandas as pd

    vol = pd.Series(np.asarray(x, float)).rolling(window, min_periods=window // 2).std().to_numpy()
    return -np.nan_to_num(vol, nan=0.0)


def _run_detector(x, name):
    tool, params = _DETECTORS[name]
    if tool == "__flatline__":
        s = _flatline(x, params["window"])
    else:
        r = aget(tool).run(x, **params)
        if not r.ok:
            return None, r.error
        a = np.asarray(r.data)
        if hasattr(r.data, "iloc"):
            a = r.data.iloc[:, 0].to_numpy()
        a = np.asarray(a, float).ravel()
        s = a.max(axis=1) if a.ndim == 2 else a
    return s, None


def _summary(s):
    s = np.asarray(s, float)
    top10 = np.argsort(-s)[:10].tolist()
    # number of consecutive runs among the top10 (fewer = more "spiky/scattered", more = more "segment-like")
    runs = 1
    for a, b in zip(top10, top10[1:]):
        if abs(b - a) > 1:
            runs += 1
    mu, sd = float(s.mean()), float(s.std())
    return {
        "n": int(s.size),
        "mean": round(mu, 3),
        "std": round(sd, 3),
        "min": round(float(s.min()), 3),
        "max": round(float(s.max()), 3),
        "top10_indices": top10,
        "top10_runs": runs,
        "n_over_3std": int((s > mu + 3 * sd).sum()),
    }


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--seq", default="TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv")
    args = ap.parse_args(argv)

    data = split_standardize(args.seq)
    x, y, meta = data["test_std"], data["y_test"], data["meta"]
    client = OllamaClient(model=args.model)

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Series source: {meta.get('src', 'unknown')} (id={meta.get('id')}).\n"
                f"Series statistics: {json.dumps(series_stats(x), ensure_ascii=False)}\n"
                "Please autonomously explore detectors and pick the optimal one."
            ),
        },
    ]

    choice = None
    for _ in range(8):
        reply = client.chat(messages, tools=_TOOLS)
        if reply["tool_calls"]:
            messages.append(
                {"role": "assistant", "content": reply["content"],
                 "tool_calls": [{"function": {"name": t["name"], "arguments": t["arguments"]}} for t in reply["tool_calls"]]}
            )
            for t in reply["tool_calls"]:
                name = t["arguments"].get("name", "")
                s, err = _run_detector(x, name)
                res = {"ok": False, "error": err} if err else {"ok": True, ** _summary(s)}
                messages.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False), "tool_name": t["name"]})
                print(f"  [call] {name} -> {json.dumps(res, ensure_ascii=False)[:200]}", flush=True)
            continue
        content = reply["content"]
        if "Choice:" in content or "Choice：" in content:
            choice = content.split("Choice")[-1].strip(" :：\n")
            choice = choice.split()[0] if choice else None
        if choice:
            break

    print(f"\n=== {Path(args.seq).name} | model {args.model} autonomous selection: {choice} ===")
    if choice in _DETECTORS:
        s, _ = _run_detector(x, choice)
        m = evaluate(s[: len(y)], y)
        for k, v in m.items():
            print(f"  {k:16s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
