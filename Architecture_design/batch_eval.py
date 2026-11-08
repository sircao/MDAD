"""Batch evaluation: run the real LLM agent over multiple TSB-AD-U series and
print metrics per series.

Usage::

    python Architecture_design/batch_eval.py <file1> <file2> ... [--model llama3.1:8b]

Each series prints one line of JSON (including the detector the LLM chose
autonomously + threshold-free metrics), which can be redirected to a JSONL
summary.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Architecture_design.agent import IndustrialAgent  # noqa: E402
from Architecture_design.client import OllamaClient  # noqa: E402
from Architecture_design.dataset import (  # noqa: E402
    series_stats,
    split_standardize,
    split_standardize_multi,
)
from Architecture_design.executor import ToolExecutor  # noqa: E402
from Architecture_design.metrics import evaluate  # noqa: E402


def run_one(path: str, model: str = "llama3.1:8b", multi: bool = False) -> dict:
    data = split_standardize_multi(path) if multi else split_standardize(path)
    meta, y_test = data["meta"], data["y_test"]
    executor = ToolExecutor(data["test_std"], data["train"], y_test, meta)
    agent = IndustrialAgent(executor, client=OllamaClient(model=model))

    t0 = time.time()
    outcome = agent.run("Detect anomalies on the test segment", meta=meta, stats=series_stats(data["test_std"]))
    secs = round(time.time() - t0, 1)

    det = executor.score_history[0] if executor.score_history else None
    metrics = evaluate(det["scores"], y_test) if det else {}
    return {
        "file": Path(path).name,
        "src": meta.get("src"),
        "multi": multi,
        "chosen": det["tool"] if det else None,
        "chosen_args": det["args"] if det else None,
        "n_steps": len(outcome["trace"]),
        "summary": outcome["parsed"].get("summary", ""),
        "secs": secs,
        "metrics": metrics,
    }


def main(argv) -> int:
    args = list(argv)
    model = "llama3.1:8b"
    multi = False
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        del args[i : i + 2]
    if "--multi" in args:
        multi = True
        args.remove("--multi")
    files = [a for a in args if not a.startswith("--")]
    if not files:
        print("Usage: batch_eval.py <file>... [--model llama3.1:8b] [--multi]", file=sys.stderr)
        return 2

    for f in files:
        try:
            row = run_one(f, model, multi)
            print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
        except Exception as exc:  # noqa: BLE001 -- a single failure must not abort the whole batch
            print(json.dumps({"file": f, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
