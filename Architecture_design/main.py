"""End-to-end demo: for one TSB-AD-U series, let an Ollama LLM schedule tools to
perform anomaly detection.

Usage::

    python Architecture_design/main.py --file 001_NAB_id_1_Facility_tr_1007_1st_2014.csv
    python Architecture_design/main.py --file ... --model llama3.1:8b --verbose
    python Architecture_design/main.py --file ... --dry-run   # offline, use a fixed decision script
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Architecture_design import config  # noqa: E402
from Architecture_design.agent import IndustrialAgent  # noqa: E402
from Architecture_design.client import OllamaClient  # noqa: E402
from Architecture_design.dataset import series_stats, split_standardize, split_standardize_multi  # noqa: E402
from Architecture_design.executor import ToolExecutor  # noqa: E402
from Architecture_design.metrics import evaluate  # noqa: E402

DEFAULT_TSBADU_DIR = ROOT / "TSB-AD-U"
DEFAULT_TSBADM_DIR = ROOT / "TSB-AD-M"


class MockClient:
    """Fixed decision script (for dry-run): search -> volatility detection -> threshold -> merge -> conclusion."""

    model = "mock"

    def __init__(self) -> None:
        self._calls = 0

    def chat(self, messages, tools=None):
        self._calls += 1
        if self._calls == 1:
            return {"content": "", "tool_calls": [{"name": "search_toolchain", "arguments": {}}]}
        if self._calls == 2:
            return {"content": "", "tool_calls": [{"name": "rolling_volatility", "arguments": {"stat": "std", "window": 100}}]}
        if self._calls == 3:
            return {"content": "", "tool_calls": [{"name": "dynamic_threshold", "arguments": {"method": "global", "quantile": 0.99}}]}
        if self._calls == 4:
            return {"content": "", "tool_calls": [{"name": "hysteresis_merge", "arguments": {"min_duration": 5, "merge_gap": 10}}]}
        return {
            "content": json.dumps(
                {
                    "toolchain": ["search_toolchain", "rolling_volatility:std:100", "dynamic_threshold:global:0.99", "hysteresis_merge"],
                    "summary": "mock decision: Pareto search picks rolling_volatility(std,100) + threshold + merge.",
                    "detected_events": [],
                    "anomaly_ratio": 0.0,
                    "confidence": "medium",
                    "reasoning": "dry-run fixed script, no real LLM called.",
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
        }


def _locate_file(name: str, multi: bool = False) -> Path:
    """Locate a CSV in the TSB-AD-U (univariate) or TSB-AD-M (multivariate) directory by filename or prefix."""
    d = DEFAULT_TSBADM_DIR if multi else DEFAULT_TSBADU_DIR
    if Path(name).exists():
        return Path(name)
    direct = d / name
    if direct.exists():
        return direct
    matches = sorted(d.glob(name + "*"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"not found: {name} (in {d})")
    raise SystemExit(f"multiple files matched; please specify exactly: {[m.name for m in matches[:5]]}")


def _print_trace(trace) -> None:
    print("\n=== Decision trace ===")
    for i, step in enumerate(trace, 1):
        d, r = step["decision"], step["result"]
        ok = "OK" if r.get("ok") else "FAIL"
        detail = r.get("error") or ""
        if r.get("kind") == "scores":
            detail = f"scores n={r['n']} mean={r['mean']} max={r['max']}"
        elif r.get("kind") == "mask":
            detail = f"mask n_alarm={r['n_alarm']}"
        elif r.get("n_alarm") is not None:
            detail = f"alarm {r['n_alarm']} points"
        elif r.get("n_events") is not None:
            detail = f"events {r['n_events']}"
        print(f"  [{i}] {d['tool']}({json.dumps(d['args'], ensure_ascii=False)}) -> {ok} {detail}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ollama LLM schedules tools to perform TSB-AD-U anomaly detection")
    ap.add_argument("--file", required=True, help="TSB-AD-U or TSB-AD-M filename or prefix")
    ap.add_argument("--multi", action="store_true", help="load as multivariate (TSB-AD-M)")
    ap.add_argument("--model", default=None, help=f"Ollama model (default {config.OLLAMA_MODEL})")
    ap.add_argument("--dry-run", action="store_true", help="use a fixed decision script without calling Ollama")
    ap.add_argument("--verbose", action="store_true", help="print the decision trace")
    ap.add_argument("--no-save", action="store_true", help="do not write the audit JSON")
    args = ap.parse_args(argv)

    path = _locate_file(args.file, multi=args.multi)
    data = split_standardize_multi(path) if args.multi else split_standardize(path)
    meta, y_test = data["meta"], data["y_test"]

    n_ch = data["test_std"].shape[1] if data["test_std"].ndim == 2 else 1
    print(f"Series: {path.name}" + (f" (multivariate, {n_ch} channels)" if args.multi else ""))
    print(f"  source={meta['src']} train_len={meta['train_len']} first_anomaly={meta['first_anomaly']} "
          f"test_points={len(data['test_std'])} true_anomaly_points={int((y_test > 0).sum())}")

    client = MockClient() if args.dry_run else OllamaClient(model=args.model)
    raw_full = np.concatenate([data["train"], data["test"]])
    y_full = np.concatenate([data["y_train"], data["y_test"]])
    executor = ToolExecutor(data["test_std"], data["train"], y_test, meta,
                            raw_full=raw_full, train_len=len(data["train"]), y_full=y_full)
    agent = IndustrialAgent(executor, client=client)

    kind = "multivariate" if args.multi else "univariate"
    task = f"Detect anomalies on the test segment of this {kind} time series and give the anomaly event intervals and a conclusion."
    outcome = agent.run(task, meta=meta, stats=series_stats(data["test_std"]))

    if args.verbose:
        _print_trace(outcome["trace"])

    print("\n=== LLM conclusion ===")
    print(json.dumps(outcome["parsed"], ensure_ascii=False, indent=2))

    # Independent evaluation: take the first detector that produced continuous scores
    print("\n=== Independent evaluation (threshold-free) ===")
    if executor.score_history:
        first = executor.score_history[0]
        if first.get("official"):
            # official metric: TSB-AD get_metrics evaluated on the full series (train+test)
            _tsbad = str(ROOT / "TSB-AD")
            if _tsbad not in sys.path:
                sys.path.insert(0, _tsbad)
            sys.path.insert(0, str(ROOT / "TSB-AD" / "stubs"))
            from TSB_AD.evaluation.metrics import get_metrics  # noqa: E402
            from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402

            _rf = raw_full if raw_full.ndim == 2 else raw_full.reshape(-1, 1)
            sw = find_length_rank(_rf[:, 0].reshape(-1, 1), rank=1)
            m = get_metrics(np.asarray(first["scores"]).ravel(), y_full, slidingWindow=sw)
            metrics = {k: round(float(v), 4) for k, v in m.items() if isinstance(v, (int, float))}
            print(f"  main detector: {first['tool']}({json.dumps(first['args'], ensure_ascii=False)}) [official metric / full series]")
            for k, v in metrics.items():
                print(f"    {k:16s} {v}")
            report = {"detector": first["tool"], "args": first["args"], "metrics": metrics}
        else:
            scores = first["scores"]
            n_test = len(y_test)
            if len(scores) != n_test:
                scores = scores[:n_test]
            metrics = evaluate(scores, y_test[: len(scores)])
            print(f"  main detector: {first['tool']}({json.dumps(first['args'], ensure_ascii=False)})")
            for k, v in metrics.items():
                print(f"    {k:16s} {v}")
            report = {"detector": first["tool"], "args": first["args"], "metrics": metrics}
    else:
        print("  (no detector produced continuous scores, so AUC-style evaluation is unavailable)")
        report = {"detector": None, "args": None, "metrics": {}}

    if not args.no_save:
        out_dir = Path(__file__).resolve().parent / "runs"
        out_dir.mkdir(exist_ok=True)
        audit = {
            "file": path.name,
            "meta": meta,
            "model": getattr(client, "model", None),
            "final": outcome["parsed"],
            "trace": outcome["trace"],
            "evaluation": report,
        }
        out_path = out_dir / (path.stem + ".json")
        out_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\nAudit written to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
