"""Smoke test: 7 detectors actually run to produce scores + 10 knowledge retrieval tools return metadata.

Usage: python Anomaly_algorithm/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Anomaly_algorithm.registry import by_category, discover, get, list_tools  # noqa: E402


def _finite(result) -> bool:
    if not result.ok:
        return False
    arr = np.asarray(result.data)
    return arr.size > 0 and np.isfinite(arr.astype(float)).all()


def main() -> int:
    tools = discover()
    print(f"Total tools: {len(tools)}")
    for cat, names in by_category().items():
        print(f"  {cat}: {len(names)} tools")

    rng = np.random.default_rng(0)
    t = np.arange(120)
    clean = np.sin(t / 10.0)
    data = clean + rng.normal(0, 0.1, 120)
    data[60:64] += 8.0  # inject a spike
    multi = np.column_stack([data, np.cos(t / 10.0) + rng.normal(0, 0.1, 120)])

    failed = []

    # 1) Real detectors: produce finite scores/labels
    detectors = [
        ("pyod", lambda: get("pyod").run(data, algorithm="iforest")),
        ("pyod(hbos)", lambda: get("pyod").run(data, algorithm="hbos")),
        ("adtk", lambda: get("adtk").run(data, detector="quantile")),
        ("merlion", lambda: get("merlion").run(data, detector="default")),
        ("tods", lambda: get("tods").run(multi, detector="zscore", return_scores=True)),
        ("pysad", lambda: get("pysad").run(data, model="exact_storm", window=60)),
        ("elki", lambda: get("elki").run(multi, algorithm="lof")),
    ]
    for name, call in detectors:
        result = call()
        ok = _finite(result)
        status = "OK" if ok else "FAIL"
        if not ok:
            failed.append((name, result.error or "non-finite output"))
        print(f"  [detector] {name:26s} {status}")

    # 2) Time-RCD: load real encoder weights and score pointwise (slower)
    try:
        result = get("time_rcd").run(data[:96])
        ok = _finite(result)
        status = "OK" if ok else "FAIL"
        if not ok:
            failed.append(("time_rcd", result.error or "non-finite output"))
    except Exception as exc:
        ok, status = False, "FAIL"
        failed.append(("time_rcd", f"{type(exc).__name__}: {exc}"))
    print(f"  [detector] time_rcd                 {status}")

    # 3) Knowledge retrieval tools: return metadata
    reference = [
        "dada", "foundation_model_eval", "adbench",
        "anomaly_detection_resources", "sota_deep_ad", "anomaly_detection_algorithms",
        "awesome_anomaly_detection", "awesome_tsad", "cran_anomaly_detection", "iad_survey",
    ]
    for name in reference:
        result = get(name).run()
        ok = result.ok and isinstance(result.data, dict) and result.data.get("name")
        status = "OK" if ok else "FAIL"
        if not ok:
            failed.append((name, result.error))
        print(f"  [ref] {name:34s} {status}")

    rows = list_tools()
    print(f"  [list_tools] metadata rows: {len(rows)}")

    if failed:
        print("\nFailed items:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return 1
    print("\nSmoke test all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
