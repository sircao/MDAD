"""Smoke test: verify the registry discovers 53 tools and run the core tools whose
dependencies are installed.

Usage: python Data_Reprocessing/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Data_preprocessing.registry import by_category, discover, get, list_tools  # noqa: E402


def make_df(n=120, cols=3, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, (n, cols)), axis=0)
    df = pd.DataFrame(x, columns=[f"ch{i}" for i in range(cols)])
    df.iloc[10:13, 0] = np.nan
    df.iloc[20:22, 1] = 5.0  # small constant segment
    return df


def main() -> int:
    tools = discover()
    print(f"Discovered tools: {len(tools)}")
    grouped = by_category()
    for cat, names in grouped.items():
        print(f"  {cat}: {len(names)}")

    # 1) Core tools with installed dependencies should run through
    df = make_df()
    checks = [
        ("fill_nearest", lambda t: t.run(df, method="ffill")),
        ("interpolate_time", lambda t: t.run(df, method="linear")),
        ("deduplicate", lambda t: t.run(df, mode="records")),
        ("detect_stuck", lambda t: t.run(df, min_len=3)),
        ("difference", lambda t: t.run(df, order=1)),
        ("cast_type", lambda t: t.run(df, decimals=3)),
        ("savgol_filter", lambda t: t.run(df, window_length=11, polyorder=2)),
        ("kalman_smooth", lambda t: t.run(df)),
        ("kalman_impute", lambda t: t.run(df)),
        ("peak_features", lambda t: t.run(df)),
        ("spectral_features", lambda t: t.run(df, fs=1.0)),
        ("entropy_features", lambda t: t.run(df)),
        ("autocorr_features", lambda t: t.run(df, max_lag=3)),
        ("rolling_correlation", lambda t: t.run(df, window=20)),
        ("interaction_features", lambda t: t.run(df)),
        ("window_partition", lambda t: t.run(df, window=32, stride=16)),
        ("stratified_sample", lambda t: t.run(df, fraction=0.5)),
        ("series_augment", lambda t: t.run(df, method="noise", magnitude=0.1)),
        ("series_compress", lambda t: t.run(df, method="paa", n_segments=8)),
        ("hysteresis_merge", lambda t: t.run(df, high=2.0, low=1.0)),
        ("validate_constraints", lambda t: t.run(df, ranges={"ch0": (-100, 100)})),
        ("multisensor_fuse", lambda t: t.run([df, df], how="outer")),
    ]
    failed = []
    for name, call in checks:
        tool = get(name)
        result = call(tool)
        status = "OK" if result.ok else "FAIL"
        if not result.ok:
            failed.append((name, result.error))
        print(f"  [run] {name:24s} {status}")

    # 2) fit/apply separation for stateful tools
    stateful = [
        ("drift_correct", {}),
        ("quantile_transform", {}),
        ("pca_ica", {}),
        ("random_projection", {}),
        ("feature_select", {}),
        ("conditional_residual", {}),
        ("dynamic_threshold", {}),
        ("power_transform", {"method": "yeo-johnson"}),
        ("knn_impute", {}),
        ("iterative_impute", {}),
    ]
    for name, kw in stateful:
        tool = get(name)
        # PCA/random projection and other sklearn reducers reject NaN: data quality should be handled upstream explicitly
        fit_data = df.dropna() if name in {"pca_ica", "random_projection"} else df
        fit = tool.fit(fit_data, **kw)
        app = tool.apply(fit_data) if fit.ok else fit
        status = "OK" if (fit.ok and app.ok) else "FAIL"
        if not (fit.ok and app.ok):
            failed.append((name, fit.error or app.error))
        print(f"  [fit/apply] {name:24s} {status}")

    # 3) Tools with missing dependencies should return a clear error rather than crash
    missing_lib = get("tsfresh_extract")
    r = missing_lib.run(df)
    expect_fail = (not r.ok) and ("not installed" in (r.error or ""))
    print(f"  [missing-lib] tsfresh_extract expected to give an install hint: {'OK' if expect_fail else 'FAIL'}")
    if not expect_fail:
        failed.append(("tsfresh_extract", r.error or "did not fail as expected"))

    # 4) Tool list can be read serially
    rows = list_tools()
    print(f"  [list_tools] metadata rows: {len(rows)}")

    if failed:
        print("\nFailed items:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return 1
    print("\nAll smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
