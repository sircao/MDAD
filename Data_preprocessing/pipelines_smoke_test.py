"""Pipeline tool smoke test: verify registry discovery + direct import of original modules.

Usage: python Data_Reprocessing/pipelines_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Data_preprocessing.registry import by_category, discover, get  # noqa: E402


def ok_or_import_error(result) -> tuple[bool, str]:
    """ok=True passes; ok=False with a 'No module named' error counts as an import failure (fatal)."""
    if result.ok:
        return True, ""
    err = result.error or ""
    if "No module named" in err or "ModuleNotFoundError" in err:
        return False, err
    return True, err  # business/contract errors are treated as "original module imported successfully" in smoke tests


def main() -> int:
    tools = discover()
    print(f"Total tools: {len(tools)}")
    grouped = by_category()
    for cat in ("RS-TOUTER", "Multi-AXIS", "MTS dataset synthesis"):
        print(f"  {cat}: {len(grouped.get(cat, []))}")

    rng = np.random.default_rng(0)
    values = np.cumsum(rng.normal(0, 1, (40, 3)), axis=0)
    row = {"series": {"values": values.tolist(), "shape": [40, 3]}, "target_interval": {"start": 5, "end": 20}}

    # (name, call)
    checks = [
        ("rs_router.drop_missing", lambda t: t.run(pd.DataFrame({"a": [1.0, np.nan, 3.0]}))),
        ("rs_router.strip_labels", lambda t: t.run(pd.DataFrame({"a": [1.0, 2.0], "Label": [0, 1]}))),
        ("rs_router.to_float_array", lambda t: t.run([[1, 2], [3, 4]])),
        ("rs_router.train_test_split", lambda t: t.run(values, train_index=20)),
        ("rs_router.select_route_span", lambda t: t.run(values, train_index=20, route_span="test")),
        ("rs_router.split_zscore", lambda t: t.run(values[:20], values[20:])),
        ("rs_router.encoder_zscore", lambda t: t.run(values[:, 0])),
        ("rs_router.acf_period", lambda t: t.run(np.sin(np.linspace(0, 40, 400)))),
        ("rs_router.sliding_window", lambda t: t.run(values[:, 0], window=10)),
        ("rs_router.internal_standardizer", lambda t: (t.fit(values[:20]), t.apply(values[20:]))[-1]),
        ("rs_router.prune_zero_columns", lambda t: (t.fit(values), t.apply(values))[-1]),
        ("rs_router.validate_array", lambda t: t.run(values)),
        ("rs_router.add_const_noise", lambda t: t.run(np.column_stack([values[:, 0], np.ones(40)]))),
        ("rs_router.mean_fuse", lambda t: t.run([values[:20, 0], values[:20, 1], values[:20, 2]])),
        ("rs_router.minmax_norm", lambda t: t.run(values[:, 0])),
        ("rs_router.validate_scores", lambda t: t.run(values[:20, 0], expected_length=20)),
        ("multi_axis.extract_values", lambda t: t.run(row)),
        ("multi_axis.extract_interval", lambda t: t.run(row, steps=40)),
        ("multi_axis.channel_zscore", lambda t: t.run(values, interval=(5, 20))),
        ("multi_axis.serialize_window", lambda t: t.run(values, interval=(5, 20))),
        ("multi_axis.canonical_channel_ids", lambda t: t.run(["ch0", "channel1"], 2)),
        ("multi_axis.canonicalize_question_refs", lambda t: t.run("compare channel 3 with ch 5")),
        ("multi_axis.canonicalize_teacher_answer", lambda t: t.run("Answer: A", "MC", True)),
        ("multi_axis.image_identity", lambda t: t.run("s_00001", (5, 20))),
        ("mts_generation.attach_channel_scales", lambda t: t.run({"series": {"values": values.tolist()}, "channels": [{"channel_id": f"ch_{i}"} for i in range(3)]})),
    ]

    failed = []
    for name, call in checks:
        tool = get(name)
        try:
            result = call(tool)
            passed, err = ok_or_import_error(result)
        except Exception as exc:  # a directly raised exception also counts as a failure
            passed, err = False, f"{type(exc).__name__}: {exc}"
        status = "OK" if passed else "FAIL"
        if not passed:
            failed.append((name, err))
        note = f"  ({err})" if (not passed or (result and not result.ok)) else ""
        print(f"  {name:42s} {status}{note}")

    # Direct import verification: confirm that all three pipelines' original modules import after ensure()
    import_checks = [
        ("TS-Router-D8FF/utils.slidingWindows", "RS_ROUTER_ROOT"),
        ("TS-Router-D8FF/models.feature", "RS_ROUTER_ROOT"),
        ("TS-Router-D8FF/utils.utility", "RS_ROUTER_ROOT"),
        ("Multi-AXIS-B1DC/src/multi_axis.data", "MULTI_AXIS_SRC"),
        ("Multi-AXIS-B1DC/src/multi_axis.channel_ids", "MULTI_AXIS_SRC"),
        ("Multi-AXIS-B1DC/src/multi_axis.response_contracts", "MULTI_AXIS_SRC"),
        ("Multi-AXIS-B1DC/MTS_Data/src/mvaxis.data_schema", "MTS_SRC"),
        ("Multi-AXIS-B1DC/MTS_Data/src/mvaxis.legacy_adapter", "MTS_SRC"),
        ("Multi-AXIS-B1DC/MTS_Data/src/mvaxis.frame_builder", "MTS_SRC"),
        ("Multi-AXIS-B1DC/MTS_Data/src/mvaxis.question_provider", "MTS_SRC"),
        ("Multi-AXIS-B1DC/MTS_Data/src/mvaxis.teacher_answer_provider", "MTS_SRC"),
    ]
    from Data_preprocessing.core.origins import MTS_SRC, MULTI_AXIS_SRC, RS_ROUTER_ROOT, ensure

    roots = {"RS_ROUTER_ROOT": RS_ROUTER_ROOT, "MULTI_AXIS_SRC": MULTI_AXIS_SRC, "MTS_SRC": MTS_SRC}
    for label, root_key in import_checks:
        ensure(roots[root_key])
        module_name = label.split("/")[-1]
        try:
            __import__(module_name)
            print(f"  [import] {module_name:34s} OK")
        except Exception as exc:
            failed.append((module_name, f"{type(exc).__name__}: {exc}"))
            print(f"  [import] {module_name:34s} FAIL ({type(exc).__name__}: {exc})")

    if failed:
        print("\nFailed items:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return 1
    print("\nAll pipeline tool smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
