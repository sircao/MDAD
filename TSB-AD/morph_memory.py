"""morph_memory.py — Load Memory_base/morph_memory.json and aggregate it into a morphology routing prior for the complement mechanism.

The memory bank is a simulation-calibrated "shape → detector catch_rate/fa_rate" map (27 fine shapes).
Here the 27 fine shapes are mapped back to the 9 coarse morphologies (aligned with classify()), the degenerate detector "stuck" is excluded,
and it aggregates: the best assistant per coarse morphology + a data-driven assist_weight.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MEM = Path(__file__).resolve().parent.parent / "Memory_base" / "morph_memory.json"

# complement pool: exclude the degenerate detector "stuck", and exclude the base (Syntropic/CHARM_kNN are bases, not assistants)
COMPLEMENT_DETECTORS = ["rolling_volatility", "two_sided_volatility",
                        "two_sided_knn", "pyod_iforest", "pyod_knn"]

# memory-bank detector name -> (registry tool name, args)
DETECTOR_TO_TOOL = {
    "rolling_volatility": ("rolling_volatility", {"stat": "std", "window": 100}),
    "two_sided_volatility": ("two_sided_volatility", {"window": 100}),
    "two_sided_knn": ("two_sided_knn", {"n_neighbors": 5}),
    "pyod_iforest": ("pyod", {"algorithm": "iforest"}),
    "pyod_knn": ("pyod", {"algorithm": "knn"}),
}

# 27 fine shapes -> 9 coarse morphologies (aligned with morphology.classify)
SHAPE_TO_COARSE = {
    "outlier": "spike", "continuous upward spike": "spike", "continuous downward spike": "spike",
    "downward spike": "spike", "upward spike": "spike", "wide downward spike": "spike",
    "wide upward spike": "spike", "increase after upward spike": "spike",
    "increase after downward spike": "spike", "decrease after upward spike": "spike",
    "decrease after downward spike": "spike", "rapid rise followed by slow decline": "spike",
    "rapid decline followed by slow rise": "spike", "slow rise followed by rapid decline": "spike",
    "slow decline followed by rapid rise": "spike",
    "sudden decrease": "level_shift", "sudden increase": "level_shift", "shift_change": "level_shift",
    "amplitude_change": "variance_change", "amplitude_scaling": "variance_change",
    "waveform_change": "variance_change", "waveform_inversion": "variance_change",
    "shake": "variance_change",
    "upward convex": "trend", "downward convex": "trend", "phase_shift": "trend",
    "noise_injection": "noise",
}


def load():
    return json.load(open(MEM, encoding="utf-8"))


def aggregate():
    """coarse morphology -> {best_detector, assist_weight, base_catch, catch_by_detector}."""
    d = load()
    atlas, aw, bs = d["shape_atlas"], d["assist_weight"], d["base_strength"]

    coarse = {}
    for shape, morph in SHAPE_TO_COARSE.items():
        if shape not in atlas:
            continue
        coarse.setdefault(morph, []).append(shape)

    out = {}
    for morph, shapes in coarse.items():
        catch = {det: float(np.mean([atlas[s][det]["catch_rate"] for s in shapes
                                     if det in atlas[s]])) for det in COMPLEMENT_DETECTORS}
        best = max(catch, key=catch.get)
        w = float(np.mean([aw[s] for s in shapes if s in aw])) if any(s in aw for s in shapes) else 0.5
        base_catch = float(np.mean([bs[s]["catch_rate"] for s in shapes if s in bs])) \
            if any(s in bs for s in shapes) else None
        out[morph] = {"best_detector": best, "assist_weight": w,
                      "base_catch": base_catch, "catch": catch}
    return out


def route(morphology: str):
    """coarse morphology -> (tool_name, args, assist_weight). Unknown morphologies degrade to the global best."""
    agg = aggregate()
    if morphology in agg:
        a = agg[morphology]
    else:
        # degrade: the assistant with the highest global catch
        best = max(COMPLEMENT_DETECTORS,
                   key=lambda det: float(np.mean([v["catch"][det] for v in aggregate().values()])))
        a = {"best_detector": best, "assist_weight": 0.5, "catch": {}}
    tool, args = DETECTOR_TO_TOOL[a["best_detector"]]
    return tool, args, a["assist_weight"]


if __name__ == "__main__":
    agg = aggregate()
    print("=== coarse morphology -> assistant routing aggregated from the memory bank ===")
    for morph in sorted(agg):
        a = agg[morph]
        print(f'  {morph:16s} best={a["best_detector"]:20s} w={a["assist_weight"]:.2f} '
              f'base_catch={a["base_catch"] if a["base_catch"] is None else round(a["base_catch"],2)}')
        print(f'    catch: ' + '  '.join(f'{d.split("_")[-1]}={v:.2f}'
                                          for d, v in sorted(a["catch"].items(), key=lambda kv: -kv[1])))
