"""train_tuning.py —— Run training only on already-generated Tuning scores (staged, not waiting for Eva).

For Tuning files that already have both Syn + CHARM scores, compute morphology + VUS-PR per file and write
training_progress.json in real time (including each loop-round trace). Eva scores are filled in as they are generated.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from eval_fullshot_standard import (DETS, _vus, ev_morph_support, load_scores,
                                    rank01, run_loop_zero, tr_of)

DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"
TUNING_LIST = ROOT / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv"
TUNING_DIR = ROOT / "eval" / "score" / "uni" / "tuning"

from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402


def main() -> None:
    tun_files = pd.read_csv(TUNING_LIST)["file_name"].tolist()
    # only look at files that already have both scores
    ready = []
    for f in tun_files:
        stem = Path(f).stem
        if all((TUNING_DIR / d / f"{stem}.npy").exists() for d in DETS):
            ready.append(f)
    print(f"Tuning ready (both scores) {len(ready)}/{len(tun_files)}", flush=True)

    rows = []
    loop_rows = []
    for i, f in enumerate(ready, 1):
        data = pd.read_csv(DATA_DIR / f).to_numpy(dtype=float)
        tr = tr_of(f)
        labels = pd.read_csv(LABEL_DIR / f)["Label"].to_numpy(dtype=float)
        sw = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
        morph, support = ev_morph_support(data, tr)
        sc = {d: rank01(load_scores(f, d, TUNING_DIR)) for d in DETS}
        v = {d: _vus(sc[d], labels, sw) for d in DETS}
        rows.append({"file": f, "morph": morph, "Syn": round(v["Syntropic"], 4),
                     "CHARM": round(v["CHARM_kNN"], 4)})

        # simple rule: use the base with the higher VUS-PR for the current file, weight 0.5
        best = max(DETS, key=lambda d: v[d])
        other = DETS[1 - DETS.index(best)]
        base = sc[best]
        o = sc[other]
        fused_w = 0.5 * base + 0.5 * o

        def _wloop(trace):
            (ROOT / "loop_progress.json").write_text(json.dumps({
                "phase": "tuning", "file": f, "trace": trace, "updated": time.time(),
            }, ensure_ascii=False), encoding="utf-8")

        loop, trace = run_loop_zero(fused_w, o, support, labels, sw, write_cb=_wloop)
        loop_rows.append({"file": f, "morph": morph, "best": best, "w": 0.5,
                          "base_vus": round(v[best], 4),
                          "fused_vus": round(_vus(fused_w, labels, sw), 4),
                          "loop_vus": round(_vus(loop, labels, sw), 4),
                          "iters": len(trace), "trace": trace})

        (ROOT / "training_progress.json").write_text(json.dumps({
            "phase": "training", "done": i, "total": len(tun_files),
            "updated": time.time(), "rows": rows, "rule": {}, "loop_rows": loop_rows,
        }, ensure_ascii=False), encoding="utf-8")
        print(f"[{i}/{len(ready)}] {f} morph={morph} best={best} "
              f"Syn={v['Syntropic']:.3f} CHARM={v['CHARM_kNN']:.3f} loop_iters={len(trace)}", flush=True)

    print(f"done training {len(ready)} ready files", flush=True)


if __name__ == "__main__":
    main()
