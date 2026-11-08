"""diag_low_vus.py — Explain why the VUS-PR of these 13 tuning files is so low.

Look at: sequence length/training split point, anomaly ratio, whether anomalies fall inside the training segment (before tr),
number and length of anomaly segments, and the deviation strength (z-score) of anomalies relative to the normal baseline."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "Dataset" / "TSB-AD-U"
LABEL_DIR = ROOT.parent / "Validation" / "TSB-AD-U"

FILES = [
    "323_UCR_id_21_HumanActivity_tr_48812_1st_128430.csv",
    "347_UCR_id_45_Sensor_tr_2851_1st_5365.csv",
    "174_MITDB_id_5_Medical_tr_50000_1st_539948.csv",
    "230_MGAB_id_6_Synthetic_tr_25000_1st_42441.csv",
    "429_UCR_id_127_Medical_tr_14825_1st_29000.csv",
    "161_Stock_id_13_Finance_tr_500_1st_3.csv",
    "033_WSD_id_5_WebService_tr_4559_1st_12588.csv",
    "053_WSD_id_25_WebService_tr_4559_1st_9198.csv",
    "045_WSD_id_17_WebService_tr_2566_1st_2666.csv",
    "312_UCR_id_10_Facility_tr_19948_1st_52000.csv",
    "217_LTDB_id_2_Medical_tr_500_1st_266.csv",
    "386_UCR_id_84_Environment_tr_2046_1st_5703.csv",
    "142_MSL_id_3_Sensor_tr_1525_1st_4575.csv",
]


def tr_of(name: str) -> int:
    return int(re.search(r"_tr_(\d+)_", name).group(1))


def segs(label):
    edges = np.diff(np.concatenate(([0], label, [0])).astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def main() -> None:
    print(f"{'file':45s} {'n':>7s} {'tr':>7s} {'lbl%':>6s} {'train%':>7s} {'test%':>6s} "
          f"{'seg':>4s} {'maxseg':>7s} {'anom|z|':>8s} {'norm|z|':>8s}")
    for name in FILES:
        x = pd.read_csv(DATA_DIR / name).to_numpy(dtype=float)
        y = pd.read_csv(LABEL_DIR / name)["Label"].to_numpy(dtype=int)
        n = len(x)
        tr = tr_of(name)
        v = x[:, 0]
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826
        mad = mad if mad > 1e-12 else float(np.std(v))
        z = np.abs(v - med) / max(mad, 1e-12)
        s = segs(y)
        anom_z = float(np.max(z[y == 1])) if y.any() else 0.0
        norm_z = float(np.median(z[y == 0])) if (y == 0).any() else 0.0
        print(f"{name:45s} {n:7d} {tr:7d} {y.mean():6.4f} {y[:tr].mean():7.4f} "
              f"{y[tr:].mean():6.4f} {len(s):4d} "
              f"{max((e - st for st, e in s), default=0):7d} {anom_z:8.2f} {norm_z:8.2f}")


if __name__ == "__main__":
    main()
