"""Time-series data augmentation.

Transforms such as noise addition, scaling, time warping, cropping, slicing, and
masking are used to expand samples during training. In anomaly detection, confirm that
augmentation does not change anomaly labels or the meaning of normal patterns. (Native
implementation; tsai provides similar augmentation.)
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class SeriesAugmentTool(BaseTool):
    name = "series_augment"
    category = "Data Augmentation, Sampling, and Post-processing"
    version = "1.0.0"
    description = "Augmentation transforms such as noise/scale/time-warp/crop/mask (for the training stage only)."
    libraries = ("numpy",)

    def _warp(self, s: np.ndarray, sigma: float) -> np.ndarray:
        n = len(s)
        t = np.arange(n, dtype=float)
        new_t = t + np.random.normal(0, sigma, n)
        new_t = np.clip(new_t, 0, n - 1)
        return np.interp(t, new_t, s)

    def run(self, data, method="noise", magnitude=0.1, seed=0) -> ToolResult:
        params = {"method": method, "magnitude": magnitude, "seed": seed}
        try:
            rng = np.random.default_rng(seed)
            df = to_frame(data)
            arr = df.to_numpy(dtype=float)
            if method == "noise":
                out = arr + rng.normal(0, magnitude * (arr.std(axis=0) + 1e-6), arr.shape)
            elif method == "scale":
                out = arr * rng.uniform(1 - magnitude, 1 + magnitude, arr.shape[1])
            elif method == "timewarp":
                out = np.column_stack([self._warp(arr[:, i], magnitude) for i in range(arr.shape[1])])
            elif method == "mask":
                out = arr.copy()
                n = len(arr)
                mask_len = max(1, int(magnitude * n))
                for i in range(arr.shape[1]):
                    start = int(rng.integers(0, max(1, n - mask_len)))
                    out[start:start + mask_len, i] = np.nan
            elif method == "crop":
                out = arr.copy()
                n = len(arr)
                keep = max(1, int((1 - magnitude) * n))
                start = int(rng.integers(0, n - keep + 1))
                out[:start] = np.nan
                out[start + keep:] = np.nan
            else:
                return self.fail(
                    ValueError(f"method only supports noise / scale / timewarp / mask / crop"), params
                )
            out_df = type(df)(out, index=df.index, columns=df.columns)
            return self.ok(out_df, params, ["For the training stage only; confirm that augmentation does not change anomaly-label semantics"])
        except Exception as exc:
            return self.fail(exc, params)


TOOL = SeriesAugmentTool()
