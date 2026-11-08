"""Peak and impulse features.

Peak count, peak height, peak width, peak spacing, crest factor, and impulse factor can represent
impulse, spike, or repeated pulse shapes. (Native implementation; tsfel also provides similar
features.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import BaseTool, ToolResult, require
from ...core.utils import to_frame


class PeakFeaturesTool(BaseTool):
    name = "peak_features"
    category = "Feature Extraction and Feature Representation"
    version = "1.0.0"
    description = "Extract pulse-shape features such as peak count/peak height/peak width/crest factor/impulse factor."
    libraries = ("scipy", "numpy")

    def _features(self, s: np.ndarray) -> dict:
        from scipy.signal import find_peaks

        s = np.asarray(s, dtype=float)
        peaks, props = find_peaks(s, width=1)
        rms = float(np.sqrt(np.mean(s ** 2))) if len(s) else 0.0
        mean_abs = float(np.mean(np.abs(s))) if len(s) else 0.0
        return {
            "n_peaks": int(len(peaks)),
            "mean_height": float(np.mean(s[peaks])) if len(peaks) else 0.0,
            "mean_width": float(np.mean(props["widths"])) if len(peaks) else 0.0,
            "mean_prominence": float(np.mean(props["prominences"])) if len(peaks) else 0.0,
            "crest_factor": float(np.max(np.abs(s)) / rms) if rms > 0 else 0.0,
            "impulse_factor": float(np.max(np.abs(s)) / mean_abs) if mean_abs > 0 else 0.0,
        }

    def run(self, data) -> ToolResult:
        params = {}
        try:
            require(["scipy"])
            df = to_frame(data)
            rows = {col: self._features(df[col].to_numpy(dtype=float)) for col in df.columns}
            out = pd.DataFrame(rows).T
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = PeakFeaturesTool()
