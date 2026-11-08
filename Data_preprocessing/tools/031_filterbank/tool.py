"""Filter bank and subband decomposition.

Split the signal into multiple frequency bands and compute energy, variance, or peak features per
band, useful for distinguishing low-frequency trends from high-frequency oscillations.
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult, require
from ...core.utils import to_frame


class FilterbankTool(BaseTool):
    name = "filterbank"
    category = "Filtering, Decomposition, and Time-Frequency Analysis"
    version = "1.0.0"
    description = "Split the signal into multiple frequency bands, returning subband signals or per-band energy/variance/peak."
    libraries = ("scipy", "numpy")

    def run(self, data, n_bands=4, fs=1.0, mode="signals", order=4) -> ToolResult:
        params = {"n_bands": n_bands, "fs": fs, "mode": mode, "order": order}
        try:
            require(["scipy"])
            from scipy import signal

            df = to_frame(data)
            nyq = fs / 2.0
            edges = np.linspace(0, nyq, n_bands + 1)
            result = {}
            for col in df.columns:
                s = df[col].to_numpy(dtype=float)
                bands = []
                for i in range(n_bands):
                    lo, hi = edges[i], edges[i + 1]
                    wn = [max(lo / nyq, 1e-6), min(hi / nyq, 1.0 - 1e-6)] if nyq > 0 else [0.0, 0.0]
                    sos = signal.butter(order, wn, btype="bandpass", output="sos")
                    bands.append(signal.sosfiltfilt(sos, s))
                if mode == "energy":
                    result[col] = [float(np.sum(b ** 2)) for b in bands]
                elif mode == "variance":
                    result[col] = [float(np.var(b)) for b in bands]
                elif mode == "peak":
                    result[col] = [float(np.max(np.abs(b))) for b in bands]
                else:
                    result[col] = bands
            return self.ok(result, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = FilterbankTool()
