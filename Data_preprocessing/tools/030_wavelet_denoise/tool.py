"""Wavelet threshold denoising.

Apply soft or hard thresholding to wavelet coefficients and then reconstruct the denoised signal.
An overly large threshold may weaken local impulse anomalies.
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult, require
from ...core.utils import to_frame


class WaveletDenoiseTool(BaseTool):
    name = "wavelet_denoise"
    category = "Filtering, Decomposition, and Time-Frequency Analysis"
    version = "1.0.0"
    description = "Wavelet soft/hard threshold denoising (automatic threshold estimated from the highest-frequency coefficients)."
    libraries = ("pywt", "numpy")

    def _denoise(self, values: np.ndarray, wavelet: str, mode: str) -> np.ndarray:
        import pywt

        coeffs = pywt.wavedec(values, wavelet)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(values)))
        coeffs[1:] = [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)[: len(values)]

    def run(self, data, wavelet="db4", mode="soft") -> ToolResult:
        params = {"wavelet": wavelet, "mode": mode}
        try:
            require(["pywt"])
            if mode not in ("soft", "hard"):
                return self.fail(ValueError(f"mode only supports soft / hard"), params)
            df = to_frame(data)
            out = df.apply(lambda s: self._denoise(s.to_numpy(dtype=float), wavelet, mode))
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = WaveletDenoiseTool()
