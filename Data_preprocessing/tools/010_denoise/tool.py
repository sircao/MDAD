"""Noise and artifact handling.

Includes median filtering, low-pass filtering, notch filtering, wavelet threshold
denoising, and local outlier handling, suited to sensor noise or known interference.
Excessive filtering strength may erase genuine anomalies.
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult, require
from ...core.utils import to_frame


class DenoiseTool(BaseTool):
    name = "denoise"
    category = "Data Cleaning and Data Quality"
    version = "1.0.0"
    description = "Median / low-pass / notch / wavelet threshold denoising, suited to sensor noise or known interference."
    libraries = ("scipy", "numpy")

    def run(self, data, method="median", **kwargs) -> ToolResult:
        params = {"method": method, **kwargs}
        try:
            require(["scipy", "numpy"])
            from scipy import ndimage, signal

            df = to_frame(data)
            if method == "median":
                size = kwargs.get("size", 3)
                out = df.apply(lambda s: ndimage.median_filter(s.to_numpy(), size=size))
            elif method == "lowpass":
                cutoff = kwargs.get("cutoff", 0.1)
                order = kwargs.get("order", 4)
                sos = signal.butter(order, cutoff, btype="low", output="sos")
                out = df.apply(lambda s: signal.sosfiltfilt(sos, s.to_numpy()))
            elif method == "notch":
                fs = kwargs.get("fs", 1.0)
                freq = kwargs.get("freq", 0.0)
                q = kwargs.get("q", 30)
                b, a = signal.iirnotch(freq, q, fs)
                out = df.apply(lambda s: signal.filtfilt(b, a, s.to_numpy()))
            elif method == "wavelet":
                require(["pywt"])
                import pywt

                wavelet = kwargs.get("wavelet", "db4")
                mode = kwargs.get("mode", "soft")
                out = df.apply(lambda s: _denoise_wavelet(s.to_numpy(), wavelet, mode))
            else:
                return self.fail(
                    ValueError(f"method only supports median / lowpass / notch / wavelet, got {method}"),
                    params,
                )
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


def _denoise_wavelet(values: np.ndarray, wavelet: str, mode: str) -> np.ndarray:
    import pywt

    coeffs = pywt.wavedec(values, wavelet)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(values)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[: len(values)]


TOOL = DenoiseTool()
