"""Spectral statistical features.

Dominant frequency, spectral centroid, spectral bandwidth, spectral flatness, spectral peak, and
band energy can represent periodic changes and spectral structure changes. (Native FFT
implementation; tsfel also provides similar features.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class SpectralFeaturesTool(BaseTool):
    name = "spectral_features"
    category = "Feature Extraction and Feature Representation"
    version = "1.0.0"
    description = "Extract spectral features such as dominant frequency/spectral centroid/spectral bandwidth/spectral flatness/band energy."
    libraries = ("numpy",)

    def _features(self, s: np.ndarray, fs: float, n_bands: int) -> dict:
        s = np.asarray(s, dtype=float)
        spectrum = np.abs(np.fft.rfft(s))
        freqs = np.fft.rfftfreq(len(s), d=1.0 / fs)
        total = spectrum.sum() + 1e-12
        centroid = float((freqs * spectrum).sum() / total)
        bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * spectrum).sum() / total))
        flatness = float(np.exp(np.mean(np.log(spectrum + 1e-12))) / (spectrum.mean() + 1e-12))
        dominant = float(freqs[int(np.argmax(spectrum))])
        bands = {}
        edges = np.linspace(0, freqs[-1], n_bands + 1)
        for i in range(n_bands):
            sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
            bands[f"band_energy_{i}"] = float(spectrum[sel].sum())
        return {
            "dominant_freq": dominant,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_flatness": flatness,
            "spectral_peak": float(spectrum.max()),
            **bands,
        }

    def run(self, data, fs=1.0, n_bands=4) -> ToolResult:
        params = {"fs": fs, "n_bands": n_bands}
        try:
            df = to_frame(data)
            rows = {
                col: self._features(df[col].to_numpy(dtype=float), fs, n_bands)
                for col in df.columns
            }
            out = pd.DataFrame(rows).T
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = SpectralFeaturesTool()
