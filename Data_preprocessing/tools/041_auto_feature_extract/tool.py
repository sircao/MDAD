"""Automated feature extraction.

TSFRESH provides many time-series feature calculators; TSFLEX can flexibly organize
window, step, and feature-computation workflows. Feature selection is usually needed
after automated extraction. This tool tries tsfresh → tsfel in order, and falls back
to a set of native basic features when neither is installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class AutoFeatureExtractTool(BaseTool):
    name = "auto_feature_extract"
    category = "Feature Extraction and Feature Representation"
    version = "1.0.0"
    description = "Automated batch feature extraction (tsfresh/tsfel, with fallback to native basic features)."
    libraries = ("tsfresh", "tsfel")

    def _native(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = {}
        for col in df.columns:
            s = df[col].to_numpy(dtype=float)
            spectrum = np.abs(np.fft.rfft(s))
            feats[col] = {
                "mean": np.mean(s), "std": np.std(s), "min": np.min(s),
                "max": np.max(s), "median": np.median(s),
                "spectral_energy": float(np.sum(spectrum ** 2)),
            }
        return pd.DataFrame(feats).T

    def run(self, data) -> ToolResult:
        params = {}
        try:
            df = to_frame(data)
            warnings = []
            if importlib.util.find_spec("tsfresh") is not None:
                from tsfresh import extract_features

                long = df.reset_index()
                out = extract_features(long, column_id="index", column_sort="index")
                return self.ok(out, params)
            if importlib.util.find_spec("tsfel") is not None:
                import tsfel

                cfg = tsfel.get_features_by_domain("statistical")
                out = tsfel.time_series_features_extractor(cfg, df, fs=1.0)
                return self.ok(out, params)
            warnings.append("tsfresh/tsfel are not installed; fell back to native basic features; you can pip install tsfresh tsfel")
            return self.ok(self._native(df), params, warnings)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = AutoFeatureExtractTool()
