"""TSFLEX: time series segmentation, sliding window, and feature extraction orchestration.

Suitable as an entry point for many types of preprocessing tasks: supports sliding windows,
varying stride, and flexible feature computation, and can work with feature calculators
such as tsfresh / tsfel / statsmodels. It orchestrates the workflow; it is not a detector.
"""

from __future__ import annotations

from ...core.base import BaseTool, ToolResult, require
from ...core.utils import to_frame


class TsflexFeatureTool(BaseTool):
    name = "tsflex_feature"
    category = "Time Series Processing and Feature Extraction"
    version = "1.0.0"
    description = "Organize segments by sliding window/stride and orchestrate multiple feature calculators to extract window-level features."
    libraries = ("tsflex",)

    def run(self, data, functions=("mean", "std"), window_size=10, stride=5) -> ToolResult:
        params = {"functions": list(functions), "window_size": window_size, "stride": stride}
        try:
            require(list(self.libraries))
            import numpy as np
            from tsflex.features import (
                FeatureCollection,
                FuncWrapper,
                MultipleFeatureDescriptors,
            )
            from tsflex.features.segmenter import StridedRolling
            from tsflex.processing import SeriesProcessor

            df = to_frame(data)
            func_map = {
                "mean": np.mean, "std": np.std, "min": np.min, "max": np.max,
                "median": np.median, "sum": np.sum, "var": np.var,
            }
            missing = [f for f in functions if f not in func_map]
            if missing:
                return self.fail(ValueError(f"Unsupported feature calculators: {missing}"), params)

            funcs = [FuncWrapper(func_map[f]) for f in functions]
            sp = SeriesProcessor(
                series_window=StridedRolling(window_size=window_size, stride=stride)
            )
            fc = FeatureCollection(
                MultipleFeatureDescriptors(
                    functions=funcs,
                    series_names=list(df.columns),
                    windows=window_size,
                )
            )
            out = fc.calculate(df, return_df=True, window_sp_func=sp)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = TsflexFeatureTool()
