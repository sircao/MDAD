"""MTS candidate window sampling — window sampling script."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MTS_SRC, ensure


class SampleWindowsTool(BaseTool):
    name = "mts_generation.sample_windows"
    category = "MTS dataset synthesis"
    version = "1.0.0"
    description = "Slices long series into target windows suitable for question generation, controlling the positive/negative sample ratio."
    libraries = ("numpy", "matplotlib")
    origin = "MTS_Data/src/mvaxis/question_provider.py:759 sample_analysis_windows()"

    def run(
        self,
        sample,
        seed=531,
        samples_per_series=None,
        min_window_size=None,
        max_window_size=None,
        anomaly_ratio=None,
    ) -> ToolResult:
        params = {
            "seed": seed, "samples_per_series": samples_per_series,
            "min_window_size": min_window_size, "max_window_size": max_window_size,
            "anomaly_ratio": anomaly_ratio,
        }
        try:
            ensure(MTS_SRC)
            from mvaxis import question_provider as qp

            windows = qp.sample_analysis_windows(
                sample,
                seed=int(seed),
                samples_per_series=samples_per_series or qp.SAMPLES_PER_SERIES,
                min_window_size=min_window_size or qp.MIN_WINDOW_SIZE,
                max_window_size=max_window_size or qp.MAX_WINDOW_SIZE,
                anomaly_ratio=anomaly_ratio if anomaly_ratio is not None else qp.ANOMALY_RATIO,
            )
            return self.ok(windows, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = SampleWindowsTool()
