"""TSFRESH: automated time series feature extraction.

Batch-extract statistical, correlation, autoregressive, and spectral features from
time series, and support feature filtering. Focused on feature generation and filtering,
it usually needs to be combined with cleaning, imputation, and scaling.
"""

from __future__ import annotations

from ...core.base import BaseTool, ToolResult, require
from ...core.utils import to_frame


class TsfreshExtractTool(BaseTool):
    name = "tsfresh_extract"
    category = "Time Series Processing and Feature Extraction"
    version = "1.0.0"
    description = "Batch-extract statistical / correlation / autoregressive / spectral features from wide-format time series."
    libraries = ("tsfresh",)

    def run(self, data, fc_parameters=None) -> ToolResult:
        params = {"fc_parameters": fc_parameters}
        try:
            require(list(self.libraries))
            from tsfresh import extract_features

            df = to_frame(data)
            df = df.reset_index()
            out = extract_features(
                df, column_id="index", column_sort="index",  # treat wide table as a single long sequence
                default_fc_parameters=fc_parameters,
            )
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = TsfreshExtractTool()
