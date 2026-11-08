"""TSFEL: multi-domain time series feature extraction.

Provides multiple kinds of feature extraction — time-domain, spectral, and
time-frequency-domain — for building window-level features.
"""

from __future__ import annotations

from ...core.base import BaseTool, ToolResult, require
from ...core.utils import to_frame


class TsfelExtractTool(BaseTool):
    name = "tsfel_extract"
    category = "Time Series Processing and Feature Extraction"
    version = "1.0.0"
    description = "Extract window-level time-domain / spectral / time-frequency-domain features (per the specified domain)."
    libraries = ("tsfel",)

    def run(self, data, domain="statistical", fs=1.0) -> ToolResult:
        params = {"domain": domain, "fs": fs}
        try:
            require(list(self.libraries))
            import tsfel

            df = to_frame(data)
            cfg = tsfel.get_features_by_domain(domain)
            out = tsfel.time_series_features_extractor(cfg, df, fs=fs)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = TsfelExtractTool()
