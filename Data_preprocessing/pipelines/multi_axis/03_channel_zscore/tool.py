"""Multi-AXIS normalization — per-channel standardization."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MULTI_AXIS_SRC, ensure


class ChannelZscoreTool(BaseTool):
    name = "multi_axis.channel_zscore"
    category = "Multi-AXIS"
    version = "1.0.0"
    description = "Per-channel zero-mean unit-variance standardization, with an invertible formula (original values can be recovered from the mean/std)."
    libraries = ("numpy", "torch")
    origin = "Multi-AXIS-B1DC/src/multi_axis/data.py:130-148 normalize_and_serialize() (mean/std part)"

    def run(self, values, interval, epsilon=1e-5, scale=100) -> ToolResult:
        params = {"interval": interval, "epsilon": epsilon, "scale": scale}
        try:
            ensure(MULTI_AXIS_SRC)
            from multi_axis.data import normalize_and_serialize

            normalized, _, means, stds = normalize_and_serialize(
                values, tuple(map(int, interval)), epsilon=epsilon, scale=scale
            )
            return self.ok({"normalized": normalized, "channel_means": means, "channel_stds": stds}, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ChannelZscoreTool()
