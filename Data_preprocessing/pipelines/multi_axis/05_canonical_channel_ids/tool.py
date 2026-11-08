"""Multi-AXIS channel id canonicalization — unify channel ids."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MULTI_AXIS_SRC, ensure


class CanonicalChannelIdsTool(BaseTool):
    name = "multi_axis.canonical_channel_ids"
    category = "Multi-AXIS"
    version = "1.0.0"
    description = "Unify the channel anchors of the numeric/visual paths to ch_<id>, avoiding misaligned references."
    libraries = ()
    origin = "Multi-AXIS-B1DC/src/multi_axis/channel_ids.py:33-50 canonical_channel_ids()"

    def run(self, values, channel_count) -> ToolResult:
        params = {"channel_count": channel_count}
        try:
            ensure(MULTI_AXIS_SRC)
            from multi_axis.channel_ids import canonical_channel_ids

            return self.ok(canonical_channel_ids(values, channel_count), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = CanonicalChannelIdsTool()
