"""MTS channel scale computation — compute channel scales."""

from __future__ import annotations

import copy

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MTS_SRC, ensure


class AttachChannelScalesTool(BaseTool):
    name = "mts_generation.attach_channel_scales"
    category = "MTS dataset synthesis"
    version = "1.0.0"
    description = "Computes and records channel statistics for later normalization and data review."
    libraries = ("numpy",)
    origin = "MTS_Data/src/mvaxis/data_schema.py:64 attach_channel_scales()"

    def run(self, sample) -> ToolResult:
        params = {}
        try:
            ensure(MTS_SRC)
            from mvaxis.data_schema import attach_channel_scales

            out = copy.deepcopy(sample)
            attach_channel_scales(out)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = AttachChannelScalesTool()
