"""MTS label structure unification — legacy data format conversion."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MTS_SRC, ensure


class ConvertLegacySampleTool(BaseTool):
    name = "mts_generation.convert_legacy_sample"
    category = "MTS dataset synthesis"
    version = "1.0.0"
    description = "Unifies various low-level outputs into the QA sample format, recovering per-channel labels while preserving the original information."
    libraries = ("numpy",)
    origin = "MTS_Data/src/mvaxis/legacy_adapter.py:343 convert_legacy_sample()"

    def run(self, sample, sample_id, base_sample_id=None) -> ToolResult:
        params = {"sample_id": sample_id, "base_sample_id": base_sample_id}
        try:
            ensure(MTS_SRC)
            from mvaxis.legacy_adapter import convert_legacy_sample

            return self.ok(convert_legacy_sample(sample, sample_id, base_sample_id=base_sample_id), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ConvertLegacySampleTool()
