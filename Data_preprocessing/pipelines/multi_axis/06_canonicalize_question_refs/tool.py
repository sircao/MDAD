"""Multi-AXIS question channel reference canonicalization — unify question channel references."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MULTI_AXIS_SRC, ensure


class CanonicalizeQuestionRefsTool(BaseTool):
    name = "multi_axis.canonicalize_question_refs"
    category = "Multi-AXIS"
    version = "1.0.0"
    description = "Unify the numeric channel anchors in the question to ch_<id>, so the question seen by the LLM shares the same protocol as the numeric window."
    libraries = ()
    origin = "Multi-AXIS-B1DC/src/multi_axis/channel_ids.py:53-62 canonicalize_question_channel_references()"

    def run(self, question) -> ToolResult:
        params = {}
        try:
            ensure(MULTI_AXIS_SRC)
            from multi_axis.channel_ids import canonicalize_question_channel_references

            return self.ok(canonicalize_question_channel_references(question), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = CanonicalizeQuestionRefsTool()
