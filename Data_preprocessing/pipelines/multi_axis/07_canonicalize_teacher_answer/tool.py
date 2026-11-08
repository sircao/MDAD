"""Multi-AXIS teacher answer canonicalization — unify the teacher answer format."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MULTI_AXIS_SRC, ensure


class CanonicalizeTeacherAnswerTool(BaseTool):
    name = "multi_axis.canonicalize_teacher_answer"
    category = "Multi-AXIS"
    version = "1.0.0"
    description = "Canonicalize teacher answers to align with the model output format, facilitating teacher forcing training and parsing."
    libraries = ()
    origin = "Multi-AXIS-B1DC/src/multi_axis/response_contracts.py:164 canonicalize_teacher_answer()"

    def run(self, answer, question_group, structured_is_anomalous=None) -> ToolResult:
        params = {"question_group": question_group, "structured_is_anomalous": structured_is_anomalous}
        try:
            ensure(MULTI_AXIS_SRC)
            from multi_axis.response_contracts import canonicalize_teacher_answer

            return self.ok(
                canonicalize_teacher_answer(
                    answer, question_group, structured_is_anomalous=structured_is_anomalous
                ),
                params,
            )
        except Exception as exc:
            return self.fail(exc, params)


TOOL = CanonicalizeTeacherAnswerTool()
