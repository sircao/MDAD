"""MTS teacher answer generation — LLM answering."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MTS_SRC, ensure


class GenerateTeacherAnswerTool(BaseTool):
    name = "mts_generation.generate_teacher_answer"
    category = "MTS dataset synthesis"
    version = "1.0.0"
    description = "Builds teacher answer messages and optionally calls an LLM to produce standard answers for supervision."
    libraries = ("numpy",)
    origin = "MTS_Data/src/mvaxis/teacher_answer_provider.py:466 build_teacher_answer_messages() (prompt :431 build_teacher_answer_prompt())"

    def run(self, row, client=None, include_image=True, max_window_rows=None, digits=3) -> ToolResult:
        params = {"include_image": include_image, "max_window_rows": max_window_rows, "digits": digits}
        try:
            ensure(MTS_SRC)
            from mvaxis.teacher_answer_provider import build_teacher_answer_messages

            messages = build_teacher_answer_messages(
                row,
                include_image=include_image,
                max_window_rows=max_window_rows,
                digits=int(digits),
            )
            if client is None:
                return self.ok({"messages": messages, "answer": None}, params)
            response = client.complete(messages)
            return self.ok({"messages": messages, "answer": str(response.content or "")}, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = GenerateTeacherAnswerTool()
