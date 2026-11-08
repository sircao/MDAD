"""MTS question stem generation — LLM question generation."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MTS_SRC, ensure


class GenerateQuestionTool(BaseTool):
    name = "mts_generation.generate_question"
    category = "MTS dataset synthesis"
    version = "1.0.0"
    description = "Generates diverse natural-language questions (MC/TF/OE) and writes them back into the sample."
    libraries = ("numpy", "matplotlib")
    origin = "MTS_Data/src/mvaxis/question_provider.py:1270 generate_question_spec_with_visual_workflow() + :984 assign_question()"

    def run(self, sample, client, answer_type="MC", seed=602, index=1, question_bank="regular") -> ToolResult:
        params = {"answer_type": answer_type, "seed": seed, "index": index, "question_bank": question_bank}
        try:
            ensure(MTS_SRC)
            from mvaxis.question_provider import (
                assign_question,
                generate_question_spec_with_visual_workflow,
                question_frame_for_sample,
            )

            frame = question_frame_for_sample(sample)
            spec = generate_question_spec_with_visual_workflow(
                sample,
                client,
                difficulty=frame,
                answer_type=answer_type,
                seed=int(seed),
                index=int(index),
                question_bank=question_bank,
            )
            if spec is None:
                return self.fail(ValueError("The LLM did not produce a valid question"), params)
            qa_row = assign_question(
                sample, spec, index=None, copy_sample=True, preserve_source_question=True
            )
            return self.ok(qa_row, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = GenerateQuestionTool()
