# mts_generation.generate_question

**Stage**: question stem generation **Role**: LLM question generation, producing diverse natural-language questions.

## Reuse source
- Original project: `Multi-AXIS-B1DC/MTS_Data`
- Source location: `src/mvaxis/question_provider.py:1270` → `generate_question_spec_with_visual_workflow()`; `question_provider.py:984` → `assign_question()`; view `question_provider.py:403` → `question_frame_for_sample()`
- Reuse method: **directly import the original module** (reproduces the call order of `04_generate_questions.py`)

## Migration benefits
- Zero porting drift: fully preserves the original call order of "get view → generate question spec → write back into sample", including the visual workflow and question bank (regular/hard) selection.
- `client` is passed in by the caller (can reuse `mvaxis.llm_client.create_llm_client`).

## Usage
```python
tool = get("mts_generation.generate_question")
r = tool.run(sample, client, answer_type="MC", question_bank="regular")
```
