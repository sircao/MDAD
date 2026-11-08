# mts_generation.generate_teacher_answer

**Stage**: teacher answer generation **Role**: LLM answering, producing standard answers for supervised training.

## Reuse source
- Original project: `Multi-AXIS-B1DC/MTS_Data`
- Source location: `src/mvaxis/teacher_answer_provider.py:466` → `build_teacher_answer_messages()`; prompt `teacher_answer_provider.py:431` → `build_teacher_answer_prompt()`
- Reuse method: **directly import the original module**

## Migration benefits
- Zero porting drift: reuses the original message/prompt construction, fully preserving the parameter semantics including images, `max_window_rows`, and `digits`.
- When `client=None`, only messages are returned (preview mode), matching `05_generate_answers.py`'s `--preview-only`.

## Usage
```python
tool = get("mts_generation.generate_teacher_answer")
r = tool.run(row, client, include_image=True, digits=3)   # r.data == {"messages", "answer"}
```
