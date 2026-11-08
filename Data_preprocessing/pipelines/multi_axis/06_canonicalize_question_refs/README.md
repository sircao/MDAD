# multi_axis.canonicalize_question_refs

**Stage**: question channel reference canonicalization **Purpose**: unify question channel references, so the question seen by the LLM uses the same channel protocol as the numeric window.

## Reuse Source
- Original project: `Multi-AXIS-B1DC`
- Source location: `src/multi_axis/channel_ids.py:53-62` → `canonicalize_question_channel_references()`
- Reuse method: **direct import of the original module**

## Migration Advantages
- Zero porting deviation: preserve the conservative semantics of "only rewriting numeric channel anchors while leaving the rest of the text byte-for-byte unchanged", avoiding accidental modification of other question content.

## Usage
```python
tool = get("multi_axis.canonicalize_question_refs")
r = tool.run(question)             # r.data is the canonicalized question
```
