# multi_axis.canonicalize_teacher_answer

**Stage**: teacher answer canonicalization **Purpose**: unify the teacher answer format, aligning the supervision target with the model output format to facilitate teacher forcing training and parsing.

## Reuse Source
- Original project: `Multi-AXIS-B1DC`
- Source location: `src/multi_axis/response_contracts.py:164` → `canonicalize_teacher_answer(answer, question_group, structured_is_anomalous)`
- Reuse method: **direct import of the original module**

## Migration Advantages
- Zero porting deviation: fully preserve the teacher answer canonicalization rules for each output contract (MC/TF/OE), keeping the supervision target consistent with the model's parsing convention.

## Usage
```python
tool = get("multi_axis.canonicalize_teacher_answer")
r = tool.run(answer, question_group="MC", structured_is_anomalous=True)
```
