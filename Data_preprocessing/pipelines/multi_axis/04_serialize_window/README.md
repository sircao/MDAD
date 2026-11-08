# multi_axis.serialize_window

**Stage**: numeric serialization **Purpose**: round to nearest integer, turning floating-point observations into integer tokens readable by the LLM, reducing text length and keeping channels aligned.

## Reuse Source
- Original project: `Multi-AXIS-B1DC`
- Source location: `src/multi_axis/data.py:140` → `np.rint(normalized[start:end].T * scale).astype(int64)`
- Reuse method: **direct import of the original module**, taking the single `serialized` value

## Migration Advantages
- Zero porting deviation: preserve the channel-major transpose, `scale=100` scaling, `np.rint` rounding, and shape consistency assertion.

## Usage
```python
tool = get("multi_axis.serialize_window")
r = tool.run(values, interval=(start, end))   # r.data is a channel-major integer window
```
