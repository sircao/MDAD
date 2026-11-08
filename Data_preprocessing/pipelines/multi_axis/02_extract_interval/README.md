# multi_axis.extract_interval

**Stage**: interval validation **Purpose**: extract the interval, ensuring the target window is valid and preventing out-of-bounds slicing.

## Reuse Source
- Original project: `Multi-AXIS-B1DC`
- Source location: `src/multi_axis/data.py:79-86` → `extract_interval(row, steps)`
- Reuse method: **direct import of the original module**

## Migration Advantages
- Zero porting deviation: preserve the `0 <= start < end <= steps` half-open interval validation and type conversion.

## Usage
```python
tool = get("multi_axis.extract_interval")
r = tool.run(row, steps=T)         # r.data == (start, end)
```
