# multi_axis.extract_values

**Stage**: input decoding **Purpose**: extract numeric values, rejecting empty/malformed sequences or sequences containing non-numeric/infinite values, ensuring the subsequent normalization and tensorization remain usable.

## Reuse Source
- Original project: `Multi-AXIS-B1DC`
- Source location: `src/multi_axis/data.py:64-76` → `extract_values(row)`
- Reuse method: **direct import of the original module**

## Migration Advantages
- Zero porting deviation: call the original function, fully preserving `series.values` extraction, `[T,C]` non-empty validation, declared shape consistency validation, and `np.isfinite` full-finiteness validation.

## Usage
```python
tool = get("multi_axis.extract_values")
r = tool.run(row)                  # r.data is a float32 2-D array [T,C]
```
