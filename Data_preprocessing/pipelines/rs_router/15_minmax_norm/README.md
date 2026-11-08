# rs_router.minmax_norm

**Stage**: Score fusion **Purpose**: Min-max normalization, normalizing a single score to a unified range when taking the top-1.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `ts_router.py:34-39` → `minmax(scores)`
- Reuse method: logic-equivalent port (line-for-line correspondence)

## Migration Benefits
- Preserves the original implementation's key details: returns all zeros (rather than dividing by zero) when `high == low` (all scores equal); errors directly on non-finite scores.

## Usage
```python
tool = get("rs_router.minmax_norm")
r = tool.run(scores)               # r.data is the [0,1] normalized score
```
