# rs_router.train_test_split

**Stage**: Data splitting **Purpose**: Divide into a reference prefix (for fitting) and a to-be-scored suffix (for outputting scores).

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `ts_router.py:140-141` → inside `predict()`, `train = data[:train_index]` and `test = data[train_index:]`
- Reuse method: logic-equivalent port

## Migration Benefits
- Preserves the prefix/suffix semantics (chronological split, no random splitting, avoiding future information leakage).
- Preserves the `0 < train_index < len(data)` boundary check.

## Usage
```python
tool = get("rs_router.train_test_split")
r = tool.run(data, train_index=500)   # r.data == {"train": ..., "test": ...}
```
