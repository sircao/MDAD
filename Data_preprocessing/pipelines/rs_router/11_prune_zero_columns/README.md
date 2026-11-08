# rs_router.prune_zero_columns

**Stage**: Model internals **Purpose**: Prune zero columns, preventing zero-variance feature columns from causing principal component decomposition to degrade or crash.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `models/PCA.py:249-263` → the `zero_pruning` branch inside `PCA.fit()`
- Reuse method: logic-equivalent port

## Migration Benefits
- Preserves the original logic: `non_zero_columns_ = np.any(X != 0, axis=0)`; if all are zero, one column is kept as a fallback (`np.ones`) to avoid a "0 feature columns" crash.
- `fit/apply` are separated: the mask is determined on the training set and then applied unchanged to new data.

## Usage
```python
tool = get("rs_router.prune_zero_columns")
tool.fit(train); r = tool.apply(test)   # r.data is the matrix after removing zero columns
```
