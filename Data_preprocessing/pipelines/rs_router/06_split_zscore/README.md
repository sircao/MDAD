# rs_router.split_zscore

**Stage**: Normalization **Purpose**: Split standardization, removing scale differences; the split form prevents test-set statistics from leaking into the training-set fit.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `ts_router.py:28-31` → `zscore()`; `ts_router.py:146` → inside `predict()`, `train, test = zscore(train), zscore(test)`
- Reuse method: logic-equivalent port (line-for-line correspondence to the original `zscore` function)

## Migration Benefits
- Fully preserves the original `zscore` semantics: column-wise (axis=0) zero mean and unit variance, with `zero_std` as a fallback for constant columns where `std==0` to avoid division by zero.
- Preserves the "train/test standardized independently" separation, matching the original inference path.

## Usage
```python
tool = get("rs_router.split_zscore")
r = tool.run(train, test)          # r.data == {"train_norm":..., "test_norm":...}
```
