# rs_router.encoder_zscore

**Stage**: Normalization **Purpose**: Encoder input standardization, normalizing the series before feeding it into the time-series foundation model to align the representation with the pretraining distribution.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `ts_router.py:99` → inside `embed()`, `normalized = zscore(series, zero_std=1e-8)`
- Reuse method: logic-equivalent port

## Migration Benefits
- Preserves the tiny `zero_std=1e-8` fallback (closer to "near-zero-variance columns stay almost unchanged" than the generic path's 1.0), exactly matching the original encoder input.
- Preserves the `float32` input convention.

## Usage
```python
tool = get("rs_router.encoder_zscore")
r = tool.run(series)              # r.data is the normalized 1-D series
```
