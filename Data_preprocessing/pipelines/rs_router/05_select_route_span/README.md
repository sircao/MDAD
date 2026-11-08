# rs_router.select_route_span

**Stage**: Route span selection **Purpose**: Control the range and size of the data output to the encoder/router, balancing efficiency and consistency.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `ts_router.py:124-130` → the `route_span` selection logic inside `predict()`
- Reuse method: logic-equivalent port

## Migration Benefits
- Fully preserves the three span semantics: univariate uses the full sequence, multivariate uses the test suffix capped at the first 10000 points, and `test` uses the test suffix.
- Preserves the `route_span ∈ {original, full, test}` validity check.

## Usage
```python
tool = get("rs_router.select_route_span")
r = tool.run(data, train_index=500, route_span="original")   # r.data is the routing sequence
```
