# rs_router.mean_fuse

**Stage**: Score fusion **Purpose**: Mean fusion, unifying the scales of scores from different experts/variables before fusing them, producing comparable final scores in [0,1].

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `ts_router.py:42-49` → `mean_fuse(score_columns)`
- Reuse method: logic-equivalent port (line-for-line correspondence)

## Migration Benefits
- Fully preserves the three steps: ① length consistency check; ② non-finite score check; ③ column-wise mean after `StandardScaler().fit_transform`, then `minmax` to [0,1].
- Shares the same `minmax` logic as the original implementation, keeping the output range and fusion semantics identical.

## Usage
```python
tool = get("rs_router.mean_fuse")
r = tool.run([scores_a, scores_b, scores_c])   # r.data is the fused [0,1] score
```
