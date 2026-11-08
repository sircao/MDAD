# multi_axis.channel_zscore

**Stage**: normalization **Purpose**: per-channel standardization, eliminating dimensional differences so numeric and image inputs use the same normalization, with an invertible formula (original values can be recovered from the mean and std).

## Reuse Source
- Original project: `Multi-AXIS-B1DC`
- Source location: `src/multi_axis/data.py:130-148` → the mean/std part of `normalize_and_serialize()`
- Reuse method: **direct import of the original module**, taking the three values `(normalized, mean, std)`

## Migration Advantages
- Zero porting deviation: reuse the original function, preserving the invertible normalization formula `(x - mean)/(std + 1e-5)` and float64 intermediate precision.
- Return mean/std, supporting recovery of the original values via the original formula.

## Usage
```python
tool = get("multi_axis.channel_zscore")
r = tool.run(values, interval=(start, end))   # r.data == {"normalized", "channel_means", "channel_stds"}
```
