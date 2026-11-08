# mts_generation.attach_channel_scales

**Stage**: channel scale computation **Role**: computes channel scales and records channel statistics for later normalization and data review.

## Reuse source
- Original project: `Multi-AXIS-B1DC/MTS_Data`
- Source location: `src/mvaxis/data_schema.py:64` → `attach_channel_scales(sample)`
- Reuse method: **directly import the original module** (deep-copies before attaching in place to avoid polluting the input)

## Migration benefits
- Zero porting drift: reuses the original function, preserving the computation and write positions of statistics such as channel mean / standard deviation / scale.

## Usage
```python
tool = get("mts_generation.attach_channel_scales")
r = tool.run(sample)               # r.data is the sample with channel scales attached
```
