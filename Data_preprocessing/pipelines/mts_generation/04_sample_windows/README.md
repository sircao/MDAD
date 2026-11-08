# mts_generation.sample_windows

**Stage**: candidate window sampling **Role**: window sampling script that slices long series into target windows suitable for question generation, controlling the positive/negative sample ratio.

## Reuse source
- Original project: `Multi-AXIS-B1DC/MTS_Data`
- Source location: `src/mvaxis/question_provider.py:759` → `sample_analysis_windows()`
- Reuse method: **directly import the original module** (default parameters reuse the module constants `SAMPLES_PER_SERIES/MIN_WINDOW_SIZE/MAX_WINDOW_SIZE/ANOMALY_RATIO`)

## Migration benefits
- Zero porting drift: fully preserves the original logic, including seed-based window sampling, positive/negative ratio control, and window size lower/upper bounds.

## Usage
```python
tool = get("mts_generation.sample_windows")
r = tool.run(sample, seed=531)     # r.data is a list of windows (with start/end/has_anomaly/variance)
```
