# mts_generation.generate_series

**Stage**: series synthesis and anomaly injection **Role**: dataset synthesis and generation tool that produces low-level time-series data with controllable ground-truth labels.

## Reuse source
- Original project: `Multi-AXIS-B1DC/MTS_Data`
- Source location: `src/mvaxis/legacy_tsad_generator.py:96` → `load_legacy_tsad_generate_dataset()`; underlying `third_party/datasets_rcd/src/generate_dataset.py:274` → `generate_dataset()`
- Reuse method: **directly import the original module**

## Migration benefits
- Zero porting drift: directly calls the original loader and underlying generator, fully preserving Datasets-RCD's multivariate series synthesis, anomaly injection, attribute-set sampling, and reproducible seed semantics.

## Usage
```python
tool = get("mts_generation.generate_series")
r = tool.run(num_samples=100, seq_len=180, num_features=10)   # r.data is a list of legacy sample rows
```
