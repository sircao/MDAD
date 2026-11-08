# mts_generation.convert_legacy_sample

**Stage**: label structure unification **Role**: legacy data format conversion that unifies various low-level outputs into the QA sample format, recovering per-channel labels while preserving the original information.

## Reuse source
- Original project: `Multi-AXIS-B1DC/MTS_Data`
- Source location: `src/mvaxis/legacy_adapter.py:343` → `convert_legacy_sample(sample, sample_id, base_sample_id)`
- Reuse method: **directly import the original module**

## Migration benefits
- Zero porting drift: fully preserves all the original logic, including legacy label matrix recovery, inference of anomaly type / root cause / affected channels, causal edge parsing, `target_output` fact card, and `evidence_card` generation.

## Usage
```python
tool = get("mts_generation.convert_legacy_sample")
r = tool.run(legacy_row, sample_id="s_00001")   # r.data is a MultivariateAnomalyQASample
```
