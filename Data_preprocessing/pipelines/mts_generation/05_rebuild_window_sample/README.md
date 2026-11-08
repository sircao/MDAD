# mts_generation.rebuild_window_sample

**Stage**: view organization and expansion **Role**: window sample reconstruction that generates two questions, an "anomaly view" and a "normal view", from the same window, expanding question diversity without rewriting ground-truth labels.

## Reuse source
- Original project: `Multi-AXIS-B1DC/MTS_Data`
- Source location: `src/mvaxis/frame_builder.py:91` → `rebuild_window_sample(raw_row, window_row)`; batch entry point `frame_builder.py:159` → `build_question_frames()`
- Reuse method: **directly import the original module**

## Migration benefits
- Zero porting drift: fully preserves the field expansion from window metadata to QA sample (`base_sample_id`/`source_window_sample_id`/`target_interval`/`root_cause`/`fact_check`/`windows`, etc.) and the view recombination logic.

## Usage
```python
tool = get("mts_generation.rebuild_window_sample")
r = tool.run(raw_row, window_row)   # r.data is the expanded QA sample
```
