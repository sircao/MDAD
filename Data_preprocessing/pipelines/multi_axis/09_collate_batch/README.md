# multi_axis.collate_batch

**Stage**: batch alignment **Purpose**: collate multi-axis sequences, turning variable-length sequences into tensors while using masks to prevent padding from being treated as real evidence.

## Reuse Source
- Original project: `Multi-AXIS-B1DC`
- Source location: `src/multi_axis/data.py:274-317` → `collate_multiaxis(samples)`
- Reuse method: **direct import of the original module**

## Migration Advantages
- Zero porting deviation: preserve the complete assembly with separate padding by `time_count/channel_count`, the `time_mask`/`channel_mask` boolean masks, and sample-level fields (question/answer/interval/channel_ids...).
- Samples must conform to the `ManifestDataset` output field structure (`normalized_series`, `time_count`, `channel_count`, etc.).

## Usage
```python
tool = get("multi_axis.collate_batch")
r = tool.run(samples)              # r.data is the collated dict (with time_mask/channel_mask)
```
