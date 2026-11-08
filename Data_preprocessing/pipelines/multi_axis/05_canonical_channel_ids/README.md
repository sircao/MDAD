# multi_axis.canonical_channel_ids

**Stage**: channel id canonicalization **Purpose**: unify channel ids, unifying the channel anchors of the numeric/visual paths to avoid misaligned references.

## Reuse Source
- Original project: `Multi-AXIS-B1DC`
- Source location: `src/multi_axis/channel_ids.py:33-50` → `canonical_channel_ids()` (including `canonical_channel_id` single-id normalization)
- Reuse method: **direct import of the original module**

## Migration Advantages
- Zero porting deviation: preserve all rules such as `ch_<id>` canonicalization, fallback to `ch_<index>`, count matching, and uniqueness validation.

## Usage
```python
tool = get("multi_axis.canonical_channel_ids")
r = tool.run(raw_ids, channel_count=C)   # r.data is ["ch_0","ch_1",...]
```
