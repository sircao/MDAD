# time_rcd

**Category**: Zero-shot & general models　**Source**: https://github.com/thu-sail-lab/Time-RCD/

**Implementation**: **actually loads the Time-RCD encoder weights in the repo** (4 encoder shards from `TS-Router-D8FF/checkpoints` + `models/time_rcd/ts_encoder_bi_bias.py`), and outputs per-channel, pointwise representation-deviation scores for unseen target series (zero-shot, no task-level training or fine-tuning).

Note: the shards exported from the repo contain only the encoder parameters; the reconstruction head / anomaly head were not exported, so the L2 distance from a local representation to the channel median representation is used as the zero-shot anomaly score (representation deviation of foundation models is a common approach for zero-shot TSAD).

**Usage**:
```python
tool = get("time_rcd")
r = tool.run(data, checkpoint_dir=None)   # default TS-Router-D8FF/checkpoints
# r.data is a [T] or [T, C] pointwise anomaly score (larger = more anomalous)
```
