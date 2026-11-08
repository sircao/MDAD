"""fast_vus.py — Vectorized VUS-PR (bit-for-bit identical to TSB-AD's official RangeAUC_volume_opt).

The official implementation is a triple Python loop (250 thresholds × windowSize windows × per segment), extremely slow on large files.
Here the "per-threshold pred / TP / N_labels / existence" are all numpy-vectorized, keeping only
the outermost window loop and per-segment constant terms; the result is identical to the official avg_ap_3d, at 10~100x speed.

Correspondence:
  fast_vus(score, label, windowSize) == get_metrics(score, label, slidingWindow=windowSize)["VUS-PR"]
"""
from __future__ import annotations

import numpy as np


def _range_convers(label: np.ndarray):
    """basic_metricor.range_convers_new: binary label → [(start, end)] segments."""
    label = np.asarray(label, dtype=int)
    starts = np.where(np.diff(label) == 1)[0] + 1
    ends = np.where(np.diff(label) == -1)[0]
    if len(ends):
        if not len(starts) or ends[0] < starts[0]:
            starts = np.concatenate([[0], starts])
    if len(starts):
        if not len(ends) or ends[-1] < starts[-1]:
            ends = np.concatenate([ends, [len(label) - 1]])
    return list(zip(starts.tolist(), ends.tolist()))


def _new_sequence(label: np.ndarray, seq, window: int):
    """basic_metricor.new_sequence: segments merged after extending by window//2."""
    a = max(seq[0][0] - window // 2, 0)
    out = []
    for i in range(len(seq) - 1):
        if seq[i][1] + window // 2 < seq[i + 1][0] - window // 2:
            out.append((a, seq[i][1] + window // 2))
            a = seq[i + 1][0] - window // 2
    out.append((a, min(seq[-1][1] + window // 2, len(label) - 1)))
    return out


def _sequencing(x: np.ndarray, L, window: int):
    """basic_metricor.sequencing: extend labels at both ends of anomaly segments with sqrt decay."""
    label = x.copy().astype(float)
    length = len(label)
    for s, e in L:
        x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
        label[x1] += np.sqrt(1.0 - (x1 - e) / window)
        x2 = np.arange(max(s - window // 2, 0), s)
        label[x2] += np.sqrt(1.0 - (s - x2) / window)
    return np.minimum(np.ones(length), label)


def _mask_of(segments, n):
    """segment list → 0/1 mask of length n."""
    m = np.zeros(n, dtype=bool)
    for s, e in segments:
        m[s:e + 1] = True
    return m


def fast_vus(score: np.ndarray, label: np.ndarray, windowSize: int, thre: int = 250) -> float:
    score = np.asarray(score, dtype=float).ravel()
    label = np.asarray(label, dtype=int)
    n = len(score)
    P = float(np.sum(label))

    seq = _range_convers(label)
    if not seq:
        return 0.0
    l_segs = _new_sequence(label, seq, windowSize)
    l_mask = _mask_of(l_segs, n)
    seq_mask = _mask_of(seq, n)

    # 250 thresholds (descending) and the precomputed pred matrix
    score_sorted = -np.sort(-score)
    idx = np.linspace(0, n - 1, thre).astype(int)
    thresholds = score_sorted[idx]
    preds = score[None, :] >= thresholds[:, None]           # (thre, n)
    N_pred = preds.sum(axis=1).astype(float)

    ap_sum = 0.0
    for window in range(windowSize + 1):
        labels_extended = _sequencing(label, seq, window)
        L = _new_sequence(labels_extended, seq, window)
        L_mask = _mask_of(L, n)

        # existence[k] = number of L segments hit by pred
        existence = np.zeros(thre, dtype=int)
        for s, e in L:
            seg_max = score[s:e + 1].max()
            existence += (seg_max >= thresholds).astype(int)
        existence_ratio = existence / len(L)

        # TP[k] = sum_p l_mask[p] * pred[k,p] * (1 if seq else labels_extended[p])
        w = np.where(seq_mask, 1.0, labels_extended)
        TP = preds @ (l_mask * w)                            # (thre,)

        # N_labels[k] = sum_p l_mask[p] * labels_mod[k,p]
        not_seq = ~seq_mask
        not_L = ~L_mask
        const = float((l_mask & seq_mask).sum()) \
            + float(np.sum(labels_extended[l_mask & not_L & not_seq]))
        vec2 = (l_mask & L_mask & not_seq).astype(float) * labels_extended
        N_labels = const + preds @ vec2                      # (thre,)

        P_new = (P + N_labels) / 2.0
        recall = np.minimum(TP / P_new, 1.0)
        TPR = recall * existence_ratio
        FPR = (N_pred - TP) / (n - P_new)
        Precision = TP / N_pred

        # official AP combination: tpr = [0, TPR_1..250, 1], prec = [1, Prec_1..250]
        tpr = np.concatenate([[0.0], TPR, [1.0]])
        prec = np.concatenate([[1.0], Precision])
        width_PR = tpr[1:-1] - tpr[:-2]
        height_PR = prec[1:]
        ap_sum += float(np.dot(width_PR, height_PR))

    return ap_sum / (windowSize + 1)
