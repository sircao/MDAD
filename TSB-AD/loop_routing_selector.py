"""loop_routing_selector.py — Label-free per-file complement selector (no server, pure table lookup + features).

On routing's (68.71/53.94) per-file VUS-PR metric, build a label-free per-file selector:
use each file's fast morphology/statistical features to judge label-free "which of Syntropic / CHARM_kNN /
CHARM_ZS should handle this file", replacing routing's "fixed pick per source". Trained with leave-one-source CV.

Four lines (per-file equal-weight mean VUS-PR):
  routing (fixed per source) / always-CHARM_kNN / label-free selector (learned) / oracle (per-file max)

This is the per-file version of "fix wrong data → swap in an algorithm that works"; the per-timestamp (point-level) version needs per-point scores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent
LDR = ROOT / "benchmark_exp" / "leaderboard_results"

MODELS = ["Syntropic", "CHARM_kNN", "CHARM_ZS"]


def norm_file(name: str) -> str:
    return name[:-4] if name.endswith(".csv") else name


def source_of(f: str) -> str:
    return f.split("_")[1]


def load_tables():
    syn = {}
    df = pd.read_csv(LDR / "Uni_Syntropic.csv")
    for _, r in df.iterrows():
        syn[norm_file(str(r["file"]))] = float(r["VUS-PR"])
    charm = {}
    df = pd.read_csv(LDR / "CHARM_uni_mergedTable_VUS-PR.csv")
    for _, r in df.iterrows():
        f = norm_file(str(r["file"]))
        charm[f] = (float(r["CHARM_kNN"]), float(r["CHARM_ZS"]))
    return syn, charm


def fast_features(x: np.ndarray) -> dict:
    """Fully vectorized O(n)~O(n log n) morphology/statistical features (replacing the slow extract_features)."""
    x = np.asarray(x, dtype=float).ravel()
    x = np.nan_to_num(x, nan=0.0)
    n = len(x)
    s = float(np.std(x))
    eps = 1e-12
    f = {}

    f["log_len"] = float(np.log1p(n))
    f["std"] = s
    f["mean_abs"] = float(np.mean(np.abs(x)))
    f["max_over_std"] = float(np.max(np.abs(x))) / s if s > eps else 0.0

    if s > eps and n > 4:
        from scipy.stats import skew, kurtosis
        f["skew"] = float(skew(x))
        f["kurt"] = float(kurtosis(x))
        f["acf1"] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        t = np.arange(n, dtype=float)
        f["slope"] = float(np.polyfit(t, x, 1)[0]) * n / s
        f["cusum"] = float(np.max(np.abs(np.cumsum(x - np.mean(x))))) / (s * np.sqrt(n))
    else:
        f.update(skew=0.0, kurt=0.0, acf1=0.0, slope=0.0, cusum=0.0)

    # vectorized rolling std (window = n//20)
    w = max(5, n // 20)
    c = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x ** 2, 0, 0.0))
    rs = c[w:] - c[:-w]
    rs2 = c2[w:] - c2[:-w]
    var = rs2 / w - (rs / w) ** 2
    rstd = np.sqrt(np.clip(var, 0, None))
    f["roll_std_mean"] = float(np.mean(rstd)) / s if s > eps else 0.0
    f["roll_std_cv"] = float(np.std(rstd)) / (float(np.mean(rstd)) + eps)

    thr = 1e-3 * (s if s > eps else 1.0)
    f["flat_ratio"] = float(np.mean(np.abs(np.diff(x)) <= thr))
    f["zero_cross"] = float(np.mean(np.diff(np.sign(x)) != 0))
    f["extreme3"] = float(np.mean(np.abs(x) > 3 * s)) if s > eps else 0.0

    if n > 16:
        spec = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
        tail = spec[1:].sum()
        if tail > eps:
            half = max(1, len(spec) // 2)
            f["hfr"] = float(spec[half:].sum() / tail)
            f["spec_peak"] = float(np.max(spec[1:]) / (np.mean(spec[1:]) + eps))
        else:
            f["hfr"] = 0.0
            f["spec_peak"] = 0.0
    else:
        f["hfr"] = 0.0
        f["spec_peak"] = 0.0
    return f


FEAT_KEYS = ["log_len", "std", "mean_abs", "max_over_std", "skew", "kurt", "acf1",
             "slope", "cusum", "roll_std_mean", "roll_std_cv", "flat_ratio",
             "zero_cross", "extreme3", "hfr", "spec_peak"]


def build_features(files, cache=ROOT / "features_uni.npz"):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {f: z[f].item() for f in z.files}
    feat = {}
    for i, f in enumerate(files):
        df = pd.read_csv(f"Datasets/TSB-AD-U/{f}.csv", usecols=lambda c: c != "Label")
        x = df.iloc[:, 0].values.astype(float)
        feat[f] = fast_features(x)
        if (i + 1) % 50 == 0:
            print(f"  features {i+1}/{len(files)}", flush=True)
    np.savez_compressed(cache, **{f: feat[f] for f in files})
    return feat


def main():
    syn, charm = load_tables()
    picks = json.load(open(ROOT / "runs" / "llm_routing_combined.json", encoding="utf-8"))
    picks_u = picks["picks_u"]

    files = sorted(syn.keys())  # Syntropic covers all 23 sources / 350 files
    # per-file VUS-PR of the three models
    vus = {f: {m: 0.0 for m in MODELS} for f in files}
    for f in files:
        vus[f]["Syntropic"] = syn[f]
        vus[f]["CHARM_kNN"] = charm[f][0]
        vus[f]["CHARM_ZS"] = charm[f][1]

    print("computing per-file features ...", flush=True)
    feat = build_features(files)
    X = np.stack([[feat[f][k] for k in FEAT_KEYS] for f in files]).astype(np.float32)

    # three baseline lines (no features)
    route = {f: vus[f][picks_u[source_of(f)]] for f in files}
    always_charm = {f: vus[f]["CHARM_kNN"] for f in files}
    oracle = {f: max(vus[f].values()) for f in files}

    # leave-one-source label-free selector (logistic regression, 3 classes)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    srcs = sorted(set(source_of(f) for f in files))
    selector = {}
    for held in srcs:
        tr = [f for f in files if source_of(f) != held]
        te = [f for f in files if source_of(f) == held]
        y_tr = np.array([int(np.argmax([vus[f][m] for m in MODELS])) for f in tr])
        if len(np.unique(y_tr)) < 2:
            # only one winner in the training sources → degenerate to picking the mode
            pred = np.full(len(te), int(np.argmax(np.bincount(y_tr))))
            for f, p in zip(te, pred):
                selector[f] = MODELS[p]
            continue
        sc = StandardScaler().fit(X[[files.index(f) for f in tr]])
        Xtr = sc.transform(X[[files.index(f) for f in tr]])
        Xte = sc.transform(X[[files.index(f) for f in te]])
        clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
        clf.fit(Xtr, y_tr)
        pred = clf.predict(Xte)
        for f, p in zip(te, pred):
            selector[f] = MODELS[p]

    sel = {f: vus[f][selector[f]] for f in files}

    def report(d, name):
        agg = defaultdict(list)
        for f, v in d.items():
            agg[source_of(f)].append(v)
        mean = float(np.mean(list(d.values()))) * 100
        print(f"\n== {name} == overall {mean:6.2f}")
        for s in sorted(agg):
            print(f"  {s:14s} {np.mean(agg[s])*100:6.2f}")
        return mean

    print("\n" + "=" * 60)
    report(route, "routing (fixed per source)")
    report(always_charm, "always CHARM_kNN")
    report(sel, "label-free per-file selector (learned)")
    report(oracle, "oracle (per-file max)")

    r = float(np.mean(list(route.values()))) * 100
    a = float(np.mean(list(always_charm.values()))) * 100
    s = float(np.mean(list(sel.values()))) * 100
    o = float(np.mean(list(oracle.values()))) * 100
    print("\n" + "=" * 60)
    print(f"routing={r:.2f} | always_CHARM={a:.2f} | label-free selector={s:.2f} | oracle={o:.2f}")
    print(f"selector vs routing: {s-r:+.2f}pp ; selector vs oracle remaining {o-s:+.2f}pp (label-free ceiling)")


if __name__ == "__main__":
    main()
