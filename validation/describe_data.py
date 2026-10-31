#!/usr/bin/env python3
"""Description and fingerprint of the three sample data files from which all
ten configurations S1-S10 are drawn: MD5 hash, number of units and raters,
missing ratings, and the distribution of each rater's ratings.

Usage: python3 describe_data.py <repo_dir> <out_json>"""
import sys, json, hashlib
import numpy as np, pandas as pd

repo, out = sys.argv[1], sys.argv[2]
FILES = ["sample-instats_Dichotomous.csv", "sample-instats_Nominal-Ordinal.csv",
         "sample-instats_Interval-Ratio.csv"]
res = {}
for f in FILES:
    path = f"{repo}/{f}"
    raw = open(path, "rb").read()
    df = pd.read_csv(path)
    raters = [c for c in df.columns if c.startswith("R")]
    X = df[raters].to_numpy(float)
    d = dict(md5=hashlib.md5(raw).hexdigest(), bytes=len(raw), columns=list(df.columns),
             raters=raters, n_units=int(len(df)), n_ratings=int(X.size),
             missing=int(np.isnan(X).sum()), values_all=sorted(float(v) for v in np.unique(X[~np.isnan(X)])),
             per_rater={})
    for r in raters:
        v = df[r].to_numpy(float)
        info = dict(min=float(np.nanmin(v)), max=float(np.nanmax(v)), mean=float(np.nanmean(v)),
                    sd=float(np.nanstd(v, ddof=1)), distinct=int(len(np.unique(v[~np.isnan(v)]))))
        if len(np.unique(v)) <= 10:
            info["counts"] = {str(int(c)): int(np.sum(v == c)) for c in np.unique(v)}
        d["per_rater"][r] = info
    res[f] = d
json.dump(res, open(out, "w"), indent=1)
for f, d in res.items():
    print(f, d["md5"], d["n_units"], "units x", len(d["raters"]), "raters; missing", d["missing"],
          "; values", d["values_all"][:6], "...")
    for r, i in d["per_rater"].items():
        print("   ", r, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in i.items()})
