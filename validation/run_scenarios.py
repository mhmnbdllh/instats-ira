#!/usr/bin/env python3
"""Run the application's own engine (modules.reliability.run_multi_irar) on
all ten data-type x rater-count configurations, exactly as the web interface
does, and write every returned quantity to JSON -- including the agreement
(cross-)tables and the ICC F-test table, which are serialised rather than
dropped.

Configurations (file, raters, declared data type, declared q, scale minimum):
  S1  dichotomous, 2 raters      S2  dichotomous, 4 raters
  S3  nominal,     2 raters      S4  nominal,     4 raters
  S5  ordinal,     2 raters      S6  ordinal,     4 raters
  S7  interval,    2 raters      S8  interval,    4 raters
  S9  ratio,       2 raters      S10 ratio,       4 raters

Usage: python3 run_scenarios.py <repo_dir> <out_json> [alpha] [n_boot]"""
import sys, json, time
import numpy as np
import pandas as pd

repo, out = sys.argv[1], sys.argv[2]
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
n_boot = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
sys.path.insert(0, repo)
from utils.file_loader import load_sample_file
from modules.reliability import run_multi_irar

TWO, FOUR = ["R1", "R2"], ["R1", "R2", "R3", "R4"]
DICH, NOMORD, INTRAT = ("sample-instats_Dichotomous.csv", "sample-instats_Nominal-Ordinal.csv",
                        "sample-instats_Interval-Ratio.csv")
SC = {
    "S1":  (DICH,   TWO,  "dichotomous", 2,    None),
    "S2":  (DICH,   FOUR, "dichotomous", 2,    None),
    "S3":  (NOMORD, TWO,  "nominal",     5,    None),
    "S4":  (NOMORD, FOUR, "nominal",     5,    None),
    "S5":  (NOMORD, TWO,  "ordinal",     5,    1),
    "S6":  (NOMORD, FOUR, "ordinal",     5,    1),
    "S7":  (INTRAT, TWO,  "interval",    None, None),
    "S8":  (INTRAT, FOUR, "interval",    None, None),
    "S9":  (INTRAT, TWO,  "ratio",       None, None),
    "S10": (INTRAT, FOUR, "ratio",       None, None),
}


def clean(o):
    if isinstance(o, pd.DataFrame):
        d = o.copy()
        d = d.map(lambda v: v.tolist() if isinstance(v, np.ndarray) else v)
        return {"index": [clean(i) for i in d.index.tolist()],
                "columns": [clean(c) for c in d.columns.tolist()],
                "data": clean(d.to_numpy().tolist())}
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


if __name__ == "__main__":
    res = {"_settings": {"alpha": alpha, "n_boot": n_boot, "bootstrap_seed": 42}}
    for key, (f, cols, dt, q, smin) in SC.items():
        df = load_sample_file(f"{repo}/{f}")
        grp = {"name": key, "items": cols, "data_type": dt, "q_declared": q, "scale_min": smin}
        t0 = time.perf_counter()
        M = run_multi_irar(df, [grp], alpha=alpha, n_boot=n_boot)
        dt_s = time.perf_counter() - t0
        c = M["constructs"][0]
        c["_file"], c["_q_declared"], c["_scale_min"], c["_runtime_s"] = f, q, smin, dt_s
        res[key] = clean(c)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)
    for k, v in res.items():
        if k.startswith("_"):
            continue
        print(f"{k:4s} {v['data_type']:11s} r={v['k_raters']} n={v['n_units']} "
              f"methods={v['applicable_methods']}  {v['_runtime_s']:.2f}s")
