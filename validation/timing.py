#!/usr/bin/env python3
"""Server-side computation time of run_multi_irar for all ten configurations
(median of five runs after one warm-up run), with 1,000 and 10,000 bootstrap
resamples for Krippendorff's alpha. Computation only; page rendering excluded.

Usage: python3 timing.py <repo_dir> <out_json>"""
import sys, json, time, statistics, platform, os
repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from utils.file_loader import load_sample_file
from modules.reliability import run_multi_irar

TWO, FOUR = ["R1", "R2"], ["R1", "R2", "R3", "R4"]
SC = {
    "S1":  ("Dichotomous",     TWO,  "dichotomous", 2,    None),
    "S2":  ("Dichotomous",     FOUR, "dichotomous", 2,    None),
    "S3":  ("Nominal-Ordinal", TWO,  "nominal",     5,    None),
    "S4":  ("Nominal-Ordinal", FOUR, "nominal",     5,    None),
    "S5":  ("Nominal-Ordinal", TWO,  "ordinal",     5,    1),
    "S6":  ("Nominal-Ordinal", FOUR, "ordinal",     5,    1),
    "S7":  ("Interval-Ratio",  TWO,  "interval",    None, None),
    "S8":  ("Interval-Ratio",  FOUR, "interval",    None, None),
    "S9":  ("Interval-Ratio",  TWO,  "ratio",       None, None),
    "S10": ("Interval-Ratio",  FOUR, "ratio",       None, None),
}


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


res = {"python": platform.python_version(), "cpu": cpu_model(), "logical_cpus": os.cpu_count()}
for nb in (1000, 10000):
    for k, (f, cols, dt, q, smin) in SC.items():
        df = load_sample_file(f"{repo}/sample-instats_{f}.csv")
        g = {"name": k, "items": cols, "data_type": dt, "q_declared": q, "scale_min": smin}
        run_multi_irar(df, [g], n_boot=nb)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            run_multi_irar(df, [g], n_boot=nb)
            ts.append(time.perf_counter() - t0)
        res[f"{k}_B{nb}"] = statistics.median(ts)
        print(k, nb, round(res[f"{k}_B{nb}"], 3), flush=True)
json.dump(res, open(out, "w"), indent=1)
