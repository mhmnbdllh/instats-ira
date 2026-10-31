#!/usr/bin/env python3
"""Data-based Monte Carlo evaluation of every interval the application
reports, for all ten configurations S1-S10.

Each sample data set is treated as a finite population. For each replicate,
86 units are drawn with replacement as whole rows (so the rater structure of
every unit is kept), and the application's engine (run_multi_irar, alpha =
0.05, 1,000 bootstrap resamples for Krippendorff's alpha) is run exactly as in
the web interface. Configurations that analyse the same columns of the same
file (S3/S5, S4/S6, S7/S9, S8/S10) receive the same resampled units, so their
results differ only through the declared data type.

Recorded per coefficient: coverage of the normal-based and t-based intervals
(or of the single interval the coefficient has), i.e. the share of replicates
whose interval contains the coefficient's value on the full data set; and the
ratio of the mean reported standard error to the standard deviation of the
replicate estimates, for each standard error that is reported.

Usage: python3 coverage_sim.py <repo_dir> <out_json> <R> <seed>"""
import sys, json, time
import numpy as np
from multiprocessing import Pool

repo, out_json, R, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
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
FILES = ("Dichotomous", "Nominal-Ordinal", "Interval-Ratio")
DATA = {k: load_sample_file(f"{repo}/sample-instats_{k}.csv") for k in FILES}


def extract(c):
    """name -> (estimate, SE for the interval, SE under H0, lo_z, hi_z, lo_t, hi_t);
    entries that do not exist for a coefficient are None."""
    o = {}
    pa = c.get("percentage_agreement")
    if pa:
        o["Percentage agreement"] = (pa["pa"], None, None, pa["ci_lo"], pa["ci_hi"], None, None)
    ck = c.get("cohen_kappa")
    if ck:
        v = ck["variants"]["unweighted"]
        o["Cohen's kappa"] = (v["kappa"], v["se_general"], v["se_null"], v["ci_lo_z"], v["ci_hi_z"], v["ci_lo_t"], v["ci_hi_t"])
    wk = c.get("weighted_kappa")
    if wk:
        for s in ("unweighted", "linear", "quadratic"):
            v = wk["variants"][s]
            o[f"Weighted kappa, {s}"] = (v["kappa"], v["se_general"], v["se_null"], v["ci_lo_z"], v["ci_hi_z"],
                                         v["ci_lo_t"], v["ci_hi_t"])
    sp = c.get("scott_pi")
    if sp:
        o["Scott's pi"] = (sp["pi"], sp["se"], sp["se_null"], sp["ci_lo_z"], sp["ci_hi_z"], sp["ci_lo_t"], sp["ci_hi_t"])
    bp = c.get("brennan_prediger")
    if bp:
        o["Brennan-Prediger"] = (bp["kappa_bp"], bp["se"], None, bp["ci_lo_z"], bp["ci_hi_z"], bp["ci_lo_t"], bp["ci_hi_t"])
    fk = c.get("fleiss_kappa")
    if fk:
        o["Fleiss' kappa"] = (fk["kappa"], fk["kappa_se_general"], fk["kappa_se"], fk["ci_lo_z"], fk["ci_hi_z"],
                              fk["ci_lo_t"], fk["ci_hi_t"])
    a1 = c.get("gwet_ac1")
    if a1:
        o["Gwet's AC1"] = (a1["ac1"], a1["se"], None, a1["ci_lo_z"], a1["ci_hi_z"], a1["ci_lo_t"], a1["ci_hi_t"])
    for sch, r in ((c.get("gwet_ac2") or {}).get("schemes") or {}).items():
        o[f"Gwet's AC2, {sch}"] = (r["ac2"], r["se"], None, r["ci_lo_z"], r["ci_hi_z"], r["ci_lo_t"], r["ci_hi_t"])
    ic = c.get("icc")
    if ic and "error" not in ic:
        for f, nm in (("1_1", "ICC(1,1)"), ("2_1", "ICC(A,1)"), ("3_1", "ICC(C,1)"),
                      ("1_k", "ICC(1,k)"), ("2_k", "ICC(A,k)"), ("3_k", "ICC(C,k)")):
            o[nm] = (ic[f"icc_{f}"], None, None, ic[f"icc_{f}_ci_lo"], ic[f"icc_{f}_ci_hi"], None, None)
    cc = c.get("lin_ccc")
    if cc:
        o["Lin's CCC"] = (cc["ccc"], None, None, cc["ci_lo"], cc["ci_hi"], None, None)
    ka = c.get("krippendorff")
    if ka:
        o["Krippendorff's alpha"] = (ka["alpha"], None, None, ka["ci_lo"], ka["ci_hi"], None, None)
    return o


def analyse(sc, df):
    f, cols, dt, q, smin = SC[sc]
    g = {"name": sc, "items": cols, "data_type": dt, "q_declared": q, "scale_min": smin}
    return extract(run_multi_irar(df, [g], alpha=0.05, n_boot=1000)["constructs"][0])


def one(args):
    sc, r = args
    f, cols, *_ = SC[sc]
    full = DATA[f][cols]
    # the same resampled units for configurations on the same columns of the same file
    rng = np.random.default_rng([seed, r, FILES.index(f), len(cols)])
    idx = rng.integers(0, len(full), len(full))
    return sc, analyse(sc, full.iloc[idx].reset_index(drop=True))


def ok(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


if __name__ == "__main__":
    t0 = time.time()
    truth = {sc: analyse(sc, DATA[SC[sc][0]][SC[sc][1]]) for sc in SC}
    jobs = [(sc, r) for sc in SC for r in range(R)]
    with Pool(2) as pool:
        res = pool.map(one, jobs, chunksize=10)
    summ = {}
    for sc in SC:
        reps = [o for s, o in res if s == sc]
        summ[sc] = {}
        for name, tv in truth[sc].items():
            pop = tv[0]
            rr = [o[name] for o in reps if name in o and ok(o[name][0])]
            est = np.array([x[0] for x in rr])
            sd = est.std(ddof=1)
            covz = [x[3] <= pop <= x[4] for x in rr if ok(x[3]) and ok(x[4])]
            covt = [x[5] <= pop <= x[6] for x in rr if ok(x[5]) and ok(x[6])]
            seg = [x[1] for x in rr if ok(x[1])]
            se0 = [x[2] for x in rr if ok(x[2])]
            summ[sc][name] = dict(pop=float(pop), n_rep=len(rr), mean_est=float(est.mean()), sd_est=float(sd),
                                  n_interval=len(covz),
                                  cover_main=float(np.mean(covz)) if covz else None,
                                  cover_t=float(np.mean(covt)) if covt else None,
                                  se_ratio=float(np.mean(seg) / sd) if seg else None,
                                  se0_ratio=float(np.mean(se0) / sd) if se0 else None)
    json.dump(dict(R=R, seed=seed, n_boot=1000, alpha=0.05, seconds=time.time() - t0, summary=summ),
              open(out_json, "w"), indent=1)
    print("done in", round(time.time() - t0), "s")
