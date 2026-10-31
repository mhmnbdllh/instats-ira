#!/usr/bin/env python3
"""Independent verification of every quantity that InStats-IrA reports, for
all ten data-type x rater-count configurations (S1-S10).

Reference values come from (a) independent third-party implementations
(scikit-learn, statsmodels, irrCAC, krippendorff, pingouin, SciPy) and
(b) formulas re-typed in this file from the primary sources. The reference
side never calls the application's own coefficient helpers; the application
side is obtained only through run_multi_irar, exactly as the web page calls it.

Usage: python3 validate_independent.py <repo_dir> <out_json>"""
import sys, json, itertools
import numpy as np, pandas as pd
from scipy import stats

repo, out_json = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from utils.file_loader import load_sample_file
from modules.reliability import run_multi_irar, _get_applicable_methods

from sklearn.metrics import cohen_kappa_score
import statsmodels.stats.inter_rater as smir
from irrCAC.raw import CAC
import krippendorff as kdp
import pingouin as pg

ALPHA, NBOOT = 0.05, 1000
TWO, FOUR = ["R1", "R2"], ["R1", "R2", "R3", "R4"]
FILES = {"Dichotomous": "sample-instats_Dichotomous.csv",
         "Nominal-Ordinal": "sample-instats_Nominal-Ordinal.csv",
         "Interval-Ratio": "sample-instats_Interval-Ratio.csv"}
D = {k: load_sample_file(f"{repo}/{v}") for k, v in FILES.items()}
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
KMETRIC = {"dichotomous": "nominal", "nominal": "nominal", "ordinal": "ordinal",
           "interval": "interval", "ratio": "ratio"}
DECLARED_CATS = {"S1": [0, 1], "S2": [0, 1], "S3": [1, 2, 3, 4, 5], "S4": [1, 2, 3, 4, 5],
                 "S5": [1, 2, 3, 4, 5], "S6": [1, 2, 3, 4, 5]}

ROWS, SE_COMPARE = [], {}
TOL_EXACT, TOL_IRRCAC, TOL_P = 1e-10, 1e-5, 1e-12


def chk(sc, coef, qty, app, ref, tol, source):
    app, ref = float(app), float(ref)
    d = abs(app - ref)
    ROWS.append(dict(scenario=sc, coefficient=coef, quantity=qty, app=app, ref=ref,
                     absdiff=d, tol=tol, ok=bool(d <= tol), source=source))


def run(sc, df=None, n_boot=NBOOT):
    f, cols, dt, q, smin = SC[sc]
    g = {"name": sc, "items": cols, "data_type": dt, "q_declared": q, "scale_min": smin}
    return run_multi_irar(D[f] if df is None else df, [g], alpha=ALPHA, n_boot=n_boot)["constructs"][0]


def X_of(sc):
    f, cols, *_ = SC[sc]
    return D[f][cols].to_numpy(float)


ZC = stats.norm.ppf(1 - ALPHA / 2)


def TC(n):
    return stats.t.ppf(1 - ALPHA / 2, n - 1)


def wald_checks(sc, coef, est, se, se0, block, n, src, est_key=None):
    """z = est/SE0, two-sided p, z- and t-based Wald intervals from SE."""
    z = est / se0
    chk(sc, coef, "z statistic", block["z"] if "z" in block else block["kappa_z"], z, TOL_EXACT, src)
    pv = block["p"] if "p" in block else block["kappa_p"]
    chk(sc, coef, "p value (two-sided)", pv, 2 * stats.norm.sf(abs(z)), TOL_P, src)
    chk(sc, coef, "CI lower, normal-based", block["ci_lo_z"], est - ZC * se, TOL_EXACT, src)
    chk(sc, coef, "CI upper, normal-based", block["ci_hi_z"], est + ZC * se, TOL_EXACT, src)
    chk(sc, coef, "CI lower, t-based", block["ci_lo_t"], est - TC(n) * se, TOL_EXACT, src)
    chk(sc, coef, "CI upper, t-based", block["ci_hi_t"], est + TC(n) * se, TOL_EXACT, src)


OUT = {sc: run(sc) for sc in SC}

# ------------------------------------------------------------------ dispatcher
TABLE1 = {
    ("dichotomous", 2): ["pct", "cohen", "gwet_ac1", "scott_pi", "bp", "krippendorff"],
    ("nominal", 2):     ["pct", "cohen", "gwet_ac1", "scott_pi", "bp", "krippendorff"],
    ("ordinal", 2):     ["pct", "weighted", "gwet_ac2", "krippendorff"],
    ("interval", 2):    ["pct", "icc", "ccc", "krippendorff"],
    ("ratio", 2):       ["pct", "icc", "ccc", "krippendorff"],
}
for dt in ("dichotomous", "nominal"):
    TABLE1[(dt, 3)] = ["pct", "fleiss", "gwet_ac1", "krippendorff"]
TABLE1[("ordinal", 3)] = ["pct", "gwet_ac2", "krippendorff"]
for dt in ("interval", "ratio"):
    TABLE1[(dt, 3)] = ["pct", "icc", "krippendorff"]
DISPATCH = []
for (dt, r), expect in TABLE1.items():
    for rr in ([2] if r == 2 else [3, 4, 5, 10, 20]):
        got = _get_applicable_methods(dt, rr)
        DISPATCH.append(dict(data_type=dt, raters=rr, expected=expect, returned=got, ok=got == expect))
for sc, (f, cols, dt, *_r) in SC.items():
    DISPATCH.append(dict(data_type=dt, raters=len(cols), scenario=sc,
                         expected=TABLE1[(dt, 2 if len(cols) == 2 else 3)],
                         returned=OUT[sc]["applicable_methods"],
                         ok=OUT[sc]["applicable_methods"] == TABLE1[(dt, 2 if len(cols) == 2 else 3)]))

# ------------------------------------------------------------------ percentage agreement
for sc in SC:
    X = X_of(sc)
    n, k = X.shape
    per = np.array([np.mean([X[i, a] == X[i, b] for a in range(k) for b in range(a + 1, k)]) for i in range(n)])
    pa, se = per.mean(), per.std(ddof=1) / np.sqrt(n)
    r = OUT[sc]["percentage_agreement"]
    src = "formula recomputed"
    chk(sc, "Percentage agreement", "estimate", r["pa"], pa, TOL_EXACT, src)
    chk(sc, "Percentage agreement", "CI lower, unit-level t", r["ci_lo"], max(0.0, pa - TC(n) * se), TOL_EXACT, src)
    chk(sc, "Percentage agreement", "CI upper, unit-level t", r["ci_hi"], min(1.0, pa + TC(n) * se), TOL_EXACT, src)
    chk(sc, "Percentage agreement", "pairwise comparisons", r["total_pairwise_comparisons"], n * k * (k - 1) / 2, 0, src)
    chk(sc, "Percentage agreement", "per-unit mean", r["mean"], per.mean(), TOL_EXACT, src)
    chk(sc, "Percentage agreement", "per-unit SD", r["sd"], per.std(ddof=1), TOL_EXACT, src)
    chk(sc, "Percentage agreement", "per-unit minimum", r["minimum"], per.min(), TOL_EXACT, src)
    chk(sc, "Percentage agreement", "per-unit maximum", r["maximum"], per.max(), TOL_EXACT, src)


# ------------------------------------------------------------------ two-rater categorical
def table_of(r1, r2, cats):
    T = np.zeros((len(cats), len(cats)))
    for a, b in zip(r1, r2):
        T[cats.index(a), cats.index(b)] += 1
    return T


def cohen_checks(sc, block, schemes, label_cats):
    X = X_of(sc).astype(int)
    r1, r2 = X[:, 0], X[:, 1]
    n = len(r1)
    obs = sorted(set(r1) | set(r2))
    # agreement (cross-)table on the observed categories, as displayed
    ct = block["crosstab"]
    T_obs = table_of(r1, r2, obs)
    chk(sc, "Cohen's kappa", "agreement table, cells", 0.0,
        np.abs(ct.to_numpy(float) - T_obs).max(), 0, "formula recomputed")
    chk(sc, "Cohen's kappa", "agreement table, row categories", 0.0,
        float(list(ct.index) != obs), 0, "formula recomputed")
    # statsmodels table over the category set used for the weights
    T = table_of(r1, r2, label_cats)
    po, pe = np.trace(T) / n, (T.sum(1) / n * T.sum(0) / n).sum()
    chk(sc, "Cohen's kappa", "observed agreement p_o", block["p_observed"], po, TOL_EXACT, "formula recomputed")
    chk(sc, "Cohen's kappa", "chance agreement p_e", block["p_expected"], pe, TOL_EXACT, "formula recomputed")
    for scheme, wt in schemes:
        v = block["variants"][scheme]
        name = "Cohen's kappa" if scheme == "unweighted" else f"Weighted kappa, {scheme}"
        sk = cohen_kappa_score(r1, r2, weights=wt, labels=label_cats)
        chk(sc, name, "estimate", v["kappa"], sk, TOL_EXACT, "scikit-learn 1.9.1")
        sm = (smir.cohens_kappa(T, return_results=True) if wt is None else
              smir.cohens_kappa(T, weights=np.arange(len(label_cats)), wt=wt, return_results=True))
        chk(sc, name, "estimate", v["kappa"], sm.kappa, TOL_EXACT, "statsmodels 0.15.0")
        chk(sc, name, "SE under H0 (FCE Eq. 9)", v["se_null"], sm.std_kappa0, TOL_EXACT, "statsmodels 0.15.0")
        chk(sc, name, "general SE (FCE Eq. 8)", v["se_general"], sm.std_kappa, TOL_EXACT, "statsmodels 0.15.0")
        wald_checks(sc, name, v["kappa"], v["se_general"], v["se_null"], v, n, "formula recomputed")
    # per-category agreement (Jaccard-style) over the category set used
    for row in block["per_category"]:
        c = row["category"]
        n1, n2, ag = int(np.sum(r1 == c)), int(np.sum(r2 == c)), int(np.sum((r1 == c) & (r2 == c)))
        chk(sc, "Cohen's kappa", f"category {c}: N rater 1", row["n_r1"], n1, 0, "formula recomputed")
        chk(sc, "Cohen's kappa", f"category {c}: N rater 2", row["n_r2"], n2, 0, "formula recomputed")
        chk(sc, "Cohen's kappa", f"category {c}: agreements", row["agree"], ag, 0, "formula recomputed")
        if n1 + n2 - ag > 0:
            chk(sc, "Cohen's kappa", f"category {c}: agreement %", row["pct"], 100 * ag / (n1 + n2 - ag),
                TOL_EXACT, "formula recomputed")
    return r1, r2, n


for sc in ("S1", "S3"):
    blk = OUT[sc]["cohen_kappa"]
    r1, r2, n = cohen_checks(sc, blk, [("unweighted", None)], sorted(set(X_of(sc).astype(int).ravel())))
    if sc == "S1":
        a, b = np.sum((r1 == 1) & (r2 == 1)), np.sum((r1 == 1) & (r2 == 0))
        c, d = np.sum((r1 == 0) & (r2 == 1)), np.sum((r1 == 0) & (r2 == 0))
        chk(sc, "Cohen's kappa", "prevalence index |a-d|/n", blk["prevalence_index"], abs(a - d) / n, TOL_EXACT,
            "formula recomputed")
        chk(sc, "Cohen's kappa", "bias index (b-c)/n", blk["bias_index"], (b - c) / n, TOL_EXACT, "formula recomputed")
        # with two categories the linear and quadratic weights coincide with identity weights
        for s in ("linear", "quadratic"):
            chk(sc, "Cohen's kappa", f"{s}-weighted equals unweighted (q = 2)", blk["variants"][s]["kappa"],
                blk["variants"]["unweighted"]["kappa"], TOL_EXACT, "identity")
    cats = DECLARED_CATS[sc]
    X = X_of(sc)
    # Scott's pi = Fleiss' kappa with two raters
    sp = OUT[sc]["scott_pi"]
    chk(sc, "Scott's pi", "estimate", sp["pi"],
        smir.fleiss_kappa(smir.aggregate_raters(X.astype(int))[0], method="fleiss"), TOL_EXACT, "statsmodels 0.15.0")
    ref = CAC(pd.DataFrame(X), categories=cats).fleiss()["est"]
    chk(sc, "Scott's pi", "estimate", sp["pi"], ref["coefficient_value"], TOL_IRRCAC, "irrCAC 0.4.4")
    chk(sc, "Scott's pi", "general SE (Gwet 2021)", sp["se"], ref["se"], TOL_IRRCAC, "irrCAC 0.4.4")
    chk(sc, "Scott's pi", "CI lower, t-based", sp["ci_lo_t"], ref["confidence_interval"][0], TOL_IRRCAC, "irrCAC 0.4.4")
    chk(sc, "Scott's pi", "CI upper, t-based", sp["ci_hi_t"], ref["confidence_interval"][1], TOL_IRRCAC, "irrCAC 0.4.4")
    po = np.mean(X[:, 0] == X[:, 1])
    obs = sorted(set(X.ravel()))
    pb = np.array([(np.mean(X[:, 0] == c_) + np.mean(X[:, 1] == c_)) / 2 for c_ in obs])
    pe = np.sum(pb ** 2)
    pi_hat = (po - pe) / (1 - pe)
    chk(sc, "Scott's pi", "estimate", sp["pi"], pi_hat, TOL_EXACT, "formula recomputed")
    chk(sc, "Scott's pi", "observed agreement P_o", sp["Po"], po, TOL_EXACT, "formula recomputed")
    chk(sc, "Scott's pi", "chance agreement P_e", sp["Pe"], pe, TOL_EXACT, "formula recomputed")
    s1, s2 = np.sum(pb * (1 - pb)), np.sum(pb * (1 - pb) * (1 - 2 * pb))
    se0 = np.sqrt(2 / (n * 2 * 1 * s1 ** 2) * (s1 ** 2 - s2))
    chk(sc, "Scott's pi", "SE under H0 (Fleiss-Nee-Landis 1979, r = 2)", sp["se_null"], se0, TOL_EXACT,
        "formula recomputed")
    agree = (X[:, 0] == X[:, 1]).astype(float)
    pe_unit = np.array([(pb[obs.index(u)] + pb[obs.index(v)]) / 2 for u, v in X])
    kstar = (agree - pe) / (1 - pe) - 2 * (1 - pi_hat) * (pe_unit - pe) / (1 - pe)
    seg = np.sqrt(np.sum((kstar - pi_hat) ** 2) / (n * (n - 1)))
    chk(sc, "Scott's pi", "general SE (Gwet 2021, r = 2)", sp["se"], seg, TOL_EXACT, "formula recomputed")
    wald_checks(sc, "Scott's pi", pi_hat, seg, se0, sp, n, "formula recomputed")
    # Brennan-Prediger
    bp = OUT[sc]["brennan_prediger"]
    q = SC[sc][3]
    ref = CAC(pd.DataFrame(X), categories=cats).bp()["est"]
    chk(sc, "Brennan-Prediger", "estimate", bp["kappa_bp"], ref["coefficient_value"], TOL_IRRCAC, "irrCAC 0.4.4")
    kbp = (po - 1 / q) / (1 - 1 / q)
    sebp = np.sqrt(po * (1 - po) / (n * (1 - 1 / q) ** 2))
    chk(sc, "Brennan-Prediger", "estimate", bp["kappa_bp"], kbp, TOL_EXACT, "formula recomputed")
    chk(sc, "Brennan-Prediger", "chance agreement 1/q", bp["Pe"], 1 / q, TOL_EXACT, "formula recomputed")
    chk(sc, "Brennan-Prediger", "SE (Cohen 1960 form, p_e = 1/q)", bp["se"], sebp, TOL_EXACT, "formula recomputed")
    wald_checks(sc, "Brennan-Prediger", kbp, sebp, sebp, bp, n, "formula recomputed")
    SE_COMPARE[f"BP {sc}"] = (bp["se"], ref["se"])
    # comparison table displayed in the Brennan-Prediger tab
    cmp = {row[0]: row[1] for row in bp["comparison_table"]}
    chk(sc, "Brennan-Prediger", "comparison table: Cohen's kappa", cmp["Cohen's κ"], OUT[sc]["cohen_kappa"]["kappa"],
        TOL_EXACT, "internal consistency")
    chk(sc, "Brennan-Prediger", "comparison table: AC1", cmp["Gwet's AC1"], OUT[sc]["gwet_ac1"]["ac1"],
        TOL_EXACT, "internal consistency")


# ------------------------------------------------------------------ AC1 (S1-S4)
def ac1_ref(Xs, cats):
    n, k = Xs.shape
    po = np.mean([np.mean([Xs[i, a] == Xs[i, b] for a in range(k) for b in range(a + 1, k)]) for i in range(n)])
    pic = np.array([np.mean(Xs == c) for c in cats])
    pe = np.sum(pic * (1 - pic)) / (len(cats) - 1)
    return (po - pe) / (1 - pe)


for sc in ("S1", "S2", "S3", "S4"):
    X, cats = X_of(sc), DECLARED_CATS[sc]
    n = len(X)
    r = OUT[sc]["gwet_ac1"]
    ref = CAC(pd.DataFrame(X), categories=cats).gwet()["est"]
    chk(sc, "Gwet's AC1", "estimate", r["ac1"], ref["coefficient_value"], TOL_IRRCAC, "irrCAC 0.4.4")
    est = ac1_ref(X, cats)
    chk(sc, "Gwet's AC1", "estimate", r["ac1"], est, TOL_EXACT, "formula recomputed")
    jk = np.array([ac1_ref(np.delete(X, i, 0), cats) for i in range(n)])
    se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    chk(sc, "Gwet's AC1", "jackknife SE", r["se"], se, TOL_EXACT, "formula recomputed")
    chk(sc, "Gwet's AC1", "declared q used", r["q"], len(cats), 0, "declaration")
    wald_checks(sc, "Gwet's AC1", est, se, se, r, n, "formula recomputed")
    SE_COMPARE[f"AC1 {sc}"] = (r["se"], ref["se"])

# ------------------------------------------------------------------ Fleiss (S2, S4)
for sc in ("S2", "S4"):
    X = X_of(sc).astype(int)
    n, k = X.shape
    fk = OUT[sc]["fleiss_kappa"]
    tab = smir.aggregate_raters(X)[0]
    chk(sc, "Fleiss' kappa", "estimate", fk["kappa"], smir.fleiss_kappa(tab, method="fleiss"), TOL_EXACT,
        "statsmodels 0.15.0")
    ref = CAC(pd.DataFrame(X), categories=DECLARED_CATS[sc]).fleiss()["est"]
    chk(sc, "Fleiss' kappa", "estimate", fk["kappa"], ref["coefficient_value"], TOL_IRRCAC, "irrCAC 0.4.4")
    chk(sc, "Fleiss' kappa", "general SE (Gwet 2021)", fk["kappa_se_general"], ref["se"], TOL_IRRCAC, "irrCAC 0.4.4")
    chk(sc, "Fleiss' kappa", "CI lower, t-based", fk["ci_lo_t"], ref["confidence_interval"][0], TOL_IRRCAC,
        "irrCAC 0.4.4")
    chk(sc, "Fleiss' kappa", "CI upper, t-based", fk["ci_hi_t"], ref["confidence_interval"][1], TOL_IRRCAC,
        "irrCAC 0.4.4")
    cats = sorted(set(X.ravel()))
    T = np.array([[np.sum(X[i] == c) for c in cats] for i in range(n)], float)
    p = T.sum(0) / (n * k)
    Pi = (T * (T - 1)).sum(1) / (k * (k - 1))
    Pbar, Pe = Pi.mean(), np.sum(p ** 2)
    kap = (Pbar - Pe) / (1 - Pe)
    chk(sc, "Fleiss' kappa", "estimate", fk["kappa"], kap, TOL_EXACT, "formula recomputed")
    chk(sc, "Fleiss' kappa", "mean observed agreement P-bar", fk["P_bar"], Pbar, TOL_EXACT, "formula recomputed")
    chk(sc, "Fleiss' kappa", "chance agreement P_e", fk["P_e"], Pe, TOL_EXACT, "formula recomputed")
    pq = p * (1 - p)
    se0 = np.sqrt(2 / (n * k * (k - 1)) * (pq.sum() ** 2 - np.sum(pq * (1 - 2 * p))) / pq.sum() ** 2)
    chk(sc, "Fleiss' kappa", "SE under H0 (Fleiss-Nee-Landis 1979)", fk["kappa_se"], se0, TOL_EXACT,
        "formula recomputed")
    pei = (T * p).sum(1) / k
    kstar = (Pi - Pe) / (1 - Pe) - 2 * (1 - kap) * (pei - Pe) / (1 - Pe)
    seg = np.sqrt(np.sum((kstar - kap) ** 2) / (n * (n - 1)))
    chk(sc, "Fleiss' kappa", "general SE (Gwet 2021)", fk["kappa_se_general"], seg, TOL_EXACT, "formula recomputed")
    wald_checks(sc, "Fleiss' kappa", kap, seg, se0, fk, n, "formula recomputed")
    for j, c in enumerate(cats):
        kc = (np.sum(T[:, j] * (T[:, j] - 1)) / (n * k * (k - 1)) - p[j] ** 2) / (p[j] * (1 - p[j]))
        chk(sc, "Fleiss' kappa", f"category {c}: kappa_j (Fleiss 1971)", fk["kappa_per_category"][str(c)]
            if str(c) in fk["kappa_per_category"] else fk["kappa_per_category"][c], kc, TOL_EXACT, "formula recomputed")
        chk(sc, "Fleiss' kappa", f"category {c}: proportion p_j", fk["p_per_category"][str(c)]
            if str(c) in fk["p_per_category"] else fk["p_per_category"][c], p[j], TOL_EXACT, "formula recomputed")

# ------------------------------------------------------------------ weighted kappa (S5)
cohen_checks("S5", OUT["S5"]["weighted_kappa"], [("unweighted", None), ("linear", "linear"),
                                                 ("quadratic", "quadratic")], [1, 2, 3, 4, 5])


# ------------------------------------------------------------------ AC2 (S5, S6)
def gwet_w(scheme, q):
    i = np.arange(q)
    d = np.abs(i[:, None] - i[None, :]).astype(float)
    if scheme == "identity":
        return np.eye(q)
    if scheme == "linear":
        return 1 - d / (q - 1)
    if scheme == "quadratic":
        return 1 - d ** 2 / (q - 1) ** 2
    if scheme == "radical":
        return 1 - np.sqrt(d / (q - 1))
    m = (d + 1) * d / 2
    return 1 - m / m.max()


def ac2_ref(X, scheme, lo=1, q=5):
    W = gwet_w(scheme, q)
    n, k = X.shape
    pw = np.mean([np.mean([W[int(X[i, a]) - lo, int(X[i, b]) - lo] for a in range(k) for b in range(a + 1, k)])
                  for i in range(n)])
    pi = np.array([np.mean(X == lo + c) for c in range(q)])
    pe = W.sum() * np.sum(pi * (1 - pi)) / (q * (q - 1))
    return (pw - pe) / (1 - pe)


for sc in ("S5", "S6"):
    X = X_of(sc)
    n = len(X)
    for sch in ("identity", "linear", "quadratic", "radical", "ordinal"):
        app = OUT[sc]["gwet_ac2"]["schemes"][sch]
        name = f"Gwet's AC2, {sch}"
        ref = CAC(pd.DataFrame(X), weights=sch, categories=[1, 2, 3, 4, 5]).gwet()["est"]
        chk(sc, name, "estimate", app["ac2"], ref["coefficient_value"], TOL_IRRCAC, "irrCAC 0.4.4")
        est = ac2_ref(X, sch)
        chk(sc, name, "estimate", app["ac2"], est, TOL_EXACT, "formula recomputed")
        jk = np.array([ac2_ref(np.delete(X, i, 0), sch) for i in range(n)])
        se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
        chk(sc, name, "jackknife SE", app["se"], se, TOL_EXACT, "formula recomputed")
        chk(sc, name, "declared q used", app["q"], 5, 0, "declaration")
        wald_checks(sc, name, est, se, se, app, n, "formula recomputed")
        SE_COMPARE[f"AC2 {sch} {sc}"] = (app["se"], ref["se"])


# ------------------------------------------------------------------ ICC (S7-S10)
def anova(X):
    n, k = X.shape
    gm = X.mean()
    msr = k * np.sum((X.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((X.mean(0) - gm) ** 2) / (k - 1)
    sst = np.sum((X - gm) ** 2)
    mse = (sst - msr * (n - 1) - msc * (k - 1)) / ((n - 1) * (k - 1))
    msw = (sst - msr * (n - 1)) / (n * (k - 1))
    return n, k, msr, msc, mse, msw


def icc_ref(X):
    """Shrout & Fleiss (1979) / McGraw & Wong (1996): estimates, F tests and
    confidence limits (McGraw & Wong, Table 7), written from the formulas."""
    n, k, msr, msc, mse, msw = anova(X)
    est = {"icc_1_1": (msr - msw) / (msr + (k - 1) * msw),
           "icc_2_1": (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n),
           "icc_3_1": (msr - mse) / (msr + (k - 1) * mse),
           "icc_1_k": (msr - msw) / msr,
           "icc_2_k": (msr - mse) / (msr + (msc - mse) / n),
           "icc_3_k": (msr - mse) / msr}
    F1, d11, d12 = msr / msw, n - 1, n * (k - 1)
    F3, d31, d32 = msr / mse, n - 1, (n - 1) * (k - 1)
    ftest = {"icc_1_1": (F1, d11, d12), "icc_1_k": (F1, d11, d12), "icc_2_1": (F3, d31, d32),
             "icc_2_k": (F3, d31, d32), "icc_3_1": (F3, d31, d32), "icc_3_k": (F3, d31, d32)}
    a2 = ALPHA / 2
    FL1, FU1 = F1 / stats.f.ppf(1 - a2, d11, d12), F1 * stats.f.ppf(1 - a2, d12, d11)
    FL3, FU3 = F3 / stats.f.ppf(1 - a2, d31, d32), F3 * stats.f.ppf(1 - a2, d32, d31)
    rho = est["icc_2_1"]
    a = k * rho / (n * (1 - rho))
    b = 1 + k * rho * (n - 1) / (n * (1 - rho))
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    Fs, Fss = stats.f.ppf(1 - a2, n - 1, v), stats.f.ppf(1 - a2, v, n - 1)
    lo2 = n * (msr - Fs * mse) / (Fs * (k * msc + (k * n - k - n) * mse) + n * msr)
    hi2 = n * (Fss * msr - mse) / (k * msc + (k * n - k - n) * mse + n * Fss * msr)
    lo2k = n * (msr - Fs * mse) / (Fs * (msc - mse) + n * msr)
    hi2k = n * (Fss * msr - mse) / (msc - mse + n * Fss * msr)
    ci = {"icc_1_1": ((FL1 - 1) / (FL1 + k - 1), (FU1 - 1) / (FU1 + k - 1)),
          "icc_3_1": ((FL3 - 1) / (FL3 + k - 1), (FU3 - 1) / (FU3 + k - 1)),
          "icc_1_k": (1 - 1 / FL1, 1 - 1 / FU1), "icc_3_k": (1 - 1 / FL3, 1 - 1 / FU3),
          "icc_2_1": (lo2, hi2), "icc_2_k": (lo2k, hi2k)}
    return est, ftest, ci


PG_NAMES = {"ICC1": "icc_1_1", "ICC2": "icc_2_1", "ICC3": "icc_3_1", "ICC1k": "icc_1_k", "ICC2k": "icc_2_k",
            "ICC3k": "icc_3_k", "ICC(1,1)": "icc_1_1", "ICC(A,1)": "icc_2_1", "ICC(C,1)": "icc_3_1",
            "ICC(1,k)": "icc_1_k", "ICC(A,k)": "icc_2_k", "ICC(C,k)": "icc_3_k"}
ICC_LABEL = {"icc_1_1": "ICC(1,1)", "icc_2_1": "ICC(A,1)", "icc_3_1": "ICC(C,1)", "icc_1_k": "ICC(1,k)",
             "icc_2_k": "ICC(A,k)", "icc_3_k": "ICC(C,k)"}
for sc in ("S7", "S8", "S9", "S10"):
    X = X_of(sc)
    n, k = X.shape
    ic = OUT[sc]["icc"]
    est, ftest, ci = icc_ref(X)
    long = pd.DataFrame({"t": np.repeat(np.arange(n), k), "r": np.tile(np.arange(k), n), "s": X.flatten()})
    pgr = pg.intraclass_corr(data=long, targets="t", raters="r", ratings="s")
    pgr["key"] = pgr["Type"].map(PG_NAMES)
    pgr = pgr.set_index("key")
    ci_col = [c for c in pgr.columns if str(c).startswith("CI")][0]
    df_rows = {PG_NAMES[row[0]]: row for row in ic["icc_df"].itertuples(index=False)}
    cols = list(ic["icc_df"].columns)
    for key in ICC_LABEL:
        nm = ICC_LABEL[key]
        chk(sc, nm, "estimate", ic[key], est[key], TOL_EXACT, "ANOVA formulas recomputed")
        chk(sc, nm, "estimate", ic[key], pgr.loc[key, "ICC"], TOL_EXACT, "pingouin 0.7.0")
        row = dict(zip(cols, df_rows[key]))
        F, d1, d2 = ftest[key]
        chk(sc, nm, "F statistic", row["F"], F, TOL_EXACT, "ANOVA formulas recomputed")
        chk(sc, nm, "df1", row["df1"], d1, 0, "ANOVA formulas recomputed")
        chk(sc, nm, "df2", row["df2"], d2, 0, "ANOVA formulas recomputed")
        chk(sc, nm, "p value (F test)", ic[f"{key}_p"], stats.f.sf(F, d1, d2), TOL_P, "ANOVA formulas recomputed")
        chk(sc, nm, "CI lower (McGraw & Wong 1996)", ic[f"{key}_ci_lo"], ci[key][0], TOL_EXACT,
            "formula recomputed")
        chk(sc, nm, "CI upper (McGraw & Wong 1996)", ic[f"{key}_ci_hi"], ci[key][1], TOL_EXACT,
            "formula recomputed")
        for side, j in (("lower", 0), ("upper", 1)):
            chk(sc, nm, f"CI {side}, two decimals", round(ic[f"{key}_ci_{'lo' if j == 0 else 'hi'}"], 2),
                pgr.loc[key, ci_col][j], 1e-12, "pingouin 0.7.0 (prints two decimals)")

# ------------------------------------------------------------------ Lin's CCC (S7, S9)
for sc in ("S7", "S9"):
    X = X_of(sc)
    x, y = X[:, 0], X[:, 1]
    n = len(x)
    mx, my, sx2, sy2 = x.mean(), y.mean(), x.var(), y.var()
    sxy = np.mean((x - mx) * (y - my))
    c = 2 * sxy / (sx2 + sy2 + (mx - my) ** 2)
    rr = np.corrcoef(x, y)[0, 1]
    u = (mx - my) / (sx2 * sy2) ** 0.25
    vz = ((1 - rr ** 2) * c ** 2 / ((1 - c ** 2) * rr ** 2) + 2 * c ** 3 * (1 - c) * u ** 2 / (rr * (1 - c ** 2) ** 2)
          - c ** 4 * u ** 4 / (2 * rr ** 2 * (1 - c ** 2) ** 2)) / (n - 2)
    cc = OUT[sc]["lin_ccc"]
    src = "formula recomputed"
    for qty, app, ref in (("estimate", cc["ccc"], c), ("Pearson r", cc["rho"], rr), ("r squared", cc["r_squared"], rr ** 2),
                          ("bias-correction factor C_b", cc["Cb"], c / rr), ("location shift u", cc["u_shift"], u),
                          ("scale shift v", cc["v_shift"], np.sqrt(sx2) / np.sqrt(sy2)),
                          ("mean, rater 1", cc["mean1"], mx), ("mean, rater 2", cc["mean2"], my),
                          ("mean difference", cc["mean_diff"], mx - my), ("SD, rater 1 (divisor n)", cc["sd1"], np.sqrt(sx2)),
                          ("SD, rater 2 (divisor n)", cc["sd2"], np.sqrt(sy2)), ("SE of Fisher z", cc["se"], np.sqrt(vz)),
                          ("CI lower (Fisher z)", cc["ci_lo"], np.tanh(np.arctanh(c) - ZC * np.sqrt(vz))),
                          ("CI upper (Fisher z)", cc["ci_hi"], np.tanh(np.arctanh(c) + ZC * np.sqrt(vz)))):
        chk(sc, "Lin's CCC", qty, app, ref, TOL_EXACT, src)
    chk(sc, "Lin's CCC", "Pearson r", cc["rho"], stats.pearsonr(x, y)[0], TOL_EXACT, "SciPy 1.18.1")

# ------------------------------------------------------------------ Krippendorff's alpha (all ten)
BOOT_REPLICATION = {}
for sc in SC:
    X = X_of(sc)
    lv = KMETRIC[SC[sc][2]]
    ka = OUT[sc]["krippendorff"]
    chk(sc, "Krippendorff's alpha", f"estimate ({lv})", ka["alpha"],
        kdp.alpha(reliability_data=X.T, level_of_measurement=lv), TOL_EXACT, "krippendorff 0.9.0")
    if lv == "nominal":
        cats = DECLARED_CATS[sc]
        chk(sc, "Krippendorff's alpha", "estimate (nominal)", ka["alpha"],
            CAC(pd.DataFrame(X), categories=cats).krippendorff()["est"]["coefficient_value"], TOL_IRRCAC,
            "irrCAC 0.4.4")
    # bootstrap interval replicated: same seed and the same unit indices,
    # alpha of every resample computed by the krippendorff package
    rng = np.random.default_rng(42)
    n_u = X.shape[0]
    reps = np.array([kdp.alpha(reliability_data=X[rng.integers(0, n_u, n_u)].T, level_of_measurement=lv)
                     for _ in range(NBOOT)])
    lo, hi = np.percentile(reps, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    chk(sc, "Krippendorff's alpha", "bootstrap CI lower (1,000 resamples)", ka["ci_lo"], lo, 1e-9,
        "krippendorff 0.9.0 on identical resamples")
    chk(sc, "Krippendorff's alpha", "bootstrap CI upper (1,000 resamples)", ka["ci_hi"], hi, 1e-9,
        "krippendorff 0.9.0 on identical resamples")
    BOOT_REPLICATION[sc] = dict(metric=lv, ci=[float(lo), float(hi)])

# ------------------------------------------------------------------ cross-configuration identities
def same(sc_a, path_a, sc_b, path_b, label):
    def get(o, p):
        for k in p:
            o = o[k]
        return o
    chk(f"{sc_a}={sc_b}", "identity", label, get(OUT[sc_a], path_a), get(OUT[sc_b], path_b), TOL_EXACT, "identity")


for f in ("kappa", "se_null", "se_general", "ci_lo_t", "ci_hi_t"):
    same("S3", ["cohen_kappa", "variants", "unweighted", f], "S5", ["weighted_kappa", "variants", "unweighted", f],
         f"Cohen's kappa (nominal S3) = unweighted kappa (ordinal S5): {f}")
for f in ("ac1", "se", "ci_lo_t", "ci_hi_t"):
    g = "ac2" if f == "ac1" else f
    same("S3", ["gwet_ac1", f], "S5", ["gwet_ac2", "schemes", "identity", g], f"AC1 (S3) = AC2 identity (S5): {f}")
    same("S4", ["gwet_ac1", f], "S6", ["gwet_ac2", "schemes", "identity", g], f"AC1 (S4) = AC2 identity (S6): {f}")


def flatten(d, p=()):
    """Every leaf of a result, including the per-unit agreement vector and the
    numeric cells of the ICC F-test table."""
    o = {}
    if isinstance(d, pd.DataFrame):
        d = d.drop(columns=[c for c in d.columns if d[c].dtype == object and c != "CI95%"], errors="ignore")
        d = {c: [list(x) if isinstance(x, np.ndarray) else x for x in d[c].tolist()] for c in d.columns}
    if isinstance(d, np.ndarray):
        d = d.tolist()
    if isinstance(d, dict):
        for k, v in d.items():
            o.update(flatten(v, p + (k,)))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            o.update(flatten(v, p + (i,)))
    else:
        o[p] = d
    return o


IR_IDENTITY = {}
for a, b in (("S7", "S9"), ("S8", "S10")):
    fa = {k: v for k, v in flatten({kk: OUT[a][kk] for kk in ("percentage_agreement", "icc", "lin_ccc")}).items()
          if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)}
    fb = flatten({kk: OUT[b][kk] for kk in ("percentage_agreement", "icc", "lin_ccc")})
    nnum = 0
    for k_, v in fa.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        nnum += 1
        chk(f"{a}={b}", "identity", "/".join(map(str, k_)), v, fb[k_], 0, "identity (bitwise)")
    IR_IDENTITY[f"{a}={b}"] = nnum

# ------------------------------------------------------------------ missing ratings (all ten)
MISSING = {}
for sc in SC:
    f, cols, dt, q, smin = SC[sc]
    X = X_of(sc)
    rng = np.random.default_rng(7)
    mask = rng.random(X.shape) < 0.10
    Xm = X.copy()
    Xm[mask] = np.nan
    dfm = pd.DataFrame(Xm, columns=cols)
    Sm = run(sc, df=dfm)
    lv = KMETRIC[dt]
    ref = kdp.alpha(reliability_data=Xm.T, level_of_measurement=lv)
    chk(sc, "Krippendorff's alpha", "estimate with 10% of ratings deleted", Sm["krippendorff"]["alpha"], ref,
        TOL_EXACT, "krippendorff 0.9.0")
    complete = ~np.isnan(Xm).any(axis=1)
    pairable = (~np.isnan(Xm)).sum(axis=1) >= 2
    chk(sc, "Missing ratings", "units used by alpha = pairable units", Sm["krippendorff"]["n_units"],
        int(pairable.sum()), 0, "formula recomputed")
    chk(sc, "Missing ratings", "units used by other coefficients = complete units", Sm["n_units"],
        int(complete.sum()), 0, "formula recomputed")
    lw = Xm[complete]
    MISSING[sc] = dict(data_type=dt, raters=len(cols), cells_total=int(mask.size), cells_removed=int(mask.sum()),
                       n_complete=int(complete.sum()), n_pairable=int(pairable.sum()),
                       alpha_app=float(Sm["krippendorff"]["alpha"]), alpha_pkg=float(ref),
                       ci=[Sm["krippendorff"]["ci_lo"], Sm["krippendorff"]["ci_hi"]],
                       alpha_complete_only=float(kdp.alpha(reliability_data=lw.T, level_of_measurement=lv)),
                       metric=lv)

# ------------------------------------------------------------------ explanatory decompositions
# Chance terms that the article uses to explain differences between
# coefficients computed on the same ratings. They are not displayed by the
# application; they are evaluated here from the defining equations.
DECOMP = {}
for sc in ("S1", "S3"):
    X, cats, q = X_of(sc), DECLARED_CATS[sc], SC[sc][3]
    pic = np.array([np.mean(X == c) for c in cats])
    DECOMP[sc] = dict(p_o=float(np.mean(X[:, 0] == X[:, 1])),
                      p_e_cohen=float(OUT[sc]["cohen_kappa"]["p_expected"]),
                      p_e_scott=float(OUT[sc]["scott_pi"]["Pe"]), p_e_bp=1.0 / q,
                      p_e_ac1=float(np.sum(pic * (1 - pic)) / (q - 1)))
for sc in ("S2", "S4"):
    X, cats, q = X_of(sc), DECLARED_CATS[sc], SC[sc][3]
    pic = np.array([np.mean(X == c) for c in cats])
    DECOMP[sc] = dict(p_a=float(OUT[sc]["fleiss_kappa"]["P_bar"]), p_e_fleiss=float(OUT[sc]["fleiss_kappa"]["P_e"]),
                      p_e_ac1=float(np.sum(pic * (1 - pic)) / (q - 1)))
for sc in ("S5", "S6"):
    X = X_of(sc)
    n, k = X.shape
    d5 = {}
    for sch in ("linear", "quadratic"):
        W = gwet_w(sch, 5)
        pw = np.mean([np.mean([W[int(X[i, a]) - 1, int(X[i, b]) - 1] for a in range(k) for b in range(a + 1, k)])
                      for i in range(n)])
        pi = np.array([np.mean(X == 1 + c) for c in range(5)])
        d5[f"ac2_{sch}_p_a"] = float(pw)
        d5[f"ac2_{sch}_p_e"] = float(W.sum() * np.sum(pi * (1 - pi)) / 20)
        d5[f"ac2_{sch}_T_w"] = float(W.sum())
        if k == 2:
            T = table_of(X[:, 0].astype(int), X[:, 1].astype(int), [1, 2, 3, 4, 5]) / n
            d5[f"kappa_{sch}_p_o"] = float((W * T).sum())
            d5[f"kappa_{sch}_p_e"] = float((W * np.outer(T.sum(1), T.sum(0))).sum())
    d5["pooled_proportions"] = {str(c): float(np.mean(X == c)) for c in range(1, 6)}
    DECOMP[sc] = d5
for sc in ("S7", "S8"):
    X = X_of(sc)
    DECOMP[sc] = dict(rater_means=[float(v) for v in X.mean(0)], value_min=float(X.min()), value_max=float(X.max()),
                      exact_agreement_units=int(np.sum([len(set(row)) < len(row) for row in X])),
                      # ratio metric = interval metric / (c + k)^2; largest/smallest (c + k)^2 over the value range
                      ratio_metric_weight_range=float((X.max() / X.min()) ** 2))
# two raters, nominal data, no missing ratings: alpha_K = pi + (1 - pi)/(2n)
for sc in ("S1", "S3"):
    n = len(X_of(sc))
    pi_, a_ = OUT[sc]["scott_pi"]["pi"], OUT[sc]["krippendorff"]["alpha"]
    DECOMP[sc].update(alpha_minus_pi=float(a_ - pi_), small_sample_term=float((1 - pi_) / (2 * n)))
# what the q-dependent coefficients would have been had q been taken from the
# categories observed in the sample (3) instead of the declared scale (5)
X3 = X_of("S3")
po3 = np.mean(X3[:, 0] == X3[:, 1])
DECOMP["S3"].update(q_observed=3, bp_if_q_observed=float((po3 - 1 / 3) / (1 - 1 / 3)),
                    ac1_if_q_observed=float(ac1_ref(X3, [3, 4, 5])))
DECOMP["S5"].update(ac2_quadratic_if_q_observed=float(ac2_ref(X_of("S5"), "quadratic", lo=3, q=3)),
                    ac2_linear_if_q_observed=float(ac2_ref(X_of("S5"), "linear", lo=3, q=3)))

out = dict(settings=dict(alpha=ALPHA, n_boot=NBOOT), rows=ROWS, decompositions=DECOMP,
           se_compare={k: [float(a), float(b)] for k, (a, b) in SE_COMPARE.items()},
           dispatch=DISPATCH, bootstrap_replication=BOOT_REPLICATION, interval_ratio_identity=IR_IDENTITY,
           missing=MISSING)
json.dump(out, open(out_json, "w"), indent=1, default=float)
bad = [r for r in ROWS if not r["ok"]]
print(f"{len(ROWS)} checks, {len(ROWS) - len(bad)} within tolerance, {len(bad)} outside")
for r in bad:
    print("  OUTSIDE:", r)
print("dispatch:", sum(d["ok"] for d in DISPATCH), "of", len(DISPATCH), "match Table 1")
print("largest |diff| overall:", max(r["absdiff"] for r in ROWS))
for kk, (a, b) in SE_COMPARE.items():
    print(f"  SE {kk}: app={a:.5f} irrCAC={b:.5f} ratio={a / b:.4f}")
print("interval/ratio identity numeric fields:", IR_IDENTITY)
