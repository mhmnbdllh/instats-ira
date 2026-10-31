"""
modules/reliability.py
======================
Computation engine of the Reliability Toolkit (this repository), used by
both of its pages:

  * Inter-Rater Agreement (InStats-IrA; views/interrateragreement.py) --
    run_multi_irar(): Percentage Agreement, Cohen's kappa (unweighted,
    linear, quadratic), Fleiss' kappa, Scott's pi, the Brennan-Prediger
    coefficient, Gwet's AC1 and AC2 (five weighting schemes), the
    intraclass correlation coefficient (six forms), Lin's concordance
    correlation coefficient, and Krippendorff's alpha (nominal, ordinal,
    interval, ratio), each computed only for the data types and rater
    counts for which it is defined (_get_applicable_methods).

  * Internal Consistency (views/internalconsistency.py) --
    run_multi_reliability(): Cronbach's alpha, KR-20/KR-21, split-half,
    Guttman's lambda1-6, McDonald's omega, ICC(C,k), item statistics.

Every formula is documented, with its primary source, in the docstring of
the function that implements it.
"""

import itertools
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    import pingouin as pg
except Exception:
    pg = None

try:
    from sklearn.metrics import cohen_kappa_score as _skl_kappa
    _SKL = True
except Exception:
    _SKL = False


# ══════════════════════════════════════════════════════════════════════════════
# KAPPA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kappa_label(k):
    """Landis & Koch (1977) inter-rater agreement benchmarks."""
    if k is None or (isinstance(k, float) and np.isnan(k)):
        return "N/A"
    k = float(k)
    if k < 0:    return "Poor (< 0: worse than chance)"
    if k < 0.20: return "Slight (0.00–0.20)"
    if k < 0.40: return "Fair (0.21–0.40)"
    if k < 0.60: return "Moderate (0.41–0.60)"
    if k < 0.80: return "Substantial (0.61–0.80)"
    return "Almost Perfect (0.81–1.00)"


def krippendorff_label(a):
    """Krippendorff (2004) reliability benchmarks.

    CORRECTION (2026-10-21): this function previously labeled values below
    0.667 with a "Slight/Fair/Moderate" 3-tier sub-scheme (cutoffs at 0.20/
    0.40/0.60) while still citing "Krippendorff (2004)" for the whole
    function. That sub-scheme is NOT Krippendorff's -- Krippendorff (2004,
    pp. 241-243) gives only the simple rule used below (confirmed via an
    independent secondary source, Wikipedia's "Krippendorff's alpha"
    article, which quotes his own three outcomes: >=0.800 reliable;
    0.667-0.800 tentative conclusions only; <0.667 discard the data --
    with NO finer breakdown below 0.667 at all). The Slight/Fair/Moderate
    labels and their 0.20/0.40/0.60 cutoffs are Landis & Koch (1977)'s
    kappa-specific scale (see kappa_label() above), copied here and
    mislabeled as Krippendorff's own. Fixed to use only Krippendorff's
    actual three outcomes; a result below 0.667 is "Unreliable" at any
    sub-level rather than being falsely parceled into borrowed kappa tiers
    Krippendorff never proposed for his own coefficient.
    """
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return "N/A"
    a = float(a)
    if a < 0.667: return "Unreliable (< 0.667 — Krippendorff recommends discarding data at this level)"
    if a < 0.80:  return "Tentative (0.667–0.80 — acceptable only for tentative conclusions)"
    return "Reliable (≥ 0.80)"


def _detect_data_type(vals):
    """Classify ratings as 'dichotomous', 'nominal', 'ordinal', or 'interval'.
    
    Decision rules:
    - Exactly 2 unique values → dichotomous
    - Non-integer or many unique values → interval
    - Integer-valued AND sequential AND min >= 1 AND n_unique 3–10 → ordinal
      (Likert-style: 1,2,3,4,5 or 1,2,3)
    - Integer-valued AND (min == 0 OR non-sequential OR n_unique > 10) → nominal
    - String/non-numeric → nominal
    """
    clean = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return "nominal"
    try:
        float_vals = [float(v) for v in clean]
    except (ValueError, TypeError):
        return "nominal"
    unique_vals = sorted(set(float_vals))
    n_unique = len(unique_vals)
    if n_unique == 2:
        return "dichotomous"
    all_int = all(float(v) == int(float(v)) for v in unique_vals)
    if not all_int or n_unique > 10:
        return "interval"
    int_vals = [int(v) for v in unique_vals]
    is_sequential = (max(int_vals) - min(int_vals) == n_unique - 1)
    min_val = min(int_vals)
    # Ordinal: sequential integers starting from 1 (Likert-style)
    # e.g., 1,2,3,4,5 → ordinal; 0,1,2 → nominal (labels)
    if is_sequential and min_val >= 1 and 3 <= n_unique <= 10:
        return "ordinal"
    # 0-based sequential (e.g., 0,1,2,3) → nominal (category labels)
    return "nominal"


def _kripp_counts(X):
    """Unit-by-value count matrix for Krippendorff's alpha.

    X: (N_units x k_raters) array, NaN = missing. Returns (C, vals) where
    vals are the distinct observed values (sorted) and C[u, c] is the number
    of raters who assigned value vals[c] to unit u."""
    X = np.asarray(X, dtype=float)
    vals = np.unique(X[~np.isnan(X)])
    C = np.stack([(X == v).sum(axis=1) for v in vals], axis=1).astype(float)
    return C, vals


def _kripp_alpha_from_counts(C, vals, data_type):
    """Krippendorff's alpha from a unit-by-value count matrix, via the
    coincidence matrix (Krippendorff, 2004, Ch. 11).

    Only pairable units (>= 2 values) enter. Coincidences:
        o_ck = sum_u [n_uc * n_uk - (c == k) * n_uc] / (m_u - 1),
    n_c = sum_k o_ck (pairable values of category c), n = sum_c n_c, and
        alpha = 1 - (n - 1) * sum o_ck d2_ck / sum n_c n_k d2_ck,
    which is identical to 1 - D_o/D_e with
        D_o = (1/n) sum_ck o_ck d2_ck,  D_e = sum_ck n_c n_k d2_ck / (n(n-1)).
    Distance d2 by metric:
        nominal  : 0 if c == k else 1
        ordinal  : (sum_{g=c..k} n_g - (n_c + n_k)/2)^2   (n_g from pairable values)
        interval : (c - k)^2
        ratio    : ((c - k)/(c + k))^2
    Returns NaN when there are no pairable units or no expected
    disagreement (all pairable values identical)."""
    m = C.sum(axis=1)
    keep = m >= 2
    C, m = C[keep], m[keep]
    if C.shape[0] == 0:
        return np.nan
    w = 1.0 / (m - 1.0)
    O = np.einsum("u,ui,uj->ij", w, C, C) - np.diag((w[:, None] * C).sum(axis=0))
    nc = O.sum(axis=1)
    n = nc.sum()
    if data_type == "nominal":
        d2 = 1.0 - np.eye(len(vals))
    elif data_type == "interval":
        d2 = (vals[:, None] - vals[None, :]) ** 2
    elif data_type == "ratio":
        tot = vals[:, None] + vals[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            d2 = np.where(tot == 0, 0.0, ((vals[:, None] - vals[None, :]) / tot) ** 2)
    else:  # ordinal
        cs = np.concatenate([[0.0], np.cumsum(nc)])
        i = np.arange(len(vals))
        lo = np.minimum(i[:, None], i[None, :])
        hi = np.maximum(i[:, None], i[None, :])
        d2 = (cs[hi + 1] - cs[lo] - (nc[:, None] + nc[None, :]) / 2.0) ** 2
        np.fill_diagonal(d2, 0.0)
    de = float((np.outer(nc, nc) * d2).sum())
    if de <= 1e-12:
        return np.nan
    return float(1.0 - (n - 1.0) * float((O * d2).sum()) / de)


def _krippendorff_alpha(X, data_type="nominal", n_boot=200, alpha=0.05):
    """Krippendorff's alpha for any number of raters and any data type,
    with missing ratings allowed.

    X: (N_units x k_raters) array, NaN = missing
    data_type: 'nominal' | 'ordinal' | 'interval' | 'ratio'
    n_boot: number of bootstrap resamples for the (1 - alpha) CI (0 = skip CI)

    Point estimate: alpha = 1 - D_o/D_e computed through the coincidence
    matrix of pairable values (values from units rated by >= 2 raters);
    see _kripp_alpha_from_counts for the formulas and distance metrics
    (Krippendorff, K. (2004). Content Analysis: An Introduction to Its
    Methodology, 2nd ed., Sage, Ch. 11). Verified equal, to within
    floating-point precision, to the independent `krippendorff` package
    for all four metrics, with and without missing data.

    Confidence interval: percentile bootstrap over UNITS -- pairable units
    are resampled with replacement (each unit keeps all of its ratings),
    alpha is recomputed on every resample, and the alpha/2 and 1-alpha/2
    percentiles of the resampled values are reported (Efron, B., &
    Tibshirani, R.J. (1993). An Introduction to the Bootstrap. Chapman &
    Hall; the same unit-level bootstrap evaluated for alpha by Zapf, A.,
    Castell, S., Morawietz, L., & Karch, A. (2016). BMC Medical Research
    Methodology, 16, 93). Fixed seed (42) for reproducibility.

    CORRECTION (2026-10-23): this function previously used the pair-pool
    bootstrap described by Krippendorff (2016, "Bootstrapping distributions
    for Krippendorff's alpha"), which resamples individual pairs of values
    as if they were independent. A data-based Monte Carlo check (1,000
    resamples of each bundled sample dataset, 200 bootstrap draws each)
    found that interval's coverage of the population alpha close to
    nominal with two raters (0.926-0.951) but far below nominal with four
    raters per unit (0.697 nominal, 0.775 ordinal, 0.766 interval), because
    the C(k,2) pairs within a unit are not independent. The unit-level
    bootstrap above gave coverage of 0.930-0.946 in the same check
    (1,000 bootstrap draws).
    """
    try:
        C, vals = _kripp_counts(X)
        if len(vals) == 0:
            return np.nan, (np.nan, np.nan)
        C = C[C.sum(axis=1) >= 2]          # pairable units only
        if C.shape[0] < 1:
            return np.nan, (np.nan, np.nan)
        alpha_val = _kripp_alpha_from_counts(C, vals, data_type)
        if np.isnan(alpha_val) or n_boot == 0 or C.shape[0] < 2:
            return alpha_val, (np.nan, np.nan)
        rng = np.random.default_rng(42)
        n_u = C.shape[0]
        reps = np.array([_kripp_alpha_from_counts(C[rng.integers(0, n_u, n_u)], vals, data_type)
                         for _ in range(n_boot)])
        reps = reps[~np.isnan(reps)]
        if len(reps) < max(20, n_boot // 2):
            return alpha_val, (np.nan, np.nan)
        ci_lo, ci_hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return alpha_val, (float(ci_lo), float(ci_hi))
    except Exception:
        return np.nan, (np.nan, np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# NEW INTER-RATER AGREEMENT METHODS
# All formulas verified against primary references.
# ══════════════════════════════════════════════════════════════════════════════

def _percentage_agreement(X, alpha=0.05):
    """Percentage Agreement — proportion of exact pairwise agreements,
    both POOLED (across all units and rater-pairs combined) and PER-UNIT
    (one agreement proportion per rated unit, matching the variable a
    researcher would construct for descriptive-statistics analysis).

    Pooled formula: PA = (1 / (N * C(k,2))) * Σ_i Σ_{j<l} I(X_ij == X_il)
    Per-unit formula (unit i): PA_i = (1/C(k_i,2)) * Σ_{j<l} I(X_ij==X_il),
    where k_i is the number of non-missing raters for unit i (so units
    with partial missingness still contribute a valid per-unit value
    based on however many pairs they do have).

    (1 - alpha) CI: unit-level one-sample t-interval on the N per-unit PA_i
    values (NOT a pair-level Wald interval -- see the CI CORRECTION
    comment in the function body for the Monte Carlo evidence, 85.1%
    vs. 94.2% empirical coverage, motivating this).

    Reference: Stemler (2004). A comparison of consensus, consistency,
    and measurement approaches to estimating interrater reliability.
    Practical Assessment, Research & Evaluation, 9(4).

    Returns a dict:
      pa, ci_lo, ci_hi          -- pooled proportion + its (1-alpha) one-
                                    sample t-interval (see CI CORRECTION
                                    comment below -- NOT a Wald interval)
      per_unit                  -- ndarray, one PA value per unit (NaN for
                                    units with <2 valid ratings)
      n_units_valid             -- count of units contributing >=1 valid pair
      mean, sd, minimum, maximum -- descriptive statistics of per_unit,
                                    computed via the standard sample-based
                                    descriptive-statistics convention for a
                                    per-unit agreement variable (sd uses
                                    ddof=1, the standard unbiased-estimator
                                    convention)
    """
    n, k = X.shape
    empty = {"pa": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
             "per_unit": np.array([]), "n_units_valid": 0,
             "mean": np.nan, "sd": np.nan, "minimum": np.nan, "maximum": np.nan}
    if k < 2:
        return empty
    agree_total = 0; pair_total = 0
    per_unit = np.full(n, np.nan)
    for i in range(n):
        unit_agree = 0; unit_pairs = 0
        for j in range(k):
            for l in range(j+1, k):
                if not (np.isnan(X[i,j]) or np.isnan(X[i,l])):
                    is_agree = int(X[i,j] == X[i,l])
                    agree_total += is_agree
                    pair_total  += 1
                    unit_agree  += is_agree
                    unit_pairs  += 1
        if unit_pairs > 0:
            per_unit[i] = unit_agree / unit_pairs
    if pair_total == 0:
        return empty
    PA = agree_total / pair_total

    valid_units = per_unit[~np.isnan(per_unit)]
    n_valid = len(valid_units)

    # CI CORRECTION: an earlier revision computed the 95% CI via a naive
    # Wald proportion interval using pair_total (= N*C(k,2)) as the
    # effective sample size, treating every within-unit pair as an
    # INDEPENDENT observation. This is statistically invalid: the C(k,2)
    # pairs from the SAME unit are not independent draws -- they share
    # that unit's own underlying agreement tendency (a unit where raters
    # happen to agree tends to have ALL of its pairs agree together, not
    # independently), a clustering/design-effect structure the naive
    # pair-level variance ignores. Direct Monte Carlo verification (3,000
    # simulated trials, clustered-agreement data structure) found the old
    # CI's empirical coverage to be 85.1% against a nominal 95% target --
    # a substantial, confirmed under-coverage, not a minor rounding issue.
    #
    # Corrected approach: treat the N independent UNITS (not the N*C(k,2)
    # correlated pairs) as the sampling unit, using the standard one-sample
    # t-interval for the mean of the N per-unit agreement proportions
    # already computed above. The SAME Monte Carlo procedure applied to
    # this corrected interval found empirical coverage of 94.2%, closely
    # matching the nominal 95% target.
    if n_valid > 1:
        se_pa_units = float(valid_units.std(ddof=1) / np.sqrt(n_valid))
        # BUG FIX (2026-10-21): t_crit was hardcoded to scipy_stats.t.ppf
        # (0.975, ...) -- always the 95% critical value regardless of the
        # alpha the user selected in the sidebar (the function did not
        # even accept an alpha parameter before this fix). Now uses the
        # actual requested confidence level.
        t_crit = float(scipy_stats.t.ppf(1 - alpha / 2, n_valid - 1))
        ci_lo = max(0.0, float(PA - t_crit * se_pa_units))
        ci_hi = min(1.0, float(PA + t_crit * se_pa_units))
    else:
        ci_lo, ci_hi = np.nan, np.nan

    return {
        "pa": float(PA), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "per_unit": per_unit, "n_units_valid": int(n_valid),
        "mean": float(valid_units.mean()) if n_valid > 0 else np.nan,
        "sd": float(valid_units.std(ddof=1)) if n_valid > 1 else np.nan,
        "minimum": float(valid_units.min()) if n_valid > 0 else np.nan,
        "maximum": float(valid_units.max()) if n_valid > 0 else np.nan,
    }


def _dual_ci(estimate, se, n, alpha=0.05):
    """Computes BOTH a z-based (asymptotic normal) and a t-based
    (df = n-1) Wald confidence interval for a sample-estimated standard
    error, for the kappa-family coefficients whose SE is derived either
    analytically (delta method) or via jackknife resampling.

    Neither basis is presented as universally "more correct" -- both
    are legitimate, but their theoretical grounding differs:

    z-basis: follows DIRECTLY from the delta method's own asymptotic
    derivation. The delta method's central result is that the estimator
    converges in distribution to a NORMAL limit (not Student-t) as
    n -> infinity (Doob, 1935; general delta-method theory). z is
    therefore the form that follows directly from the SE formula's own
    theoretical origin, not an arbitrary default.

    z-basis for the analytic case (Brennan-Prediger only): the SE is
    Cohen's (1960) large-sample approximation sqrt(Po(1-Po)/(N(1-Pe)^2)),
    Educational and Psychological Measurement, 20(1), 37-46. It treats Pe as
    a fixed constant, which is exact for Brennan-Prediger because Pe = 1/q
    is fixed by the declared number of categories. (Scott's pi no longer
    uses this formula: its SEs now come from the Fleiss-kappa variance
    literature -- see _scott_pi.)

    t-basis (df=n-1): a more conservative alternative. For
    jackknife-derived SEs (Gwet's AC1/AC2), this has independent,
    well-established support in the general jackknife resampling
    literature -- Miller, R.G. (1974). The jackknife: A review.
    Biometrika, 61(1), 1-15; Efron, B., & Tibshirani, R.J. (1993). An
    Introduction to the Bootstrap. Chapman & Hall, Ch. 11 -- both of
    which discuss the "jackknife t-interval" as an established,
    conservative convention distinct from the plain jackknife variance
    estimate. The t-basis is also the convention of Gwet's (2014)
    inference framework as implemented in Stata's kappaetc command
    (Klein, D., 2018. Implementing a general framework for assessing
    interrater agreement in Stata. The Stata Journal, 18(4), 871-901).
    For Scott's pi (= Fleiss' kappa with two raters) it is the interval
    Gwet (2021) evaluated for the general Fleiss-kappa variance. For
    Brennan-Prediger it rests on the Gwet/Klein convention alone.

    Returns dict: ci_lo_z, ci_hi_z, ci_lo_t, ci_hi_t, df, z_critical,
    t_critical.
    """
    if (se is None or (isinstance(se, float) and np.isnan(se))
            or n is None or n < 2):
        return {"ci_lo_z": np.nan, "ci_hi_z": np.nan,
                "ci_lo_t": np.nan, "ci_hi_t": np.nan,
                "df": max((n or 0) - 1, 0), "z_critical": np.nan,
                "t_critical": np.nan}
    df = n - 1
    z_crit = float(scipy_stats.norm.ppf(1 - alpha / 2))
    t_crit = float(scipy_stats.t.ppf(1 - alpha / 2, df)) if df > 0 else np.nan
    return {
        "ci_lo_z": float(estimate - z_crit * se),
        "ci_hi_z": float(estimate + z_crit * se),
        "ci_lo_t": float(estimate - t_crit * se) if not np.isnan(t_crit) else np.nan,
        "ci_hi_t": float(estimate + t_crit * se) if not np.isnan(t_crit) else np.nan,
        "df": df, "z_critical": z_crit, "t_critical": t_crit,
    }


def _scott_pi(r1, r2, alpha=0.05):
    """Scott's Pi (pi) -- inter-rater agreement for 2 raters, nominal/dichotomous.

    Point estimate (Scott, 1955):
        pi = (P_o - P_e) / (1 - P_e),  P_e = sum_c ((p_c(r1) + p_c(r2)) / 2)^2,
    i.e. chance agreement from the AVERAGED marginal proportions (Cohen's
    kappa uses their product instead).

    With two raters Scott's pi is identical to Fleiss' (1971) kappa, so its
    standard errors are taken from the Fleiss-kappa variance literature:
      se_null -- null-hypothesis SE of Fleiss, Nee & Landis (1979), used
                 ONLY for z and the two-sided p-value of H0: pi = 0;
      se      -- general-purpose (non-null) SE of Gwet (2021), Equations
                 4 and 12, used for the confidence intervals (z-basis and
                 t-basis with df = n - 1, the interval Gwet (2021)
                 evaluated).

    CORRECTION (2026-10-23): this function previously used Cohen's (1960)
    approximation sqrt(P_o(1 - P_o) / (n(1 - P_e)^2)), which treats P_e as a
    fixed constant. For Scott's pi P_e is estimated from the data, and a
    Monte Carlo check with independent raters showed that SE exceeding the
    true sampling SD by a factor that grows with prevalence skew (about 1.0
    at a 50/50 split, 1.3 at 75/25, 1.7 at 85/15 and 2.2 at 90/10), i.e.
    the intervals were too wide. The replacement SE agrees with the irrCAC
    package's Fleiss-kappa SE for two raters.

    References:
    - Scott, W. A. (1955). Public Opinion Quarterly, 19(3), 321-325.
    - Fleiss, J. L. (1971). Psychological Bulletin, 76(5), 378-382.
    - Fleiss, J. L., Nee, J. C. M., & Landis, J. R. (1979). Psychological
      Bulletin, 86(5), 974-977.
    - Gwet, K. L. (2021). Educational and Psychological Measurement,
      81(4), 781-790.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    mask = ~(np.isnan(r1) | np.isnan(r2))
    r1, r2 = r1[mask], r2[mask]
    n = len(r1)
    empty = {"pi": np.nan, "se": np.nan, "se_null": np.nan, "z": np.nan,
             "p": np.nan, "ci_lo_z": np.nan, "ci_hi_z": np.nan,
             "ci_lo_t": np.nan, "ci_hi_t": np.nan, "ci_df": max(n - 1, 0),
             "Po": np.nan, "Pe": np.nan, "n": n, "label": "N/A"}
    if n < 2:
        return empty
    cats = np.union1d(r1, r2)
    # unit-by-category counts (each unit has exactly 2 ratings)
    table = np.stack([(r1 == c).astype(float) + (r2 == c).astype(float) for c in cats], axis=1)
    p_j = table.sum(axis=0) / (2.0 * n)            # averaged marginal proportions
    P_i = (table * (table - 1)).sum(axis=1) / 2.0  # 1 if the two ratings agree, else 0
    Po = float(P_i.mean())
    Pe = float(np.sum(p_j ** 2))
    if abs(1 - Pe) < 1e-12:
        return dict(empty, Po=Po, Pe=Pe)
    pi_val = (Po - Pe) / (1 - Pe)
    # Null-hypothesis SE (Fleiss, Nee & Landis, 1979) with k = 2 raters
    S1 = float(np.sum(p_j * (1 - p_j)))
    S2 = float(np.sum(p_j * (1 - p_j) * (1 - 2 * p_j)))
    var0 = (2.0 / (n * 2 * 1 * max(S1 ** 2, 1e-12))) * (S1 ** 2 - S2)
    se_null = float(np.sqrt(max(var0, 0.0)))
    # General-purpose SE (Gwet, 2021): linearised unit contributions
    pe_i = (table * p_j[None, :]).sum(axis=1) / 2.0
    kap_i = (P_i - Pe) / (1 - Pe)
    kap_star = kap_i - 2 * (1 - pi_val) * (pe_i - Pe) / (1 - Pe)
    se = float(np.sqrt(np.sum((kap_star - pi_val) ** 2) / (n * (n - 1))))
    z = pi_val / se_null if se_null > 1e-12 else np.nan
    p = float(2 * scipy_stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
    _dual = _dual_ci(pi_val, se, n, alpha)
    return {"pi":      pi_val,
            "se":      se,
            "se_null": se_null,
            "z":       z if not np.isnan(z) else np.nan,
            "p":       p if not np.isnan(p) else np.nan,
            "ci_lo_z": _dual["ci_lo_z"], "ci_hi_z": _dual["ci_hi_z"],
            "ci_lo_t": _dual["ci_lo_t"], "ci_hi_t": _dual["ci_hi_t"],
            "ci_df":   _dual["df"],
            "Po": Po, "Pe": Pe,
            "n": n, "label": kappa_label(pi_val)}


def _brennan_prediger(r1, r2, alpha=0.05, q_declared=None):
    """Brennan-Prediger Coefficient (κ_BP).

    Chance agreement is fixed at P_e = 1/q, the agreement expected if
    raters assigned the q declared categories uniformly at random. P_e
    therefore does not depend on the observed marginal distributions.
    Note: 1/q is the SMALLEST chance term of the Scott/Fleiss type
    (sum_c p_c^2 >= 1/q), so kappa_BP >= Scott's pi for the same data; it
    is not a lower bound on agreement. (An earlier revision described it as
    "the most conservative" estimate, which was wrong.)

    Formula (Brennan & Prediger, 1981):
        κ_BP = (P_o − 1/q) / (1 − 1/q)
    SE: var(κ_BP) ≈ P_o(1−P_o) / (N × (1−1/q)²)

    SE -- CITATION AND ACCURACY (added 2026-10-21, as part of a repo-wide
    audit): this formula is Cohen, J. (1960)'s original large-sample SE
    approximation for kappa-family coefficients (A coefficient of
    agreement for nominal scales. Educational and Psychological
    Measurement, 20(1), 37-46), adapted here to Brennan-Prediger's own
    Pe=1/q. It had previously gone uncited in this docstring, and the
    shared _dual_ci()/_dual_ci_note() machinery that builds this
    coefficient's confidence interval had wrongly attributed it to
    Fleiss, Cohen, & Everitt (1969) instead (now corrected -- see
    _dual_ci()'s own docstring for the full citation-correction note).

    Pe = 1/q is a FIXED design constant (the declared category count), not
    a quantity estimated from the data, so Cohen's (1960) fixed-Pe
    assumption holds exactly for this coefficient. Independently re-verified by
    direct Monte Carlo (not assumed to transfer from Scott's pi's
    result): (a) resampling from sample-instats_Dichotomous.csv's real
    R1/R2 joint distribution, q=2, N=86/200/500/1500, 10,000 replicates
    each -- SE/true-SD ratio 1.0076/0.9820/1.0065/1.0020; (b) 3-category
    nominal data, independent raters (near-null, the harder case), q=3,
    same N's -- ratio 0.9836/0.9978/0.9992/1.0001. Both setups show the
    ratio close to 1.0 at every N with no systematic drift -- the
    expected signature of a correctly-specified large-sample SE, not a
    biased approximation.

    q -- THEORETICAL BASIS (verified against primary/authoritative source):
    q is the number of categories in the measurement instrument's coding
    scheme -- the space a rater selects among -- not merely the count of
    distinct values that happen to appear in a given sample. Brennan &
    Prediger (1981, p. 693) themselves describe q as "the number of
    categories"; this is the same q-family as Bennett, Alpert, & Goldstein's
    (1954) S coefficient and Gwet's AC1, both of which use an a priori,
    declared category count (see _gwet_ac1's docstring for the fuller
    theoretical argument, independently verified via Gwet's Handbook of
    Inter-Rater Reliability, 4th ed., Ch. 4). If q_declared is not supplied, this
    function falls back to the count of distinct observed values, WHICH
    UNDERSTATES q -- and therefore OVERSTATES Pe and UNDERSTATES kappa_bp
    -- whenever the instrument has a valid category that happens not to
    appear in this specific sample. q_declared should be supplied whenever
    the true instrument scale is known.

    Reference:
    Brennan, R. L., & Prediger, D. J. (1981). Coefficient kappa: Some uses,
    misuses, and alternatives. Educational and Psychological Measurement, 41(3).
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    mask = ~(np.isnan(r1) | np.isnan(r2))
    r1, r2 = r1[mask], r2[mask]
    n = len(r1)
    if n < 2:
        return {"kappa_bp": np.nan, "se": np.nan, "z": np.nan,
                "p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n": n, "q": 0, "q_observed": 0, "q_declared": q_declared, "label": "N/A"}
    cats = sorted(int(c) for c in np.union1d(r1, r2))
    # q resolved via the SAME shared helper AC1/AC2 use (_resolve_q), so
    # Brennan-Prediger agrees with them on q for identical data -- required
    # for the auto-comparison table (below, built by run_multi_irar) that
    # presents BP/Cohen's kappa/AC1 side by side as sharing one Po with
    # different Pe models: that comparison is only fair if q also matches.
    q, q_observed, has_gaps, _lo = _resolve_q(cats, q_declared, scale_min=None)
    if q < 2:
        return {"kappa_bp": np.nan, "se": np.nan, "z": np.nan,
                "p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n": n, "q": q, "q_observed": q_observed, "q_declared": q_declared, "label": "N/A"}
    Po    = float(np.mean(r1 == r2))
    Pe    = 1.0 / q
    kbp   = (Po - Pe) / (1 - Pe)
    se    = float(np.sqrt(Po*(1-Po) / (n * (1-Pe)**2)))
    z     = kbp / se if se > 1e-12 else np.nan
    p     = float(2 * scipy_stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
    _dual = _dual_ci(kbp, se, n, alpha)
    return {"kappa_bp": kbp,
            "se":       se,
            "z":        z if not np.isnan(z) else np.nan,
            # Not rounded here -- preserve precision for genuinely tiny p.
            "p":        p if not np.isnan(p) else np.nan,
            "ci_lo_z":  _dual["ci_lo_z"], "ci_hi_z": _dual["ci_hi_z"],
            "ci_lo_t":  _dual["ci_lo_t"], "ci_hi_t": _dual["ci_hi_t"],
            "ci_df":    _dual["df"],
            "Po": Po, "Pe": Pe,
            "n": n, "q": q, "q_observed": q_observed, "q_declared": q_declared,
            "has_gaps": has_gaps,
            "label": kappa_label(kbp)}


def _resolve_q(cats, q_declared, scale_min=None, require_span=False):
    """Resolves the effective category count 'q' shared by Gwet's AC1,
    Gwet's AC2, and Brennan-Prediger -- all three define q as THE
    INSTRUMENT'S CATEGORY COUNT (an a priori property of the measurement
    scale: Gwet, 2014, Ch.4; Brennan & Prediger, 1981, p.693), not merely
    the count of distinct values that happen to appear in one sample.
    Factored out so all three coefficients agree on q for the SAME data
    -- a hard requirement for their documented, regression-tested
    identities (AC2 with identity weights == AC1 exactly; the BP/Cohen/
    AC1 comparison table, which presents all three as sharing one Po
    with different Pe models and is only a fair comparison if they also
    share q).

    BUG FIX (2026-10-21): a prior revision let q default to q_observed =
    len(cats) -- a plain COUNT of distinct observed values -- whenever
    q_declared was absent. This is only equal to the numeric SPAN those
    values occupy (max-min+1) when the observed category labels are
    perfectly consecutive integers (the normal case for dense Likert/
    rubric-style data, e.g. 2,3,4,5). For sparse numeric labels with
    gaps (e.g. raw scores 58,59,60,...,64,66,67,... with some integers
    never observed), the count UNDERSTATES the span. _gwet_ac2 indexes
    its weight matrix by RAW VALUE POSITION (cat_idx[c] = c - lo), which
    requires q >= span -- using the plain count there produced a live
    crash ("IndexError: index N out of bounds...") on exactly this kind
    of data declared "ordinal". The fix widens q to cover the full span
    whenever it would otherwise be too small; the "extra", never-observed
    integers in between are treated exactly like any other declared-but-
    unobserved category elsewhere in this module (pi_c=0, so they add
    nothing to Σπk(1-πk), but DO count as one of the q categories in the
    (q-1)/(q(q-1)) normalizer -- the same treatment already given to a
    genuinely-declared Likert category absent from a given sample).

    CORRECTION (2026-10-21): the span-widening fix above is only a valid
    requirement for AC2's own weight-matrix indexing -- AC1 and Brennan-
    Prediger use q only as a plain scalar in their Pe formulas and have
    no positional-indexing requirement at all, so span-widening q for
    them is not a fix of anything, only a new bug: it silently overrides
    a correctly-declared q with the numeric SPAN of whatever integer
    *codes* happen to label the categories -- meaningless for NOMINAL
    data, where the codes are arbitrary labels, not a measurement scale.
    Verified: the same underlying pattern, q_declared=3 with category
    codes {1,2,3}, relabeled to the semantically-identical {10,20,30},
    left q=3 for the first (correct) but produced q=21 (=30-10+1) for
    the second -- AC1 and Brennan-Prediger results changed completely
    (e.g. AC1 0.4419->0.6123 on one test dataset) despite the underlying
    agreement pattern being unchanged. The parameter `require_span`
    (default False) now gates the span-widening behaviour so it applies
    ONLY where it is actually needed (AC2's call site passes
    require_span=True); AC1 and Brennan-Prediger now use the plain
    declared-or-observed count with no span involvement, exactly as
    their own formulas require.

    Returns (q, q_observed, has_gaps, lo):
      q_observed -- count of distinct observed category values.
      lo         -- scale_min if declared, else the lowest observed value.
      has_gaps   -- True only if the OBSERVED values themselves skip an
                    integer somewhere between the lowest and highest value
                    actually observed (observed_span = max(cats)-min(cats)+1
                    > q_observed) -- e.g. raw scores 58,...,64,66,...,
                    skipping 65. Deliberately does NOT fire merely because
                    a user-declared scale_min/q_declared extends the range
                    below/beyond what this particular sample happened to
                    use (e.g. a declared 2-5 Likert scale where this
                    sample only used 3,4,5) -- that is normal, by-design
                    "declared-but-unobserved category" behaviour, not a
                    sign the data doesn't fit a q-category model. Intended
                    for an optional UI advisory ("this data may not suit
                    an ordinal/AC2 model") that should fire on genuinely
                    gapped raw numeric data, not on ordinary Likert data
                    with an unused declared category.
      q          -- declared-or-observed count; widened to span-from-lo
                    ONLY when require_span=True (AC2's weight-matrix
                    sizing). For AC1/Brennan-Prediger (require_span=False,
                    the default), q is never widened by raw code span.
    """
    cats = sorted(cats)
    q_observed = len(cats)
    if q_observed == 0:
        return 0, 0, False, 0
    lo = int(scale_min) if scale_min is not None else int(cats[0])
    span = int(cats[-1]) - lo + 1
    # VERIFIED: int(q_declared) below silently truncates a non-integer
    # q_declared (e.g. 5.7 -> 5) with no warning -- flagged as a possible
    # bug and checked against every UI path in views/interrateragreement.py
    # that supplies q_declared. The nominal branch feeds it from a
    # st.number_input created with step=1 (an int, not a float) and no
    # custom `format=` override, which makes Streamlit constrain the widget
    # to integer-only entry and return a Python int; the dichotomous branch
    # sets it to the literal 2; the ordinal branch derives it from integer
    # scale minimum and maximum. A user therefore cannot enter a fractional
    # value through the UI, so int(q_declared) is unreachable with a
    # genuinely fractional value there; it is retained only as defensive
    # input hygiene for any future or programmatic (non-UI) caller that
    # might pass a float.
    q_candidate = int(q_declared) if (q_declared is not None and q_declared >= q_observed) else q_observed
    q = max(q_candidate, span) if require_span else q_candidate
    observed_span = int(cats[-1]) - int(cats[0]) + 1   # ignores scale_min padding
    has_gaps = observed_span > q_observed
    return q, q_observed, has_gaps, lo


def _gwet_ac1(X, alpha=0.05, q_declared=None):
    """Gwet's AC1 — for 2 or more raters, nominal/dichotomous data.

    More stable than Cohen's κ when marginal distributions are extreme
    (prevalence paradox). Uses jackknife variance estimation.

    Formula (Gwet, 2008, 2014):
        P_e(AC1) = (1/(q−1)) × Σ_c π_c(1−π_c)
        where π_c = (1/(N×k)) × Σ_i Σ_j I(X_ij = c)  [grand marginal proportion]
        AC1 = (P_o − P_e) / (1 − P_e)

    For k=2: P_o = proportion of matching pairs.
    For k≥2: P_o = (1/(N×k×(k-1)/2)) × Σ_i Σ_{j<l} I(X_ij == X_il)
    Variance: delete-one-unit jackknife (Efron & Tibshirani, 1993, Ch. 11).

    q -- THEORETICAL BASIS (verified directly against the primary source):
    Gwet, Handbook of Inter-Rater Reliability (4th ed., 2014), Advanced
    Analytics LLC -- Part II, "Chance-Corrected Agreement Coefficients" --
    Ch. 4 ("Constructing Agreement Coefficients: AC1 and Aickin's alpha"),
    Sec. 4.2 ("Gwet's AC1 and Aickin's alpha for two Raters"), Sec. 4.5
    ("Calculating AC1 for three Raters or More"), and Sec. 4.6 ("AC2: the
    AC1 Coefficient for Ordinal and Interval Data"): AC1/AC2 is built on a
    q-category nominal (or, for AC2, ordinal/interval) coding scheme fixed
    by the measurement instrument -- i.e. q is the size of the category
    space a rater selects among (an instrument property, fixed a priori),
    NOT the count of distinct values that happen to appear in a specific
    sample. (The chance-agreement denominator (q-1) used below matches the
    book's own published erratum -- the original 2014 printing had a typo,
    q*(q-1), later corrected by the author to (q-1); see agreestat.com/
    book4/errors_4ed.html, correction to Eq. 2.4.6.)
    If q_declared is not supplied, this function falls back to the count of
    distinct observed values, WHICH UNDERSTATES q whenever the instrument
    has a valid category absent from this specific sample -- inflating Pe
    and understating AC1. The Σ_c π_c(1−π_c) sum itself is unaffected by
    unobserved categories (each contributes exactly 0), so only the
    (q−1) denominator requires correction; no reconstruction of the
    unobserved category identities is needed.

    References:
    Gwet, K. L. (2008). Computing inter-rater reliability and its variance
    in the presence of high agreement. British Journal of Mathematical and
    Statistical Psychology, 61(1), 29–48.
    Gwet, K. L. (2014). Handbook of Inter-Rater Reliability (4th ed.),
    Ch. 4. Advanced Analytics, LLC.
    """
    X = np.array(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    if k < 2 or n < 3:
        return {"ac1": np.nan, "se": np.nan, "z": np.nan,
                "p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n": n, "k": k, "label": "N/A"}

    cats = sorted(set(X[~np.isnan(X)].flatten().astype(int).tolist()))
    q, q_observed, has_gaps, _lo = _resolve_q(cats, q_declared, scale_min=None)
    if q < 2:
        return {"ac1": np.nan, "se": np.nan, "z": np.nan,
                "p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n": n, "k": k, "label": "N/A"}

    def _compute_ac1(X_):
        n_, k_ = X_.shape
        # Observed agreement per unit
        gamma = np.zeros(n_)
        for i in range(n_):
            row = X_[i, :]
            valid = row[~np.isnan(row)]
            m = len(valid)
            if m < 2: continue
            pairs = m*(m-1)/2
            agree = sum(1 for a in range(m) for b in range(a+1,m)
                        if valid[a]==valid[b])
            gamma[i] = agree / pairs
        Po = float(np.mean(gamma))
        # Grand marginal proportions (sum over OBSERVED cats only -- any
        # unobserved declared category has pi_c=0 and contributes nothing
        # to the sum below, so only the (q-1) denominator needs q_declared)
        all_valid = X_[~np.isnan(X_)]
        pi_c = np.array([np.sum(all_valid==c)/len(all_valid) for c in cats])
        Pe = float((1/(q-1)) * np.sum(pi_c*(1-pi_c)))
        if abs(1-Pe) < 1e-12: return np.nan
        return (Po - Pe)/(1 - Pe)

    ac1_full = _compute_ac1(X)

    # Jackknife variance
    ac1_jack = np.array([_compute_ac1(np.delete(X, i, axis=0))
                         for i in range(n)])
    ac1_jack = ac1_jack[~np.isnan(ac1_jack)]
    if len(ac1_jack) < 5:
        se = np.nan
    else:
        var_j = ((n-1)/n) * np.sum((ac1_jack - np.mean(ac1_jack))**2)
        se    = float(np.sqrt(var_j))

    z     = ac1_full/se if (not np.isnan(se) and se>1e-12) else np.nan
    p     = float(2 * scipy_stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
    _dual = _dual_ci(ac1_full, se, n, alpha)

    return {"ac1":    float(ac1_full) if not np.isnan(ac1_full) else np.nan,
            "se":     se if not np.isnan(se) else np.nan,
            "z":      z if not np.isnan(z) else np.nan,
            # Not rounded here -- preserve precision for genuinely tiny p.
            "p":      p if not np.isnan(p) else np.nan,
            "ci_lo_z": _dual["ci_lo_z"], "ci_hi_z": _dual["ci_hi_z"],
            "ci_lo_t": _dual["ci_lo_t"], "ci_hi_t": _dual["ci_hi_t"],
            "ci_df":  _dual["df"],
            "n": n, "k": k, "q": q, "q_observed": q_observed, "q_declared": q_declared,
            "has_gaps": has_gaps,
            "label":  kappa_label(ac1_full) if not np.isnan(ac1_full) else "N/A"}


def _build_ac2_weight_matrix(cats, q, weights, scale_min=None):
    """Builds the q x q weight matrix for Gwet's AC2 over the FULL declared
    category range [lo, lo+q-1] (see the IMPORTANT note below).

    Five weighting schemes, all taken from the weight family defined in
    Gwet, K.L. (2014). Handbook of Inter-Rater Reliability (4th ed.),
    Advanced Analytics, LLC, and each checked numerically against the
    weight functions of an independent implementation of that family
    (the irrCAC package's Weights class):

      identity  -- w=1 if c=c' else 0. Reduces AC2 exactly to AC1.
      linear    -- w = 1 - |c-c'|/(q-1) (Cicchetti & Allison, 1971).
      quadratic -- w = 1 - (c-c')^2/(q-1)^2 (Fleiss & Cohen, 1973).
      radical   -- w = 1 - sqrt(|c-c'|/(q-1)).
      ordinal   -- rank-based ordinal weights: with n_cc' = |c-c'| + 1,
        M_cc' = n_cc'(n_cc'-1)/2 (the number of category pairs spanned
        between c and c'), w = 1 - M_cc'/max(M). Depends only on the
        category positions, never on the observed rating frequencies.
        (Corrected 2026-10-23: this scheme previously used a frequency-
        based, Krippendorff-alpha-style distance, which is not one of the
        weights defined for AC2 in Gwet (2014); replaced by the ordinal
        weights that Gwet (2014) does define.)

    IMPORTANT (corrected in this revision): the weight matrix is built
    over the FULL DECLARED category range [lo, lo+q-1] -- q categories,
    q possibly larger than the number actually observed -- not merely
    over the categories that happen to appear in the data. This matters
    because the corrected Pe formula (see _gwet_ac2) requires u = the sum
    of the ENTIRE q x q weight matrix, including cells for declared-but-
    unobserved categories; building the matrix only over observed
    categories was found, during verification against Gwet's identity-
    weights-equals-AC1 property, to silently understate u whenever
    q_declared exceeds the observed category count, producing an
    incorrect Pe. lo defaults to the minimum OBSERVED category if
    scale_min is not supplied.

    Returns: (q x q) ndarray W indexed by position within the FULL
    declared range [lo, lo+q-1] -- i.e. W[i,j] corresponds to categories
    (lo+i) and (lo+j), regardless of whether either was actually observed.
    """
    lo = int(scale_min) if scale_min is not None else int(min(cats))
    full_range = list(range(lo, lo + q))
    n_full = len(full_range)
    W = np.zeros((n_full, n_full))

    if weights == "identity":
        for i in range(n_full):
            W[i, i] = 1.0
        return W
    if weights == "linear":
        v = 1
        for i, ci in enumerate(full_range):
            for j, cj in enumerate(full_range):
                W[i, j] = 1 - (abs(ci - cj) ** v) / ((q - 1) ** v) if q > 1 else 1.0
        return W
    if weights == "radical":
        for i, ci in enumerate(full_range):
            for j, cj in enumerate(full_range):
                W[i, j] = 1 - np.sqrt(abs(ci - cj) / (q - 1)) if q > 1 else 1.0
        return W
    if weights == "ordinal":
        # Gwet (2014) ordinal weights: M_ij = n_ij(n_ij-1)/2 with
        # n_ij = |i-j|+1, then w_ij = 1 - M_ij / max(M).
        for i in range(n_full):
            for j in range(n_full):
                n_ij = abs(i - j) + 1
                W[i, j] = n_ij * (n_ij - 1) / 2.0
        m_max = W.max()
        return 1 - W / m_max if m_max > 0 else np.ones_like(W)
    # default / "quadratic" (Fleiss-Cohen)
    v = 2
    for i, ci in enumerate(full_range):
        for j, cj in enumerate(full_range):
            W[i, j] = 1 - (abs(ci - cj) ** v) / ((q - 1) ** v) if q > 1 else 1.0
    return W


def _gwet_ac2(X, weights="quadratic", alpha=0.05, q_declared=None, scale_min=None):
    """Gwet's AC2 — weighted version of AC1 for ordinal data.

    Supports five weighting schemes (see _build_ac2_weight_matrix for
    individually verified formulas and primary sources): "quadratic"
    (Fleiss-Cohen, default), "linear" (Cicchetti-Allison), "radical",
    "ordinal" (Gwet's rank-based ordinal weights), and "identity"
    (unweighted -- reduces exactly to AC1).

    Formula (Gwet, 2014, Chapter 4, Sec. 4.6, corrected -- see the note
    below), for the label-based schemes:
        w_cc' = 1 − |c−c'|^v / (q−1)^v   (v=2 quadratic, v=1 linear)
        P_o_w = (1/N) Σ_i Σ_{j<l} w(X_ij, X_il) / C(k,2)
        P_e_w = u × Σ_k π_k(1−π_k) / (q(q−1)),
            u = sum of ALL cells of the q×q weight matrix (incl. diagonal)
        AC2 = (P_o_w − P_e_w) / (1 − P_e_w)
    Variance: jackknife.

    IMPORTANT, documented for traceability: an earlier revision of this
    function implemented P_e_w = (1/(q−1)) Σ_c π_c Σ_{c'≠c} w_cc' (summing
    only OFF-diagonal weights per row). This formula was found, during
    verification, NOT to reduce correctly to AC1's own P_e formula for
    "identity" (unweighted) weights -- it collapsed to P_e_w=0 instead of
    matching AC1 exactly, despite AC2 with identity weights being
    mathematically required to equal AC1 (confirmed against an explicit
    cell-by-cell worked formula: Real Statistics Using Excel, "Gwet's AC2
    Basic Concepts"). The corrected formula above was independently
    verified to reduce exactly to AC1's P_e when W=identity (u=q in that
    case, giving P_e_w = q·Σπk(1−πk)/(q(q−1)) = Σπk(1−πk)/(q−1), Gwet's
    own AC1 formula) -- confirmed by permanent regression test.

    q -- THEORETICAL BASIS: same argument as _gwet_ac1 -- q is the
    instrument's declared category count, not merely the count of
    distinct observed values. See _gwet_ac1's docstring for the full,
    source-verified argument (Gwet, Handbook of Inter-Rater Reliability,
    4th ed., Ch. 4, Sec. 4.6). The weight matrix is built over the full
    declared range [lo, lo+q-1] (lo = scale_min when declared), so the
    sum u of all its cells, and the position of every observed category
    within it, reflect the instrument's scale rather than only the
    categories that happen to appear in this sample.

    Reference:
    Gwet, K. L. (2014). Handbook of Inter-Rater Reliability (4th ed.),
    Ch. 4, Sec. 4.6. Advanced Analytics, LLC.
    """
    X = np.array(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    if k < 2 or n < 3:
        return {"ac2": np.nan, "se": np.nan, "z": np.nan,
                "p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n": n, "k": k, "label": "N/A", "weights": weights}

    cats = sorted(set(X[~np.isnan(X)].flatten().astype(int).tolist()))
    # q is resolved via the SAME shared logic AC1 and Brennan-Prediger use
    # (_resolve_q) -- required so all three agree on q for the same data
    # (see _resolve_q's docstring for the full rationale and the bug this
    # fixes: AC2's weight-matrix lookup indexes by raw category VALUE
    # position, which crashes on sparse/gapped numeric labels unless q
    # covers the full observed span, not merely a count of distinct values).
    q, q_observed, has_gaps, lo = _resolve_q(cats, q_declared, scale_min=scale_min,
                                              require_span=True)
    if q < 2:
        return {"ac2": np.nan, "se": np.nan, "z": np.nan,
                "p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n": n, "k": k, "label": "N/A", "weights": weights}

    # NOTE: identity (unweighted) AC2 is, by construction, mathematically
    # identical to AC1 -- confirmed both theoretically (Real Statistics
    # Using Excel's Gwet documentation: "AC1 is the same as AC2 when
    # restricted to categorical data") and, with the corrected Pe_w
    # formula below, ALGEBRAICALLY: when W=identity, u=q, so
    # Pe_w = q*Σπk(1-πk)/(q(q-1)) = Σπk(1-πk)/(q-1), exactly AC1's Pe.
    # This identity now holds naturally from the general formula; no
    # special-case delegation to _gwet_ac1 is required (an earlier
    # revision patched this case via direct delegation, working around
    # a since-corrected error in the previously implemented Pe_w formula).

    W = _build_ac2_weight_matrix(cats, q, weights, scale_min=scale_min)
    # cat_idx maps each OBSERVED category value to its position WITHIN
    # THE FULL DECLARED RANGE [lo, lo+q-1] -- the same indexing basis
    # _build_ac2_weight_matrix uses for W -- not a separate 0..q_observed-1
    # indexing over only the observed categories. 'lo' and 'q' are the
    # same values already computed above (q now safety-widened to cover
    # the full observed span -- see BUG FIX note), so this stays
    # consistent with _build_ac2_weight_matrix's own indexing by
    # construction; recomputing 'lo' separately here was the source of
    # the original indexing bug and has been removed.
    cat_idx = {c: (c - lo) for c in cats}
    # u = sum of ALL cells of the weight matrix (including the diagonal).
    # This is the scalar multiplier in the verified Pe formula below; it
    # is NOT recomputed per replicate since it depends only on W (fixed
    # once cats/q/weights are fixed), not on the specific data realization.
    u_weight_sum = float(W.sum())

    def _compute_ac2(X_):
        n_, k_ = X_.shape
        # Observed weighted agreement
        Po_w = 0.0; cnt = 0
        for i in range(n_):
            row = X_[i, :]
            valid = row[~np.isnan(row)]
            m = len(valid)
            if m < 2: continue
            for a in range(m):
                for b in range(a+1, m):
                    ia = cat_idx.get(int(valid[a]))
                    ib = cat_idx.get(int(valid[b]))
                    if ia is not None and ib is not None:
                        Po_w += W[ia, ib]; cnt += 1
        if cnt == 0: return np.nan
        Po_w /= cnt
        # Grand marginal proportions (over observed cats; see docstring --
        # unobserved declared categories contribute 0 and need no
        # reconstruction here, only the (q-1) normalizer below needs q)
        all_v = X_[~np.isnan(X_)]
        pi_c  = np.array([np.sum(all_v==c)/len(all_v) for c in cats])
        # Expected weighted agreement -- CORRECTED formula, verified
        # against an explicit cell-by-cell worked formula (Real
        # Statistics Using Excel, "Gwet's AC2 Basic Concepts"):
        #     Pe_w = u * Σ_k πk(1-πk) / (q*(q-1))
        #     u = sum of ALL entries of the weight matrix (incl. diagonal)
        # This form was independently confirmed correct by verifying it
        # reduces EXACTLY to AC1's Pe formula for identity weights
        # (u=q when W=I, giving Pe_w = q*Σπk(1-πk)/(q(q-1)) =
        # Σπk(1-πk)/(q-1), Gwet's own AC1 formula) -- unlike the
        # previously implemented formula (Pe_w = (1/(q-1))Σ_c πc Σ_c'≠c
        # w_cc'), which was found NOT to reduce correctly for identity
        # weights (collapsing to 0), indicating it did not correctly
        # match Gwet's actual general formula for ANY weighting scheme.
        sum_pi_1_minus_pi = float(np.sum(pi_c * (1 - pi_c)))
        Pe_w = u_weight_sum * sum_pi_1_minus_pi / (q * (q - 1))
        if abs(1-Pe_w) < 1e-12: return np.nan
        return (Po_w - Pe_w) / (1 - Pe_w)

    ac2_full = _compute_ac2(X)

    # Jackknife
    ac2_jack = np.array([_compute_ac2(np.delete(X, i, axis=0))
                         for i in range(n)])
    ac2_jack = ac2_jack[~np.isnan(ac2_jack)]
    if len(ac2_jack) < 5:
        se = np.nan
    else:
        var_j = ((n-1)/n)*np.sum((ac2_jack-np.mean(ac2_jack))**2)
        se    = float(np.sqrt(var_j))

    z     = ac2_full/se if (not np.isnan(se) and se>1e-12) else np.nan
    p     = float(2 * scipy_stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
    _dual = _dual_ci(ac2_full, se, n, alpha)

    return {"ac2":     float(ac2_full) if not np.isnan(ac2_full) else np.nan,
            "se":      se if not np.isnan(se) else np.nan,
            "z":       z if not np.isnan(z) else np.nan,
            # Not rounded here -- preserve precision for genuinely tiny p.
            "p":       p if not np.isnan(p) else np.nan,
            "ci_lo_z": _dual["ci_lo_z"], "ci_hi_z": _dual["ci_hi_z"],
            "ci_lo_t": _dual["ci_lo_t"], "ci_hi_t": _dual["ci_hi_t"],
            "ci_df":   _dual["df"],
            "n": n, "k": k, "q": q, "q_observed": q_observed, "q_declared": q_declared,
            "weights": weights, "has_gaps": has_gaps,
            "label":   kappa_label(ac2_full) if not np.isnan(ac2_full) else "N/A"}


def _lin_ccc(r1, r2, alpha=0.05):
    """Lin's Concordance Correlation Coefficient (CCC/ρ_c).

    Simultaneously evaluates PRECISION (correlation, ρ) and ACCURACY (how
    close the data are to the line of perfect concordance, C_b). Only valid
    for exactly 2 raters with interval/ratio data.

    Formula (Lin, 1989):
        ρ_c = 2σ_xy / (σ_x² + σ_y² + (μ_x − μ_y)²)
            = ρ × C_b
    where:
        ρ   = Pearson correlation (precision)
        C_b = 2/(v + 1/v + u²)  (accuracy/bias correction factor)
        v   = σ_x/σ_y  (scale shift)
        u   = (μ_x − μ_y)/(σ_x·σ_y)^0.5  (location shift)

    CONFIDENCE INTERVAL -- FULL Lin (1989) large-sample variance formula
    for the Fisher-Z-transformed CCC (an earlier revision of this
    function used a simplified, "matched-means-only" approximation,
    SE=1/sqrt(N-2), which discards the location-shift (u) contribution
    entirely -- this was found, via direct external cross-validation
    against an independent reference calculator on a real dataset
    (N=86), to produce a confidence interval visibly different from the
    reference tool's own reported Fisher-Z-based interval, even though
    the point estimates for CCC, rho, and Cb all matched closely. The
    FULL formula below, obtained from NCSS PASS's technical
    documentation (a professional statistical-software vendor's own
    published technical reference, itself citing Lin 1989/1992, Lin,
    Hedayat, Sinha & Yang 2002, Lin, Hedayat & Wu 2012, and McBride 2005
    directly) and independently reconstructed
    and verified against that same real dataset, reproduced the
    reference tool's reported interval to within 0.001 -- confirming
    this is the correct, complete formula and that the earlier
    simplified approximation was the source of the discrepancy):

        Var[Z(ρ̂_c)] = (1/(n-2)) × [ (1-ρ²)ρ_c² / ((1-ρ_c²)ρ²)
                                    + 2ρ_c³(1-ρ_c)u² / (ρ(1-ρ_c²)²)
                                    − ρ_c⁴u⁴ / (2ρ²(1-ρ_c²)²) ]

    where Z(ρ̂_c) = arctanh(ρ̂_c) is the Fisher transform, and u is the
    SAME location-shift quantity defined above for C_b (verified: at
    u=0 -- i.e. no location bias -- this reduces exactly to the simpler
    "matched-means" special-case formula, (1-ρ²)ρ_c²/((1-ρ_c²)ρ²)/(n-2),
    consistent with multiple independent secondary sources for that
    special case).

    References:
    Lin, L. I. (1989). A concordance correlation coefficient to evaluate
    reproducibility. Biometrics, 45(1), 255–268.
    Lin, L. I. (1992). Assay validation using the concordance correlation
    coefficient. Biometrics, 48(2), 599–604.
    Lin, L., Hedayat, A.S., Sinha, B., & Yang, M. (2002). Statistical
    methods in assessing agreement: Models, issues, and tools. Journal
    of the American Statistical Association, 97(457), 257–270.
    Lin, L., Hedayat, A. S., & Wu, W. (2012). Statistical Tools for
    Measuring Agreement. Springer.
    McBride, G. B. (2005). A proposal for strength-of-agreement criteria
    for Lin's Concordance Correlation Coefficient. NIWA Client Report:
    HAM2005-062.
    Carrasco, J. L., & Jover, L. (2003). Estimating the generalized
    concordance correlation coefficient through variance components.
    Statistics in Medicine, 22(1), 163–175.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    mask = ~(np.isnan(r1)|np.isnan(r2))
    r1, r2 = r1[mask], r2[mask]
    n = len(r1)
    if n < 3:
        return {"ccc": np.nan, "rho": np.nan, "Cb": np.nan,
                "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n": n, "label": "N/A"}

    # BUG FIX (2026-10-21): sx2/sy2/sxy previously used ddof=1 (the n-1,
    # "sample"/unbiased divisor). Lin's (1989) original CCC formula is
    # defined using POPULATION moments (divisor n, ddof=0) throughout --
    # using ddof=1 here produces a point estimate that drifts slightly
    # from Lin's own published formula (verified: ~0.0008 absolute
    # difference on this module's own N=86 real dataset). ddof=0 matches
    # Lin (1989) exactly and keeps the point estimate, rho, Cb, u, and v
    # all internally consistent with the same population-moment basis
    # the CI variance formula below (also from Lin's framework) assumes.
    mx, my = float(np.mean(r1)), float(np.mean(r2))
    sx2  = float(np.var(r1, ddof=0))
    sy2  = float(np.var(r2, ddof=0))
    sxy  = float(np.cov(r1, r2, ddof=0)[0, 1])
    denom = sx2 + sy2 + (mx - my)**2
    if denom < 1e-12:
        return {"ccc": np.nan, "rho": np.nan, "Cb": np.nan,
                "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n": n, "label": "N/A"}

    ccc = 2*sxy / denom
    rho = float(np.corrcoef(r1, r2)[0, 1])

    # Explicit u (location shift) and v (scale shift) -- previously only
    # implicit (Cb was backed out as ccc/rho without materialising these
    # named quantities); both are needed directly by the full CI variance
    # formula below, and both are independently meaningful diagnostics
    # (v: is one method's spread larger than the other's; u: how large
    # is the mean gap relative to the pooled scale).
    sx, sy = float(np.sqrt(sx2)), float(np.sqrt(sy2))
    v_shift = sx / sy if sy > 1e-12 else np.nan          # scale shift
    u_shift = (mx - my) / np.sqrt(sx * sy) if (sx > 1e-12 and sy > 1e-12) else np.nan  # location shift
    Cb  = ccc/rho if abs(rho) > 1e-12 else np.nan

    # 95% CI via Fisher z on CCC, FULL Lin (1989) variance (see docstring)
    ccc_clipped = max(min(ccc, 0.9999), -0.9999)
    z_ccc = np.arctanh(ccc_clipped)
    if (abs(rho) > 1e-9 and not np.isnan(u_shift)
            and abs(1 - ccc_clipped**2) > 1e-9 and n > 2):
        c, r_, u_ = ccc_clipped, rho, u_shift
        term1 = (1 - r_**2) * c**2 / ((1 - c**2) * r_**2)
        term2 = 2 * c**3 * (1 - c) * u_**2 / (r_ * (1 - c**2)**2)
        term3 = -(c**4 * u_**4) / (2 * r_**2 * (1 - c**2)**2)
        var_z = (term1 + term2 + term3) / (n - 2)
        se_z = float(np.sqrt(var_z)) if var_z > 0 else np.nan
    else:
        se_z = np.nan
    if not np.isnan(se_z):
        # BUG FIX (2026-10-21): z_crit was hardcoded to 1.96 (the 95%
        # two-tailed value) even though this function already accepted
        # an `alpha` parameter -- the parameter was silently unused.
        z_crit = float(scipy_stats.norm.ppf(1 - alpha / 2))
        ci_lo = float(np.tanh(z_ccc - z_crit*se_z))
        ci_hi = float(np.tanh(z_ccc + z_crit*se_z))
    else:
        ci_lo = ci_hi = np.nan

    # ICC benchmark labels for CCC (Koo & Li 2016 adapted)
    def _ccc_label(v):
        if np.isnan(v): return "N/A"
        if v < 0.50: return "Poor (<0.50)"
        if v < 0.75: return "Moderate (0.50–0.75)"
        if v < 0.90: return "Good (0.75–0.90)"
        return "Excellent (>0.90)"

    return {"ccc":    ccc,
            "rho":    rho,
            "r_squared": float(rho**2) if not np.isnan(rho) else np.nan,
            "Cb":     Cb if not np.isnan(Cb) else np.nan,
            "u_shift": u_shift,     # location shift
            "v_shift": v_shift,     # scale shift
            "mean1": mx, "mean2": my, "sd1": sx, "sd2": sy,
            "mean_diff": float(mx - my),
            "se":     se_z,
            "ci_lo":  ci_lo,
            "ci_hi":  ci_hi,
            "n": n,
            "label":  _ccc_label(ccc)}


def _dual_ci_note(coeff_name, se_basis, ci_lo_z, ci_hi_z, ci_lo_t, ci_hi_t, df,
                  alpha=0.05):
    """Generates the shared narrative note explaining the z-basis and
    t-basis confidence intervals for Scott's pi, Brennan-Prediger,
    Gwet's AC1, and Gwet's AC2 -- with citation strength presented
    honestly and asymmetrically rather than implying both bases rest on
    equally strong primary-source footing.

    se_basis: "analytic" (Brennan-Prediger: Cohen's 1960 SE with fixed
    Pe = 1/q), "gwet2021" (Scott's pi: general Fleiss-kappa variance of
    Gwet, 2021) or "jackknife" (Gwet's AC1/AC2).
    """
    if se_basis == "jackknife":
        # AC1/AC2: delete-one-unit jackknife SE (Efron & Tibshirani, 1993).
        z_evidence = (
            'the SE is a delete-one-unit jackknife estimate; its large-sample '
            'normal approximation follows general jackknife theory (Efron, B., '
            '&amp; Tibshirani, R.J. (1993). <i>An Introduction to the '
            'Bootstrap</i>. Chapman &amp; Hall, Ch. 11).')
        t_evidence = (
            "Intervals with Student's t on n − 1 degrees of freedom are the "
            "convention of Gwet's (2014) inference framework as implemented in "
            "Stata's <i>kappaetc</i> command (Klein, D. (2018). <i>The Stata "
            "Journal</i>, 18(4), 871–901).")
    elif se_basis == "gwet2021":
        # Scott's pi = Fleiss' kappa with two raters.
        z_evidence = (
            'the SE is the general-purpose large-sample variance of Fleiss’ '
            'kappa (Gwet, K.L. (2021). <i>Educational and Psychological '
            'Measurement</i>, 81(4), 781–790), which applies here because '
            'with two raters Fleiss’ kappa equals Scott’s π. The '
            'z-test and p-value use the null-hypothesis SE of Fleiss, Nee '
            '&amp; Landis (1979) instead, which is valid only under '
            'H<sub>0</sub>: π = 0.')
        t_evidence = (
            "Gwet (2021) evaluated this variance with Student's t on n − 1 "
            "degrees of freedom and reported coverage close to the nominal "
            "level.")
    else:
        # Brennan-Prediger: Cohen's (1960) SE with Pe = 1/q fixed.
        z_evidence = (
            'the SE is Cohen’s (1960) large-sample approximation '
            '(<i>Educational and Psychological Measurement</i>, 20(1), '
            '37–46), which treats P<sub>e</sub> as fixed; this holds '
            'exactly here because P<sub>e</sub> = 1/q is set by the declared '
            'number of categories.')
        t_evidence = (
            "Intervals with Student's t on n − 1 degrees of freedom are the "
            "convention of Gwet's (2014) inference framework as implemented in "
            "Stata's <i>kappaetc</i> command (Klein, D. (2018). <i>The Stata "
            "Journal</i>, 18(4), 871–901).")
    return (
        f'<div class="dual-ci-note">'
        f'<b>Two confidence intervals are reported for {coeff_name}, on two different, '
        f'both-legitimate theoretical bases:</b><br/>'
        f'&bull; <b>z-basis ({round((1 - alpha) * 100):g}% CI): [{_f_local(ci_lo_z)}, {_f_local(ci_hi_z)}]</b> \u2014 '
        f'uses the standard normal critical value; '
        f'{z_evidence}<br/>'
        f'&bull; <b>t-basis (df={df}): [{_f_local(ci_lo_t)}, {_f_local(ci_hi_t)}]</b> \u2014 '
        f'a more conservative alternative (always at least as wide as the z-basis '
        f'interval). {t_evidence}<br/>'
        f'<i>The two intervals converge as n grows; report the basis that '
        f'matches the convention of your field and cite its source.</i>'
        f'</div>')


def _f_local(v):
    """Minimal 4dp formatter for use within engine-generated narrative
    text (the engine has no dependency on the UI's display formatters)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "\u2014"
    return f"{v:.4f}"


def _gwet_ac_interp(value, label, method_name, n_units, k_raters,
                     data_type, n_boot_or_jack="jackknife"):
    """Descriptive academic interpretation for Gwet's AC1/AC2."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Result could not be computed for this configuration."
    v = float(value)
    if v < 0:
        strength = "poor (below chance level)"
        impl = ("The raters disagreed more than expected by chance. "
                "Rater training and clearer operational definitions are strongly recommended.")
    elif v < 0.20:
        strength = "slight"
        impl = ("Agreement barely exceeds chance level. "
                "Rating criteria should be revised and raters re-trained.")
    elif v < 0.40:
        strength = "fair"
        impl = ("Fair agreement — substantially influenced by chance. "
                "Rater calibration is recommended before using these ratings for research.")
    elif v < 0.60:
        strength = "moderate"
        impl = ("Moderate agreement, acceptable for exploratory research. "
                "Improvement is recommended for high-stakes or publishable assessment.")
    elif v < 0.80:
        strength = "substantial"
        impl = ("Substantial agreement — generally acceptable for most research contexts.")
    else:
        strength = "almost perfect"
        impl = ("Almost perfect agreement — excellent inter-rater reliability.")

    dtype_str = {"dichotomous": "dichotomous (2 categories)",
                 "nominal": "nominal (unordered categories)",
                 "ordinal": "ordinal (ordered categories)"}.get(data_type, data_type)

    return (
        f"Gwet's {method_name} = {v:.3f} indicates <b>{strength} agreement</b> "
        f"among {k_raters} raters rating {n_units} units "
        f"({dtype_str} data; {label}). "
        f"{impl} "
        f"Variance estimated via {n_boot_or_jack}. "
        f"Note: Gwet's coefficients are more stable than Cohen's κ under "
        f"extreme marginal distributions (prevalence paradox). "
        f"Benchmarks: Landis & Koch (1977), applied here only as a "
        f"conventional descriptive scale, not because it was derived for "
        f"this coefficient -- Vach, W., &amp; Gerke, O. (2023). Gwet's AC1 "
        f"is not a substitute for Cohen's kappa: a comparison of basic "
        f"properties. <i>MethodsX</i>, 10, 102212 argue AC1 and κ respond "
        f"differently to prevalence (for a fixed observed agreement rate, "
        f"AC1 rises as prevalence moves away from 0.5 while κ falls) and "
        f"that AC1, unlike κ, can be nonzero even when raters show no "
        f"association at all -- so a verbal scale calibrated for κ does "
        f"not necessarily carry the same meaning for AC1/AC2."
    )


def _scott_pi_interp(pi, label, n, data_type):
    """Descriptive interpretation for Scott's π.

    Uses if-elif chain — NOT a dict with boolean keys.
    The dict-boolean-key anti-pattern evaluates all conditions at dict creation
    time, collapsing multiple True keys into one and producing incorrect
    strength labels for all values below the last True condition.
    """
    if pi is None or (isinstance(pi, float) and np.isnan(pi)):
        return "Result could not be computed."
    v = float(pi)
    if v < 0:
        strength = "poor"
        impl = "worse than chance; systematic bias or confusion between categories."
    elif v < 0.20:
        strength = "slight"
        impl = "barely above chance; rating criteria need revision."
    elif v < 0.40:
        strength = "fair"
        impl = "fair but chance-influenced; rater calibration recommended."
    elif v < 0.60:
        strength = "moderate"
        impl = "moderate; acceptable for exploratory research."
    elif v < 0.80:
        strength = "substantial"
        impl = "substantial; generally acceptable for research."
    else:
        strength = "almost perfect"
        impl = "excellent reliability."
    dtype_str = {"dichotomous": "dichotomous", "nominal": "nominal"}.get(data_type, data_type)
    return (
        f"Scott's π = {v:.3f} indicates <b>{strength} agreement</b> "
        f"between 2 raters on {n} units ({dtype_str} data; {label}). "
        f"The result is {impl} "
        f"Scott's π uses averaged marginal proportions to estimate expected "
        f"agreement, unlike Cohen's κ which uses the product of separate "
        f"marginals — yielding slightly different estimates when rater "
        f"marginals differ. Reference: Scott (1955)."
    )


def _bp_interp(kbp, label, n, q, data_type):
    """Descriptive interpretation for Brennan-Prediger coefficient."""
    if kbp is None or (isinstance(kbp, float) and np.isnan(kbp)):
        return "Result could not be computed."
    v = float(kbp)
    if v < 0:     strength = "poor (below chance)"
    elif v < 0.20: strength = "slight"
    elif v < 0.40: strength = "fair"
    elif v < 0.60: strength = "moderate"
    elif v < 0.80: strength = "substantial"
    else:          strength = "almost perfect"
    dtype_str = {"dichotomous": "dichotomous", "nominal": "nominal",
                 "ordinal": "ordinal"}.get(data_type, data_type)
    return (
        f"Brennan-Prediger coefficient = {v:.3f} indicates <b>{strength} agreement</b> "
        f"between 2 raters on {n} units ({dtype_str}, {q} categories; {label}). "
        f"Chance agreement is fixed at P_e = 1/{q} = {1/q:.3f}, the value expected "
        f"if raters used the {q} declared categories uniformly at random, so it "
        f"does not depend on the observed category prevalence; the coefficient "
        f"therefore depends on the declared number of categories q. With q = 2 it "
        f"equals the prevalence-adjusted bias-adjusted kappa (Byrt, Bishop & "
        f"Carlin, 1993). Reference: Brennan & Prediger (1981)."
    )


def _ccc_interp(ccc, rho, Cb, label, n):
    """Descriptive interpretation for Lin's CCC."""
    if ccc is None or (isinstance(ccc, float) and np.isnan(ccc)):
        return "Result could not be computed."
    v = float(ccc)
    if v < 0.50:   strength = "poor"
    elif v < 0.75: strength = "moderate"
    elif v < 0.90: strength = "good"
    else:          strength = "excellent"
    rho_str = f"{rho:.3f}" if not np.isnan(rho) else "N/A"
    Cb_str  = f"{Cb:.3f}"  if not np.isnan(Cb)  else "N/A"
    return (
        f"Lin's Concordance Correlation Coefficient (CCC = {v:.3f}) indicates "
        f"<b>{strength} concordance</b> between 2 raters on {n} units ({label}). "
        f"CCC simultaneously evaluates precision (Pearson r = {rho_str}) and "
        f"accuracy (bias correction factor C_b = {Cb_str}). "
        f"A high Pearson r with low C_b indicates raters agree on rank order "
        f"but differ systematically in absolute scoring (scale or location bias). "
        f"|CCC| ≤ |Pearson r| always holds, because CCC = r × C_b with 0 < C_b ≤ 1; "
        f"the gap between them reflects systematic bias. "
        f"Benchmarks (Koo & Li, 2016 adapted): CCC > 0.90 excellent; "
        f"0.75–0.90 good; 0.50–0.75 moderate; < 0.50 poor. "
        f"Reference: Lin (1989, 1992) Biometrics; Lin, Hedayat, Sinha & Yang (2002) JASA; "
        f"McBride (2005)."
    )


def _pa_interp(pa, ci_lo, ci_hi, n, k, alpha=0.05):
    """Descriptive interpretation for Percentage Agreement."""
    if pa is None or (isinstance(pa, float) and np.isnan(pa)):
        return "Result could not be computed."
    ci_str = (f"[{ci_lo:.3f}, {ci_hi:.3f}]"
              if not (np.isnan(ci_lo) or np.isnan(ci_hi)) else "—")
    # No verbal grade is attached: percentage agreement has no published,
    # chance-adjusted benchmark scale, so an "excellent/good/..." label
    # would be unsupported.
    return (
        f"Percentage Agreement = {pa*100:.1f}% ({round((1 - alpha) * 100):g}% CI {ci_str}) "
        f"across {k} raters on {n} units. "
        f"Percentage Agreement does not correct for chance, so it should "
        f"always be reported alongside a chance-corrected coefficient "
        f"(Cohen's κ, Gwet's AC1, or Krippendorff's α). "
        f"It is reported here as a descriptive baseline. "
        f"Reference: Stemler (2004). Practical Assessment, Research & Evaluation."
    )


def _irar_method_context(data_type, k_raters, applicable_methods):
    """Generate academic-English contextual description of why specific methods
    are computed for a given data type and rater count.
    Displayed prominently above the method tabs in the web output.

    CORRECTED (per direct user testing): three defects fixed in this
    revision, all in this single function.
    (1) Content previously used markdown **bold** syntax, but is rendered
        via st.markdown(..., unsafe_allow_html=True) wrapped in a raw
        <div> block -- markdown syntax is not processed inside a raw HTML
        block by standard Markdown parsers, so **text** displayed as
        literal asterisks. Fixed: all emphasis now uses literal <b> tags,
        consistent with every other interpretive-text function in this
        module.
    (2) The "guidance" text was previously a single flat paragraph.
        Fixed: now an HTML list, one item per coefficient.
    (3) MOST SUBSTANTIVE: "guidance" was previously a STATIC paragraph per
        data_type, independent of applicable_methods -- it could both
        OMIT coefficients that ARE computed for a configuration (e.g. the
        dichotomous/nominal branch never mentioned Scott's π, Brennan-
        Prediger, or Krippendorff's α, despite all three always being
        computed for that data type) AND INCLUDE coefficients that are
        NOT applicable for a specific k_raters (e.g. "Weighted Cohen's κ"
        was recommended even for k>=3 ordinal configurations, where
        Weighted κ is never computed, since it is a k=2-only method).
        Fixed: guidance is now built dynamically, iterating
        applicable_methods directly -- the exact same source of truth
        already used for method_list -- so it can never omit an applicable
        coefficient or reference an inapplicable one, for any
        data_type x k_raters combination.
    """
    dtype_full = {"dichotomous": "dichotomous (2 categories, e.g., Correct/Incorrect, Pass/Fail)",
                  "nominal":     "nominal (unordered categories, e.g., error type classification)",
                  "ordinal":     "ordinal (ordered categories, e.g., Likert scale, rubric scores)",
                  "interval":    "interval (continuous scale, e.g., test scores 0–100)",
                  "ratio":       "ratio (continuous with true zero, e.g., time, frequency)"
                  }.get(data_type, data_type)

    method_descs = {
        "cohen":        "<b>Cohen's κ</b> — chance-corrected agreement for 2 raters, nominal/dichotomous (Cohen, 1960)",
        "gwet_ac1":     "<b>Gwet's AC1</b> — chance-corrected agreement with a chance model based on pooled category prevalence (Gwet, 2008)",
        "scott_pi":     "<b>Scott's π</b> — agreement using averaged marginal proportions (Scott, 1955)",
        "bp":           "<b>Brennan-Prediger</b> — chance agreement fixed at 1/q for the q declared categories (Brennan & Prediger, 1981)",
        "pct":          "<b>Percentage Agreement</b> — observed agreement without chance correction (reported as baseline)",
        "fleiss":       "<b>Fleiss' κ</b> — generalises Cohen's κ to ≥3 raters, nominal/dichotomous (Fleiss, 1971)",
        "weighted":     "<b>Weighted Cohen's κ</b> — ordinal agreement for 2 raters with linear and quadratic weights (Cohen, 1968)",
        "gwet_ac2":     "<b>Gwet's AC2</b> — weighted version of AC1 for ordered categories (Gwet, 2014)",
        "icc":          "<b>ICC (6 forms)</b> — one-way, two-way absolute-agreement (ICC-A) and two-way consistency (ICC-C) models, single and average measures (Shrout & Fleiss, 1979; McGraw & Wong, 1996)",
        "ccc":          "<b>Lin's CCC</b> — measures precision and accuracy simultaneously for 2 raters (Lin, 1989)",
        "krippendorff": "<b>Krippendorff's α</b> — any number of raters, nominal to ratio metrics, missing ratings allowed (Krippendorff, 2004)",
    }

    # Short, method-specific guidance fragments. Combined dynamically
    # below, strictly following applicable_methods -- never a static
    # per-data_type paragraph -- so every computed method is addressed
    # and no inapplicable method is ever recommended.
    guidance_fragments = {
        "pct":          "Report as a descriptive baseline alongside any chance-corrected coefficient below; it is not itself chance-corrected.",
        "cohen":        "The established chance-corrected index for 2-rater nominal/dichotomous data (Cohen, 1960); read it together with the prevalence and bias indices shown in its tab (Byrt, Bishop & Carlin, 1993).",
        "gwet_ac1":     "Less sensitive than κ to skewed category prevalence (Gwet, 2008); it responds differently from κ to prevalence, so the two should be read side by side rather than interchangeably (Vach & Gerke, 2023).",
        "scott_pi":     "A useful cross-check against Cohen's κ; differs only in how chance agreement is modelled from the marginals.",
        "bp":           "Its chance term depends only on the declared number of categories, not on the observed marginals, so it is unaffected by category prevalence through P_e.",
        "fleiss":       "The standard index for ≥3-rater nominal/dichotomous agreement; generalises Cohen's κ.",
        "weighted":     "Gives partial credit to near-agreements on an ordered scale; quadratic weights penalise large disagreements most (Cohen, 1968).",
        "gwet_ac2":     "Weighted extension of AC1 for ordered categories, reported under five weighting schemes so the effect of the weighting choice is visible (Gwet, 2014).",
        "icc":          "ICC(A,1) is recommended for single-rater absolute-agreement reliability; use ICC(A,k) if the mean of all raters' scores will be used in practice.",
        "ccc":          "Recommended when both precision (correlation) and accuracy (mean bias) need to be assessed simultaneously.",
        "krippendorff": "A universal complementary estimate that handles missing data and remains comparable across data types.",
    }

    intro = (
        f"Data type: <b>{dtype_full}</b> | Raters: <b>{k_raters}</b> | "
        f"Methods computed: <b>{len(applicable_methods)}</b><br/><br/>"
        f"The following inter-rater agreement methods are automatically computed "
        f"based on the data type and number of raters. Each method addresses "
        f"a different aspect of rater reliability. Results are presented in "
        f"separate tabs below."
    )

    method_list = "<br/>".join(f"&bull; {method_descs[m]}" for m in applicable_methods
                                if m in method_descs)

    # Dynamic guidance: one list item per APPLICABLE method, in the same
    # order as method_list, so nothing computed is ever left undiscussed
    # and nothing uncomputed is ever recommended. No coefficient is tagged
    # as the single "primary" index: all applicable coefficients are
    # reported side by side.
    guidance_items = []
    for m in applicable_methods:
        if m not in method_descs or m not in guidance_fragments:
            continue
        name_only = method_descs[m].split(" — ")[0]  # keeps the <b>...</b> pair intact
        guidance_items.append(f"&bull; {name_only} — {guidance_fragments[m]}")
    guidance = "<br/>".join(guidance_items)

    return {"intro": intro, "methods": method_list, "guidance": guidance}


def _get_applicable_methods(data_type, k_raters):
    """Return ordered list of applicable inter-rater method keys.
    All methods in the list are computed automatically.
    """
    methods = []
    if data_type in ("dichotomous", "nominal"):
        methods.append("pct")          # Percentage Agreement — always first
        if k_raters == 2:
            methods += ["cohen", "gwet_ac1", "scott_pi", "bp"]
        if k_raters >= 3:
            methods += ["fleiss", "gwet_ac1"]
        methods.append("krippendorff")
    elif data_type == "ordinal":
        methods.append("pct")
        if k_raters == 2:
            methods += ["weighted", "gwet_ac2"]
        if k_raters >= 3:
            methods.append("gwet_ac2")
        methods.append("krippendorff")
    else:  # interval / ratio
        methods.append("pct")
        methods.append("icc")
        if k_raters == 2:
            methods.append("ccc")
        methods.append("krippendorff")
    return methods

def _cohen_kappa_weight_matrix(k, weights):
    """Builds the k x k AGREEMENT weight matrix (w_ii=1 for exact agreement,
    decreasing toward 0 for more severe disagreement), matching the exact
    convention used internally by scikit-learn's cohen_kappa_score() for
    weights='linear'/'quadratic' -- verified by direct numerical comparison:
    kappa computed via this module's own Equation-8/9-based formula below
    reproduces sklearn's own point estimate exactly for k=2 and k=4 test
    cases, for all three schemes.

    weights: None/"unweighted" -> identity (w_ii=1, else 0)
             "linear"          -> w_ij = 1 - |i-j|/(k-1)
             "quadratic"       -> w_ij = 1 - (i-j)^2/(k-1)^2
    """
    idx = np.arange(k)
    if weights in (None, "unweighted"):
        return np.eye(k)
    diff = np.subtract.outer(idx, idx).astype(float)
    if weights == "linear":
        return 1 - np.abs(diff) / max(k - 1, 1)
    if weights == "quadratic":
        return 1 - (diff ** 2) / max((k - 1) ** 2, 1)
    raise ValueError(f"Unknown weights scheme: {weights}")


def _kappa_se_z_p_exact(O_counts, W, alpha=0.05):
    """EXACT large-sample standard errors of (weighted) kappa, implementing
    Equation 8 (general-purpose variance, "useful in setting confidence
    limits") and Equation 9 (null-hypothesis variance, "may be used in
    testing the hypothesis that kappa=0") of Fleiss, Cohen & Everitt (1969),
    Psychological Bulletin, 72(5), 323-327 -- parameterized by an explicit
    k x k agreement weight matrix W (identity for unweighted kappa).

    CORRECTION, replacing an earlier implementation: a previous revision of
    this module computed kappa's standard error via a simplified
    approximation, ASE = sqrt[P_o(1-P_o) / (N(1-P_e)^2)], attributing it to
    Fleiss, Cohen & Everitt (1969) without directly verifying it against
    that paper's own stated formula. Direct reconstruction of the primary
    source (obtained as full text) found this simplified formula does NOT
    match either of the two formulas the paper actually presents (Equation
    8 or Equation 9), and can diverge from the paper's own Equation 9 by a
    magnitude that materially changes z and p for realistic data -- found
    via external cross-validation against independent reference statistical software on a real
    N=86 dataset ([[12,12],[7,55]]), where the simplified formula gave
    z=3.4817 against both external tools' z=3.8811, while THIS exact
    Equation-9 implementation reproduces z=3.8811 to four decimal places.
    This function's exact implementation of both equations was itself
    verified, before this correction was adopted, by reproducing the
    primary source's own two worked numerical examples (Table 1, weighted
    kappa, N=200: Var=.002885/.003082; Table 2, unweighted kappa, N=200:
    same values, since unweighted kappa is the identity-weight special
    case) to six decimal places.

    Returns dict: kappa, se_general (Eq 8), se_null (Eq 9), z, p -- z and p
    use se_null (Eq 9), consistent with these being the correct quantities
    for the H0:kappa=0 significance test the primary source itself states
    Equation 9 is "used in testing" -- se_general (Eq 8) is retained in the
    return value for transparency and potential future use, but is NOT
    currently used to construct a confidence interval by this module (see
    kappa_ci_note in _cohen_kappa(), which explains this design choice and
    is unaffected by this formula correction).
    """
    N = float(O_counts.sum())
    if N <= 0:
        return {"kappa": np.nan, "se_general": np.nan, "se_null": np.nan,
                "z": np.nan, "p": np.nan}
    p = O_counts / N
    k = O_counts.shape[0]
    row_m = p.sum(axis=1)
    col_m = p.sum(axis=0)

    p_o = float((W * p).sum())
    p_e = float(sum(row_m[i] * col_m[j] * W[i, j]
                     for i in range(k) for j in range(k)))
    if abs(1 - p_e) < 1e-12:
        return {"kappa": np.nan, "se_general": np.nan, "se_null": np.nan,
                "z": np.nan, "p": np.nan}
    kappa = (p_o - p_e) / (1 - p_e)

    w_bar_row = np.array([sum(W[i, j] * col_m[j] for j in range(k)) for i in range(k)])
    w_bar_col = np.array([sum(W[i, j] * row_m[i] for i in range(k)) for j in range(k)])

    # Equation 8 -- general-purpose variance
    term1_gen = sum(
        p[i, j] * ((1 - p_e) * W[i, j] - (1 - p_o) * (w_bar_row[i] + w_bar_col[j])) ** 2
        for i in range(k) for j in range(k))
    term2_gen = (p_o * p_e - 2 * p_e + p_o) ** 2
    denom_gen = N * (1 - p_e) ** 4
    var_gen = (term1_gen - term2_gen) / denom_gen if denom_gen > 1e-15 else np.nan
    se_general = float(np.sqrt(max(var_gen, 0))) if not np.isnan(var_gen) else np.nan

    # Equation 9 -- null-hypothesis (H0: kappa=0) variance
    term1_null = sum(
        row_m[i] * col_m[j] * (W[i, j] - (w_bar_row[i] + w_bar_col[j])) ** 2
        for i in range(k) for j in range(k))
    denom_null = N * (1 - p_e) ** 2
    var_null = (term1_null - p_e ** 2) / denom_null if denom_null > 1e-15 else np.nan
    se_null = float(np.sqrt(max(var_null, 0))) if not np.isnan(var_null) else np.nan

    if se_null is not None and not np.isnan(se_null) and se_null > 1e-12:
        z = kappa / se_null
        p_val = float(2 * scipy_stats.norm.sf(abs(z)))
    else:
        z, p_val = np.nan, np.nan

    return {"kappa": float(kappa), "se_general": se_general, "se_null": se_null,
            "z": float(z) if not np.isnan(z) else np.nan,
            "p": p_val}


def _cohen_kappa(X, rater_names, alpha=0.05, data_type=None,
                 q_declared=None, scale_min=None):
    """Cohen's Kappa for exactly 2 raters (columns of X).
    Returns a result dict or None on failure.
    Supports: standard κ (all types) + weighted κ linear & quadratic (ordinal).

    data_type: the data type the user declared for the rating group
    ("dichotomous", "nominal" or "ordinal"); auto-detected only when not
    supplied. For declared ordinal data with a declared scale (scale_min,
    q_declared), the category set is the full declared scale
    [scale_min, scale_min + q_declared - 1] (plus any observed value outside
    it), so the linear/quadratic weights w_ij = 1 - |i-j|/(q-1) and
    1 - (i-j)^2/(q-1)^2 use the instrument's q rather than only the
    categories observed in this sample (Cohen, 1968). Categories that no
    rater used contribute empty rows/columns and leave unweighted kappa
    unchanged."""
    if not _SKL or X.shape[1] != 2:
        return None
    try:
        r1 = X[:, 0].astype(int)
        r2 = X[:, 1].astype(int)
    except Exception:
        return None
    n    = len(r1)
    dtype = data_type if data_type else _detect_data_type(list(r1) + list(r2))
    names = (rater_names if rater_names and len(rater_names) >= 2
             else ["Rater 1", "Rater 2"])

    cats_sorted = sorted(set(r1.tolist() + r2.tolist()))
    if dtype == "ordinal" and q_declared and scale_min is not None:
        cats_sorted = sorted(set(range(int(scale_min), int(scale_min) + int(q_declared)))
                             | set(cats_sorted))

    def _safe_kappa(a, b, weights=None):
        try:
            return float(_skl_kappa(a, b, weights=weights, labels=cats_sorted))
        except Exception:
            return np.nan

    nc = len(cats_sorted)
    cat_map = {c: i for i, c in enumerate(cats_sorted)}
    O_counts = np.zeros((nc, nc))
    for a, b in zip(r1, r2):
        O_counts[cat_map[a], cat_map[b]] += 1

    # ── All three weighting schemes, ALWAYS computed together (not gated
    # by dtype=="ordinal" as an earlier revision did -- linear/quadratic
    # weighted kappa is well-defined and computable for dichotomous data
    # too, even though it is then mathematically identical to unweighted
    # kappa, per Fleiss & Cohen, 1973; this module now always computes
    # and reports all three, matching the presentation convention of
    # independent external reference tools). ──
    variants = {}
    for scheme in ("unweighted", "linear", "quadratic"):
        W = _cohen_kappa_weight_matrix(nc, scheme)
        exact = _kappa_se_z_p_exact(O_counts, W)
        k_sklearn = _safe_kappa(r1, r2, weights=None if scheme == "unweighted" else scheme)
        k_val = k_sklearn if not np.isnan(k_sklearn) else exact["kappa"]
        # Confidence interval from the general-purpose variance (Fleiss,
        # Cohen & Everitt, 1969, Eq. 8, "useful in setting confidence
        # limits"); the null variance (Eq. 9) is used only for z and p.
        _dual = _dual_ci(k_val, exact["se_general"], n, alpha)
        variants[scheme] = {
            "kappa": k_val,
            "se_general": exact["se_general"],
            "se_null": exact["se_null"],
            "z": exact["z"],
            "p": exact["p"],
            "ci_lo_z": _dual["ci_lo_z"], "ci_hi_z": _dual["ci_hi_z"],
            "ci_lo_t": _dual["ci_lo_t"], "ci_hi_t": _dual["ci_hi_t"],
            "ci_df": _dual["df"],
            "label": kappa_label(k_val),
        }

    k_std  = variants["unweighted"]["kappa"]
    ase_s  = variants["unweighted"]["se_null"]
    z_s    = variants["unweighted"]["z"]
    p_s    = variants["unweighted"]["p"]

    ct = pd.crosstab(
        pd.Series(r1.tolist(), name=names[0]),
        pd.Series(r2.tolist(), name=names[1]))

    # ── Per-category agreement breakdown: for each category c, N_r1 = how
    # many times rater 1 used c, N_r2 = how many times rater 2 used c,
    # Agree = how many units both raters assigned c, Pct = the Jaccard-style
    # overlap Agree / (N_r1 + N_r2 - Agree) computed below. ──
    per_category = []
    for c in cats_sorted:
        ci = cat_map[c]
        n_r1 = int(O_counts[ci, :].sum())
        n_r2 = int(O_counts[:, ci].sum())
        agree = int(O_counts[ci, ci])
        # Jaccard-style overlap: Agree / (N_r1 + N_r2 - Agree), i.e.
        # intersection over union of the two raters' respective usage of
        # this category -- verified against an external reference tool's
        # own reported per-category percentages on a real dataset
        # (categories 0/1, N_r1/N_r2=24/19 and 62/67, Agree=12/55: this
        # formula reproduces the reference tool's reported 38.7%/74.3%
        # exactly; a simpler Agree/max(N_r1,N_r2) convention does not).
        denom = n_r1 + n_r2 - agree
        pct = (agree / denom * 100) if denom > 0 else np.nan
        per_category.append({"category": c, "n_r1": n_r1, "n_r2": n_r2,
                              "agree": agree, "pct": pct})

    p_o_std = float(np.diag(O_counts / n).sum())
    p_e_std = float(sum((O_counts.sum(axis=1)/n)[i] * (O_counts.sum(axis=0)/n)[j]
                         for i in range(nc) for j in range(nc) if i == j))
    # Recompute p_e_std via the standard product-of-marginals definition
    # (equivalent for the unweighted case; kept as an explicit, independent
    # line for clarity in the breakdown text rather than reusing a
    # weighted-general intermediate).
    row_m_std = O_counts.sum(axis=1) / n
    col_m_std = O_counts.sum(axis=0) / n
    p_e_std = float((row_m_std * col_m_std).sum())

    res = {
        "n": n, "dtype": dtype,
        "rater1": names[0], "rater2": names[1],
        "categories": cats_sorted,
        "n_categories": nc,
        "crosstab":   ct,
        "variants": variants,
        "per_category": per_category,
        "p_observed": p_o_std,
        "p_expected": p_e_std,
        "kappa":      k_std if not np.isnan(k_std) else np.nan,
        "kappa_ase":  ase_s if not np.isnan(ase_s) else np.nan,
        "kappa_z":    z_s if not np.isnan(z_s) else np.nan,
        # NOTE: kappa_p is NOT rounded here (unlike kappa/ase/z above).
        # A genuinely tiny p-value (e.g. z~42 with strong agreement + large
        # N is entirely realistic — verified empirically) would otherwise be
        # rounded to exactly 0.0 at the engine level, which is statistically
        # dishonest (p is never exactly zero) and destroys precision before
        # the display formatter (_p(), which correctly renders "< .00001"
        # for near-zero values) ever sees the true value. Same root-cause
        # class of bug as the ICC p-value issue found and fixed earlier.
        "kappa_p":    p_s if not np.isnan(p_s) else np.nan,
        "kappa_label": kappa_label(k_std),
    }
    res["agreement_breakdown_text"] = (
        f"P(observed) = {p_o_std:.4f}; P(expected by chance) = {p_e_std:.4f}; "
        f"\u03ba = (P_o \u2212 P_e) / (1 \u2212 P_e) = "
        f"({p_o_std:.4f} \u2212 {p_e_std:.4f}) / (1 \u2212 {p_e_std:.4f}) = {k_std:.4f}"
    )
    res["formula_reference_note"] = (
        "Point estimate: Cohen, J. (1960). A coefficient of agreement for "
        "nominal scales. Educational and Psychological Measurement, 20(1), "
        "37\u201346 (unweighted); Cohen, J. (1968). Weighted kappa: Nominal "
        "scale agreement with provision for scaled disagreement or partial "
        "credit. Psychological Bulletin, 70(4), 213\u2013220 (linear/"
        "quadratic weighting; weight convention w_ij=1\u2212|i\u2212j|/(q\u22121) "
        "for linear, w_ij=1\u2212(i\u2212j)\u00b2/(q\u22121)\u00b2 for quadratic, matching "
        "scikit-learn's own internal convention, verified by direct "
        "numerical comparison). Standard error, z, and p-value: Fleiss, "
        "J.L., Cohen, J., & Everitt, B.S. (1969). Large sample standard "
        "errors of kappa and weighted kappa. Psychological Bulletin, "
        "72(5), 323\u2013327, Equation 9 (the null-hypothesis, H0:\u03ba=0 "
        "variance -- this module's exact implementation of Equation 9 was "
        "verified by reproducing the primary source's own worked numerical "
        "example, N=200, to six decimal places, and by external "
        "cross-validation against independent reference statistical software on a real N=86 "
        "dataset). Interpretation benchmarks: Landis, J.R., & Koch, G.G. "
        "(1977). The measurement of observer agreement for categorical "
        "data. Biometrics, 33(1), 159\u2013174."
    )
    # Flag: perfect agreement with zero ASE → z/p undefined (not computable)
    res["kappa_undefined_reason"] = (
        "Sig. (2-tailed) is not computable: agreement is perfect (\u03ba = 1.000) "
        "and ASE = 0, so z = \u03ba / ASE is undefined (division by zero). "
        "This occurs when both raters\u2019 ratings have zero or near-zero "
        "variance (e.g., identical or near-identical scores for every unit). "
        "This is expected behaviour, consistent with standard statistical software, which also reports "
        "this as a system-missing value under these conditions."
        if (not np.isnan(k_std) and abs(k_std - 1.0) < 1e-9 and
            not np.isnan(ase_s) and ase_s < 1e-9)
        else None
    )
    # Two standard errors are reported (Fleiss, Cohen & Everitt, 1969):
    # Equation 9 (null-hypothesis variance) for z and p, and Equation 8
    # (general-purpose variance) for the confidence interval.
    res["kappa_ci_lo"] = variants["unweighted"]["ci_lo_z"]
    res["kappa_ci_hi"] = variants["unweighted"]["ci_hi_z"]
    res["kappa_ci_note"] = (
        "Two standard errors from Fleiss, Cohen & Everitt (1969) are "
        "reported. The null-hypothesis SE (Equation 9, valid when "
        "kappa = 0) is used only for the z statistic and p-value. The "
        "confidence interval uses the general-purpose SE (Equation 8), "
        "which the authors give for setting confidence limits; z-basis "
        "and t-basis (df = n - 1) intervals are both shown.")
    if len(cats_sorted) == 2:
        # Prevalence Index (PI) and Bias Index (BI), Byrt, Bishop, &
        # Carlin (1993, J Clin Epidemiol, 46(5), 423-429): from the 2x2
        # table a (both rate "1"), b (rater1=1,rater2=0), c (rater1=0,
        # rater2=1), d (both rate "0"), n=a+b+c+d:
        #   PI = |a - d| / n
        #   BI = (b - c) / n
        # CORRECTION (2026-10-21): the previous implementation computed
        # |p1-p2|/2 and (p1-p2)/2 from the rater-level marginal
        # proportions p1=mean(r1), p2=mean(r2) -- this uses the WRONG
        # table cells (b,c only, via p1-p2=(b-c)/n, not a,d for PI) and
        # an extraneous /2 divisor not in Byrt et al.'s formula. Verified
        # on the 2 x 2 table of the bundled dichotomous sample (raters R1
        # and R2: a=55, b=7, c=12, d=12, n=86): correct PI=0.5000,
        # BI=-0.0581; the previous code returned PI=0.0291, BI=-0.0291.
        # The higher of the two category codes is treated as the
        # "positive" category (code 1 for the usual 0/1 coding).
        _neg, _pos = cats_sorted
        a = float(np.sum((r1 == _pos) & (r2 == _pos)))
        b = float(np.sum((r1 == _pos) & (r2 == _neg)))
        c = float(np.sum((r1 == _neg) & (r2 == _pos)))
        d = float(np.sum((r1 == _neg) & (r2 == _neg)))
        n_pi = a + b + c + d
        res["prevalence_index"] = abs(a - d) / n_pi if n_pi > 0 else np.nan
        res["bias_index"]       = (b - c) / n_pi if n_pi > 0 else np.nan
    # Linear/quadratic mirrored into legacy top-level keys for backward
    # compatibility with existing UI code -- now populated for EVERY
    # dtype (not only "ordinal" as an earlier revision restricted this
    # to), since weighted kappa is well-defined for any category count.
    for scheme, wk in [("linear", "kappa_linear"), ("quadratic", "kappa_quadratic")]:
        v = variants[scheme]
        res[wk]              = v["kappa"] if not np.isnan(v["kappa"]) else np.nan
        res[f"{wk}_ase"]     = v["se_null"] if not np.isnan(v["se_null"]) else np.nan
        res[f"{wk}_z"]       = v["z"] if not np.isnan(v["z"]) else np.nan
        res[f"{wk}_p"]       = v["p"] if not np.isnan(v["p"]) else np.nan
        res[f"{wk}_label"]   = v["label"]

    # ── Chi-Square Test (standard crosstabulation output) ──────────────────
    try:
        chi2, p_chi2, dof, expected = scipy_stats.chi2_contingency(ct.values)
        n_cells_lt5 = int((expected < 5).sum())
        pct_cells_lt5 = 100 * n_cells_lt5 / expected.size
        res["chi_square"] = {
            "chi2": float(chi2), "df": int(dof),
            "p": float(p_chi2),
            "n_cells_lt5": n_cells_lt5, "pct_cells_lt5": pct_cells_lt5,
            "min_expected": float(expected.min()),
        }
        # Likelihood Ratio (also commonly reported alongside Pearson chi-square)
        with np.errstate(divide="ignore", invalid="ignore"):
            obs = ct.values.astype(float)
            lr = 2 * np.nansum(np.where(obs > 0, obs * np.log(obs / expected), 0))
        res["chi_square"]["lr_chi2"] = float(lr)
        res["chi_square"]["lr_p"]    = float(1 - scipy_stats.chi2.cdf(lr, dof))
    except Exception:
        res["chi_square"] = None

    # ── Symmetric Measures: Correlation (standard crosstabulation output) ──
    # Nominal -> Phi & Cramer's V. Ordinal -> Spearman & Pearson (linear-by-linear).
    sym = {}
    try:
        n_tot = ct.values.sum()
        chi2_v = res["chi_square"]["chi2"] if res.get("chi_square") else np.nan
        if dtype in ("dichotomous", "nominal") and not np.isnan(chi2_v):
            phi = float(np.sqrt(chi2_v / n_tot))
            r_dim, c_dim = ct.shape
            cramers_v = float(np.sqrt(chi2_v / (n_tot * (min(r_dim, c_dim) - 1)))
                              ) if min(r_dim, c_dim) > 1 else np.nan
            sym["phi"]        = phi
            sym["cramers_v"]  = cramers_v if not np.isnan(cramers_v) else np.nan
            sym["chi2_for_phi"] = chi2_v
            sym["phi_p"] = res["chi_square"]["p"] if res.get("chi_square") else np.nan
        if dtype == "ordinal":
            rho, rho_p = scipy_stats.spearmanr(r1, r2)
            pear, pear_p = scipy_stats.pearsonr(r1.astype(float), r2.astype(float))
            sym["spearman_rho"]   = float(rho)
            sym["spearman_rho_p"] = float(rho_p)
            sym["pearson_r"]      = float(pear)
            sym["pearson_r_p"]    = float(pear_p)
    except Exception:
        pass
    res["symmetric_measures"] = sym if sym else None

    return res


def _fleiss_kappa(X, rater_names, alpha=0.05):
    """Fleiss' Kappa for ≥ 3 raters — pure numpy, no statsmodels dependency.
    X shape: (N_units, k_raters) — integer category codes.

    Formula (Fleiss, 1971):
        κ = (P̄ - P̄_e) / (1 - P̄_e)
    where:
        P̄   = mean of per-subject agreement proportions (observed agreement)
        P̄_e = sum of squared marginal proportions (expected by chance)

    IMPORTANT finding regarding SE/z/p below: this function's standard
    error uses the Fleiss, Nee, & Landis (1979) large-sample variance
    formula -- the same formula widely used across major general-purpose
    statistical software packages, including R's "rel" package. Gwet,
    K.L. (2021). Large-sample variance of Fleiss generalized kappa.
    Educational and Psychological Measurement, 81(4), 781-790,
    demonstrates explicitly and formally that this specific variance
    formula "is not meant to be used for quantifying the precision of
    Fleiss' kappa. Instead, it should be used solely for testing the
    hypothesis of zero agreement among raters" (i.e. it is valid for the
    H0: κ=0 significance test reported below, NOT for confidence-interval
    construction around the actual estimated κ). It is therefore used here
    only for z and p. The confidence interval is built from the general-
    purpose variance that Gwet (2021, Equations 4-6) derived for interval
    estimation (see the code below), reported on a t-basis (df = n - 1,
    as evaluated in Gwet's own Monte Carlo study) and a z-basis.
    (Revised 2026-10-23: an earlier revision reported no interval at all.)

    CORRECTION (2026-10-21): an earlier revision of this function used a
    SE formula derived independently "from first principles" rather than
    the actual closed-form Fleiss, Nee & Landis (1979) expression, and
    that independent derivation was wrong -- it returned a value
    approximately 1.94x the correct SE (e.g. 0.08485 vs. the correct
    0.04402 on this module's own N=86, k=4 real sample dataset),
    confirmed by an independent 20,000-replication Monte Carlo simulation
    under H0 using that same dataset's actual marginal prevalence
    (empirical SD = 0.04378, within 0.6% of the formula below -- the
    earlier value was off by a factor of ~2, not a rounding difference).
    The formula actually implemented below is the real Fleiss, Nee, &
    Landis (1979) closed form,

        Var(kappa) = [2 / (N k(k-1) S1^2)] * [S1^2 - S2],
        S1 = sum_j p_j(1-p_j),   S2 = sum_j p_j(1-p_j)(1-2 p_j),

    cross-checked against an independent secondary reproduction of the
    1979 paper's Equation (2) and against the Monte Carlo result above
    before being adopted here.

    Primary sources:
    Fleiss, J.L. (1971). Measuring nominal scale agreement among many
    raters. Psychological Bulletin, 76(5), 378-382.
    Fleiss, J.L., Nee, J.C.M., & Landis, J.R. (1979). Large sample
    variance of kappa in the case of different sets of raters.
    Psychological Bulletin, 86(5), 974-977.
    Gwet, K.L. (2021). Large-sample variance of Fleiss generalized
    kappa. Educational and Psychological Measurement, 81(4), 781-790.
    """
    if X.shape[1] < 3:
        return None
    try:
        n, k  = X.shape   # n=units, k=raters
        dtype = _detect_data_type(X.flatten().tolist())
        cats  = sorted(set(int(v) for v in X.flatten().tolist()
                          if not (isinstance(v, float) and np.isnan(v))))
        nc    = len(cats)
        cat_map = {c: i for i, c in enumerate(cats)}

        # Build N × n_categories rating table
        table = np.zeros((n, nc), dtype=int)
        for i in range(n):
            for j in range(k):
                table[i, cat_map[int(X[i, j])]] += 1

        # Marginal proportions p_j
        p_j = table.sum(axis=0) / (n * k)

        # Per-subject agreement P_i
        # P_i = (Σ_j n_ij(n_ij-1)) / (k(k-1))
        P_i = np.array([
            np.sum(table[i] * (table[i] - 1)) / (k * (k - 1))
            for i in range(n)
        ])
        P_bar = P_i.mean()      # overall observed agreement
        P_e   = np.sum(p_j**2) # expected agreement by chance

        if abs(1 - P_e) < 1e-12:
            return None

        kappa_overall = float((P_bar - P_e) / (1 - P_e))

        # SE and z — null-hypothesis (H0: κ=0) large-sample variance,
        # the actual closed-form formula of Fleiss, Nee, & Landis (1979,
        # Psychological Bulletin, 86(5), 974-977, Equation 2):
        #
        #   S1 = Σ_j p_j(1-p_j)   [ = 1 - P_e ]
        #   S2 = Σ_j p_j(1-p_j)(1-2p_j)
        #   Var(κ) = [2 / (N·k(k-1)·S1²)] · [S1² - S2]
        #
        # Verified (2026-10-21) against an independent 20,000-replication
        # Monte Carlo simulation under H0 (raters drawing independently
        # from the pooled marginal distribution) on this module's own
        # N=86, k=4 real sample dataset: formula SE = 0.04402 vs.
        # empirical Monte Carlo SD = 0.04378 (within 0.6%). An earlier
        # revision of this function used a different, self-derived
        # formula that was wrong by a factor of ~2 (see docstring
        # CORRECTION note above) -- this formula replaces it.
        S1 = float(np.sum(p_j * (1 - p_j)))              # = 1 - P_e
        S2 = float(np.sum(p_j * (1 - p_j) * (1 - 2 * p_j)))
        var_kappa = (2.0 / (n * k * (k - 1) * max(S1**2, 1e-12))) * (S1**2 - S2)
        var_kappa = max(var_kappa, 0.0)    # guard against fp noise
        se  = float(np.sqrt(var_kappa))
        z   = kappa_overall / se if se > 1e-12 else np.nan
        p   = float(2 * scipy_stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan

        # General-purpose variance for interval estimation (Gwet, K.L.
        # (2021). Large-sample variance of Fleiss generalized kappa.
        # Educational and Psychological Measurement, 81(4), 781-790,
        # Equations 4-6, finite-population correction f = 0):
        #   kappa_i  = (pa_i - pe)/(1 - pe),  pa_i = sum_c n_ic(n_ic-1)/(k(k-1))
        #   pe_i     = sum_c p_c n_ic / k
        #   kappa*_i = kappa_i - 2(1 - kappa)(pe_i - pe)/(1 - pe)
        #   var_G    = sum_i (kappa*_i - kappa)^2 / (n(n-1))
        pe_i = (table * p_j[None, :]).sum(axis=1) / k
        kap_i = (P_i - P_e) / (1 - P_e)
        kap_star = kap_i - 2 * (1 - kappa_overall) * (pe_i - P_e) / (1 - P_e)
        se_general = float(np.sqrt(np.sum((kap_star - kappa_overall) ** 2) / (n * (n - 1)))) if n > 1 else np.nan
        _dual = _dual_ci(kappa_overall, se_general, n, alpha)

        # Per-category κ (Fleiss 1971, eq. 14)
        kappa_per_cat = {}
        p_per_cat     = {}
        for ci, cat in enumerate(cats):
            pj   = p_j[ci]
            num  = (np.sum(table[:, ci] * (table[:, ci] - 1)) / max(n*k*(k-1), 1)) - pj**2
            den  = pj * (1 - pj)
            kappa_per_cat[cat] = float(num/den) if abs(den) > 1e-10 else np.nan
            p_per_cat[cat]     = float(pj)

        names = (rater_names if rater_names and len(rater_names) >= k
                 else [f"Rater {i+1}" for i in range(k)])
        return {
            "n": n, "k": k, "dtype": dtype,
            "categories":  cats,
            "rater_names": names,
            "kappa":       kappa_overall,
            "P_bar":       float(P_bar),   # observed agreement (Po)
            "P_e":         float(P_e),     # expected/chance agreement (Pe)
            "kappa_se":    se if not np.isnan(se) else np.nan,
            "kappa_z":     z if not np.isnan(z) else np.nan,
            # Not rounded here -- preserve precision for genuinely tiny p.
            # (p_per_category above is a marginal PROPORTION, not a p-value
            # -- exact 0 there is legitimate and unrelated to this fix.)
            "kappa_p":     p if not np.isnan(p) else np.nan,
            "kappa_label": kappa_label(kappa_overall),
            "kappa_se_general": se_general,
            "ci_lo_z": _dual["ci_lo_z"], "ci_hi_z": _dual["ci_hi_z"],
            "ci_lo_t": _dual["ci_lo_t"], "ci_hi_t": _dual["ci_hi_t"],
            "ci_df": _dual["df"],
            "kappa_ci_lo": _dual["ci_lo_t"],
            "kappa_ci_hi": _dual["ci_hi_t"],
            "kappa_ci_note": (
                "Two standard errors are reported. The null-hypothesis SE "
                "of Fleiss, Nee & Landis (1979) is valid only for testing "
                "H0: kappa = 0 and is used only for z and p (Gwet, 2021). "
                "The confidence interval uses Gwet's (2021) general-purpose "
                "variance (Equations 4-6), with the t-basis interval "
                "(df = n - 1) that Gwet (2021) evaluated and the z-basis "
                "interval both shown."),
            "kappa_per_category": kappa_per_cat,
            "p_per_category":     p_per_cat,
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def alpha_label(a):
    """Descriptive label for a reliability coefficient (George & Mallery, 2003)."""
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return "undefined"
    if a >= 0.9:
        return "excellent"
    if a >= 0.8:
        return "good"
    if a >= 0.7:
        return "acceptable"
    if a >= 0.6:
        return "questionable"
    if a >= 0.5:
        return "poor"
    return "unacceptable"


# ══════════════════════════════════════════════════════════════════════════════
# CORE COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════════════

def _cronbach_alpha(X):
    """Cronbach's alpha from an N x k item matrix (raw)."""
    k = X.shape[1]
    if k < 2:
        return np.nan
    item_var = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return float((k / (k - 1)) * (1 - item_var.sum() / total_var))


def _standardised_alpha(X):
    """Standardised Cronbach's alpha (based on the correlation matrix)."""
    k = X.shape[1]
    if k < 2:
        return np.nan
    R = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices(k, 1)
    rbar = R[iu].mean()
    if np.isnan(rbar):
        return np.nan
    return float(k * rbar / (1 + (k - 1) * rbar))


def _alpha_ci(alpha, n, k, conf=0.95):
    """Feldt (1965) confidence interval for Cronbach's alpha."""
    if alpha is None or np.isnan(alpha) or n < 3:
        return np.nan, np.nan
    a = 1 - conf
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    try:
        f_lo = scipy_stats.f.ppf(1 - a / 2, df1, df2)
        f_hi = scipy_stats.f.ppf(a / 2, df1, df2)
        lo = 1 - (1 - alpha) * f_lo
        hi = 1 - (1 - alpha) * f_hi
        return float(lo), float(hi)
    except Exception:
        return np.nan, np.nan


def _kr20(X):
    """KR-20 for dichotomous (0/1) items.

    KR-20 is algebraically equivalent to Cronbach's α when all items are
    dichotomous (Kuder & Richardson, 1937). This equivalence requires that
    the item variance be computed as the SAMPLE variance p·q·n/(n−1), not
    the population variance p·q. Using population variance breaks the identity
    and produces a systematic positive bias in KR-20 relative to Cronbach's α.

    Formula (Kuder & Richardson, 1937; Lord & Novick, 1968):
        KR-20 = k/(k−1) · (1 − Σσ²ᵢ / σ²_total)
    where σ²ᵢ = pᵢ·qᵢ·n/(n−1)  [sample variance of dichotomous item i]
          σ²_total = variance of total scores  (ddof=1)

    Returns a dict with:
      kr20            -- the coefficient
      item_p          -- proportion-correct per item (array)
      item_pq_raw     -- RAW population-style p·q per item (array), the
                         textbook/most commonly recognised form of the
                         per-item variance term
      item_pq_sample  -- SAMPLE-corrected p·q·n/(n−1) per item (array),
                         the form that actually enters the KR-20 formula
                         above. item_pq_raw and item_pq_sample are BOTH
                         surfaced (rather than only the sample-corrected
                         form previously returned) because different
                         reference tools conventionally display the raw,
                         uncorrected p·q as the per-item diagnostic value
                         even when the underlying reliability formula
                         itself uses the sample-corrected version
                         internally -- showing both avoids a researcher
                         being unable to reconcile a displayed p·q figure
                         against an external reference's own p·q column,
                         despite the final KR-20 value itself matching.
    """
    n, k = X.shape
    if k < 2:
        return {"kr20": np.nan, "item_p": np.array([]),
                "item_pq_raw": np.array([]), "item_pq_sample": np.array([])}
    p  = X.mean(axis=0)               # proportion correct per item
    q  = 1 - p
    pq_raw = p * q                    # RAW population-style p·q
    # Sample variance for dichotomous item: p·q·n/(n−1)
    # This is equivalent to X[:,j].var(ddof=1) for 0/1 items, and guarantees
    # KR-20 = Cronbach's α exactly for dichotomous data.
    pq_sample = pq_raw * n / (n - 1)  # sample variance, NOT population p·q
    total_var  = X.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return {"kr20": np.nan, "item_p": p,
                "item_pq_raw": pq_raw, "item_pq_sample": pq_sample}
    kr = float((k / (k - 1)) * (1 - pq_sample.sum() / total_var))
    return {"kr20": kr, "item_p": p,
            "item_pq_raw": pq_raw, "item_pq_sample": pq_sample}


def _kr21(X):
    """Kuder-Richardson Formula 21 (Kuder & Richardson, 1937).

    A simplified, more conservative form of KR-20 that assumes EQUAL
    item difficulty across all items (every item's true p equals the
    same p̄), avoiding the need to compute each item's individual p
    separately -- historically valuable before computers made per-item
    computation routine. Uses only the mean and variance of total scores.

    Formula (Kuder & Richardson, 1937, exact -- verified against the
    formula as they themselves present it, and independently confirmed
    by exact numerical reconstruction against an external reference
    calculator's output on a real dataset, N=86, k=4: removing an
    earlier, undocumented n/(n-1) sample-correction factor that had
    been added to this implementation reproduced the external
    reference's value to within 9x10^-6):
        KR-21 = [k/(k−1)] × [1 − M(k−M) / (k·σ²_total)]
    where M = mean of total scores (sample), σ²_total = sample variance
    (ddof=1) of total scores, k = number of items. Unlike KR-20
    (Section on _kr20), this formula applies NO further sample-size
    correction to the M(k-M)/k term -- KR-21's entire simplifying
    purpose is to require only the total-score mean and variance, with
    no per-item computation at all, so it has no per-item p*q term for
    a sample-variance correction to apply to in the first place. An
    earlier revision of this function incorrectly multiplied M(k-M) by
    an additional n/(n-1) factor, reasoning by analogy to KR-20's own
    item-level correction -- this was not verified against the primary
    source before being added, and was found, when tested against an
    external reference implementation, to be unsupported: the
    uncorrected formula (as shown above, and as already stated in this
    docstring prior to that revision) is the one that reproduces
    external reference values exactly.

    This is the KR-20 formula with every item's p·q·n/(n−1) term replaced
    by the SAME value, M(k−M)/k -- i.e., substituting the
    average item difficulty p̄=M/k uniformly for every item's individual
    pᵢ. By Jensen's inequality (p(1-p) is concave), the RAW (uncorrected)
    per-item terms satisfy Σpᵢqᵢ ≤ k·p̄q̄ -- so on a MATCHED basis (both
    raw, or both sample-corrected), KR-21 ≤ KR-20 always, with equality
    only when items are genuinely equally difficult.

    IMPORTANT, PRECISE SCOPE (verified by direct construction of a
    counter-example, then confirmed to vanish at large n): because this
    function's KR-21 correctly uses the UNCORRECTED M(k-M)/k term (per
    the primary source and external validation -- see below), while
    _kr20() correctly uses the SAMPLE-CORRECTED Σpᵢqᵢ·n/(n-1) term (per
    the separately, independently proven Bernoulli-variance identity
    that Section on _kr20 relies on), the two formulas are being
    compared on DIFFERENT bases for finite n. The n/(n-1) correction
    factor -> 1 as n -> infinity, so KR-21 <= KR-20 holds as an
    asymptotic property and holds in practice for typical sample sizes,
    but a small, finite-sample violation is mathematically possible
    (confirmed by direct construction: n=86 can produce a violation on
    the order of 0.003-0.004; n=2000 constructions tested did not
    produce any violation in 20 trials). This is not an implementation
    error in either formula -- both are individually correct and
    externally validated on their own terms -- but a genuine
    consequence of comparing a sample-corrected quantity (KR-20) against
    an uncorrected one (KR-21), which the classical Jensen's-inequality
    argument above does not, by itself, account for.

    Reference: Kuder, G.F., & Richardson, M.W. (1937). The theory of the
    estimation of test reliability. Psychometrika, 2(3), 151–160.
    """
    n, k = X.shape
    if k < 2:
        return np.nan
    total = X.sum(axis=1)
    M = float(total.mean())
    var_total = float(total.var(ddof=1))
    if var_total == 0:
        return np.nan
    kr21 = (k / (k - 1)) * (1 - (M * (k - M)) / (k * var_total))
    return float(kr21)


def _sem(sd_total, reliability):
    """Standard Error of Measurement (SEM).

    Estimates the standard deviation of measurement error expected
    around a single respondent's observed total score, under classical
    test theory -- widely used to construct approximate confidence
    bands around an individual score (e.g., observed ± 1.96·SEM).

    Formula: SEM = SD_total × √(1 − reliability)

    where SD_total is the observed (sample) standard deviation of total
    scores, and 'reliability' is a reliability coefficient for the same
    scale (conventionally KR-20 or Cronbach's α, whichever is the
    primary headline estimate for the analysis at hand).

    Reference: Lord, F.M., & Novick, M.R. (1968). Statistical Theories
    of Mental Test Scores. Addison-Wesley, Chapter 3. [Foundational
    classical-test-theory source for this formula; also presented in,
    e.g., Crocker, L., & Algina, J. (1986). Introduction to Classical
    and Modern Test Theory. Holt, Rinehart and Winston.]
    """
    if reliability is None or (isinstance(reliability, float) and np.isnan(reliability)):
        return np.nan
    if reliability > 1 or (1 - reliability) < 0:
        return np.nan  # SEM undefined when reliability exceeds 1 (see KR-20 out-of-range note)
    return float(sd_total * np.sqrt(1 - reliability))


def _is_dichotomous(X):
    """True if every item is coded strictly 0/1 (genuine dichotomous/binary data).

    BUG FIX (2026-10-21, fatal, confirmed via audit): the original check
    only required "at most two distinct values per column" -- NOT that
    those two values were actually 0 and 1. This is reachable and
    seriously wrong in practice: this module's reverse-scoring feature
    (auto_rev, 'theoretical' method) is available for ANY construct
    regardless of data type, and the UI's own scale-bounds inputs default
    to scale_min=1.0, scale_max=5.0 (views/internalconsistency.py)
    regardless of whether the underlying items are actually 0/1. Reversing
    a genuinely dichotomous 0/1 item with those defaults maps it to
    {(1+5)-0, (1+5)-1} = {6, 5} -- still only two distinct values, so the
    OLD check still called the column "dichotomous" and routed it into
    _kr20(), which assumes every column's MEAN is itself a valid Bernoulli
    proportion p (with q = 1-p). For a column coded {5,6}, p ~= 5.5 and
    q = 1-p ~= -4.5, so that item's p*q "variance" term is strongly
    NEGATIVE (not a valid variance at all). Confirmed by direct
    construction: reversing a single flagged dichotomous item with the
    UI's default theoretical bounds drove KR-20 from a normal in-range
    value to 122.42 -- not a borderline "just over 1.0" case, a two-
    orders-of-magnitude blow-up.

    This is the real, reachable mechanism behind a KR-20 "out of range"
    (> 1) value in this application -- NOT "near-zero item variance", as
    this module's UI-facing explanation (views/internalconsistency.py)
    previously claimed. That claim was independently re-verified and found
    UNSUPPORTED, not merely unproven: a 200,000-trial Monte Carlo search
    over complete (listwise, no missing data), correctly-coded 0/1 binary
    data -- including data deliberately constructed with near-zero-
    variance items and strong latent positive inter-item correlation,
    specifically to push KR-20 as high as mathematically possible -- never
    produced a value exceeding 1.0 by more than floating-point noise (max
    observed over 200,000 trials: 1.0000000000000007). This is not a
    coincidence: for any COMPLETE sample item matrix, the sample
    covariance matrix is guaranteed positive semi-definite, which in turn
    guarantees Cov(Xi,Xj) <= (Var(Xi)+Var(Xj))/2 for every item pair --
    and that inequality is exactly what keeps KR-20 (and Cronbach's alpha)
    from exceeding 1 for genuinely 0/1-coded, complete data. The
    "out-of-range (>1)" UI branch is retained purely as a defensive
    floating-point-precision safety net (see views/internalconsistency.py),
    not because near-zero-variance items genuinely produce it.

    Fixed by requiring the (at most two) distinct values actually BE 0 and
    1 (within floating-point tolerance), not merely "two distinct values
    of any kind". Genuinely binary-but-not-0/1-coded data (e.g. 1/2
    coding, or a 0/1 item that has been reverse-scored with non-0/1
    bounds) now correctly falls through to the "not dichotomous" branch,
    where Cronbach's alpha -- which, unlike KR-20, is invariant to any
    such linear recoding -- remains the valid, reported reliability
    estimate.
    """
    for j in range(X.shape[1]):
        u = np.unique(X[:, j])
        if len(u) > 2:
            return False
        if not np.all(np.isin(np.round(u, 6), [0.0, 1.0])):
            return False
    return True


def _split_half_for_indices(X, idx1, idx2):
    """Core split-half computation for an ARBITRARY partition of item
    indices into two groups. Used by all three split-half variants
    (first-second, odd-even, maximum) so the underlying formula is
    defined and verified in exactly one place.

    Reference: Spearman (1910), Brown (1910) — equal-length correction;
    Guttman (1945) — unequal-variance split-half formula.
    """
    X1 = X[:, idx1]; X2 = X[:, idx2]
    h1 = X1.sum(axis=1); h2 = X2.sum(axis=1)
    try:
        r = np.corrcoef(h1, h2)[0, 1]
    except Exception:
        r = np.nan
    sb = (2 * r / (1 + r)) if not np.isnan(r) else np.nan
    var1 = h1.var(ddof=1); var2 = h2.var(ddof=1)
    vart = X.sum(axis=1).var(ddof=1)
    guttman = (4 * r * np.sqrt(var1 * var2) / vart) if (vart > 0 and not np.isnan(r)) else np.nan
    a1 = _cronbach_alpha(X1) if X1.shape[1] >= 2 else np.nan
    a2 = _cronbach_alpha(X2) if X2.shape[1] >= 2 else np.nan
    return {"error": False,
            "spearman_brown": float(sb) if not np.isnan(sb) else np.nan,
            "guttman": float(guttman) if not np.isnan(guttman) else np.nan,
            "r_halves": float(r) if not np.isnan(r) else np.nan,
            "alpha_part1": float(a1) if not np.isnan(a1) else np.nan,
            "alpha_part2": float(a2) if not np.isnan(a2) else np.nan,
            "n_part1": len(idx1), "n_part2": len(idx2)}


def _bootstrap_ci_percentile(X, compute_fn, n_boot=1000, alpha=0.05, seed=42,
                              min_valid=20):
    """General-purpose percentile bootstrap confidence interval for any
    reliability coefficient computed directly from an N x k item matrix.

    Resamples RESPONDENTS (rows) with replacement, recomputes
    compute_fn(X_boot) on each resample, and returns the percentile-based
    interval [100*alpha/2, 100*(1-alpha/2)] of the resulting bootstrap
    distribution.

    This is the method explicitly recommended, following an extensive
    review of alternatives, by Kelley, K., & Pornprasertmanit, S. (2016).
    Confidence intervals for population reliability coefficients:
    Evaluation of methods, recommendations, and software for composite
    measures. Psychological Methods, 21(1), 69-92 -- who recommend
    bootstrapping for both alpha- and omega-type reliability
    coefficients over competing analytic approaches. A Monte Carlo
    comparison of six omega-CI methods (see e.g. the applied literature
    citing this recommendation) found bootstrap methods have the best
    coverage performance specifically for the smaller sample sizes
    common in applied reliability research (N<100).

    Appropriate ONLY for coefficients whose per-replicate computational
    cost is small and does not itself involve a combinatorial search or
    iterative optimization that scales unfavourably with k (item count)
    -- i.e. NOT for Guttman's lambda4, the maximum-Spearman-Brown
    split-half variant, or McDonald's omega-hierarchical, all three of
    which require a dedicated, cost-aware bootstrap strategy
    (`_bootstrap_ci_fixed_search`) instead.

    Uses a fixed random seed for reproducibility, matching the existing
    convention already established for Krippendorff's alpha's own
    bootstrap CI elsewhere in this module.

    Returns dict: ci_lo, ci_hi, n_boot_valid, method="bootstrap".
    Returns NaN bounds if fewer than `min_valid` replicates succeed
    (e.g. due to degenerate resamples).
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_boot = X[idx, :]
        try:
            val = compute_fn(X_boot)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                boot_vals.append(float(val))
        except Exception:
            continue
    if len(boot_vals) < min_valid:
        return {"ci_lo": np.nan, "ci_hi": np.nan,
                "n_boot_valid": len(boot_vals), "n_boot_requested": n_boot,
                "method": "bootstrap_insufficient_replicates"}
    boot_arr = np.array(boot_vals)
    ci_lo = float(np.percentile(boot_arr, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
    return {"ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_boot_valid": len(boot_vals), "n_boot_requested": n_boot,
            "method": "bootstrap"}


# Threshold (number of items, k) below which a full, per-replicate
# re-optimization bootstrap is computationally practical (benchmarked
# empirically: k=12 completes 1000 replicates in well under 15 seconds
# for lambda4, the maximum-Spearman-Brown split, and omega-hierarchical
# alike; k=14-18 scale to several minutes or worse for these three
# coefficients specifically, since each involves either a combinatorial
# search over 2^(k-1)-1 partitions or an iterative oblique-rotation
# optimization whose cost grows with the number of extracted factors).
_BOOTSTRAP_SEARCH_K_THRESHOLD = 12


def _bootstrap_ci_fixed_search(X, k, search_and_compute_fn,
                                recompute_given_result_fn,
                                n_boot=1000, alpha=0.05, seed=42,
                                min_valid=20,
                                k_threshold=_BOOTSTRAP_SEARCH_K_THRESHOLD):
    """Bootstrap confidence interval for coefficients whose point
    estimate requires an expensive per-call search or iterative
    optimization (Guttman's lambda4, the maximum-Spearman-Brown
    split-half variant, and McDonald's omega-hierarchical) -- for which
    a naive percentile bootstrap that re-runs the full search on every
    replicate does not scale to realistic item counts (benchmarked: k=18
    naive bootstrap projects to 35-135+ minutes for these coefficients,
    versus ~1 second for a simple closed-form statistic at the same k).

    Two regimes, selected automatically by k:

      k <= k_threshold (default 12): FULL bootstrap. The expensive
        search/optimization (`search_and_compute_fn`) is re-run on every
        bootstrap replicate, exactly as it is for the point estimate --
        the most rigorous option, used whenever it remains
        computationally practical (benchmarked well under 15 seconds for
        1000 replicates at k=12 for all three affected coefficients).

      k > k_threshold: FIXED-SEARCH bootstrap. The expensive
        search/optimization is run ONCE, on the original (full) sample,
        to find the best partition (lambda4, split-half maximum) or
        rotated factor solution (omega-hierarchical). This result is
        then held FIXED, and each bootstrap replicate only recomputes
        the coefficient's VALUE given that fixed partition/solution on
        the resampled data (`recompute_given_result_fn`) -- a cheap
        operation with no search involved, restoring bootstrap speed to
        that of a simple closed-form statistic.

        IMPORTANT, DISCLOSED LIMITATION: this fixed-search CI answers
        "how precise is this coefficient's value, GIVEN the partition/
        solution already found on this sample" -- not "how precise
        would the coefficient be if the optimal partition/solution were
        also re-selected in every hypothetical repeated sample." This is
        a standard, named simplification in the bootstrap literature for
        statistics involving a search or selection step (sometimes
        discussed under "post-selection inference"); it is not
        presented as equivalent in rigor to the full bootstrap used for
        k <= k_threshold, and the application discloses which of the two
        methods was used in every case, rather than presenting both
        identically.

    Returns dict: ci_lo, ci_hi, n_boot_valid, method
    ("bootstrap_full_search" or "bootstrap_fixed_search"), k_threshold.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if k <= k_threshold:
        boot_vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            X_boot = X[idx, :]
            try:
                val = search_and_compute_fn(X_boot)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    boot_vals.append(float(val))
            except Exception:
                continue
        method = "bootstrap_full_search"
    else:
        try:
            fixed_result = search_and_compute_fn(X, return_search_result=True)
        except TypeError:
            fixed_result = search_and_compute_fn(X)
        boot_vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            X_boot = X[idx, :]
            try:
                val = recompute_given_result_fn(X_boot, fixed_result)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    boot_vals.append(float(val))
            except Exception:
                continue
        method = "bootstrap_fixed_search"
    if len(boot_vals) < min_valid:
        return {"ci_lo": np.nan, "ci_hi": np.nan,
                "n_boot_valid": len(boot_vals), "n_boot_requested": n_boot,
                "method": method + "_insufficient_replicates",
                "k_threshold": k_threshold}
    boot_arr = np.array(boot_vals)
    ci_lo = float(np.percentile(boot_arr, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
    return {"ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_boot_valid": len(boot_vals), "n_boot_requested": n_boot,
            "method": method, "k_threshold": k_threshold}


def _lambda4_given_split(X, Vt, best_split):
    """Recomputes Guttman's lambda4's value for a PRE-DETERMINED split
    (item indices in group A), without searching -- the cheap companion
    to `_lambda4_exhaustive`, used by `_bootstrap_ci_fixed_search` for
    k > 12 (see that function's docstring for the full rationale)."""
    k = X.shape[1]
    idx1 = list(best_split)
    idx2 = [i for i in range(k) if i not in idx1]
    h1 = X[:, idx1].sum(axis=1)
    h2 = X[:, idx2].sum(axis=1)
    return 2 * (1 - (h1.var(ddof=1) + h2.var(ddof=1)) / Vt)


def _spearman_brown_given_split(X, best_split):
    """Recomputes the Spearman-Brown split-half value for a PRE-
    DETERMINED split, without searching -- the cheap companion to
    `_max_spearman_brown_split_exhaustive`."""
    k = X.shape[1]
    idx1 = list(best_split)
    idx2 = [i for i in range(k) if i not in idx1]
    h1 = X[:, idx1].sum(axis=1)
    h2 = X[:, idx2].sum(axis=1)
    r_corr = np.corrcoef(h1, h2)[0, 1]
    if np.isnan(r_corr):
        return np.nan
    return 2 * r_corr / (1 + r_corr)


def _split_half(X):
    """Split-half reliability, THREE variants, all built on the same
    verified core (_split_half_for_indices):

      1. First-second half split (PRIMARY / always shown) — items 1..k/2
         vs k/2+1..k. Numerically validated against an independent
         external reference calculator to <0.0001 on a real reference
         dataset; this remains the headline result, unchanged.
      2. Odd-even split (supplementary) — odd-indexed vs even-indexed
         items, the classical alternative split convention.
      3. Maximum split-half (supplementary) — the split that maximises
         the Spearman-Brown coefficient specifically, via its own
         independent exhaustive search (_max_spearman_brown_split_
         exhaustive). This matches what external tools such as R's
         psych::splitHalf() report as "maximum split-half reliability".

    IMPORTANT (verified during development): the split that maximises
    Guttman's lambda4 (variance-ratio formula) is NOT always the same
    split that maximises the Spearman-Brown coefficient — confirmed on
    a real reference dataset where the two criteria selected genuinely
    different splits. This function therefore performs its OWN
    Spearman-Brown-maximising search for the "maximum" variant, rather
    than reusing lambda4's optimal split — reusing it would silently
    under-report the true achievable maximum Spearman-Brown value and
    mislabel it. lambda4's own (separately reported, in the Guttman's
    Lambda section) result remains the correct maximiser of ITS OWN
    variance-ratio criterion.

    MANDATORY GATE: split-half analysis requires a minimum of 4 items
    (see docstring history: Spearman-Brown's equal-length assumption is
    not meaningful below 4 items). Applies to all three variants.
    """
    k = X.shape[1]
    if k < 4:
        err = {"error": True, "message": "Split-half analysis requires at least 4 items.",
               "n_items": k, "spearman_brown": np.nan, "guttman": np.nan}
        return {"error": True, "message": err["message"], "n_items": k,
                "spearman_brown": np.nan, "guttman": np.nan,
                "first_second": err, "odd_even": err, "maximum": err}

    half = k // 2
    first_second = _split_half_for_indices(X, list(range(half)), list(range(half, k)))

    idx = np.arange(k)
    odd_even = _split_half_for_indices(X, list(idx[idx % 2 == 0]), list(idx[idx % 2 == 1]))

    sb_max_result = _max_spearman_brown_split_exhaustive(X)
    if sb_max_result.get("best_split") is not None:
        best_idx1 = list(sb_max_result["best_split"])
        best_idx2 = [i for i in range(k) if i not in best_idx1]
        maximum = _split_half_for_indices(X, best_idx1, best_idx2)
        maximum["method"] = sb_max_result["method"]
    else:
        maximum = dict(first_second)
        maximum["method"] = "unavailable"

    # Top-level keys mirror first_second for backward compatibility with
    # existing callers that expect the flat (pre-refactor) structure.
    out = dict(first_second)
    out["error"] = False
    out["first_second"] = first_second
    out["odd_even"] = odd_even
    out["maximum"] = maximum
    return out


def _sample_random_partitions(k, n_samples, seed=42):
    """Generates n_samples random, non-trivial partitions of k items into
    two non-empty groups, as a boolean indicator matrix (True = group A).

    Each item is independently assigned to group A or B via an unbiased
    coin flip; any degenerate draw (all items in one group) is redrawn.
    This is the random-sampling methodology used, for exactly this
    problem, by Revelle, W. psych::splitHalf() -- see the calling
    functions' docstrings for the full citation and rationale.

    Returns: boolean ndarray of shape (n_samples, k).
    """
    rng = np.random.default_rng(seed)
    indicator = rng.random((n_samples, k)) < 0.5
    degenerate = indicator.all(axis=1) | (~indicator).all(axis=1)
    redraws = 0
    while degenerate.any() and redraws < 20:
        n_deg = int(degenerate.sum())
        indicator[degenerate] = rng.random((n_deg, k)) < 0.5
        degenerate = indicator.all(axis=1) | (~indicator).all(axis=1)
        redraws += 1
    return indicator


def _best_partition_vectorized(X, indicator, criterion, Vt=None):
    """Vectorized evaluation of every partition (row) in `indicator`
    (an (n_partitions, k) boolean matrix -- either the full exhaustive
    set or a random sample from `_sample_random_partitions`) at once,
    via matrix operations, avoiding a per-partition Python loop.

    criterion: "variance_ratio" (Guttman's lambda4) or "correlation"
    (Spearman-Brown-maximizing split-half).

    Returns: (best_value, best_partition_item_indices_tuple, n_evaluated)
    """
    n_partitions = indicator.shape[0]
    indicator_f = indicator.astype(np.float64)
    H1 = X @ indicator_f.T                      # (n, n_partitions)
    total = X.sum(axis=1)
    H2 = total[:, None] - H1
    if criterion == "variance_ratio":
        var_h1 = H1.var(axis=0, ddof=1)
        var_h2 = H2.var(axis=0, ddof=1)
        vals = 2 * (1 - (var_h1 + var_h2) / Vt)
    else:  # "correlation" -- Spearman-Brown from the h1/h2 correlation
        H1c = H1 - H1.mean(axis=0, keepdims=True)
        H2c = H2 - H2.mean(axis=0, keepdims=True)
        num = (H1c * H2c).sum(axis=0)
        den = np.sqrt((H1c ** 2).sum(axis=0) * (H2c ** 2).sum(axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den > 1e-12, num / den, np.nan)
            vals = np.where(np.isnan(r), -np.inf, 2 * r / (1 + r))
    vals = np.nan_to_num(vals, nan=-np.inf)
    best_idx = int(np.argmax(vals))
    best_partition = tuple(np.where(indicator[best_idx])[0].tolist())
    return float(vals[best_idx]), best_partition, n_partitions


def _lambda4_exhaustive(X, Vt, max_exhaustive_k=16):
    """Guttman's lambda4: the TRUE maximum split-half coefficient over
    all possible non-trivial splits of k items into two groups.

    Reference: Guttman, L. (1945). A basis for analyzing test-retest
    reliability. Psychometrika, 10(4), 255-282.

    For k items there are 2^(k-1) - 1 distinct non-trivial splits (A vs
    B is the same partition as B vs A, so the naive 2^k - 2 count is
    halved). This is exhaustively searched whenever k <= max_exhaustive_k.

    THRESHOLD AND FALLBACK METHOD, both aligned with established
    precedent: R's widely-used psych package switches its own
    splitHalf() function from exhaustive search to random sampling at
    the SAME general item-count threshold used here ("only sample
    splits if the number of possible splits exceeds 10,000 ... more
    than 16 items in a scale" -- psych package NEWS documentation), and
    an independently developed package for exactly this problem
    (splithalf, Parsons, Kruijt, & Fox et al.) uses the same random-
    sampling philosophy. This function follows that same precedent
    rather than a cruder single-split (e.g. even/odd) approximation:
    for k > max_exhaustive_k, `n_boot`-many random, non-trivial
    partitions are sampled (default 15,000, matching the sample count
    used in more recent psych::splitHalf versions) and the best value
    found among the SAMPLE is reported -- explicitly labelled "sampled"
    in the returned method field, never presented as the guaranteed
    true maximum. Sampling is vectorized (a single matrix operation
    across all sampled partitions at once, per `_best_partition_vectorized`)
    rather than a per-partition Python loop, keeping this fast regardless
    of k (unlike exhaustive search, whose cost grows as 2^(k-1)).

    Reference for the sampling methodology: Revelle, W. (2024). psych:
    Procedures for Psychological, Psychometric, and Personality Research
    [R package], splitHalf() function and package NEWS.

    Returns dict: {"lambda4": float, "method": "exhaustive"|"sampled",
    "n_splits_evaluated": int, "best_split": (tuple of item indices in
    group A) or None}.
    """
    k = X.shape[1]
    if k < 2 or Vt <= 0:
        return {"lambda4": np.nan, "method": "n/a", "n_splits_evaluated": 0, "best_split": None}

    if k <= max_exhaustive_k:
        best_l4 = -np.inf
        best_split = None
        n_eval = 0
        # only need combinations of size 1..k//2 to avoid double-counting
        # (size k-r is the complement of size r)
        for r in range(1, k // 2 + 1):
            for combo in itertools.combinations(range(k), r):
                # skip the mirror-image duplicate when k is even and r==k/2
                if k % 2 == 0 and r == k // 2 and 0 not in combo:
                    continue
                idx1 = list(combo)
                idx2 = [i for i in range(k) if i not in combo]
                h1 = X[:, idx1].sum(axis=1)
                h2 = X[:, idx2].sum(axis=1)
                l4_split = 2 * (1 - (h1.var(ddof=1) + h2.var(ddof=1)) / Vt)
                n_eval += 1
                if l4_split > best_l4:
                    best_l4 = l4_split
                    best_split = tuple(idx1)
        return {"lambda4": float(best_l4), "method": "exhaustive",
                "n_splits_evaluated": n_eval, "best_split": best_split}
    else:
        n_sample = 15000
        indicator = _sample_random_partitions(k, n_sample, seed=42)
        best_l4, best_split, n_eval = _best_partition_vectorized(
            X, indicator, criterion="variance_ratio", Vt=Vt)
        return {"lambda4": best_l4, "method": "sampled",
                "n_splits_evaluated": n_eval, "best_split": best_split}


def _max_spearman_brown_split_exhaustive(X, max_exhaustive_k=16):
    """Finds the split that maximises the SPEARMAN-BROWN (correlation-
    based) split-half coefficient specifically — NOT the same target as
    Guttman's lambda4 (variance-ratio) maximisation.

    IMPORTANT, verified during development: the split that maximises
    Guttman's lambda4 formula (2*(1-(Va+Vb)/Vt)) is NOT guaranteed to be
    the same split that maximises the Spearman-Brown formula (2r/(1+r))
    when item variances are heterogeneous — confirmed on a real
    reference dataset where the two criteria selected genuinely
    different "best" splits (lambda4-optimal: items {1,2} vs {3,4};
    SB-optimal: item {3} alone vs {1,2,4}). Reusing lambda4's optimal
    split to report a "maximum" Spearman-Brown value would silently
    under-report the true achievable maximum and mislabel it. This
    function performs its own independent search so that "maximum
    Spearman-Brown split-half" is reported honestly against its own
    actual optimisation target.

    THRESHOLD AND FALLBACK METHOD, both aligned with established
    precedent: R's widely-used psych package switches its own
    splitHalf() function from exhaustive search to random sampling at
    the SAME general item-count threshold used here ("only sample
    splits if the number of possible splits exceeds 10,000 ... more
    than 16 items in a scale" -- psych package NEWS documentation), and
    an independently developed package for exactly this problem
    (splithalf, Parsons, Kruijt, & Fox et al.) uses the same random-
    sampling philosophy. For k > max_exhaustive_k, this function
    likewise switches to random sampling (15,000 non-trivial partitions
    by default, vectorized via `_best_partition_vectorized` rather than
    a per-partition Python loop) instead of a cruder single-split
    (even/odd) approximation, explicitly labelled "sampled" in the
    returned method field.

    Reference: Revelle, W. (2024). psych: Procedures for Psychological,
    Psychometric, and Personality Research (splitHalf function docs and
    package NEWS); Spearman (1910); Brown (1910).
    """
    k = X.shape[1]
    if k < 4:
        return {"spearman_brown": np.nan, "method": "n/a",
                "n_splits_evaluated": 0, "best_split": None}
    if k <= max_exhaustive_k:
        best_sb = -np.inf
        best_split = None
        n_eval = 0
        for r in range(1, k // 2 + 1):
            for combo in itertools.combinations(range(k), r):
                if k % 2 == 0 and r == k // 2 and 0 not in combo:
                    continue
                idx1 = list(combo)
                idx2 = [i for i in range(k) if i not in combo]
                h1 = X[:, idx1].sum(axis=1)
                h2 = X[:, idx2].sum(axis=1)
                try:
                    r_corr = np.corrcoef(h1, h2)[0, 1]
                    sb = 2 * r_corr / (1 + r_corr) if not np.isnan(r_corr) else -np.inf
                except Exception:
                    sb = -np.inf
                n_eval += 1
                if sb > best_sb:
                    best_sb = sb
                    best_split = tuple(idx1)
        return {"spearman_brown": float(best_sb), "method": "exhaustive",
                "n_splits_evaluated": n_eval, "best_split": best_split}
    else:
        n_sample = 15000
        indicator = _sample_random_partitions(k, n_sample, seed=42)
        best_sb, best_split, n_eval = _best_partition_vectorized(
            X, indicator, criterion="correlation")
        return {"spearman_brown": best_sb, "method": "sampled",
                "n_splits_evaluated": n_eval, "best_split": best_split}


def _guttman_lambdas_fast_only(X):
    """Computes ONLY lambda1, lambda2, lambda3, lambda5, lambda6 -- the
    five Guttman-lambda forms that are genuinely closed-form and fast
    at any item count (benchmarked: 1000 evaluations at k=18 complete
    in well under 0.2 seconds total).

    Deliberately EXCLUDES lambda4 (the true-maximum split-half
    coefficient), which requires an exhaustive combinatorial search and
    is the ENTIRE reason `_guttman_lambdas_single_basis` becomes slow at
    large k (benchmarked: lambda4 alone accounts for ~99% of that
    function's total runtime at k=18). This function exists specifically
    so the five fast lambdas can be bootstrapped via
    `_bootstrap_ci_percentile` without incurring lambda4's search cost on
    every one of potentially 1000 replicates -- lambda4's own confidence
    interval is instead computed via `_bootstrap_ci_fixed_search`
    (Guttman's lambda4 point-estimate computation, `_guttman_lambdas_single_basis`,
    is otherwise unchanged and still computes all six forms together for
    the point estimate itself).

    Reference: Guttman, L. (1945). Psychometrika, 10(4), 255-282.
    """
    k = X.shape[1]
    if k < 2:
        return {"lambda1": np.nan, "lambda2": np.nan, "lambda3": np.nan,
                "lambda5": np.nan, "lambda6": np.nan}
    try:
        C = np.cov(X, rowvar=False, ddof=1)
        Vt = X.sum(axis=1).var(ddof=1)
        if Vt <= 0:
            return {"lambda1": np.nan, "lambda2": np.nan, "lambda3": np.nan,
                    "lambda5": np.nan, "lambda6": np.nan}
        item_var = np.diag(C)
        offdiag = C - np.diag(item_var)

        l1 = 1 - item_var.sum() / Vt
        l3 = (k / (k - 1)) * l1
        sum_sq_off = np.sum(offdiag ** 2)
        l2 = l1 + np.sqrt((k / (k - 1)) * sum_sq_off) / Vt
        try:
            Rinv = np.linalg.inv(np.corrcoef(X, rowvar=False))
            smc = 1 - 1 / np.diag(Rinv)
            e = item_var * (1 - smc)
            l6 = 1 - e.sum() / Vt
        except Exception:
            l6 = np.nan
        col_sq = np.sum(offdiag ** 2, axis=0)
        l5 = l1 + 2 * np.sqrt(np.max(col_sq)) / Vt

        return {"lambda1": float(l1), "lambda2": float(l2), "lambda3": float(l3),
                "lambda5": float(l5), "lambda6": float(l6) if not np.isnan(l6) else np.nan}
    except Exception:
        return {"lambda1": np.nan, "lambda2": np.nan, "lambda3": np.nan,
                "lambda5": np.nan, "lambda6": np.nan}


def _guttman_lambdas_single_basis(X):
    """Guttman's lambda1..lambda6, computed from whichever matrix X is
    passed in. Calling this with raw item scores gives the RAW
    (covariance-based) lambda family; calling it with column-standardised
    (z-scored, unit-variance) item scores gives the STANDARDISED
    (correlation-based) lambda family. This is the same formula family
    in both cases — only the input basis differs.

    Reference: Guttman, L. (1945). Psychometrika, 10(4), 255-282,
    Equations as individually cited per coefficient below.
    """
    k = X.shape[1]
    out = {f"lambda{i}": np.nan for i in range(1, 7)}
    out["lambda4_method"] = "n/a"
    out["lambda4_n_splits"] = 0
    if k < 2:
        return out
    try:
        C = np.cov(X, rowvar=False, ddof=1)
        Vt = X.sum(axis=1).var(ddof=1)
        if Vt <= 0:
            return out
        item_var = np.diag(C)
        sum_item_var = item_var.sum()
        offdiag = C - np.diag(item_var)

        # lambda1 (Guttman, 1945, Eq. 1) — weakest lower bound
        l1 = 1 - sum_item_var / Vt
        out["lambda1"] = float(l1)

        # lambda3 (Guttman, 1945, Eq. 3) — algebraically identical to
        # Cronbach's alpha on this same basis
        l3 = (k / (k - 1)) * l1
        out["lambda3"] = float(l3)

        # lambda2 (Guttman, 1945, Eq. 2) — improves on lambda1 using the
        # off-diagonal covariance structure
        sum_sq_off = np.sum(offdiag ** 2)
        l2 = l1 + np.sqrt((k / (k - 1)) * sum_sq_off) / Vt
        out["lambda2"] = float(l2)

        # lambda4 (Guttman, 1945, Eq. 4) — TRUE maximum split-half,
        # via exhaustive search (see _lambda4_exhaustive)
        l4_result = _lambda4_exhaustive(X, Vt)
        out["lambda4"] = l4_result["lambda4"]
        out["lambda4_method"] = l4_result["method"]
        out["lambda4_n_splits"] = l4_result["n_splits_evaluated"]

        # lambda6 (Guttman, 1945, Eq. 6) — uses squared multiple
        # correlation (SMC) to isolate unique/error variance per item
        try:
            Rinv = np.linalg.inv(np.corrcoef(X, rowvar=False))
            smc = 1 - 1 / np.diag(Rinv)
            e = item_var * (1 - smc)
            l6 = 1 - e.sum() / Vt
            out["lambda6"] = float(l6)
        except Exception:
            out["lambda6"] = np.nan

        # lambda5 (Guttman, 1945, Eq. 5)
        # λ5 = λ1 + 2·√(e**) / Vt, e** = max_j(Σ_{i≠j} s²_ij)
        col_sq = np.sum(offdiag ** 2, axis=0)
        l5 = l1 + 2 * np.sqrt(np.max(col_sq)) / Vt
        out["lambda5"] = float(l5)

        # ── Dimensionality signal: Kaiser eigenvalue criterion ───────────
        # An earlier version of this function attempted a "lambda4 minus
        # mean of other lambdas" gap as a multidimensionality heuristic.
        # This was tested via Monte Carlo simulation (100 replications
        # each) and found NOT to discriminate reliably: under a
        # genuinely unidimensional null, the gap has mean ~0.08 (90th
        # percentile ~0.10) purely from lambda4's inherent selection
        # bias (it is defined as the MAXIMUM over many candidate splits,
        # which is systematically inflated relative to non-maximised
        # bounds even with no true sub-structure present) -- and this
        # null distribution substantially overlaps the distribution
        # observed under genuine 2-cluster bifactor data (mean ~0.11).
        # A simple gap threshold could not be found that discriminates
        # the two scenarios reliably, so it was NOT shipped as a flag.
        #
        # Instead, this reports the number of eigenvalues > 1 (Kaiser,
        # 1960) of the inter-item correlation matrix -- the same,
        # separately-validated criterion already used to gate whether
        # omega_hierarchical is computable. This is informational, not
        # a pass/fail flag: more than one eigenvalue > 1 suggests
        # possible multidimensionality worth following up (e.g. via the
        # omega_hierarchical result, if available, or a dedicated EFA),
        # but is not by itself definitive.
        try:
            R_for_eig = np.corrcoef(X, rowvar=False)
            eigvals = np.linalg.eigvalsh(R_for_eig)[::-1]
            out["n_eigenvalues_above_1"] = int(np.sum(eigvals > 1.0))
        except Exception:
            out["n_eigenvalues_above_1"] = None
    except Exception:
        pass
    return out


def _guttman_lambdas(X):
    """Guttman's lambda1..lambda6, reported on BOTH the raw-covariance
    basis and the standardised-correlation basis.

    IMPORTANT: these are NOT two different theories or two different
    families of coefficient. This is the same Guttman (1945) lambda
    family, computed on two different, both-legitimate measurement
    bases. The two sets will generally differ whenever item variances
    are heterogeneous — the more item SDs differ from one another, the
    larger the raw-vs-standardised gap typically is. Neither basis is
    "more correct" in the abstract: use the raw basis when the original
    item-scale differences are substantively meaningful; use the
    standardised basis when items should be equalised (unit variance)
    before combination (e.g. items measured on different scales).

    Returns a dict with keys "raw" and "standardized", each holding a
    full lambda1..lambda6 (+ lambda4 method metadata) sub-dict from
    _guttman_lambdas_single_basis().
    """
    raw = _guttman_lambdas_single_basis(X)
    k = X.shape[1]
    if k >= 2:
        sd = X.std(axis=0, ddof=1)
        sd_safe = np.where(sd > 1e-12, sd, 1.0)
        X_std = (X - X.mean(axis=0)) / sd_safe
    else:
        X_std = X
    standardized = _guttman_lambdas_single_basis(X_std)
    return {"raw": raw, "standardized": standardized}


def _item_total_stats(X, item_names, alpha_full):
    """Corrected item-total correlation, alpha-if-deleted, scale stats."""
    k = X.shape[1]
    total = X.sum(axis=1)
    rows = []
    for j in range(k):
        rest = total - X[:, j]
        try:
            cit = np.corrcoef(X[:, j], rest)[0, 1]
        except Exception:
            cit = np.nan
        # alpha if item j deleted
        Xd = np.delete(X, j, axis=1)
        a_del = _cronbach_alpha(Xd) if Xd.shape[1] >= 2 else np.nan
        # squared multiple correlation (item j predicted by the rest)
        try:
            Rinv = np.linalg.inv(np.corrcoef(X, rowvar=False))
            smc = 1 - 1 / np.diag(Rinv)
            smc_j = float(smc[j])
        except Exception:
            smc_j = np.nan
        rows.append({
            "Item": item_names[j],
            "Scale Mean if Item Deleted": float((total - X[:, j]).mean()),
            "Scale Variance if Item Deleted": float(rest.var(ddof=1)),
            "Corrected Item-Total Correlation": cit if not np.isnan(cit) else np.nan,
            "Squared Multiple Correlation": smc_j if not np.isnan(smc_j) else np.nan,
            "Cronbach's Alpha if Item Deleted": a_del if not np.isnan(a_del) else np.nan,
        })
    df = pd.DataFrame(rows)
    # Flag negative alpha-if-deleted values (footnote "a")
    df.attrs["has_negative_alpha"] = bool(
        (df["Cronbach's Alpha if Item Deleted"] < 0).any())
    return df


def _paf_single_factor(R, max_iter=300, tol=1e-9):
    """Iterated Principal Axis Factoring, single common factor.

    Reference: classical iterative principal-factor method (see e.g.
    Gorsuch, 1983, Factor Analysis, 2nd ed., Ch.5). Communalities are
    initialised via SMC (squared multiple correlation = 1 - 1/diag(R^-1))
    and refined iteratively until convergence:
      1. Replace R's diagonal with current communality estimates h².
      2. Eigendecompose the reduced matrix; take the largest
         eigenvalue/eigenvector as the factor loading: λ = sqrt(e1)·v1.
      3. Update h² = λ², repeat until h² stabilises.

    This is a genuine common-factor extraction (unlike PCA, which uses
    the unreduced correlation matrix with 1s on the diagonal) — it
    partitions each item's variance into common (factor-explained) and
    unique components, which is what McDonald's omega formula requires.

    Handles the k=2 case (SMC undefined via matrix inverse for a 2x2
    correlation matrix) with the well-defined closed form h² = r_12².

    Returns (loadings, communalities) as 1-D arrays of length k.
    """
    k = R.shape[0]
    if k == 2:
        h2 = np.array([R[0, 1] ** 2, R[0, 1] ** 2])
    else:
        Rinv = np.linalg.inv(R)
        h2 = np.clip(1 - 1 / np.diag(Rinv), 0.0, 0.999)
    loading = np.zeros(k)
    for _ in range(max_iter):
        R_reduced = R.copy()
        np.fill_diagonal(R_reduced, h2)
        vals, vecs = np.linalg.eigh(R_reduced)
        idx = np.argmax(vals)
        loading = vecs[:, idx] * np.sqrt(max(vals[idx], 0.0))
        if loading.sum() < 0:
            loading = -loading
        h2_new = np.clip(loading ** 2, 0.0, 0.999)
        if np.max(np.abs(h2_new - h2)) < tol:
            h2 = h2_new
            break
        h2 = h2_new
    return loading, h2


def _paf_multi_factor(R, m, max_iter=500, tol=1e-9):
    """Iterated Principal Axis Factoring, m>1 common factors (unrotated).
    Same iterative-communality principle as _paf_single_factor, extended
    to retain the m largest eigenvalue/eigenvector pairs per iteration.
    Returns (pattern matrix k×m, communalities length-k)."""
    k = R.shape[0]
    Rinv = np.linalg.inv(R)
    h2 = np.clip(1 - 1 / np.diag(Rinv), 0.0, 0.999)
    L = np.zeros((k, m))
    for _ in range(max_iter):
        R_reduced = R.copy()
        np.fill_diagonal(R_reduced, h2)
        vals, vecs = np.linalg.eigh(R_reduced)
        idx = np.argsort(vals)[::-1][:m]
        L = vecs[:, idx] * np.sqrt(np.clip(vals[idx], 0.0, None))
        h2_new = np.clip(np.sum(L ** 2, axis=1), 0.0, 0.999)
        if np.max(np.abs(h2_new - h2)) < tol:
            h2 = h2_new
            break
        h2 = h2_new
    return L, h2


def _gpa_oblimin(A, max_iter=5000, tol=1e-9):
    """Oblique rotation via the Gradient Projection Algorithm, direct
    oblimin criterion (gamma=0, i.e. direct quartimin — the standard
    default oblique rotation used by R's psych::fa() and most major
    EFA software).

    Reference: Bernaards, C.A. & Jennrich, R.I. (2005). Gradient
    Projection Algorithms and Software for Arbitrary Rotation Criteria
    in Factor Analysis. Educational and Psychological Measurement,
    65(5), 676-696.

    Input: A (k×m) unrotated loading matrix from _paf_multi_factor.
    Returns: (pattern matrix k×m, factor correlation matrix Φ m×m).

    Verified (see project audit records): gradient converges to ~0,
    result is invariant to the rotation starting point, and the rotated
    solution reproduces the correlation matrix with error identical to
    the unrotated solution (pure reparameterisation, no information
    loss) — confirmed on synthetic bifactor data with known structure.
    """
    k, m = A.shape
    T = np.eye(m)
    alpha = 1.0

    def _crit(L):
        L2 = L ** 2
        N = np.ones((m, m)) - np.eye(m)
        return np.sum(L2 * (L2 @ N)) / 4, L * (L2 @ N)

    Tinv = np.linalg.inv(T)
    L = A @ Tinv.T
    Q, Gq = _crit(L)
    for _ in range(max_iter):
        G = -(L.T @ Gq @ Tinv).T
        Gp = G - T @ np.diag(np.sum(T * G, axis=0))
        s = np.linalg.norm(Gp)
        if s < tol:
            break
        alpha = 2 * alpha
        for _ in range(30):
            Tt = T - alpha * Gp
            v = 1 / np.sqrt(np.sum(Tt ** 2, axis=0))
            Tt = Tt @ np.diag(v)
            Tinv_t = np.linalg.inv(Tt)
            Lt = A @ Tinv_t.T
            Qt, Gqt = _crit(Lt)
            if Qt < Q - 0.5 * s ** 2 * alpha:
                break
            alpha = alpha / 2
        T, L, Q, Gq = Tt, Lt, Qt, Gqt
        Tinv = Tinv_t
    return L, T.T @ T


def _schmid_leiman(Pattern, Phi):
    """Schmid-Leiman bifactor transformation of an oblique multi-factor
    EFA solution into an orthogonalised general factor + orthogonalised
    group/specific factors.

    Reference: Schmid, J. & Leiman, J.M. (1957). The development of
    hierarchical factor solutions. Psychometrika, 22(1), 53-61.
    Applied to omega-hierarchical specifically: Zinbarg, R.E., Revelle,
    W., Yovel, I., & Li, W. (2005). Cronbach's alpha, Revelle's beta,
    and McDonald's omega_H: their relations with each other and two
    alternative conceptualizations of reliability. Psychometrika, 70(1),
    123-133; McDonald, R.P. (1999), Test Theory: A Unified Treatment.

    Procedure:
      1. A second single-factor PAF extraction is applied to Φ itself
         (the group-factor correlation matrix) to obtain each group
         factor's own loading g_j on a general (second-order) factor.
         This must use the SAME iterated-communality PAF procedure as
         the first-order extraction, not a raw eigendecomposition of Φ
         (whose diagonal is 1, not a reduced/communality diagonal) —
         using the raw diagonal was verified during development to
         produce an invalid decomposition (Φ not correctly reproduced).
      2. General-factor item loadings: G = Pattern @ g
      3. Specific/group-factor item loadings (orthogonalised against the
         general factor): S[:,j] = Pattern[:,j] * sqrt(1 - g_j²)

    Verified identity (see project audit records): G_i² + Σⱼ S_ij²
    exactly reproduces each item's original communality from the
    first-order solution, and G@G' + S@S' + diag(uniqueness) reproduces
    the correlation matrix with error identical to the original
    (Pattern, Φ) solution — confirming this is an exact, information-
    preserving reparameterisation.

    Returns (G: general loadings length-k, S: specific loadings k×m,
    g: group-factor-on-general loadings length-m).
    """
    m = Pattern.shape[1]
    if m == 2:
        g, _ = _paf_single_factor(Phi)
    else:
        g, _ = _paf_single_factor(Phi)
    G = Pattern @ g
    if G.sum() < 0:
        G = -G
        g = -g
    S = Pattern * np.sqrt(np.clip(1 - g ** 2, 0.0, None))[None, :]
    return G, S, g


def _mcdonald_omega(X, item_names=None):
    """McDonald's Omega, unidimensional model (\u03c9u), via iterated
    Principal Axis Factoring — computed on BOTH denominator bases
    described in the primary literature.

    Reference (primary, for this exact distinction): Flora, D.B. (2020).
    Your coefficient alpha is probably wrong, but which coefficient
    omega is right? A tutorial on using R to obtain better reliability
    estimates. Advances in Methods and Practices in Psychological
    Science, 3(4), 484-501. https://doi.org/10.1177/2515245920951747
    Foundational reference: McDonald, R.P. (1999). Test Theory: A
    Unified Treatment. Erlbaum.

    Flora (2020) shows that \u03c9u = (\u03a3\u03bbj)\u00b2 / \u03c3\u0302X\u00b2, where the
    denominator \u03c3\u0302X\u00b2 "can be calculated either as the sample
    variance of the total score X or as the model-implied variance of
    X" (Flora, 2020, p. 488) — i.e. \u03c9u is ONE coefficient computable
    on two different, both-legitimate variance bases, not two
    differently-named coefficients. This function reports both:

      omega_model_implied: \u03c3\u0302X\u00b2 = (\u03a3\u03bbj)\u00b2 + \u03a3(1-\u03bbj\u00b2), the
        variance implied BY the fitted one-factor model itself (sum of
        modelled common + modelled unique variance). Matches the rows
        labelled "omega"/"omega2" in R's semTools::reliability().

      omega_observed: \u03c3\u0302X\u00b2 = the ACTUAL observed total-score
        variance (k + 2\u00b7\u03a3 off-diagonal correlations for standardised
        items), independent of how well the one-factor model fits.
        Matches the row labelled "omega3" in semTools::reliability().

    Flora's footnote 7 explicitly warns that some software documentation
    (and Kelley & Pornprasertmanit, 2016) mislabel the observed-variance
    form "hierarchical omega" — this is avoided here: "omega
    hierarchical" (\u03c9h) is reserved exclusively for the bifactor-model
    estimate (see _mcdonald_omega_hierarchical), consistent with most of
    the psychometric literature (Zinbarg et al., 2005; Rodriguez, Reise,
    & Haviland, 2016).

    When the one-factor model fits the data well, the two bases give
    near-identical values (Flora, 2020, footnote 4); a large gap between
    them is itself diagnostic of one-factor model misfit (Bentler, 2009;
    Kelley & Pornprasertmanit, 2016).

    METHODOLOGICAL NOTE (historical correction): an earlier version of
    this function derived loadings from the first principal COMPONENT of
    the raw correlation matrix (PCA), which does not partition variance
    into common vs unique components and cannot validly be labelled a
    McDonald's omega estimate. This function uses iterated Principal
    Axis Factoring, a genuine common-factor extraction. Verified on a
    real reference dataset (N=86, k=4): loadings matched an independent
    external reference calculator to 3-4 decimal places.

    Returns a dict with omega_model_implied, omega_observed, plus full
    item-level decomposition (loading, communality, unique variance per
    item). Returns NaN/empty arrays if k < 3 or computation fails.
    """
    k = X.shape[1]
    names = item_names if item_names and len(item_names) >= k else [f"Item {i+1}" for i in range(k)]
    empty = {"omega_model_implied": np.nan, "omega_observed": np.nan,
             "loadings": np.array([]), "communalities": np.array([]),
             "unique_variances": np.array([]), "item_names": names}
    if k < 3:
        return empty
    try:
        R = np.corrcoef(X, rowvar=False)
        loading, comm = _paf_single_factor(R)
        err = 1 - comm
        Vt_observed = k + 2 * np.sum(np.triu(R, 1))
        Vt_model_implied = float(loading.sum() ** 2 + err.sum())
        omega_observed = float(loading.sum() ** 2 / Vt_observed) if Vt_observed > 1e-9 else np.nan
        omega_model_implied = (float(loading.sum() ** 2 / Vt_model_implied)
                                if Vt_model_implied > 1e-9 else np.nan)
        return {"omega_observed": omega_observed,
                "omega_model_implied": omega_model_implied,
                "loadings": loading,
                "communalities": comm,
                "unique_variances": err,
                "item_names": names}
    except Exception:
        return empty


def _mcdonald_omega_hierarchical(X, item_names=None, min_items=6, max_iter_note=True,
                                  max_rotation_iter=None):
    """McDonald's Omega Hierarchical (ωh) via Schmid-Leiman bifactor
    transformation.

    Reference: McDonald, R.P. (1999). Test Theory: A Unified Treatment.
    Erlbaum; Schmid, J. & Leiman, J.M. (1957). Psychometrika, 22(1),
    53-61; Zinbarg, Revelle, Yovel & Li (2005). Psychometrika, 70(1),
    123-133.

    max_rotation_iter: if None (default), the oblique rotation
    (`_gpa_oblimin`) uses its own default max_iter (5000), unchanged
    from prior behaviour -- used for the point estimate. If an integer
    is supplied, this caps the rotation's iteration budget, bounding
    worst-case per-call time; used specifically by the bootstrap
    confidence-interval mechanism (`_bootstrap_ci_fixed_search`), since
    bootstrap-resampled covariance structures were found to occasionally
    require far more rotation iterations to converge than the original
    (non-resampled) sample, causing highly variable, sometimes very slow
    per-replicate cost that is NOT simply a function of item count k.

    ωh answers a different question than ωt: it isolates the proportion
    of total variance attributable specifically to a single general
    factor common to all items, as distinct from ωt's "how much total
    variance is systematic at all" (general + group-specific combined).
    ωh is always <= ωt (within the same bifactor decomposition) because
    it credits only the general-factor component of common variance.

    ωh should NOT automatically replace ωt as the default headline
    reliability statistic: most applications (unidimensional or
    short scales) are adequately summarised by ωt alone, and ωh
    specifically answers a multidimensionality-aware question that is
    only meaningful when a genuine general-plus-group-factor structure
    exists in the data.

    GATING (honesty over forced computation): a bifactor structure
    requires at least two extractable group factors, each with enough
    items to be identified. This function requires:
      (a) k >= min_items (default 6) -- with fewer items, a 2+ group
          factor oblique rotation is under-identified / unstable.
      (b) The number of eigenvalues > 1 (Kaiser criterion) in R is >= 2
          -- if only one factor is substantively supported by the data,
          there is no second group factor to extract, and ωh is not a
          meaningful quantity for this construct (it should be reported
          as "not computable", not forced to a possibly-unstable value).
    If either condition fails, ωh is reported as unavailable with an
    explicit, honest reason -- never silently omitted or approximated.

    Returns a dict: {"omega_h": float or nan, "computable": bool,
    "reason": str, "n_factors_used": int, "general_loadings": array,
    "specific_loadings": array (k×m), "item_names": list}
    """
    k = X.shape[1]
    names = item_names if item_names and len(item_names) >= k else [f"Item {i+1}" for i in range(k)]
    base = {"omega_h": np.nan, "computable": False, "reason": "", "n_factors_used": 0,
            "general_loadings": np.array([]), "specific_loadings": np.array([]),
            "item_names": names}
    if k < min_items:
        base["reason"] = (f"Omega hierarchical requires at least {min_items} items to "
                           f"support an identifiable bifactor (general + group-factor) "
                           f"structure; this construct has {k}.")
        return base
    try:
        R = np.corrcoef(X, rowvar=False)
        vals = np.linalg.eigvalsh(R)[::-1]
        n_factors = int(np.sum(vals > 1.0))
        if n_factors < 2:
            base["reason"] = (f"Only {n_factors} eigenvalue(s) of the inter-item "
                               f"correlation matrix exceed 1.0 (Kaiser criterion); the "
                               f"data support a single dominant factor, not a "
                               f"multi-group-factor bifactor structure. Omega "
                               f"hierarchical is not a meaningful quantity for an "
                               f"essentially unidimensional construct — use omega "
                               f"total (\u03c9t) instead.")
            return base
        m = min(n_factors, k // 3)   # practical cap: keep >=3 items/factor on average
        m = max(m, 2)
        Pattern, comm_mf = _paf_multi_factor(R, m)
        Pattern_rot, Phi = _gpa_oblimin(
            Pattern, max_iter=max_rotation_iter if max_rotation_iter is not None else 5000)
        G, S, g_on_groups = _schmid_leiman(Pattern_rot, Phi)
        Vt_observed = k + 2 * np.sum(np.triu(R, 1))
        if Vt_observed <= 1e-9:
            base["reason"] = "Total observed variance is degenerate (near zero)."
            return base
        omega_h = float(G.sum() ** 2 / Vt_observed)
        omega_t_bifactor = float((G.sum() ** 2 + np.sum(S.sum(axis=0) ** 2)) / Vt_observed)
        return {"omega_h": omega_h, "computable": True,
                "reason": (f"Estimated via a {m}-factor oblique PAF extraction "
                            f"(direct oblimin rotation) followed by Schmid-Leiman "
                            f"bifactor transformation."),
                "n_factors_used": m,
                "general_loadings": G, "specific_loadings": S,
                "omega_t_bifactor": omega_t_bifactor,
                "item_names": names}
    except Exception as e:
        base["reason"] = f"Omega hierarchical could not be estimated ({type(e).__name__})."
        return base


def _icc_if_deleted(X, item_names):
    """ICC(C,k) recomputed with each item removed in turn — diagnostic
    analogue of "Cronbach's alpha if item deleted" (see _item_total_stats),
    applied to the Internal-Consistency-context primary ICC metric.

    OPTIMIZATION (found necessary during unrelated bootstrap-CI
    development work, when testing at larger item counts k revealed this
    function as an unexpected, pre-existing performance bottleneck):
    rather than recomputing ALL SIX ICC forms via pingouin's
    intraclass_corr() for each of the k item-deleted scenarios --
    pingouin's own pandas-heavy DataFrame construction made each such
    call take roughly 0.9-1 second, so the full loop scaled to ~k
    seconds, e.g. ~14.5s at k=15 -- this function instead computes
    Cronbach's alpha directly (pure NumPy, effectively instantaneous)
    for each reduced item set, relying on the PROVEN mathematical
    identity ICC(C,k) = Cronbach's alpha (both equal (MSB-MSE)/MSB from
    the same two-way ANOVA decomposition; already documented and relied
    upon elsewhere in this module's ICC section). Verified to agree with
    pingouin's own ICC(C,k) output to machine precision (~1e-16 absolute
    difference) before this change was made. This is a pure speed
    optimization with NO change to the reported values.

    For each item j, the k-1 remaining items are re-submitted and the
    ICC(C,k)-equivalent value is extracted. This lets a user identify
    which specific item, if dropped, would most improve (or most harm)
    the two-way-consistency reliability estimate — the same diagnostic
    purpose as alpha-if-deleted, applied to the ANOVA-based estimator
    (identical in value to Cronbach's alpha for this specific ICC form).

    Returns a DataFrame with columns ["Item", "ICC(C,k) if Item Deleted"],
    or an empty DataFrame if k < 4 (need at least 3 remaining items for a
    meaningful ICC(C,k) on the reduced matrix).
    """
    k = X.shape[1]
    rows = []
    if k < 4:
        return pd.DataFrame(columns=["Item", "ICC(C,k) if Item Deleted"])
    for j in range(k):
        Xd = np.delete(X, j, axis=1)
        try:
            val = _cronbach_alpha(Xd)
            if isinstance(val, float) and np.isnan(val):
                val = np.nan
        except Exception:
            val = np.nan
        rows.append({"Item": item_names[j], "ICC(C,k) if Item Deleted": val})
    return pd.DataFrame(rows)


_PINGOUIN_LEGACY_ICC_TYPES = {
    # pingouin < 0.7 labelled the six forms ICC1 ... ICC3k; pingouin >= 0.7
    # uses the McGraw & Wong (1996) notation below. Both are accepted.
    "ICC1": "ICC(1,1)", "ICC2": "ICC(A,1)", "ICC3": "ICC(C,1)",
    "ICC1k": "ICC(1,k)", "ICC2k": "ICC(A,k)", "ICC3k": "ICC(C,k)",
}


def _icc_confidence_intervals(X, alpha=0.05):
    """(1 - alpha) confidence intervals for the six ICC forms, using the
    F-distribution-based formulas of McGraw, K.O., & Wong, S.P. (1996).
    Forming inferences about some intraclass correlation coefficients.
    Psychological Methods, 1(1), 30-46 -- the same formulas pingouin's
    intraclass_corr() applies, there with the confidence level fixed at
    95%. Evaluated here at the user-selected alpha; at alpha=0.05 the
    bounds are identical to pingouin's own (checked in the test suite).

    X: (n targets x k raters) complete matrix. Returns dict Type -> (lo, hi).
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    grand = X.mean()
    ss_rows = k * np.sum((X.mean(axis=1) - grand) ** 2)
    ss_cols = n * np.sum((X.mean(axis=0) - grand) ** 2)
    ss_tot = np.sum((X - grand) ** 2)
    ss_err = ss_tot - ss_rows - ss_cols
    df_r, df_c, df_e, df_w = n - 1, k - 1, (n - 1) * (k - 1), n * (k - 1)
    msb = ss_rows / df_r
    msj = ss_cols / df_c
    mse = ss_err / df_e
    msw = (ss_tot - ss_rows) / df_w
    f1 = msb / msw
    f3 = msb / mse
    icc2 = (msb - mse) / (msb + (k - 1) * mse + k * (msj - mse) / n)
    fdist = scipy_stats.f
    # One-way (case 1) and consistency (case 3)
    f1l = f1 / fdist.ppf(1 - alpha / 2, df_r, df_w)
    f1u = f1 * fdist.ppf(1 - alpha / 2, df_w, df_r)
    f3l = f3 / fdist.ppf(1 - alpha / 2, df_r, df_e)
    f3u = f3 * fdist.ppf(1 - alpha / 2, df_e, df_r)
    l1, u1 = (f1l - 1) / (f1l + (k - 1)), (f1u - 1) / (f1u + (k - 1))
    l3, u3 = (f3l - 1) / (f3l + (k - 1)), (f3u - 1) / (f3u + (k - 1))
    # Absolute agreement (case 2): Satterthwaite-approximated df v
    fj = msj / mse
    vn = df_e * (k * icc2 * fj + n * (1 + (k - 1) * icc2) - k * icc2) ** 2
    vd = df_r * k ** 2 * icc2 ** 2 * fj ** 2 + (n * (1 + (k - 1) * icc2) - k * icc2) ** 2
    v = vn / vd
    f2u = fdist.ppf(1 - alpha / 2, n - 1, v)
    f2l = fdist.ppf(1 - alpha / 2, v, n - 1)
    l2 = n * (msb - f2u * mse) / (f2u * (k * msj + (k * n - k - n) * mse) + n * msb)
    u2 = n * (f2l * msb - mse) / (k * msj + (k * n - k - n) * mse + n * f2l * msb)
    return {
        "ICC(1,1)": (l1, u1),
        "ICC(A,1)": (l2, u2),
        "ICC(C,1)": (l3, u3),
        "ICC(1,k)": (1 - 1 / f1l, 1 - 1 / f1u),
        "ICC(A,k)": (l2 * k / (1 + l2 * (k - 1)), u2 * k / (1 + u2 * (k - 1))),
        "ICC(C,k)": (1 - 1 / f3l, 1 - 1 / f3u),
    }


def _icc(X, item_names, alpha=0.05):
    """Intraclass correlation, all six forms, via pingouin's
    intraclass_corr() (point estimates, F tests, p-values), with the
    confidence intervals evaluated at the requested alpha by
    _icc_confidence_intervals() (McGraw & Wong, 1996).

    Notation (McGraw & Wong, 1996; Shrout & Fleiss, 1979 numbering in
    brackets):
        ICC(1,1)  one-way random effects, single rater          [ICC(1,1)]
        ICC(A,1)  two-way, absolute agreement, single rater     [ICC(2,1)]
        ICC(C,1)  two-way, consistency, single rater            [ICC(3,1)]
        ICC(1,k), ICC(A,k), ICC(C,k): the same three, for the mean of k raters.

    The CI column is named "CI<level>%" (e.g. "CI95%" for alpha=0.05).
    Returns DataFrame or None when data is degenerate.
    """
    if pg is None:
        return None
    try:
        if float(np.var(X)) < 1e-12:
            return None
        item_vars = np.var(X, axis=0)
        row_vars  = np.var(X, axis=1)
        if (item_vars < 1e-12).any() or (row_vars < 1e-12).all():
            return None
        n, k = X.shape
        long = pd.DataFrame({
            "target": np.repeat(np.arange(n), k),
            "rater":  np.tile(np.arange(k), n),
            "score":  X.flatten(),
        })
        res = pg.intraclass_corr(data=long, targets="target",
                                 raters="rater", ratings="score")
        out = res[["Type", "ICC", "F", "df1", "df2", "pval"]].copy()
        out["Type"] = out["Type"].replace(_PINGOUIN_LEGACY_ICC_TYPES)
        if out["ICC"].isna().all():
            return None
        ci_by_type = _icc_confidence_intervals(X, alpha=alpha)
        ci_col = f"CI{round((1 - alpha) * 100):g}%"
        out[ci_col] = [np.array(ci_by_type.get(t, (np.nan, np.nan)), dtype=float)
                       for t in out["Type"]]
        # ICC and F as plain floats; pval is NOT rounded here (a genuinely
        # tiny p-value must not become a literal 0.0 -- display-time
        # formatting is the responsibility of utils/formatters.py).
        for col in ["ICC", "F"]:
            out[col] = out[col].apply(lambda v: float(v) if pd.notna(v) else np.nan)
        return out
    except Exception:
        return None


def _parse_icc_df(icc_df):
    """Parse the ICC DataFrame returned by _icc() into a named dict.
    Type names follow McGraw & Wong (1996): ICC(1,1), ICC(A,1), ICC(C,1),
    ICC(1,k), ICC(A,k), ICC(C,k) [Shrout & Fleiss numbering: ICC(1,1),
    ICC(2,1), ICC(3,1), ICC(1,k), ICC(2,k), ICC(3,k)]. The confidence-
    interval column is whichever "CI<level>%" column _icc() produced.
    """
    if icc_df is None or not isinstance(icc_df, pd.DataFrame):
        return None
    ci_col = next((c for c in icc_df.columns if str(c).startswith("CI")), None)
    _ci_level = ci_col[2:] if ci_col else "95%"

    def _get(itype):
        row = icc_df[icc_df["Type"] == itype]
        if row.empty:
            return np.nan, np.nan, np.nan, np.nan
        v  = float(row["ICC"].iloc[0])
        p  = float(row["pval"].iloc[0]) if "pval" in row.columns else np.nan
        ci = row[ci_col].iloc[0] if ci_col else [np.nan, np.nan]
        if isinstance(ci, str):
            # Parse string like "[-0.32, 0.15]"
            try:
                parts = ci.strip("[]").split(",")
                lo, hi = float(parts[0]), float(parts[1])
            except Exception:
                lo, hi = np.nan, np.nan
        elif hasattr(ci, "__len__") and len(ci) >= 2:
            lo, hi = float(ci[0]), float(ci[1])
        else:
            lo, hi = np.nan, np.nan
        return v, p, lo, hi

    # NOTE: values are returned at FULL floating-point precision here.
    # Rounding was previously applied at this parsing stage (round to 3dp
    # for ICC/CI, 4dp for p) — this silently converted any genuinely tiny
    # but nonzero p-value (e.g. 3.1e-10, entirely plausible with a large
    # F-statistic under strong rater agreement) into a literal 0.0 float,
    # which downstream code cannot distinguish from "not computed" and
    # which is statistically dishonest to report as an exact zero. All
    # display-time rounding and near-zero thresholding is now the sole
    # responsibility of the _f()/_p() formatters in utils/formatters.py,
    # which already implement correct "< .00001"-style threshold notation.
    def _lbl(v):
        if isinstance(v, float) and np.isnan(v): return "—"
        if v < 0.50: return "Poor (<0.50)"
        if v < 0.75: return "Moderate (0.50–0.75)"
        if v < 0.90: return "Good (0.75–0.90)"
        return "Excellent (>0.90)"

    # All 6 ICC forms from pingouin (Shrout & Fleiss, 1979):
    # ICC(1,1) = one-way random, single rater
    # ICC(A,1) = two-way random, absolute agreement, single rater  [PRIMARY]
    # ICC(C,1) = two-way mixed, consistency, single rater
    # ICC(1,k) = one-way random, mean of k raters
    # ICC(A,k) = two-way random, absolute agreement, mean of k raters
    # ICC(C,k) = two-way mixed, consistency, mean of k raters
    i11, p11, l11, h11 = _get("ICC(1,1)")
    i21, p21, l21, h21 = _get("ICC(A,1)")
    i31, p31, l31, h31 = _get("ICC(C,1)")
    i1k, p1k, l1k, h1k = _get("ICC(1,k)")
    i2k, p2k, l2k, h2k = _get("ICC(A,k)")
    i3k, p3k, l3k, h3k = _get("ICC(C,k)")

    return {
        "icc_df":           icc_df,
        # ICC(1,1) — one-way random, single rater
        "icc_1_1":          i11, "icc_1_1_p":     p11,
        "icc_1_1_ci_lo":    l11, "icc_1_1_ci_hi": h11,
        "icc_1_1_label":    _lbl(i11),
        # ICC(A,1) — two-way random, absolute agreement, single rater [PRIMARY]
        "icc_2_1":          i21, "icc_2_1_p":     p21,
        "icc_2_1_ci_lo":    l21, "icc_2_1_ci_hi": h21,
        "icc_2_1_label":    _lbl(i21),
        # ICC(C,1) — two-way mixed, consistency, single rater
        "icc_3_1":          i31, "icc_3_1_p":     p31,
        "icc_3_1_ci_lo":    l31, "icc_3_1_ci_hi": h31,
        "icc_3_1_label":    _lbl(i31),
        # ICC(1,k) — one-way random, mean of k raters
        "icc_1_k":          i1k, "icc_1_k_p":     p1k,
        "icc_1_k_ci_lo":    l1k, "icc_1_k_ci_hi": h1k,
        "icc_1_k_label":    _lbl(i1k),
        # ICC(A,k) — two-way random, absolute agreement, mean of k raters
        "icc_2_k":          i2k, "icc_2_k_p":     p2k,
        "icc_2_k_ci_lo":    l2k, "icc_2_k_ci_hi": h2k,
        "icc_2_k_label":    _lbl(i2k),
        # ICC(C,k) — two-way mixed, consistency, mean of k raters
        "icc_3_k":          i3k, "icc_3_k_p":     p3k,
        "icc_3_k_ci_lo":    l3k, "icc_3_k_ci_hi": h3k,
        "icc_3_k_label":    _lbl(i3k),
        "interpretation_text": "",  # filled by caller
        "formula_note": (
            "Point estimates, F tests and p-values: pingouin intraclass_corr "
            "(Shrout & Fleiss, 1979; McGraw & Wong, 1996). "
            "ICC(2,1) [pingouin: ICC(A,1)]: two-way random effects, absolute agreement, single rater — "
            "PRIMARY index: use when each unit is rated by the same raters drawn randomly from a larger pool. "
            "ICC(2,k) [pingouin: ICC(A,k)]: absolute agreement for mean of k raters — "
            "use when you will average all raters' scores in practice. "
            "ICC(3,1) [pingouin: ICC(C,1)]: two-way mixed, consistency, single rater — "
            "use when the same fixed set of raters rates all units (no random sampling of raters). "
            "ICC(3,k) [pingouin: ICC(C,k)]: consistency for mean of k raters. "
            "Absolute agreement (ICC2) is stricter: it penalises systematic rater bias. "
            "Consistency (ICC3) does not. "
            f"{_ci_level} CI via the F distribution (McGraw & Wong, 1996). "
            "Benchmarks — Koo & Li (2016): <0.50 Poor; 0.50–0.75 Moderate; "
            "0.75–0.90 Good; >0.90 Excellent."),
    }


# ══════════════════════════════════════════════════════════════════════════════
# REVERSE-ITEM HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def _reverse(X, method, scale_min=None, scale_max=None, which=None):
    """Return a copy of X with the selected columns reversed."""
    Xr = X.astype(float).copy()
    which = which if which is not None else range(X.shape[1])
    info = []
    for j in which:
        col = Xr[:, j]
        if method == "theoretical" and scale_min is not None and scale_max is not None:
            Xr[:, j] = (scale_min + scale_max) - col
            info.append((j, f"({scale_min} + {scale_max}) \u2212 x"))
        elif method == "observed":
            mn, mx = col.min(), col.max()
            Xr[:, j] = (mn + mx) - col
            info.append((j, f"({mn:g} + {mx:g}) \u2212 x [observed]"))
        elif method == "zscore":
            sd = col.std(ddof=1)
            Xr[:, j] = -((col - col.mean()) / sd) if sd > 0 else col
            info.append((j, "\u2212z(x)"))
    return Xr, info


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT (single construct)
# ══════════════════════════════════════════════════════════════════════════════

def run_reliability(matrix, item_names, alpha=0.05, construct_name="Construct",
                    reverse_method=None, reverse_items=None,
                    scale_min=None, scale_max=None,
                    compute_bootstrap_ci=False, n_boot_ic=1000,
                    compute_omega_h=False):
    """Run reliability analysis on one construct (item matrix).

    compute_omega_h : bool, default False
        When True, computes McDonald's omega-hierarchical (Section 4.3.4).
        Defaults to False (skipped) because investigation found its
        oblique-rotation extraction can take substantial, unpredictable
        time for certain data structures (observed: single point-estimate
        calls ranging from milliseconds to over ten seconds depending on
        the specific data realization, not reliably predictable from item
        count k or a fixed iteration cap alone) -- this protects the
        speed of every other coefficient in this module, none of which
        share this issue, for the default analysis run. The interactive
        Web interface explicitly opts in, behind its own loading
        indicator and an explicit warning that computation may take
        several seconds, when the user requests this specific coefficient.

    compute_bootstrap_ci : bool, default False
        When True, additionally computes bootstrap confidence intervals
        for Guttman's lambda family (all 6 forms, both bases), all three
        Split-Half variants, and McDonald's omega-u (both bases) --
        Section 4.3-4.5 CI additions. Defaults to False (skipped
        entirely) because this computation, while individually fast per
        coefficient, is meaningfully slower in aggregate across the ~15
        distinct bootstrap runs involved (empirically: roughly 15-35
        seconds total, depending on item count k) than the base
        reliability computation without it (a fraction of a second) --
        callers that do not need these confidence intervals (including
        the permanent automated test suite's existing coverage of other
        functionality) should not silently pay this cost. The
        interactive Web interface explicitly opts in via this parameter,
        shown behind its own loading indicator (Section: UI), when the
        user is viewing results for a construct where these coefficients
        apply.

    n_boot_ic : int, default 1000
        Bootstrap replicate count used when compute_bootstrap_ci=True.

    Parameters
    ----------
    matrix : array-like (N, k)  -- listwise-complete numeric item scores
    item_names : list[str], length k
    alpha : float -- significance level for CIs (default 0.05 -> 95%)
    construct_name : str
    reverse_method : {None,'theoretical','observed','zscore'}
    reverse_items : list[int] | None -- indices to reverse; None = auto-detect
    scale_min, scale_max : numbers (for 'theoretical')

    Returns
    -------
    dict
    """
    X = np.asarray(matrix, dtype=float)
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    N, k = X.shape
    item_names = list(item_names)
    conf = 1 - alpha
    pct = int(conf * 100)

    # ── Detect reverse items (negative corrected item-total correlation) ──
    flagged = []
    total = X.sum(axis=1)
    for j in range(k):
        rest = total - X[:, j]
        try:
            cit = np.corrcoef(X[:, j], rest)[0, 1]
            if cit < 0:
                flagged.append(item_names[j])
        except Exception:
            pass

    # ── Apply reversal if requested ──
    reverse_info = []
    if reverse_method:
        if reverse_items is None:
            # auto: reverse the flagged items
            idx = [item_names.index(v) for v in flagged]
        else:
            idx = reverse_items
        if idx:
            X, rinfo = _reverse(X, reverse_method, scale_min, scale_max, idx)
            reverse_info = [(item_names[j], f) for j, f in rinfo]

    # ── Coefficients (each protected) ──
    results = {}
    try: results["alpha"] = _cronbach_alpha(X)
    except Exception: results["alpha"] = np.nan
    try: results["alpha_std"] = _standardised_alpha(X)
    except Exception: results["alpha_std"] = np.nan
    try: results["alpha_ci"] = _alpha_ci(results["alpha"], N, k, conf)
    except Exception: results["alpha_ci"] = (np.nan, np.nan)
    try:
        results["dichotomous"] = _is_dichotomous(X)
        if results["dichotomous"]:
            _kr_res = _kr20(X)
            results["kr20"]           = _kr_res["kr20"]
            results["item_p"]         = _kr_res["item_p"]      # proportion correct per item
            results["item_pq"]        = _kr_res["item_pq_sample"]  # backward-compat alias
            results["item_pq_raw"]    = _kr_res["item_pq_raw"]
            results["item_pq_sample"] = _kr_res["item_pq_sample"]
            results["kr21"]           = _kr21(X)
        else:
            results["kr20"] = np.nan
            results["kr21"] = np.nan
            results["item_p"]  = np.array([])
            results["item_pq"] = np.array([])
            results["item_pq_raw"]    = np.array([])
            results["item_pq_sample"] = np.array([])
    except Exception:
        results["dichotomous"] = False; results["kr20"] = np.nan; results["kr21"] = np.nan
    try: results["lambdas"] = _guttman_lambdas(X)
    except Exception: results["lambdas"] = {}
    try:
        results["split_half"] = _split_half(X)
    except Exception:
        results["split_half"] = {}
    try:
        results["omega_detail"] = _mcdonald_omega(X, item_names)  # full per-item breakdown
    except Exception:
        results["omega_detail"] = {"omega_model_implied": np.nan,
                                    "omega_observed": np.nan, "loadings": np.array([]),
                                    "communalities": np.array([]),
                                    "unique_variances": np.array([]), "item_names": []}
    try:
        if compute_omega_h:
            # SAFETY BOUND: investigation during this development cycle
            # found the oblique rotation's default max_iter (5000) can,
            # for certain data, take many seconds for a SINGLE point-
            # estimate call -- empirically confirmed NON-monotonic with
            # iteration count in some cases (cap=100 completed instantly
            # with a value identical to much slower higher caps), but
            # NOT reliably resolved by capping alone: different random
            # data realizations at the same k and same cap were also
            # found to occasionally take several seconds regardless.
            # Given this could not be resolved to a uniformly fast,
            # predictable bound, omega-hierarchical's ENTIRE computation
            # (not merely its bootstrap CI) is opt-in via this parameter,
            # defaulting to False/skipped, protecting the speed of every
            # other coefficient in this module (which have no such
            # issue) for the default case. The 200-iteration cap is
            # still applied when explicitly requested, as a partial
            # mitigation.
            results["omega_h_detail"] = _mcdonald_omega_hierarchical(
                X, item_names, max_rotation_iter=200)
        else:
            results["omega_h_detail"] = {
                "omega_h": np.nan, "computable": False,
                "reason": (
                    "Omega-hierarchical was not computed for this run. Its "
                    "computation is opt-in (compute_omega_h=True) because "
                    "investigation found its oblique-rotation extraction can "
                    "take substantial, unpredictable time for certain data "
                    "structures, independent of item count -- this protects "
                    "the speed of every other reliability coefficient, which "
                    "have no such issue, for the default analysis run."),
                "n_factors_used": 0, "general_loadings": np.array([]),
                "specific_loadings": np.array([]), "item_names": item_names,
                "not_computed_by_default": True}
    except Exception:
        results["omega_h_detail"] = {"omega_h": np.nan, "computable": False,
                                      "reason": "Omega hierarchical computation failed.",
                                      "n_factors_used": 0, "general_loadings": np.array([]),
                                      "specific_loadings": np.array([]), "item_names": item_names}

    # ══════════════════════════════════════════════════════════════════
    # Bootstrap confidence intervals: Guttman's lambda family, Split-Half
    # (all three variants), and McDonald's Omega (both omega-u bases and
    # omega-hierarchical). Method recommended by Kelley, K., &
    # Pornprasertmanit, S. (2016). Psychological Methods, 21(1), 69-92,
    # for reliability coefficients generally (bootstrap methods showed
    # the best empirical coverage for the smaller sample sizes common in
    # applied reliability research, per subsequent Monte Carlo
    # comparisons of omega-CI methods specifically).
    #
    # Fast, closed-form coefficients (lambda1/2/3/5/6 both bases,
    # omega-u both bases) use a direct percentile bootstrap
    # (`_bootstrap_ci_percentile`) at the full requested n_boot.
    #
    # Coefficients requiring an expensive per-call search or iterative
    # optimisation (lambda4, the maximum-Spearman-Brown split-half
    # variant, omega-hierarchical) use `_bootstrap_ci_fixed_search`,
    # which automatically selects a full re-search bootstrap for k <= 12
    # items or a fixed-search (search once, bootstrap the value only)
    # approach for k > 12 -- see that function's docstring for the full
    # rationale, benchmarks, and the explicitly disclosed limitation of
    # the fixed-search regime.
    # ══════════════════════════════════════════════════════════════════
    _n_boot_ic = n_boot_ic
    try:
        if compute_bootstrap_ci and k >= 2:
            # -- Guttman's lambda: fast forms (both bases) --
            for _basis_key in ("raw", "standardized"):
                if _basis_key not in results["lambdas"]:
                    continue
                _bX = X if _basis_key == "raw" else (
                    (X - X.mean(axis=0)) / np.where(X.std(axis=0, ddof=1) > 1e-12, X.std(axis=0, ddof=1), 1.0))
                _sub = results["lambdas"][_basis_key]
                for _lam_key in ["lambda1", "lambda2", "lambda3", "lambda5", "lambda6"]:
                    _ci = _bootstrap_ci_percentile(
                        _bX, lambda Xb, _lk=_lam_key: _guttman_lambdas_fast_only(Xb).get(_lk, np.nan),
                        n_boot=_n_boot_ic)
                    _sub[f"{_lam_key}_ci_lo"] = _ci["ci_lo"]
                    _sub[f"{_lam_key}_ci_hi"] = _ci["ci_hi"]
                    _sub[f"{_lam_key}_ci_method"] = _ci["method"]
                # -- lambda4: fixed-search bootstrap, k-dependent regime --
                def _l4_search(Xb, return_search_result=False):
                    _r = _lambda4_exhaustive(Xb, Xb.sum(axis=1).var(ddof=1))
                    return _r if return_search_result else _r["lambda4"]
                def _l4_recompute(Xb, fixed_result):
                    return _lambda4_given_split(Xb, Xb.sum(axis=1).var(ddof=1), fixed_result["best_split"])
                _ci4 = _bootstrap_ci_fixed_search(_bX, k, _l4_search, _l4_recompute, n_boot=_n_boot_ic)
                _sub["lambda4_ci_lo"] = _ci4["ci_lo"]
                _sub["lambda4_ci_hi"] = _ci4["ci_hi"]
                _sub["lambda4_ci_method"] = _ci4["method"]
                _sub["lambda4_ci_k_threshold"] = _BOOTSTRAP_SEARCH_K_THRESHOLD

            # -- Split-Half: first-second and odd-even (fast) --
            if results.get("split_half") and not results["split_half"].get("error"):
                for _variant in ["first_second", "odd_even"]:
                    if _variant not in results["split_half"]:
                        continue
                    _v = results["split_half"][_variant]
                    if _v.get("error"):
                        continue
                    for _stat_key in ["spearman_brown", "guttman"]:
                        def _sh_compute(Xb, _variant=_variant, _stat_key=_stat_key):
                            k_ = Xb.shape[1]
                            if _variant == "first_second":
                                idx1 = list(range(k_ // 2)); idx2 = list(range(k_ // 2, k_))
                            else:
                                idx1 = list(range(0, k_, 2)); idx2 = list(range(1, k_, 2))
                            h1 = Xb[:, idx1].sum(axis=1); h2 = Xb[:, idx2].sum(axis=1)
                            r = np.corrcoef(h1, h2)[0, 1]
                            if np.isnan(r): return np.nan
                            if _stat_key == "spearman_brown":
                                return 2*r/(1+r)
                            else:
                                return 4*r*np.sqrt(h1.var(ddof=1)*h2.var(ddof=1)) / Xb.sum(axis=1).var(ddof=1)
                        _ci = _bootstrap_ci_percentile(X, _sh_compute, n_boot=_n_boot_ic)
                        _v[f"{_stat_key}_ci_lo"] = _ci["ci_lo"]
                        _v[f"{_stat_key}_ci_hi"] = _ci["ci_hi"]
                        _v[f"{_stat_key}_ci_method"] = _ci["method"]
                # -- Split-Half Maximum: fixed-search bootstrap --
                if "maximum" in results["split_half"] and not results["split_half"]["maximum"].get("error"):
                    def _sb_search(Xb, return_search_result=False):
                        _r = _max_spearman_brown_split_exhaustive(Xb)
                        return _r if return_search_result else _r["spearman_brown"]
                    def _sb_recompute(Xb, fixed_result):
                        return _spearman_brown_given_split(Xb, fixed_result["best_split"])
                    _ci_max = _bootstrap_ci_fixed_search(X, k, _sb_search, _sb_recompute, n_boot=_n_boot_ic)
                    results["split_half"]["maximum"]["spearman_brown_ci_lo"] = _ci_max["ci_lo"]
                    results["split_half"]["maximum"]["spearman_brown_ci_hi"] = _ci_max["ci_hi"]
                    results["split_half"]["maximum"]["spearman_brown_ci_method"] = _ci_max["method"]
                    results["split_half"]["maximum"]["ci_k_threshold"] = _BOOTSTRAP_SEARCH_K_THRESHOLD

            # -- McDonald's omega-u: both bases (fast) --
            for _ok in ["omega_model_implied", "omega_observed"]:
                _ci = _bootstrap_ci_percentile(
                    X, lambda Xb, _ok=_ok: _mcdonald_omega(Xb, item_names).get(_ok, np.nan),
                    n_boot=_n_boot_ic)
                results["omega_detail"][f"{_ok}_ci_lo"] = _ci["ci_lo"]
                results["omega_detail"][f"{_ok}_ci_hi"] = _ci["ci_hi"]
                results["omega_detail"][f"{_ok}_ci_method"] = _ci["method"]

            # -- McDonald's omega-hierarchical: CI deliberately NOT computed --
            # Investigated during development: even the POINT ESTIMATE alone
            # (a single, uncapped call) was found to take several seconds for
            # certain real (non-pathological) datasets -- specifically, data
            # that only marginally satisfies the >=2-eigenvalue applicability
            # gate without a genuinely well-separated bifactor structure,
            # causing the oblique-rotation optimizer to struggle to converge.
            # Bootstrap resampling compounds this: across genuine bootstrap
            # replicates of such data, omega_h values were observed to vary
            # dramatically (e.g. 0.18-0.61 across 30 replicates of the same
            # underlying data in one tested case) even with a reduced
            # rotation-iteration cap -- indicating the oblique rotation does
            # not reliably converge to a comparable solution across resamples
            # for this class of data. A confidence interval computed from an
            # unstable bootstrap distribution would convey false precision
            # rather than genuine uncertainty, so none is computed. This is a
            # deliberate, disclosed limitation, not merely a "not yet
            # implemented" gap: the point estimate itself remains fully
            # computed and reported.
            if results["omega_h_detail"].get("computable"):
                results["omega_h_detail"]["omega_h_ci_lo"] = np.nan
                results["omega_h_detail"]["omega_h_ci_hi"] = np.nan
                results["omega_h_detail"]["omega_h_ci_method"] = "not_computed_disclosed"
                results["omega_h_detail"]["omega_h_ci_reason"] = (
                    "A bootstrap confidence interval is not computed for "
                    "omega-hierarchical. Investigation found that its "
                    "oblique-rotation extraction can require substantial, "
                    "unpredictable computation time for certain data "
                    "structures (data that only marginally satisfies this "
                    "coefficient's applicability gate without a strongly "
                    "separated bifactor structure), and that bootstrap "
                    "replicate values under such conditions can vary "
                    "dramatically rather than converging to a stable "
                    "distribution -- a confidence interval computed under "
                    "these conditions would convey false precision. This "
                    "is a disclosed limitation, not an oversight: the "
                    "point estimate above remains fully computed.")
    except Exception:
        pass

    try: results["item_stats"] = _item_total_stats(X, item_names, results["alpha"])
    except Exception: results["item_stats"] = pd.DataFrame()
    try: results["icc"] = _icc(X, item_names)
    except Exception: results["icc"] = None
    try: results["icc_if_deleted"] = _icc_if_deleted(X, item_names)
    except Exception: results["icc_if_deleted"] = pd.DataFrame(columns=["Item", "ICC(C,k) if Item Deleted"])

    # ── Inter-item correlation & covariance ──
    try:
        results["inter_item_corr"] = pd.DataFrame(
            np.corrcoef(X, rowvar=False),
            index=item_names, columns=item_names)
        iu = np.triu_indices(k, 1)
        results["avg_inter_item_r"] = float(np.corrcoef(X, rowvar=False)[iu].mean())
        results["inter_item_cov"] = pd.DataFrame(
            np.cov(X, rowvar=False, ddof=1),
            index=item_names, columns=item_names)
    except Exception:
        results["inter_item_corr"] = pd.DataFrame()
        results["avg_inter_item_r"] = np.nan
        results["inter_item_cov"] = pd.DataFrame()
    # ── Item statistics (incl. Median + extended descriptives) ──
    item_rows = []
    for j in range(k):
        d = X[:, j]
        nn = len(d)
        skew = float(scipy_stats.skew(d, bias=False)) if nn > 2 else np.nan
        kurt = float(scipy_stats.kurtosis(d, bias=False)) if nn > 3 else np.nan
        item_rows.append({
            "Item": item_names[j], "N": N,
            "Mean": float(d.mean()),
            "Median": float(np.median(d)),
            "Std. Deviation": float(d.std(ddof=1)),
            "Variance": float(d.var(ddof=1)),
            "Minimum": float(d.min()),
            "Maximum": float(d.max()),
            "Range": float(d.max() - d.min()),
            "Skewness": skew if not np.isnan(skew) else np.nan,
            "Kurtosis": kurt if not np.isnan(kurt) else np.nan,
        })
    results["item_descriptives"] = pd.DataFrame(item_rows)

    # ── Summary Item Statistics (aggregate across items) ──
    try:
        means = X.mean(axis=0)
        variances = X.var(axis=0, ddof=1)
        # inter-item correlations (off-diagonal)
        Rc = np.corrcoef(X, rowvar=False)
        iu = np.triu_indices(k, 1)
        inter_r = Rc[iu]
        Cc = np.cov(X, rowvar=False, ddof=1)
        inter_cov = Cc[iu]

        def _summ(arr):
            arr = np.asarray(arr, dtype=float)
            mn, mx = float(arr.min()), float(arr.max())
            rng = mx - mn
            ratio = (mx / mn) if mn != 0 else np.nan
            return [float(arr.mean()), mn, mx,
                    rng, ratio if not np.isnan(ratio) else np.nan,
                    float(arr.var(ddof=1)) if len(arr) > 1 else np.nan,
                    float(np.median(arr))]

        summ_cols = ["", "Mean", "Minimum", "Maximum", "Range",
                     "Maximum / Minimum", "Variance", "Median"]
        summ_rows = [summ_cols]
        summ_rows.append(["Item Means"] + _summ(means))
        summ_rows.append(["Item Variances"] + _summ(variances))
        if len(inter_r) > 0:
            summ_rows.append(["Inter-Item Correlations"] + _summ(inter_r))
            summ_rows.append(["Inter-Item Covariances"] + _summ(inter_cov))
        results["summary_item_stats"] = pd.DataFrame(summ_rows[1:], columns=summ_cols)
    except Exception:
        results["summary_item_stats"] = pd.DataFrame()

    # ── Scale Statistics (full) ──
    tot = X.sum(axis=1)
    nt = len(tot)
    sskew = float(scipy_stats.skew(tot, bias=False)) if nt > 2 else np.nan
    skurt = float(scipy_stats.kurtosis(tot, bias=False)) if nt > 3 else np.nan
    results["scale_stats"] = {
        "n": N, "k": k,
        "mean": float(tot.mean()),
        "median": float(np.median(tot)),
        "variance": float(tot.var(ddof=1)),
        "sd": float(tot.std(ddof=1)),
        "min": float(tot.min()),
        "max": float(tot.max()),
        "range": float(tot.max() - tot.min()),
        "skewness": sskew if not np.isnan(sskew) else np.nan,
        "kurtosis": skurt if not np.isnan(skurt) else np.nan,
    }

    # ── Standard Error of Measurement (SEM) ──
    # Uses the PRIMARY reliability estimate for this construct's context:
    # KR-20 for dichotomous data, Cronbach's alpha otherwise -- the same
    # coefficient already used as this module's headline reliability
    # figure for the relevant data type.
    _sem_reliability = (results.get("kr20") if results.get("dichotomous")
                         else results.get("alpha"))
    results["sem"] = _sem(results["scale_stats"]["sd"], _sem_reliability)
    results["sem_basis"] = "KR-20" if results.get("dichotomous") else "Cronbach's alpha"


    # ── Descriptive interpretation ──
    a = results["alpha"]
    lbl = alpha_label(a)
    ci = results.get("alpha_ci", (np.nan, np.nan))
    interp = (
        f"For the construct <b>{construct_name}</b> ({k} items, N = {N}), "
        f"Cronbach\u2019s alpha = {a:.3f}, indicating <b>{lbl}</b> internal "
        f"consistency."
    )
    if not np.isnan(ci[0]):
        interp += f" The {pct}% confidence interval is [{ci[0]:.3f}, {ci[1]:.3f}]."
    # alpha-if-deleted advice
    its = results.get("item_stats")
    advice = ""
    if its is not None and not its.empty and not np.isnan(a):
        bad = its[its["Cronbach's Alpha if Item Deleted"] > a + 0.001]
        if not bad.empty:
            items = ", ".join(bad["Item"].tolist())
            advice = (f" Removing {items} would increase alpha, suggesting "
                      f"these item(s) may not fit the construct well.")
    interp += advice
    if flagged:
        interp += (f" Note: {', '.join(flagged)} show negative item-total "
                   f"correlations and may be reverse-scored.")
    results["interpretation"] = interp

    results.update({
        "construct_name": construct_name, "n": N, "k": k,
        "item_names": item_names, "alpha_level": alpha,
        "flagged_reverse": flagged, "reverse_info": reverse_info,
        "reverse_method": reverse_method,
    })

    # Inter-rater agreement is computed separately via run_irar()
    # which is called from run_multi_reliability when mode = Inter-Rater Agreement.
    # Here we only zero-out the keys so downstream code is safe.
    results["cohen_kappa"]   = None
    results["fleiss_kappa"]  = None
    results["krippendorff"]  = None
    results["irar_method"]   = None
    results["irar_data_type"]= None

    return results


def _irar_strength_interp(method_key, value, label, n_units, k_raters, data_type):
    """Generate full academic interpretation text for an inter-rater result."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Result could not be computed for this configuration."
    v = float(value)
    # Strength
    if v < 0:
        strength = "poor (worse than chance)"
        implication = (
            "The raters disagreed more than expected by chance, suggesting "
            "systematic bias or fundamental disagreement in rating criteria. "
            "Rater training and clarification of rating criteria are strongly recommended.")
    elif v < 0.20:
        strength = "slight"
        implication = (
            "Agreement barely exceeds chance. The raters are largely inconsistent. "
            "The rating protocol should be thoroughly revised and rater training conducted.")
    elif v < 0.40:
        strength = "fair"
        implication = (
            "Agreement is fair but substantially influenced by chance. "
            "Rater calibration sessions and clearer operational definitions of "
            "rating categories are recommended before using these ratings for research.")
    elif v < 0.60:
        strength = "moderate"
        implication = (
            "Moderate agreement indicates reasonable but imperfect consistency. "
            "Acceptable for exploratory research; improvement is recommended for "
            "high-stakes assessment or publication purposes.")
    elif v < 0.80:
        strength = "substantial"
        implication = (
            "Substantial agreement indicates good reliability. "
            "This level is generally acceptable for most research and applied contexts.")
    else:
        strength = "almost perfect"
        implication = (
            "Almost perfect agreement indicates excellent inter-rater reliability. "
            "The raters demonstrate very high consistency in their assessments.")

    # Method-specific symbol and threshold note
    if "Krippendorff" in method_key:
        sym = "α"; thr = ("Krippendorff (2004) recommends α ≥ 0.800 for reliable data "
                          "and considers α ≥ 0.667 the minimum acceptable threshold; "
                          "he gives no finer breakdown below that threshold -- any "
                          "value under 0.667 is treated as a reason to discard the "
                          "data, not placed on a graded scale.")
        bench_ref = "Krippendorff (2004)"
        # BUG FIX (2026-10-21): `strength`/`implication` just above were
        # computed using the generic Landis & Koch (1977) kappa-style
        # bands (slight/fair/moderate/substantial/almost perfect at
        # 0.20/0.40/0.60/0.80) UNCONDITIONALLY, before this branch runs --
        # wrong for Krippendorff's alpha, which uses its own, different
        # thresholds (0.667/0.800) with no finer tiers at all. This
        # produced genuinely self-contradictory interpretation text for
        # every Krippendorff's alpha result in the app -- e.g. for
        # alpha=0.45, the old text said "indicates MODERATE inter-rater
        # agreement... Acceptable for exploratory research" in the same
        # sentence as "Krippendorff (2004) recommends... the minimum
        # acceptable threshold [is 0.667, above this value]." Overridden
        # here, the same way the ICC branch below already overrides its
        # own strength/implication after the generic computation.
        if v < 0.667:
            strength = "unreliable"
            implication = (
                "Krippendorff (2004) recommends discarding data at this level "
                "rather than drawing conclusions from it -- the raters' "
                "classifications are not consistent enough to support "
                "substantive interpretation. Rater training and clearer "
                "operational definitions of the categories are strongly "
                "recommended before re-collecting these ratings.")
        elif v < 0.80:
            strength = "tentative"
            implication = (
                "This level supports only tentative conclusions (Krippendorff, "
                "2004) -- adequate for preliminary or exploratory work, but "
                "improvement is recommended before relying on these ratings "
                "for confirmatory research or high-stakes decisions.")
        else:
            strength = "reliable"
            implication = (
                "This level meets Krippendorff's (2004) threshold for treating "
                "the data as reliable, supporting confident substantive "
                "conclusions from these ratings.")
    elif "ICC" in method_key:
        sym = "ICC"
        if v >= 0.90:   icc_str = "Excellent"
        elif v >= 0.75: icc_str = "Good"
        elif v >= 0.50: icc_str = "Moderate"
        else:           icc_str = "Poor"
        thr = (f"Koo & Li (2016): {icc_str} reliability "
               f"(thresholds: >0.90 Excellent, 0.75–0.90 Good, "
               f"0.50–0.75 Moderate, <0.50 Poor).")
        bench_ref = "Koo & Li (2016)"
        strength  = icc_str.lower()
        # The Landis & Koch implication sentence set above does not apply
        # to the ICC, whose verbal band comes from Koo & Li (2016) alone;
        # it is dropped so that the text states one band only.
        implication = ""
    else:
        sym = "κ"; thr = ("Landis & Koch (1977) benchmarks: <0 Poor; "
                          "0.00–0.20 Slight; 0.21–0.40 Fair; 0.41–0.60 Moderate; "
                          "0.61–0.80 Substantial; 0.81–1.00 Almost Perfect.")
        bench_ref = "Landis & Koch (1977)"

    dtype_str = {"dichotomous": "dichotomous (2 categories)",
                 "nominal":     "nominal (unordered categorical)",
                 "ordinal":     "ordinal (ordered categorical)",
                 "interval":    "interval/continuous",
                 "ratio":       "ratio/continuous"}.get(data_type, data_type)

    return (
        f"The {method_key} ({sym}\u2009=\u2009{v:.3f}) indicates "
        f"<b>{strength} inter-rater agreement</b> "
        f"among {k_raters} raters rating {n_units} units "
        f"({dtype_str} data; {label}). "
        + (f"{implication} " if implication else "")
        + thr
    )


def run_multi_reliability(df, constructs, alpha=0.05, reverse_method=None,
                          auto_reverse=False, scale_min=None, scale_max=None,
                          compute_bootstrap_ci=False, n_boot_ic=1000,
                          compute_omega_h=False):
    """Run reliability for several constructs.

    Parameters
    ----------
    df : DataFrame
    constructs : list of dict {'name': str, 'items': [col, ...]}
    alpha : float
    reverse_method : None | 'theoretical' | 'observed' | 'zscore'
    auto_reverse : bool -- if True, auto-reverse flagged items using reverse_method
    compute_bootstrap_ci : bool, default False -- see run_reliability's
        own docstring; passed through unchanged, defaulting to skipped
        for speed unless the caller explicitly opts in.
    n_boot_ic : int, default 1000 -- bootstrap replicate count when
        compute_bootstrap_ci=True.
    compute_omega_h : bool, default False -- see run_reliability's own
        docstring; passed through unchanged.
    """
    out = {"constructs": [], "alpha_level": alpha}
    summary_rows = []
    for c in constructs:
        name = c["name"]
        items = c["items"]
        sub = df[items].apply(pd.to_numeric, errors="coerce").dropna()
        rm = reverse_method if auto_reverse else None
        res = run_reliability(
            sub.values, items, alpha=alpha, construct_name=name,
            reverse_method=rm, reverse_items=None,
            scale_min=scale_min, scale_max=scale_max,
            compute_bootstrap_ci=compute_bootstrap_ci, n_boot_ic=n_boot_ic,
            compute_omega_h=compute_omega_h)
        out["constructs"].append(res)
        sh = res.get("split_half", {})
        # BUG FIX (2026-10-21): this summary row previously read
        # omega_detail["omega_observed"] -- but views/internalconsistency.py's
        # own per-construct detail tab explicitly labels omega_model_implied
        # "PRIMARY / headline reliability estimate for this module" and
        # omega_observed "SECONDARY / DIAGNOSTIC estimate" (see
        # _coeff_blocks()'s omega_mi_section / omega_obs_section). This
        # top-level, across-constructs summary table -- whose own caption
        # calls it "a quick numeric comparison" -- was silently showing the
        # SECONDARY basis while every detailed per-construct tab shows the
        # PRIMARY basis as the headline number, a genuine cross-consistency
        # defect: a user comparing the summary table's omega figure against
        # the same construct's own detail tab could see two different
        # numbers with no indication why. Confirmed computationally on the
        # Interval-Ratio sample dataset: omega_observed=0.867356,
        # omega_model_implied=0.868584 (gap=0.001227) -- small here, but the
        # app's own "automatic note" elsewhere warns a larger gap signals
        # one-factor model misfit, so the two values can diverge more
        # substantially on other data. Fixed to use omega_model_implied,
        # matching the designated headline/primary estimate everywhere else
        # in this module.
        summary_rows.append({
            "Construct": name,
            "Items": res["k"],
            "N": res["n"],
            "Cronbach's \u03b1": res["alpha"],
            "\u03b1 (std.)": res.get("alpha_std", np.nan),
            "McDonald's \u03c9": res.get("omega_detail", {}).get("omega_model_implied", np.nan),
            "Guttman \u03bb2": res.get("lambdas", {}).get("raw", {}).get("lambda2", np.nan),
            "Split-Half (SB)": sh.get("spearman_brown", np.nan),
        })
    out["summary"] = pd.DataFrame(summary_rows)
    return out


def run_multi_irar(df, groups, alpha=0.05, n_boot=1000):
    """Run inter-rater agreement for multiple rating groups.
    Each group specifies its own data_type. All relevant methods
    are computed automatically — no single-method selection.

    Parameters
    ----------
    df     : DataFrame
    groups : list of {"name": str, "items": [col,...], "data_type": str}
    alpha  : significance level
    n_boot : bootstrap resamples for Krippendorff's α

    Returns
    -------
    dict with "constructs" and "irar_summary"
    """
    constructs   = []
    summary_rows = []

    for grp in groups:
        name      = grp["name"]
        cols      = grp["items"]
        data_type = grp.get("data_type","nominal")
        sub_full  = df[cols].apply(pd.to_numeric, errors="coerce")
        # BUG FIX (2026-10-21): every coefficient below used to receive
        # this SAME listwise-complete-case matrix (dropna() on ALL rater
        # columns together), including Krippendorff's alpha -- even
        # though _krippendorff_alpha's own pairable-values algorithm is
        # specifically designed to use whatever ratings ARE present in a
        # unit (>=2 valid) without requiring every rater to have scored
        # it. A single missing cell therefore silently dropped an entire
        # unit from Krippendorff's alpha too, contradicting this
        # application's own "handles missing data" characterization of
        # that coefficient. X_complete (listwise-deleted) is kept for
        # every OTHER coefficient here, all of which genuinely require a
        # complete row of k ratings per unit (Cohen's/Fleiss' kappa,
        # AC1/AC2, ICC) or are themselves pairwise with internal NaN-
        # masking already applied on a 2-column slice (Scott's pi,
        # Brennan-Prediger, Lin's CCC); X_with_missing (NaN preserved) is
        # used ONLY for Krippendorff's alpha, below.
        sub       = sub_full.dropna()
        X         = sub.values.astype(float)
        X_with_missing = sub_full.values.astype(float)
        n_units, k_raters = X.shape

        # Guard: minimum requirements for IRAR computation
        if k_raters < 2:
            constructs.append({
                "construct_name": name, "data_type": data_type,
                "n_units": n_units,     "k_raters": k_raters,
                "rater_names": cols,    "applicable_methods": [],
                "error": ("At least 2 rater columns are required. "
                          "Please select ≥ 2 columns for this rating group."),
            })
            continue
        if n_units < 2:
            constructs.append({
                "construct_name": name, "data_type": data_type,
                "n_units": n_units,     "k_raters": k_raters,
                "rater_names": cols,    "applicable_methods": [],
                "error": (f"Insufficient data: {n_units} complete row(s) found after "
                          f"removing rows with missing values. At least 2 complete rows "
                          f"are required. Check for missing values in the selected columns."),
            })
            continue

        # Categorical declarations are binding: the ratings must be integer
        # codes that fit the declared number of categories or scale.
        if data_type in ("dichotomous", "nominal", "ordinal"):
            _codes = np.unique(X_with_missing[~np.isnan(X_with_missing)])
            _q_decl, _s_min = grp.get("q_declared"), grp.get("scale_min")
            _err = None
            if np.any(_codes != np.round(_codes)):
                _err = ("Categorical data types require integer category codes; "
                        "non-integer values were found in the selected columns.")
            elif data_type == "dichotomous" and len(_codes) > 2:
                _err = (f"Dichotomous data have two categories, but {len(_codes)} "
                        f"distinct codes were found in the selected columns.")
            elif data_type == "nominal" and _q_decl is not None and len(_codes) > int(_q_decl):
                _err = (f"{len(_codes)} distinct codes were found, more than the "
                        f"{int(_q_decl)} categories declared for this group.")
            elif (data_type == "ordinal" and _q_decl is not None and _s_min is not None
                  and (_codes.min() < _s_min or _codes.max() > _s_min + _q_decl - 1)):
                _err = (f"Ratings outside the declared scale {int(_s_min)}–"
                        f"{int(_s_min + _q_decl - 1)} were found in the selected columns.")
            if _err:
                constructs.append({
                    "construct_name": name, "data_type": data_type,
                    "n_units": n_units,     "k_raters": k_raters,
                    "rater_names": cols,    "applicable_methods": [],
                    "error": _err,
                })
                continue

        applicable = _get_applicable_methods(data_type, k_raters)

        # ── Mandatory, user-declared instrument-scale information ────────
        # Required (not optional) whenever a q-dependent coefficient
        # (AC1, AC2, Brennan-Prediger) is applicable -- see the primary-
        # theory audit documented in _gwet_ac1's docstring: q must reflect
        # the instrument's declared category count, not merely the count
        # of values observed in this specific sample. The UI enforces
        # this field's presence before allowing computation to run; the
        # engine additionally guards defensively here.
        q_declared      = grp.get("q_declared")
        scale_min       = grp.get("scale_min")

        res = {
            "construct_name":     name,
            "data_type":          data_type,
            "n_units":            n_units,
            "k_raters":           k_raters,
            "rater_names":        cols,
            "applicable_methods": applicable,
            "method_context":     _irar_method_context(data_type, k_raters, applicable),
            # Existing methods
            "cohen_kappa":        None,
            "weighted_kappa":     None,
            "fleiss_kappa":       None,
            "icc":                None,
            "krippendorff":       None,
            # New methods
            "percentage_agreement": None,
            "scott_pi":           None,
            "brennan_prediger":   None,
            "gwet_ac1":           None,
            "gwet_ac2":           None,
            "lin_ccc":            None,
        }

        # ── Percentage Agreement (always first, all configurations) ──────
        if "pct" in applicable:
            pa_res = _percentage_agreement(X, alpha=alpha)
            res["percentage_agreement"] = {
                "pa": pa_res["pa"], "ci_lo": pa_res["ci_lo"], "ci_hi": pa_res["ci_hi"],
                "n": n_units, "k": k_raters,
                "total_pairwise_comparisons": int(n_units * (k_raters*(k_raters-1)//2)),
                "per_unit": pa_res["per_unit"], "n_units_valid": pa_res["n_units_valid"],
                "mean": pa_res["mean"], "sd": pa_res["sd"],
                "minimum": pa_res["minimum"], "maximum": pa_res["maximum"],
                "interpretation_text": _pa_interp(pa_res["pa"], pa_res["ci_lo"],
                                                    pa_res["ci_hi"], n_units, k_raters,
                                                    alpha=alpha),
            }

        # ── Cohen's κ (dichotomous/nominal, k=2) ─────────────────────────
        if "cohen" in applicable and k_raters == 2:
            ck = _cohen_kappa(X, cols, alpha, data_type=data_type)
            if ck:
                ck["interpretation_text"] = _irar_strength_interp(
                    "Cohen's κ", ck["kappa"], ck["kappa_label"],
                    n_units, k_raters, data_type)
                ck["formula_note"] = (
                    "κ = (P_o − P_e) / (1 − P_e). "
                    "P_o = observed proportion of agreement. "
                    "P_e = Σ_c p_c(r1) × p_c(r2) = expected agreement by chance. "
                    "SE for z and p: Fleiss, Cohen & Everitt (1969), Eq. 9 (H0: κ = 0); "
                    "SE for the confidence interval: Eq. 8 of the same paper. "
                    "Prevalence Index (PI) = |a − d| / n; Bias Index (BI) = (b − c) / n, "
                    "from the 2×2 table a=both rate '1', b=rater1 '1'/rater2 '0', "
                    "c=rater1 '0'/rater2 '1', d=both rate '0' (Byrt, Bishop, & Carlin, 1993). "
                    "Reference: Cohen (1960). Educational and Psychological Measurement.")
            res["cohen_kappa"] = ck

        # ── Weighted Cohen's κ (ordinal, k=2) ────────────────────────────
        if "weighted" in applicable and k_raters == 2:
            ck_w = _cohen_kappa(X, cols, alpha, data_type=data_type,
                                q_declared=q_declared, scale_min=scale_min)
            if ck_w and "kappa_quadratic" in ck_w:
                ck_w["primary_value"] = ck_w.get("kappa_quadratic", np.nan)
                ck_w["primary_label"] = ck_w.get("kappa_quadratic_label","—")
                ck_w["interpretation_text"] = _irar_strength_interp(
                    "Weighted Cohen's κ (quadratic)",
                    ck_w.get("kappa_quadratic", np.nan),
                    ck_w.get("kappa_quadratic_label","—"),
                    n_units, k_raters, data_type)
                ck_w["formula_note"] = (
                    "κ_w = (Σ w_ij p_ij − Σ w_ij p_i· p_·j) / (1 − Σ w_ij p_i· p_·j) "
                    "(Cohen, 1968), with agreement weights over the q categories of "
                    "the declared scale: quadratic w_ij = 1 − (i−j)²/(q−1)², "
                    "linear w_ij = 1 − |i−j|/(q−1). "
                    "Quadratic weights penalise larger disagreements "
                    "disproportionately more; with quadratic weights κ_w is "
                    "closely related to the intraclass correlation coefficient "
                    "(Fleiss & Cohen, 1973). SE, z and p: Fleiss, Cohen & "
                    "Everitt (1969).")
            res["weighted_kappa"] = ck_w

        # ── Fleiss' κ (dichotomous/nominal, k≥3) ─────────────────────────
        if "fleiss" in applicable and k_raters >= 3:
            fk = _fleiss_kappa(X, cols, alpha)
            if fk:
                fk["interpretation_text"] = _irar_strength_interp(
                    "Fleiss' κ", fk["kappa"], fk["kappa_label"],
                    n_units, k_raters, data_type)
                fk["formula_note"] = (
                    "κ = (P̄ − P̄_e) / (1 − P̄_e) (Fleiss, 1971). "
                    "P̄ = (1/N)Σ_i[(1/k(k−1))Σ_j n_ij(n_ij−1)]. "
                    "P̄_e = Σ_j p_j² (expected by chance). "
                    "SE for z and p (H0: κ = 0): Fleiss, Nee & Landis (1979); "
                    "SE for the confidence interval: Gwet (2021). "
                    "Per-category κ_j also reported. "
                    "Note: Fleiss' κ treats all categories as nominal — "
                    "for ordinal ≥3 raters, Krippendorff's α (ordinal metric) is more appropriate.")
            res["fleiss_kappa"] = fk

        # ── ICC (interval/ratio) ──────────────────────────────────────────
        if "icc" in applicable:
            try:
                icc_df = _icc(X, cols, alpha=alpha)
                if icc_df is not None and isinstance(icc_df, pd.DataFrame):
                    icc_res = _parse_icc_df(icc_df)
                    if icc_res is not None:
                        i21 = icc_res.get("icc_2_1", np.nan)
                        icc_res["interpretation_text"] = _irar_strength_interp(
                            "ICC(A,1)", i21, icc_res.get("icc_2_1_label","—"),
                            n_units, k_raters, data_type)
                        res["icc"] = icc_res
                    else:
                        res["icc"] = {"error": "ICC parsing failed."}
                else:
                    res["icc"] = {"error": "ICC not computable. Check: pingouin installed? ≥2 raters? No zero-variance columns?"}
            except Exception as e:
                res["icc"] = {"error": str(e)}

        # ── Krippendorff's α (all types, all rater counts) ───────────────
        if "krippendorff" in applicable:
            dt_map = {"dichotomous":"nominal","nominal":"nominal",
                      "ordinal":"ordinal","interval":"interval","ratio":"ratio"}
            dt_k = dt_map.get(data_type,"nominal")
            ka_val, ka_ci = _krippendorff_alpha(X_with_missing, data_type=dt_k, n_boot=n_boot, alpha=alpha)
            ka_lbl = krippendorff_label(ka_val)
            # Units actually contributing to alpha: any unit with >=2 valid
            # (non-missing) ratings -- NOT the listwise-complete-case count
            # `n_units` used by every other coefficient above (see the
            # BUG FIX note on X_with_missing where it is built). Reporting
            # the complete-case n_units here would understate how much
            # data alpha actually used now that the listwise-deletion bug
            # is fixed.
            ka_n_units = int(np.sum((~np.isnan(X_with_missing)).sum(axis=1) >= 2))
            res["krippendorff"] = {
                "alpha":    float(ka_val) if not np.isnan(ka_val) else np.nan,
                "ci_lo":    float(ka_ci[0]) if not np.isnan(ka_ci[0]) else np.nan,
                "ci_hi":    float(ka_ci[1]) if not np.isnan(ka_ci[1]) else np.nan,
                "data_type":dt_k, "n_units":ka_n_units, "k_raters":k_raters,
                "n_boot":   n_boot, "label": ka_lbl,
                "interpretation_text": _irar_strength_interp(
                    "Krippendorff's α", ka_val, ka_lbl, ka_n_units, k_raters, dt_k),
                "formula_note": (
                    f"α = 1 − D_o/D_e (Krippendorff, 2004). "
                    f"D_o = observed disagreement (mean pairwise distance within units). "
                    f"D_e = expected disagreement (mean pairwise distance across all values). "
                    f"Distance metric for {dt_k} data: "
                    + {"nominal":  "d(a,b) = 0 if a=b, else 1.",
                       "ordinal":  "d(a,b) = (Σn_g from a to b − (n_a+n_b)/2)² "
                                   "(cumulative marginal distribution).",
                       "interval": "d(a,b) = (a−b)².",
                       "ratio":    "d(a,b) = ((a−b)/(a+b))²."}[dt_k]
                    + f" {round((1 - alpha) * 100):g}% CI: percentile bootstrap over rated units "
                    f"({n_boot:,} resamples of units with replacement, seed=42). "
                    f"Handles missing data and any number of raters. "
                    f"Thresholds: α≥0.800 reliable; 0.667≤α<0.800 tentatively acceptable; "
                    f"α<0.667 unreliable (Krippendorff, 2004)."),
            }

        # ── Gwet's AC1 (dichotomous/nominal, k=2 or k≥3) ────────────────
        if "gwet_ac1" in applicable:
            ac1_res = _gwet_ac1(X, alpha=alpha, q_declared=q_declared)
            if ac1_res.get("ac1") is not None:
                ac1_res["interpretation_text"] = _gwet_ac_interp(
                    ac1_res["ac1"], ac1_res["label"], "AC1",
                    n_units, k_raters, data_type)
                ac1_res["formula_note"] = (
                    "AC1 = (P_o − P_e) / (1 − P_e). "
                    "P_e(AC1) = (1/(q−1)) × Σ_c π_c(1−π_c), "
                    "where π_c = grand marginal proportion of category c, and "
                    "q = user-declared number of categories in the coding "
                    "scheme (not merely the count observed in this sample -- "
                    "see Gwet, Handbook of Inter-Rater Reliability, 4th ed., Ch. 4). "
                    "Gwet (2008) proposed this P_e to reduce the dependence of the "
                    "coefficient on category prevalence. Variance: delete-one-unit "
                    "jackknife (Efron & Tibshirani, 1993). "
                    "Reference: Gwet (2008). British Journal of Mathematical and "
                    "Statistical Psychology, 61(1), 29–48.")
                ac1_res["dual_ci_note"] = _dual_ci_note(
                    "Gwet's AC1", "jackknife",
                    ac1_res.get("ci_lo_z"), ac1_res.get("ci_hi_z"),
                    ac1_res.get("ci_lo_t"), ac1_res.get("ci_hi_t"),
                    ac1_res.get("ci_df"), alpha=alpha)
            res["gwet_ac1"] = ac1_res

        # ── Scott's π (dichotomous/nominal, k=2 only) ────────────────────
        if "scott_pi" in applicable and k_raters == 2:
            sp_res = _scott_pi(X[:,0], X[:,1], alpha=alpha)
            sp_res["interpretation_text"] = _scott_pi_interp(
                sp_res["pi"], sp_res["label"], sp_res["n"], data_type)
            if sp_res.get("pi") is not None and not np.isnan(sp_res.get("pi", np.nan)):
                sp_res["dual_ci_note"] = _dual_ci_note(
                    "Scott's \u03c0", "gwet2021",
                    sp_res.get("ci_lo_z"), sp_res.get("ci_hi_z"),
                    sp_res.get("ci_lo_t"), sp_res.get("ci_hi_t"),
                    sp_res.get("ci_df"), alpha=alpha)
            res["scott_pi"] = sp_res

        # ── Brennan-Prediger (dichotomous/nominal, k=2 only) ─────────────
        if "bp" in applicable and k_raters == 2:
            bp_res = _brennan_prediger(X[:,0], X[:,1], alpha=alpha, q_declared=q_declared)
            bp_res["interpretation_text"] = _bp_interp(
                bp_res["kappa_bp"], bp_res["label"],
                bp_res["n"], bp_res["q"], data_type)
            # ── Comparison with other chance-corrected coefficients ──────
            # Brennan-Prediger, Cohen's kappa, and Gwet's AC1 are all
            # applicable together for dichotomous/nominal k=2 data and
            # differ ONLY in how expected chance agreement (Pe) is
            # modelled -- BP: uniform 1/q; Cohen's: product of each
            # rater's own marginals; AC1: pooled/grand marginals via the
            # (q-1) denominator. Both Cohen's kappa and AC1 are already
            # computed earlier in this same function call (see above);
            # this reuses those results rather than recomputing them.
            _bp_comparison = []
            _ck_for_cmp = res.get("cohen_kappa")
            if _ck_for_cmp and _ck_for_cmp.get("kappa") is not None:
                _bp_comparison.append(("Cohen's \u03ba", _ck_for_cmp["kappa"],
                                        "Product of each rater's own marginal proportions"))
            _ac1_for_cmp = res.get("gwet_ac1")
            if _ac1_for_cmp and _ac1_for_cmp.get("ac1") is not None and not np.isnan(_ac1_for_cmp.get("ac1", np.nan)):
                _bp_comparison.append(("Gwet's AC1", _ac1_for_cmp["ac1"],
                                        "Pooled (grand) marginal proportions across both raters"))
            _bp_comparison.append(("Brennan-Prediger", bp_res["kappa_bp"],
                                    "Uniform 1/q -- assumes all categories equally likely a priori"))
            bp_res["comparison_table"] = _bp_comparison
            bp_res["comparison_note"] = (
                "Brennan-Prediger, Cohen's \u03ba, and Gwet's AC1 all take the same "
                "general form \u03ba=(Po\u2212Pe)/(1\u2212Pe) and share the identical "
                "observed agreement Po; they differ only in how Pe (expected "
                "chance agreement) is modelled. Brennan-Prediger's uniform-"
                "probability Pe makes it immune to the prevalence and bias "
                "paradoxes that can distort Cohen's \u03ba when category use is "
                "very unequal between or within raters -- but this comes at the "
                "cost of assuming a chance model (equal category probability) "
                "that may not reflect the actual rating task.")
            if bp_res.get("kappa_bp") is not None and not np.isnan(bp_res.get("kappa_bp", np.nan)):
                bp_res["dual_ci_note"] = _dual_ci_note(
                    "Brennan-Prediger", "analytic",
                    bp_res.get("ci_lo_z"), bp_res.get("ci_hi_z"),
                    bp_res.get("ci_lo_t"), bp_res.get("ci_hi_t"),
                    bp_res.get("ci_df"), alpha=alpha)
            res["brennan_prediger"] = bp_res

        # ── Gwet's AC2 (ordinal, k=2 or k≥3) ────────────────────────────
        # Computes ALL FIVE weighting schemes simultaneously (not a
        # single user-selected scheme) -- per explicit agreement: the
        # application presents every scheme's result together, with its
        # own narrative and reference, rather than requiring the user to
        # pick one via a dropdown.
        if "gwet_ac2" in applicable:
            _ac2_scheme_defs = {
                "quadratic": {
                    "label": "Quadratic (Fleiss-Cohen)",
                    "formula": "w_cc' = 1 − (c−c')²/(q−1)²",
                    "reference": "Fleiss, J.L., & Cohen, J. (1973). The equivalence of "
                                 "weighted kappa and the intraclass correlation coefficient "
                                 "as measures of reliability. Educational and Psychological "
                                 "Measurement, 33(3), 613–619.",
                    "when_to_use": "The most commonly used default weighting scheme; "
                                   "penalises disagreements in proportion to the SQUARE of "
                                   "their category distance, so distant disagreements are "
                                   "penalised disproportionately more than near ones. "
                                   "Appropriate as a general-purpose default for ordinal "
                                   "scales when there is no specific substantive reason to "
                                   "prefer a different decay pattern.",
                },
                "linear": {
                    "label": "Linear (Cicchetti-Allison)",
                    "formula": "w_cc' = 1 − |c−c'|/(q−1)",
                    "reference": "Cicchetti, D., & Allison, T. (1971). A new procedure for "
                                 "assessing reliability of scoring EEG sleep recordings. "
                                 "American Journal of EEG Technology, 11, 101–109.",
                    "when_to_use": "Penalises disagreements in direct proportion to category "
                                   "distance (no disproportionate penalty for distant "
                                   "disagreements). Appropriate when every unit of category "
                                   "distance is considered substantively equally serious, "
                                   "rather than distant disagreements being considered "
                                   "disproportionately worse.",
                },
                "radical": {
                    "label": "Radical",
                    "formula": "w_cc' = 1 − √(|c−c'|/(q−1))",
                    "reference": "Gwet, K.L. (2014). Handbook of Inter-Rater Reliability "
                                 "(4th ed.). Advanced Analytics, LLC.",
                    "when_to_use": "The most severe of the three distance-based schemes: "
                                   "credit falls fastest at the FIRST unit of disagreement "
                                   "and more slowly afterwards, so any partial disagreement "
                                   "receives less credit than under linear weights (which in "
                                   "turn give less than quadratic weights). Appropriate when "
                                   "even a one-category disagreement should already cost "
                                   "most of the credit.",
                },
                "ordinal": {
                    "label": "Ordinal (Gwet)",
                    "formula": "w_cc' = 1 − M_cc'/max(M), with M_cc' = n(n−1)/2 and "
                               "n = |c−c'| + 1 (category positions on the declared scale)",
                    "reference": "Gwet, K.L. (2014). Handbook of Inter-Rater Reliability "
                                 "(4th ed.). Advanced Analytics, LLC.",
                    "when_to_use": "Uses only the rank order of the categories, not their "
                                   "numeric spacing: the penalty grows with the number of "
                                   "category pairs spanned between the two ratings. "
                                   "Appropriate when categories are ordered but the "
                                   "distances between adjacent categories should not be "
                                   "treated as equal-interval.",
                },
                "identity": {
                    "label": "Identity (unweighted)",
                    "formula": "w_cc' = 1 if c=c' else 0",
                    "reference": "Mathematically identical to Gwet's AC1 (verified exactly: "
                                 "see Gwet, K.L. (2014). Handbook of Inter-Rater Reliability "
                                 "(4th ed.). Advanced Analytics, LLC., Chapter 4).",
                    "when_to_use": "No partial credit for any disagreement, however close. "
                                   "Included for direct comparability with AC1 and with "
                                   "Cohen's/Fleiss' unweighted \u03ba; not genuinely an ordinal "
                                   "weighting choice, since it discards the ordering "
                                   "information the other four schemes use.",
                },
            }
            _ac2_schemes_computed = {}
            for _scheme_key, _scheme_meta in _ac2_scheme_defs.items():
                _r = _gwet_ac2(X, weights=_scheme_key, alpha=alpha,
                                q_declared=q_declared, scale_min=scale_min)
                if _r.get("ac2") is not None and not np.isnan(_r.get("ac2", np.nan)):
                    _r["interpretation_text"] = _gwet_ac_interp(
                        _r["ac2"], _r["label"], "AC2", n_units, k_raters, data_type)
                    _r["scheme_label"] = _scheme_meta["label"]
                    _r["weight_formula"] = _scheme_meta["formula"]
                    _r["when_to_use"] = _scheme_meta["when_to_use"]
                    _r["formula_note"] = (
                        f"AC2 = (P_o_w − P_e_w) / (1 − P_e_w). "
                        f"{_scheme_meta['label']}: {_scheme_meta['formula']}. "
                        f"P_e_w = u·Σ_k π_k(1−π_k) / (q(q−1)), where u = the sum of "
                        f"every cell of the q×q weight matrix (including the diagonal) "
                        f"and q = the user-declared number of categories in the coding "
                        f"scheme (not merely the count observed in this sample). "
                        f"Variance via jackknife. Reference: {_scheme_meta['reference']}")
                    _r["dual_ci_note"] = _dual_ci_note(
                        f"Gwet's AC2 ({_scheme_meta['label']})", "jackknife",
                        _r.get("ci_lo_z"), _r.get("ci_hi_z"),
                        _r.get("ci_lo_t"), _r.get("ci_hi_t"),
                        _r.get("ci_df"), alpha=alpha)
                _ac2_schemes_computed[_scheme_key] = _r
            res["gwet_ac2"] = {"schemes": _ac2_schemes_computed,
                                "n": n_units, "k": k_raters,
                                "q_declared": q_declared, "scale_min": scale_min}


        # ── Lin's CCC (interval/ratio, k=2 only) ─────────────────────────
        if "ccc" in applicable and k_raters == 2:
            ccc_res = _lin_ccc(X[:,0], X[:,1], alpha=alpha)
            ccc_res["interpretation_text"] = _ccc_interp(
                ccc_res["ccc"], ccc_res["rho"], ccc_res["Cb"],
                ccc_res["label"], ccc_res["n"])
            ccc_res["formula_note"] = (
                "ρ_c = 2σ_xy / (σ_x² + σ_y² + (μ_x−μ_y)²) = ρ × C_b. "
                "ρ = Pearson correlation (precision); "
                "C_b = 2/(v + 1/v + u²) = bias correction factor (accuracy); "
                "v = σ_x/σ_y (scale shift); u = (μ_x−μ_y)/(σ_x·σ_y)^0.5 (location shift). "
                f"{round((1 - alpha) * 100):g}% CI via Fisher z-transformation on ρ_c. "
                "Reference: Lin (1989). Biometrics, 45(1), 255–268.")
            res["lin_ccc"] = ccc_res

        constructs.append(res)

        # Build comprehensive summary row — all methods
        def _sv(v):
            try: return float(v) if not (isinstance(v,float) and np.isnan(v)) else "—"
            except: return "—"
        def _sp(p):
            try:
                pf=float(p)
                return "< .001" if pf<.001 else f"{pf:.3f}"
            except: return "—"

        row = {
            "Group":       name,
            "Data Type":   data_type.title(),
            "k (Raters)":  k_raters,
            "N (Units)":   n_units,
            "Methods":     ", ".join(applicable),
        }
        # Cohen's κ columns
        ck = res.get("cohen_kappa") or {}
        row["Cohen's κ"]  = _sv(ck.get("kappa"))
        row["κ Sig."]     = _sp(ck.get("kappa_p")) if ck else "—"
        row["κ Label"]    = ck.get("kappa_label","—") if ck else "—"
        # Weighted κ columns
        wk = res.get("weighted_kappa") or {}
        row["Weighted κ (quad)"] = _sv(wk.get("kappa_quadratic")) if wk else "—"
        row["Wκ Label"]           = wk.get("kappa_quadratic_label","—") if wk else "—"
        # Fleiss' κ
        fk = res.get("fleiss_kappa") or {}
        row["Fleiss' κ"]  = _sv(fk.get("kappa")) if fk else "—"
        row["Fκ Sig."]    = _sp(fk.get("kappa_p")) if fk else "—"
        row["Fκ Label"]   = fk.get("kappa_label","—") if fk else "—"
        # ICC
        icc = res.get("icc") or {}
        row["ICC(2,1)"]       = _sv(icc.get("icc_2_1")) if icc and "error" not in icc else "—"
        row["ICC(2,k)"]       = _sv(icc.get("icc_2_k")) if icc and "error" not in icc else "—"
        row["ICC(3,1)"]       = _sv(icc.get("icc_3_1")) if icc and "error" not in icc else "—"
        row["ICC(3,k)"]       = _sv(icc.get("icc_3_k")) if icc and "error" not in icc else "—"
        row["ICC Label"]      = icc.get("icc_2_1_label","—") if icc and "error" not in icc else "—"
        # Krippendorff's α
        ka = res.get("krippendorff") or {}
        row["Kripp. α"]   = _sv(ka.get("alpha")) if ka else "—"
        _lo = ka.get("ci_lo",float("nan")); _hi = ka.get("ci_hi",float("nan"))
        try:
            row["Kripp. CI"] = (f"[{_lo:.3f}, {_hi:.3f}]"
                                    if not (np.isnan(_lo) or np.isnan(_hi)) else "—")
        except: row["Kripp. CI"] = "—"
        row["Kripp. Label"] = ka.get("label","—") if ka else "—"
        summary_rows.append(row)

    return {
        "constructs":   constructs,
        "irar_summary": pd.DataFrame(summary_rows),
        "alpha":        alpha,
        "n_boot":       n_boot,
    }

