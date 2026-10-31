"""
views/internalconsistency.py
=============================
Streamlit page for Internal Consistency reliability analysis.

Covers Cronbach's alpha, KR-20/KR-21, Split-Half reliability (First-Second,
Odd-Even, Maximum variants), Guttman's Lambda (6 forms, raw + standardized
basis), McDonald's Omega (model-implied, observed, and hierarchical), and
ICC(C,k). Also reports full Item Statistics, Summary Item Statistics,
Item-Total Statistics, the Inter-Item Correlation matrix, and Scale
Statistics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Internal Consistency | Reliability Toolkit",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.styles      import load_page_css
from utils.file_loader import (load_file, validate_file,
                                file_metadata, get_numeric_cols)
from utils.formatters  import _f as _f_base, _p as _p_base
from utils.tables      import html_tbl
from utils.plots       import fig_to_bytes

from modules.reliability import run_multi_reliability, alpha_label

# ── Reliability-module display precision (4 decimal places) ───────────────
def _f(v, d=4):
    return _f_base(v, d)

def _p(v, d=4):
    return _p_base(v, d)

def _fmt_ci(lo, hi):
    """Bracket-formatted confidence interval, e.g. '[0.4700, 0.8000]'."""
    try:
        return (f"[{_f(float(lo))}, {_f(float(hi))}]"
                if not (np.isnan(float(lo)) or np.isnan(float(hi))) else "—")
    except Exception:
        return "—"

def _fmt_p_dash(p):
    try:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return "—"
        return _p(float(p))
    except Exception:
        return "—"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = ["#1a1a2e", "#e94560", "#0f3460", "#16213e"]
BG  = "#f8fafc"


# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION HELPERS + PER-COEFFICIENT BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

def _icc_label(v):
    """Koo & Li (2016) interpretation for ICC."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "undefined"
    if v < 0.5:
        return "poor"
    if v < 0.75:
        return "moderate"
    if v < 0.90:
        return "good"
    return "excellent"


def _coeff_blocks(res, alpha):
    """Ordered per-coefficient blocks shared by web/HTML/CSV.

    Each block: dict(key, title, cards=[(label,value,sub)], table=(cols,rows)|None,
                      params, interp, note)
    """
    pct = int((1 - res.get("alpha_level", alpha)) * 100)
    a = res.get("alpha"); ci = res.get("alpha_ci", (np.nan, np.nan))
    a_std = res.get("alpha_std")
    omh = res.get("omega_h_detail") or {}
    lam = res.get("lambdas", {}); sh = res.get("split_half", {})
    kr = res.get("kr20"); dich = res.get("dichotomous")
    k = res["k"]; n = res["n"]
    blocks = []

    a_neg = (a is not None and not (isinstance(a, float) and np.isnan(a)) and a < 0) or \
            (a_std is not None and not (isinstance(a_std, float) and np.isnan(a_std)) and a_std < 0)
    blocks.append({
        "key": "Alpha", "title": "Cronbach's Alpha",
        "cards": [("Cronbach's \u03b1", _f(a), alpha_label(a)),
                  (f"{pct}% CI", f"[{_f(ci[0])}, {_f(ci[1])}]", "Feldt method"),
                  ("Standardised \u03b1", _f(a_std), "from correlations")],
        "table": (["Cronbach's Alpha" + ("\u1d43" if a_neg else ""),
                   "Cronbach's Alpha Based on Standardized Items" + ("\u1d43" if a_neg else ""),
                   f"{pct}% CI", "N of Items"],
                  [[_f(a), _f(a_std), f"[{_f(ci[0])}, {_f(ci[1])}]", str(k)]]),
        "table_left": set(),
        "footnote": ("a. The value is negative due to a negative average covariance "
                     "among items. This violates reliability model assumptions. You "
                     "may want to check item codings.") if a_neg else "",
        "params": ("Thresholds (George &amp; Mallery, 2003): \u2265.9 excellent, "
                   ".8\u2013.89 good, .7\u2013.79 acceptable, .6\u2013.69 questionable, "
                   ".5\u2013.59 poor, &lt;.5 unacceptable."),
        "interp": (f"Cronbach\u2019s alpha = {_f(a)}, indicating <b>{alpha_label(a)}</b> "
                   f"internal consistency for {k} items (N = {n}). The {pct}% "
                   f"confidence interval is [{_f(ci[0])}, {_f(ci[1])}]."
                   + (" A negative value indicates a negative average inter-item "
                      "covariance \u2014 check for reverse-scored items." if a_neg else "")),
        "note": ("Alpha assumes (essentially) tau-equivalent items and "
                 "unidimensionality; it is a lower bound to true reliability."),
    })

    # ── McDonald's Omega: ONE tab, TWO clearly-headed sections ─────────────
    # omega_t (primary/required) and omega_h (advanced/optional) are
    # presented together under a single "McDonald's Omega" tab, each with
    # its own sub-heading — rather than as two separate top-level tabs,
    # which was found to be confusing in practice (they are two facets
    # of the same underlying construct-reliability question, not two
    # unrelated statistics).
    om_detail = res.get("omega_detail") or {}
    om_loadings = om_detail.get("loadings", np.array([]))
    om_has_detail = om_loadings is not None and len(om_loadings) > 0
    if om_has_detail:
        om_names = om_detail.get("item_names", [f"Item {i+1}" for i in range(len(om_loadings))])
        om_comm  = om_detail.get("communalities", np.array([]))
        om_err   = om_detail.get("unique_variances", np.array([]))
        omega_table2 = (["Item", "Factor Loading (\u03bb)", "Communality (\u03bb\u00b2)",
                          "Unique Variance (1\u2212\u03bb\u00b2)"],
                         [[om_names[j], _f(om_loadings[j]), _f(om_comm[j]), _f(om_err[j])]
                          for j in range(len(om_names))])
        omega_note = ("Estimated via iterated Principal Axis Factoring (PAF) \u2014 a "
                       "genuine common-factor extraction, not principal component "
                       "analysis (PCA). PCA does not partition variance into common "
                       "vs unique components and cannot be validly labelled McDonald's "
                       "omega; PAF iteratively estimates each item's communality (via "
                       "squared multiple correlation, then refines through the reduced "
                       "correlation matrix) until convergence \u2014 see Gorsuch (1983), "
                       "Factor Analysis, Ch. 5. The table above shows each item's "
                       "common-factor loading, communality (\u03bb\u00b2, proportion of "
                       "that item's variance explained by the common factor), and "
                       "unique variance (1\u2212\u03bb\u00b2, residual/error variance).")
    else:
        omega_table2 = None
        omega_note = ("Estimated via iterated Principal Axis Factoring (a genuine "
                       "common-factor extraction). Item-level decomposition requires "
                       "at least 3 items and a non-degenerate correlation matrix; not "
                       "available for this construct" +
                       (f" (only {k} item{'s' if k != 1 else ''})." if k < 3 else "."))

    omega_model_implied = om_detail.get("omega_model_implied", np.nan)
    omega_observed = om_detail.get("omega_observed", np.nan)
    omega_gap = (abs(omega_model_implied - omega_observed)
                 if (not (isinstance(omega_model_implied, float) and np.isnan(omega_model_implied))
                     and not (isinstance(omega_observed, float) and np.isnan(omega_observed)))
                 else np.nan)
    gap_note = (f" <b>Automatic note:</b> the two \u03c9u estimates differ by "
                f"{_f(omega_gap)}. Per Bentler (2009) and Kelley &amp; "
                f"Pornprasertmanit (2016), a larger gap here is itself diagnostic of "
                f"one-factor model misfit \u2014 i.e., the single-factor model does not "
                f"fully reproduce the observed inter-item correlations. Consider "
                f"examining the inter-item correlation matrix and item-level "
                f"loadings above for items that may be poorly explained by a single "
                f"common factor."
                if (not np.isnan(omega_gap) and omega_gap > 0.01) else
                (f" <b>Automatic note:</b> the two \u03c9u estimates are very close "
                 f"(gap = {_f(omega_gap)}), consistent with good one-factor model fit."
                 if not np.isnan(omega_gap) else ""))

    terminology_note = (" <b>On terminology:</b> both estimates above are the SAME "
                         "coefficient \u2014 McDonald's \u03c9u ('omega, unidimensional "
                         "model') \u2014 evaluated on two different, both-legitimate "
                         "variance bases, not two different-named coefficients. "
                         "Software and secondary sources are inconsistent here: R's "
                         "semTools::reliability() labels the model-implied-basis rows "
                         "\u2018omega\u2019/\u2018omega2\u2019 and the observed-variance-basis "
                         "row \u2018omega3\u2019; some tools and secondary sources instead "
                         "use the labels \u03c9t (\u2018omega total\u2019) and \u03c9u "
                         "interchangeably for these two bases, or mislabel the "
                         "observed-variance form \u2018omega hierarchical\u2019 \u2014 a "
                         "usage Flora (2020, footnote 7) explicitly flags as "
                         "incorrect. In this application, \u2018\u03c9 hierarchical "
                         "(\u03c9h)\u2019 is reserved exclusively for the bifactor-model "
                         "estimate below (Zinbarg et al., 2005; Rodriguez, Reise, "
                         "&amp; Haviland, 2016), never for either \u03c9u basis.")

    omega_mi_has_ci = "omega_model_implied_ci_lo" in om_detail
    _omega_mi_ci_txt = "\u2014"
    if omega_mi_has_ci:
        try:
            _omega_mi_ci_txt = _fmt_ci(om_detail.get("omega_model_implied_ci_lo", np.nan),
                                        om_detail.get("omega_model_implied_ci_hi", np.nan))
        except Exception:
            pass

    omega_mi_section = {
        "heading": "McDonald's \u03c9u \u2014 Model-Implied Variance (Primary)",
        "cards": [("McDonald's \u03c9u (model-implied)", _f(omega_model_implied),
                   alpha_label(omega_model_implied))],
        "table": (["Statistic", "Value"], [
            ["McDonald's \u03c9u (model-implied variance)", _f(omega_model_implied)],
        ] + ([["95% CI (bootstrap)", _omega_mi_ci_txt]] if omega_mi_has_ci else []) + [
            ["N of Items", str(k)],
            ["Extraction method", "Iterated Principal Axis Factoring"],
            ["Denominator basis", "(\u03a3\u03bb\u2c7c)\u00b2 + \u03a3(1\u2212\u03bb\u00b2\u2c7c)  (model-implied)"],
        ]),
        "table2": omega_table2,
        "table2_note": ("\u03bb\u2c7c estimated via iterated common-factor extraction "
                         "(PAF) on the inter-item correlation matrix; sign fixed so "
                         "\u03a3\u03bb\u2c7c \u2265 0.") if om_has_detail else "",
        "params": ("PRIMARY / headline reliability estimate for this module. "
                   "\u03c9u = (\u03a3\u03bb\u2c7c)\u00b2 / \u03c3\u0302X\u00b2, where \u03c3\u0302X\u00b2 here is the "
                   "variance IMPLIED BY the fitted one-factor model itself "
                   "[(\u03a3\u03bb\u2c7c)\u00b2 + \u03a3(1\u2212\u03bb\u00b2\u2c7c)]. Unlike Cronbach's \u03b1, "
                   "\u03c9u does not assume tau-equivalence (equal item loadings)."),
        "interp": (f"McDonald\u2019s \u03c9u (model-implied) = {_f(omega_model_implied)}, "
                   f"indicating <b>{alpha_label(omega_model_implied)}</b> reliability "
                   f"of the composite score according to the fitted congeneric "
                   f"(one-factor) measurement model. This is the recommended default "
                   f"\u03c9u estimate for standard internal-consistency reporting "
                   f"(Flora, 2020)."
                   + (f" Compared to Cronbach\u2019s alpha ({_f(a)}), \u03c9u is "
                      f"{'higher' if (omega_model_implied is not None and a is not None and not np.isnan(omega_model_implied) and not np.isnan(a) and omega_model_implied > a) else 'lower' if (omega_model_implied is not None and a is not None and not np.isnan(omega_model_implied) and not np.isnan(a) and omega_model_implied < a) else 'essentially equal'} "
                      f"by {_f(abs(omega_model_implied - a)) if (a is not None and not (isinstance(omega_model_implied,float) and np.isnan(omega_model_implied)) and not (isinstance(a,float) and np.isnan(a))) else '.'}"
                      + (f", suggesting departure from strict tau-equivalence (unequal "
                         f"common-factor loadings) \u2014 \u03c9u is the more appropriate "
                         f"estimate in that case." if (a is not None
                             and not (isinstance(omega_model_implied,float) and np.isnan(omega_model_implied))
                             and not (isinstance(a,float) and np.isnan(a))
                             and abs(omega_model_implied-a) > 0.02) else ".")
                      if (a is not None and not (isinstance(omega_model_implied,float) and np.isnan(omega_model_implied))
                      and not (isinstance(a,float) and np.isnan(a))) else "")),
        "note": (omega_note + " Purpose: estimates composite-score reliability with "
                 "respect to the target construct represented by the fitted "
                 "congeneric factor model, without assuming tau-equivalence. When "
                 "to use: as the PRIMARY, default \u03c9u estimate for standard "
                 "internal-consistency reporting (Flora, 2020, p. 488). Limitation: "
                 "assumes a single common factor (unidimensionality) and that the "
                 "one-factor model is not badly misspecified; for multidimensional "
                 "scales, see \u03c9 hierarchical below. References: Flora, D.B. "
                 "(2020). Advances in Methods and Practices in Psychological "
                 "Science, 3(4), 484\u2013501; McDonald, R.P. (1999). Test Theory: A "
                 "Unified Treatment. Erlbaum." + terminology_note),
    }

    omega_obs_has_ci = "omega_observed_ci_lo" in om_detail
    _omega_obs_ci_txt = "\u2014"
    if omega_obs_has_ci:
        try:
            _omega_obs_ci_txt = _fmt_ci(om_detail.get("omega_observed_ci_lo", np.nan),
                                         om_detail.get("omega_observed_ci_hi", np.nan))
        except Exception:
            pass

    omega_obs_section = {
        "heading": "McDonald's \u03c9u \u2014 Observed Variance (Secondary / Diagnostic)",
        "cards": [("McDonald's \u03c9u (observed)", _f(omega_observed),
                   alpha_label(omega_observed))],
        "table": (["Statistic", "Value"], [
            ["McDonald's \u03c9u (observed variance)", _f(omega_observed)],
        ] + ([["95% CI (bootstrap)", _omega_obs_ci_txt]] if omega_obs_has_ci else []) + [
            ["N of Items", str(k)],
            ["Denominator basis", "sample/observed Var(X)"],
            ["Gap vs. model-implied \u03c9u", _f(omega_gap)],
        ]),
        "params": ("SECONDARY / DIAGNOSTIC estimate. \u03c9u = (\u03a3\u03bb\u2c7c)\u00b2 / \u03c3\u0302X\u00b2, "
                   "where \u03c3\u0302X\u00b2 here is the ACTUAL observed sample variance of "
                   "the total score, independent of how well the one-factor model "
                   "fits the data."),
        "interp": (f"McDonald\u2019s \u03c9u (observed variance) = {_f(omega_observed)}: "
                   f"the proportion of the ACTUAL, empirically observed total-score "
                   f"variance attributable to the specified common factor."),
        "note": ("Purpose: relates model-based common-factor variance directly to "
                 "the actually observed composite-score variance, rather than to "
                 "the model's own implied variance. When to use: as a secondary "
                 "diagnostic specifically when the researcher wants this "
                 "model-to-observed comparison, or per Kelley & Pornprasertmanit "
                 "(2016), who found confidence-interval coverage for \u03c9u tends to "
                 "be better when the observed sample variance is used as the "
                 "denominator. This is NOT automatically the preferred estimate "
                 "over the model-implied form above \u2014 both are legitimate; the "
                 "gap between them (reported above) is itself informative. "
                 "Reference: Kelley, K., & Pornprasertmanit, S. (2016). "
                 "Psychological Methods, 21(1), 69\u201392." + gap_note),
    }

    omh_computable = omh.get("computable", False)
    omh_value = omh.get("omega_h", np.nan)
    if omh_computable:
        omh_general = omh.get("general_loadings", np.array([]))
        omh_specific = omh.get("specific_loadings", np.array([]))
        omh_names = omh.get("item_names", [f"Item {i+1}" for i in range(len(omh_general))])
        omh_t_bifactor = omh.get("omega_t_bifactor", np.nan)
        omh_table = (["Statistic", "Value"], [
            ["McDonald's Omega Hierarchical (\u03c9h)", _f(omh_value)],
            ["95% CI (bootstrap)", "Not computed \u2014 see note below"],
            ["N of Items", str(k)],
            ["Group factors extracted", str(omh.get("n_factors_used", "\u2014"))],
            ["\u03c9 total within this bifactor model", _f(omh_t_bifactor)],
        ])
        # Specific/group-factor loadings (Schmid-Leiman matrix S) were
        # previously computed by the engine but discarded before reaching
        # the UI. Surfaced here as additional columns alongside the
        # general-factor loading, so a user can see each item's full
        # bifactor decomposition (general + every specific factor), not
        # only its general-factor component.
        n_specific = omh_specific.shape[1] if (hasattr(omh_specific, "shape") and omh_specific.ndim == 2) else 0
        omh_table2_header = ["Item", "General-Factor Loading"] + [
            f"Specific Factor {j+1} Loading" for j in range(n_specific)]
        omh_table2_rows = []
        for i in range(len(omh_names)):
            row = [omh_names[i], _f(omh_general[i])]
            for j in range(n_specific):
                row.append(_f(omh_specific[i, j]))
            omh_table2_rows.append(row)
        omh_table2 = (omh_table2_header, omh_table2_rows)
        ratio = (omh_value/omh_t_bifactor) if (omh_t_bifactor and omh_t_bifactor > 1e-9) else np.nan
        omh_interp = (f"McDonald\u2019s \u03c9h = {_f(omh_value)}: an estimated "
                      f"{_f(ratio*100 if not np.isnan(ratio) else np.nan)}% of this "
                      f"construct's total common variance is attributable specifically "
                      f"to the single general factor shared by all items, with the "
                      f"remainder attributable to {omh.get('n_factors_used')} "
                      f"group-specific factor(s). {omh.get('reason','')}")
        omh_note = ("\u03c9h answers a DIFFERENT question than \u03c9t: \u03c9t asks "
                    "\u2018how much total variance is systematic at all\u2019 (general + "
                    "group-specific combined); \u03c9h asks \u2018how much reflects "
                    "the single construct common to every item\u2019 (general only). "
                    "\u03c9h \u2264 \u03c9t always. \u03c9h should NOT automatically "
                    "replace \u03c9t as the headline statistic \u2014 most scales are "
                    "adequately summarised by \u03c9t alone; \u03c9h is specifically "
                    "informative when a scale is knowingly multidimensional and the "
                    "question is how much reliability survives after separating out "
                    "the general construct from its sub-facets. Method: "
                    f"{omh.get('n_factors_used')}-factor oblique Principal Axis "
                    "Factoring (direct oblimin rotation) followed by the Schmid-Leiman "
                    "bifactor transformation. References: Schmid, J. & Leiman, J.M. "
                    "(1957). Psychometrika, 22(1), 53\u201361; Zinbarg, R.E., Revelle, "
                    "W., Yovel, I., & Li, W. (2005). Psychometrika, 70(1), 123\u2013133; "
                    "McDonald, R.P. (1999). Test Theory: A Unified Treatment.<br/><br/>"
                    "<b>Why no confidence interval:</b> " +
                    omh.get("omega_h_ci_reason",
                            "A bootstrap confidence interval is not computed for "
                            "omega-hierarchical, since its oblique-rotation extraction "
                            "was found to take substantial, unpredictable time for "
                            "certain data structures, independent of item count."))
        omega_h_section = {
            "heading": "McDonald's Omega Hierarchical (\u03c9h) \u2014 Advanced / Optional",
            "cards": [("McDonald's \u03c9h", _f(omh_value),
                       f"{_f(ratio*100 if not np.isnan(ratio) else np.nan)}% general-factor")],
            "table": omh_table,
            "table2": omh_table2,
            "table2_note": ("General-factor loading and specific/group-factor loading(s) "
                             "per item, after Schmid-Leiman orthogonalisation (general and "
                             "specific factors are mutually uncorrelated by construction). "
                             "A high specific-factor loading alongside a low general-factor "
                             "loading indicates that item is driven mainly by its group-"
                             "specific facet rather than the construct common to all items."),
            "params": ("ADVANCED / OPTIONAL. Requires a genuine bifactor (general + "
                       "group-factor) structure; see note for gating criteria and "
                       "why this was computable for this construct."),
            "interp": omh_interp,
            "note": omh_note,
        }
    else:
        omh_reason = omh.get("reason", "Omega hierarchical was not computed for this construct.")
        omega_h_section = {
            "heading": "McDonald's Omega Hierarchical (\u03c9h) \u2014 Advanced / Optional",
            "cards": [("McDonald's \u03c9h", "N/A", "Not computable")],
            "table": (["Statistic", "Value"],
                      [["Status", "Not computed"], ["Reason", omh_reason]]),
            "params": ("ADVANCED / OPTIONAL. \u03c9h requires a genuine bifactor "
                       "(general + \u22652 group-factor) structure: at least 6 items, "
                       "and at least 2 eigenvalues of the inter-item correlation "
                       "matrix exceeding 1.0 (Kaiser criterion)."),
            "interp": (f"<b>\u03c9 hierarchical was not computed for this construct.</b> "
                       f"{omh_reason} This is an intentional methodological safeguard, "
                       f"<b>not a computation error</b> \u2014 forcing a bifactor "
                       f"estimate on data that do not support one would produce an "
                       f"unstable, misleading number. \u03c9t (above) remains the "
                       f"valid, primary reliability estimate for this construct."),
            "note": ("\u03c9h answers a narrower question than \u03c9t (reliability "
                     "attributable to a single general factor specifically, as "
                     "distinct from all common variance combined) and is only "
                     "meaningful for scales with a genuine multi-facet structure. "
                     "References: Schmid, J. & Leiman, J.M. (1957). Psychometrika, "
                     "22(1), 53\u201361; McDonald, R.P. (1999). Test Theory: A Unified "
                     "Treatment."),
        }

    blocks.append({
        "key": "Omega", "title": "McDonald's Omega",
        "sections": [omega_mi_section, omega_obs_section, omega_h_section],
        # Top-level fields kept minimal/unused when "sections" is present;
        # required keys still populated defensively for any legacy caller.
        "cards": [], "table": None, "interp": "", "params": "", "note": "",
    })

    # ── Guttman's Lambda: dual-basis (raw covariance + standardized) ──────
    # These are NOT two different theories: same Guttman (1945) lambda
    # family, computed on two legitimate but different measurement bases.
    lam_raw = lam.get("raw", {})
    lam_std = lam.get("standardized", {})

    def _lambda_rows_with_ci(lam_basis):
        """Builds Guttman-lambda rows, adding a 95% CI column only when
        bootstrap CI data is actually present (compute_bootstrap_ci was
        requested) -- avoids a column of blank dashes when the user has
        not opted in."""
        _has_ci = any(f"lambda{i}_ci_lo" in lam_basis for i in range(1, 7))
        rows = []
        for i in range(1, 7):
            row = [f"Guttman \u03bb{i}", _f(lam_basis.get(f'lambda{i}'))]
            if _has_ci:
                _lo = lam_basis.get(f"lambda{i}_ci_lo", np.nan)
                _hi = lam_basis.get(f"lambda{i}_ci_hi", np.nan)
                _method = lam_basis.get(f"lambda{i}_ci_method", "")
                try:
                    _ci_txt = _fmt_ci(_lo, _hi)
                except Exception:
                    _ci_txt = "\u2014"
                row.append(_ci_txt)
            rows.append(row)
        return rows, _has_ci

    lam_raw_rows, _lam_raw_has_ci = _lambda_rows_with_ci(lam_raw)
    lam_std_rows, _lam_std_has_ci = _lambda_rows_with_ci(lam_std)
    l2r, l3r = lam_raw.get("lambda2"), lam_raw.get("lambda3")
    l4_method = lam_raw.get("lambda4_method", "n/a")
    l4_n_splits = lam_raw.get("lambda4_n_splits", 0)
    l4_method_txt = ({
        "exhaustive": f"true maximum \u2014 exhaustively searched across all "
                      f"{l4_n_splits} distinct non-trivial splits",
        "sampled": (f"best value found among {l4_n_splits:,} randomly sampled "
                    f"non-trivial splits (item count exceeds the exhaustive-search "
                    f"threshold of 16 items) \u2014 not the guaranteed true maximum, "
                    f"but a close estimate; methodology follows Revelle, W. psych "
                    f"package's splitHalf() function, which switches from exhaustive "
                    f"search to random sampling at the same item-count threshold"),
        "n/a": "not computed",
    }).get(l4_method, l4_method)

    # Automatic recommendation: raw vs standardized basis divergence
    # signals item-scale heterogeneity (large SD differences across items).
    l3_std_val = lam_std.get("lambda3")
    basis_gap = (abs(l3r - l3_std_val)
                 if (l3r is not None and l3_std_val is not None
                     and not (isinstance(l3r, float) and np.isnan(l3r))
                     and not (isinstance(l3_std_val, float) and np.isnan(l3_std_val)))
                 else np.nan)
    basis_recommendation = (
        f" <b>Automatic note:</b> raw-basis \u03bb\u2083 ({_f(l3r)}) and "
        f"standardized-basis \u03bb\u2083 ({_f(l3_std_val)}) differ by {_f(basis_gap)} "
        f"\u2014 a gap this size typically indicates items have substantially "
        f"different standard deviations (scale heterogeneity). Consider whether "
        f"raw or standardized combination is more substantively appropriate for "
        f"this scale."
        if (not np.isnan(basis_gap) and basis_gap > 0.02) else "")

    # Automatic recommendation: dimensionality signal via Kaiser eigenvalue
    # criterion (validated via Monte Carlo simulation to correctly identify
    # unidimensional vs multi-factor data in ~80-100% of cases — a simpler
    # "lambda4 minus mean of other lambdas" gap heuristic was tested first
    # and found NOT to discriminate reliably, so it was not used).
    n_eig = lam_raw.get("n_eigenvalues_above_1")
    dimensionality_note = ""
    if n_eig is not None and n_eig >= 2:
        dimensionality_note = (
            f" <b>Automatic note:</b> {n_eig} eigenvalues of the inter-item "
            f"correlation matrix exceed 1.0 (Kaiser, 1960 criterion), suggesting "
            f"this construct may not be strictly unidimensional. If McDonald's "
            f"\u03c9 hierarchical is reported above as computable, its value "
            f"reflects this structure directly; otherwise, consider a dedicated "
            f"exploratory factor analysis to examine potential sub-facets before "
            f"treating this scale as measuring a single construct.")

    lambda_raw_section = {
        "heading": "Raw Covariance Basis",
        "cards": [("\u03bb2 (raw)", _f(l2r), alpha_label(l2r)),
                  ("\u03bb3 = \u03b1 (raw)", _f(l3r), alpha_label(l3r))],
        "table": (["Coefficient", "Value", "95% CI (bootstrap)"] if _lam_raw_has_ci
                  else ["Coefficient", "Value"], lam_raw_rows),
        "params": ("Computed from the item COVARIANCE matrix (original item scales "
                   "preserved). \u03bb\u2083 equals raw-score Cronbach's alpha. \u03bb\u2084 "
                   f"method: {l4_method_txt}."),
        "interp": (f"Guttman\u2019s \u03bb\u2082 (raw) = {_f(l2r)} (<b>{alpha_label(l2r)}</b>); "
                   f"\u03bb\u2083 (raw) = {_f(l3r)}, equal to Cronbach\u2019s alpha computed on "
                   f"the original item scales." + basis_recommendation + dimensionality_note),
        "note": ("\u03bb\u2081: weakest lower bound (Guttman, 1945, Eq. 1). \u03bb\u2082: "
                 "improves on \u03bb\u2081 using covariance structure (Eq. 2). \u03bb\u2083: "
                 "= Cronbach's \u03b1 (Eq. 3). \u03bb\u2084: true maximum split-half "
                 "reliability via exhaustive search over all non-trivial splits "
                 "(Eq. 4) \u2014 the tightest split-half-type bound. \u03bb\u2085: uses the "
                 "largest column-sum of squared covariances (Eq. 5). \u03bb\u2086: uses "
                 "squared multiple correlations (SMC) to isolate unique variance "
                 "(Eq. 6); requires an invertible correlation matrix, may be "
                 "undefined for collinear items. Proven ordering \u03bb\u2081 \u2264 \u03bb\u2083 "
                 "\u2264 \u03bb\u2082 holds when \u03bb\u2081 \u2265 0 (Guttman, 1945); no universal "
                 "ordering is proven among \u03bb\u2084, \u03bb\u2085, \u03bb\u2086 \u2014 their relative "
                 "size depends on the specific covariance structure. Reference: "
                 "Guttman, L. (1945). A basis for analyzing test-retest reliability. "
                 "Psychometrika, 10(4), 255\u2013282."
                 + (" 95% confidence intervals are percentile-bootstrap, following "
                    "the method recommended for reliability coefficients by Kelley, "
                    "K., &amp; Pornprasertmanit, S. (2016). Psychological Methods, "
                    "21(1), 69\u201392." if _lam_raw_has_ci else "")),
    }

    l2s = lam_std.get("lambda2")
    lambda_std_section = {
        "heading": "Standardized Correlation Basis",
        "cards": [("\u03bb2 (standardized)", _f(l2s), alpha_label(l2s)),
                  ("\u03bb3 = \u03b1 (standardized)", _f(l3_std_val), alpha_label(l3_std_val))],
        "table": (["Coefficient", "Value", "95% CI (bootstrap)"] if _lam_std_has_ci
                  else ["Coefficient", "Value"], lam_std_rows),
        "params": ("Same Guttman (1945) lambda family as above (see Raw Covariance "
                   "Basis section), computed on the item CORRELATION matrix instead "
                   "(each item implicitly rescaled to unit variance before "
                   "combination). \u03bb\u2083 here equals STANDARDIZED Cronbach's alpha."),
        "interp": (f"Guttman\u2019s \u03bb\u2082 (standardized) = {_f(l2s)} "
                   f"(<b>{alpha_label(l2s)}</b>); \u03bb\u2083 (standardized) = "
                   f"{_f(l3_std_val)}."),
        "note": ("Use this basis when items are measured on different scales and "
                 "should be equalised (unit variance) before combination. The raw "
                 "basis (above) is preferable when the original item-scale "
                 "differences are themselves substantively meaningful. Neither "
                 "basis is universally \u2018more correct\u2019 \u2014 the appropriate "
                 "choice depends on measurement context, not on which number is "
                 "larger. Reference: Guttman, L. (1945). Psychometrika, 10(4), "
                 "255\u2013282."
                 + (" 95% confidence intervals are percentile-bootstrap (Kelley &amp; "
                    "Pornprasertmanit, 2016)." if _lam_std_has_ci else "")),
    }

    blocks.append({
        "key": "Lambda", "title": "Guttman's Lambda",
        "sections": [lambda_raw_section, lambda_std_section],
        "cards": [], "table": None, "interp": "", "params": "", "note": "",
    })

    if sh.get("error"):
        # k < 4: Spearman-Brown requires equal-length halves, which is not
        # meaningful below 4 items (a 3-item split gives 1 item vs 2 items).
        # Render an explicit, visible rejection message instead of silently
        # falling through to a table of "." placeholders with no explanation
        # — an empty-looking table with no message reads as a software error
        # to users, when it is in fact a deliberate, documented minimum.
        _n_items_sh = sh.get("n_items", k)
        blocks.append({
            "key": "Split-half", "title": "Split-Half",
            "cards": [("Split-Half", "N/A", "Requires \u2265 4 items")],
            "table": (["Statistic", "Value"],
                      [["Status", "Not computed"],
                       ["Reason", sh.get("message", "Split-half analysis requires at least 4 items.")],
                       ["N of Items (this construct)", str(_n_items_sh)],
                       ["Minimum Required", "4"]]),
            "params": ("Spearman-Brown assumes equal-length, equally-reliable "
                       "halves; with fewer than 4 items a first/second split "
                       "cannot produce two halves of comparable length "
                       "(e.g. 3 items \u2192 1 item vs 2 items), so the "
                       "correction is not meaningful."),
            "interp": (f"<b>Split-half reliability was not computed for this "
                       f"construct.</b> It contains {_n_items_sh} item"
                       f"{'s' if _n_items_sh != 1 else ''}, below the minimum "
                       f"of 4 required for a meaningful equal-length split. "
                       f"This is an intentional methodological safeguard, "
                       f"<b>not a computation error</b> \u2014 no split-half "
                       f"value can be validly reported for scales this short."),
            "note": ("To obtain split-half reliability for this construct, "
                     "include at least 4 items. Cronbach's alpha, McDonald's "
                     "omega, and Guttman's lambda coefficients remain valid "
                     "and are reported above regardless of item count."),
        })
    else:
        sb = sh.get("spearman_brown"); gut = sh.get("guttman"); rh = sh.get("r_halves")
        a1 = sh.get("alpha_part1"); a2 = sh.get("alpha_part2")
        n1 = sh.get("n_part1", ""); n2 = sh.get("n_part2", "")
        ntot = (int(n1) + int(n2)) if (str(n1).isdigit() and str(n2).isdigit()) else ""

        # Supplementary variants (additive — the first-second half result
        # above remains PRIMARY and unchanged; these are shown alongside,
        # not in place of it, per spec).
        oe = sh.get("odd_even", {})
        mx = sh.get("maximum", {})
        fsh = sh.get("first_second", {})
        mx_method = mx.get("method", "n/a")
        mx_n = mx.get("n_splits_evaluated", 0)
        mx_method_label = {
            "exhaustive": f"exhaustive, n={mx_n:,}",
            "sampled": f"sampled, n={mx_n:,}",
            "n/a": "n/a",
        }.get(mx_method, mx_method)
        _sh_has_ci = "spearman_brown_ci_lo" in fsh
        variants_table2 = None
        variants_note = ""
        if oe or mx:
            if _sh_has_ci:
                def _ci_cell(d, key):
                    try:
                        return _fmt_ci(d.get(f"{key}_ci_lo", np.nan), d.get(f"{key}_ci_hi", np.nan))
                    except Exception:
                        return "\u2014"
                variants_table2 = (
                    ["Split Method", "r", "Spearman-Brown", "SB 95% CI",
                     "Guttman SH", "Guttman 95% CI"],
                    [["First-Second Half (primary, above)", _f(rh), _f(sb),
                      _ci_cell(fsh, "spearman_brown"), _f(gut), _ci_cell(fsh, "guttman")],
                     ["Odd-Even Split", _f(oe.get("r_halves")), _f(oe.get("spearman_brown")),
                      _ci_cell(oe, "spearman_brown"), _f(oe.get("guttman")), _ci_cell(oe, "guttman")],
                     [f"Maximum Split-Half ({mx_method_label})",
                      _f(mx.get("r_halves")), _f(mx.get("spearman_brown")),
                      _ci_cell(mx, "spearman_brown"), _f(mx.get("guttman")),
                      "\u2014 (Guttman-SH not separately optimized for this variant)"]])
            else:
                variants_table2 = (
                    ["Split Method", "r (between halves)", "Spearman-Brown", "Guttman Split-Half"],
                    [["First-Second Half (primary, above)", _f(rh), _f(sb), _f(gut)],
                     ["Odd-Even Split", _f(oe.get("r_halves")), _f(oe.get("spearman_brown")),
                      _f(oe.get("guttman"))],
                     [f"Maximum Split-Half ({mx_method_label})",
                      _f(mx.get("r_halves")), _f(mx.get("spearman_brown")), _f(mx.get("guttman"))]])
            variants_note = (
                "Odd-even split (items at odd vs even positions) is the classical "
                "alternative to the first-second-half convention shown above. "
                "Maximum split-half is found via its own search (exhaustive for "
                "\u226416 items; for larger scales, "
                f"{mx_n:,} randomly sampled non-trivial partitions, following the "
                "same threshold and methodology as R's psych::splitHalf() -- "
                "Revelle, W. psych package documentation and NEWS) specifically "
                "maximising the Spearman-Brown coefficient. "
                "<b>Important:</b> this search target is deliberately independent "
                "from Guttman's \u03bb\u2084 (see Guttman's Lambda section), which "
                "maximises a different criterion (the variance-ratio formula). "
                "When item variances are heterogeneous, the split that maximises "
                "\u03bb\u2084 is not always the same split that maximises "
                "Spearman-Brown \u2014 the two are related but distinct "
                "optimisation targets, so this module reports each against its "
                "own correct maximum rather than reusing one for the other. "
                "All three split-half variants are legitimate; they will "
                "generally differ, and no single one is the definitive "
                "\u2018true\u2019 split-half estimate. The first-second-half result "
                "above remains the primary, headline split-half statistic for "
                "this module."
            )

        blocks.append({
            "key": "Split-half", "title": "Split-Half",
            "cards": [("Spearman-Brown", _f(sb), alpha_label(sb)),
                      ("Guttman split-half", _f(gut), alpha_label(gut))],
            "table": (["Statistic", "Part 1", "Part 2", "Overall"],
                      [["Cronbach's Alpha", _f(a1), _f(a2), ""],
                       ["N of Items", str(n1), str(n2), str(ntot)],
                       ["Correlation Between Forms", "", "", _f(rh)],
                       ["Spearman-Brown Coefficient", "", "", _f(sb)],
                       ["Guttman Split-Half Coefficient", "", "", _f(gut)]]),
            "table2": variants_table2,
            "table2_note": variants_note,
            "params": ("PRIMARY result: items split into first vs second half. "
                       "Spearman-Brown assumes equal-length, equally-reliable "
                       "halves. Supplementary split methods (odd-even, maximum) "
                       "are tabulated below for comparison."),
            "interp": (f"The Spearman-Brown split-half coefficient = {_f(sb)} "
                       f"(<b>{alpha_label(sb)}</b>); the Guttman split-half = {_f(gut)}. "
                       f"The two halves correlate r = {_f(rh)}."),
            "note": ("A low or negative split-half value alongside an adequate omega "
                     "often signals a reverse-scored item or multidimensionality. "
                     "References: Spearman, C. (1910); Brown, W. (1910); Guttman, L. "
                     "(1945). Psychometrika, 10(4), 255\u2013282."),
        })

    if dich:
        _kr_oor  = (kr is not None and not (isinstance(kr, float) and np.isnan(kr))
                    and kr > 1)
        _item_p  = res.get("item_p",  np.array([]))
        _item_pq = res.get("item_pq", np.array([]))
        _item_names = res.get("item_names",
                              [f"Item {i+1}" for i in range(len(_item_p))])
        # Item-level diagnostic table
        _item_rows = [["Item", "p (proportion correct)",
                       "p\u00d7q (item variance)", "Diagnosis"]]
        for _i, (_pv, _pqv) in enumerate(zip(_item_p, _item_pq)):
            _nm = _item_names[_i] if _i < len(_item_names) else f"Item {_i+1}"
            if float(_pv) < 0.1 or float(_pv) > 0.9:
                _dx = "\u26a0\ufe0f Near-zero variance (p extreme)"
            elif float(_pqv) < 0.05:
                _dx = "\u26a0\ufe0f Very low item variance"
            else:
                _dx = "\u2713 OK"
            _item_rows.append([_nm, _f(float(_pv)), _f(float(_pqv)), _dx])
        _has_diag = len(_item_rows) > 1  # more than just header

        # KR-20 table: two rows when out-of-range (original + capped for appendix)
        _kr_capped = min(float(kr), 1.0) if (kr is not None and not
                        (isinstance(kr, float) and np.isnan(kr))) else np.nan
        _kr21_val = res.get("kr21", np.nan)
        _astd_val = res.get("alpha_std", np.nan)
        _sem_val = res.get("sem", np.nan)
        _sem_basis = res.get("sem_basis", "KR-20")
        _sstats = res.get("scale_stats", {})

        # ── Section 1: Reliability Coefficients ─────────────────────────
        if _kr_oor:
            _rel_rows = [["KR-20 (calculated)", _f(kr)],
                         ["KR-20 (adjusted for report, capped at 1.000)", _f(_kr_capped)],
                         ["KR-21", _f(_kr21_val)],
                         ["Standardized Alpha (correlation-basis)", _f(_astd_val)],
                         ["N of Items", str(k)]]
            _rel_note = (
                # CORRECTION (2026-10-21): this note previously claimed the
                # out-of-range value is a "known, well-documented
                # mathematical property" that "occurs when one or more
                # items have near-zero variance." That causal claim was
                # independently re-verified and found UNSUPPORTED: for any
                # COMPLETE (listwise, no missing data) sample of correctly-
                # coded 0/1 items, the sample covariance matrix is
                # guaranteed positive semi-definite, which mathematically
                # guarantees KR-20 (and Cronbach's alpha) cannot exceed
                # 1.0 -- confirmed by a 200,000-trial Monte Carlo search
                # (including deliberately near-zero-variance items and
                # maximal positive inter-item correlation) that never
                # produced a value exceeding 1.0 by more than floating-
                # point noise. The genuine, reachable cause found in this
                # application was a different, unrelated bug: reverse-
                # scoring a dichotomous item with the sidebar's default
                # non-0/1 theoretical bounds (scale_min=1, scale_max=5)
                # produced non-Bernoulli-coded values that were still
                # (wrongly) treated as "dichotomous," corrupting the KR-20
                # computation -- now fixed at the source (see
                # modules.reliability._is_dichotomous). This branch remains
                # purely as a defensive floating-point-precision safety
                # net; a genuine out-of-range value appearing now most
                # likely indicates near-exact floating-point noise around
                # 1.0 rather than a meaningful "near-zero item variance"
                # phenomenon.
                "<b>\u26a0\ufe0f KR-20 = " + _f(kr) + " \u2014 out-of-range (value &gt; 1).</b> "
                "This is <b>not a calculation error</b>. For complete, "
                "correctly-coded 0/1 data, KR-20 is mathematically bounded "
                "at \u2264 1.000, so a value this far above 1 generally "
                "signals floating-point noise at the boundary rather than a "
                "substantively meaningful result. Use the adjusted value "
                "(1.000) in report appendices, with a methodological note. "
                "See Item-Level Diagnostics below for the underlying items.")
            _card_lbl = "\u26a0\ufe0f Out-of-range (> 1)"
        else:
            _rel_rows = [["KR-20", _f(kr)],
                         ["KR-21", _f(_kr21_val)],
                         ["Standardized Alpha (correlation-basis)", _f(_astd_val)],
                         ["N of Items", str(k)]]
            _rel_note = ""
            _card_lbl = alpha_label(kr)

        _gap_kr20_kr21 = (abs(kr - _kr21_val)
                          if (not (isinstance(kr, float) and np.isnan(kr))
                              and not (isinstance(_kr21_val, float) and np.isnan(_kr21_val)))
                          else np.nan)
        reliability_section = {
            "heading": "Reliability Coefficients",
            "cards": [("KR-20", _f(_kr_capped if _kr_oor else kr), _card_lbl),
                      ("KR-21", _f(_kr21_val), alpha_label(_kr21_val)),
                      ("Standardized \u03b1", _f(_astd_val), alpha_label(_astd_val))],
            "table": (["Statistic", "Value"], _rel_rows),
            "params": (
                "KR-20 applies to dichotomous (0/1) items and equals "
                "Cronbach\u2019s alpha exactly for binary data (Kuder & "
                "Richardson, 1937) -- and, for complete (listwise) data, is "
                "itself mathematically bounded at \u2264 1.000, so a reported "
                "value above that reflects floating-point noise, not a "
                "meaningful result. KR-21 is a simplified, more conservative "
                "form assuming equal item difficulty; KR-21 \u2264 KR-20 "
                "asymptotically (as N\u2192\u221e) and in typical samples, but a "
                "small finite-sample violation is mathematically possible "
                "(confirmed by direct construction: e.g. N=86 can produce a "
                "violation on the order of 0.002\u20130.004) since the two "
                "formulas are not computed on an identical basis for finite "
                "N -- this is not a calculation error in either coefficient. "
                "Standardized Alpha uses the correlation (not covariance) "
                "basis \u2014 for genuinely dichotomous 0/1 items this will "
                "generally differ from KR-20/raw-\u03b1 since item variances "
                "p\u00b7q differ across items even when all are dichotomous."),
            "interp": ((f"KR-20 = {_f(kr)}, indicating <b>{alpha_label(kr)}</b> "
                       f"internal consistency for these dichotomous items."
                       + (f" KR-21 = {_f(_kr21_val)} (gap to KR-20 = {_f(_gap_kr20_kr21)}), "
                          f"the more conservative equal-difficulty approximation."
                          if not np.isnan(_gap_kr20_kr21) else "")) if not _kr_oor else
                       f"<b>\u26a0\ufe0f KR-20 = {_f(kr)} \u2014 out-of-range (value &gt; 1).</b> "
                       f"This is <b>not a calculation error</b>, but for complete, "
                       f"correctly-coded 0/1 data KR-20 is mathematically bounded "
                       f"at \u2264 1.000 (see Decision Parameters below), so this "
                       f"most likely reflects floating-point noise at the "
                       f"boundary. Use the <b>adjusted value "
                       f"({_f(_kr_capped)})</b> in your report, indicating "
                       f"<b>{alpha_label(_kr_capped)}</b> internal consistency."
                       + (f" KR-21 = {_f(_kr21_val)}, the more conservative "
                          f"equal-difficulty approximation."
                          if not np.isnan(_kr21_val) and not (isinstance(_kr21_val,float) and np.isnan(_kr21_val)) else "")),
            "note": ("Reported because all items in this construct are "
                     "dichotomous. References: Kuder, G.F., & Richardson, "
                     "M.W. (1937). Psychometrika, 2(3), 151\u2013160." + _rel_note),
        }

        # ── Section 2: Scale-Level Statistics ────────────────────────────
        scale_section = {
            "heading": "Scale-Level Statistics",
            "cards": [("Mean (total score)", _f(_sstats.get("mean", np.nan)), ""),
                      ("SD (total score)", _f(_sstats.get("sd", np.nan)), ""),
                      ("SEM", _f(_sem_val), f"basis: {_sem_basis}")],
            "table": (["Statistic", "Value"], [
                ["Mean (total score)", _f(_sstats.get("mean", np.nan))],
                ["Median (total score)", _f(_sstats.get("median", np.nan))],
                ["SD (total score)", _f(_sstats.get("sd", np.nan))],
                ["Variance (total score)", _f(_sstats.get("variance", np.nan))],
                ["Min / Max", f"{_f(_sstats.get('min', np.nan))} / {_f(_sstats.get('max', np.nan))}"],
                ["Standard Error of Measurement (SEM)", _f(_sem_val)],
            ]),
            "params": (f"SEM = SD_total \u00d7 \u221a(1 \u2212 reliability), using "
                      f"{_sem_basis} as the reliability estimate for this "
                      f"dichotomous-item construct (Lord & Novick, 1968)."),
            "interp": (f"The observed total-score mean is {_f(_sstats.get('mean', np.nan))} "
                      f"(SD = {_f(_sstats.get('sd', np.nan))}). The Standard Error of "
                      f"Measurement is {_f(_sem_val)}, meaning an individual "
                      f"respondent\u2019s true score is expected to fall within "
                      f"approximately \u00b1{_f(1.96*_sem_val) if not np.isnan(_sem_val) else '\u2014'} "
                      f"of their observed score, 95% of the time, under classical "
                      f"test theory."),
            "note": ("SEM is useful for constructing approximate confidence "
                     "bands around an individual respondent\u2019s observed score "
                     "\u2014 distinct from the reliability coefficients above, "
                     "which describe the scale\u2019s overall consistency rather "
                     "than uncertainty around any one person\u2019s score. "
                     "Reference: Lord, F.M., & Novick, M.R. (1968). Statistical "
                     "Theories of Mental Test Scores. Addison-Wesley."),
        }

        # ── Section 3: Item-Level Diagnostics (consolidated) ─────────────
        _istats_df = res.get("item_stats")
        _item_total_r_map = {}
        _kr20_del_map = {}
        if _istats_df is not None and not _istats_df.empty:
            for _, _row in _istats_df.iterrows():
                _item_total_r_map[_row["Item"]] = _row.get("Corrected Item-Total Correlation", np.nan)
                _kr20_del_map[_row["Item"]] = _row.get("Cronbach's Alpha if Item Deleted", np.nan)

        _pq_raw_arr = res.get("item_pq_raw", np.array([]))
        _pq_sample_arr = res.get("item_pq_sample", np.array([]))
        if _istats_df is not None and not _istats_df.empty and "Item" in _istats_df.columns:
            _item_names_kr = list(_istats_df["Item"])
        else:
            _item_names_kr = [f"Item {i+1}" for i in range(len(_item_p))]

        _diag_rows = [["Item", "p (proportion)", "p\u00d7q (raw)",
                       "p\u00d7q (sample-corrected)", "Item-Total r",
                       "KR-20 if Deleted", "Diagnosis"]]
        for _i in range(len(_item_p)):
            _nm = _item_names_kr[_i] if _i < len(_item_names_kr) else f"Item {_i+1}"
            _pv = float(_item_p[_i])
            _pqr = float(_pq_raw_arr[_i]) if _i < len(_pq_raw_arr) else np.nan
            _pqs = float(_pq_sample_arr[_i]) if _i < len(_pq_sample_arr) else np.nan
            _itr = _item_total_r_map.get(_nm, np.nan)
            _kdel = _kr20_del_map.get(_nm, np.nan)
            if _pv < 0.1 or _pv > 0.9:
                _dx = "\u26a0\ufe0f Near-zero variance (p extreme)"
            elif _pqs < 0.05:
                _dx = "\u26a0\ufe0f Very low item variance"
            elif not np.isnan(_itr) and _itr < 0.1:
                _dx = "\u26a0\ufe0f Weak item-total correlation"
            else:
                _dx = "\u2713 OK"
            _diag_rows.append([_nm, _f(_pv), _f(_pqr), _f(_pqs),
                               _f(_itr), _f(_kdel), _dx])
        # Summary rows: Sum and Mean item difficulty
        _diag_rows.append(["Sum", "\u2014", _f(np.sum(_pq_raw_arr)) if len(_pq_raw_arr) else "\u2014",
                           _f(np.sum(_pq_sample_arr)) if len(_pq_sample_arr) else "\u2014",
                           "\u2014", "\u2014", ""])
        _diag_rows.append(["Mean item difficulty", _f(np.mean(_item_p)) if len(_item_p) else "\u2014",
                           "\u2014", "\u2014", "\u2014", "\u2014", ""])

        diagnostics_section = {
            "heading": "Item-Level Diagnostics",
            "table": (_diag_rows[0], _diag_rows[1:]),
            "table_left": {0, 6},
            "params": ("p\u00d7q (raw) is the population-style item variance "
                      "p(1\u2212p) \u2014 the conventional textbook quantity. "
                      "p\u00d7q (sample-corrected) = p(1\u2212p)\u00b7n/(n\u22121) is the "
                      "value that actually enters the KR-20 formula\u2019s "
                      "numerator sum. Both are shown because the two "
                      "conventions differ systematically by exactly the factor "
                      "n/(n\u22121), and different reference tools may display "
                      "either one \u2014 showing both allows direct reconciliation "
                      "without implying either alone is incorrect."),
            "interp": ("Items flagged \u26a0\ufe0f above warrant review: extreme p "
                      "(&lt;.10 or &gt;.90) indicates near-unanimous responses "
                      "and contributes little discriminating variance; weak "
                      "item-total correlation (&lt;.10) suggests the item may "
                      "not measure the same construct as the rest of the scale. "
                      "KR-20-if-deleted values above the scale\u2019s overall "
                      "KR-20 indicate removing that item would improve "
                      "reliability."),
            "note": ("KR-20-if-Deleted is numerically identical to Cronbach\u2019s "
                     "alpha-if-deleted for these dichotomous items (the two "
                     "coefficients are algebraically identical; see Kuder & "
                     "Richardson, 1937), recomputed here on the (k\u22121)-item "
                     "reduced scale for each item in turn."),
        }

        blocks.append({
            "key":        "KR-20",
            "title":      "Kuder-Richardson 20",
            "sections": [reliability_section, scale_section, diagnostics_section],
            "cards": [], "table": None, "interp": "", "params": "", "note": "",
        })
    else:
        blocks.append({
            "key": "KR-20", "title": "Kuder-Richardson 20",
            "cards": [("KR-20", "\u2014", "not applicable")],
            "table": None,
            "params": "KR-20 is defined only for dichotomous (0/1) items.",
            "interp": ("KR-20 is not applicable here because the items are not "
                       "dichotomous. Use Cronbach\u2019s alpha instead."),
            "note": "",
        })

    icc = res.get("icc")
    if icc is None:
        blocks.append({
            "key": "ICC", "title": "Intraclass Correlation Coefficient",
            "cards": [("ICC", "\u2014", "not computable")],
            "table": None,
            "params": ("ICC requires variance in both subjects and items. "
                       "It cannot be computed when all items or all persons "
                       "have identical scores (zero variance)."),
            "interp": ("\u2139\ufe0f ICC could not be computed for this construct. "
                       "This typically occurs when item scores are extremely "
                       "homogeneous \u2014 e.g., all respondents answered "
                       "identically, or all items have the same value for every "
                       "respondent. Ensure there is sufficient variability in the "
                       "data before re-running."),
            "note": ("ICC is based on a one-way or two-way random/mixed effects "
                     "model. Computed via pingouin; requires non-degenerate data "
                     "(both between-subject and within-subject variance > 0)."),
        })
    if icc is not None and not icc.empty:
        cols = list(icc.columns)
        def _icc_row_cell(v):
            # pingouin's CI95% column returns a numpy.ndarray, NOT a
            # Python list -- confirmed via direct inspection (type(v) is
            # numpy.ndarray). An isinstance(v, list) check alone misses
            # this, silently falling through to str(v) and producing
            # numpy's raw repr (e.g. "[-0.05  0.03]", double-spaced, no
            # comma, 3dp) instead of the intended 5dp bracket format.
            if isinstance(v, (list, np.ndarray)) and len(v) == 2:
                return _fmt_ci(v[0], v[1])
            if isinstance(v, (list, np.ndarray)):
                return str(v)
            if isinstance(v, (int, float, np.floating)):
                return "\u2014" if np.isnan(float(v)) else _f(v)
            return str(v)
        rows = [[_icc_row_cell(v) for v in icc.loc[i]] for i in icc.index]

        # ── BUGFIX (fatal, confirmed via audit): the previous headline
        # selection used icc["Type"].str.contains("k"), which matches
        # ICC(1,k), ICC(A,k), AND ICC(C,k) (all three contain the letter
        # "k"), then took .iloc[0] — the FIRST matching row in pingouin's
        # fixed output order, which is ICC(1,k) (one-way random). This
        # produced a headline "internal consistency" metric drawn from a
        # ONE-WAY model, which assumes each subject is rated by a
        # DIFFERENT randomly-selected set of items — the opposite of a
        # questionnaire design, where every respondent answers the exact
        # same fixed set of items (a two-way, fixed-items design).
        # Cronbach's alpha is mathematically identical to ICC(C,k)
        # [two-way mixed effects, CONSISTENCY, average measures] — NOT
        # to ICC(1,k) or ICC(A,k). Verified with a realistic test case
        # (heterogeneous item means): the old buggy selection produced
        # ICC(1,k) = -0.110 ("Poor"), while the correct ICC(C,k) = 0.773
        # ("Good") — a 0.883 gap capable of reversing a published
        # reliability conclusion. Fixed to an EXPLICIT key lookup,
        # identical in style to how the Inter-Rater Agreement context
        # already correctly selects icc_2_1 by exact key (never had this
        # bug) rather than by fuzzy string matching.
        ck_row = icc[icc["Type"] == "ICC(C,k)"]
        head = float(ck_row["ICC"].iloc[0]) if not ck_row.empty else np.nan
        head_ci_row = ck_row
        head_p = float(head_ci_row["pval"].iloc[0]) if (not head_ci_row.empty and "pval" in head_ci_row.columns) else np.nan

        a_val = res.get("alpha", np.nan)
        n_val = res.get("n", "\u2014"); k_val = res.get("k", "\u2014")
        avg_r = res.get("avg_inter_item_r", np.nan)
        alpha_gap = (abs(head - a_val) if (not (isinstance(head, float) and np.isnan(head))
                     and not (isinstance(a_val, float) and np.isnan(a_val))) else np.nan)

        icc_deleted_df = res.get("icc_if_deleted")
        has_deleted = icc_deleted_df is not None and not icc_deleted_df.empty
        deleted_table = None
        if has_deleted:
            deleted_table = (["Item", "ICC(C,k) if Item Deleted"],
                              [[r["Item"], _f(r["ICC(C,k) if Item Deleted"])]
                               for _, r in icc_deleted_df.iterrows()])

        primary_section = {
            "heading": "Primary Result \u2014 ICC(C,k)",
            "cards": [("ICC(C,k) \u2014 Primary", _f(head), _icc_label(head)),
                      ("Cronbach's \u03b1 (cross-check)", _f(a_val), alpha_label(a_val)),
                      ("Mean Inter-Item r", _f(avg_r), "")],
            "table": (["Statistic", "Value"], [
                ["N (respondents)", str(n_val)],
                ["k (items)", str(k_val)],
                ["ICC(C,k)", _f(head)],
                ["p (F-test)", _fmt_p_dash(head_p)],
                ["Cronbach's \u03b1 (same scale, covariance-ratio estimator)", _f(a_val)],
                ["|ICC(C,k) \u2212 \u03b1|", _f(alpha_gap)],
                ["Mean Inter-Item Correlation", _f(avg_r)],
            ]),
            "table2": deleted_table,
            "table2_note": ("Each row: ICC(C,k) recomputed with that item removed \u2014 "
                             "the ANOVA-based analogue of \u2018Cronbach\u2019s alpha if item "
                             "deleted\u2019. A value ABOVE the full-scale ICC(C,k) "
                             f"({_f(head)}) indicates removing that item would improve "
                             "consistency; requires \u22654 items."
                             if has_deleted else ""),
            "params": ("MODEL: two-way MIXED effects, CONSISTENCY, average measures "
                       "\u2014 the model that matches a fixed-items questionnaire design, "
                       "where every respondent answers the identical set of items (not "
                       "a randomly-varying set per respondent). This is the ICC form "
                       "that is mathematically identical to Cronbach's alpha (both "
                       "equal (MSB\u2212MSE)/MSB from the same two-way ANOVA "
                       "decomposition, without the between-item/column mean square "
                       "in the formula) \u2014 the near-zero gap above is the expected, "
                       "verifiable signature of that identity, not a coincidence. "
                       "Koo & Li (2016): <0.50 poor; 0.50\u20130.75 moderate; "
                       "0.75\u20130.90 good; >0.90 excellent."),
            "interp": (f"ICC(C,k) = {_f(head)}, indicating <b>{_icc_label(head)}</b> "
                       f"internal consistency \u2014 mathematically the same quantity as "
                       f"Cronbach's \u03b1 ({_f(a_val)}), confirmed above by their small "
                       f"numerical gap ({_f(alpha_gap)}), computed here via an explicit "
                       f"two-way ANOVA decomposition (Shrout & Fleiss, 1979) rather "
                       f"than the covariance-ratio formula. Use average measures "
                       f"(\u2018k\u2019) \u2014 not single measures (\u20181\u2019) \u2014 as the "
                       f"mandatory standard for overall scale reliability: single-"
                       f"measures forms evaluate the reliability of one individual "
                       f"item alone, not the combined scale."),
            "note": ("WHEN TO USE THIS (IC) CONTEXT vs the Inter-Rater Agreement "
                     "module's ICC: use this context when your question is about "
                     "item-level internal consistency of a scale; use the Inter-Rater "
                     "Agreement module when your question is about agreement between "
                     "distinct raters assessing the same units. Both contexts share "
                     "identical underlying mathematics (Shrout & Fleiss, 1979) but "
                     "answer different substantive questions, and by design select "
                     "DIFFERENT primary forms: this context selects ICC(C,k) "
                     "(consistency, matching Cronbach's alpha); the Inter-Rater "
                     "Agreement context selects ICC(A,1)/ICC(2,1) (absolute "
                     "agreement, penalising systematic rater bias) \u2014 see the "
                     "reference guide below for the full six-form breakdown. "
                     "Reference: Shrout, P.E., & Fleiss, J.L. (1979). Psychological "
                     "Bulletin, 86(2), 420\u2013428."),
        }

        # ── Structured reference guide: all six ICC forms ──────────────────
        # Content specified per explicit user requirement: for each form,
        # When to Use / Model Parameters / Academic Reference, tabulated
        # together so a user can see all six side by side rather than only
        # the one being used as the headline.
        reference_table = (
            ["Form", "Model", "Agreement Type", "Unit", "When to Use", "Reference"],
            [
                ["ICC(1,1)", "One-way random", "Absolute agreement", "Single",
                 "Different, randomly-selected items/raters evaluate each subject; "
                 "unit of interest is a single item/rater.",
                 "Shrout & Fleiss (1979); McGraw & Wong (1996)"],
                ["ICC(A,1)\n[ICC(2,1)]", "Two-way random", "Absolute agreement", "Single",
                 "The same set of random items/raters evaluate all subjects, and "
                 "exact score agreement (incl. systematic bias) for a single item matters.",
                 "McGraw & Wong (1996); Koo & Li (2016)"],
                ["ICC(C,1)\n[ICC(3,1)]", "Two-way mixed", "Consistency", "Single",
                 "The same items evaluate all subjects, measuring rank-order "
                 "consistency of a single item while ignoring systematic mean differences.",
                 "Shrout & Fleiss (1979)"],
                ["ICC(1,k)", "One-way random", "Absolute agreement", "Average (k)",
                 "Different items/raters assess each subject, and the target score "
                 "is the average across all k items.",
                 "Shrout & Fleiss (1979)"],
                ["ICC(A,k)\n[ICC(2,k)]", "Two-way random", "Absolute agreement", "Average (k)",
                 "The same items evaluate all subjects, assessing absolute agreement "
                 "of the overall scale score (average of k items) including systematic "
                 "item differences.",
                 "McGraw & Wong (1996); Koo & Li (2016)"],
                ["ICC(C,k)\n[ICC(3,k)] \u2014 PRIMARY HERE", "Two-way mixed", "Consistency", "Average (k)",
                 "STANDARD METRIC for scale internal consistency (Cronbach's alpha "
                 "equivalent). Evaluates consistency of the averaged scale score "
                 "when all subjects answer the same fixed items.",
                 "Shrout & Fleiss (1979); McGraw & Wong (1996); Koo & Li (2016)"],
            ])
        full_results_section = {
            "heading": "Full Results \u2014 All Six ICC Forms",
            "table": (cols, rows),
            "table_left": {0},
            "note": ("Complete numeric output (value, F, df1, df2, p, 95% CI) for "
                     "all six ICC forms, as returned by pingouin.intraclass_corr(). "
                     "ICC(C,k) (highlighted in the Primary Result above) is the "
                     "recommended headline metric for internal consistency; the "
                     "remaining five forms are provided here in full for "
                     "transparency and for researchers who need single-measures "
                     "forms, absolute-agreement forms, or one-way forms for a "
                     "specific purpose. See the Reference Guide below for when "
                     "each form is appropriate."),
        }

        reference_section = {
            "heading": "Reference Guide \u2014 All Six ICC Variants",
            "table": reference_table,
            "table_left": {0, 1, 2, 3, 4, 5},
            "note": ("Interpretation benchmarks (Koo & Li, 2016), applied identically "
                     "to all six forms: <0.50 poor; 0.50\u20130.75 moderate; 0.75\u20130.90 "
                     "good; >0.90 excellent \u2014 presented as a sourced heuristic, not "
                     "a universal statistical law. Full numeric result table for all "
                     "six computed forms (value, F, df1, df2, p, 95% CI) is provided "
                     "in the \u2018Full Results\u2019 section above."),
        }

        blocks.append({
            "key": "ICC", "title": "Intraclass Correlation Coefficient",
            "sections": [primary_section, full_results_section, reference_section],
            "cards": [], "table": None, "interp": "", "params": "", "note": "",
        })

    return blocks




def _item_tables(res):
    out = {}
    its = res.get("item_descriptives")
    if its is not None and not its.empty:
        out["Item Statistics"] = (list(its.columns),
            [[_f(v) if isinstance(v, (int, float, np.floating)) else str(v)
              for v in its.loc[i]] for i in its.index])
    summ = res.get("summary_item_stats")
    if summ is not None and not summ.empty:
        out["Summary Item Statistics"] = (list(summ.columns),
            [[_f(v) if isinstance(v, (int, float, np.floating)) else str(v)
              for v in summ.loc[i]] for i in summ.index])
    itot = res.get("item_stats")
    if itot is not None and not itot.empty:
        out["Item-Total Statistics"] = (list(itot.columns),
            [[_f(v) if isinstance(v, (int, float, np.floating)) else str(v)
              for v in itot.loc[i]] for i in itot.index],
            bool(itot.attrs.get("has_negative_alpha")))
    iic = res.get("inter_item_corr")
    if iic is not None and not iic.empty:
        out["Inter-Item Correlation Matrix"] = ([""] + list(iic.columns),
            [[idx] + [_f(v) for v in iic.loc[idx]] for idx in iic.index])
    return out




def _scale_rows(res):
    s = res.get("scale_stats", {})
    return (["Statistic", "Value"],
            [["Mean", _f(s.get("mean"))], ["Median", _f(s.get("median"))],
             ["Variance", _f(s.get("variance"))],
             ["Std. Deviation", _f(s.get("sd"))],
             ["Minimum", _f(s.get("min"))], ["Maximum", _f(s.get("max"))],
             ["Range", _f(s.get("range"))],
             ["Skewness", _f(s.get("skewness"))],
             ["Kurtosis", _f(s.get("kurtosis"))],
             ["N of Items", str(s.get("k", ""))],
             ["N of Cases", str(s.get("n", ""))]])


# ══════════════════════════════════════════════════════════════════════════════
# METHODOLOGICAL NOTES
# ══════════════════════════════════════════════════════════════════════════════

def _methodological_notes(M, alpha):
    """Methodological notes specific to Internal Consistency analysis."""
    pct = int((1 - alpha) * 100)
    rm = M.get("reverse_method")
    rm_name = {"theoretical": "theoretical scale endpoints ((min+max) − x)",
               "observed": "observed min/max ((min+max) − x)",
               "zscore": "z-score reversal (−z)"}.get(rm, "none (reversal off)")
    return [
        ("Cronbach's Alpha",
         "α = (k/(k−1))(1 − Σ s²ᵢ / s²_total). Standardised alpha uses the "
         "inter-item correlation matrix. The "
         f"{pct}% CI uses the Feldt (1965) F-based method. "
         "Interpretation benchmarks (George & Mallery, 2003): "
         "α ≥ .90 Excellent; .80–.89 Good; .70–.79 Acceptable; "
         ".60–.69 Questionable; .50–.59 Poor; < .50 Unacceptable."),
        ("McDonald's Omega",
         "Estimated from single-factor loadings (one-factor congeneric model). "
         "ω = (Σλᵢ)² / ((Σλᵢ)² + Σθᵢᵢ), where λᵢ are factor loadings and "
         "θᵢᵢ are unique variances. Preferred over α when items are not "
         "strictly tau-equivalent (unequal true-score variances). "
         "Reference: McDonald (1999). Test Theory: A Unified Treatment."),
        ("Guttman's Lambda",
         "λ₁–λ₆ from the covariance matrix. λ₃ = Cronbach's α. "
         "λ₂ is frequently the tightest lower bound to reliability. "
         "λ₆ (corrected split-half) requires an invertible item correlation matrix. "
         "Reference: Guttman (1945). Psychometrika."),
        ("Split-Half Reliability",
         "First half vs second half of items; "
         "Spearman-Brown formula assumes equal-length, parallel halves; "
         "Guttman split-half does not require this assumption. "
         "Reference: Spearman (1910); Brown (1910)."),
        ("KR-20",
         "Kuder-Richardson Formula 20 — the special case of Cronbach's α "
         "for strictly dichotomous (0/1) items. "
         "KR-20 = (k/(k−1)) × (1 − Σ(pᵢqᵢ) / s²_total). "
         "Reported automatically only when all items are 0/1 coded. "
         "Reference: Kuder & Richardson (1937). Psychometrika."),
        ("Intraclass Correlation (ICC)",
         "Two-way ANOVA decomposition (pingouin; Shrout & Fleiss, 1979). "
         "Six standard forms reported: ICC(1,1), ICC(A,1), ICC(C,1) "
         "for single-rater, and ICC(1,k), ICC(A,k), ICC(C,k) for mean of k raters. "
         "Absolute agreement (A) penalises systematic rater bias; "
         "consistency (C) does not. "
         "Benchmarks (Koo & Li, 2016): < .50 Poor; .50–.75 Moderate; "
         ".75–.90 Good; > .90 Excellent."),
        ("Item-Total Statistics",
         "Corrected item-total correlation (r_it): Pearson correlation of the item "
         "with the sum of remaining items. r_it < .30 suggests weak discrimination. "
         "Squared multiple correlation (R²): proportion of item variance explained "
         "by all other items. α-if-deleted: Cronbach's α computed with the item removed; "
         "a value exceeding the full-scale α indicates the item reduces reliability. "
         "Negative α-if-deleted values are flagged (footnote a) and signal model "
         "violations (e.g., un-reversed reverse-coded items)."),
        ("Reverse-Scored Items",
         f"Reversal method applied: {rm_name}. Items with negative corrected "
         "item-total correlations are automatically flagged. Reversal is OFF by "
         "default; when enabled, the exact items reversed and the formula used "
         "are reported per construct."),
        ("Missing Data",
         "Listwise deletion per construct: only cases with complete data across "
         "all items in a construct are included. N per construct is reported."),
        ("Interpretation Thresholds",
         "Cronbach's α / McDonald's ω (George & Mallery, 2003): "
         "≥ .90 Excellent; .80–.89 Good; .70–.79 Acceptable; "
         ".60–.69 Questionable; .50–.59 Poor; < .50 Unacceptable. "
         "ICC (Koo & Li, 2016): < .50 Poor; .50–.75 Moderate; "
         ".75–.90 Good; > .90 Excellent."),
        ("References",
         "Cronbach (1951) Psychometrika; Kuder & Richardson (1937) Psychometrika; "
         "Guttman (1945) Psychometrika; Feldt (1965) Psychometrika; "
         "Shrout & Fleiss (1979) Psychological Bulletin; McDonald (1999); "
         "George & Mallery (2003); Koo & Li (2016) Journal of Chiropractic Medicine."),
    ]




# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _alpha_bar(summary):
    names = summary["Construct"].tolist()
    vals = [float(v) if v == v else 0 for v in summary["Cronbach's \u03b1"]]
    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.6 * len(names) + 1)))
    colors = ["#16a34a" if v >= .8 else "#0284c7" if v >= .7
              else "#f59e0b" if v >= .6 else "#dc2626" for v in vals]
    ax.barh(names, vals, color=colors, edgecolor="white")
    for thr in (.7, .8, .9):
        ax.axvline(thr, color="#94a3b8", ls="--", lw=.8)
    ax.set_xlim(0, 1); ax.set_xlabel("Cronbach's \u03b1", fontsize=9, color=PAL[0])
    ax.set_title("Reliability by Construct", fontsize=11, fontweight="bold", color=PAL[0])
    for i, v in enumerate(vals):
        ax.text(min(v + .02, .95), i, f"{v:.3f}", va="center", fontsize=8, color=PAL[0])
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("white"); fig.tight_layout()
    return fig


def _item_total_plot(res):
    its = res.get("item_stats")
    if its is None or its.empty:
        return None
    items = its["Item"].tolist()
    vals = [float(v) if v == v else 0 for v in its["Corrected Item-Total Correlation"]]
    fig, ax = plt.subplots(figsize=(7, max(2, 0.5 * len(items) + 1)))
    colors = ["#dc2626" if v < 0 else "#0284c7" if v < .3 else "#16a34a" for v in vals]
    ax.barh(items, vals, color=colors, edgecolor="white")
    ax.axvline(.3, color="#94a3b8", ls="--", lw=.8)
    ax.axvline(0, color="#1a1a2e", lw=.8)
    ax.set_xlabel("Corrected Item-Total Correlation", fontsize=9, color=PAL[0])
    ax.set_title(f"Item Discrimination \u2014 {res['construct_name']}",
                 fontsize=10, fontweight="bold", color=PAL[0])
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("white"); fig.tight_layout()
    return fig




# ══════════════════════════════════════════════════════════════════════════════
# CSS + HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(load_page_css(), unsafe_allow_html=True)

st.markdown("""
<div class="main-hdr">
  <h1>&#129514; Internal Consistency
    <span class="badge">Publication-Grade</span>
    <span class="badge">Multi-Construct</span>
    <span class="badge">Extended Diagnostics</span>
  </h1>
  <p>Cronbach&#39;s &alpha; &middot; KR-20 &middot; Split-half &middot;
     Guttman &lambda; &middot; ICC &middot; McDonald&#39;s &omega; &nbsp;&middot;&nbsp;
     Analyse multiple constructs from a single file in one run</p>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:1rem;">
  <p style="color:#64748b;font-size:.9rem;margin-bottom:.4rem;">
    &#9749; Support this work &mdash; scan QRIS:</p>
  <img src="https://muhaiminabdullah.com/media/images/qris_muhaiminabdullahdotcom.png"
       style="width:180px;border-radius:10px;border:2px solid #e94560;display:block;"/>
</div>""", unsafe_allow_html=True)



st.info(
    "ℹ️ The first row of your file is treated as the header "
    "(variable names) and is never counted as data. Upload one file "
    "containing all scale items, choose how many constructs you have, "
    "then assign items to each construct.")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.page_link(st.session_state["_home_page"], label="Home", icon="🏠")
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    alpha = st.selectbox("\u03b1 (for confidence intervals)",
                         [0.05, 0.01, 0.10], index=0)
    
    analysis_mode = "Internal Consistency"
    st.markdown("---")

    uploaded = st.file_uploader("&#128194; Upload Data File",
                                type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            df = load_file(uploaded)
            fm = file_metadata(uploaded, df)
            val = validate_file(df)
            if val["errors"]:
                for e in val["errors"]: st.error(e)
                df = None
            else:
                st.success(f"&#10003; {fm['Rows']:,} rows \u00d7 {fm['Columns']} cols")
                for wm in val["warnings"]: st.warning(wm)
        except Exception as e:
            st.error(str(e)); df = None
    
    else:
        df = None

    cfg = None
    if df is not None:
        num_cols = get_numeric_cols(df)

        def _is_id(col, series):
            cl = col.lower().strip()
            idn = {'id', 'no', 'num', 'number', 'index', 'subject_id',
                   'respondent', 'participant', 'sampel', 'nomor'}
            if not (cl in idn or cl.endswith(('_id', '_no'))
                    or cl.startswith(('id_', 'subject'))):
                return False
            s = series.dropna()
            try:
                return bool((s == s.astype(int)).all()) and s.nunique() == len(s)
            except Exception:
                return False
        avail = [c for c in num_cols if not _is_id(c, df[c])]

        st.markdown("---")

        n_constructs = st.number_input(
            "\U0001f4ca Number of constructs",
            min_value=1, max_value=20, value=1, step=1)
        constructs = []; used = []
        for i in range(int(n_constructs)):
            st.markdown(f"**Construct {i+1}**")
            cname = st.text_input(
                "Name", value=f"Construct {i+1}", key=f"rel_name_{i}")
            citems = st.multiselect(
                "Items (\u2265 2)", avail, key=f"rel_items_{i}")
            if citems:
                constructs.append({"name": cname, "items": citems})
                used.extend(citems)
        dupes = set([x for x in used if used.count(x) > 1])
        if dupes:
            st.warning(
                f"\u26a0\ufe0f Items in multiple constructs: {', '.join(dupes)}")

        st.markdown("---")
        st.markdown("**\U0001f504 Reverse-Item Handling**")
        auto_rev = st.checkbox(
            "Auto-reverse flagged items", value=False,
            help="Off by default. When on, items with a negative "
                 "item-total correlation are reversed.")
        rev_method = "theoretical"; smin = smax = None
        if auto_rev:
            rev_method = st.selectbox(
                "Reversal method",
                ["theoretical", "observed", "zscore"], index=0)
            if rev_method == "theoretical":
                c1, c2 = st.columns(2)
                smin = c1.number_input("Scale min", value=1.0)
                smax = c2.number_input("Scale max", value=5.0)
                # SAFEGUARD (2026-10-21): added after confirming, by direct
                # construction, that reversing a dichotomous (0/1) item
                # with these UNCHANGED defaults silently produced non-0/1
                # values ({5,6}) that an older version of
                # modules.reliability._is_dichotomous still (incorrectly)
                # treated as "dichotomous," corrupting KR-20 (observed:
                # drove it from a normal in-range value to 122.42). That
                # root cause is fixed at the source (_is_dichotomous now
                # requires the values literally be 0/1), but a declared
                # scale range that does not actually bound the selected
                # items' observed data is still a strong, visible sign of
                # a mismatched reversal setup -- most commonly, exactly
                # this scenario (dichotomous or other non-1..5 data left
                # at the 1..5 default). Surfaced directly to the user
                # rather than silently falling back to a different
                # (still correct, but likely unintended) result.
                if used:
                    _obs_vals = pd.to_numeric(
                        df[used].stack(), errors="coerce").dropna()
                    if not _obs_vals.empty:
                        _omin, _omax = float(_obs_vals.min()), float(_obs_vals.max())
                        if _omin < smin or _omax > smax:
                            st.warning(
                                f"⚠️ The declared theoretical scale "
                                f"[{smin:g}, {smax:g}] does not cover the "
                                f"observed data range [{_omin:g}, {_omax:g}] "
                                f"for the selected item(s). Reversal computes "
                                f"(scale min + scale max) − x, so an "
                                f"incorrect range will silently produce the "
                                f"wrong reversed values. Set Scale min/max to "
                                f"match your actual response scale (e.g. 0 "
                                f"and 1 for dichotomous items) before relying "
                                f"on auto-reverse.")

        st.markdown("---")
        st.markdown("**\U0001f4c8 Additional Analyses (Optional)**")
        compute_bootstrap_ci = st.checkbox(
            "Compute bootstrap confidence intervals",
            value=False,
            help="Adds 95% percentile-bootstrap confidence intervals for "
                 "Guttman's Lambda (all 6 forms, both bases), all three "
                 "Split-Half variants, and McDonald's Omega-u (both "
                 "bases) \u2014 following the method recommended by "
                 "Kelley & Pornprasertmanit (2016) for reliability "
                 "coefficients generally. Off by default: adds "
                 "meaningfully to computation time (typically a few "
                 "seconds to under a minute, depending on item count) "
                 "since it involves ~15 separate bootstrap runs.")
        compute_omega_h = st.checkbox(
            "Compute McDonald's Omega-Hierarchical (\u03c9h)",
            value=False,
            help="Off by default. Investigation found this coefficient's "
                 "oblique-rotation extraction can take substantial, "
                 "unpredictable time (milliseconds to 10+ seconds) for "
                 "certain data structures, independent of item count \u2014 "
                 "this protects the speed of every other coefficient, "
                 "none of which share this issue. A confidence interval "
                 "is not computed for \u03c9h even when enabled (see the "
                 "\u03c9h section for the disclosed reason).")



        valid = [c for c in constructs if len(c.get("items", [])) >= 2]
        if len(valid) < len(constructs):
            st.warning("⚠️ Each construct needs at least 2 items.")
        if valid:
            cfg = {"constructs": valid, "auto_rev": auto_rev,
                   "rev_method": rev_method, "smin": smin, "smax": smax,
                   "compute_bootstrap_ci": compute_bootstrap_ci,
                   "compute_omega_h": compute_omega_h}
        st.markdown("---")
        run_btn = st.button("🚀 Run Analysis", type="primary",
                            width="stretch")
    else:
        run_btn = False

# ── Data preview ─────────────────────────────────────────────────────────────
if df is not None:
    with st.expander("🔍 Data Preview (row 1 = header, not data)", expanded=False):
        st.dataframe(df.head(20), width="stretch")

# ── Welcome ──────────────────────────────────────────────────────────────────
if not run_btn and "ic_M" not in st.session_state:
    st.markdown(
        '<div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);'
        'box-shadow:inset 0 0 0 1px rgba(2,132,199,.22);border-radius:12px;'
        'padding:1.2rem 1.4rem;margin:.8rem 0;">'
        '<h4 style="margin:0 0 .4rem;font-size:.95rem;font-weight:700;">'
        '🧪 Internal Consistency</h4>'
        '<p style="margin:0;font-size:.84rem;color:#475569;">'
        'Upload one file and assign columns to constructs to compute '
        'Cronbach’s α, KR-20, McDonald’s ω, '
        'Guttman’s λ, Split-Half reliability, and ICC(C,k).'
        '</p></div>',
        unsafe_allow_html=True)
    st.info("👇 Configure constructs in the sidebar, then click **Run Analysis**.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    if cfg is None:
        st.error("⚠️ Define at least one construct with ≥ 2 items.")
        st.stop()
    with st.spinner("Computing analysis…"):
        try:
            M = run_multi_reliability(
                df, cfg["constructs"], alpha=alpha,
                reverse_method=cfg["rev_method"], auto_reverse=cfg["auto_rev"],
                scale_min=cfg["smin"], scale_max=cfg["smax"],
                compute_bootstrap_ci=cfg.get("compute_bootstrap_ci", False),
                compute_omega_h=cfg.get("compute_omega_h", False))
            M["reverse_method"] = cfg["rev_method"] if cfg["auto_rev"] else None
            figs_b = [fig_to_bytes(_alpha_bar(M["summary"]))]
            for res in M["constructs"]:
                f = _item_total_plot(res)
                figs_b.append(fig_to_bytes(f) if f is not None else None)
        except Exception as e:
            import traceback as _tb
            st.error(f"⚠️ Analysis could not be completed: {e}")
            st.code(_tb.format_exc()); st.stop()

    st.session_state.update({"ic_M": M, "ic_figs": figs_b, "ic_alpha": alpha})

# ══════════════════════════════════════════════════════════════════════════════
# LOAD FROM SESSION
# ══════════════════════════════════════════════════════════════════════════════
M      = st.session_state["ic_M"]
figs_b = st.session_state["ic_figs"]
alpha  = st.session_state.get("ic_alpha", 0.05)

st.success("✓ Analysis complete!")

# ── Internal Consistency summary ──────────────────────────────────────
st.markdown('<div class="sec-title">Reliability Summary (All Constructs)</div>',
            unsafe_allow_html=True)
srows = [list(M["summary"].columns)] + [
    [_f(v) if isinstance(v,(int,float,np.floating)) else str(v)
     for v in M["summary"].loc[i]] for i in M["summary"].index]
st.markdown(html_tbl(srows, left_cols={0}), unsafe_allow_html=True)
if figs_b and figs_b[0] is not None:
    st.image(figs_b[0], width="stretch")
st.markdown('<p class="note-txt">Figures only \u2014 a quick numeric comparison. '
            'Full detail per coefficient in each tab below.</p>',
            unsafe_allow_html=True)



ctabs = st.tabs([res["construct_name"] for res in M["constructs"]] + ["📝 Notes"])
for ti, res in enumerate(M["constructs"]):
    with ctabs[ti]:
        # ── Internal Consistency rendering (unchanged) ────────────────
        a = res["alpha"]
        cls = "use-param" if (a == a and a >= 0.7) else "use-nonparam"
        st.markdown(
            f'<div class="decision-banner {cls}">Cronbach\u2019s \u03b1 = {_f(a)} '
            f'\u2014 <b>{alpha_label(a)}</b> internal consistency '
            f'({res["k"]} items, N = {res["n"]})</div>', unsafe_allow_html=True)
        if res.get("flagged_reverse") and not res.get("reverse_info"):
            st.warning("\u26a0\ufe0f Possible reverse-scored item(s): "
                       f"{', '.join(res['flagged_reverse'])}. "
                       "Enable auto-reverse in the sidebar if appropriate.")

        blocks = _coeff_blocks(res, alpha)
        stabs = st.tabs([b["title"] for b in blocks] + ["Item Statistics"])
        for si, blk in enumerate(blocks):
            with stabs[si]:
                if blk.get("sections"):
                    # Multi-section tab: several logically-distinct results
                    # (e.g. omega_t + omega_h, or raw-basis + standardized-
                    # basis lambdas) grouped under one tab with their own
                    # sub-headings, rather than scattered across separate
                    # top-level tabs.
                    for sec in blk["sections"]:
                        st.markdown(f'<div class="sub-title">{sec["heading"]}</div>',
                                    unsafe_allow_html=True)
                        if sec.get("cards"):
                            cols = st.columns(len(sec["cards"]))
                            for c, (lab, val, sub) in zip(cols, sec["cards"]):
                                c.markdown(
                                    f'<div class="metric-card"><div class="metric-val">{val}</div>'
                                    f'<div class="metric-lbl">{lab}</div>'
                                    f'<div style="font-size:.72rem;color:#e94560;font-weight:600;'
                                    f'margin-top:2px;">{sub}</div></div>', unsafe_allow_html=True)
                            st.markdown("")
                        if sec.get("table"):
                            st.markdown(html_tbl([sec["table"][0]] + sec["table"][1],
                                                left_cols=sec.get("table_left", {0})),
                                        unsafe_allow_html=True)
                            if sec.get("table_note"):
                                st.markdown(f'<p class="note-txt">{sec["table_note"]}</p>',
                                            unsafe_allow_html=True)
                        if sec.get("table2"):
                            st.markdown(html_tbl([sec["table2"][0]] + sec["table2"][1],
                                                left_cols=sec.get("table_left", {0})),
                                        unsafe_allow_html=True)
                            if sec.get("table2_note"):
                                st.markdown(f'<p class="note-txt">{sec["table2_note"]}</p>',
                                            unsafe_allow_html=True)
                        if sec.get("interp"):
                            st.markdown(f'<div class="interp-box sig">{sec["interp"]}</div>',
                                        unsafe_allow_html=True)
                        if sec.get("params"):
                            st.markdown(f'<p class="note-txt"><b>Decision parameters:</b> '
                                        f'{sec["params"]}</p>', unsafe_allow_html=True)
                        if sec.get("note"):
                            st.markdown(f'<p class="note-txt">{sec["note"]}</p>',
                                        unsafe_allow_html=True)
                        st.markdown("---")
                    continue
                cols = st.columns(len(blk["cards"]))
                for c, (lab, val, sub) in zip(cols, blk["cards"]):
                    c.markdown(
                        f'<div class="metric-card"><div class="metric-val">{val}</div>'
                        f'<div class="metric-lbl">{lab}</div>'
                        f'<div style="font-size:.72rem;color:#e94560;font-weight:600;'
                        f'margin-top:2px;">{sub}</div></div>', unsafe_allow_html=True)
                st.markdown("")
                if blk["table"]:
                    st.markdown(html_tbl([blk["table"][0]] + blk["table"][1],
                                        left_cols=blk.get("table_left", {0})),
                                unsafe_allow_html=True)
                    if blk.get("table_note"):
                        st.markdown(f'<p class="note-txt">{blk["table_note"]}</p>',
                                    unsafe_allow_html=True)
                if blk.get("table2"):
                    st.markdown(html_tbl([blk["table2"][0]] + blk["table2"][1],
                                        left_cols=blk.get("table_left", {0})),
                                unsafe_allow_html=True)
                    if blk.get("table2_note"):
                        st.markdown(f'<p class="note-txt">{blk["table2_note"]}</p>',
                                    unsafe_allow_html=True)
                if blk.get("footnote"):
                    st.markdown(f'<p class="note-txt">{blk["footnote"]}</p>',
                                unsafe_allow_html=True)
                st.markdown(f'<div class="interp-box sig">{blk["interp"]}</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<p class="note-txt"><b>Decision parameters:</b> '
                            f'{blk["params"]}</p>', unsafe_allow_html=True)
                if blk["note"]:
                    st.markdown(f'<p class="note-txt">{blk["note"]}</p>',
                                unsafe_allow_html=True)
        with stabs[-1]:
            for tname, payload in _item_tables(res).items():
                st.markdown(f'<div class="sub-title">{tname}</div>',
                            unsafe_allow_html=True)
                if tname == "Item-Total Statistics":
                    cols, rows, neg = payload
                    st.markdown(html_tbl([cols] + rows, left_cols={0}),
                                unsafe_allow_html=True)
                    if neg:
                        st.markdown('<p class="note-txt">a. A negative value '
                                    'indicates a negative average covariance among '
                                    'items, violating reliability model assumptions '
                                    '\u2014 check item codings (possible reverse item).</p>',
                                    unsafe_allow_html=True)
                    f = figs_b[ti + 1]
                    if f is not None:
                        st.image(f, width="stretch")
                else:
                    cols, rows = payload
                    st.markdown(html_tbl([cols] + rows, left_cols={0}),
                                unsafe_allow_html=True)
            scols, srows2 = _scale_rows(res)
            st.markdown('<div class="sub-title">Scale Statistics</div>',
                        unsafe_allow_html=True)
            st.markdown(html_tbl([scols] + srows2, left_cols={0}),
                        unsafe_allow_html=True)
            if res.get("reverse_info"):
                ritems = "; ".join(f"{it}: {f}" for it, f in res["reverse_info"])
                st.markdown(f'<p class="note-txt">Reverse-scored items applied: '
                            f'{ritems}.</p>', unsafe_allow_html=True)



with ctabs[-1]:
    st.markdown('<div class="sec-title">Methodological Notes</div>',
                unsafe_allow_html=True)
    for title, txt in _methodological_notes(M, alpha):
        st.markdown(f'<div class="norm-rec-box"><b>{title}.</b> {txt}</div>',
                    unsafe_allow_html=True)
