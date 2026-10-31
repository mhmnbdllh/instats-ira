"""
views/interrateragreement.py
=============================
Streamlit page for Inter-Rater Agreement analysis.

Covers Percentage Agreement, Cohen's kappa (unweighted + weighted, all
three weighting schemes), Fleiss' kappa, Gwet's AC1, Gwet's AC2 (all
five weighting schemes), Scott's pi, the Brennan-Prediger coefficient,
the Intraclass Correlation Coefficient (all six forms), Lin's
Concordance Correlation Coefficient, and Krippendorff's alpha (nominal,
ordinal, interval, and ratio, with a percentile-bootstrap confidence
interval).
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_APP_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_FILES = {
    "Dichotomous (0/1, N=86, 4 raters)":        "sample-instats_Dichotomous.csv",
    "Nominal / Ordinal (N=86, 4 raters)":       "sample-instats_Nominal-Ordinal.csv",
    "Interval / Ratio (N=86, 4 raters)":        "sample-instats_Interval-Ratio.csv",
}

st.set_page_config(
    page_title="InStats-IrA | Inter-Rater Agreement",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.styles      import load_page_css
from utils.file_loader import (load_file, validate_file,
                                file_metadata, get_numeric_cols,
                                load_sample_file)
from utils.formatters  import _f as _f_base, _p as _p_base
from utils.tables      import html_tbl
from utils.plots       import fig_to_bytes

from modules.reliability import run_multi_irar

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
# METHODOLOGICAL NOTES
# ══════════════════════════════════════════════════════════════════════════════

def _methodological_notes(M, alpha):
    """Methodological notes specific to Inter-Rater Agreement analysis."""
    pct = int((1 - alpha) * 100)
    constructs = M.get("constructs", [])
    if not constructs:
        return []
    # Gather all methods computed across all groups
    all_methods = set()
    for c in constructs:
        all_methods.update(c.get("applicable_methods", []))

    notes = []
    notes.append((
        "Percentage Agreement",
        "Observed proportion of exact pairwise agreements across all rater pairs. "
        "PA = (number of agreements) / (total pairwise comparisons). "
        "Does not correct for chance agreement and therefore should always be "
        "reported alongside a chance-corrected index. "
        "Reported here as a descriptive baseline. "
        "Reference: Stemler (2004). Practical Assessment, Research & Evaluation, 9(4)."
    ))

    if "cohen" in all_methods:
        notes.append((
            "Cohen's Kappa (κ)",
            "κ = (P_o − P_e) / (1 − P_e). P_o = observed agreement; "
            "P_e = expected agreement by chance = Σ p_c(r1) × p_c(r2). "
            "SE for z and two-tailed p: Fleiss, Cohen & Everitt (1969), Eq. 9 "
            "(H0: κ = 0); SE for the confidence interval: Eq. 8 of the same paper. "
            "Known limitation: κ can be paradoxically low when one category "
            "dominates (prevalence paradox) or when raters differ in marginal "
            "distributions (bias paradox). "
            "Benchmarks (Landis & Koch, 1977): < 0 Poor; 0.00–0.20 Slight; "
            "0.21–0.40 Fair; 0.41–0.60 Moderate; 0.61–0.80 Substantial; "
            "0.81–1.00 Almost Perfect. "
            "Reference: Cohen (1960). Educational and Psychological Measurement, 20(1)."
        ))

    if "gwet_ac1" in all_methods:
        notes.append((
            "Gwet's AC1",
            "AC1 = (P_o − P_e_AC1) / (1 − P_e_AC1). "
            "P_e(AC1) = (1/(q−1)) × Σ_c π_c(1−π_c), where π_c is the grand "
            "marginal proportion of category c across all raters. "
            "More stable than Cohen's κ under extreme marginal distributions "
            "(Gwet, 2008). Variance: delete-one-unit jackknife (Efron & Tibshirani, 1993). "
            "Benchmarks: Landis & Koch (1977), used here only as a conventional "
            "descriptive scale — Vach & Gerke (2023, MethodsX, 10, 102212) argue "
            "AC1 and κ respond differently to prevalence and that AC1 can be "
            "nonzero even with no real rater association, so a κ-calibrated "
            "verbal scale does not necessarily carry the same meaning for AC1. "
            "References: Gwet (2008). British Journal of Mathematical and Statistical "
            "Psychology, 61(1), 29–48. Gwet (2014). Handbook of Inter-Rater Reliability (4th ed.)."
        ))

    if "scott_pi" in all_methods:
        notes.append((
            "Scott's Pi (π)",
            "π = (P_o − P_e_scott) / (1 − P_e_scott). "
            "P_e(Scott) = Σ_c ((p_c(r1) + p_c(r2))/2)², using averaged (pooled) "
            "marginal proportions rather than their product. "
            "Differs from Cohen's κ only in P_e computation; typically yields "
            "slightly lower estimates when rater marginals differ. "
            "With two raters Scott's π equals Fleiss' (1971) κ, so its standard "
            "errors follow that literature: the null-hypothesis SE of Fleiss, Nee "
            "& Landis (1979) for z and p, and the general-purpose SE of Gwet "
            "(2021) for the confidence intervals. "
            "Reference: Scott (1955). Public Opinion Quarterly, 19(3), 321–325."
        ))

    if "bp" in all_methods:
        notes.append((
            "Brennan-Prediger Coefficient (κ_BP)",
            "κ_BP = (P_o − 1/q) / (1 − 1/q). Chance agreement is fixed at "
            "P_e = 1/q, the agreement expected if raters used the q declared "
            "categories uniformly at random; it does not depend on the observed "
            "marginals. Because 1/q is the smallest chance term of the Scott/Fleiss "
            "type, κ_BP is never below Scott's π for the same data. With q = 2 it "
            "equals the prevalence-adjusted bias-adjusted kappa (Byrt, Bishop & "
            "Carlin, 1993). "
            "Reference: Brennan & Prediger (1981). Educational and Psychological "
            "Measurement, 41(3), 687–699."
        ))

    if "fleiss" in all_methods:
        notes.append((
            "Fleiss' Kappa (κ)",
            "Generalises Cohen's κ to ≥ 3 raters. "
            "κ = (P̄ − P̄_e) / (1 − P̄_e). "
            "P̄ = mean per-unit agreement across all rater pairs; "
            "P̄_e = Σ_c p̄²_c (expected by chance). "
            "SE for z and p: Fleiss, Nee & Landis (1979) (valid under H0: κ = 0); "
            "SE for the confidence interval: Gwet (2021). "
            "Per-category κ_j also reported. "
            "Note: Fleiss' κ treats all categories as nominal (unordered). "
            "For ordinal multi-rater data, Gwet's AC2 or Krippendorff's α "
            "(ordinal metric) is preferred. "
            "Reference: Fleiss (1971). Psychological Bulletin, 76(5), 378–382."
        ))

    if "weighted" in all_methods:
        notes.append((
            "Weighted Cohen's Kappa (κ_w)",
            "κ_w = (Σ w_cc' p_cc' − Σ w_cc' p_c· p_·c') / (1 − Σ w_cc' p_c· p_·c'), "
            "with agreement weights over the q categories of the declared scale: "
            "quadratic w_cc' = 1 − (c−c')² / (q−1)²; linear w_cc' = 1 − |c−c'| / (q−1). "
            "Quadratic weights penalise larger disagreements disproportionately "
            "more; with quadratic weights κ_w is closely related to the intraclass "
            "correlation coefficient (Fleiss & Cohen, 1973). SE, z and p: Fleiss, "
            "Cohen & Everitt (1969). "
            "Reference: Cohen (1968). Psychological Bulletin, 70(4), 213–220."
        ))

    if "gwet_ac2" in all_methods:
        notes.append((
            "Gwet's AC2",
            "Weighted version of AC1 for ordinal data, computed under all five "
            "standard weighting schemes (Quadratic/Fleiss-Cohen, Linear/Cicchetti-"
            "Allison, Radical, Ordinal, and Identity; Gwet, 2014) — see the "
            "AC2 tab for the complete set with individual guidance and references. "
            "AC2 = (P_o_w − P_e_w) / (1 − P_e_w). "
            "P_e_w = u × Σ_k π_k(1−π_k) / (q(q−1)), where u = the sum of every "
            "cell of the q×q weight matrix (including the diagonal) and q = the "
            "user-declared number of categories in the coding scheme. This "
            "formula is verified to reduce exactly to AC1's own P_e formula "
            "when Identity (unweighted) weights are used. "
            "More stable than Weighted Cohen's κ under unequal marginals. "
            "Variance estimated via jackknife. "
            "Reference: Gwet (2014). Handbook of Inter-Rater Reliability (4th ed.), Chapter 4."
        ))

    if "icc" in all_methods:
        notes.append((
            "Intraclass Correlation Coefficient (ICC)",
            "Two-way ANOVA decomposition (pingouin; Shrout & Fleiss, 1979). "
            "Six forms: ICC(1,1) one-way random; ICC(A,1) two-way random absolute, "
            "single rater [PRIMARY for single-rater reliability]; "
            "ICC(C,1) two-way mixed consistency, single rater; "
            "ICC(1,k), ICC(A,k), ICC(C,k) = corresponding average-of-k forms. "
            "Absolute agreement (A) penalises systematic rater bias; "
            f"consistency (C) does not. {pct}% CI via F-distribution. "
            "Benchmarks (Koo & Li, 2016): < .50 Poor; .50–.75 Moderate; "
            ".75–.90 Good; > .90 Excellent. "
            "Reference: Shrout & Fleiss (1979). Psychological Bulletin, 86(2)."
        ))

    if "ccc" in all_methods:
        notes.append((
            "Lin's Concordance Correlation Coefficient (CCC / ρ_c)",
            "ρ_c = 2σ_xy / (σ_x² + σ_y² + (μ_x−μ_y)²) = ρ × C_b. "
            "ρ = Pearson correlation (precision); "
            "C_b = bias correction factor (accuracy). "
            "CCC simultaneously evaluates how tightly raters agree (precision) "
            "and how close their scores are to the line of perfect concordance (accuracy). "
            "|ρ_c| ≤ |r| always, because ρ_c = r × C_b with 0 < C_b ≤ 1; the gap "
            "between them reflects systematic bias. "
            f"{pct}% CI via Fisher z-transformation. "
            "Benchmarks (Koo & Li, 2016 adapted): ρ_c > .90 Excellent; "
            ".75–.90 Good; .50–.75 Moderate; < .50 Poor. "
            "References: Lin (1989). Biometrics, 45(1), 255–268. "
            "Carrasco & Jover (2003). Statistics in Medicine, 22(1), 163–175."
        ))

    if "krippendorff" in all_methods:
        _actual_n_boot = next(
            (c.get("krippendorff", {}).get("n_boot")
             for c in constructs if (c.get("krippendorff") or {}).get("n_boot")),
            None)
        _boot_txt = (f"{_actual_n_boot:,} resamples" if _actual_n_boot
                     else "the number of resamples selected in the sidebar")
        notes.append((
            "Krippendorff's Alpha (α)",
            "α = 1 − D_o/D_e. D_o = observed disagreement (mean pairwise distance "
            "within units); D_e = expected disagreement (mean pairwise distance "
            "across all values). Distance metric varies by data type: "
            "nominal d(a,b)=0/1; ordinal d(a,b)=(Σn_g from a to b − (n_a+n_b)/2)²; "
            "interval d(a,b)=(a−b)²; ratio d(a,b)=((a−b)/(a+b))². "
            "Handles missing data and any number of raters. "
            f"{pct}% CI: percentile bootstrap over rated units ({_boot_txt}, "
            "units resampled with replacement, seed=42). "
            "Thresholds (Krippendorff, 2004): α ≥ .800 reliable; "
            ".667 ≤ α < .800 tentatively acceptable; α < .667 unreliable. "
            "Reference: Krippendorff (2004). Content Analysis (2nd ed.)."
        ))

    notes.append((
        "Agreement Benchmarks — Summary",
        "Kappa family (Landis & Koch, 1977): < 0 Poor; 0.00–0.20 Slight; "
        "0.21–0.40 Fair; 0.41–0.60 Moderate; 0.61–0.80 Substantial; "
        "0.81–1.00 Almost Perfect. "
        "ICC / CCC (Koo & Li, 2016): < .50 Poor; .50–.75 Moderate; "
        ".75–.90 Good; > .90 Excellent. "
        "Krippendorff's α: ≥ .800 reliable; .667–.800 tentatively acceptable. "
        "Note: Percentage Agreement does not correct for chance and should "
        "not be used as a standalone reliability index."
    ))

    notes.append((
        "Missing Data",
        "Krippendorff's α uses every unit rated by at least two raters in the "
        "group (pairable values), so a unit with some missing ratings still "
        "contributes. Every other coefficient requires a rating from every "
        "selected rater and therefore uses only the units with complete "
        "ratings (listwise deletion within the rating group). The number of "
        "units used is reported with each coefficient."
    ))

    return notes




# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _irar_icc_bar(icc_res, title="ICC — All Forms"):
    """Bar chart of all 6 ICC forms — bbox labels, no overlap."""
    forms  = ["ICC(1,1)\nOne-way random\n1 rater",
              "ICC(A,1) ★ PRIMARY\nTwo-way random\nAbsolute, 1 rater",
              "ICC(C,1)\nTwo-way mixed\nConsistency, 1 rater",
              "ICC(1,k)\nOne-way random\nMean of k raters",
              "ICC(A,k)\nTwo-way random\nAbsolute, mean",
              "ICC(C,k)\nTwo-way mixed\nConsistency, mean"]
    keys   = ["icc_1_1","icc_2_1","icc_3_1","icc_1_k","icc_2_k","icc_3_k"]
    real_vals = []
    for k in keys:
        v = icc_res.get(k, np.nan)
        try:    real_vals.append(float(v) if not np.isnan(float(v)) else np.nan)
        except: real_vals.append(np.nan)
    disp = [max(min(v,1.0),-0.30) if not np.isnan(v) else 0.0 for v in real_vals]
    def _col(v):
        if np.isnan(v): return "#cbd5e1"
        if v>=.90: return "#16a34a"
        if v>=.75: return "#0284c7"
        if v>=.50: return "#f59e0b"
        return "#dc2626"
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.barh(forms, disp, color=[_col(v) for v in real_vals],
            edgecolor="white", height=0.52, zorder=3)
    for thr, col in [(0.50,"#f59e0b"),(0.75,"#0284c7"),(0.90,"#16a34a")]:
        ax.axvline(thr, color=col, ls="--", lw=0.9, alpha=0.45, zorder=1)
    ax.axvline(0, color="#334155", lw=0.9, zorder=2)
    ax.set_xlim(-0.42, 1.32)
    ax.set_xlabel("ICC value", fontsize=9, color=PAL[0])
    ax.set_title(title + "\n★ Primary: ICC(A,1) — absolute agreement, single rater"
                 "  |  Benchmarks (Koo & Li, 2016): <0.50 Poor · 0.50–0.75 Moderate · "
                 "0.75–0.90 Good · >0.90 Excellent",
                 fontsize=7.5, color=PAL[0], pad=8)
    for i, (rv, dv) in enumerate(zip(real_vals, disp)):
        if not np.isnan(rv):
            lx = dv+0.04 if dv>=0 else dv-0.04
            ha = "left" if dv>=0 else "right"
            ax.text(lx, i, f"{rv:.3f}", va="center", ha=ha,
                    fontsize=9, fontweight="bold", color=PAL[0], zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.88))
    ax.spines[["top","right"]].set_visible(False)
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=1.5)
    return fig


def _irar_kripp_gauge(alpha_val, label, title="Krippendorff's α"):
    """Horizontal gauge — colour zones, bbox label, zero overlap."""
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.set_xlim(-0.15, 1.10); ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines[["top","right","left"]].set_visible(False)
    ax.barh([0.5],[0.667],left=-0.15,height=0.38,color="#fee2e2",alpha=0.50,zorder=1)
    ax.barh([0.5],[0.133],left=0.667,height=0.38,color="#fef9c3",alpha=0.50,zorder=1)
    ax.barh([0.5],[0.233],left=0.800,height=0.38,color="#dcfce7",alpha=0.50,zorder=1)
    for mid, col, lbl in [(0.26,"#dc2626","Unreliable"),
                           (0.733,"#ca8a04","Tentative"),
                           (0.900,"#16a34a","Reliable")]:
        ax.text(mid, 0.50, lbl, ha="center", va="center", fontsize=7.5,
                color=col, alpha=0.65,
                transform=ax.get_xaxis_transform(), zorder=2)
    for thr, col, lbl in [(0.667,"#ca8a04","0.667"),(0.800,"#16a34a","0.800")]:
        ax.axvline(thr, color=col, lw=1.8, zorder=4, alpha=0.85)
        ax.text(thr, 0.97, lbl, ha="center", va="top", fontsize=8.5,
                color=col, fontweight="bold",
                transform=ax.get_xaxis_transform(), zorder=5,
                bbox=dict(boxstyle="round,pad=0.20",fc="white",ec="none",alpha=0.88))
    try:
        av = float(alpha_val)
        if not np.isnan(av):
            av_c = max(min(av,1.07),-0.12)
            col = "#16a34a" if av>=.80 else "#ca8a04" if av>=.667 else "#dc2626"
            ax.plot([av_c],[0.50],marker="v",markersize=22,color=col,zorder=6)
            ax.text(av_c,0.91,f"α = {av:.3f}",ha="center",va="top",
                    fontsize=11,fontweight="bold",color=col,
                    transform=ax.get_xaxis_transform(),zorder=7,
                    bbox=dict(boxstyle="round,pad=0.30",fc="white",
                              ec=col,alpha=0.97,lw=1.5))
    except Exception:
        pass
    ax.set_xlabel("Krippendorff's α", fontsize=9, color=PAL[0])
    ax.set_title(f"{title} — {label}", fontsize=10, fontweight="bold",
                 color=PAL[0], pad=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=1.3)
    return fig


def _irar_pa_gauge(pa_val, ci_lo, ci_hi, title="Percentage Agreement", ci_level="95%"):
    """Gauge for % Agreement — same readability standard as Kripp gauge."""
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.set_xlim(-0.02, 1.10); ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines[["top","right","left"]].set_visible(False)
    # Neutral axis: percentage agreement has no published benchmark scale,
    # so no evaluative zones or threshold lines are drawn.
    ax.barh([0.5],[1.0],left=0.0, height=0.38,color="#f1f5f9",alpha=0.90,zorder=1)
    try:
        if not (np.isnan(float(ci_lo)) or np.isnan(float(ci_hi))):
            ax.barh([0.5],[float(ci_hi)-float(ci_lo)],left=float(ci_lo),
                    height=0.38,color="#bae6fd",alpha=0.90,zorder=2)
    except Exception:
        pass
    try:
        pv = float(pa_val)
        if not np.isnan(pv):
            pv_c = max(min(pv,1.07),0.0)
            col = "#0369a1"
            ax.plot([pv_c],[0.50],marker="v",markersize=22,color=col,zorder=6)
            lbl_txt = f"PA = {pv*100:.1f}%"
            try:
                if not (np.isnan(float(ci_lo)) or np.isnan(float(ci_hi))):
                    lbl_txt += f"\n{ci_level} CI [{float(ci_lo)*100:.1f}–{float(ci_hi)*100:.1f}%]"
            except Exception:
                pass
            ax.text(pv_c,0.91,lbl_txt,ha="center",va="top",fontsize=10,
                    fontweight="bold",color=col,
                    transform=ax.get_xaxis_transform(),zorder=7,
                    bbox=dict(boxstyle="round,pad=0.30",fc="white",
                              ec=col,alpha=0.97,lw=1.5))
    except Exception:
        pass
    ax.set_xlabel("Percentage Agreement", fontsize=9, color=PAL[0])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x*100:.0f}%"))
    ax.set_title(title, fontsize=10, fontweight="bold", color=PAL[0], pad=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=1.3)
    return fig


def _irar_single_bar(value, label, method_name, group_name,
                     benchmarks=None, xlim=(-0.22, 1.12)):
    """Universal single-value bar — bbox label, no line overlap."""
    if benchmarks is None:
        benchmarks = [(0.20,"#94a3b8"),(0.40,"#f59e0b"),
                      (0.60,"#0284c7"),(0.80,"#16a34a")]
    try:    fv = float(value)
    except: fv = np.nan
    dv = max(min(fv,xlim[1]-0.06),xlim[0]+0.06) if not np.isnan(fv) else 0.0
    col = ("#16a34a" if not np.isnan(fv) and fv>=.80 else
           "#0284c7" if not np.isnan(fv) and fv>=.60 else
           "#f59e0b" if not np.isnan(fv) and fv>=.40 else "#dc2626")
    fig, ax = plt.subplots(figsize=(8.5, 1.85))
    ax.barh([method_name],[dv],color=col if not np.isnan(fv) else "#cbd5e1",
            edgecolor="white",height=0.45,zorder=3)
    for thr, tcol in benchmarks:
        ax.axvline(thr,color=tcol,ls="--",lw=0.9,alpha=0.45,zorder=1)
    ax.axvline(0,color="#334155",lw=0.9,zorder=2)
    ax.set_xlim(xlim)
    ax.set_xlabel("Value",fontsize=9,color=PAL[0])
    ax.set_title(f"{method_name} = {fv:.3f} — {label} — {group_name}",
                 fontsize=10,fontweight="bold",color=PAL[0],pad=6)
    if not np.isnan(fv):
        lx = dv+0.04 if dv>=0 else dv-0.04
        ha = "left" if dv>=0 else "right"
        ax.text(lx,0,f"{fv:.3f}",va="center",ha=ha,
                fontsize=11,fontweight="bold",color=PAL[0],zorder=5,
                bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="none",alpha=0.92))
    ax.set_yticks([])
    ax.spines[["top","right","left"]].set_visible(False)
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=1.2)
    return fig


def _irar_kappa_heatmap(ct_df, r1_name="Rater 1", r2_name="Rater 2",
                         title="Confusion Matrix"):
    """Heatmap — contrast cell text, colorbar outside, no overlap."""
    if ct_df is None or ct_df.empty: return None
    n = max(len(ct_df.columns), len(ct_df.index))
    sz = max(3.8, 0.95*n + 1.4)
    fig, ax = plt.subplots(figsize=(sz+1.2, sz))
    arr = ct_df.values.astype(float)
    im = ax.imshow(arr, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(ct_df.columns)))
    ax.set_xticklabels(ct_df.columns, fontsize=9)
    ax.set_yticks(range(len(ct_df.index)))
    ax.set_yticklabels(ct_df.index, fontsize=9)
    ax.set_xlabel(r2_name, fontsize=9, color=PAL[0], labelpad=7)
    ax.set_ylabel(r1_name, fontsize=9, color=PAL[0], labelpad=7)
    vmax = arr.max() if arr.size else 1
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i,j]
            ax.text(j, i, str(int(v)), ha="center", va="center",
                    fontsize=12, fontweight="bold", zorder=4,
                    color="white" if v>vmax*0.58 else PAL[0])
    ax.set_title(title, fontsize=10, fontweight="bold", color=PAL[0], pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.08)
    cbar.ax.tick_params(labelsize=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=1.3)
    return fig


def _irar_fleiss_bar(kappa_val, per_cat, title="Fleiss' κ — Per-Category Agreement"):
    """Bar: overall κ + per-category — bbox labels, no overlap."""
    if not per_cat: return None
    cats = ["Overall κ"] + [str(c) for c in per_cat.keys()]
    vals = [kappa_val] + [
        float(v) if v is not None and not (isinstance(v,float) and np.isnan(v)) else np.nan
        for v in per_cat.values()]
    disp = [max(min(v,1.05),-0.22) if not np.isnan(v) else 0.0 for v in vals]
    def _col(v, i):
        if np.isnan(v): return "#cbd5e1"
        if i==0: return "#1a1a2e"
        if v>=.80: return "#16a34a"
        if v>=.60: return "#0284c7"
        if v>=.40: return "#f59e0b"
        return "#dc2626"
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.70*len(cats)+1.5)))
    ax.barh(cats, disp, color=[_col(vals[i],i) for i in range(len(vals))],
            edgecolor="white", height=0.50, zorder=3)
    for thr, col in [(0.20,"#94a3b8"),(0.40,"#f59e0b"),
                     (0.60,"#0284c7"),(0.80,"#16a34a")]:
        ax.axvline(thr, color=col, ls="--", lw=0.9, alpha=0.45, zorder=1)
    ax.axvline(0, color="#334155", lw=0.9, zorder=2)
    ax.set_xlim(-0.32, 1.22)
    ax.set_xlabel("Kappa (κ)", fontsize=9, color=PAL[0])
    ax.set_title(title, fontsize=10, fontweight="bold", color=PAL[0], pad=6)
    for i, (v, dv) in enumerate(zip(vals, disp)):
        if not np.isnan(v):
            lx = dv+0.04 if dv>=0 else dv-0.04
            ha = "left" if dv>=0 else "right"
            ax.text(lx, i, f"{v:.3f}", va="center", ha=ha,
                    fontsize=9, fontweight="bold", color=PAL[0], zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18",fc="white",ec="none",alpha=0.90))
    ax.spines[["top","right"]].set_visible(False)
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=1.5)
    return fig


# NOTE (2026-10-21): an empty "HTML EXPORT" section header used to sit here
# with no code under it (confirmed by reading through to the next section
# header with nothing in between) -- removed as a leftover stub from a
# feature that was never implemented on this page, rather than leaving a
# misleading placeholder in place.


# ══════════════════════════════════════════════════════════════════════════════
# CSS + HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(load_page_css(), unsafe_allow_html=True)

st.markdown("""
<div class="main-hdr">
  <h1>&#129514; Inter-Rater Agreement
    <span class="badge">InStats-IrA</span>
    <span class="badge">Inter-rater Agreement</span>
    <span class="badge">Multi-Group</span>
    <span class="badge">Extended Diagnostics</span>
  </h1>
  <p>Cohen&#39;s &kappa; &middot; Fleiss&#39; &kappa; &middot; Gwet&#39;s AC1/AC2 &middot;
     Scott&#39;s &pi; &middot; Brennan-Prediger &middot; ICC &middot; Lin&#39;s CCC &middot;
     Krippendorff&#39;s &alpha; &nbsp;&middot;&nbsp;
     Analyse multiple rating groups from a single file in one run</p>
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
    "(variable names) and is never counted as data. Each column should "
    "represent one rater/judge, each row one unit being rated.")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.page_link(st.session_state["_home_page"], label="Home", icon="🏠")
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    alpha = st.selectbox("\u03b1 (for confidence intervals)",
                         [0.05, 0.01, 0.10], index=0)
    st.markdown("---")

    _SAMPLE_NONE = "— None (I'll upload my own file) —"
    with st.expander("📂 Try with sample data"):
        st.caption(
            # The sample files are described as "illustrative" because
            # they serve to demonstrate the procedure; they carry no
            # personal identifiers.
            "Three illustrative rating datasets (N=86 units, up to 4 raters each) — "
            "one per data type family. Pick one below to load it straight into "
            "the analysis, or download it first to inspect or edit."
        )
        _samp_choice = st.selectbox(
            "Load a sample dataset directly",
            [_SAMPLE_NONE] + list(_SAMPLE_FILES.keys()),
            index=0,
            key="ira_samp_choice",
        )
        for _label, _fname in _SAMPLE_FILES.items():
            _fpath = _APP_ROOT / _fname
            if _fpath.exists():
                st.download_button(
                    f"⬇ {_label}",
                    data=_fpath.read_bytes(),
                    file_name=_fname,
                    mime="text/csv",
                    key=f"dl_{_fname}",
                    width="stretch",
                )

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
    elif st.session_state.get("ira_samp_choice", _SAMPLE_NONE) != _SAMPLE_NONE:
        # Direct sample-data loading (restores a capability present in the
        # original reliability.py / 6_reliability.py views, which let a user
        # pick a built-in sample and have it load automatically -- the
        # dedicated InStats-IrA view had only offered download-then-reupload).
        _samp_fname = _SAMPLE_FILES[st.session_state["ira_samp_choice"]]
        _samp_fpath = _APP_ROOT / _samp_fname
        try:
            df = load_sample_file(_samp_fpath)
            st.info(f"ℹ️ Using built-in sample data — {st.session_state['ira_samp_choice']}")
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

        
        st.info(
            "📋 **Data format:** Each **column** = one rater. "
            "Each **row** = one unit rated (student, essay, item). "
            "All raters must score the same set of units.")

        n_groups = st.number_input(
            "Number of rating groups",
            min_value=1, max_value=20, value=1, step=1,
            help="Each group = one set of raters rating the same units on one aspect.")

        dtype_options = {
            "Dichotomous (two categories coded, e.g., 0/1)":  "dichotomous",
            "Nominal (unordered categories, integer-coded)":   "nominal",
            "Ordinal (Likert 1–5, rubric 1–4, ordered)":       "ordinal",
            "Interval (scores 0–100, continuous)":             "interval",
            "Ratio (time, weight, continuous with true zero)":  "ratio",
        }

        constructs = []; used_globally = []
        kripp_will_run = False  # track if any group needs Krippendorff
        scale_declarations_ok = True  # tracks whether every group that
        # needs a declared instrument scale (AC1/AC2/Brennan-Prediger)
        # has actually supplied it -- gates the Run button below.
        for i in range(int(n_groups)):
            st.markdown(f"**Rating Group {i+1}**")
            # Separate widget keys for IRAR vs IC (irar_ prefix)
            cname = st.text_input(
                "Group name", value=f"Rating Group {i+1}",
                key=f"irar_name_{i}")
            avail_now = [c for c in avail if c not in used_globally]
            citems = st.multiselect(
                "Select rater columns (≥ 2)",
                avail_now, key=f"irar_items_{i}",
                help="Each column = one rater. Columns used in other groups are excluded.")
            dtype_choice = st.selectbox(
                "Data type",
                list(dtype_options.keys()),
                key=f"irar_dtype_{i}",
                help=(
                    "Dichotomous: exactly 2 categories, integer-coded (e.g., 0 = No, 1 = Yes).\n"
                    "Nominal: 3+ unordered categories, each coded as an integer "
                    "(e.g., 1 = error type A, 2 = type B, 3 = type C).\n"
                    "Ordinal: ordered levels (Likert scale, rubric).\n"
                    "Interval: continuous scores without true zero.\n"
                    "Ratio: continuous with true zero (time, frequency)."))
            irar_dtype_i = dtype_options[dtype_choice]
            q_declared_i = None; scale_min_i = None
            if citems:
                k = len(citems)
                from modules.reliability import _get_applicable_methods
                methods = _get_applicable_methods(irar_dtype_i, k)
                if "krippendorff" in methods:
                    kripp_will_run = True
                method_names = {
                    "pct":          "% Agreement",
                    "cohen":        "Cohen's κ",
                    "gwet_ac1":     "Gwet's AC1",
                    "scott_pi":     "Scott's π",
                    "bp":           "Brennan-Prediger",
                    "fleiss":       "Fleiss' κ",
                    "weighted":     "Weighted Cohen's κ",
                    "gwet_ac2":     "Gwet's AC2",
                    "icc":          "ICC (all 6 forms)",
                    "ccc":          "Lin's CCC",
                    "krippendorff": "Krippendorff's α",
                }
                methods_str = " + ".join(method_names.get(m, m) for m in methods)
                st.markdown(
                    f'<div style="font-size:.78rem;color:#0284c7;'
                    f'padding:4px 8px;background:#f0f9ff;border-radius:4px;">'
                    f'&#10003; {k} rater{"s" if k>1 else ""} | '
                    f'{irar_dtype_i.title()} → '
                    f'<b>{methods_str}</b>'
                    f'</div>', unsafe_allow_html=True)

                # ── Instrument-scale declaration ──────────────────────
                # The q-dependent coefficients (Gwet's AC1/AC2, Brennan-
                # Prediger) need the number of categories the instrument
                # defines. The data alone cannot reveal it when a valid
                # category is absent from the sample, so it is declared,
                # not inferred (Gwet, Handbook of Inter-Rater Reliability,
                # 4th ed., Ch. 4; Brennan & Prediger, 1981). The one
                # exception is dichotomous data, which have exactly two
                # categories by definition: q is fixed at 2 and nothing
                # is asked. For the other declarations the fields start
                # EMPTY (no plausible-looking default) and the Run button
                # stays disabled until every such group has a value.
                needs_q = any(m in methods for m in ("gwet_ac1", "bp"))
                needs_ac2 = "gwet_ac2" in methods
                if irar_dtype_i == "dichotomous":
                    # Two categories by definition; q is not asked.
                    q_declared_i = 2
                elif needs_ac2 and irar_dtype_i == "ordinal":
                    st.markdown(
                        '<p style="font-size:.78rem;color:#9a3412;'
                        'font-weight:600;margin:6px 0 2px;">'
                        '&#9888; Required: declare the full instrument scale '
                        '(not just the values observed in this data) — see '
                        'Gwet (2014) / Brennan &amp; Prediger (1981).</p>',
                        unsafe_allow_html=True)
                    sc1, sc2 = st.columns(2)
                    # Integer-typed inputs (step=1): the widget returns a
                    # whole number, so the declared bounds, q and the scale
                    # minimum passed to the engine are integers by
                    # construction (rating-scale categories carry integer
                    # codes).
                    smin_in = sc1.number_input(
                        "Scale minimum", value=None, step=1,
                        key=f"irar_scalemin_{i}", placeholder="e.g., 1",
                        help="The lowest possible value on the rating instrument, "
                             "whether or not it was actually used by raters.")
                    smax_in = sc2.number_input(
                        "Scale maximum", value=None, step=1,
                        key=f"irar_scalemax_{i}", placeholder="e.g., 5",
                        help="The highest possible value on the rating instrument, "
                             "whether or not it was actually used by raters.")
                    if smin_in is not None and smax_in is not None and smax_in > smin_in:
                        q_declared_i = int(smax_in - smin_in + 1)
                        scale_min_i = int(smin_in)
                        st.caption(f"Declared scale: {int(smin_in)}–{int(smax_in)} "
                                   f"→ q = {q_declared_i} categories. Gwet's AC2 will "
                                   f"be computed under all five standard weighting "
                                   f"schemes (see the AC2 tab for the full set).")
                    else:
                        scale_declarations_ok = False
                        st.error("Enter both scale minimum and maximum "
                                 "(maximum must exceed minimum) to proceed.")
                elif needs_q and not needs_ac2:
                    # Nominal: no inherent order, so a bare category count
                    # is requested rather than a minimum and maximum.
                    st.markdown(
                        '<p style="font-size:.78rem;color:#9a3412;'
                        'font-weight:600;margin:6px 0 2px;">'
                        '&#9888; Required: declare the full instrument scale '
                        '(not just the values observed in this data) — see '
                        'Gwet (2014) / Brennan &amp; Prediger (1981).</p>',
                        unsafe_allow_html=True)
                    q_in = st.number_input(
                        "Total number of categories in the coding scheme",
                        value=None, step=1, min_value=2,
                        key=f"irar_qcount_{i}", placeholder="e.g., 5",
                        help="The number of categories the coding scheme "
                             "defines, whether or not every category was "
                             "actually used by raters in this dataset.")
                    if q_in is not None:
                        q_declared_i = int(q_in)
                    else:
                        scale_declarations_ok = False
                        st.error("Enter the total number of categories to proceed.")

                constructs.append({
                    "name": cname, "items": citems,
                    "data_type": irar_dtype_i,
                    "q_declared": q_declared_i,
                    "scale_min": scale_min_i})
                used_globally.extend(citems)
            st.markdown("")

        # Bootstrap option — shown only when Krippendorff will run
        n_boot_irar = 1000
        if kripp_will_run:
            st.markdown("---")
            st.markdown("**Krippendorff's α — Bootstrap CI**")
            _boot_opts = {
                "1,000 resamples":  1000,
                "2,000 resamples":  2000,
                "5,000 resamples":  5000,
                "10,000 resamples": 10000,
            }
            _boot_choice = st.selectbox(
                f"Bootstrap resamples for the {round((1 - alpha) * 100):g}% CI",
                list(_boot_opts.keys()),
                index=0,
                key="irar_n_boot",
                help=(
                    "Rated units are resampled with replacement and α is "
                    "recomputed on each resample. More resamples give a "
                    "smoother, more reproducible interval; 10,000 resamples "
                    "take a few seconds."))
            n_boot_irar = _boot_opts[_boot_choice]

        # DEAD CODE REMOVED (2026-10-21): `auto_rev`, `rev_method`, `smin`,
        # `smax` were assigned here but never read anywhere else in this
        # file (confirmed by a full line-by-line read of the whole file,
        # not just a grep for these names). They are reverse-scoring
        # concepts specific to the Internal Consistency module (automatic
        # vs. theoretical reverse-item detection, scale min/max) that had
        # no business being declared on this Inter-Rater Agreement page at
        # all -- almost certainly left over from copying the sidebar
        # structure across the two modules.

        valid = [c for c in constructs if len(c.get("items", [])) >= 2]
        if len(valid) < len(constructs):
            st.warning("⚠️ Each rating group needs at least 2 raters.")
        if valid:
            cfg = {"constructs": valid, "n_boot": n_boot_irar}
        st.markdown("---")
        if not scale_declarations_ok:
            st.info("🔒 Complete every required instrument-scale field "
                     "above (marked with ⚠️) to enable analysis.")
        run_btn = st.button("🚀 Run Analysis", type="primary",
                            width="stretch",
                            disabled=not scale_declarations_ok)
    else:
        run_btn = False

# ── Data preview ─────────────────────────────────────────────────────────────
if df is not None:
    with st.expander("🔍 Data Preview (row 1 = header, not data)", expanded=False):
        st.dataframe(df.head(20), width="stretch")

# ── Welcome ──────────────────────────────────────────────────────────────────
if not run_btn and "ira_M" not in st.session_state:
    st.markdown(
        '<div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);'
        'box-shadow:inset 0 0 0 1px rgba(2,132,199,.22);border-radius:12px;'
        'padding:1.2rem 1.4rem;margin:.8rem 0;">'
        '<h4 style="margin:0 0 .4rem;font-size:.95rem;font-weight:700;">'
        '🧪 Inter-Rater Agreement</h4>'
        '<p style="margin:0;font-size:.84rem;color:#475569;">'
        'Upload one file where each column is a rater and each row is a '
        'rated unit, declare the data type per rating group, then compute '
        'Cohen’s/Fleiss’ &kappa;, Gwet’s AC1/AC2, Scott’s &pi;, '
        'Brennan-Prediger, ICC, Lin’s CCC, and Krippendorff’s &alpha;.'
        '</p></div>',
        unsafe_allow_html=True)
    st.info("👇 Configure rating groups in the sidebar, then click **Run Analysis**.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    if cfg is None:
        st.error("⚠️ Define at least one rating group with ≥ 2 raters.")
        st.stop()
    with st.spinner("Computing analysis…"):
        try:
            M = run_multi_irar(df, cfg["constructs"], alpha=alpha,
                               n_boot=cfg.get("n_boot", 1000))
        except Exception as e:
            import traceback as _tb
            st.error(f"⚠️ Analysis could not be completed: {e}")
            st.code(_tb.format_exc()); st.stop()

    # DEAD CODE REMOVED (2026-10-21): this used to also set
    # figs_b = [None], store it into st.session_state as "ira_figs", and
    # read it straight back below into a module-level `figs_b` variable --
    # confirmed, by reading every remaining line of this file, that
    # `figs_b` is never referenced again after that round trip. All
    # charts in this page are built fresh, inline, inside each method's
    # own tab (e.g. _irar_icc_bar, _irar_pa_gauge) -- there is no
    # mechanism anywhere that reads a pre-built figure list back out.
    st.session_state.update({"ira_M": M, "ira_alpha": alpha})

# ══════════════════════════════════════════════════════════════════════════════
# LOAD FROM SESSION
# ══════════════════════════════════════════════════════════════════════════════
M      = st.session_state["ira_M"]
alpha  = st.session_state.get("ira_alpha", 0.05)
_cl    = f"{round((1 - alpha) * 100):g}%"   # confidence level label, e.g. "95%"

st.success("✓ Analysis complete!")

# ── Comprehensive Inter-Rater Summary ───────────────────────
st.markdown('<div class="sec-title">Inter-Rater Agreement Summary — All Groups</div>',
            unsafe_allow_html=True)

# Redesigned summary: per-group metric cards, not one wide table
for _grp_res in M.get("constructs",[]):
    _gname   = _grp_res.get("construct_name","—")
    _gdtype  = _grp_res.get("data_type","—").title()
    _gnunits = _grp_res.get("n_units",0)
    _gk      = _grp_res.get("k_raters",0)
    _gappl   = _grp_res.get("applicable_methods",[])

    st.markdown(
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
        f'border-radius:8px;padding:14px 18px;margin-bottom:12px;">'
        f'<div style="font-size:.95rem;font-weight:700;color:#1a1a2e;'
        f'margin-bottom:8px;">&#128202; {_gname}</div>'
        f'<div style="font-size:.78rem;color:#64748b;margin-bottom:10px;">'
        f'Data type: <b>{_gdtype}</b> &nbsp;|&nbsp; '
        f'{_gk} raters &nbsp;|&nbsp; {_gnunits} units &nbsp;|&nbsp; '
        f'Methods: {", ".join(_gappl)}</div>',
        unsafe_allow_html=True)

    # Build metric pairs for applicable methods
    _metrics = []
    _pa = (_grp_res.get("percentage_agreement") or {}).get("pa")
    if _pa is not None and not (isinstance(_pa,float) and np.isnan(_pa)):
        _metrics.append(("% Agreement", f"{_pa*100:.1f}%", "baseline"))

    _ck = (_grp_res.get("cohen_kappa") or {}).get("kappa")
    if _ck is not None and not (isinstance(_ck,float) and np.isnan(_ck)):
        _ck_lbl = (_grp_res.get("cohen_kappa") or {}).get("kappa_label","—")
        _metrics.append(("Cohen's κ", f"{_ck:.5f}", _ck_lbl))

    _ac1 = (_grp_res.get("gwet_ac1") or {}).get("ac1")
    if _ac1 is not None and not (isinstance(_ac1,float) and np.isnan(_ac1)):
        _ac1_lbl = (_grp_res.get("gwet_ac1") or {}).get("label","—")
        _metrics.append(("Gwet's AC1", f"{_ac1:.5f}", _ac1_lbl))

    _sp = (_grp_res.get("scott_pi") or {}).get("pi")
    if _sp is not None and not (isinstance(_sp,float) and np.isnan(_sp)):
        _metrics.append(("Scott's π", f"{_sp:.5f}",
                         (_grp_res.get("scott_pi") or {}).get("label","—")))

    _bp2 = (_grp_res.get("brennan_prediger") or {}).get("kappa_bp")
    if _bp2 is not None and not (isinstance(_bp2,float) and np.isnan(_bp2)):
        _metrics.append(("Brennan-Prediger", f"{_bp2:.5f}",
                         (_grp_res.get("brennan_prediger") or {}).get("label","—")))

    _fk = (_grp_res.get("fleiss_kappa") or {}).get("kappa")
    if _fk is not None and not (isinstance(_fk,float) and np.isnan(_fk)):
        _metrics.append(("Fleiss' κ", f"{_fk:.5f}",
                         (_grp_res.get("fleiss_kappa") or {}).get("kappa_label","—")))

    _wkq = (_grp_res.get("weighted_kappa") or {}).get("kappa_quadratic")
    if _wkq is not None and not (isinstance(_wkq,float) and np.isnan(_wkq)):
        _metrics.append(("Weighted κ (quad)", f"{_wkq:.5f}",
                         (_grp_res.get("weighted_kappa") or {}).get("kappa_quadratic_label","—")))

    _ac2_quad = ((_grp_res.get("gwet_ac2") or {}).get("schemes") or {}).get("quadratic") or {}
    _ac2 = _ac2_quad.get("ac2")
    if _ac2 is not None and not (isinstance(_ac2,float) and np.isnan(_ac2)):
        _metrics.append(("Gwet's AC2 (quadratic)", f"{_ac2:.5f}",
                         _ac2_quad.get("label","—")))

    _icc_r = _grp_res.get("icc") or {}
    _ia1 = _icc_r.get("icc_2_1") if "error" not in _icc_r else None
    if _ia1 is not None and not (isinstance(_ia1,float) and np.isnan(_ia1)):
        _metrics.append(("ICC(A,1) — primary", f"{_ia1:.5f}",
                         _icc_r.get("icc_2_1_label","—")))

    _ccc2 = (_grp_res.get("lin_ccc") or {}).get("ccc")
    if _ccc2 is not None and not (isinstance(_ccc2,float) and np.isnan(_ccc2)):
        _metrics.append(("Lin's CCC", f"{_ccc2:.5f}",
                         (_grp_res.get("lin_ccc") or {}).get("label","—")))

    _ka2 = (_grp_res.get("krippendorff") or {}).get("alpha")
    if _ka2 is not None and not (isinstance(_ka2,float) and np.isnan(_ka2)):
        _metrics.append(("Krippendorff's α", f"{_ka2:.5f}",
                         (_grp_res.get("krippendorff") or {}).get("label","—")))

    # Display in 3-column grid
    if _metrics:
        _ncols = min(3, len(_metrics))
        _cols = st.columns(_ncols)
        for _mi, (_lbl, _val, _interp) in enumerate(_metrics):
            with _cols[_mi % _ncols]:
                st.markdown(
                    f'<div style="background:white;border:1px solid #e2e8f0;'
                    f'border-radius:6px;padding:10px;text-align:center;'
                    f'margin-bottom:8px;">'
                    f'<div style="font-size:1.1rem;font-weight:700;color:#1a1a2e;">'
                    f'{_val}</div>'
                    f'<div style="font-size:.73rem;color:#64748b;font-weight:600;">'
                    f'{_lbl}</div>'
                    f'<div style="font-size:.68rem;color:#94a3b8;margin-top:2px;">'
                    f'{_interp}</div>'
                    f'</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Benchmark reference note
st.markdown(
    '<div style="background:#f0f9ff;box-shadow:inset 0 0 0 1px rgba(2,132,199,.22);'
    'padding:10px 14px;border-radius:8px;font-size:.77rem;color:#1e3a5f;'
    'margin-bottom:12px;">'
    '<b>Benchmark References:</b> '
    'Kappa family — Landis & Koch (1977): &lt;0 Poor; 0.00–0.20 Slight; '
    '0.21–0.40 Fair; 0.41–0.60 Moderate; 0.61–0.80 Substantial; '
    '0.81–1.00 Almost Perfect. '
    'ICC / CCC — Koo & Li (2016): &lt;0.50 Poor; 0.50–0.75 Moderate; '
    '0.75–0.90 Good; &gt;0.90 Excellent. '
    'Krippendorff (2004): α&ge;0.800 reliable; 0.667&le;α&lt;0.800 tentatively acceptable. '
    '"—" = method not applicable for this data type / rater count. '
    'See each group\'s tab below for full detail, visualisations, and interpretation.'
    '</div>',
    unsafe_allow_html=True)


ctabs = st.tabs([res["construct_name"] for res in M["constructs"]] + ["📝 Notes"])
for ti, res in enumerate(M["constructs"]):
    with ctabs[ti]:
        if res.get("error"):
            st.error(
                f"&#9888;&#65039; **{res.get('construct_name', f'Group {ti+1}')}** "
                f"could not be analysed: {res['error']}")
            continue

        # ── NEW: clean per-group display from run_multi_irar ─────────
        from modules.reliability import _get_applicable_methods
        name         = res.get("construct_name", f"Group {ti+1}")
        irar_dtype   = res.get("data_type","")
        r_names      = res.get("rater_names",[])
        applicable   = res.get("applicable_methods",[])
        n_units      = res.get("n_units",0)
        k_raters     = res.get("k_raters",0)
        method_ctx   = res.get("method_context",{})

        # Context banner — academic English, professional
        st.markdown(
            f'<div class="decision-banner use-param">&#128202; '
            f'<b>{irar_dtype.title()} data</b> &nbsp;|&nbsp; '
            f'<b>{k_raters} raters</b> &nbsp;|&nbsp; '
            f'<b>{n_units} units</b> &nbsp;|&nbsp; '
            f'<b>{len(applicable)} methods computed</b></div>',
            unsafe_allow_html=True)

        # Method context description
        if method_ctx:
            with st.expander("📋 Method Selection Rationale & Guidance"):
                st.markdown(
                    f'<div class="norm-rec-box">'
                    f'{method_ctx.get("methods","")}<br/><br/>'
                    f'<b>Guidance for this configuration:</b><br/>'
                    f'{method_ctx.get("guidance","")}'
                    f'</div>', unsafe_allow_html=True)

        # Method tabs
        method_tab_names = {
            "pct":          "% Agreement",
            "cohen":        "Cohen's κ",
            "gwet_ac1":     "Gwet's AC1",
            "scott_pi":     "Scott's π",
            "bp":           "Brennan-Prediger",
            "fleiss":       "Fleiss' κ",
            "weighted":     "Weighted κ",
            "gwet_ac2":     "Gwet's AC2",
            "icc":          "ICC",
            "ccc":          "Lin's CCC",
            "krippendorff": "Krippendorff's α",
        }
        tab_labels = [method_tab_names.get(m, m) for m in applicable]
        if not tab_labels:
            st.warning("No applicable methods for this configuration.")
        else:
            mtabs = st.tabs(tab_labels)
            for mi, mkey in enumerate(applicable):
                with mtabs[mi]:

                    # ── Percentage Agreement ──────────────────────────
                    if mkey == "pct":
                        pa_res = res.get("percentage_agreement") or {}
                        if not pa_res:
                            st.info("Percentage Agreement not computed.")
                        else:
                            _pa = pa_res.get("pa", np.nan)
                            _lo = pa_res.get("ci_lo", np.nan)
                            _hi = pa_res.get("ci_hi", np.nan)
                            try: _ci_pa=_fmt_ci(_lo, _hi)
                            except: _ci_pa="—"
                            c1,c2,c3=st.columns(3)
                            c1.metric("Percentage Agreement",
                                      f"{_pa*100:.1f}%" if not np.isnan(_pa) else "—")
                            c2.metric(f"{_cl} CI", _ci_pa)
                            c3.metric("N units / Raters",
                                      f"{n_units} / {k_raters}")
                            _total_pairs = pa_res.get("total_pairwise_comparisons",
                                                        n_units*k_raters*(k_raters-1)//2)
                            rows=[["Statistic","Value"],
                                  ["Percentage Agreement",
                                   f"{_pa*100:.2f}%" if not np.isnan(_pa) else "—"],
                                  ["As proportion", _f(_pa)],
                                  [f"{_cl} CI (unit-level t-interval)", _ci_pa],
                                  ["N (units)", str(n_units)],
                                  ["k (raters)", str(k_raters)],
                                  ["Total pairwise comparisons", str(_total_pairs)]]
                            st.markdown(html_tbl(rows,left_cols={0}),
                                        unsafe_allow_html=True)

                            # ── Per-unit descriptive statistics ───────
                            # Matches the variable a researcher would
                            # construct for descriptive-statistics analysis: one
                            # agreement proportion per rated unit, then
                            # Mean/SD/Min/Max across units.
                            _pu_mean = pa_res.get("mean", np.nan)
                            _pu_sd   = pa_res.get("sd", np.nan)
                            _pu_min  = pa_res.get("minimum", np.nan)
                            _pu_max  = pa_res.get("maximum", np.nan)
                            _n_valid = pa_res.get("n_units_valid", 0)
                            if _n_valid > 0:
                                st.markdown(
                                    '<div class="sub-title">Per-Unit Agreement — '
                                    'Descriptive Statistics</div>',
                                    unsafe_allow_html=True)
                                desc_rows = [["Statistic", "Value"],
                                    ["N (units with \u22651 valid pair)", str(_n_valid)],
                                    ["Minimum", _f(_pu_min)],
                                    ["Maximum", _f(_pu_max)],
                                    ["Mean", _f(_pu_mean)],
                                    ["Std. Deviation", _f(_pu_sd)]]
                                st.markdown(html_tbl(desc_rows, left_cols={0}),
                                            unsafe_allow_html=True)
                                _pu_gap = (abs(_pu_mean - _pa)
                                           if (not np.isnan(_pu_mean) and not np.isnan(_pa))
                                           else np.nan)
                                st.markdown(
                                    '<p class="note-txt">The Mean above is the average '
                                    'of PER-UNIT agreement proportions (one value per '
                                    'rated unit \u2014 the proportion of that unit\u2019s '
                                    'rater-pairs that agreed exactly), matching what a '
                                    'DESCRIPTIVES-style analysis on a per-unit agreement '
                                    'variable would report. It differs conceptually from '
                                    'the POOLED Percentage Agreement above (which pools '
                                    'all rater-pairs across all units into one '
                                    'proportion) but is numerically equal to it when '
                                    'every unit has the same number of valid rater-pairs '
                                    f'(here, gap = {_f(_pu_gap)}). '
                                    'A non-zero Std. Deviation indicates agreement is not '
                                    'uniform across units \u2014 the Minimum/Maximum show '
                                    'the range of unit-level agreement observed, which the '
                                    'single pooled proportion alone cannot reveal. '
                                    'Purpose: identifies whether disagreement is spread '
                                    'evenly across units or concentrated in a subset of '
                                    'problematic units. Reference: Stemler, S.E. (2004). '
                                    'Practical Assessment, Research &amp; Evaluation, 9(4).'
                                    '</p>', unsafe_allow_html=True)
                            # Visual: PA gauge (same standard as Krippendorff)
                            try:
                                _pagauge = _irar_pa_gauge(
                                    _pa, _lo, _hi,
                                    f"Percentage Agreement — {name}",
                                    ci_level=_cl)
                                if _pagauge:
                                    st.image(fig_to_bytes(_pagauge),
                                             width="stretch")
                                    plt.close("all")
                            except Exception:
                                pass
                            st.markdown(
                                '<p class="note-txt">Percentage Agreement does not correct '
                                'for chance agreement. It is reported as a descriptive '
                                'baseline only. Always report alongside a chance-corrected '
                                'coefficient (Cohen\'s κ, Gwet\'s AC1, or Krippendorff\'s α). '
                                'Reference: Stemler (2004). Practical Assessment, Research '
                                '&amp; Evaluation, 9(4).</p>',
                                unsafe_allow_html=True)
                            it=pa_res.get("interpretation_text","")
                            if it: st.markdown(f'<div class="interp-box sig">{it}</div>',
                                               unsafe_allow_html=True)

                    # ── Cohen's κ ────────────────────────────────────
                    elif mkey == "cohen":
                        ck = res.get("cohen_kappa") or {}
                        if not ck:
                            st.info("Cohen's κ not computed for this group.")
                        else:
                            variants = ck.get("variants", {})
                            _kv=ck.get("kappa",np.nan); _klbl=ck.get("kappa_label","—")
                            _kp=ck.get("kappa_p",np.nan)
                            c1,c2,c3=st.columns(3)
                            c1.metric("Cohen's κ (unweighted)",_f(_kv))
                            c2.metric(f"{_cl} CI (z-basis)",
                                      _fmt_ci(variants.get("unweighted",{}).get("ci_lo_z",np.nan),
                                              variants.get("unweighted",{}).get("ci_hi_z",np.nan)))
                            c3.metric("Sig.(2-tailed)",_p(_kp) if not np.isnan(_kp) else "—")

                            st.markdown(
                                f'<p class="note-txt">Rater 1: {ck.get("rater1","—")} | '
                                f'Rater 2: {ck.get("rater2","—")} | N: {ck.get("n",n_units)} | '
                                f'Categories: {ck.get("n_categories","—")}</p>',
                                unsafe_allow_html=True)

                            # ── Confusion matrix / Agreement Table ──
                            ct=ck.get("crosstab")
                            if ct is not None:
                                st.markdown('<div class="sub-title">Agreement Table</div>',
                                            unsafe_allow_html=True)
                                ct_rows=[[""] + [str(c) for c in ct.columns]]
                                for idx in ct.index:
                                    ct_rows.append([str(idx)]+[str(ct.loc[idx,c]) for c in ct.columns])
                                st.markdown(html_tbl(ct_rows,left_cols={0}),unsafe_allow_html=True)
                                r1n=r_names[0] if r_names else "Rater 1"
                                r2n=r_names[1] if len(r_names)>1 else "Rater 2"
                                hm=_irar_kappa_heatmap(ct, r1n, r2n,
                                                       f"Confusion Matrix — {name}")
                                if hm:
                                    st.image(fig_to_bytes(hm),width="stretch")
                                    plt.close("all")

                            # ── Kappa result (unweighted: Cohen's κ for nominal or
                            # dichotomous categories; weighted κ, which assumes
                            # ordered categories, is reported in the Weighted κ
                            # tab of ordinal rating groups) ──
                            st.markdown('<div class="sub-title">Kappa Result</div>',
                                        unsafe_allow_html=True)
                            v = variants.get("unweighted", {})
                            scheme_rows = [["Coefficient","κ","SE (null, Eq.9)","z","Sig.(2-tailed)",
                                            "SE (general, Eq.8)",f"{_cl} CI (z-basis)",
                                            f"{_cl} CI (t-basis, df={v.get('ci_df','—')})","Interpretation"],
                                           ["Cohen's κ (unweighted)",
                                            _f(v.get("kappa",np.nan)),
                                            _f(v.get("se_null",np.nan)),
                                            _f(v.get("z",np.nan)),
                                            _p(v.get("p",np.nan)) if not np.isnan(v.get("p",np.nan)) else "—",
                                            _f(v.get("se_general",np.nan)),
                                            _fmt_ci(v.get("ci_lo_z",np.nan), v.get("ci_hi_z",np.nan)),
                                            _fmt_ci(v.get("ci_lo_t",np.nan), v.get("ci_hi_t",np.nan)),
                                            v.get("label","—").split(" (")[0] if v.get("label") else "—"]]
                            st.markdown(html_tbl(scheme_rows,left_cols={0}),unsafe_allow_html=True)
                            st.markdown(
                                '<p class="note-txt">SE (null, Eq.9) is the H0: κ = 0 standard error '
                                'used for z and p; SE (general, Eq.8) is the large-sample standard '
                                'error of the estimate and is used for the confidence interval '
                                '(Fleiss, Cohen &amp; Everitt, 1969).</p>',
                                unsafe_allow_html=True)

                            # ── Agreement Breakdown ──
                            st.markdown('<div class="sub-title">Agreement Breakdown</div>',
                                        unsafe_allow_html=True)
                            st.markdown(f'<p class="note-txt">{ck.get("agreement_breakdown_text","")}</p>',
                                        unsafe_allow_html=True)

                            # ── Per-Category Agreement ──
                            per_cat = ck.get("per_category", [])
                            if per_cat:
                                st.markdown('<div class="sub-title">Per-Category Agreement</div>',
                                            unsafe_allow_html=True)
                                pc_rows = [["Category","N (Rater 1)","N (Rater 2)","Agree","Pct"]]
                                for row in per_cat:
                                    pc_rows.append([
                                        str(row["category"]), str(row["n_r1"]), str(row["n_r2"]),
                                        str(row["agree"]),
                                        f'{row["pct"]:.1f}%' if not np.isnan(row["pct"]) else "—"])
                                st.markdown(html_tbl(pc_rows,left_cols={0}),unsafe_allow_html=True)
                                st.markdown(
                                    '<p class="note-txt">Pct = Agree / (N_Rater1 + N_Rater2 − Agree), '
                                    'a Jaccard-style (intersection-over-union) same-category '
                                    'concordance rate.</p>', unsafe_allow_html=True)

                            if ck.get("dtype") == "dichotomous":
                                st.markdown(
                                    f'<p class="note-txt"><b>Prevalence Index:</b> '
                                    f'{_f(ck.get("prevalence_index",np.nan))} &nbsp;|&nbsp; '
                                    f'<b>Bias Index:</b> {_f(ck.get("bias_index",np.nan))}</p>',
                                    unsafe_allow_html=True)

                            _ck_ci_note = ck.get("kappa_ci_note","")
                            if _ck_ci_note:
                                st.markdown(f'<p class="note-txt"><b>Standard errors and confidence interval:</b> {_ck_ci_note}</p>',unsafe_allow_html=True)

                            # ── Referensi lengkap ──
                            _ref_note = ck.get("formula_reference_note","")
                            if _ref_note:
                                st.markdown(
                                    f'<div class="sub-title">References</div>'
                                    f'<p class="note-txt">{_ref_note}</p>',
                                    unsafe_allow_html=True)

                            it=ck.get("interpretation_text","")
                            if it: st.markdown(f'<div class="interp-box sig">{it}</div>',
                                               unsafe_allow_html=True)

                    # Weighted Cohen's κ
                    elif mkey == "weighted":
                        wk = res.get("weighted_kappa") or {}
                        if not wk:
                            st.info("Weighted Cohen's κ not computed.")
                        else:
                            _kwq=wk.get("kappa_quadratic",np.nan)
                            _kwl=wk.get("kappa_linear",np.nan)
                            _ku=wk.get("kappa",np.nan)
                            _lbl=wk.get("kappa_quadratic_label","—")
                            c1,c2,c3=st.columns(3)
                            c1.metric("κ_w (quadratic — primary)",_f(_kwq))
                            c2.metric("κ_w (linear)",_f(_kwl))
                            c3.metric("κ (unweighted)",_f(_ku))
                            _wv = wk.get("variants", {})
                            rows=[["Statistic","Value","SE (null, Eq.9)","z","Sig.",
                                   "SE (general, Eq.8)",f"{_cl} CI (z-basis)",f"{_cl} CI (t-basis)"]]
                            for _wkey, _wlab in (("unweighted","κ (unweighted)"),
                                                 ("linear","κ_w (linear)"),
                                                 ("quadratic","κ_w (quadratic — primary)")):
                                _vv = _wv.get(_wkey, {})
                                rows.append([_wlab, _f(_vv.get("kappa",np.nan)),
                                             _f(_vv.get("se_null",np.nan)), _f(_vv.get("z",np.nan)),
                                             _p(_vv.get("p",np.nan)) if not np.isnan(_vv.get("p",np.nan)) else "—",
                                             _f(_vv.get("se_general",np.nan)),
                                             _fmt_ci(_vv.get("ci_lo_z",np.nan), _vv.get("ci_hi_z",np.nan)),
                                             _fmt_ci(_vv.get("ci_lo_t",np.nan), _vv.get("ci_hi_t",np.nan))])
                            st.markdown(html_tbl(rows,left_cols={0}),unsafe_allow_html=True)
                            # Visual: multi-value bar — quadratic primary on top
                            try:
                                _wk_vals = {
                                    "κ_w (quadratic) ★": float(_kwq) if not np.isnan(float(_kwq)) else np.nan,
                                    "κ_w (linear)":       float(_kwl) if not np.isnan(float(_kwl)) else np.nan,
                                    "κ (unweighted)":     float(_ku)  if not np.isnan(float(_ku))  else np.nan,
                                }
                                _wk_colors = ["#e94560","#0284c7","#94a3b8"]
                                _wfig, _wax = plt.subplots(figsize=(8.5, 2.5))
                                _wlbls = list(_wk_vals.keys())
                                _wdisp = [max(min(v,1.05),-0.22) if not np.isnan(v) else 0.0
                                          for v in _wk_vals.values()]
                                _wax.barh(_wlbls, _wdisp, color=_wk_colors,
                                          edgecolor="white", height=0.45, zorder=3)
                                for thr, col in [(0.40,"#f59e0b"),(0.60,"#0284c7"),(0.80,"#16a34a")]:
                                    _wax.axvline(thr, color=col, ls="--", lw=0.9, alpha=0.45, zorder=1)
                                _wax.axvline(0, color="#334155", lw=0.9, zorder=2)
                                _wax.set_xlim(min(-0.22, min(_wdisp)-0.05), 1.18)
                                _wax.set_xlabel("Kappa value", fontsize=9, color=PAL[0])
                                _wax.set_title(f"Weighted Cohen's κ — {name}  (★ primary)",
                                               fontsize=10, fontweight="bold", color=PAL[0])
                                for _bi, (_lk, _bv) in enumerate(zip(_wlbls, _wdisp)):
                                    _rv = list(_wk_vals.values())[_bi]
                                    if not np.isnan(_rv):
                                        _bx = _bv+0.04 if _bv>=0 else _bv-0.04
                                        _wax.text(_bx, _bi, f"{_rv:.3f}", va="center",
                                                  ha="left" if _bv>=0 else "right",
                                                  fontsize=9, fontweight="bold", color=PAL[0],
                                                  zorder=5, bbox=dict(boxstyle="round,pad=0.15",
                                                  fc="white", ec="none", alpha=0.90))
                                _wax.spines[["top","right"]].set_visible(False)
                                _wfig.patch.set_facecolor("white")
                                _wfig.tight_layout(pad=1.3)
                                st.image(fig_to_bytes(_wfig), width="stretch")
                                plt.close("all")
                            except Exception:
                                pass
                            st.markdown(f'<p class="note-txt"><b>Primary</b>: κ_w (quadratic) = {_f(_kwq)} — {_lbl}.<br/>'
                                        f'{wk.get("formula_note","")}</p>',unsafe_allow_html=True)
                            ct=wk.get("crosstab")
                            if ct is not None:
                                st.markdown('<div class="sub-title">Confusion Matrix</div>',unsafe_allow_html=True)
                                ct_rows=[[""] + [str(c) for c in ct.columns]]
                                for idx in ct.index:
                                    ct_rows.append([str(idx)]+[str(ct.loc[idx,c]) for c in ct.columns])
                                st.markdown(html_tbl(ct_rows,left_cols={0}),unsafe_allow_html=True)
                                hm=_irar_kappa_heatmap(ct,
                                    r_names[0] if r_names else "Rater 1",
                                    r_names[1] if len(r_names)>1 else "Rater 2",
                                    f"Confusion Matrix — {name}")
                                if hm: st.image(fig_to_bytes(hm),width="stretch"); plt.close("all")
                            it=wk.get("interpretation_text","")
                            if it: st.markdown(f'<div class="interp-box sig">{it}</div>',unsafe_allow_html=True)

                    # Fleiss' κ
                    elif mkey == "fleiss":
                        fk = res.get("fleiss_kappa") or {}
                        if not fk:
                            st.info("Fleiss' κ not computed.")
                        else:
                            _fk=fk.get("kappa",np.nan); _flbl=fk.get("kappa_label","—")
                            _fp2=fk.get("kappa_p",np.nan)
                            c1,c2,c3=st.columns(3)
                            c1.metric("Fleiss' κ",_f(_fk))
                            c2.metric(f"{_cl} CI (t-basis)",
                                      _fmt_ci(fk.get("ci_lo_t",np.nan), fk.get("ci_hi_t",np.nan)))
                            c3.metric("Sig.(2-tailed)",_p(_fp2) if not np.isnan(_fp2) else "—")
                            rows=[["Statistic","Value"],
                                  ["Fleiss' κ",_f(_fk)],
                                  ["P\u0304 (observed agreement)",_f(fk.get("P_bar",np.nan))],
                                  ["P\u0304_e (expected/chance agreement)",_f(fk.get("P_e",np.nan))],
                                  ["SE (null; Fleiss, Nee & Landis, 1979)",_f(fk.get("kappa_se",np.nan))],
                                  ["z",_f(fk.get("kappa_z",np.nan))],
                                  ["Sig.(2-tailed)",_p(_fp2) if not np.isnan(_fp2) else "—"],
                                  ["SE (general; Gwet, 2021)",_f(fk.get("kappa_se_general",np.nan))],
                                  [f"{_cl} CI (t-basis, df={fk.get('ci_df','—')})",
                                   _fmt_ci(fk.get("ci_lo_t",np.nan), fk.get("ci_hi_t",np.nan))],
                                  [f"{_cl} CI (z-basis)",
                                   _fmt_ci(fk.get("ci_lo_z",np.nan), fk.get("ci_hi_z",np.nan))],
                                  ["Interpretation",_flbl],
                                  ["N (units)",str(n_units)],["k (raters)",str(k_raters)]]
                            st.markdown(html_tbl(rows,left_cols={0}),unsafe_allow_html=True)
                            pc=fk.get("kappa_per_category",{})
                            if pc:
                                st.markdown('<div class="sub-title">Per-Category κ</div>',unsafe_allow_html=True)
                                pc_rows=[["Category","κ_j","p̂ (proportion)"]]
                                for cat,kv in pc.items():
                                    pv=fk.get("p_per_category",{}).get(cat,np.nan)
                                    pc_rows.append([str(cat),_f(kv),_f(pv)])
                                st.markdown(html_tbl(pc_rows,left_cols={0}),unsafe_allow_html=True)
                            st.markdown(f'<p class="note-txt">{fk.get("formula_note","")}</p>',unsafe_allow_html=True)
                            _fk_ci_note = fk.get("kappa_ci_note","")
                            if _fk_ci_note:
                                st.markdown(f'<p class="note-txt"><b>Standard errors and confidence interval:</b> {_fk_ci_note}</p>',unsafe_allow_html=True)
                            it=fk.get("interpretation_text","")
                            if it: st.markdown(f'<div class="interp-box sig">{it}</div>',unsafe_allow_html=True)

                    # ICC
                    elif mkey == "icc":
                        icc = res.get("icc") or {}
                        if "error" in icc:
                            st.error(f"⚠️ ICC: {icc['error']}")
                        elif not icc:
                            st.info("ICC not computed.")
                        else:
                            def _fv(v):
                                try: return _f(v) if not (isinstance(v,float) and np.isnan(v)) else "—"
                                except: return "—"
                            _p21=icc.get("icc_2_1",np.nan)
                            # Metric card labels kept short to prevent truncation
                            # Full label in the detail table below
                            _plbl_short = (icc.get("icc_2_1_label","—") or "—").split(" ")[0]
                            _p2k=icc.get("icc_2_k",np.nan)
                            c1,c2,c3=st.columns(3)
                            c1.metric("ICC(A,1) — Primary",_fv(_p21))
                            c2.metric("ICC(A,k)",_fv(_p2k))
                            c3.metric("Interpretation",_plbl_short)
                            rows=[["ICC Form","Description","ICC",f"{_cl} CI","p (F-test)"],
                                  ["ICC(1,1)",
                                   "One-way random, single rater",
                                   _fv(icc.get("icc_1_1","—")),
                                   _fmt_ci(icc.get("icc_1_1_ci_lo",np.nan),icc.get("icc_1_1_ci_hi",np.nan)),
                                   _fmt_p_dash(icc.get("icc_1_1_p",np.nan))],
                                  ["ICC(A,1) — PRIMARY",
                                   "Two-way random, absolute agreement, single rater",
                                   _fv(icc.get("icc_2_1")),
                                   _fmt_ci(icc.get("icc_2_1_ci_lo",np.nan),icc.get("icc_2_1_ci_hi",np.nan)),
                                   _fmt_p_dash(icc.get("icc_2_1_p",np.nan))],
                                  ["ICC(C,1)",
                                   "Two-way mixed, consistency, single rater",
                                   _fv(icc.get("icc_3_1")),
                                   _fmt_ci(icc.get("icc_3_1_ci_lo",np.nan),icc.get("icc_3_1_ci_hi",np.nan)),
                                   _fmt_p_dash(icc.get("icc_3_1_p",np.nan))],
                                  ["ICC(1,k)",
                                   f"One-way random, mean of {k_raters} raters",
                                   _fv(icc.get("icc_1_k","—")),
                                   _fmt_ci(icc.get("icc_1_k_ci_lo",np.nan),icc.get("icc_1_k_ci_hi",np.nan)),
                                   _fmt_p_dash(icc.get("icc_1_k_p",np.nan))],
                                  ["ICC(A,k)",
                                   f"Two-way random, absolute agreement, mean of {k_raters} raters",
                                   _fv(icc.get("icc_2_k")),
                                   _fmt_ci(icc.get("icc_2_k_ci_lo",np.nan),icc.get("icc_2_k_ci_hi",np.nan)),
                                   _fmt_p_dash(icc.get("icc_2_k_p",np.nan))],
                                  ["ICC(C,k)",
                                   f"Two-way mixed, consistency, mean of {k_raters} raters",
                                   _fv(icc.get("icc_3_k")),
                                   _fmt_ci(icc.get("icc_3_k_ci_lo",np.nan),icc.get("icc_3_k_ci_hi",np.nan)),
                                   _fmt_p_dash(icc.get("icc_3_k_p",np.nan))]]
                            st.markdown(html_tbl(rows,left_cols={0,1}),unsafe_allow_html=True)
                            _icc_fig = _irar_icc_bar(icc, f"ICC — {name}")
                            if _icc_fig:
                                st.image(fig_to_bytes(_icc_fig), width="stretch")
                                plt.close("all")
                            st.markdown(f'<p class="note-txt">{icc.get("formula_note","")}</p>',unsafe_allow_html=True)
                            st.markdown(
                                '<p class="note-txt"><b>Model in this context:</b> ICC here treats '
                                'the actual raters in this group as independent assessors of the '
                                'same rated units \u2014 this is agreement between distinct raters. '
                                '<b>This differs from the Internal Consistency module\u2019s ICC</b>, '
                                'where items (not raters) are treated as interchangeable measures of '
                                'a construct within respondents. <b>When to use this (Inter-Rater) '
                                'context vs the Internal Consistency context:</b> use this context '
                                'when your question is about agreement between raters; use the '
                                'Internal Consistency module when your question is about a scale\u2019s '
                                'item-level consistency. Both share identical underlying mathematics '
                                '(Shrout &amp; Fleiss, 1979) but answer different substantive '
                                'questions. <b>ICC(A,1)</b> is the primary estimate here because '
                                'absolute agreement (penalising systematic rater bias) is typically '
                                'the more relevant standard for inter-rater work.</p>',
                                unsafe_allow_html=True)
                            it=icc.get("interpretation_text","")
                            if it: st.markdown(f'<div class="interp-box sig">{it}</div>',unsafe_allow_html=True)

                    # Krippendorff's α
                    elif mkey == "krippendorff":
                        ka = res.get("krippendorff") or {}
                        if not ka:
                            st.info("Krippendorff's α not computed.")
                        else:
                            _kav=ka.get("alpha",np.nan); _klbl=ka.get("label","—")
                            _nb=ka.get("n_boot",1000)
                            _lo=ka.get("ci_lo",np.nan); _hi=ka.get("ci_hi",np.nan)
                            try: _kci=_fmt_ci(_lo, _hi)
                            except: _kci="—"
                            c1,c2,c3=st.columns(3)
                            c1.metric("Krippendorff's α",_f(_kav))
                            c2.metric(f"{_cl} CI ({_nb:,} resamples)",_kci)
                            c3.metric("Interpretation",_klbl.split(" (")[0] if _klbl else "—")
                            rows=[["Statistic","Value"],
                                  ["Krippendorff's α",_f(_kav)],
                                  [f"{_cl} CI ({_nb:,} bootstrap resamples)",_kci],
                                  ["Data type / metric",ka.get("data_type","—").title()],
                                  # α uses every unit with at least two ratings
                                  # (pairable values), which can exceed the
                                  # complete-case count used by the other tabs.
                                  ["N (units with ≥ 2 ratings)",str(ka.get("n_units", n_units))],
                                  ["k (raters)",str(k_raters)],
                                  ["Threshold — min acceptable","α ≥ 0.667 (Krippendorff, 2004)"],
                                  ["Threshold — reliable","α ≥ 0.800 (Krippendorff, 2004)"],
                                  ["Interpretation",_klbl]]
                            st.markdown(html_tbl(rows,left_cols={0}),unsafe_allow_html=True)
                            _kg = _irar_kripp_gauge(_kav, _klbl, f"Krippendorff's α — {name}")
                            if _kg:
                                st.image(fig_to_bytes(_kg), width="stretch")
                                plt.close("all")
                            st.markdown(f'<p class="note-txt">{ka.get("formula_note","")}</p>',unsafe_allow_html=True)
                            it=ka.get("interpretation_text","")
                            if it: st.markdown(f'<div class="interp-box sig">{it}</div>',unsafe_allow_html=True)

                    # ── Gwet's AC1 ────────────────────────────────────
                    elif mkey == "gwet_ac1":
                        ac1 = res.get("gwet_ac1") or {}
                        if not ac1 or ac1.get("ac1") is None:
                            st.info("Gwet's AC1 not computed.")
                        else:
                            _av=ac1.get("ac1",np.nan); _albl=ac1.get("label","—")
                            _ase=ac1.get("se",np.nan); _az=ac1.get("z",np.nan)
                            _ap=ac1.get("p",np.nan)
                            _alo_z=ac1.get("ci_lo_z",np.nan); _ahi_z=ac1.get("ci_hi_z",np.nan)
                            _alo_t=ac1.get("ci_lo_t",np.nan); _ahi_t=ac1.get("ci_hi_t",np.nan)
                            try: _aci_z=_fmt_ci(_alo_z, _ahi_z)
                            except: _aci_z="—"
                            try: _aci_t=_fmt_ci(_alo_t, _ahi_t)
                            except: _aci_t="—"
                            c1,c2,c3=st.columns(3)
                            c1.metric("Gwet's AC1",_f(_av))
                            c2.metric(f"{_cl} CI (z-basis)",_aci_z)
                            c3.metric("Interpretation",_albl.split(" (")[0] if _albl else "—")
                            rows=[["Statistic","Value"],
                                  ["Gwet's AC1",_f(_av)],
                                  ["SE (jackknife)",_f(_ase)],
                                  ["z",_f(_az)],
                                  ["p (2-tailed)",_p(_ap) if not np.isnan(_ap) else "—"],
                                  [f"{_cl} CI (z-basis)",_aci_z],
                                  [f"{_cl} CI (t-basis, df="+str(ac1.get("ci_df","—"))+")",_aci_t],
                                  ["N (units)",str(n_units)],
                                  ["k (raters)",str(k_raters)],
                                  ["q (categories)",str(ac1.get("q","—"))],
                                  ["Interpretation",_albl]]
                            st.markdown(html_tbl(rows,left_cols={0}),unsafe_allow_html=True)
                            # Visual: single bar
                            try:
                                _ac1fig = _irar_single_bar(_av, _albl, "Gwet's AC1", name)
                                if _ac1fig:
                                    st.image(fig_to_bytes(_ac1fig), width="stretch")
                                    plt.close("all")
                            except Exception: pass
                            st.markdown(f'<p class="note-txt">{ac1.get("formula_note","")}</p>',unsafe_allow_html=True)
                            if ac1.get("dual_ci_note"):
                                st.markdown(ac1["dual_ci_note"], unsafe_allow_html=True)
                            it=ac1.get("interpretation_text","")
                            if it: st.markdown(f'<div class="interp-box sig">{it}</div>',unsafe_allow_html=True)

                    # ── Scott's π ─────────────────────────────────────
                    elif mkey == "scott_pi":
                        sp = res.get("scott_pi") or {}
                        if not sp or np.isnan(sp.get("pi",np.nan)):
                            st.info("Scott's π not computed.")
                        else:
                            _sv2=sp.get("pi",np.nan); _slbl=sp.get("label","—")
                            _sse=sp.get("se",np.nan); _sz=sp.get("z",np.nan); _spp=sp.get("p",np.nan)
                            _slo_z=sp.get("ci_lo_z",np.nan); _shi_z=sp.get("ci_hi_z",np.nan)
                            _slo_t=sp.get("ci_lo_t",np.nan); _shi_t=sp.get("ci_hi_t",np.nan)
                            try: _sci_z=_fmt_ci(_slo_z, _shi_z)
                            except: _sci_z="—"
                            try: _sci_t=_fmt_ci(_slo_t, _shi_t)
                            except: _sci_t="—"
                            c1,c2,c3=st.columns(3)
                            c1.metric("Scott's π",_f(_sv2))
                            c2.metric(f"{_cl} CI (z-basis)",_sci_z)
                            c3.metric("Interpretation",_slbl.split(" (")[0] if _slbl else "—")
                            rows=[["Statistic","Value"],
                                  ["Scott's π",_f(_sv2)],
                                  ["P_o (observed)",_f(sp.get("Po",np.nan))],
                                  ["P_e (Scott averaged marginals)",_f(sp.get("Pe",np.nan))],
                                  ["SE under H0 (Fleiss, Nee & Landis, 1979; for z and p)",_f(sp.get("se_null",np.nan))],
                                  ["z",_f(_sz)],
                                  ["p (2-tailed)",_p(_spp) if not np.isnan(_spp) else "—"],
                                  ["SE, general (Gwet, 2021; for the CI)",_f(_sse)],
                                  [f"{_cl} CI (z-basis)",_sci_z],
                                  [f"{_cl} CI (t-basis, df="+str(sp.get("ci_df","—"))+")",_sci_t],
                                  ["N (units)",str(n_units)],
                                  ["Interpretation",_slbl]]
                            st.markdown(html_tbl(rows,left_cols={0}),unsafe_allow_html=True)
                            # Visual
                            try:
                                _spfig = _irar_single_bar(_sv2, _slbl, "Scott's π", name)
                                if _spfig:
                                    st.image(fig_to_bytes(_spfig), width="stretch")
                                    plt.close("all")
                            except Exception: pass
                            if sp.get("dual_ci_note"):
                                st.markdown(sp["dual_ci_note"], unsafe_allow_html=True)
                            it=sp.get("interpretation_text","")
                            if it: st.markdown(f'<div class="interp-box sig">{it}</div>',unsafe_allow_html=True)

                    # ── Brennan-Prediger ───────────────────────────────
                    elif mkey == "bp":
                        bp_r = res.get("brennan_prediger") or {}
                        if not bp_r or np.isnan(bp_r.get("kappa_bp",np.nan)):
                            st.info("Brennan-Prediger coefficient not computed.")
                        else:
                            _bv=bp_r.get("kappa_bp",np.nan); _blbl=bp_r.get("label","—")
                            _bse=bp_r.get("se",np.nan); _bz=bp_r.get("z",np.nan)
                            _bpp=bp_r.get("p",np.nan)
                            _blo_z=bp_r.get("ci_lo_z",np.nan); _bhi_z=bp_r.get("ci_hi_z",np.nan)
                            _blo_t=bp_r.get("ci_lo_t",np.nan); _bhi_t=bp_r.get("ci_hi_t",np.nan)
                            try: _bci_z=_fmt_ci(_blo_z, _bhi_z)
                            except: _bci_z="—"
                            try: _bci_t=_fmt_ci(_blo_t, _bhi_t)
                            except: _bci_t="—"
                            c1,c2,c3=st.columns(3)
                            c1.metric("Brennan-Prediger",_f(_bv))
                            c2.metric(f"{_cl} CI (z-basis)",_bci_z)
                            c3.metric("Interpretation",_blbl.split(" (")[0] if _blbl else "—")
                            rows=[["Statistic","Value"],
                                  ["Brennan-Prediger (κ_BP)",_f(_bv)],
                                  ["P_o (observed agreement)",_f(bp_r.get("Po",np.nan))],
                                  ["P_e (= 1/q — uniform)",_f(bp_r.get("Pe",np.nan))],
                                  ["q (categories)",str(bp_r.get("q","—"))],
                                  ["SE",_f(_bse)],["z",_f(_bz)],
                                  ["p (2-tailed)",_p(_bpp) if not np.isnan(_bpp) else "—"],
                                  [f"{_cl} CI (z-basis)",_bci_z],
                                  [f"{_cl} CI (t-basis, df="+str(bp_r.get("ci_df","—"))+")",_bci_t],
                                  ["N (units)",str(n_units)],
                                  ["Interpretation",_blbl]]
                            st.markdown(html_tbl(rows,left_cols={0}),unsafe_allow_html=True)
                            try:
                                _bpfig = _irar_single_bar(_bv, _blbl, "Brennan-Prediger", name)
                                if _bpfig:
                                    st.image(fig_to_bytes(_bpfig), width="stretch")
                                    plt.close("all")
                            except Exception: pass
                            if bp_r.get("dual_ci_note"):
                                st.markdown(bp_r["dual_ci_note"], unsafe_allow_html=True)
                            _bp_cmp = bp_r.get("comparison_table")
                            if _bp_cmp:
                                st.markdown(
                                    '<div class="sub-title">Comparison with Other Coefficients</div>',
                                    unsafe_allow_html=True)
                                _cmp_rows = [["Coefficient", "Value", "Chance-Agreement Model"]]
                                for _cname, _cval, _cdesc in _bp_cmp:
                                    _cmp_rows.append([_cname, _f(_cval), _cdesc])
                                st.markdown(html_tbl(_cmp_rows, left_cols={0,2}),
                                            unsafe_allow_html=True)
                                if bp_r.get("comparison_note"):
                                    st.markdown(f'<p class="note-txt">{bp_r["comparison_note"]}</p>',
                                                unsafe_allow_html=True)
                            it=bp_r.get("interpretation_text","")
                            if it: st.markdown(f'<div class="interp-box sig">{it}</div>',unsafe_allow_html=True)

                    # ── Gwet's AC2 — all five weighting schemes ────────
                    elif mkey == "gwet_ac2":
                        ac2_all = res.get("gwet_ac2") or {}
                        _schemes = ac2_all.get("schemes", {})
                        _computed = {k: v for k, v in _schemes.items()
                                     if v.get("ac2") is not None and not np.isnan(v.get("ac2", np.nan))}
                        if not _computed:
                            st.info("Gwet's AC2 not computed.")
                        else:
                            st.markdown(
                                '<p class="note-txt">Gwet\u2019s AC2 is computed under '
                                '<b>all five standard weighting schemes</b> below, so the '
                                'sensitivity of the result to the choice of weighting '
                                'function is directly visible; no single scheme is '
                                'selected on the user\u2019s behalf.</p>',
                                unsafe_allow_html=True)
                            if any(v.get("has_gaps") for v in _computed.values()):
                                st.warning(
                                    "\u26a0 This group's rated values skip some integers between "
                                    "its lowest and highest value (e.g. a value like 65 never "
                                    "appears, even though lower and higher values do). Gwet's "
                                    "AC2 treats every integer in that range as a distinct, "
                                    "equally-spaced category, which fits a rating-scale/rubric "
                                    "(e.g. 1\u20135) but not a wide-range numeric score. If this "
                                    "group is really a raw score or other continuous-like "
                                    "measurement, consider declaring it Interval or Ratio "
                                    "instead, which use ICC/Lin's CCC and a distance-based "
                                    "Krippendorff's \u03b1 rather than a category count."
                                )
                            _scheme_order = ["quadratic", "linear", "radical", "ordinal", "identity"]
                            for _sk in _scheme_order:
                                if _sk not in _computed:
                                    continue
                                ac2 = _computed[_sk]
                                _a2v=ac2.get("ac2",np.nan); _a2lbl=ac2.get("label","—")
                                _a2se=ac2.get("se",np.nan); _a2z=ac2.get("z",np.nan)
                                _a2p=ac2.get("p",np.nan)
                                _a2lo_z=ac2.get("ci_lo_z",np.nan); _a2hi_z=ac2.get("ci_hi_z",np.nan)
                                _a2lo_t=ac2.get("ci_lo_t",np.nan); _a2hi_t=ac2.get("ci_hi_t",np.nan)
                                try: _a2ci_z=_fmt_ci(_a2lo_z, _a2hi_z)
                                except: _a2ci_z="—"
                                try: _a2ci_t=_fmt_ci(_a2lo_t, _a2hi_t)
                                except: _a2ci_t="—"
                                st.markdown(
                                    f'<div class="sub-title">{ac2.get("scheme_label", _sk.title())}</div>',
                                    unsafe_allow_html=True)
                                c1,c2,c3=st.columns(3)
                                c1.metric(f"Gwet's AC2 ({_sk})",_f(_a2v))
                                c2.metric(f"{_cl} CI (z-basis)",_a2ci_z)
                                c3.metric("Interpretation",_a2lbl.split(" (")[0] if _a2lbl else "—")
                                rows=[["Statistic","Value"],
                                      [f"Gwet's AC2 ({ac2.get('scheme_label','')})",_f(_a2v)],
                                      ["SE (jackknife)",_f(_a2se)],
                                      ["z",_f(_a2z)],
                                      ["p (2-tailed)",_p(_a2p) if not np.isnan(_a2p) else "—"],
                                      [f"{_cl} CI (z-basis)",_a2ci_z],
                                      [f"{_cl} CI (t-basis, df="+str(ac2.get("ci_df","—"))+")",_a2ci_t],
                                      ["N (units)",str(n_units)],
                                      ["k (raters)",str(k_raters)],
                                      ["q (categories)",str(ac2.get("q","—"))],
                                      ["Weight function",ac2.get("weight_formula","—")],
                                      ["Interpretation",_a2lbl]]
                                st.markdown(html_tbl(rows,left_cols={0}),unsafe_allow_html=True)
                                if ac2.get("when_to_use"):
                                    st.markdown(
                                        f'<p class="note-txt"><b>When to use this scheme:</b> '
                                        f'{ac2["when_to_use"]}</p>', unsafe_allow_html=True)
                                st.markdown(f'<p class="note-txt">{ac2.get("formula_note","")}</p>',unsafe_allow_html=True)
                                if ac2.get("dual_ci_note"):
                                    st.markdown(ac2["dual_ci_note"], unsafe_allow_html=True)
                                it=ac2.get("interpretation_text","")
                                if it: st.markdown(f'<div class="interp-box sig">{it}</div>',unsafe_allow_html=True)
                                st.markdown("---")


                    # ── Lin's CCC ─────────────────────────────────────
                    elif mkey == "ccc":
                        ccc_r = res.get("lin_ccc") or {}
                        if not ccc_r or ccc_r.get("ccc") is None:
                            st.info("Lin's CCC not computed.")
                        else:
                            _cv=ccc_r.get("ccc",np.nan); _clbl=ccc_r.get("label","—")
                            _crho=ccc_r.get("rho",np.nan); _cCb=ccc_r.get("Cb",np.nan)
                            _clo=ccc_r.get("ci_lo",np.nan); _chi2=ccc_r.get("ci_hi",np.nan)
                            try: _cci=_fmt_ci(_clo, _chi2)
                            except: _cci="—"
                            c1,c2,c3=st.columns(3)
                            c1.metric("Lin's CCC (ρ_c)",_f(_cv))
                            c2.metric(f"{_cl} CI (Fisher z)",_cci)
                            c3.metric("Interpretation",_clbl.split(" (")[0] if _clbl else "—")
                            rows=[["Statistic","Value","Interpretation"],
                                  ["CCC (ρ_c) — overall concordance",_f(_cv),_clbl],
                                  ["Pearson r (ρ) — precision",_f(_crho),
                                   "How correlated the two raters are"],
                                  ["r² (r-squared)",_f(_crho**2) if not np.isnan(_crho) else "—",""],
                                  ["Bias correction factor (C_b) — accuracy",
                                   _f(_cCb) if not (isinstance(_cCb,float) and np.isnan(_cCb)) else "—",
                                   "1.0 = no bias; < 1 = systematic bias present"],
                                  ["Mean (Rater/Method 1)",_f(ccc_r.get("mean1",np.nan)),""],
                                  ["Mean (Rater/Method 2)",_f(ccc_r.get("mean2",np.nan)),""],
                                  ["Mean difference",_f(ccc_r.get("mean_diff",np.nan)),""],
                                  ["SD (Rater/Method 1)",_f(ccc_r.get("sd1",np.nan)),""],
                                  ["SD (Rater/Method 2)",_f(ccc_r.get("sd2",np.nan)),""],
                                  ["Location shift (u)",_f(ccc_r.get("u_shift",np.nan)),
                                   "Mean gap relative to pooled scale"],
                                  ["Scale shift (v)",_f(ccc_r.get("v_shift",np.nan)),
                                   "Ratio of the two SDs"],
                                  [f"{_cl} CI (Fisher z-transform)",_cci,""],
                                  ["N (units)",str(n_units),""],
                                  ["|ρ_c| ≤ |Pearson r|",
                                   ("Satisfied" if abs(_cv) <= abs(_crho) + 1e-12 else "Not satisfied")
                                   if not (np.isnan(_cv) or np.isnan(_crho)) else "—",
                                   "Gap = extent of systematic bias"]]
                            st.markdown(html_tbl(rows,left_cols={0}),unsafe_allow_html=True)
                            # Visual: precision vs accuracy decomposition
                            try:
                                _cfig,_cax=plt.subplots(figsize=(7,2.5))
                                _vals=[max(min(float(v),1.0),-0.2) if not (isinstance(v,float) and np.isnan(v)) else 0
                                       for v in [_cv,_crho,_cCb]]
                                _labels=["CCC ρ_c\n(overall)","Pearson r\n(precision)","C_b\n(accuracy)"]
                                _cols=["#e94560","#0284c7","#16a34a"]
                                _cax.barh(_labels,_vals,color=_cols,edgecolor="white",height=0.5)
                                # BUG FIX (2026-10-21): this loop unpacked each
                                # 3-element (threshold, color, label) tuple into
                                # only 2 loop variables (thr, col) -- a
                                # ValueError ("too many values to unpack") on
                                # every single run, silently swallowed by the
                                # broad `except Exception: pass` below, which
                                # discarded this ENTIRE visual (the whole bar
                                # chart, not just the threshold lines) with no
                                # error ever shown to the user. Confirmed by
                                # direct execution of the same unpacking
                                # pattern (ValueError reproduced), not merely
                                # inferred from reading. Fixed by unpacking all
                                # three elements, and the previously-unused
                                # qualitative label is now actually rendered
                                # (it was dead data before this fix).
                                for thr,col,qlbl in [(0.75,"#0284c7","Good"),(0.90,"#16a34a","Excellent")]:
                                    _cax.axvline(thr,color=col,ls="--",lw=0.9,alpha=0.7)
                                    _cax.text(thr,len(_labels)-0.1,f"≥{thr} ({qlbl})",ha="center",
                                              fontsize=7,color=col)
                                _cax.axvline(0,color=PAL[0],lw=0.8)
                                _cax.set_xlim(-0.1,1.10)
                                _cax.set_xlabel("Value",fontsize=9,color=PAL[0])
                                _cax.set_title(f"Lin's CCC — Precision vs Accuracy — {name}",
                                              fontsize=10,fontweight="bold",color=PAL[0])
                                for i,v in enumerate(_vals):
                                    _cax.text(v+0.02,i,f"{v:.3f}",va="center",fontsize=9,color=PAL[0])
                                _cax.spines[["top","right"]].set_visible(False)
                                _cfig.patch.set_facecolor("white"); _cfig.tight_layout()
                                st.image(fig_to_bytes(_cfig),width="stretch"); plt.close("all")
                            except Exception: pass
                            st.markdown(f'<p class="note-txt">{ccc_r.get("formula_note","")}</p>',unsafe_allow_html=True)
                            st.markdown(
                                f'<p class="note-txt"><b>{_cl} CI methodology:</b> Fisher-Z transformation '
                                'using the FULL large-sample variance formula from Lin (1989, Biometrics '
                                '45:255\u2013268; see also Lin, Hedayat, Sinha &amp; Yang, 2002, JASA '
                                '97(457):257\u2013270) \u2014 this accounts for both precision (r) AND the '
                                'location-shift/bias term (u) shown above, not a simplified '
                                'matched-means-only approximation.</p>', unsafe_allow_html=True)
                            it=ccc_r.get("interpretation_text","")
                            if it: st.markdown(f'<div class="interp-box sig">{it}</div>',unsafe_allow_html=True)


with ctabs[-1]:
    st.markdown('<div class="sec-title">Methodological Notes</div>',
                unsafe_allow_html=True)
    for title, txt in _methodological_notes(M, alpha):
        st.markdown(f'<div class="norm-rec-box"><b>{title}.</b> {txt}</div>',
                    unsafe_allow_html=True)
