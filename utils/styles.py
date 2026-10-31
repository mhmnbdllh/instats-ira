"""
utils/styles.py
===============
Centralised CSS styles for the Reliability Toolkit (Internal Consistency
and Inter-Rater Agreement). All styles are defined as string constants
and injected via st.markdown().

Usage:
    from utils.styles import load_css
    st.markdown(load_css(), unsafe_allow_html=True)
"""


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE FONTS IMPORT
# ══════════════════════════════════════════════════════════════════════════════

FONTS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
"""


# ══════════════════════════════════════════════════════════════════════════════
# BASE
# ══════════════════════════════════════════════════════════════════════════════

BASE = """
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
section[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
}
button[kind="header"] { display: none !important; }
"""


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

HEADER = """
.main-hdr {
    position: relative;
    background: radial-gradient(1200px 400px at 85% -10%, rgba(233,69,96,.35), transparent 60%),
                linear-gradient(120deg, #0a0a0a 0%, #1a1a2e 45%, #16213e 100%);
    padding: 2.2rem 2.4rem;
    border-radius: 18px;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(15, 23, 42, .22);
    overflow: hidden;
}
.main-hdr::after {
    content: '';
    position: absolute;
    top: -40%; right: -10%;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(233,69,96,.25), transparent 70%);
    pointer-events: none;
}
.main-hdr h1 {
    color: #fff;
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 .4rem 0;
    letter-spacing: -.4px;
}
.main-hdr p {
    color: #aeb8d4;
    margin: 0;
    font-size: .95rem;
    max-width: 48ch;
}
.badge {
    display: inline-block;
    background: #e94560;
    color: #fff;
    font-size: .7rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-left: 8px;
    font-weight: 600;
    vertical-align: middle;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# HOMEPAGE CARDS
# ══════════════════════════════════════════════════════════════════════════════

CARDS = """
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.4rem;
    margin-bottom: 1.6rem;
}

/* ---- hub card: whole tile is clickable via a stretched st.page_link ----
   st.markdown() and st.page_link() each render into their own sibling
   "stElementContainer" inside the SAME stColumn -- an unclosed <div> from
   one markdown call does NOT actually wrap a later call's widget (each
   call is parsed as its own isolated HTML fragment). The real shared
   ancestor of the visible card markup and the page_link widget is the
   stColumn itself, so that is what gets position:relative + the hover
   lift, and the page_link is stretched to fill it with :has(). */
div[data-testid="stColumn"]:has(.hub-card-wrap) {
    position: relative;
    isolation: isolate;
    border-radius: 18px;
    overflow: hidden;
    transition: transform .25s ease, box-shadow .25s ease;
    box-shadow: 0 4px 16px rgba(15, 23, 42, .08);
}
div[data-testid="stColumn"]:has(.hub-card-wrap):hover {
    transform: translateY(-5px);
    box-shadow: 0 16px 36px rgba(15, 23, 42, .16);
}
div[data-testid="stColumn"]:has(.hub-card-wrap):hover .hub-card-cta .arrow {
    transform: translateX(4px);
}
.hub-card-wrap {
    border-radius: 18px;
    overflow: hidden;
}
.hub-card {
    padding: 1.7rem 1.8rem 1.4rem;
    color: #fff;
    min-height: 210px;
    display: flex;
    flex-direction: column;
    position: relative;
}
.hub-card.ic {
    background: radial-gradient(900px 300px at 110% -20%, rgba(255,255,255,.18), transparent 60%),
                linear-gradient(145deg, #0f766e 0%, #0891b2 55%, #0e7490 100%);
}
.hub-card.ira {
    background: radial-gradient(900px 300px at 110% -20%, rgba(255,255,255,.18), transparent 60%),
                linear-gradient(145deg, #6d28d9 0%, #9333ea 55%, #c026d3 100%);
}
.hub-card-icon {
    width: 46px; height: 46px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(255,255,255,.16);
    border-radius: 12px;
    font-size: 1.4rem;
    margin-bottom: .9rem;
    backdrop-filter: blur(2px);
}
.hub-card-title {
    color: #fff;
    font-size: 1.08rem;
    font-weight: 700;
    margin: 0 0 .5rem;
    letter-spacing: -.2px;
}
.hub-card p {
    color: rgba(255,255,255,.88);
    font-size: .83rem;
    margin: 0 0 1rem;
    line-height: 1.65;
    flex-grow: 1;
}
.hub-card .nonparam-tag {
    display: inline-block;
    background: rgba(255,255,255,.16);
    color: #fff;
    font-size: .68rem;
    padding: 3px 9px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    margin: 0 .3rem .3rem 0;
}
.hub-card-cta {
    display: flex;
    align-items: center;
    gap: .4rem;
    font-size: .8rem;
    font-weight: 700;
    color: #fff;
    margin-top: .8rem;
    opacity: .95;
}
.hub-card-cta .arrow { display: inline-block; transition: transform .2s ease; }

/* the real navigation control: an invisible st.page_link stretched over
   the whole stColumn (which also contains the card markup above as a
   sibling), so the whole tile -- not just a small button -- is the
   click target, while still using Streamlit's own page-routing.

   Streamlit itself puts `position: relative` on the stPageLink's own
   immediate parent (stElementContainer), which becomes the CSS
   "containing block" for an absolutely-positioned child -- so without
   this next rule, `width/height: 100%` below would resolve against
   that tiny, content-sized wrapper (it shrinks to the link's own
   collapsed size) instead of against the stColumn we actually want.
   Neutralizing position on *only* that one wrapper (scoped narrowly to
   the wrapper that itself contains a stPageLink, inside one of our two
   hub-card columns) lets the absolute positioning skip past it to the
   next positioned ancestor up the chain, which is the stColumn
   targeted by the rule just above. */
div[data-testid="stColumn"]:has(.hub-card-wrap) div[data-testid="stElementContainer"]:has([data-testid="stPageLink"]) {
    position: static !important;
}
div[data-testid="stColumn"]:has(.hub-card-wrap) [data-testid="stPageLink"] {
    position: absolute !important;
    top: 0 !important; left: 0 !important;
    width: 100% !important; height: 100% !important;
    margin: 0 !important;
    z-index: 5;
}
div[data-testid="stColumn"]:has(.hub-card-wrap) [data-testid="stPageLink"] > div,
div[data-testid="stColumn"]:has(.hub-card-wrap) [data-testid="stPageLink"] a,
div[data-testid="stColumn"]:has(.hub-card-wrap) [data-testid="stPageLink"] span {
    display: block !important;
    width: 100% !important; height: 100% !important;
    margin: 0 !important;
}
div[data-testid="stColumn"]:has(.hub-card-wrap) [data-testid="stPageLink"] a {
    opacity: 0;
}

/* ---- usage-guide box below the two hub cards ---- */
.guide-box {
    background: #fff;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, .05);
}
.guide-box-hdr {
    display: flex;
    align-items: center;
    gap: .6rem;
    margin-bottom: .3rem;
}
.guide-box-hdr .ic {
    width: 34px; height: 34px;
    display: flex; align-items: center; justify-content: center;
    background: #f1f5f9;
    border-radius: 9px;
    font-size: 1.05rem;
}
.guide-box-title {
    margin: 0;
    font-size: 1rem;
    font-weight: 700;
    color: #1a1a2e;
}
.guide-box-sub {
    color: #64748b;
    font-size: .82rem;
    margin: 0 0 .4rem;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION PANEL
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = """
.cfg-panel {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, .05);
}
.cfg-panel h3 {
    margin: 0 0 1rem;
    font-size: .95rem;
    font-weight: 700;
    color: #1a1a2e;
    border-bottom: 2px solid #e94560;
    padding-bottom: .5rem;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION TITLES & SUB-TITLES
# ══════════════════════════════════════════════════════════════════════════════

TITLES = """
.sec-title {
    background: linear-gradient(90deg, #1a1a2e, #16213e);
    color: #e2e8f0;
    padding: 9px 16px;
    border-radius: 6px 6px 0 0;
    font-weight: 600;
    font-size: .84rem;
    letter-spacing: .6px;
    margin-top: 1.4rem;
    font-family: 'DM Mono', monospace;
    border-bottom: 2px solid #e94560;
}
.sub-title {
    background: #f1f5f9;
    color: #1a1a2e;
    padding: 7px 14px 7px 28px;
    border-radius: 6px;
    font-weight: 600;
    font-size: .82rem;
    margin: 1rem 0 .4rem;
    position: relative;
}
.sub-title::before {
    content: "";
    position: absolute;
    left: 12px;
    top: 50%;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #e94560;
    transform: translateY(-50%);
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL OUTPUT TABLES
# ══════════════════════════════════════════════════════════════════════════════

TABLES = """
.stat-tbl-wrap {
    overflow-x: auto;
    margin-bottom: .4rem;
}
.stat-tbl {
    font-family: 'DM Mono', monospace;
    font-size: .77rem;
    border-collapse: collapse;
    width: 100%;
    min-width: 400px;
}
.stat-tbl th {
    background: #1a1a2e;
    color: #e2e8f0;
    padding: 7px 12px;
    text-align: center;
    font-weight: 600;
    border: 1px solid #334155;
    font-size: .74rem;
    white-space: nowrap;
}
.stat-tbl td {
    padding: 5px 12px;
    border: 1px solid #e2e8f0;
    text-align: right;
    color: #1e293b;
    white-space: nowrap;
    background: #fff;
}
.stat-tbl tr:nth-child(even) td {
    background: #f8fafc;
}
.stat-tbl td.left {
    text-align: left;
    font-weight: 500;
    background: #f1f5f9 !important;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# METRIC CARDS
# ══════════════════════════════════════════════════════════════════════════════

METRICS = """
.metric-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: .9rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0, 0, 0, .05);
}
.metric-val {
    font-size: 1.45rem;
    font-weight: 700;
    color: #1a1a2e;
    font-family: 'DM Mono', monospace;
}
.metric-lbl {
    font-size: .7rem;
    color: #64748b;
    margin-top: .25rem;
    text-transform: uppercase;
    letter-spacing: .6px;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# DECISION BANNERS
# ══════════════════════════════════════════════════════════════════════════════

BANNERS = """
.decision-banner {
    padding: 1rem 1.4rem;
    border-radius: 10px;
    margin: 1rem 0;
    font-size: .9rem;
    font-weight: 600;
}
.decision-banner.use-param {
    background: #dcfce7;
    color: #14532d;
    border: 1px solid #86efac;
}
.decision-banner.use-nonparam {
    background: #ffedd5;
    color: #7c2d12;
    border: 1px solid #fdba74;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION BOXES
# ══════════════════════════════════════════════════════════════════════════════

INTERP = """
.interp-box {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    box-shadow: inset 0 0 0 1px rgba(2, 132, 199, .18);
    padding: 1rem 1.2rem;
    border-radius: 10px;
    margin: .7rem 0;
    font-size: .87rem;
    line-height: 1.8;
    color: #1e293b;
}
.interp-box b { color: #0284c7; }
.interp-box.sig {
    box-shadow: inset 0 0 0 1px rgba(22, 163, 74, .22);
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
}
.interp-box.sig b { color: #16a34a; }
"""


# ══════════════════════════════════════════════════════════════════════════════
# ALERT & INFO BOXES
# ══════════════════════════════════════════════════════════════════════════════

ALERTS = """
.warn-box {
    background: #fffbeb;
    box-shadow: inset 0 0 0 1px rgba(245, 158, 11, .32);
    padding: .7rem 1rem;
    border-radius: 8px;
    font-size: .82rem;
    color: #92400e;
    margin: .4rem 0;
}
.info-box {
    background: #eff6ff;
    box-shadow: inset 0 0 0 1px rgba(59, 130, 246, .28);
    padding: .7rem 1rem;
    border-radius: 8px;
    font-size: .82rem;
    color: #1e40af;
    margin: .4rem 0;
}
.norm-rec-box {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    box-shadow: inset 0 0 0 1px rgba(2, 132, 199, .24);
    padding: .8rem 1rem;
    border-radius: 10px;
    font-size: .82rem;
    color: #0c4a6e;
    margin: .5rem 0;
    line-height: 1.7;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

TEXT = """
.note-txt {
    font-size: .74rem;
    color: #64748b;
    font-style: italic;
    margin-top: .3rem;
    line-height: 1.6;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOADER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def load_css():
    """
    Return the complete CSS string for injection via st.markdown().

    Usage:
        import streamlit as st
        from utils.styles import load_css
        st.markdown(load_css(), unsafe_allow_html=True)
    """
    all_css = (
        FONTS
        + BASE
        + HEADER
        + CARDS
        + CONFIG
        + TITLES
        + TABLES
        + METRICS
        + BANNERS
        + INTERP
        + ALERTS
        + TEXT
    )
    return f"<style>{all_css}</style>"


def load_page_css():
    """
    Return CSS for the analysis pages only (excludes homepage card
    styles). Use this in views/internalconsistency.py and
    views/interrateragreement.py.
    """
    page_css = (
        FONTS
        + BASE
        + HEADER
        + CONFIG
        + TITLES
        + TABLES
        + METRICS
        + BANNERS
        + INTERP
        + ALERTS
        + TEXT
    )
    return f"<style>{page_css}</style>"
