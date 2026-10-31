"""
app.py
======
Entry point for the Reliability Toolkit. Shows a two-card landing page
and registers the two analyses as separate routed pages:
    /internalconsistency
    /interrateragreement
"""

import streamlit as st

from utils.styles import load_css


def home():
    st.set_page_config(
        page_title="Reliability Toolkit",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(load_css(), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="main-hdr">
          <h1>&#129514; Reliability Toolkit</h1>
          <p>Choose an analysis below to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card-grid">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="hub-card-wrap">
              <div class="hub-card ic">
                <div class="hub-card-icon">&#128202;</div>
                <div class="hub-card-title">Internal Consistency</div>
                <p>For scales made of multiple items measuring one construct.
                Computes Cronbach&#39;s &alpha;, KR-20 / KR-21, Split-Half
                reliability, Guttman&#39;s &lambda; (6 forms), McDonald&#39;s
                &omega; (model-implied, observed, and hierarchical), and
                ICC(C,k) &mdash; plus full item-level diagnostics.</p>
                <div>
                  <span class="nonparam-tag">Cronbach&#39;s &alpha;</span>
                  <span class="nonparam-tag">KR-20</span>
                  <span class="nonparam-tag">McDonald&#39;s &omega;</span>
                  <span class="nonparam-tag">ICC</span>
                </div>
                <div class="hub-card-cta">Open module <span class="arrow">&#8594;</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(
            "views/internalconsistency.py",
            label="Open Internal Consistency",
        )

    with col2:
        st.markdown(
            """
            <div class="hub-card-wrap">
              <div class="hub-card ira">
                <div class="hub-card-icon">&#129309;</div>
                <div class="hub-card-title">Inter-Rater Agreement
                  <span style="font-size:.6em;font-weight:600;color:#94a3b8;
                  vertical-align:middle;">(InStats-IrA)</span></div>
                <p>For agreement between two or more raters/judges scoring
                the same units. Computes Percentage Agreement, Cohen&#39;s /
                Fleiss&#39; &kappa;, Gwet&#39;s AC1 / AC2, Scott&#39;s &pi;,
                the Brennan-Prediger coefficient, ICC (all six forms),
                Lin&#39;s CCC, and Krippendorff&#39;s &alpha;.</p>
                <div>
                  <span class="nonparam-tag">Cohen&#39;s/Fleiss&#39; &kappa;</span>
                  <span class="nonparam-tag">Gwet&#39;s AC1/AC2</span>
                  <span class="nonparam-tag">Krippendorff&#39;s &alpha;</span>
                </div>
                <div class="hub-card-cta">Open module <span class="arrow">&#8594;</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(
            "views/interrateragreement.py",
            label="Open Inter-Rater Agreement",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    _render_instats_ira_box()

    _render_usage_guide()


def _render_instats_ira_box():
    st.markdown(
        """
        <div class="guide-box" style="margin-bottom:1.2rem;">
          <div class="guide-box-hdr">
            <div class="ic">&#129309;</div>
            <div class="guide-box-title">About InStats-IrA</div>
          </div>
          <p class="guide-box-sub" style="margin-bottom:0;">
          InStats-IrA is this toolkit&#39;s dedicated Inter-Rater Agreement
          module &mdash; a formula-documented implementation of eleven
          agreement coefficients (Percentage Agreement, Cohen&#39;s / Fleiss&#39;
          &kappa;, weighted &kappa;, Scott&#39;s &pi;, the Brennan-Prediger
          coefficient, Gwet&#39;s AC1 / AC2, ICC, Lin&#39;s CCC, and
          Krippendorff&#39;s &alpha;) spanning dichotomous, nominal, ordinal,
          interval, and ratio rating data with two or more raters. The app
          restricts each coefficient to the data types and rater counts for
          which it is statistically appropriate, reports a standard error or
          confidence interval wherever one is valid, and documents every
          formula with its primary-source citation in the results&#39; Notes
          tab &mdash; so the output can be read, checked, and cited directly
          in a methods write-up.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_usage_guide():
    st.markdown(
        """
        <div class="guide-box">
          <div class="guide-box-hdr">
            <div class="ic">&#128218;</div>
            <div class="guide-box-title">How to use this toolkit</div>
          </div>
          <p class="guide-box-sub">A quick walkthrough for either module &mdash;
          expand a section below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    g1, g2 = st.columns(2)

    with g1:
        with st.expander("📁  1. Prepare your data file"):
            st.markdown(
                "- Upload a **CSV** or **Excel (.xlsx/.xls)** file.\n"
                "- The **first row** of the file is always treated as the "
                "header (column names) and is never read as data.\n"
                "- **Internal Consistency:** each column is one scale "
                "**item**; each row is one **respondent**.\n"
                "- **Inter-Rater Agreement:** each column is one **rater**; "
                "each row is one **unit** being rated (an essay, a "
                "response, a case). All raters must score the same set "
                "of units.\n"
                "- Missing cells are allowed. In the Inter-Rater Agreement "
                "module, Krippendorff's alpha uses every unit rated by at "
                "least two raters; every other coefficient uses only the "
                "units rated by all selected raters (listwise deletion "
                "within the rating group)."
            )
        with st.expander("⚙️  2. Configure the analysis"):
            st.markdown(
                "- **Internal Consistency:** select the item columns for "
                "your scale and whether items are dichotomous (right/"
                "wrong) or polytomous (rated on a scale).\n"
                "- **Inter-Rater Agreement:** group your rater columns "
                "into one or more **rating groups** (use *Number of "
                "rating groups* in the sidebar if several independent "
                "sets of raters/aspects are in the same file), then pick "
                "each group's **data type** "
                "(dichotomous, nominal, ordinal, interval, or ratio) and "
                "**number of raters**.\n"
                "- When Gwet's AC1, AC2, or the Brennan-Prediger "
                "coefficient applies, you must declare the instrument's "
                "**full category count or scale range** (not just what "
                "happens to appear in your sample) &mdash; this is "
                "intentional: it keeps the coefficient tied to the true "
                "rating scale rather than to whichever categories this "
                "particular sample happened to use."
            )

    with g2:
        with st.expander("▶️  3. Run and read the results"):
            st.markdown(
                "- Click **Run Analysis**. The app automatically computes "
                "every coefficient that is theoretically appropriate for "
                "the data type and number of raters you selected &mdash; "
                "there is no coefficient to pick by hand.\n"
                "- Each result includes the point estimate, its standard "
                "error/confidence interval where one is statistically "
                "valid, and a plain-language interpretation banner.\n"
                "- Krippendorff's alpha's confidence interval is a "
                "percentile bootstrap; you can choose how many resamples "
                "to use (more resamples = smoother interval, slower "
                "computation) from the sidebar.\n"
                "- A **Notes** tab at the end of the results lists every "
                "formula and primary-source reference used, for citing "
                "in your own write-up."
            )
        with st.expander("🧪  4. Try it with sample data"):
            st.markdown(
                "- The Inter-Rater Agreement page has a **Try with sample "
                "data** panel in the sidebar with three illustrative, "
                "ready-to-use datasets (dichotomous, nominal/ordinal, and "
                "interval/ratio ratings, 86 units each) &mdash; pick one from "
                "the dropdown to load it directly, or download it first if "
                "you'd rather inspect or edit it before using your own "
                "data.\n"
                "- Uploaded data is held only in memory for your own "
                "session and is never written to a database or disk, and "
                "is discarded once your session ends &mdash; it is not "
                "retained or shared across sessions or users. Note this "
                "is an in-browser *display* of a server-side app: your "
                "data IS processed on the application server while a run "
                "is underway (as with any Streamlit app), it simply is "
                "not persisted anywhere afterward."
            )


home_page = st.Page(home, title="Home", icon="🏠", default=True)
ic_page = st.Page(
    "views/internalconsistency.py",
    title="Internal Consistency",
    icon="📊",
    url_path="internalconsistency",
)
ira_page = st.Page(
    "views/interrateragreement.py",
    title="Inter-Rater Agreement",
    icon="🤝",
    url_path="interrateragreement",
)

# Callable-based pages (like `home_page` here) have no file-path identity,
# so sub-pages cannot link back to them with st.page_link("app.py", ...).
# Stash the actual Page object in session_state so every page can retrieve
# the same object and pass it to st.page_link() directly.
st.session_state["_home_page"] = home_page

pg = st.navigation([home_page, ic_page, ira_page], position="hidden")
pg.run()
