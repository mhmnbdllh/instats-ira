"""
utils/tables.py
===============
HTML table builder for statistical output in Streamlit UI.
Returns an HTML string rendered via st.markdown(unsafe_allow_html=True).
"""

import html as _html


# ══════════════════════════════════════════════════════════════════════════════
# CORE TABLE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def html_tbl(rows, left_cols=None):
    """
    Build a formatted HTML table from a list of rows.

    Parameters
    ----------
    rows : list of list
        First row is the header. Subsequent rows are data.
    left_cols : set of int, optional
        Column indices to left-align. Default: {0}.

    Returns
    -------
    str : HTML string ready for st.markdown(unsafe_allow_html=True)

    DEAD CODE REMOVED (2026-10-21): this function used to also accept
    `center_cols` and `highlight_cols` parameters. Confirmed, by searching
    every call site of html_tbl() across the whole repo (views/
    interrateragreement.py and views/internalconsistency.py -- the only
    two callers), that NEITHER was ever passed anywhere: every single call
    uses only `left_cols`. They are removed rather than left in place,
    which also removes a latent bug they had: the if/elif chain checked
    left_cols first, so a column in BOTH left_cols (default {0}) and
    highlight_cols would have silently never received highlight styling
    -- moot now that the dead parameter is gone rather than fixed in code
    nothing calls.
    """
    if left_cols is None:
        left_cols = {0}

    # BUG FIX (2026-10-21): cell/header values were interpolated straight
    # into the HTML string with no escaping. Confirmed by direct test: a
    # category label like "<5 years" (a rating/category value a user's
    # own uploaded data could plausibly contain, e.g. an age bracket or a
    # "<some threshold" code) produced the literal fragment
    # "<td class=\"left\"><5 years</td>" -- malformed HTML, since "<5" is
    # parsed as the start of an (invalid) tag, not literal text. Row and
    # column labels throughout this app come from the user's own data
    # (rater column names, observed category values), so this was a real,
    # reachable rendering bug, not a hypothetical one. Checked every
    # html_tbl() call site in the repo first to confirm no caller relies
    # on passing deliberate HTML inside a cell (e.g. a <b> tag) that
    # escaping would now break -- none do; all intentional HTML markup in
    # this app is built in separate st.markdown() calls outside html_tbl.
    html = '<div class="stat-tbl-wrap"><table class="stat-tbl"><thead><tr>'
    for cell in rows[0]:
        html += f"<th>{_html.escape(str(cell))}</th>"
    html += "</tr></thead><tbody>"

    for row in rows[1:]:
        html += "<tr>"
        for i, v in enumerate(row):
            css = ' class="left"' if i in left_cols else ""
            html += f"<td{css}>{_html.escape(str(v))}</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    return html
