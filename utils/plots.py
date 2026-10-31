"""
utils/plots.py
==============
Shared plotting utility. Converts a matplotlib Figure to PNG bytes for
Streamlit's st.image(). The actual figure-drawing functions live in each
view (views/internalconsistency.py, views/interrateragreement.py), since
they are specific to the coefficients each page reports.
"""

from io import BytesIO


def fig_to_bytes(fig, dpi=150):
    """
    Convert a matplotlib Figure to PNG bytes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    dpi : int

    Returns
    -------
    bytes
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()
