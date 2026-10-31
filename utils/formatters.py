"""
utils/formatters.py
===================
Shared numeric formatting utilities used by the reliability engine and
both views. Pure functions, no side effects.
"""

import numpy as np
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation


# ══════════════════════════════════════════════════════════════════════════════
# NUMERIC FORMATTERS
# ══════════════════════════════════════════════════════════════════════════════

def _f(v, d=3):
    """
    Format a numeric value to d decimal places.
    Returns '.' for None or NaN (SPSS convention).
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "."
    return f"{v:.{d}f}"


def _p(v, d=3):
    """
    Format a p-value to d decimal places (default 3, matching the
    conventional display precision for most modules).
    Uses standard rounding first (round half up).
    Shows '< .0001'-style threshold (one order below d) only when the
    rounded value is exactly zero at d decimals.
    Shows '.' for None or NaN.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "."
    try:
        v = float(v)
    except Exception:
        return "."
    # BUG FIX (2026-10-21): this docstring has always claimed "round half
    # up," but `round(v, d)` is Python's BUILT-IN round, which actually
    # uses round-HALF-TO-EVEN ("banker's rounding") -- confirmed directly
    # (round(0.5,0)=0.0, round(2.5,0)=2.0; a true round-half-up would give
    # 1.0 and 3.0). In practice this almost never changes a displayed
    # p-value, since a continuous test statistic landing exactly on a
    # decimal half-boundary is a measure-zero event -- but the documented
    # behavior and the actual behavior disagreed, so the code is now
    # fixed to match what it always claimed, using Decimal to round
    # correctly rather than silently weakening the docstring instead.
    try:
        rounded = float(Decimal(str(v)).quantize(
            Decimal('1.' + '0' * d) if d > 0 else Decimal('1'),
            rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, OverflowError):
        rounded = round(v, d)  # fallback for non-finite/pathological v
    if rounded == 0.0:
        thresh = "0" * (d - 1) + "1"
        return f"< .{thresh}"
    return f"{rounded:.{d}f}"
