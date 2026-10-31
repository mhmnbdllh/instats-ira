"""
utils/file_loader.py
====================
File loading utilities for CSV and Excel uploads.
Supports: .csv, .xlsx, .xls
Includes validation, preview metadata, and column type detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# SUPPORTED FORMATS
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")

EXTENSION_LABELS = {
    ".csv":  "CSV (Comma-Separated Values)",
    ".xlsx": "Excel Workbook (Office Open XML)",
    ".xls":  "Excel Workbook (Legacy Binary)",
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_file(f):
    """
    Load a CSV or Excel file uploaded via Streamlit file_uploader.

    Parameters
    ----------
    f : UploadedFile
        Streamlit UploadedFile object.

    Returns
    -------
    pd.DataFrame
        Loaded and lightly cleaned DataFrame.

    Raises
    ------
    ValueError
        If the file format is unsupported or the file cannot be parsed.
    """
    name = f.name.lower()

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(f)
        except Exception as e:
            raise ValueError(
                f"Could not parse CSV file '{f.name}'. "
                f"Please ensure the file uses comma delimiters and UTF-8 encoding. "
                f"Details: {e}"
            )

    elif name.endswith(".xlsx"):
        try:
            df = pd.read_excel(f, engine="openpyxl")
        except Exception as e:
            raise ValueError(
                f"Could not parse Excel file '{f.name}'. "
                f"Please ensure the file is a valid .xlsx workbook. "
                f"Details: {e}"
            )

    elif name.endswith(".xls"):
        try:
            df = pd.read_excel(f, engine="xlrd")
        except Exception as e:
            raise ValueError(
                f"Could not parse Excel file '{f.name}'. "
                f"Please ensure the file is a valid .xls workbook. "
                f"Details: {e}"
            )

    else:
        ext = name.rsplit(".", 1)[-1] if "." in name else "unknown"
        raise ValueError(
            f"Unsupported file format: '.{ext}'. "
            f"Accepted formats: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    df = _clean(df)
    return df


def load_sample_file(path):
    """
    Load one of the app's bundled sample CSV files directly from a local
    path (not from a Streamlit file_uploader object), through the SAME
    cleaning pipeline as an uploaded file, so a sample dataset behaves
    identically to a user's own upload once loaded.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a bundled sample CSV file on disk.

    Returns
    -------
    pd.DataFrame
        Loaded and lightly cleaned DataFrame.

    Raises
    ------
    ValueError
        If the file cannot be parsed.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Could not parse sample file '{path}'. Details: {e}")
    return _clean(df)


# ══════════════════════════════════════════════════════════════════════════════
# CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def _clean(df):
    """
    Apply light cleaning to a freshly loaded DataFrame:
    - Strip leading/trailing whitespace from string column names
    - Strip leading/trailing whitespace from string cell values
    - Drop completely empty rows and columns
    - Attempt to coerce object columns that look numeric to float

    BUG FIX (2026-10-21): the original order of operations here was
    (1) dropna(how="all") on rows/columns, THEN (2) strip whitespace from
    string cells, THEN (3) numeric-coerce. That order missed a whole
    class of effectively-blank rows: a row whose cells are all
    whitespace-only strings (e.g. "   ,   ,   " -- a common artifact of
    a trailing blank line or a spreadsheet export quirk) is NOT NaN, so
    step (1)'s dropna never removed it; it only became a clean empty
    string "" once step (2) ran, by which point the row had already
    survived the "drop empty rows" step. Confirmed directly: a 4-row,
    3-numeric-column CSV with one such blank row produced
    get_numeric_cols(df) == [] (ALL THREE otherwise-clean numeric rater
    columns vanished from numeric-column detection) -- because the
    leftover "" values counted as "non-null" in step (3)'s 80% numeric-
    success threshold, dragging every column's measured success rate
    below the cutoff and leaving them as string dtype. In the app this
    silently empties the "Select rater columns" multiselect for a file
    that should have worked perfectly, with no error explaining why.
    Fixed by stripping whitespace and converting now-empty strings to
    real NaN BEFORE dropping empty rows/columns, so a whitespace-only
    row is correctly recognized and dropped as blank, and never pollutes
    the later numeric-coercion success-rate calculation.
    """
    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Strip string cells FIRST, and normalize now-empty strings to real
    # missing values, so a whitespace-only cell/row is treated as blank.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        df[col] = df[col].replace("", np.nan)

    # Drop fully empty rows / columns (now correctly catches whitespace-
    # only rows too, since they are real NaN by this point)
    df = df.dropna(how="all").reset_index(drop=True)
    df = df.dropna(axis=1, how="all")

    # Attempt numeric coercion for object columns
    for col in df.select_dtypes(include="object").columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        # Only convert if > 80% of non-null values are numeric
        non_null = df[col].notna().sum()
        if non_null > 0 and coerced.notna().sum() / non_null >= 0.80:
            df[col] = coerced

    return df


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_file(df, min_rows=3):
    """
    Validate a loaded DataFrame for minimum usability.

    Parameters
    ----------
    df : pd.DataFrame
    min_rows : int
        Minimum number of complete rows required.

    Returns
    -------
    dict with keys:
        'valid'    : bool
        'errors'   : list of str — blocking issues
        'warnings' : list of str — non-blocking notices
    """
    errors   = []
    warnings = []

    if df is None or df.empty:
        errors.append("The uploaded file is empty or could not be read.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if len(df) < min_rows:
        errors.append(
            f"The dataset contains only {len(df)} row(s). "
            f"A minimum of {min_rows} complete observations is required."
        )

    if len(df.columns) < 1:
        errors.append("The dataset contains no columns.")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) == 0:
        errors.append(
            "No numeric columns detected. "
            "At least one numeric variable is required for analysis."
        )

    # Warnings
    missing_pct = df.isnull().mean().mean() * 100
    if missing_pct > 0:
        warnings.append(
            f"The dataset contains missing values "
            f"({missing_pct:.1f}% of all cells). "
            "Each module's Notes tab states how missing values are handled "
            "by each coefficient."
        )

    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        warnings.append(
            f"{int(dup_rows)} duplicate row(s) detected. "
            "Please verify that these are legitimate observations "
            "and not the result of accidental double entry."
        )

    return {
        "valid":    len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
    }


# ══════════════════════════════════════════════════════════════════════════════
# METADATA
# ══════════════════════════════════════════════════════════════════════════════

def file_metadata(f, df):
    """
    Generate a metadata summary for a loaded file.

    Parameters
    ----------
    f   : UploadedFile
    df  : pd.DataFrame

    Returns
    -------
    dict with descriptive metadata for display.
    """
    name      = f.name
    ext       = "." + name.lower().rsplit(".", 1)[-1] if "." in name else ""
    fmt_label = EXTENSION_LABELS.get(ext, ext)
    size_kb   = round(f.size / 1024, 1) if hasattr(f, "size") else "N/A"

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    missing  = int(df.isnull().sum().sum())

    return {
        "File Name":         name,
        "Format":            fmt_label,
        "File Size":         f"{size_kb} KB",
        "Rows":              len(df),
        "Columns":           len(df.columns),
        "Numeric Columns":   len(num_cols),
        "Categorical Columns": len(cat_cols),
        "Missing Values":    missing,
        "Loaded at":         datetime.now().strftime("%B %d, %Y %H:%M"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN TYPE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_numeric_cols(df):
    """Return list of numeric column names."""
    return df.select_dtypes(include=np.number).columns.tolist()
