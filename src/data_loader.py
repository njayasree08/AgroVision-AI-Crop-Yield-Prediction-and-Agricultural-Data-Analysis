"""Loading and validating datasets (built-in file or user upload).

Uploaded files are only ever parsed as CSV text with pandas. Nothing in an
uploaded file is executed or unpickled.
"""
from __future__ import annotations

import pandas as pd

from .config import (DATA_PATH, MAX_UPLOAD_MB, MAX_UPLOAD_ROWS, MIN_ROWS,
                     NUMERIC_FEATURES, REG_TARGET, REQUIRED_COLUMNS)


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Read the built-in dataset; generate it first if it does not exist yet."""
    if not path.exists():
        from .data_generator import main as generate
        generate()
    return pd.read_csv(path)


def read_uploaded_csv(file_obj) -> pd.DataFrame:
    """Safely parse an uploaded CSV. Raises ValueError with a friendly message."""
    size = getattr(file_obj, "size", None)
    if size is not None and size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File is larger than the {MAX_UPLOAD_MB} MB limit.")
    try:
        df = pd.read_csv(file_obj, nrows=MAX_UPLOAD_ROWS + 1)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read the file as CSV: {exc}") from exc
    if len(df) > MAX_UPLOAD_ROWS:
        raise ValueError(f"File has more than {MAX_UPLOAD_ROWS:,} rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). An empty error list means the data can be used."""
    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required column(s): " + ", ".join(missing))
    if len(df) < MIN_ROWS:
        errors.append(f"At least {MIN_ROWS} rows are required (found {len(df)}).")

    for col in NUMERIC_FEATURES + [REG_TARGET]:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            bad = int(converted.isna().sum() - df[col].isna().sum())
            if bad > 0.3 * max(len(df), 1):
                errors.append(f"Column '{col}' must be numeric but {bad} values are not numbers.")
            elif bad > 0:
                warnings.append(f"Column '{col}' has {bad} non-numeric value(s); they will be treated as missing.")
    if len(df) and REG_TARGET in df.columns and pd.to_numeric(df[REG_TARGET], errors="coerce").isna().all():
        errors.append(f"'{REG_TARGET}' contains no usable numbers.")
    if MIN_ROWS <= len(df) < 100:
        warnings.append("Small dataset: model metrics will be unstable.")
    return errors, warnings
