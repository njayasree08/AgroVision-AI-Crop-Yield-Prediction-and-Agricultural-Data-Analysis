"""Cleaning, target creation, encoding/scaling pipelines and train/test splitting.

Leakage rules followed in this project
--------------------------------------
* Row-level cleaning that uses only fixed rules (valid ranges, spelling) is done
  once, before splitting. Nothing is learned from the data there.
* Anything *learned* from data (imputation values, scaling statistics, category
  lists) lives inside a scikit-learn Pipeline and is fitted on the training
  split only.
* Crop_Yield and Yield_Category are never used as classification inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (CATEGORICAL_FEATURES, CATEGORY_SYNONYMS, CLASS_ORDER, CLS_TARGET,
                     DOMAIN_BOUNDS, FEATURES, ID_COLUMN, MIN_ROWS, NUMERIC_FEATURES,
                     RANDOM_SEED, REG_TARGET, TEST_SIZE)

_TEXT_COLUMNS = CATEGORICAL_FEATURES + ["State", "District"]


# ------------------------------------------------------------------ helpers
def split_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, non_numeric_columns)."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    other = [c for c in df.columns if c not in numeric]
    return numeric, other


def _normalise_text(value):
    if not isinstance(value, str):
        return value
    text = " ".join(value.strip().split())
    if not text:
        return np.nan
    key = text.lower()
    if key in CATEGORY_SYNONYMS:
        return CATEGORY_SYNONYMS[key]
    return text.title()


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    out = pd.DataFrame({"Column": counts.index, "Missing": counts.values})
    out["Missing_%"] = (out["Missing"] / max(len(df), 1) * 100).round(2)
    return out.sort_values("Missing", ascending=False).reset_index(drop=True)


def count_duplicates(df: pd.DataFrame) -> int:
    subset = [c for c in df.columns if c != ID_COLUMN]
    return int(df.duplicated(subset=subset).sum())


def outlier_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """IQR (1.5 x IQR) outlier counts per numeric column. Informational only."""
    cols = columns or [c for c in NUMERIC_FEATURES + [REG_TARGET] if c in df.columns]
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"Column": col, "Q1": round(q1, 2), "Q3": round(q3, 2),
                     "Lower_Fence": round(lo, 2), "Upper_Fence": round(hi, 2),
                     "Outliers": n_out, "Outlier_%": round(n_out / len(s) * 100, 2)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- cleaning
def clean_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply fixed, rule-based cleaning. Returns (clean_df, report)."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    report: dict = {"rows_in": int(len(df))}
    report["missing_before"] = int(df[[c for c in df.columns if c != ID_COLUMN]].isna().sum().sum())

    non_numeric, invalid = {}, {}
    for col in NUMERIC_FEATURES + [REG_TARGET]:
        if col not in df.columns:
            continue
        na_before = int(df[col].isna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        non_numeric[col] = int(df[col].isna().sum()) - na_before
        lo, hi = DOMAIN_BOUNDS[col]
        mask = (df[col] < lo) | (df[col] > hi)
        invalid[col] = int(mask.sum())
        df.loc[mask, col] = np.nan
    report["non_numeric_to_missing"] = {k: v for k, v in non_numeric.items() if v}
    report["out_of_range_to_missing"] = {k: v for k, v in invalid.items() if v}

    fixes = 0
    for col in _TEXT_COLUMNS:
        if col not in df.columns:
            continue
        original = df[col].astype("object")
        cleaned = original.map(_normalise_text)
        fixes += int((original.fillna("<NA>") != cleaned.fillna("<NA>")).sum())
        df[col] = cleaned
    report["category_values_standardised"] = fixes

    dup_mask = df.duplicated(subset=[c for c in df.columns if c != ID_COLUMN])
    report["duplicates_removed"] = int(dup_mask.sum())
    df = df.loc[~dup_mask]

    if REG_TARGET in df.columns:
        no_target = df[REG_TARGET].isna()
        report["rows_dropped_missing_target"] = int(no_target.sum())
        df = df.loc[~no_target]

    df = df.reset_index(drop=True)
    report["rows_out"] = int(len(df))
    report["missing_after_cleaning"] = int(df[[c for c in df.columns if c != ID_COLUMN]].isna().sum().sum())
    return df, report


# ---------------------------------------------------------------- target
def compute_yield_thresholds(yield_values: pd.Series) -> tuple[float, float]:
    """Low/Medium and Medium/High cut-offs = 33.3rd and 66.7th percentile of Crop_Yield."""
    y = pd.to_numeric(yield_values, errors="coerce").dropna()
    low, high = y.quantile(1 / 3), y.quantile(2 / 3)
    return float(round(low, 4)), float(round(high, 4))


def assign_yield_category(yield_values: pd.Series, thresholds: tuple[float, float]) -> pd.Series:
    """Low: < t1,  Medium: t1 <= y < t2,  High: >= t2."""
    t1, t2 = thresholds
    cat = pd.cut(pd.to_numeric(yield_values, errors="coerce"), bins=[-np.inf, t1, t2, np.inf],
                 labels=CLASS_ORDER, right=False)
    return cat.astype("object")


def prepare_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float], dict]:
    """Clean raw data and add the Yield_Category target. Returns (df, thresholds, report)."""
    df, report = clean_dataframe(raw)
    if len(df) < MIN_ROWS:
        raise ValueError(f"Only {len(df)} usable rows after cleaning; at least {MIN_ROWS} are needed.")
    thresholds = compute_yield_thresholds(df[REG_TARGET])
    df[CLS_TARGET] = assign_yield_category(df[REG_TARGET], thresholds)
    report["yield_thresholds"] = thresholds
    return df, thresholds, report


# ------------------------------------------------------ pipelines & splits
def build_preprocessor(numeric: list[str] | None = None, categorical: list[str] | None = None,
                       scale: bool = True) -> ColumnTransformer:
    """Imputation + (optional) scaling for numbers; imputation + one-hot for categories."""
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    transformers = []
    if numeric:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), list(numeric)))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categorical)))
    if not transformers:
        raise ValueError("Select at least one feature.")
    return ColumnTransformer(transformers, remainder="drop")


def make_split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED) -> dict:
    """One shared, stratified split used by classification AND regression.

    Stratifying on Yield_Category keeps class proportions equal in both parts
    and makes the classification and regression test sets identical rows.
    """
    X = df[FEATURES]
    strat = df[CLS_TARGET] if df[CLS_TARGET].value_counts().min() >= 2 else None
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, df[CLS_TARGET], df[REG_TARGET], test_size=test_size, random_state=seed, stratify=strat)
    return {"X_train": X_tr, "X_test": X_te, "y_cls_train": yc_tr, "y_cls_test": yc_te,
            "y_reg_train": yr_tr, "y_reg_test": yr_te, "test_size": test_size, "seed": seed}


def encoding_preview(train_df: pd.DataFrame) -> pd.DataFrame:
    """Show which numeric columns the one-hot encoder creates (fitted on training data only)."""
    pre = build_preprocessor()
    pre.fit(train_df[FEATURES])
    names = [n for n in pre.get_feature_names_out() if n.startswith("cat__")]
    return pd.DataFrame({"Encoded_Column": [n.replace("cat__", "") for n in names]})


def scaling_preview(train_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of numeric features before and after StandardScaler (fitted on train only)."""
    num = train_df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    pre = build_preprocessor(NUMERIC_FEATURES, [], scale=True).fit(num)
    scaled = pd.DataFrame(pre.transform(num), columns=NUMERIC_FEATURES)
    return pd.DataFrame({
        "Feature": NUMERIC_FEATURES,
        "Mean_before": num.mean().round(3).values, "Std_before": num.std().round(3).values,
        "Mean_after": scaled.mean().round(3).values, "Std_after": scaled.std().round(3).values,
    })
