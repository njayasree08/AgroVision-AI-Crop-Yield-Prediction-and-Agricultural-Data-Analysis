"""Validation and prediction for a single set of field conditions."""
from __future__ import annotations

import pandas as pd

from .config import CATEGORICAL_FEATURES, DOMAIN_BOUNDS, FEATURES, NUMERIC_FEATURES
from .preprocessing import assign_yield_category


def validate_inputs(inputs: dict) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errors = []
    for col in FEATURES:
        if col not in inputs or inputs[col] in (None, ""):
            errors.append(f"{col} is required.")
    if errors:
        return errors
    for col in NUMERIC_FEATURES:
        try:
            value = float(inputs[col])
        except (TypeError, ValueError):
            errors.append(f"{col} must be a number.")
            continue
        lo, hi = DOMAIN_BOUNDS[col]
        if not (lo <= value <= hi):
            errors.append(f"{col} must be between {lo:g} and {hi:g} (got {value:g}).")
    for col in CATEGORICAL_FEATURES:
        if not str(inputs[col]).strip():
            errors.append(f"{col} must not be empty.")
    return errors


def training_range_warnings(inputs: dict, reference: pd.DataFrame) -> list[str]:
    """Warn (not block) when an input lies outside the range seen in the data."""
    warnings = []
    for col in NUMERIC_FEATURES:
        if col in reference.columns:
            lo, hi = reference[col].min(), reference[col].max()
            if not (lo <= float(inputs[col]) <= hi):
                warnings.append(f"{col} = {float(inputs[col]):g} is outside the range seen in the data "
                                f"({lo:g} - {hi:g}); the estimate is an extrapolation.")
    return warnings


def make_input_frame(inputs: dict) -> pd.DataFrame:
    row = {c: float(inputs[c]) for c in NUMERIC_FEATURES}
    row.update({c: str(inputs[c]).strip() for c in CATEGORICAL_FEATURES})
    return pd.DataFrame([row], columns=FEATURES)


def predict_one(reg_bundle: dict, cls_bundle: dict | None, inputs: dict) -> dict:
    """Run the saved regression (and optionally classification) pipelines."""
    errors = validate_inputs(inputs)
    if errors:
        raise ValueError(" ".join(errors))
    X = make_input_frame(inputs)
    out = {"yield_estimate": max(0.0, float(reg_bundle["pipeline"].predict(X)[0]))}
    thresholds = reg_bundle.get("thresholds") or (cls_bundle or {}).get("thresholds")
    if thresholds:
        out["category_from_estimate"] = str(assign_yield_category(pd.Series([out["yield_estimate"]]), thresholds).iloc[0])
    if cls_bundle is not None:
        pipe = cls_bundle["pipeline"]
        out["classifier_category"] = str(pipe.predict(X)[0])
        if hasattr(pipe, "predict_proba"):
            out["class_probabilities"] = {str(c): float(p) for c, p in zip(pipe.classes_, pipe.predict_proba(X)[0])}
    return out
