"""Tests for dataset generation, loading, cleaning and the target variable."""
import io
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from src.config import (CLASS_ORDER, CLS_TARGET, DATA_PATH, FEATURES, NUMERIC_FEATURES, REG_TARGET,
                        REQUIRED_COLUMNS)
from src.data_generator import generate_dataset
from src.data_loader import read_uploaded_csv, validate_schema
from src.preprocessing import (assign_yield_category, clean_dataframe, compute_yield_thresholds,
                               count_duplicates, outlier_summary, prepare_dataset)
from src.prediction import validate_inputs


@lru_cache(maxsize=1)
def raw_data() -> pd.DataFrame:
    return generate_dataset()


def test_dataset_has_exactly_700_unique_records():
    df = raw_data()
    assert len(df) == 700
    assert df["Record_ID"].is_unique
    assert set(REQUIRED_COLUMNS).issubset(df.columns)
    assert df.shape[1] == 17


def test_generation_is_reproducible():
    pd.testing.assert_frame_equal(generate_dataset(seed=42), generate_dataset(seed=42))
    assert not generate_dataset(seed=1).equals(generate_dataset(seed=2))


def test_saved_csv_matches_generator():
    if DATA_PATH.exists():
        saved = pd.read_csv(DATA_PATH)
        assert len(saved) == 700
        assert list(saved.columns) == list(raw_data().columns)


def test_dataset_contains_quality_problems_to_clean():
    df = raw_data()
    assert df[FEATURES].isna().sum().sum() > 0
    assert df[REG_TARGET].notna().all()
    assert df["Annual_Rainfall_mm"].max() > 3000  # injected impossible values


def test_cleaning_fixes_categories_and_invalid_values():
    clean, report = clean_dataframe(raw_data())
    assert set(clean["Crop_Type"].dropna().unique()) == {"Rice", "Wheat", "Maize", "Cotton", "Soybean",
                                                         "Groundnut", "Bajra", "Chickpea"}
    assert set(clean["Season"].dropna().unique()) <= {"Kharif", "Rabi", "Summer"}
    assert set(clean["Irrigation_Type"].dropna().unique()) <= {"Canal", "Drip", "Sprinkler", "Tube Well", "Rainfed"}
    assert clean["Annual_Rainfall_mm"].max() <= 3000
    assert report["category_values_standardised"] > 0
    assert report["out_of_range_to_missing"]["Annual_Rainfall_mm"] >= 1
    assert len(clean) == 700 and count_duplicates(clean) == 0


def test_cleaning_removes_duplicates_and_missing_targets():
    base = generate_dataset(inject_issues=False).head(60)
    dup = pd.concat([base, base.iloc[[0]].assign(Record_ID="AGV-9999")], ignore_index=True)
    dup.loc[1, REG_TARGET] = np.nan
    clean, report = clean_dataframe(dup)
    assert report["duplicates_removed"] == 1
    assert report["rows_dropped_missing_target"] == 1
    assert len(clean) == 59


def test_yield_category_thresholds_and_labels():
    df, thresholds, _ = prepare_dataset(raw_data())
    low, high = thresholds
    assert low < high
    assert set(df[CLS_TARGET].unique()) == set(CLASS_ORDER)
    assert df[CLS_TARGET].notna().all()
    assert (df.loc[df[CLS_TARGET] == "Low", REG_TARGET] < low).all()
    assert (df.loc[df[CLS_TARGET] == "High", REG_TARGET] >= high).all()
    share = df[CLS_TARGET].value_counts(normalize=True)
    assert share.min() > 0.28  # tertiles -> near-balanced classes
    assert compute_yield_thresholds(df[REG_TARGET]) == thresholds
    assert list(assign_yield_category(pd.Series([0.1, 5.0]), thresholds)) == ["Low", "High"]


def test_target_not_in_features():
    assert REG_TARGET not in FEATURES and CLS_TARGET not in FEATURES and "Record_ID" not in FEATURES


def test_outlier_summary_columns():
    clean, _ = clean_dataframe(raw_data())
    summary = outlier_summary(clean)
    assert {"Column", "Outliers", "Lower_Fence", "Upper_Fence"} <= set(summary.columns)
    assert len(summary) == len(NUMERIC_FEATURES) + 1


def test_schema_validation_messages():
    errors, _ = validate_schema(raw_data().drop(columns=["Soil_pH", "Crop_Type"]))
    assert any("Soil_pH" in e and "Crop_Type" in e for e in errors)
    errors, _ = validate_schema(raw_data().head(10))
    assert any("At least" in e for e in errors)
    errors, _ = validate_schema(raw_data())
    assert errors == []


def test_upload_parsing_is_plain_csv_only():
    csv = raw_data().head(60).to_csv(index=False).encode()
    df = read_uploaded_csv(io.BytesIO(csv))
    assert len(df) == 60
    with pytest.raises(ValueError):
        read_uploaded_csv(io.BytesIO(b""))


def test_prediction_input_validation():
    ok = {"Crop_Type": "Rice", "Season": "Kharif", "Soil_Type": "Loamy", "Irrigation_Type": "Canal",
          "Area_Hectares": 2.0, "Annual_Rainfall_mm": 1000.0, "Average_Temperature_C": 27.0, "Soil_pH": 6.5,
          "Nitrogen": 200.0, "Phosphorus": 30.0, "Potassium": 180.0, "Fertilizer_Usage": 120.0,
          "Previous_Year_Yield": 4.0}
    assert validate_inputs(ok) == []
    assert validate_inputs({**ok, "Soil_pH": 15}) != []
    assert validate_inputs({**ok, "Annual_Rainfall_mm": "abc"}) != []
    broken = dict(ok)
    broken.pop("Crop_Type")
    assert validate_inputs(broken) != []
