import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.config import DOMAIN_BOUNDS, RANDOM_SEED, REG_TARGET, TEST_SIZE
from src.data_generator import IRRIGATION_NAMES, SOILS, CROPS
from src.model_io import load_bundle
from src.pipeline import train_and_save_all
from src.prediction import predict_one, training_range_warnings, validate_inputs
from src.ui import get_dataset, setup_page, show_df, synthetic_notice

setup_page("Crop Yield Prediction")
synthetic_notice()
df, thresholds, report = get_dataset()

reg_bundle = load_bundle("best_regressor.joblib")
cls_bundle = load_bundle("best_classifier.joblib")

# ----------------------------------------------------------- model status
if reg_bundle is None or cls_bundle is None:
    st.warning("No trained models were found in the `models/` folder (or they could not be read). "
               "Train them now; this takes a few seconds.")
    if st.button("Train and save models"):
        with st.spinner("Training classification and regression models..."):
            train_and_save_all(df, thresholds, RANDOM_SEED, TEST_SIZE)
        st.rerun()
    st.stop()

st.caption(f"Regression model: **{reg_bundle['name']}** (test RMSE {reg_bundle['metrics']['rmse']:.3f} t/ha, "
           f"R2 {reg_bundle['metrics']['r2']:.3f})  |  Classification model: **{cls_bundle['name']}** "
           f"(test accuracy {cls_bundle['metrics']['accuracy']:.3f}). Saved pipelines include their own preprocessing.")
if st.button("Retrain models on the current dataset"):
    with st.spinner("Retraining..."):
        train_and_save_all(df, thresholds, RANDOM_SEED, TEST_SIZE)
    st.rerun()


def options(col, fallback):
    found = sorted(df[col].dropna().astype(str).unique()) if col in df.columns else []
    return found or sorted(fallback)


# -------------------------------------------------------------------- form
with st.form("predict_form"):
    st.subheader("Field conditions")
    a, b, c = st.columns(3)
    crop = a.selectbox("Crop type", options("Crop_Type", CROPS))
    season = b.selectbox("Season", options("Season", ["Kharif", "Rabi", "Summer"]))
    irrigation = c.selectbox("Irrigation type", options("Irrigation_Type", IRRIGATION_NAMES))

    a, b, c = st.columns(3)
    area = a.number_input("Area (hectares)", DOMAIN_BOUNDS["Area_Hectares"][0], DOMAIN_BOUNDS["Area_Hectares"][1], 3.0, 0.5)
    rain = b.number_input("Annual rainfall (mm)", DOMAIN_BOUNDS["Annual_Rainfall_mm"][0], DOMAIN_BOUNDS["Annual_Rainfall_mm"][1], 900.0, 10.0)
    temp = c.number_input("Average temperature (°C)", DOMAIN_BOUNDS["Average_Temperature_C"][0], DOMAIN_BOUNDS["Average_Temperature_C"][1], 26.0, 0.5)

    a, b, c = st.columns(3)
    soil = a.selectbox("Soil type", options("Soil_Type", SOILS))
    ph = b.number_input("Soil pH", DOMAIN_BOUNDS["Soil_pH"][0], DOMAIN_BOUNDS["Soil_pH"][1], 6.8, 0.1)
    fert = c.number_input("Fertilizer usage (kg/ha)", DOMAIN_BOUNDS["Fertilizer_Usage"][0], DOMAIN_BOUNDS["Fertilizer_Usage"][1], 120.0, 5.0)

    a, b, c, d = st.columns(4)
    n_val = a.number_input("Nitrogen (kg/ha)", DOMAIN_BOUNDS["Nitrogen"][0], DOMAIN_BOUNDS["Nitrogen"][1], 220.0, 5.0)
    p_val = b.number_input("Phosphorus (kg/ha)", DOMAIN_BOUNDS["Phosphorus"][0], DOMAIN_BOUNDS["Phosphorus"][1], 30.0, 1.0)
    k_val = c.number_input("Potassium (kg/ha)", DOMAIN_BOUNDS["Potassium"][0], DOMAIN_BOUNDS["Potassium"][1], 190.0, 5.0)
    prev = d.number_input("Previous-year yield (t/ha)", DOMAIN_BOUNDS["Previous_Year_Yield"][0], DOMAIN_BOUNDS["Previous_Year_Yield"][1], 3.5, 0.1)
    submitted = st.form_submit_button("Predict yield")

if submitted:
    inputs = {"Crop_Type": crop, "Season": season, "Soil_Type": soil, "Irrigation_Type": irrigation,
              "Area_Hectares": area, "Annual_Rainfall_mm": rain, "Average_Temperature_C": temp, "Soil_pH": ph,
              "Nitrogen": n_val, "Phosphorus": p_val, "Potassium": k_val, "Fertilizer_Usage": fert,
              "Previous_Year_Yield": prev}
    problems = validate_inputs(inputs)
    if problems:
        for p in problems:
            st.error(p)
    else:
        for w in training_range_warnings(inputs, df):
            st.warning(w)
        try:
            out = predict_one(reg_bundle, cls_bundle, inputs)
        except Exception as exc:  # e.g. model trained on a different schema
            st.error(f"Prediction failed: {exc}. Try retraining the models.")
            st.stop()
        left, right = st.columns(2)
        with left:
            st.markdown(f"<div class='result-card'><div class='small'>Regression model: estimated yield</div>"
                        f"<div class='big'>{out['yield_estimate']:.2f} t/ha</div>"
                        f"<div class='small'>Typical test error about ±{reg_bundle['metrics']['mae']:.2f} t/ha (MAE)."
                        f" This is a numeric estimate.</div></div>", unsafe_allow_html=True)
        with right:
            st.markdown(f"<div class='result-card'><div class='small'>Classification model: predicted yield category</div>"
                        f"<div class='big'>{out['classifier_category']}</div>"
                        f"<div class='small'>A separate model that predicts Low / Medium / High directly.</div></div>",
                        unsafe_allow_html=True)
        st.info("These are **two different predictions** from two different models, so they can occasionally disagree "
                f"near a category boundary. For reference, the regression estimate falls in the **{out.get('category_from_estimate', 'n/a')}** "
                f"band (Low < {thresholds[0]:.2f}, High >= {thresholds[1]:.2f} t/ha).")
        if "class_probabilities" in out:
            st.markdown("**Classifier confidence**")
            show_df(pd.DataFrame({"Category": list(out["class_probabilities"]),
                                                    "Probability": [round(v, 3) for v in out["class_probabilities"].values()]}))
        st.caption("Estimates come from models trained on the active dataset; with the built-in synthetic data they "
                   "illustrate the method and must not guide real farming decisions.")
