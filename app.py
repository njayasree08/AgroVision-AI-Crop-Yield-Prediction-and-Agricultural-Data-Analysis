"""AgroVision AI - Home dashboard.  Run with:  streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src import visualization as viz
from src.config import CLS_TARGET, CLASS_ORDER, REG_TARGET
from src.evaluation import generate_key_findings
from src.preprocessing import missing_summary, split_column_types
from src.ui import get_dataset, get_raw_dataset, setup_page, show_df, show_plot, synthetic_notice

setup_page("AgroVision AI: Crop Yield Prediction and Agricultural Data Analysis")
st.write("Explore agricultural records, group similar fields with hierarchical clustering, classify yield "
         "categories (KNN, Naive Bayes, Decision Tree) and predict crop yield (Linear Regression, Random Forest, "
         "XGBoost). Use the sidebar to move between pages.")
synthetic_notice()

df, thresholds, report = get_dataset()
raw = get_raw_dataset()
numeric_cols, other_cols = split_column_types(raw.drop(columns=[c for c in ["Record_ID"] if c in raw.columns]))

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total records", f"{len(raw):,}")
c2.metric("Features (columns)", len(raw.columns) - (1 if "Record_ID" in raw.columns else 0))
c3.metric("Numerical", len(numeric_cols))
c4.metric("Categorical", len(other_cols))
c5.metric("Average yield", f"{df[REG_TARGET].mean():.2f} t/ha")
c6.metric("Missing values", f"{int(raw.isna().sum().sum()):,}")

st.subheader("Key findings")
for finding in generate_key_findings(df):
    st.markdown(f"<div class='finding-box'>{finding}</div>", unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    show_plot(viz.bar_counts(df, "Crop_Type", "Crop distribution"), key="home_crop")
with right:
    show_plot(viz.bar_counts(df, "Season", "Season distribution"), key="home_season")

left, right = st.columns(2)
with left:
    ordered = df.assign(**{CLS_TARGET: df[CLS_TARGET].astype(str)})
    show_plot(viz.bar_counts(ordered, CLS_TARGET, "Yield-category distribution", viz.CATEGORY_COLORS), key="home_cat")
    st.caption(f"Categories use the 33rd and 67th percentiles of yield: Low < {thresholds[0]:.2f} t/ha, "
               f"Medium {thresholds[0]:.2f} to {thresholds[1]:.2f}, High >= {thresholds[1]:.2f}.")
with right:
    st.markdown("**Missing-value summary (raw data)**")
    summary = missing_summary(raw)
    show_df(summary[summary["Missing"] > 0] if summary["Missing"].sum() else summary.head(5))
    st.caption("Missing values are imputed inside each model pipeline (median / most frequent), "
               "using training data only.")

st.subheader("Workflow")
st.markdown(
    "1. **Dataset Explorer**: inspect, filter, download or upload data.\n"
    "2. **Data Preprocessing**: missing values, duplicates, outliers, encoding, scaling.\n"
    "3. **Classification**, **Clustering**, **Regression**: tune and compare algorithms.\n"
    "4. **Crop Yield Prediction**: enter field conditions and get an estimate.\n"
    "5. **Reports**: download tables and a consolidated HTML report.")
