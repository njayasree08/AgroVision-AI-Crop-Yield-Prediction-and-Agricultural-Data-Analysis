import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src import visualization as viz
from src.config import CATEGORICAL_FEATURES, DOMAIN_BOUNDS, FEATURES, NUMERIC_FEATURES, TEST_SIZE, RANDOM_SEED
from src.preprocessing import (count_duplicates, encoding_preview, make_split, missing_summary,
                               outlier_summary, scaling_preview)
from src.ui import download_csv, get_dataset, get_raw_dataset, setup_page, show_df, show_plot, synthetic_notice

setup_page("Data Preprocessing")
synthetic_notice()
raw = get_raw_dataset()
clean, thresholds, report = get_dataset()

st.write("Cleaning uses **fixed rules only** (valid ranges, consistent spelling). Anything learned from the data "
         "(imputation values, scaling statistics, category lists) is fitted on the **training split only**, inside "
         "each model pipeline, so no information leaks from the test set.")

tabs = st.tabs(["Overview", "Missing values", "Duplicates", "Outliers", "Categorical encoding", "Feature scaling"])

with tabs[0]:
    a, b, c, d = st.columns(4)
    a.metric("Rows in", report["rows_in"])
    b.metric("Rows out", report["rows_out"])
    c.metric("Spellings standardised", report["category_values_standardised"])
    d.metric("Out-of-range -> missing", sum(report["out_of_range_to_missing"].values()))
    st.markdown("**Before and after cleaning**")
    show_df(pd.DataFrame({
        "Step": ["Missing cells before cleaning", "Non-numeric text set to missing", "Out-of-range values set to missing",
                 "Missing cells after cleaning (imputed later, per pipeline)", "Duplicate rows removed",
                 "Rows without a yield value removed"],
        "Count": [report["missing_before"], sum(report["non_numeric_to_missing"].values()),
                  sum(report["out_of_range_to_missing"].values()), report["missing_after_cleaning"],
                  report["duplicates_removed"], report.get("rows_dropped_missing_target", 0)]}))
    if report["out_of_range_to_missing"]:
        st.caption("Out-of-range values by column: " + ", ".join(f"{k}: {v}" for k, v in report["out_of_range_to_missing"].items()))
    st.markdown("**Valid ranges used**")
    show_df(pd.DataFrame([{"Column": k, "Min": v[0], "Max": v[1]} for k, v in DOMAIN_BOUNDS.items()]))
    download_csv(clean, "Download processed (cleaned) dataset", "agrovision_processed.csv", "dl_processed")

with tabs[1]:
    before, after = missing_summary(raw), missing_summary(clean)
    c1, c2 = st.columns(2)
    c1.markdown("**Before cleaning (raw)**")
    with c1:
        show_df(before[before["Missing"] > 0])
    c2.markdown("**After cleaning**")
    with c2:
        show_df(after[after["Missing"] > 0])
    if before["Missing"].sum():
        show_plot(viz.missing_bar(before), key="pp_missing")
    st.info("Handling strategy: numeric columns are imputed with the **median**, categorical columns with the "
            "**most frequent** value. The imputer is fitted on training data only.")

with tabs[2]:
    st.metric("Duplicate records in raw data (ignoring Record_ID)", count_duplicates(raw))
    st.write("Exact duplicates (all columns except the ID) are removed during cleaning. "
             f"Removed in this run: **{report['duplicates_removed']}**.")

with tabs[3]:
    summary = outlier_summary(clean)
    show_df(summary)
    st.caption("IQR rule: values beyond 1.5 x IQR from the quartiles are flagged. They are *shown*, not deleted: "
               "skewed variables such as farm area legitimately have large values. Only physically impossible "
               "values (outside the valid ranges above) are removed and imputed.")
    col = st.selectbox("Inspect a column", NUMERIC_FEATURES + ["Crop_Yield"])
    show_plot(viz.box_plot(clean, col), key="pp_box")

split = make_split(clean, TEST_SIZE, RANDOM_SEED)
with tabs[4]:
    st.write("Categorical columns: " + ", ".join(CATEGORICAL_FEATURES) + ". One-hot encoding is used; categories "
             "unseen during training are ignored at prediction time.")
    st.markdown("**Example of the original columns**")
    show_df(split["X_train"][CATEGORICAL_FEATURES].head(8))
    enc = encoding_preview(split["X_train"])
    st.markdown(f"**Encoded columns created from the training split ({len(enc)})**")
    show_df(enc)

with tabs[5]:
    st.write("Numeric columns are standardised (mean 0, standard deviation 1) using the **training split only**. "
             "KNN and Naive Bayes use the scaled values; tree models do not need scaling but share the pipeline.")
    show_df(scaling_preview(split["X_train"]))
    st.caption(f"Training rows: {len(split['X_train'])}, test rows: {len(split['X_test'])} "
               f"(stratified by yield category, seed {RANDOM_SEED}).")
