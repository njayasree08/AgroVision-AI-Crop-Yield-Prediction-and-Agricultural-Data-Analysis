import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src import visualization as viz
from src.config import REQUIRED_COLUMNS
from src.data_loader import read_uploaded_csv, validate_schema
from src.preprocessing import count_duplicates, split_column_types
from src.ui import (BUILTIN_LABEL, download_csv, get_dataset, get_raw_dataset, setup_page, show_df,
                    show_pyplot, synthetic_notice)

setup_page("Dataset Explorer")
synthetic_notice()

# ------------------------------------------------------------------ upload
with st.expander("Upload your own CSV dataset", expanded=False):
    st.write("Required columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS) +
             ". Optional: `Record_ID`, `State`, `District`. The file is only parsed as CSV text.")
    upload = st.file_uploader("Choose a CSV file", type=["csv"])
    if upload is not None:
        try:
            candidate = read_uploaded_csv(upload)
            errors, warnings = validate_schema(candidate)
        except ValueError as exc:
            candidate, errors, warnings = None, [str(exc)], []
        for e in errors:
            st.error(e)
        for w in warnings:
            st.warning(w)
        if candidate is not None and not errors:
            st.success(f"File looks valid: {len(candidate)} rows, {len(candidate.columns)} columns.")
            if st.button("Use this dataset in all pages"):
                st.session_state["uploaded_raw"] = candidate
                st.session_state["data_source"] = f"Uploaded file: {upload.name}"
                st.rerun()
    if st.session_state.get("uploaded_raw") is not None and st.button("Go back to the built-in dataset"):
        st.session_state.pop("uploaded_raw", None)
        st.session_state["data_source"] = BUILTIN_LABEL
        st.rerun()

raw = get_raw_dataset()
clean, thresholds, report = get_dataset()
view = st.radio("Show", ["Cleaned data (used by the models)", "Raw data (as loaded)"], horizontal=True)
data = clean if view.startswith("Cleaned") else raw

# ------------------------------------------------------- search & filtering
st.subheader("Browse")
c1, c2 = st.columns([2, 3])
search = c1.text_input("Search text in any column")
all_cols = list(data.columns)
chosen = c2.multiselect("Columns to show", all_cols, default=all_cols)

filtered = data
filter_cols = [c for c in ("Crop_Type", "Season", "State", "Soil_Type", "Irrigation_Type", "Yield_Category") if c in data.columns]
fcols = st.columns(max(len(filter_cols), 1))
for col, box in zip(filter_cols, fcols):
    options = sorted(data[col].dropna().astype(str).unique())
    picked = box.multiselect(col.replace("_", " "), options)
    if picked:
        filtered = filtered[filtered[col].astype(str).isin(picked)]
if search:
    mask = filtered.astype(str).apply(lambda s: s.str.contains(search, case=False, na=False, regex=False)).any(axis=1)
    filtered = filtered[mask]

st.caption(f"{len(filtered):,} of {len(data):,} rows match.")
shown = filtered[chosen] if chosen else filtered
show_df(shown.head(500))
if len(shown) > 500:
    st.caption("Only the first 500 rows are displayed; the download contains all matching rows.")
download_csv(shown, "Download shown data as CSV", "agrovision_filtered_data.csv", "dl_explorer")

# -------------------------------------------------------------- statistics
st.subheader("Summary")
a, b, c = st.columns(3)
a.metric("Rows x columns", f"{data.shape[0]:,} x {data.shape[1]}")
b.metric("Duplicate records (ignoring ID)", count_duplicates(raw))
c.metric("Missing cells (raw)", f"{int(raw.isna().sum().sum()):,}")

tab1, tab2, tab3, tab4 = st.tabs(["Descriptive statistics", "Data types", "Missing values", "Yield distribution"])
with tab1:
    numeric, others = split_column_types(data)
    st.markdown("**Numerical columns**")
    st.dataframe(data[numeric].describe().T.round(3), use_container_width=True)
    if others:
        st.markdown("**Categorical columns**")
        st.dataframe(data[others].astype("object").describe().T, use_container_width=True)
with tab2:
    show_df(pd.DataFrame({"Column": data.columns, "Data type": [str(t) for t in data.dtypes],
                          "Unique values": [int(data[c].nunique()) for c in data.columns]}))
with tab3:
    miss = pd.DataFrame({"Column": data.columns, "Missing": data.isna().sum().values})
    miss["Missing %"] = (miss["Missing"] / max(len(data), 1) * 100).round(2)
    show_df(miss.sort_values("Missing", ascending=False))
with tab4:
    if "Crop_Type" in clean.columns:
        show_pyplot(viz.yield_distribution_seaborn(clean))
