import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import pandas as pd
import streamlit as st

from src import classification as clf
from src import regression as reg
from src.config import RANDOM_SEED, TEST_SIZE
from src.evaluation import build_html_report
from src.pipeline import best_name
from src.ui import (cached_full_analysis, download_csv, get_dataset, get_raw_dataset, setup_page, show_df,
                    synthetic_notice)

setup_page("Reports and Downloads")
synthetic_notice()
df, thresholds, report = get_dataset()
raw = get_raw_dataset()

st.write("This page runs all three analyses with **default settings** (seed 42, 20% test set) and packages the results. "
         "Use the individual analysis pages to experiment with parameters.")
analysis = cached_full_analysis(df, RANDOM_SEED, TEST_SIZE)
cls, rg, cl = analysis["classification"], analysis["regression"], analysis["clustering"]

st.subheader("Automatic summary")
for line in analysis["summary"]:
    st.markdown(f"<div class='finding-box'>{line}</div>", unsafe_allow_html=True)

best_c, best_r = best_name(cls["table"]), best_name(rg["table"])
cls_pred = clf.predictions_frame(df, cls["split"], cls["results"][best_c])
reg_pred = reg.predictions_frame(df, rg["split"], rg["results"][best_r])
assign = df.reset_index(drop=True).copy()
assign["Cluster"] = cl["labels"]

evaluation = {
    "dataset_records": len(df), "random_seed": RANDOM_SEED, "test_size": TEST_SIZE,
    "yield_thresholds_t_per_ha": list(thresholds), "best_classifier": best_c, "best_regressor": best_r,
    "clustering": {"linkage": cl["linkage"], "n_clusters": cl["n_clusters"], "silhouette": cl["silhouette"]},
    "classification": cls["table"].to_dict("records"), "regression": rg["table"].to_dict("records"),
    "summary": [s.replace("**", "") for s in analysis["summary"]],
}

dataset_info = {"Records": len(df), "Source": st.session_state.get("data_source", "Built-in synthetic dataset"),
                "Yield thresholds (t/ha)": f"Low < {thresholds[0]:.2f}; High >= {thresholds[1]:.2f}",
                "Train / test rows": f"{len(cls['split']['X_train'])} / {len(cls['split']['X_test'])}",
                "Random seed": RANDOM_SEED}
sections = {"Classification comparison (test set)": cls["table"], "Regression comparison (test set)": rg["table"],
            f"Clustering: silhouette by number of clusters ({cl['linkage']} linkage)": cl["table"],
            f"Cluster profiles (k = {cl['n_clusters']})": cl["profile"],
            f"Classification report: {best_c}": cls["results"][best_c]["report_text"]}
html_report = build_html_report(sections, analysis["summary"], dataset_info)

st.subheader("Downloads")
a, b, c = st.columns(3)
with a:
    st.markdown("**Data**")
    download_csv(df, "Processed dataset", "processed_dataset.csv", "r_proc")
    download_csv(assign, "Cluster assignments", "cluster_assignments.csv", "r_assign")
with b:
    st.markdown("**Predictions**")
    download_csv(cls_pred, f"Classification predictions ({best_c})", "classification_predictions.csv", "r_cp")
    download_csv(reg_pred, f"Regression predictions ({best_r})", "regression_predictions.csv", "r_rp")
with c:
    st.markdown("**Tables and reports**")
    download_csv(cls["table"], "Classification comparison", "classification_comparison.csv", "r_ct")
    download_csv(rg["table"], "Regression comparison", "regression_comparison.csv", "r_rt")
    download_csv(cl["table"], "Clustering comparison", "cluster_comparison.csv", "r_clt")
    st.download_button("Evaluation summary (JSON)", json.dumps(evaluation, indent=2, default=str).encode("utf-8"),
                       file_name="evaluation_summary.json", mime="application/json", key="r_json")
    st.download_button("Consolidated report (HTML)", html_report.encode("utf-8"),
                       file_name="agrovision_report.html", mime="text/html", key="r_html")

st.subheader("Preview")
show_df(cls["table"].round(4))
show_df(rg["table"].round(4))
show_df(cl["table"].round(4))
