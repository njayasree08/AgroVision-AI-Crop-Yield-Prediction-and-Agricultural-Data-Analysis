import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src import visualization as viz
from src.classification import CLASSIFIER_NAMES, knn_k_sweep, predictions_frame
from src.config import RANDOM_SEED
from src.evaluation import feature_importance_table
from src.ui import (cached_classification, download_csv, get_dataset, setup_page, show_df, show_plot,
                    show_pyplot, synthetic_notice)

setup_page("Classification Analysis")
synthetic_notice()
df, thresholds, report = get_dataset()

st.write(f"**Target:** `Yield_Category`, derived from `Crop_Yield` with fixed percentile cut-offs "
         f"(Low < {thresholds[0]:.2f} t/ha, Medium {thresholds[0]:.2f} to {thresholds[1]:.2f}, High >= {thresholds[1]:.2f}). "
         "`Crop_Yield` and the ID are never used as inputs.")
dist = df["Yield_Category"].value_counts()
st.caption("Class distribution: " + ", ".join(f"{k}: {v}" for k, v in dist.items()) +
           ". Classes are near-balanced; stratified splitting, macro-averaged metrics and (for the tree) "
           "class weights are used to handle any imbalance.")

# ------------------------------------------------------------- parameters
with st.sidebar:
    st.markdown("#### Classification settings")
    test_size = st.slider("Test set share", 0.1, 0.4, 0.2, 0.05)
    st.markdown("**KNN**")
    k = st.slider("K (neighbours)", 1, 31, 7, 2)
    weights = st.selectbox("Neighbour weights", ["uniform", "distance"])
    metric = st.selectbox("Distance metric", ["euclidean", "manhattan"])
    st.markdown("**Naive Bayes**")
    smoothing_exp = st.slider("var_smoothing (10^x)", -12, -3, -9)
    st.markdown("**Decision Tree**")
    depth = st.slider("max_depth", 1, 20, 6)
    min_split = st.slider("min_samples_split", 2, 50, 10)
    min_leaf = st.slider("min_samples_leaf", 1, 30, 4)
    criterion = st.selectbox("criterion", ["gini", "entropy"])
    cw = st.selectbox("class_weight", ["balanced", "none"])

params = {"KNN": {"n_neighbors": k, "weights": weights, "metric": metric},
          "Naive Bayes": {"var_smoothing": 10.0 ** smoothing_exp},
          "Decision Tree": {"max_depth": depth, "min_samples_split": min_split, "min_samples_leaf": min_leaf,
                            "criterion": criterion, "class_weight": cw}}
run = cached_classification(df, params, float(test_size), RANDOM_SEED)
results, table, split = run["results"], run["table"], run["split"]

tabs = st.tabs(["Comparison"] + CLASSIFIER_NAMES + ["Downloads"])

with tabs[0]:
    best = table.iloc[0]
    st.success(f"Best on this test set by macro-F1: **{best['Model']}** (F1 {best['F1 (macro)']:.3f}, "
               f"accuracy {best['Accuracy']:.3f}). Ranking is computed from the actual predictions below.")
    show_df(table.round(4))
    show_plot(viz.metric_comparison_bar(table, ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"],
                                        "Classification algorithm comparison (test set)"), key="cls_cmp")
    st.caption(f"Train rows: {len(split['X_train'])}, test rows: {len(split['X_test'])}. All models use the same split. "
               "'CV F1 macro (train)' is 5-fold cross-validation on the training data only. "
               "Synthetic data can show the workflow but cannot establish real-world accuracy.")

for tab, name in zip(tabs[1:4], CLASSIFIER_NAMES):
    r = results[name]
    with tab:
        m = r["metrics"]
        c = st.columns(5)
        c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
        c[1].metric("Precision (macro)", f"{m['precision_macro']:.3f}")
        c[2].metric("Recall (macro)", f"{m['recall_macro']:.3f}")
        c[3].metric("F1 (macro)", f"{m['f1_macro']:.3f}")
        c[4].metric("Train / predict time", f"{m['fit_time_s'] * 1000:.0f} / {m['predict_time_s'] * 1000:.0f} ms")
        left, right = st.columns(2)
        with left:
            show_plot(viz.confusion_matrix_fig(r["confusion"], r["labels"], f"{name}: confusion matrix"), key=f"cm_{name}")
        with right:
            st.markdown("**Classification report**")
            st.code(r["report_text"])
            st.caption(f"Weighted: precision {m['precision_weighted']:.3f}, recall {m['recall_weighted']:.3f}, "
                       f"F1 {m['f1_weighted']:.3f}.")

        if name == "KNN":
            st.markdown("**Experiment with K** (cross-validated on training data; the test set is not used)")
            sweep = knn_k_sweep(split, [1, 3, 5, 7, 9, 11, 15, 21, 31], weights, metric)
            show_plot(viz.knn_k_line(sweep), key="knn_sweep")
            st.caption("KNN classifies by majority vote of the K nearest records, so features must be on a similar "
                       "scale (done by the pipeline's StandardScaler).")
        if name == "Naive Bayes":
            with st.expander("Independence assumption and its limits", expanded=True):
                st.markdown(
                    "Naive Bayes assumes each feature is **conditionally independent** given the class, and "
                    "Gaussian NB assumes each numeric feature is **normally distributed** within each class. "
                    "In agriculture, rainfall, temperature, soil and fertiliser interact, and one-hot encoded "
                    "category columns (0/1) are not Gaussian, so the assumptions are violated. The model is fast "
                    "and a useful baseline, but it can be poorly calibrated or ignore a class entirely, "
                    "as the confusion matrix may show.")
        if name == "Decision Tree":
            imp = feature_importance_table(r["pipeline"])
            if imp is not None:
                show_plot(viz.feature_importance_bar(imp, "Decision Tree feature importance (grouped by original column)"), key="dt_imp")
            show_depth = st.slider("Levels of the tree to draw", 1, 5, 3)
            show_pyplot(viz.decision_tree_figure(r["pipeline"], show_depth))

with tabs[4]:
    choice = st.selectbox("Model", CLASSIFIER_NAMES)
    preds = predictions_frame(df, split, results[choice])
    st.caption(f"{len(preds)} test-set predictions, accuracy {preds['Correct'].mean():.3f}.")
    show_df(preds.head(200))
    download_csv(preds, f"Download {choice} predictions", f"classification_predictions_{choice.lower().replace(' ', '_')}.csv", "dl_cls")
    download_csv(table, "Download model comparison table", "classification_comparison.csv", "dl_cls_table")
