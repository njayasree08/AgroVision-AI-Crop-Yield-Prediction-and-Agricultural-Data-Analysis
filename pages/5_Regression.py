import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src import visualization as viz
from src.config import RANDOM_SEED
from src.evaluation import feature_importance_table
from src.regression import BOOSTED, FOREST, LINEAR, REGRESSOR_NAMES, XGBOOST_AVAILABLE, predictions_frame
from src.ui import cached_regression, download_csv, get_dataset, setup_page, show_df, show_plot, synthetic_notice

setup_page("Regression Analysis")
synthetic_notice()
df, thresholds, report = get_dataset()

st.write("Predict the numeric **Crop_Yield (t/ha)** from the 12 field/soil/management inputs. All models share the same "
         "preprocessing (imputation, scaling, one-hot encoding) and the same train/test rows.")
if not XGBOOST_AVAILABLE:
    st.warning("The `xgboost` package is not installed, so scikit-learn's **HistGradientBoostingRegressor** is used as a "
               "documented substitute (also gradient-boosted trees). Install it with `pip install xgboost` to use XGBoost.")

with st.sidebar:
    st.markdown("#### Regression settings")
    test_size = st.slider("Test set share", 0.1, 0.4, 0.2, 0.05)
    st.markdown("**Random Forest**")
    rf_trees = st.slider("n_estimators", 50, 500, 200, 50)
    rf_depth = st.slider("max_depth (0 = unlimited)", 0, 30, 12)
    st.markdown("**" + ("XGBoost" if XGBOOST_AVAILABLE else "HistGradientBoosting") + "**")
    gb_trees = st.slider("Boosting rounds", 50, 600, 300, 50)
    gb_lr = st.select_slider("Learning rate", [0.01, 0.02, 0.05, 0.1, 0.2, 0.3], value=0.05)
    gb_depth = st.slider("Tree depth", 2, 10, 4)

params = {FOREST: {"n_estimators": rf_trees, "max_depth": rf_depth, "min_samples_leaf": 2},
          BOOSTED: {"n_estimators": gb_trees, "learning_rate": gb_lr, "max_depth": gb_depth}}
run = cached_regression(df, params, float(test_size), RANDOM_SEED)
results, table, split = run["results"], run["table"], run["split"]

tabs = st.tabs(["Comparison"] + REGRESSOR_NAMES + ["Downloads"])
with tabs[0]:
    best = table.iloc[0]
    st.success(f"Lowest test RMSE: **{best['Model']}** ({best['RMSE']:.3f} t/ha, R2 = {best['R2']:.3f}). "
               "Ranking is computed from actual predictions on the held-out test set.")
    show_df(table.round(4))
    a, b = st.columns(2)
    with a:
        show_plot(viz.metric_comparison_bar(table, ["MAE", "RMSE"], "Prediction error (lower is better)", "t/ha"), key="reg_err")
    with b:
        show_plot(viz.single_metric_bar(table, "R2", "R-squared (higher is better)"), key="reg_r2")
    st.caption("A very high R2 is partly expected here: the synthetic yield follows simple rules and "
               "Previous_Year_Yield is informative. Real data are noisier. A simple linear model can beat "
               "complex ones when the signal is mostly additive.")

for tab, name in zip(tabs[1:4], REGRESSOR_NAMES):
    r = results[name]
    with tab:
        m = r["metrics"]
        c = st.columns(5)
        c[0].metric("MAE", f"{m['mae']:.3f}")
        c[1].metric("MSE", f"{m['mse']:.3f}")
        c[2].metric("RMSE", f"{m['rmse']:.3f}")
        c[3].metric("R2", f"{m['r2']:.3f}")
        c[4].metric("Training time", f"{m['fit_time_s'] * 1000:.0f} ms")
        a, b = st.columns(2)
        with a:
            show_plot(viz.actual_vs_predicted(r["y_true"], r["y_pred"], f"{name}: actual vs predicted"), key=f"avp_{name}")
        with b:
            show_plot(viz.residual_plot(r["y_pred"], r["residuals"], f"{name}: residuals"), key=f"res_{name}")
        show_plot(viz.residual_histogram(r["residuals"], f"{name}: residual distribution"), key=f"resh_{name}")
        st.caption("Residuals should scatter randomly around zero. Curves or funnels suggest missed patterns or "
                   "unequal error across yield levels.")
        imp = feature_importance_table(r["pipeline"])
        if imp is not None:
            show_plot(viz.feature_importance_bar(imp, f"{name}: feature importance"), key=f"imp_{name}")

with tabs[4]:
    choice = st.selectbox("Model", REGRESSOR_NAMES)
    preds = predictions_frame(df, split, results[choice])
    show_df(preds.head(200))
    download_csv(preds, f"Download {choice} predictions", "regression_predictions.csv", "dl_reg")
    download_csv(table, "Download model comparison table", "regression_comparison.csv", "dl_reg_table")
