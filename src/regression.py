"""Crop-yield regression: Linear Regression, Random Forest, XGBoost (or a documented fallback).

If the optional `xgboost` package cannot be imported, scikit-learn's
HistGradientBoostingRegressor is used in its place and labelled as such in
every table and chart (it is also a gradient-boosted tree ensemble).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from .config import ID_COLUMN, RANDOM_SEED, TEST_SIZE
from .evaluation import regression_metrics
from .preprocessing import build_preprocessor, make_split

try:  # optional dependency
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:  # ImportError or a broken binary install
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

LINEAR, FOREST = "Linear Regression", "Random Forest"
BOOSTED = "XGBoost" if XGBOOST_AVAILABLE else "HistGradientBoosting (XGBoost substitute)"
REGRESSOR_NAMES = [LINEAR, FOREST, BOOSTED]

DEFAULT_PARAMS = {
    LINEAR: {},
    FOREST: {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2},
    BOOSTED: {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
}


def build_regressor(name: str, params: dict | None = None, seed: int = RANDOM_SEED) -> Pipeline:
    p = {**DEFAULT_PARAMS[name], **(params or {})}
    if name == LINEAR:
        model = LinearRegression()
    elif name == FOREST:
        depth = p["max_depth"]
        model = RandomForestRegressor(n_estimators=int(p["n_estimators"]),
                                      max_depth=None if depth in (None, 0) else int(depth),
                                      min_samples_leaf=int(p["min_samples_leaf"]), random_state=seed, n_jobs=-1)
    elif name == BOOSTED and XGBOOST_AVAILABLE:
        model = XGBRegressor(n_estimators=int(p["n_estimators"]), learning_rate=float(p["learning_rate"]),
                             max_depth=int(p["max_depth"]), subsample=0.9, colsample_bytree=0.9,
                             tree_method="hist", random_state=seed, n_jobs=1, verbosity=0)
    elif name == BOOSTED:
        model = HistGradientBoostingRegressor(max_iter=int(p["n_estimators"]), learning_rate=float(p["learning_rate"]),
                                              max_depth=int(p["max_depth"]), random_state=seed)
    else:
        raise ValueError(f"Unknown regressor: {name}")
    return Pipeline([("preprocess", build_preprocessor(scale=True)), ("model", model)])


def train_and_evaluate(split: dict, name: str, params: dict | None = None, seed: int = RANDOM_SEED) -> dict:
    pipe = build_regressor(name, params, seed)
    t0 = time.perf_counter()
    pipe.fit(split["X_train"], split["y_reg_train"])
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = np.asarray(pipe.predict(split["X_test"]), dtype=float)
    predict_time = time.perf_counter() - t0
    y_true = split["y_reg_test"].to_numpy(dtype=float)
    metrics = regression_metrics(y_true, y_pred)
    metrics["fit_time_s"], metrics["predict_time_s"] = fit_time, predict_time
    return {"name": name, "pipeline": pipe, "params": {**DEFAULT_PARAMS[name], **(params or {})},
            "metrics": metrics, "y_true": y_true, "y_pred": y_pred, "residuals": y_true - y_pred}


def comparison_table(results: dict) -> pd.DataFrame:
    rows = [{"Model": n, "MAE": r["metrics"]["mae"], "MSE": r["metrics"]["mse"], "RMSE": r["metrics"]["rmse"],
             "R2": r["metrics"]["r2"], "Training time (s)": r["metrics"]["fit_time_s"],
             "Prediction time (s)": r["metrics"]["predict_time_s"]} for n, r in results.items()]
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    return table


def run_regression(df: pd.DataFrame, params: dict | None = None, test_size: float = TEST_SIZE,
                   seed: int = RANDOM_SEED) -> dict:
    split = make_split(df, test_size, seed)
    params = params or {}
    results = {n: train_and_evaluate(split, n, params.get(n), seed) for n in REGRESSOR_NAMES}
    return {"split": split, "results": results, "table": comparison_table(results)}


def predictions_frame(df: pd.DataFrame, split: dict, result: dict) -> pd.DataFrame:
    idx = split["X_test"].index
    out = pd.DataFrame(index=idx)
    if ID_COLUMN in df.columns:
        out[ID_COLUMN] = df.loc[idx, ID_COLUMN]
    out["Actual_Yield"] = result["y_true"].round(3)
    out["Predicted_Yield"] = result["y_pred"].round(3)
    out["Residual"] = result["residuals"].round(3)
    out["Abs_Error"] = np.abs(result["residuals"]).round(3)
    return out.reset_index(drop=True)
