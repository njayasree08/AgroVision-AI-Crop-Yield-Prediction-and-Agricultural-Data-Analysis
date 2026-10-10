"""Yield-category classification: KNN, Gaussian Naive Bayes, Decision Tree."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import CLASS_ORDER, CLS_TARGET, ID_COLUMN, RANDOM_SEED, TEST_SIZE
from .evaluation import classification_metrics, confusion, report_text_and_dict
from .preprocessing import build_preprocessor, make_split

CLASSIFIER_NAMES = ["KNN", "Naive Bayes", "Decision Tree"]

DEFAULT_PARAMS = {
    "KNN": {"n_neighbors": 7, "weights": "uniform", "metric": "euclidean"},
    "Naive Bayes": {"var_smoothing": 1e-9},
    "Decision Tree": {"max_depth": 6, "min_samples_split": 10, "min_samples_leaf": 4,
                      "criterion": "gini", "class_weight": "balanced"},
}


def build_classifier(name: str, params: dict | None = None, n_train: int | None = None,
                     seed: int = RANDOM_SEED) -> Pipeline:
    """Return an unfitted Pipeline(preprocess -> model). Scaling is part of the pipeline,
    so it is fitted on training data only. Scaling matters for KNN (distance based) and
    Naive Bayes; the tree is scale-invariant but shares the same preprocessing for fairness."""
    p = {**DEFAULT_PARAMS[name], **(params or {})}
    if name == "KNN":
        k = int(p["n_neighbors"])
        if n_train is not None:
            k = min(k, n_train)
        model = KNeighborsClassifier(n_neighbors=max(1, k), weights=p["weights"], metric=p["metric"])
    elif name == "Naive Bayes":
        model = GaussianNB(var_smoothing=float(p["var_smoothing"]))
    elif name == "Decision Tree":
        cw = p.get("class_weight")
        model = DecisionTreeClassifier(
            max_depth=None if p["max_depth"] in (None, 0) else int(p["max_depth"]),
            min_samples_split=int(p["min_samples_split"]), min_samples_leaf=int(p["min_samples_leaf"]),
            criterion=p["criterion"], class_weight=None if cw in (None, "none") else cw, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier: {name}")
    return Pipeline([("preprocess", build_preprocessor(scale=True)), ("model", model)])


def _cv_f1(pipeline: Pipeline, X, y, seed: int) -> float:
    min_class = int(y.value_counts().min())
    folds = min(5, min_class)
    if folds < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return float(cross_val_score(clone(pipeline), X, y, cv=cv, scoring="f1_macro").mean())


def train_and_evaluate(split: dict, name: str, params: dict | None = None, seed: int = RANDOM_SEED) -> dict:
    X_tr, X_te = split["X_train"], split["X_test"]
    y_tr, y_te = split["y_cls_train"], split["y_cls_test"]
    pipe = build_classifier(name, params, n_train=len(X_tr), seed=seed)

    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = pipe.predict(X_te)
    predict_time = time.perf_counter() - t0

    labels = [c for c in CLASS_ORDER if c in set(y_tr) | set(y_te)]
    metrics = classification_metrics(y_te, y_pred, labels)
    metrics["fit_time_s"], metrics["predict_time_s"] = fit_time, predict_time
    metrics["cv_f1_macro_train"] = _cv_f1(pipe, X_tr, y_tr, seed)
    text, rep = report_text_and_dict(y_te, y_pred, labels)
    proba = pipe.predict_proba(X_te) if hasattr(pipe, "predict_proba") else None
    return {"name": name, "pipeline": pipe, "params": {**DEFAULT_PARAMS[name], **(params or {})},
            "metrics": metrics, "confusion": confusion(y_te, y_pred, labels), "labels": labels,
            "report_text": text, "report_dict": rep, "y_pred": np.asarray(y_pred),
            "y_proba": proba, "classes": list(pipe.classes_)}


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        m = r["metrics"]
        rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision (macro)": m["precision_macro"],
                     "Recall (macro)": m["recall_macro"], "F1 (macro)": m["f1_macro"],
                     "F1 (weighted)": m["f1_weighted"], "CV F1 macro (train)": m["cv_f1_macro_train"],
                     "Training time (s)": m["fit_time_s"], "Prediction time (s)": m["predict_time_s"]})
    table = pd.DataFrame(rows).sort_values(["F1 (macro)", "Accuracy"], ascending=False).reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    return table


def run_classification(df: pd.DataFrame, params: dict | None = None, test_size: float = TEST_SIZE,
                       seed: int = RANDOM_SEED) -> dict:
    """Train all three classifiers on the same stratified split."""
    split = make_split(df, test_size, seed)
    params = params or {}
    results = {n: train_and_evaluate(split, n, params.get(n), seed) for n in CLASSIFIER_NAMES}
    return {"split": split, "results": results, "table": comparison_table(results)}


def knn_k_sweep(split: dict, ks: list[int], weights: str = "uniform", metric: str = "euclidean",
                seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Cross-validated macro-F1 on the TRAINING split for several K (test set stays untouched)."""
    rows = []
    for k in ks:
        pipe = build_classifier("KNN", {"n_neighbors": k, "weights": weights, "metric": metric},
                                n_train=len(split["X_train"]), seed=seed)
        rows.append({"K": k, "CV F1 macro (train)": _cv_f1(pipe, split["X_train"], split["y_cls_train"], seed)})
    return pd.DataFrame(rows)


def predictions_frame(df: pd.DataFrame, split: dict, result: dict) -> pd.DataFrame:
    idx = split["X_test"].index
    out = pd.DataFrame(index=idx)
    if ID_COLUMN in df.columns:
        out[ID_COLUMN] = df.loc[idx, ID_COLUMN]
    out["Actual"] = split["y_cls_test"]
    out["Predicted"] = result["y_pred"]
    out["Correct"] = out["Actual"] == out["Predicted"]
    if result["y_proba"] is not None:
        for j, c in enumerate(result["classes"]):
            out[f"P({c})"] = result["y_proba"][:, j].round(4)
    return out.reset_index(drop=True)
