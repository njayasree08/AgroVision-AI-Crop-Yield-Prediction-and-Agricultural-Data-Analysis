"""Tests for splitting, leakage prevention, the three task modules and model persistence."""
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src import classification as clf
from src import clustering as clu
from src import regression as reg
from src.config import CLS_TARGET, FEATURES, REG_TARGET
from src.data_generator import generate_dataset
from src.model_io import load_bundle
from src.pipeline import analyse_clustering, train_and_save_all, run_full_analysis
from src.prediction import predict_one
from src.preprocessing import build_preprocessor, make_split, prepare_dataset


@lru_cache(maxsize=1)
def data():
    return prepare_dataset(generate_dataset())


@lru_cache(maxsize=1)
def cls_run():
    return clf.run_classification(data()[0])


@lru_cache(maxsize=1)
def reg_run():
    return reg.run_regression(data()[0])


# ------------------------------------------------------------------ split
def test_split_is_disjoint_stratified_and_reproducible():
    df = data()[0]
    s1, s2 = make_split(df), make_split(df)
    assert set(s1["X_train"].index).isdisjoint(s1["X_test"].index)
    assert len(s1["X_train"]) + len(s1["X_test"]) == len(df)
    assert list(s1["X_test"].index) == list(s2["X_test"].index)
    overall = df[CLS_TARGET].value_counts(normalize=True)
    test_share = s1["y_cls_test"].value_counts(normalize=True)
    assert (overall - test_share).abs().max() < 0.03


def test_no_target_columns_in_model_inputs():
    split = make_split(data()[0])
    assert list(split["X_train"].columns) == FEATURES
    assert REG_TARGET not in split["X_train"].columns and CLS_TARGET not in split["X_train"].columns


def test_preprocessing_is_fitted_on_training_data_only():
    split = make_split(data()[0])
    pre = build_preprocessor().fit(split["X_train"])
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    full_mean = data()[0]["Annual_Rainfall_mm"].mean()
    idx = list(pre.transformers_[0][2]).index("Annual_Rainfall_mm")
    assert scaler.mean_[idx] == pytest.approx(split["X_train"]["Annual_Rainfall_mm"].mean(), rel=0.02)
    assert scaler.mean_[idx] != full_mean


# --------------------------------------------------------- classification
def test_three_classifiers_with_computed_metrics():
    run = cls_run()
    assert set(run["results"]) == {"KNN", "Naive Bayes", "Decision Tree"}
    y_te = run["split"]["y_cls_test"]
    for name, r in run["results"].items():
        m = r["metrics"]
        assert 0 <= m["accuracy"] <= 1 and 0 <= m["f1_macro"] <= 1
        assert m["accuracy"] == pytest.approx(accuracy_score(y_te, r["y_pred"]))
        assert m["f1_macro"] == pytest.approx(f1_score(y_te, r["y_pred"], average="macro"))
        assert r["confusion"].sum() == len(y_te)
        assert m["fit_time_s"] > 0 and m["predict_time_s"] >= 0
    table = run["table"]
    assert list(table["Rank"]) == [1, 2, 3]
    assert table["F1 (macro)"].is_monotonic_decreasing


def test_classifiers_beat_random_guessing():
    best = cls_run()["table"].iloc[0]
    assert best["Accuracy"] > 0.5  # three balanced classes -> chance is about 0.33


def test_knn_k_is_configurable_and_clamped():
    split = make_split(data()[0])
    sweep = clf.knn_k_sweep(split, [1, 5, 15])
    assert list(sweep["K"]) == [1, 5, 15] and sweep["CV F1 macro (train)"].notna().all()
    pipe = clf.build_classifier("KNN", {"n_neighbors": 9999}, n_train=len(split["X_train"]))
    assert pipe.named_steps["model"].n_neighbors == len(split["X_train"])


def test_decision_tree_depth_parameter_and_importance():
    split = make_split(data()[0])
    r = clf.train_and_evaluate(split, "Decision Tree", {"max_depth": 2})
    assert r["pipeline"].named_steps["model"].get_depth() <= 2
    from src.evaluation import feature_importance_table
    imp = feature_importance_table(r["pipeline"])
    assert imp["Importance"].sum() == pytest.approx(1.0)
    assert set(imp["Feature"]) <= set(FEATURES)


# ------------------------------------------------------------- regression
def test_three_regressors_with_consistent_metrics():
    run = reg_run()
    assert len(run["results"]) == 3 and reg.LINEAR in run["results"] and reg.FOREST in run["results"]
    for r in run["results"].values():
        m = r["metrics"]
        assert m["rmse"] == pytest.approx(np.sqrt(m["mse"]))
        assert m["mae"] <= m["rmse"] + 1e-9
        assert m["mse"] == pytest.approx(np.mean((r["y_true"] - r["y_pred"]) ** 2))
        assert np.allclose(r["residuals"], r["y_true"] - r["y_pred"])
    assert run["table"]["RMSE"].is_monotonic_increasing
    assert run["table"]["R2"].max() > 0.5


def test_regression_and_classification_share_test_rows():
    assert list(cls_run()["split"]["X_test"].index) == list(reg_run()["split"]["X_test"].index)


# ------------------------------------------------------------- clustering
def test_agglomerative_clustering_outputs():
    df = data()[0]
    X, names = clu.build_cluster_matrix(df)
    assert X.shape[0] == len(df) and X.shape[1] == len(names)
    labels = clu.run_agglomerative(X, 4, "ward")
    assert sorted(set(labels)) == [1, 2, 3, 4]
    sil = clu.safe_silhouette(X, labels)
    assert -1 <= sil <= 1
    prof = clu.cluster_profiles(df, labels)
    assert prof["Records"].sum() == len(df) and len(prof) == 4
    assert len(clu.interpret_clusters(prof, df)) == 4


def test_clustering_does_not_use_target():
    _, names = clu.build_cluster_matrix(data()[0])
    assert not any("Crop_Yield" in n or "Yield_Category" in n or "Previous_Year_Yield" in n for n in names)


def test_cluster_parameter_validation():
    assert clu.validate_cluster_params(700, 4, "ward", ["a", "b"]) == []
    assert clu.validate_cluster_params(700, 1, "ward", ["a", "b"])
    assert clu.validate_cluster_params(5, 5, "ward", ["a", "b"])
    assert clu.validate_cluster_params(700, 3, "bogus", ["a", "b"])
    assert clu.validate_cluster_params(700, 3, "ward", ["a"])
    with pytest.raises(ValueError):
        clu.run_agglomerative(np.zeros((10, 3)), 50, "ward")
    assert clu.safe_silhouette(np.zeros((5, 2)), np.ones(5, dtype=int)) is None


@pytest.mark.parametrize("linkage", clu.LINKAGES)
def test_all_linkages_and_dendrogram_matrix(linkage):
    X, _ = clu.build_cluster_matrix(data()[0])
    Z, idx, sampled = clu.compute_linkage_matrix(X, linkage, max_samples=100)
    assert sampled and len(idx) == 100 and Z.shape == (99, 4)
    assert clu.cut_height(Z, 4) is not None and clu.cut_height(Z, 1000) is None
    assert len(set(clu.run_agglomerative(X, 3, linkage))) == 3


def test_pca_projection_shape():
    X, _ = clu.build_cluster_matrix(data()[0])
    coords, evr = clu.pca_projection(X)
    assert coords.shape == (len(X), 2) and 0 < evr.sum() <= 1


def test_analyse_clustering_chooses_valid_k():
    res = analyse_clustering(data()[0])
    assert 2 <= res["n_clusters"] <= 8 and len(res["labels"]) == len(data()[0])


# ----------------------------------------------------- persistence + predict
def test_save_load_and_predict_with_saved_pipelines():
    df, thresholds, _ = data()
    with tempfile.TemporaryDirectory() as tmp:
        analysis = {"classification": cls_run(), "regression": reg_run(), "clustering": analyse_clustering(df),
                    "summary": ["test"]}
        train_and_save_all(df, thresholds, directory=Path(tmp), analysis=analysis)
        reg_b, cls_b = load_bundle("best_regressor.joblib", Path(tmp)), load_bundle("best_classifier.joblib", Path(tmp))
        assert reg_b is not None and cls_b is not None
        assert load_bundle("missing.joblib", Path(tmp)) is None
        row = df.iloc[0]
        inputs = {c: row[c] for c in FEATURES}
        for c in ("Soil_pH", "Nitrogen", "Potassium", "Fertilizer_Usage", "Previous_Year_Yield"):
            if pd.isna(inputs[c]):
                inputs[c] = float(df[c].median())
        out = predict_one(reg_b, cls_b, inputs)
        assert out["yield_estimate"] > 0
        assert out["classifier_category"] in {"Low", "Medium", "High"}
        assert out["category_from_estimate"] in {"Low", "Medium", "High"}
        assert sum(out["class_probabilities"].values()) == pytest.approx(1.0)
        with pytest.raises(ValueError):
            predict_one(reg_b, cls_b, {**inputs, "Soil_pH": 99})


def test_full_analysis_summary_is_generated_from_results():
    result = run_full_analysis(data()[0])
    text = " ".join(result["summary"])
    assert result["classification"]["table"].iloc[0]["Model"] in text
    assert result["regression"]["table"].iloc[0]["Model"] in text
