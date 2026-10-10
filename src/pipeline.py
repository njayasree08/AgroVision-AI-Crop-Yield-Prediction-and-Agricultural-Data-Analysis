"""End-to-end orchestration (no UI code): full analysis, model saving, table export."""
from __future__ import annotations

import pandas as pd

from . import classification as clf
from . import clustering as clu
from . import regression as reg
from .config import CLS_TARGET, FEATURES, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED, REG_TARGET, TEST_SIZE
from .evaluation import best_model_summary
from .model_io import save_bundle, save_json, slug
from .preprocessing import prepare_dataset


def analyse_clustering(df: pd.DataFrame, linkage: str = "ward", n_clusters: int | None = None,
                       features: list[str] | None = None) -> dict:
    """Cluster with k chosen by best silhouette (k = 2..8) unless n_clusters is given."""
    X, names = clu.build_cluster_matrix(df, features)
    table = clu.compare_cluster_counts(X, linkage)
    if n_clusters is None:
        valid = table.dropna(subset=["Silhouette"])
        n_clusters = int(valid.sort_values("Silhouette", ascending=False).iloc[0]["Clusters"]) if len(valid) else 3
    labels = clu.run_agglomerative(X, n_clusters, linkage)
    profile = clu.cluster_profiles(df, labels)
    return {"X": X, "feature_names": names, "labels": labels, "n_clusters": n_clusters, "linkage": linkage,
            "silhouette": clu.safe_silhouette(X, labels), "table": table, "profile": profile,
            "interpretation": clu.interpret_clusters(profile, df)}


def run_full_analysis(df: pd.DataFrame, seed: int = RANDOM_SEED, test_size: float = TEST_SIZE) -> dict:
    """Run all three techniques with default parameters on one dataset."""
    cls = clf.run_classification(df, None, test_size, seed)
    rg = reg.run_regression(df, None, test_size, seed)
    cl = analyse_clustering(df)
    summary = best_model_summary(cls["table"], rg["table"], cl["table"], cl["linkage"])
    return {"classification": cls, "regression": rg, "clustering": cl, "summary": summary}


def best_name(table: pd.DataFrame) -> str:
    return str(table.iloc[0]["Model"])  # tables are sorted best-first


def train_and_save_all(df: pd.DataFrame, thresholds: tuple[float, float], seed: int = RANDOM_SEED,
                       test_size: float = TEST_SIZE, directory=MODELS_DIR, analysis: dict | None = None) -> dict:
    """Train everything and save each full pipeline plus 'best_classifier' / 'best_regressor'."""
    analysis = analysis or run_full_analysis(df, seed, test_size)
    common = {"features": FEATURES, "thresholds": tuple(thresholds), "n_train": len(analysis["classification"]["split"]["X_train"]),
              "seed": seed, "test_size": test_size}
    saved = {}
    for kind, key, target, best_table in (("cls", "classification", CLS_TARGET, analysis["classification"]["table"]),
                                          ("reg", "regression", REG_TARGET, analysis["regression"]["table"])):
        best = best_name(best_table)
        for name, r in analysis[key]["results"].items():
            bundle = {**common, "pipeline": r["pipeline"], "name": name, "target": target,
                      "metrics": r["metrics"], "params": r["params"]}
            save_bundle(bundle, f"{kind}_{slug(name)}.joblib", directory)
            if name == best:
                save_bundle(bundle, "best_classifier.joblib" if kind == "cls" else "best_regressor.joblib", directory)
                saved[f"best_{key}"] = name
    save_json({"best_classifier": saved.get("best_classification"), "best_regressor": saved.get("best_regression"),
               "summary": analysis["summary"], "thresholds": list(thresholds), "n_records": len(df),
               "classification_table": analysis["classification"]["table"].to_dict("records"),
               "regression_table": analysis["regression"]["table"].to_dict("records")},
              "training_summary.json", directory)
    return saved


def export_tables(analysis: dict, df: pd.DataFrame, directory=OUTPUTS_DIR) -> list[str]:
    """Write the main result tables as CSV files into outputs/."""
    directory.mkdir(parents=True, exist_ok=True)
    cls, rg, cl = analysis["classification"], analysis["regression"], analysis["clustering"]
    best_c, best_r = best_name(cls["table"]), best_name(rg["table"])
    files = {
        "processed_dataset.csv": df,
        "classification_comparison.csv": cls["table"],
        "classification_predictions.csv": clf.predictions_frame(df, cls["split"], cls["results"][best_c]),
        "regression_comparison.csv": rg["table"],
        "regression_predictions.csv": reg.predictions_frame(df, rg["split"], rg["results"][best_r]),
        "cluster_comparison.csv": cl["table"],
        "cluster_profiles.csv": cl["profile"],
    }
    assign = df.reset_index(drop=True).copy()
    assign["Cluster"] = cl["labels"]
    files["cluster_assignments.csv"] = assign
    for name, frame in files.items():
        frame.to_csv(directory / name, index=False)
    return list(files)
