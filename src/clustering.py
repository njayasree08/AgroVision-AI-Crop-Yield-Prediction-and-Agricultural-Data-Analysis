"""Agglomerative (hierarchical) clustering, dendrogram data, PCA and cluster profiling.

Clustering is unsupervised: neither Crop_Yield nor Yield_Category is used to form
clusters. Yield is only added afterwards to *describe* the clusters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

from .config import CLUSTER_DEFAULT_FEATURES, NUMERIC_FEATURES, RANDOM_SEED, REG_TARGET
from .preprocessing import build_preprocessor

LINKAGES = ["ward", "complete", "average", "single"]
MAX_CLUSTERS = 10


def validate_cluster_params(n_rows: int, n_clusters: int, linkage: str, features: list[str]) -> list[str]:
    errors = []
    if linkage not in LINKAGES:
        errors.append(f"Linkage must be one of {LINKAGES}.")
    if len(features) < 2:
        errors.append("Select at least two features.")
    upper = min(MAX_CLUSTERS, n_rows - 1)
    if not (2 <= n_clusters <= upper):
        errors.append(f"Number of clusters must be between 2 and {upper} for {n_rows} records.")
    return errors


def build_cluster_matrix(df: pd.DataFrame, features: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Impute, one-hot encode categoricals and standardise numerics (unsupervised, whole dataset)."""
    features = features or CLUSTER_DEFAULT_FEATURES
    numeric = [f for f in features if f in NUMERIC_FEATURES]
    categorical = [f for f in features if f not in NUMERIC_FEATURES]
    pre = build_preprocessor(numeric, categorical, scale=True)
    X = pre.fit_transform(df[features])
    return np.asarray(X, dtype=float), [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]


def run_agglomerative(X: np.ndarray, n_clusters: int, linkage: str = "ward") -> np.ndarray:
    """Return 1-based cluster labels (labels are arbitrary identifiers, not a ranking)."""
    errors = validate_cluster_params(len(X), n_clusters, linkage, ["a", "b"])
    errors = [e for e in errors if "features" not in e]
    if errors:
        raise ValueError(" ".join(errors))
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return model.fit_predict(X) + 1


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette needs 2 <= number of clusters <= n_samples - 1."""
    n_labels = len(set(labels))
    if n_labels < 2 or n_labels > len(X) - 1:
        return None
    return float(silhouette_score(X, labels))


def compare_cluster_counts(X: np.ndarray, linkage: str, k_values=range(2, 9)) -> pd.DataFrame:
    rows = []
    for k in k_values:
        if k > len(X) - 1:
            continue
        labels = run_agglomerative(X, k, linkage)
        sil = safe_silhouette(X, labels)
        rows.append({"Clusters": k, "Silhouette": sil,
                     "Davies-Bouldin (lower is better)": float(davies_bouldin_score(X, labels)),
                     "Smallest cluster": int(pd.Series(labels).value_counts().min())})
    return pd.DataFrame(rows)


def compute_linkage_matrix(X: np.ndarray, method: str, max_samples: int = 150, seed: int = RANDOM_SEED):
    """SciPy linkage on (a random sample of) the data so the dendrogram stays readable."""
    n = len(X)
    sampled = n > max_samples
    idx = np.sort(np.random.default_rng(seed).choice(n, size=max_samples, replace=False)) if sampled else np.arange(n)
    Z = scipy_linkage(X[idx], method=method, metric="euclidean")
    return Z, idx, sampled


def cut_height(Z: np.ndarray, n_clusters: int) -> float | None:
    """Dendrogram height that yields n_clusters flat clusters (None if impossible)."""
    if n_clusters < 2 or n_clusters > len(Z):
        return None
    return float((Z[-(n_clusters - 1), 2] + Z[-n_clusters, 2]) / 2)


def pca_projection(X: np.ndarray, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[1] < 2:
        raise ValueError("PCA needs at least two encoded columns.")
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def _mode(s: pd.Series):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else "n/a"


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    d = df.reset_index(drop=True).copy()
    d["Cluster"] = np.asarray(labels)
    num = [c for c in NUMERIC_FEATURES + [REG_TARGET] if c in d.columns]
    prof = d.groupby("Cluster")[num].mean().round(2)
    prof.insert(0, "Records", d["Cluster"].value_counts().sort_index())
    prof.insert(1, "Share_%", (prof["Records"] / len(d) * 100).round(1))
    for col, name in (("Crop_Type", "Top_Crop"), ("Season", "Top_Season"),
                      ("Soil_Type", "Top_Soil"), ("Irrigation_Type", "Top_Irrigation")):
        if col in d.columns:
            prof[name] = d.groupby("Cluster")[col].agg(_mode)
    return prof.reset_index()


def interpret_clusters(profile: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    """Generate plain-language cluster descriptions by comparing each cluster with the overall mean."""
    def rel(cluster_val, overall):
        if overall == 0 or pd.isna(overall):
            return "similar to"
        diff = (cluster_val - overall) / abs(overall) * 100
        if abs(diff) < 5:
            return "about average for"
        return f"{abs(diff):.0f}% {'above' if diff > 0 else 'below'}"

    lines = []
    for _, row in profile.iterrows():
        bits = []
        for col, label in (("Annual_Rainfall_mm", "rainfall"), ("Average_Temperature_C", "temperature"),
                           ("Fertilizer_Usage", "fertilizer use"), (REG_TARGET, "mean yield")):
            if col in profile.columns:
                bits.append(f"{label} {rel(row[col], df[col].mean())} the dataset average")
        crop = f" Most common crop: {row['Top_Crop']} ({row.get('Top_Season', 'n/a')} season)," \
               f" soil: {row.get('Top_Soil', 'n/a')}." if "Top_Crop" in profile.columns else ""
        lines.append(f"**Cluster {int(row['Cluster'])}** ({int(row['Records'])} records, {row['Share_%']}%): "
                     + "; ".join(bits) + "." + crop)
    return lines
