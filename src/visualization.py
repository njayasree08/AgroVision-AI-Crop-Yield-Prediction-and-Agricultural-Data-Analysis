"""Plotly (interactive) and Matplotlib (dendrogram, tree) figure builders.

Every function returns a figure; none of them touches Streamlit.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: safe inside a web server
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

from .clustering import cut_height  # noqa: E402
from .config import CLASS_ORDER, REG_TARGET  # noqa: E402

DARK, MID, LIGHT = "#1B4332", "#40916C", "#B7E4C7"
COLORWAY = ["#1B4332", "#52B788", "#E9A23B", "#3A7CA5", "#B56576", "#8D6B94", "#6C757D", "#C8553D", "#95D5B2", "#264653"]
CATEGORY_COLORS = {"Low": "#C8553D", "Medium": "#E9A23B", "High": "#2D6A4F"}
LABELS = {
    "Annual_Rainfall_mm": "Annual rainfall (mm)", "Average_Temperature_C": "Average temperature (°C)",
    "Crop_Yield": "Crop yield (t/ha)", "Previous_Year_Yield": "Previous-year yield (t/ha)",
    "Fertilizer_Usage": "Fertilizer usage (kg/ha)", "Area_Hectares": "Area (ha)", "Crop_Type": "Crop",
}


def _style(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(size=16)) if title else None,
                      font=dict(family="Segoe UI, Arial, sans-serif", size=13, color="#1f2d27"),
                      colorway=COLORWAY, height=height, margin=dict(l=50, r=20, t=60 if title else 20, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="left", x=0))
    return fig


# ------------------------------------------------------------ data overview
def bar_counts(df: pd.DataFrame, col: str, title: str, color_map: dict | None = None) -> go.Figure:
    counts = df[col].value_counts().rename_axis(col).reset_index(name="Records")
    fig = px.bar(counts, x=col, y="Records", text="Records", color=col, color_discrete_map=color_map,
                 labels={col: LABELS.get(col, col.replace("_", " "))})
    fig.update_layout(showlegend=False)
    fig.update_traces(textposition="outside")
    return _style(fig, title, 360)


def seasonal_crop_distribution(df: pd.DataFrame) -> go.Figure:
    g = df.groupby(["Season", "Crop_Type"]).size().reset_index(name="Records")
    fig = px.bar(g, x="Season", y="Records", color="Crop_Type", barmode="stack", labels={"Crop_Type": "Crop"})
    return _style(fig, "Crop distribution by season", 420)


def correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    corr = df[columns].apply(pd.to_numeric, errors="coerce").corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    labels=dict(color="Pearson r"))
    return _style(fig, "Correlation heatmap (numeric columns)", 560)


def scatter_vs_yield(df: pd.DataFrame, x: str, title: str) -> go.Figure:
    fig = px.scatter(df, x=x, y=REG_TARGET, color="Crop_Type", opacity=0.7,
                     hover_data=[c for c in ("Record_ID", "State", "Season", "Soil_Type") if c in df.columns],
                     labels={**LABELS, "Crop_Type": "Crop"})
    return _style(fig, title, 460)


def nutrient_comparison(df: pd.DataFrame) -> go.Figure:
    long = df.melt(id_vars=["Soil_Type"], value_vars=["Nitrogen", "Phosphorus", "Potassium"],
                   var_name="Nutrient", value_name="Level (kg/ha)")
    fig = px.box(long, x="Soil_Type", y="Level (kg/ha)", color="Nutrient", labels={"Soil_Type": "Soil type"})
    return _style(fig, "Soil nutrient levels by soil type", 440)


def box_plot(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.box(df, y=col, points="outliers", labels={col: LABELS.get(col, col)})
    return _style(fig, f"Distribution and IQR outliers: {LABELS.get(col, col)}", 380)


def missing_bar(summary: pd.DataFrame) -> go.Figure:
    s = summary[summary["Missing"] > 0]
    fig = px.bar(s, x="Column", y="Missing", text="Missing")
    return _style(fig, "Missing values per column", 360)


# --------------------------------------------------------- classification
def confusion_matrix_fig(cm: np.ndarray, labels: list[str], title: str) -> go.Figure:
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Greens", aspect="auto",
                    labels=dict(x="Predicted class", y="Actual class", color="Records"))
    return _style(fig, title, 380)


def metric_comparison_bar(table: pd.DataFrame, metrics: list[str], title: str, y_title: str = "Score") -> go.Figure:
    long = table.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name=y_title)
    fig = px.bar(long, x="Model", y=y_title, color="Metric", barmode="group", text_auto=".3f")
    return _style(fig, title, 430)


def single_metric_bar(table: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = px.bar(table, x="Model", y=metric, color="Model", text_auto=".3f")
    fig.update_layout(showlegend=False)
    return _style(fig, title, 380)


def feature_importance_bar(table: pd.DataFrame, title: str, top: int = 12) -> go.Figure:
    t = table.head(top).iloc[::-1]
    fig = px.bar(t, x="Importance", y="Feature", orientation="h", text_auto=".3f")
    return _style(fig, title, 420)


def knn_k_line(sweep: pd.DataFrame) -> go.Figure:
    fig = px.line(sweep, x="K", y="CV F1 macro (train)", markers=True)
    return _style(fig, "KNN: cross-validated macro-F1 for different K (training data only)", 360)


def decision_tree_figure(pipeline, display_depth: int = 3):
    """Matplotlib figure of the fitted tree (top `display_depth` levels shown)."""
    tree = pipeline.named_steps["model"]
    names = [n.split("__", 1)[-1] for n in pipeline.named_steps["preprocess"].get_feature_names_out()]
    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(tree, feature_names=names, class_names=[str(c) for c in tree.classes_], filled=True, rounded=True,
              max_depth=display_depth, fontsize=8, impurity=False, ax=ax)
    ax.set_title(f"Decision tree (first {display_depth} levels shown)")
    return fig


# ------------------------------------------------------------- clustering
def dendrogram_figure(Z: np.ndarray, n_clusters: int, p: int = 30, method: str = "ward"):
    """SciPy dendrogram (truncated to the last `p` merged clusters) with the cut line."""
    fig, ax = plt.subplots(figsize=(11, 5))
    cut = cut_height(Z, n_clusters)
    dendrogram(Z, truncate_mode="lastp", p=p, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=9,
               color_threshold=cut, above_threshold_color="#6C757D", ax=ax)
    if cut is not None:
        ax.axhline(cut, color="#C8553D", linestyle="--", linewidth=1.2, label=f"cut for {n_clusters} clusters")
        ax.legend(loc="upper right")
    ax.set_title(f"Agglomerative clustering dendrogram ({method} linkage, truncated)")
    ax.set_xlabel("Cluster (number of records in brackets)")
    ax.set_ylabel("Merge distance")
    fig.tight_layout()
    return fig


def pca_scatter(coords: np.ndarray, labels: np.ndarray, evr: np.ndarray, hover: pd.DataFrame | None = None) -> go.Figure:
    d = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": [f"Cluster {c}" for c in labels]})
    if hover is not None:
        for col in hover.columns:
            d[col] = hover[col].to_numpy()
    fig = px.scatter(d.sort_values("Cluster"), x="PC1", y="PC2", color="Cluster", opacity=0.75,
                     hover_data=list(hover.columns) if hover is not None else None,
                     labels={"PC1": f"PC1 ({evr[0] * 100:.1f}% variance)", "PC2": f"PC2 ({evr[1] * 100:.1f}% variance)"})
    return _style(fig, "Clusters projected onto two principal components", 480)


def cluster_size_bar(labels: np.ndarray) -> go.Figure:
    s = pd.Series(labels).value_counts().sort_index()
    d = pd.DataFrame({"Cluster": [f"Cluster {c}" for c in s.index], "Records": s.values})
    fig = px.bar(d, x="Cluster", y="Records", text="Records", color="Cluster")
    fig.update_layout(showlegend=False)
    return _style(fig, "Cluster sizes", 360)


def silhouette_line(table: pd.DataFrame) -> go.Figure:
    fig = px.line(table, x="Clusters", y="Silhouette", markers=True)
    return _style(fig, "Silhouette score by number of clusters (higher is better)", 340)


# ------------------------------------------------------------- regression
def actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> go.Figure:
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="Test records",
                             marker=dict(color=MID, opacity=0.7),
                             hovertemplate="Actual %{x:.2f}<br>Predicted %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(color="#C8553D", dash="dash")))
    fig.update_xaxes(title="Actual yield (t/ha)")
    fig.update_yaxes(title="Predicted yield (t/ha)")
    return _style(fig, title, 420)


def residual_plot(y_pred: np.ndarray, residuals: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=y_pred, y=residuals, mode="markers", marker=dict(color=MID, opacity=0.7),
                               name="Residual", hovertemplate="Predicted %{x:.2f}<br>Residual %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="#C8553D")
    fig.update_xaxes(title="Predicted yield (t/ha)")
    fig.update_yaxes(title="Residual = actual - predicted (t/ha)")
    return _style(fig, title, 380)


def residual_histogram(residuals: np.ndarray, title: str) -> go.Figure:
    fig = px.histogram(x=residuals, nbins=25, labels={"x": "Residual (t/ha)"})
    fig.update_traces(marker_color=MID)
    return _style(fig, title, 340)


def yield_distribution_seaborn(df: pd.DataFrame):
    """Seaborn histogram of crop yield, stacked by crop (static Matplotlib figure)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.histplot(data=df, x=REG_TARGET, hue="Crop_Type", multiple="stack", bins=30, palette=COLORWAY[:df["Crop_Type"].nunique()], ax=ax)
    ax.set_xlabel("Crop yield (t/ha)")
    ax.set_ylabel("Records")
    ax.set_title("Yield distribution by crop")
    fig.tight_layout()
    return fig
