"""Metrics, rankings, generated summaries and the HTML report."""
from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, precision_recall_fscore_support,
                             r2_score)

from .config import CATEGORICAL_FEATURES, CLS_TARGET, REG_TARGET


# ----------------------------------------------------------- classification
def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = float(p), float(r), float(f)
    return out


def confusion(y_true, y_pred, labels: list[str]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def report_text_and_dict(y_true, y_pred, labels: list[str]) -> tuple[str, dict]:
    text = classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=3)
    d = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    return text, d


# --------------------------------------------------------------- regression
def regression_metrics(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {"mae": float(mean_absolute_error(y_true, y_pred)), "mse": mse,
            "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y_true, y_pred))}


# -------------------------------------------------------- feature importance
def feature_importance_table(pipeline, aggregate: bool = True) -> pd.DataFrame | None:
    """Importance from a fitted preprocess+model pipeline; None if unsupported."""
    model = pipeline.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    table = pd.DataFrame({"Feature": names, "Importance": np.asarray(importances, dtype=float)})
    table["Feature"] = table["Feature"].str.replace(r"^(num|cat)__", "", regex=True)
    if aggregate:
        def origin(name: str) -> str:
            for cat in CATEGORICAL_FEATURES:
                if name.startswith(cat + "_"):
                    return cat
            return name
        table["Feature"] = table["Feature"].map(origin)
        table = table.groupby("Feature", as_index=False)["Importance"].sum()
    return table.sort_values("Importance", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------- insights
def generate_key_findings(df: pd.DataFrame) -> list[str]:
    """Plain-language findings computed from the data (nothing hard-coded)."""
    findings = []
    if REG_TARGET not in df.columns or df.empty:
        return findings
    by_crop = df.groupby("Crop_Type")[REG_TARGET].mean().sort_values(ascending=False)
    findings.append(f"Highest average yield: **{by_crop.index[0]}** ({by_crop.iloc[0]:.2f} t/ha); "
                    f"lowest: **{by_crop.index[-1]}** ({by_crop.iloc[-1]:.2f} t/ha).")
    if "Irrigation_Type" in df:
        by_irr = df.groupby("Irrigation_Type")[REG_TARGET].mean().sort_values(ascending=False)
        findings.append(f"Irrigation type with the highest mean yield: **{by_irr.index[0]}** ({by_irr.iloc[0]:.2f} t/ha), "
                        f"compared with **{by_irr.index[-1]}** ({by_irr.iloc[-1]:.2f} t/ha). "
                        "Crop mix differs between groups, so this is not a causal effect.")
    if "Season" in df:
        by_season = df.groupby("Season")[REG_TARGET].mean().sort_values(ascending=False)
        findings.append(f"Season with the highest mean yield: **{by_season.index[0]}** ({by_season.iloc[0]:.2f} t/ha).")
    numeric = [c for c in ("Annual_Rainfall_mm", "Average_Temperature_C", "Soil_pH", "Nitrogen", "Phosphorus",
                           "Potassium", "Fertilizer_Usage", "Area_Hectares") if c in df]
    if numeric:
        corr = df[numeric + [REG_TARGET]].corr()[REG_TARGET].drop(REG_TARGET).dropna()
        if not corr.empty:
            top = corr.abs().sort_values(ascending=False).index[0]
            findings.append(f"Numeric input most correlated with yield (linear, Pearson): **{top}** (r = {corr[top]:+.2f}).")
    if CLS_TARGET in df:
        share = df[CLS_TARGET].value_counts(normalize=True).mul(100).round(1)
        findings.append("Yield categories: " + ", ".join(f"{k} {v}%" for k, v in share.items()) + ".")
    return findings


def best_model_summary(cls_table: pd.DataFrame | None, reg_table: pd.DataFrame | None,
                       cluster_table: pd.DataFrame | None, linkage: str = "ward") -> list[str]:
    lines = []
    if cls_table is not None and len(cls_table):
        b = cls_table.sort_values(["F1 (macro)", "Accuracy"], ascending=False).iloc[0]
        lines.append(f"Classification: **{b['Model']}** has the best test macro-F1 ({b['F1 (macro)']:.3f}) "
                     f"and accuracy {b['Accuracy']:.3f}.")
    if reg_table is not None and len(reg_table):
        b = reg_table.sort_values("RMSE").iloc[0]
        lines.append(f"Regression: **{b['Model']}** has the lowest test RMSE ({b['RMSE']:.3f} t/ha) "
                     f"with R² = {b['R2']:.3f}.")
    if cluster_table is not None and len(cluster_table):
        valid = cluster_table.dropna(subset=["Silhouette"])
        if len(valid):
            b = valid.sort_values("Silhouette", ascending=False).iloc[0]
            lines.append(f"Clustering ({linkage} linkage): k = **{int(b['Clusters'])}** gives the highest silhouette "
                         f"score ({b['Silhouette']:.3f}). Silhouette is a guide, not proof of meaningful groups.")
    lines.append("These results come from synthetic data. They demonstrate the workflow but do not "
                 "establish real-world predictive performance.")
    return lines


# -------------------------------------------------------------- HTML report
def _table_html(df: pd.DataFrame | None) -> str:
    return "<p><em>Not available.</em></p>" if df is None else df.to_html(index=False, border=0, classes="t", float_format=lambda v: f"{v:.4g}")


def build_html_report(sections: dict, summary_lines: list[str], dataset_info: dict) -> str:
    """Self-contained HTML report. sections: {title: DataFrame | str}."""
    esc = html.escape

    def md(text: str) -> str:  # tiny **bold** support, escaped first
        parts = esc(text).split("**")
        return "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(parts))

    body = [f"<h1>AgroVision AI - Analysis Report</h1><p class='m'>Generated {datetime.now():%Y-%m-%d %H:%M}</p>",
            "<div class='n'>Synthetic dataset notice: unless you uploaded your own data, all records are synthetic "
            "and the results illustrate the workflow only.</div>",
            "<h2>Dataset</h2><ul>" + "".join(f"<li>{esc(str(k))}: {esc(str(v))}</li>" for k, v in dataset_info.items()) + "</ul>",
            "<h2>Summary</h2><ul>" + "".join(f"<li>{md(s)}</li>" for s in summary_lines) + "</ul>"]
    for title, content in sections.items():
        body.append(f"<h2>{esc(title)}</h2>")
        body.append(_table_html(content) if isinstance(content, pd.DataFrame) or content is None
                    else f"<pre>{esc(str(content))}</pre>")
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:2rem auto;color:#1f2d27;padding:0 1rem}"
           "h1{color:#1B4332}h2{color:#2D6A4F;border-bottom:2px solid #B7E4C7;padding-bottom:.2rem;margin-top:2rem}"
           "table.t{border-collapse:collapse;width:100%;font-size:.9rem}.t th{background:#1B4332;color:#fff;text-align:left}"
           ".t th,.t td{padding:.35rem .6rem;border-bottom:1px solid #d8e5dd}.m{color:#6c757d}"
           ".n{background:#fff6e0;border-left:4px solid #e9a23b;padding:.6rem 1rem;margin:1rem 0}pre{background:#f1f8f4;padding:1rem;overflow:auto}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>AgroVision AI Report</title><style>{css}</style></head><body>{''.join(body)}</body></html>"
