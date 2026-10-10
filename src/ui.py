"""Streamlit helpers shared by all pages (page setup, cached data/training, display wrappers)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .classification import run_classification
from .config import ASSETS_DIR
from .data_loader import load_dataset
from .pipeline import analyse_clustering, run_full_analysis
from .preprocessing import prepare_dataset
from .regression import run_regression

BUILTIN_LABEL = "Built-in synthetic dataset (700 records)"


def setup_page(title: str) -> None:
    """Must be the first Streamlit call on every page."""
    st.set_page_config(page_title=f"{title} | AgroVision AI", layout="wide", initial_sidebar_state="expanded")
    css = ASSETS_DIR / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    st.sidebar.markdown("### AgroVision AI")
    st.sidebar.caption("Crop yield prediction and agricultural data analysis")
    st.sidebar.caption("Data source: " + st.session_state.get("data_source", BUILTIN_LABEL))
    st.markdown(f"<h1 class='page-title'>{title}</h1>", unsafe_allow_html=True)


def synthetic_notice() -> None:
    if st.session_state.get("data_source", BUILTIN_LABEL) == BUILTIN_LABEL:
        st.info("The built-in dataset is **synthetic** (generated with a fixed random seed). Results demonstrate "
                "the workflow and are not real field measurements or real-world performance.")


# ------------------------------------------------------------------- data
def get_raw_dataset() -> pd.DataFrame:
    uploaded = st.session_state.get("uploaded_raw")
    return uploaded if uploaded is not None else _builtin_raw()


@st.cache_data(show_spinner=False)
def _builtin_raw() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner=False)
def _prepare(raw: pd.DataFrame):
    return prepare_dataset(raw)


def get_dataset():
    """Return (clean_df, thresholds, cleaning_report) for the active dataset, or stop the page with an error."""
    try:
        return _prepare(get_raw_dataset())
    except ValueError as exc:
        st.error(f"The active dataset cannot be used: {exc}")
        st.stop()


# ------------------------------------------------- cached model training
@st.cache_resource(show_spinner="Training classifiers...")
def cached_classification(df: pd.DataFrame, params: dict, test_size: float, seed: int):
    return run_classification(df, params, test_size, seed)


@st.cache_resource(show_spinner="Training regression models...")
def cached_regression(df: pd.DataFrame, params: dict, test_size: float, seed: int):
    return run_regression(df, params, test_size, seed)


@st.cache_resource(show_spinner="Running all analyses with default settings...")
def cached_full_analysis(df: pd.DataFrame, seed: int, test_size: float):
    return run_full_analysis(df, seed, test_size)


@st.cache_data(show_spinner="Clustering...")
def cached_cluster_matrix(df: pd.DataFrame, features: tuple):
    from .clustering import build_cluster_matrix
    return build_cluster_matrix(df, list(features))


# ------------------------------------------------------ display wrappers
# Kept in one place so a future Streamlit API change needs a one-line fix.
def show_df(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def show_plot(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key)


def show_pyplot(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def download_csv(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", key=key)
