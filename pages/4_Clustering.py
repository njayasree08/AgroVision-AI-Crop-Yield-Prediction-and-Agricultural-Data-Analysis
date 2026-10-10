import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src import clustering as clu
from src import visualization as viz
from src.config import CATEGORICAL_FEATURES, CLUSTER_DEFAULT_FEATURES, FEATURES, RANDOM_SEED
from src.ui import cached_cluster_matrix, download_csv, get_dataset, setup_page, show_df, show_plot, show_pyplot, synthetic_notice

setup_page("Clustering Analysis")
synthetic_notice()
df, thresholds, report = get_dataset()

st.write("**Agglomerative (hierarchical) clustering** starts with every record as its own cluster and repeatedly "
         "merges the two closest clusters. Yield is **not** used to form clusters; it is added afterwards only to "
         "describe them. `Previous_Year_Yield` is excluded by default for the same reason.")

with st.sidebar:
    st.markdown("#### Clustering settings")
    features = st.multiselect("Features used for clustering", FEATURES, default=CLUSTER_DEFAULT_FEATURES)
    n_clusters = st.slider("Number of clusters", 2, clu.MAX_CLUSTERS, 4)
    linkage = st.selectbox("Linkage", clu.LINKAGES, help="ward minimises within-cluster variance; "
                           "complete/average/single use maximum/mean/minimum pairwise distance.")
    sample_size = st.slider("Dendrogram sample size", 30, 300, 120, 10)
    p_leaves = st.slider("Dendrogram leaves shown", 10, 60, 30, 5)

errors = clu.validate_cluster_params(len(df), n_clusters, linkage, features)
if errors:
    for e in errors:
        st.error(e)
    st.stop()

X, names = cached_cluster_matrix(df, tuple(features))
labels = clu.run_agglomerative(X, n_clusters, linkage)
silhouette = clu.safe_silhouette(X, labels)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clusters", n_clusters)
c2.metric("Silhouette score", "n/a" if silhouette is None else f"{silhouette:.3f}")
c3.metric("Encoded input columns", X.shape[1])
c4.metric("Smallest cluster", int(min((labels == c).sum() for c in set(labels))))
st.caption("Silhouette ranges from -1 to 1; values near 0 mean clusters overlap. Agricultural records often form "
           "continuous gradients, so low scores are normal. **Cluster numbers are arbitrary labels**: Cluster 4 is "
           "not better or worse than Cluster 1.")

tab_names = ["Dendrogram", "PCA view", "Sizes and profiles", "Choosing k", "Interpretation", "Download"]
tabs = st.tabs(tab_names)

with tabs[0]:
    Z, idx, sampled = clu.compute_linkage_matrix(X, linkage, max_samples=sample_size, seed=RANDOM_SEED)
    if sampled:
        st.info(f"The dendrogram is built from a random sample of {len(idx)} of {len(X)} records (seed {RANDOM_SEED}) "
                f"and truncated to the last {p_leaves} merges so it stays readable. The red line marks where the "
                f"tree is cut for {n_clusters} clusters; cluster assignments elsewhere use all records.")
    else:
        st.info(f"All {len(X)} records are used; the tree is truncated to the last {p_leaves} merges for readability.")
    show_pyplot(viz.dendrogram_figure(Z, n_clusters, p_leaves, linkage))

with tabs[1]:
    coords, evr = clu.pca_projection(X, RANDOM_SEED)
    hover = df[[c for c in ("Record_ID", "Crop_Type", "Season", "Crop_Yield") if c in df.columns]]
    show_plot(viz.pca_scatter(coords, labels, evr, hover), key="clu_pca")
    st.caption(f"The two components explain {evr.sum() * 100:.1f}% of the variance; points that look close here "
               "may still differ in the dimensions that are hidden.")

profile = clu.cluster_profiles(df, labels)
with tabs[2]:
    show_plot(viz.cluster_size_bar(labels), key="clu_size")
    st.markdown("**Cluster profiles** (means of numeric columns; most frequent category)")
    show_df(profile)

with tabs[3]:
    table = clu.compare_cluster_counts(X, linkage)
    show_df(table.round(4))
    show_plot(viz.silhouette_line(table), key="clu_sil")
    st.caption("Compare k values by silhouette and Davies-Bouldin together with agronomic sense; the highest "
               "silhouette is not automatically the most useful grouping.")

with tabs[4]:
    for line in clu.interpret_clusters(profile, df):
        st.markdown(f"<div class='finding-box'>{line}</div>", unsafe_allow_html=True)
    st.caption("Descriptions are generated from the cluster means versus the dataset averages. "
               "They describe patterns, not causes.")

with tabs[5]:
    assigned = df.reset_index(drop=True).copy()
    assigned["Cluster"] = labels
    download_csv(assigned, "Download cluster assignments", "cluster_assignments.csv", "dl_clu")
    download_csv(profile, "Download cluster profiles", "cluster_profiles.csv", "dl_clu_prof")
