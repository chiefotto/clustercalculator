from data_helpers.cluster_loader import (
    available_seasons,
    get_feature_columns,
    load_defense_clusters,
)

import streamlit as st


def read_cluster_file():
    """Backward-compatible wrapper used by other modules."""
    seasons = available_seasons()
    season = seasons[-1] if seasons else "2025-26"
    df = load_defense_clusters(season)
    if df.empty:
        import pandas as pd
        return pd.DataFrame(columns=["TEAM_ID", "cluster"]), {}

    cluster_df = df[["TEAM_ID", "CLUSTER"]].rename(columns={"CLUSTER": "cluster"})
    cluster_map = cluster_df.groupby("cluster")["TEAM_ID"].apply(list).to_dict()
    return cluster_df, cluster_map


def clusters_frontend():
    seasons = available_seasons()
    if not seasons:
        st.info("No cluster data found. Run the pipeline first.")
        return

    season = st.selectbox("Season", seasons, index=len(seasons) - 1, key="clusters_season")
    df = load_defense_clusters(season)
    if df.empty:
        return

    feature_cols = get_feature_columns(df)
    st.dataframe(
        df[["TEAM_NAME", "CLUSTER"] + feature_cols],
        use_container_width=True,
        hide_index=True,
    )

    cluster_map = df.groupby("CLUSTER")["TEAM_NAME"].apply(list).to_dict()
    st.subheader("Cluster → Teams")
    for c, team_list in sorted(cluster_map.items()):
        st.markdown(f"**Cluster {c}:** {', '.join(team_list)}")
