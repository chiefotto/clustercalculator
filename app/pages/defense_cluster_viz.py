"""
Defense Clusters -- Visualizations
==================================
Dedicated Streamlit page for exploring and interpreting team defensive
clusters.  Supports on-demand recomputation via an "Update clusters now"
button that mirrors the player-logs update pattern (cached load + action
+ save + rerun).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from data_helpers.cluster_loader import (
    available_seasons,
    compute_league_means,
    compute_league_stds,
    get_feature_columns,
    get_or_build_defense_clusters,
    invalidate_cache,
    last_updated,
    load_defense_clusters,
    load_diagnostics,
)

CLUSTER_PALETTE = px.colors.qualitative.Set2

# ── Page header ──────────────────────────────────────────────────────────

st.title("Defense Clusters -- Visualizations")

# ── Sidebar: season selector + update controls ───────────────────────────

seasons = available_seasons()

with st.sidebar:
    st.header("Controls")

    if seasons:
        selected_season = st.selectbox(
            "Season", seasons, index=len(seasons) - 1,
        )
    else:
        selected_season = None

    _current_year = 2025
    _all_seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(_current_year, _current_year - 10, -1)]
    _missing = [s for s in _all_seasons if s not in seasons]

    if _missing:
        new_season = st.selectbox(
            "Or generate a new season",
            options=[""] + _missing,
            format_func=lambda s: "-- select --" if s == "" else s,
            help="Pick a season to fetch and cluster from the NBA API.",
        )
    else:
        new_season = ""

    if new_season:
        selected_season = new_season
    elif selected_season is None:
        selected_season = "2025-26"

    # -- Last-updated display --
    ts = last_updated(selected_season)
    if ts:
        st.caption(f"Last updated: {ts[:19].replace('T', ' ')} UTC")
    else:
        st.caption("Last updated: never")

    # -- Update / Refresh button --
    update_clicked = st.button(
        f"Update clusters ({selected_season})",
        type="primary",
        help=(
            f"Re-fetch NBA API data for **{selected_season}**, re-cluster, "
            "and save a new season-scoped Parquet. Other seasons are untouched."
        ),
    )

# ── Handle update action (before loading data) ──────────────────────────

if update_clicked:
    with st.status(
        f"Recomputing clusters for {selected_season} ...", expanded=True,
    ) as status:
        progress_container = st.empty()

        def _log(msg):
            progress_container.write(msg)

        try:
            get_or_build_defense_clusters(
                selected_season, force_refresh=True, log_fn=_log,
            )
            status.update(label="Clusters updated!", state="complete")
            st.toast(f"Pipeline finished for {selected_season}.")
            st.rerun()

        except Exception as exc:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Pipeline error: {exc}")
            st.stop()

# ── Load cached data ─────────────────────────────────────────────────────

if not seasons and not new_season:
    st.warning(
        "No cluster files found. Enter a season above and click "
        "**Update clusters** to generate the initial clustering."
    )
    st.stop()

df = load_defense_clusters(selected_season)
if df.empty:
    st.stop()

feature_cols = get_feature_columns(df)
clusters = sorted(df["CLUSTER"].unique())
teams = sorted(df["TEAM_NAME"].unique())
diag = load_diagnostics(selected_season)

# ── Sidebar: cluster / feature / team selectors ─────────────────────────

with st.sidebar:
    k_used = int(df["K_USED"].iloc[0])
    st.metric("K (clusters)", k_used)

    selected_cluster = st.selectbox("Cluster", clusters)
    selected_feature = st.selectbox("Feature (for distribution)", feature_cols)
    selected_team = st.selectbox("Team (for drilldown)", teams)

    # -- Pipeline diagnostics expander --
    with st.expander("Pipeline diagnostics"):
        if diag:
            st.markdown(f"**Version:** {diag.get('pipeline_version')}")
            st.markdown(f"**Residualized:** {diag.get('residualized')}")
            st.markdown(
                f"**PCA components:** {diag.get('pca_components_kept')} "
                f"(target {diag.get('pca_variance_target', 0.85):.0%})"
            )
            pc1_r = diag.get("pc1_quality_corr", 0)
            dropped = diag.get("dropped_pc1", False)
            st.markdown(
                f"**PC1 ~ DEF_RATING:** r = {pc1_r:+.3f} "
                f"({'dropped' if dropped else 'kept'})"
            )
            st.markdown(f"**Features:** {len(diag.get('features_after_pruning', []))}")
        else:
            st.caption("No diagnostics file found.")

league_means = compute_league_means(df)
league_stds = compute_league_stds(df)

# ── K-evaluation table ───────────────────────────────────────────────────

if diag and "k_evaluation" in diag:
    with st.expander("K-selection diagnostics", expanded=False):
        k_df = pd.DataFrame(diag["k_evaluation"])
        st.dataframe(
            k_df.style.highlight_max(subset=["silhouette"], color="#d4edda"),
            use_container_width=True,
            hide_index=True,
        )
        if diag.get("dropped_pc1"):
            st.info("PC1 was dropped from the clustering space because it "
                     "correlated strongly with DEF_RATING (quality axis).")

# ═════════════════════════════════════════════════════════════════════════
#  A) PCA SCATTER
# ═════════════════════════════════════════════════════════════════════════

st.subheader("PCA Scatter -- All Teams")

hover_feats = feature_cols[:5]
fig_pca = px.scatter(
    df,
    x="PCA_1",
    y="PCA_2",
    color=df["CLUSTER"].astype(str),
    hover_name="TEAM_NAME",
    hover_data={f: ":.2f" for f in hover_feats},
    labels={"PCA_1": "PC 1", "PCA_2": "PC 2", "color": "Cluster"},
    color_discrete_sequence=CLUSTER_PALETTE,
)

show_labels = st.toggle("Show team labels", value=True)
if show_labels:
    fig_pca.update_traces(
        text=df["TEAM_NAME"],
        textposition="top center",
        textfont_size=9,
        mode="markers+text",
    )

fig_pca.update_layout(height=520, legend_title_text="Cluster")
st.plotly_chart(fig_pca, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
#  B) CLUSTER CENTROID PROFILE  (original features, not PCs)
# ═════════════════════════════════════════════════════════════════════════

st.subheader(f"Cluster {selected_cluster} -- Defining Features")

cluster_mask = df["CLUSTER"] == selected_cluster
cluster_vals = df.loc[cluster_mask, feature_cols].mean()
deviations = (cluster_vals - league_means) / league_stds
deviations = deviations.dropna().sort_values()

n_show = min(6, len(deviations))
top_feats = pd.concat([deviations.head(n_show), deviations.tail(n_show)])
top_feats = top_feats[~top_feats.index.duplicated()]

colors = ["#3498db" if v < 0 else "#e74c3c" for v in top_feats.values]

fig_bar = go.Figure(
    go.Bar(
        x=top_feats.values,
        y=top_feats.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in top_feats.values],
        textposition="outside",
    )
)
fig_bar.update_layout(
    xaxis_title="Std. deviations from league mean",
    yaxis=dict(autorange="reversed"),
    height=max(360, 32 * len(top_feats)),
    margin=dict(l=10, r=10, t=10, b=30),
)
st.plotly_chart(fig_bar, use_container_width=True)

# -- Explain this cluster --

with st.expander("Explain this cluster", expanded=True):
    above = deviations[deviations > 0.5].sort_values(ascending=False)
    below = deviations[deviations < -0.5].sort_values()

    parts: list[str] = []
    if not above.empty:
        high_txt = ", ".join(
            f"**{f}** (+{v:.1f}\u03c3)" for f, v in above.head(4).items()
        )
        parts.append(f"High on {high_txt}")
    if not below.empty:
        low_txt = ", ".join(
            f"**{f}** ({v:.1f}\u03c3)" for f, v in below.head(4).items()
        )
        parts.append(f"Low on {low_txt}")

    if parts:
        st.markdown(f"Cluster {selected_cluster}: " + " | ".join(parts))
    else:
        st.info("This cluster is close to the league average across all features.")

# -- Teams in cluster --

st.markdown(f"**Teams in Cluster {selected_cluster}**")

show_cols = ["TEAM_NAME", "W", "L"]
if "DEF_RATING" in df.columns:
    show_cols.append("DEF_RATING")
cluster_teams = df.loc[cluster_mask, show_cols]
st.dataframe(
    cluster_teams.reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)


# ═════════════════════════════════════════════════════════════════════════
#  C) DISTRIBUTION VIEW
# ═════════════════════════════════════════════════════════════════════════

st.subheader(f"Distribution -- {selected_feature}")

fig_box = px.box(
    df,
    x=df["CLUSTER"].astype(str),
    y=selected_feature,
    color=df["CLUSTER"].astype(str),
    points="all",
    hover_name="TEAM_NAME",
    labels={"x": "Cluster", "color": "Cluster"},
    color_discrete_sequence=CLUSTER_PALETTE,
)
fig_box.update_layout(height=420, showlegend=False, xaxis_title="Cluster")
st.plotly_chart(fig_box, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
#  D) TEAM DRILLDOWN  (z-scores + cosine similarity on standardized features)
# ═════════════════════════════════════════════════════════════════════════

st.subheader(f"Team Drilldown -- {selected_team}")

team_row = df[df["TEAM_NAME"] == selected_team]
if team_row.empty:
    st.warning("Team not found in this season's data.")
    st.stop()

team_vals = team_row[feature_cols].iloc[0]
team_z = ((team_vals - league_means) / league_stds).dropna().sort_values()

colors_team = ["#3498db" if v < 0 else "#e74c3c" for v in team_z.values]

fig_team = go.Figure(
    go.Bar(
        x=team_z.values,
        y=team_z.index,
        orientation="h",
        marker_color=colors_team,
        text=[f"{v:+.2f}" for v in team_z.values],
        textposition="outside",
    )
)
fig_team.update_layout(
    xaxis_title="Z-score vs league mean",
    yaxis=dict(autorange="reversed"),
    height=max(400, 30 * len(team_z)),
    margin=dict(l=10, r=10, t=10, b=30),
)
st.plotly_chart(fig_team, use_container_width=True)

# -- Closest teams (cosine similarity in standardized feature space) --

st.markdown(f"**5 Most Similar Teams to {selected_team}**")

scaler = StandardScaler()
X_all = scaler.fit_transform(df[feature_cols].values)
sim = cosine_similarity(X_all)

team_idx = df.index.get_loc(team_row.index[0])
sim_series = pd.Series(sim[team_idx], index=df["TEAM_NAME"].values)
sim_series = sim_series.drop(selected_team, errors="ignore")
closest = sim_series.nlargest(5).reset_index()
closest.columns = ["Team", "Cosine Similarity"]
closest["Cosine Similarity"] = closest["Cosine Similarity"].round(3)

st.dataframe(closest, use_container_width=True, hide_index=True)
