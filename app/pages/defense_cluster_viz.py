"""
Defense Clusters -- Visualizations
==================================
Dedicated Streamlit page for exploring and interpreting team defensive
clusters.  Data is prefilled by running scripts/backfill_defense_clusters.py
from the repo root; this page only loads from disk.
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
    band_color,
    cluster_display_name,
    compute_cluster_labels,
    compute_league_means,
    compute_league_stds,
    format_feature_value,
    get_feature_columns,
    interpret_pc_axis,
    last_updated,
    list_available_artifacts,
    load_defense_clusters,
    load_diagnostics,
    percentile_to_band,
    resolve_latest_artifact,
    z_to_percentile,
)

CLUSTER_PALETTE = px.colors.qualitative.Set2

# ── Page header ──────────────────────────────────────────────────────────

st.title("Defense Clusters")

# ── Sidebar: season + artifact (data prefilled by backfill script) ────────

seasons = available_seasons()

with st.sidebar:
    st.header("Controls")

    if seasons:
        selected_season = st.selectbox(
            "Season", seasons, index=len(seasons) - 1,
        )
    else:
        selected_season = "2025-26"

    _current_year = 2025
    _all_seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(_current_year, _current_year - 10, -1)]

    # -- Artifact mode: Full (historical) vs Latest (current season only) --
    artifact_keys = list_available_artifacts(selected_season)
    if not artifact_keys:
        artifact_keys = ["full"]
    _latest_key = resolve_latest_artifact(selected_season)
    if _latest_key != "full" and _latest_key not in artifact_keys and selected_season == _all_seasons[0]:
        artifact_keys = ["full", _latest_key]
    artifact_options = artifact_keys
    artifact_labels = {
        "full": "Full (historical)",
        **{k: f"Latest ({k})" for k in artifact_options if k != "full"},
    }
    selected_artifact = st.selectbox(
        "Artifact",
        artifact_options,
        format_func=lambda k: artifact_labels.get(k, k),
        help="Full = full-season artifact; Latest = rolling/asof for current season.",
    )

    # -- Last-updated + pipeline version --
    ts = last_updated(selected_season, selected_artifact)
    if ts:
        st.caption(f"Last updated: {ts[:19].replace('T', ' ')} UTC")
    else:
        st.caption("Last updated: never")
    diag_sidebar = load_diagnostics(selected_season, selected_artifact)
    if diag_sidebar:
        st.caption(f"Pipeline: {diag_sidebar.get('pipeline_version', '?')}")

# ── Load data from disk ───────────────────────────────────────────────────

if not seasons:
    st.warning(
        "No cluster data found. Prefill by running from the repo root:\n\n"
        "`python scripts/backfill_defense_clusters.py`"
    )
    st.stop()

df = load_defense_clusters(selected_season, selected_artifact)
if df.empty:
    st.stop()

feature_cols = get_feature_columns(df)
clusters = sorted(df["CLUSTER"].unique())
teams = sorted(df["TEAM_NAME"].unique())
diag = load_diagnostics(selected_season, selected_artifact)

league_means = compute_league_means(df)
league_stds = compute_league_stds(df)

cluster_labels = compute_cluster_labels(df, league_means, league_stds)

# ── Sidebar: units + cluster / feature / team selectors ──────────────────

with st.sidebar:
    units_mode = st.selectbox(
        "Units",
        ["Z-score (σ)", "Percentile", "Band (percentile)"],
        index=0,
        help="Controls how feature values are displayed on bar labels. "
             "Hover always shows percentile + band + z-score.",
    )

    st.divider()

    k_used = int(df["K_USED"].iloc[0])
    st.metric("K (clusters)", k_used)

    selected_cluster = st.selectbox(
        "Cluster",
        clusters,
        format_func=lambda c: cluster_display_name(c, cluster_labels),
    )
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

    with st.expander("Band legend"):
        st.markdown(
            "**Very Low** (<10th pct) · **Low** (10–25th) · "
            "**Typical** (25–75th) · **High** (75–90th) · "
            "**Very High** (≥90th)"
        )

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


# ── Shared chart helpers ─────────────────────────────────────────────────

def _build_z_bar(
    z_series: pd.Series,
    units_mode: str,
    xaxis_title: str = "Std. deviations from league mean",
) -> go.Figure:
    """
    Horizontal bar chart from a feature→z Series.
    Bar length is always z-score; labels + colors respect *units_mode*.
    Hover always shows percentile (band) + z.
    """
    feats = z_series.index.tolist()
    zvals = z_series.values.astype(float)
    pcts = z_to_percentile(zvals).astype(float)
    bands = [percentile_to_band(p) for p in pcts]
    colors = [band_color(b) for b in bands]
    texts = [format_feature_value(z, units_mode) for z in zvals]

    hover_texts = [
        f"<b>{f}</b><br>{p:.0f}th pct ({b}) · {z:+.2f}σ"
        for f, z, p, b in zip(feats, zvals, pcts, bands)
    ]

    fig = go.Figure(go.Bar(
        x=zvals,
        y=feats,
        orientation="h",
        marker_color=colors,
        text=texts,
        textposition="outside",
        hovertext=hover_texts,
        hoverinfo="text",
    ))
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis=dict(autorange="reversed"),
        height=max(360, 32 * len(feats)),
        margin=dict(l=10, r=10, t=10, b=30),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  A) PCA SCATTER
# ═════════════════════════════════════════════════════════════════════════

pc1_subtitle, pc2_subtitle = "", ""
if diag and "pca_components" in diag and "pca_feature_names" in diag:
    pca_feat_names = diag["pca_feature_names"]
    pca_comps = diag["pca_components"]
    evr = diag.get("pca_explained_variance", [])

    if len(pca_comps) >= 1:
        pc1_subtitle, pc1_pos, pc1_neg = interpret_pc_axis(
            pca_comps[0], pca_feat_names,
        )
    if len(pca_comps) >= 2:
        pc2_subtitle, pc2_pos, pc2_neg = interpret_pc_axis(
            pca_comps[1], pca_feat_names,
        )

pc1_label = f"PC1 — {pc1_subtitle}" if pc1_subtitle else "PC 1"
pc2_label = f"PC2 — {pc2_subtitle}" if pc2_subtitle else "PC 2"

st.subheader("PCA Scatter — All Teams")

# Build rich hover data for scatter: team z-scores for top features
hover_feats = feature_cols[:5]
hover_z = pd.DataFrame(index=df.index)
for f in hover_feats:
    if f in league_means.index and f in league_stds.index and league_stds[f] != 0:
        hover_z[f] = (df[f] - league_means[f]) / league_stds[f]

customdata_cols = []
hover_parts = []
for i, f in enumerate(hover_feats):
    if f not in hover_z.columns:
        continue
    zv = hover_z[f].values.astype(float)
    pv = z_to_percentile(zv).astype(float)
    bv = np.array([percentile_to_band(p) for p in pv])
    customdata_cols.append(np.column_stack([zv, pv, bv]))
    hover_parts.append(
        f"<b>{f}</b>: %{{customdata[{i}][1]:.0f}}th pct "
        f"(%{{customdata[{i}][2]}}) · %{{customdata[{i}][0]:+.2f}}σ"
    )

if customdata_cols:
    customdata = np.concatenate(customdata_cols, axis=1).reshape(len(df), -1, 3)
else:
    customdata = None

fig_pca = px.scatter(
    df,
    x="PCA_1",
    y="PCA_2",
    color=df["CLUSTER"].astype(str),
    hover_name="TEAM_NAME",
    hover_data={f: ":.2f" for f in hover_feats},
    labels={"PCA_1": pc1_label, "PCA_2": pc2_label, "color": "Cluster"},
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

# -- PCA axis explanation expander --
if diag and "pca_components" in diag and "pca_feature_names" in diag:
    with st.expander("What do PC1 and PC2 represent?"):
        evr = diag.get("pca_explained_variance", [])
        pca_feat_names = diag["pca_feature_names"]
        pca_comps = diag["pca_components"]

        for pc_idx, pc_name in enumerate(["PC1", "PC2"]):
            if pc_idx >= len(pca_comps):
                break
            subtitle, top_pos, top_neg = interpret_pc_axis(
                pca_comps[pc_idx], pca_feat_names,
            )
            var_pct = f"{evr[pc_idx] * 100:.1f}%" if pc_idx < len(evr) else "?"
            st.markdown(f"**{pc_name}** ({var_pct} variance) — *{subtitle}*")

            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("Top positive loadings")
                for feat, val in top_pos:
                    st.markdown(f"- `{feat}` {val:+.3f}")
            with col_neg:
                st.markdown("Top negative loadings")
                for feat, val in top_neg:
                    st.markdown(f"- `{feat}` {val:+.3f}")


# ═════════════════════════════════════════════════════════════════════════
#  B) CLUSTER CENTROID PROFILE  (original features, not PCs)
# ═════════════════════════════════════════════════════════════════════════

cluster_name = cluster_display_name(selected_cluster, cluster_labels)
st.subheader(f"Cluster {cluster_name} — Defining Features")

cluster_mask = df["CLUSTER"] == selected_cluster
cluster_vals = df.loc[cluster_mask, feature_cols].mean()
deviations = (cluster_vals - league_means) / league_stds
deviations = deviations.dropna().sort_values()

n_show = min(6, len(deviations))
top_feats = pd.concat([deviations.head(n_show), deviations.tail(n_show)])
top_feats = top_feats[~top_feats.index.duplicated()]

fig_bar = _build_z_bar(top_feats, units_mode)
st.plotly_chart(fig_bar, use_container_width=True)

# -- Explain this cluster --

with st.expander("Explain this cluster", expanded=True):
    above = deviations[deviations > 0.5].sort_values(ascending=False)
    below = deviations[deviations < -0.5].sort_values()

    parts: list[str] = []
    if not above.empty:
        high_txt = ", ".join(
            f"**{f}** ({z_to_percentile(v):.0f}th pct, +{v:.1f}σ)"
            for f, v in above.head(4).items()
        )
        parts.append(f"High on {high_txt}")
    if not below.empty:
        low_txt = ", ".join(
            f"**{f}** ({z_to_percentile(v):.0f}th pct, {v:.1f}σ)"
            for f, v in below.head(4).items()
        )
        parts.append(f"Low on {low_txt}")

    if parts:
        st.markdown(f"Cluster {cluster_name}: " + " | ".join(parts))
    else:
        st.info("This cluster is close to the league average across all features.")

# -- Teams in cluster --

st.markdown(f"**Teams in Cluster {cluster_name}**")

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

st.subheader(f"Distribution — {selected_feature}")

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

st.subheader(f"Team Drilldown — {selected_team}")

team_row = df[df["TEAM_NAME"] == selected_team]
if team_row.empty:
    st.warning("Team not found in this season's data.")
    st.stop()

team_vals = team_row[feature_cols].iloc[0]
team_z = ((team_vals - league_means) / league_stds).dropna().sort_values()

fig_team = _build_z_bar(team_z, units_mode, xaxis_title="Z-score vs league mean")
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
