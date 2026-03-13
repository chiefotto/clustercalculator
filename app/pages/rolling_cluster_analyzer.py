"""
Rolling Cluster Analyzer — build and visualize windowed cluster artifacts.

Window types: Full, Last N games, Month, Season segment. Only Full is
supported by the pipeline today; others raise a clear UnsupportedFilterError.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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
    invalidate_cache,
    last_updated,
    list_available_artifacts,
    load_defense_clusters,
    load_diagnostics,
    percentile_to_band,
    z_to_percentile,
)
from data_helpers.defense_clusters_pipeline import run_pipeline, UnsupportedFilterError

CLUSTER_PALETTE = px.colors.qualitative.Set2

st.title("Rolling Cluster Analyzer")

seasons = available_seasons()
if not seasons:
    st.warning("No seasons found. Build a full-season artifact from Defense Clusters — Viz first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    selected_season = st.selectbox("Season", seasons, index=len(seasons) - 1, key="roll_season")
    window_type = st.selectbox(
        "Window type",
        ["Full", "Last N games", "Month", "Season segment"],
        index=0,
        key="roll_type",
    )
    artifact_key = "full"
    window_params = None
    if window_type == "Full":
        artifact_key = "full"
    elif window_type == "Last N games":
        n_games = st.number_input("N games", min_value=1, max_value=82, value=12, key="roll_n")
        artifact_key = f"last{int(n_games)}"
        window_params = {"type": "last_n_games", "n": int(n_games)}
    elif window_type == "Month":
        month_name = st.selectbox(
            "Month",
            ["January", "February", "March", "April", "October", "November", "December"],
            key="roll_month",
        )
        artifact_key = f"month_{month_name}"
        window_params = {"type": "month", "month": month_name}
    else:
        segment = st.selectbox("Segment", ["PreAllStar", "PostAllStar"], key="roll_seg")
        artifact_key = f"segment_{segment}"
        window_params = {"type": "segment", "segment": segment}

    ts = last_updated(selected_season, artifact_key)
    if ts:
        st.caption(f"Last updated: {ts[:19].replace('T', ' ')} UTC")
    else:
        st.caption("Last updated: never")

    diag_sb = load_diagnostics(selected_season, artifact_key)
    if diag_sb:
        st.metric("chosen_k", diag_sb.get("chosen_k", "—"))
        k_eval = diag_sb.get("k_evaluation", [])
        sil = None
        if k_eval and diag_sb.get("chosen_k") is not None:
            for r in k_eval:
                if r.get("k") == diag_sb["chosen_k"]:
                    sil = r.get("silhouette")
                    break
        if sil is not None:
            st.metric("silhouette", f"{sil:.3f}")

    build_clicked = st.button(
        "Build / Update window clusters",
        type="primary",
        help="Run the pipeline for this season and window. Only Full is supported by the API.",
        key="roll_build",
    )

# ── Build on button click (no auto-build) ──────────────────────────────────
if build_clicked:
    with st.status("Building window clusters...", expanded=True) as status:
        progress = st.empty()
        def _log(msg):
            progress.write(msg)
        try:
            run_pipeline(
                selected_season,
                artifact_key=artifact_key,
                window_params=window_params,
                log_fn=_log,
            )
            invalidate_cache()
            status.update(label="Done", state="complete")
            st.toast(f"Artifact {artifact_key} saved.")
            st.rerun()
        except UnsupportedFilterError as e:
            status.update(label="Filter not supported", state="error")
            st.error(
                "**Window not supported.** The NBA API endpoints used by this pipeline "
                "do not support last N games, month, or segment filters. Use **Full** for "
                "full-season clustering."
            )
            st.exception(e)
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(str(e))
            st.exception(e)

# ── Load artifact and show same 4 charts as defense_cluster_viz ─────────────
df = load_defense_clusters(selected_season, artifact_key)
if df.empty:
    st.info(
        f"No artifact **{artifact_key}** for **{selected_season}**. "
        "Click **Build / Update window clusters** (only Full is supported by the API)."
    )
    st.stop()

feature_cols = get_feature_columns(df)
clusters = sorted(df["CLUSTER"].unique())
teams = sorted(df["TEAM_NAME"].unique())
diag = load_diagnostics(selected_season, artifact_key)
league_means = compute_league_means(df)
league_stds = compute_league_stds(df)
cluster_labels = compute_cluster_labels(df, league_means, league_stds)

with st.sidebar:
    units_mode = st.selectbox(
        "Units",
        ["Z-score (σ)", "Percentile", "Band (percentile)"],
        index=0,
        key="roll_units",
    )
    selected_cluster = st.selectbox(
        "Cluster",
        clusters,
        format_func=lambda c: cluster_display_name(c, cluster_labels),
        key="roll_cluster",
    )
    selected_feature = st.selectbox("Feature (distribution)", feature_cols, key="roll_feat")
    selected_team = st.selectbox("Team (drilldown)", teams, key="roll_team")

def _build_z_bar(z_series, units_mode, xaxis_title="Std. deviations from league mean"):
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
        x=zvals, y=feats, orientation="h",
        marker_color=colors, text=texts, textposition="outside",
        hovertext=hover_texts, hoverinfo="text",
    ))
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis=dict(autorange="reversed"),
        height=max(360, 32 * len(feats)),
        margin=dict(l=10, r=10, t=10, b=30),
    )
    return fig

# PCA scatter
pc1_subtitle, pc2_subtitle = "", ""
if diag and "pca_components" in diag and "pca_feature_names" in diag:
    pca_feat = diag["pca_feature_names"]
    pca_comps = diag["pca_components"]
    if len(pca_comps) >= 1:
        pc1_subtitle, _, _ = interpret_pc_axis(pca_comps[0], pca_feat)
    if len(pca_comps) >= 2:
        pc2_subtitle, _, _ = interpret_pc_axis(pca_comps[1], pca_feat)
pc1_label = f"PC1 — {pc1_subtitle}" if pc1_subtitle else "PC 1"
pc2_label = f"PC2 — {pc2_subtitle}" if pc2_subtitle else "PC 2"

st.subheader("PCA Scatter")
fig_pca = px.scatter(
    df, x="PCA_1", y="PCA_2",
    color=df["CLUSTER"].astype(str),
    hover_name="TEAM_NAME",
    labels={"PCA_1": pc1_label, "PCA_2": pc2_label, "color": "Cluster"},
    color_discrete_sequence=CLUSTER_PALETTE,
)
fig_pca.update_layout(height=520, legend_title_text="Cluster")
st.plotly_chart(fig_pca, use_container_width=True)

if diag and "pca_components" in diag:
    with st.expander("What do PC1 and PC2 represent?"):
        evr = diag.get("pca_explained_variance", [])
        for pc_idx, pc_name in enumerate(["PC1", "PC2"]):
            if pc_idx >= len(diag["pca_components"]):
                break
            subtitle, top_pos, top_neg = interpret_pc_axis(
                diag["pca_components"][pc_idx], diag["pca_feature_names"],
            )
            var_pct = f"{evr[pc_idx] * 100:.1f}%" if pc_idx < len(evr) else "?"
            st.markdown(f"**{pc_name}** ({var_pct}) — *{subtitle}*")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("Top positive loadings")
                for feat, val in top_pos:
                    st.markdown(f"- `{feat}` {val:+.3f}")
            with c2:
                st.markdown("Top negative loadings")
                for feat, val in top_neg:
                    st.markdown(f"- `{feat}` {val:+.3f}")

# Cluster defining features
cluster_name = cluster_display_name(selected_cluster, cluster_labels)
st.subheader(f"Cluster {cluster_name} — Defining Features")
cluster_mask = df["CLUSTER"] == selected_cluster
cluster_vals = df.loc[cluster_mask, feature_cols].mean()
deviations = (cluster_vals - league_means) / league_stds
deviations = deviations.dropna().sort_values()
n_show = min(6, len(deviations))
top_feats = pd.concat([deviations.head(n_show), deviations.tail(n_show)])
top_feats = top_feats[~top_feats.index.duplicated()]
st.plotly_chart(_build_z_bar(top_feats, units_mode), use_container_width=True)

# Distribution
st.subheader(f"Distribution — {selected_feature}")
fig_box = px.box(
    df, x=df["CLUSTER"].astype(str), y=selected_feature,
    color=df["CLUSTER"].astype(str),
    points="all", hover_name="TEAM_NAME",
    color_discrete_sequence=CLUSTER_PALETTE,
)
fig_box.update_layout(height=420, showlegend=False, xaxis_title="Cluster")
st.plotly_chart(fig_box, use_container_width=True)

# Team drilldown
st.subheader(f"Team Drilldown — {selected_team}")
team_row = df[df["TEAM_NAME"] == selected_team]
if not team_row.empty:
    team_vals = team_row[feature_cols].iloc[0]
    team_z = ((team_vals - league_means) / league_stds).dropna().sort_values()
    st.plotly_chart(_build_z_bar(team_z, units_mode), use_container_width=True)
    st.markdown("**5 Most Similar Teams**")
    scaler = StandardScaler()
    X_all = scaler.fit_transform(df[feature_cols].values)
    sim = cosine_similarity(X_all)
    team_idx = df.index.get_loc(team_row.index[0])
    sim_series = pd.Series(sim[team_idx], index=df["TEAM_NAME"].values).drop(selected_team, errors="ignore")
    closest = sim_series.nlargest(5).reset_index()
    closest.columns = ["Team", "Cosine Similarity"]
    st.dataframe(closest.round(3), use_container_width=True, hide_index=True)
