"""
Compare Clusters — cluster vs cluster across seasons.

Select Season A + Artifact A + Cluster A and Season B + Artifact B + Cluster B;
compare defining features side-by-side using season-normalized z-scores.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_helpers.cluster_loader import (
    available_seasons,
    cluster_display_name,
    compute_cluster_labels,
    compute_league_means,
    compute_league_stds,
    format_feature_value,
    get_feature_columns,
    get_or_build_defense_clusters,
    hover_line,
    invalidate_cache,
    last_updated,
    list_available_artifacts,
    load_defense_clusters,
    load_diagnostics,
    z_to_percentile,
)

st.title("Compare Clusters")

# ── Season list (last 10-ish from available) ───────────────────────────────
seasons = available_seasons()
if not seasons:
    st.warning("No cluster artifacts found. Run the backfill or build a season from Defense Clusters — Viz.")
    st.stop()
_season_opts = seasons

# ── Sidebar: A and B selections ───────────────────────────────────────────
with st.sidebar:
    st.header("Cluster A")
    season_a = st.selectbox("Season A", _season_opts, key="sea_a")
    artifacts_a = list_available_artifacts(season_a) or ["full"]
    artifact_a = st.selectbox("Artifact A", artifacts_a, key="art_a")
    df_a = load_defense_clusters(season_a, artifact_a)
    feat_a = get_feature_columns(df_a) if not df_a.empty else []
    league_means_a = compute_league_means(df_a) if not df_a.empty else pd.Series(dtype=float)
    league_stds_a = compute_league_stds(df_a) if not df_a.empty else pd.Series(dtype=float)
    labels_a = compute_cluster_labels(df_a, league_means_a, league_stds_a) if not df_a.empty else {}
    clusters_a = sorted(df_a["CLUSTER"].unique()) if not df_a.empty else []
    cluster_a = st.selectbox(
        "Cluster A",
        clusters_a if clusters_a else [0],
        format_func=lambda c: cluster_display_name(c, labels_a),
        key="cl_a",
    ) if clusters_a else 0

    st.divider()
    st.header("Cluster B")
    season_b = st.selectbox("Season B", _season_opts, key="sea_b")
    artifacts_b = list_available_artifacts(season_b) or ["full"]
    artifact_b = st.selectbox("Artifact B", artifacts_b, key="art_b")
    df_b = load_defense_clusters(season_b, artifact_b)
    feat_b = get_feature_columns(df_b) if not df_b.empty else []
    league_means_b = compute_league_means(df_b) if not df_b.empty else pd.Series(dtype=float)
    league_stds_b = compute_league_stds(df_b) if not df_b.empty else pd.Series(dtype=float)
    labels_b = compute_cluster_labels(df_b, league_means_b, league_stds_b) if not df_b.empty else {}
    clusters_b = sorted(df_b["CLUSTER"].unique()) if not df_b.empty else []
    cluster_b = st.selectbox(
        "Cluster B",
        clusters_b if clusters_b else [0],
        format_func=lambda c: cluster_display_name(c, labels_b),
        key="cl_b",
    ) if clusters_b else 0

    feature_set = st.radio(
        "Feature set",
        ["Intersection only", "Union"],
        index=0,
        help="Intersection: only features present in both. Union: all features (missing as NaN).",
    )
    use_intersection = feature_set == "Intersection only"

    build_a = st.button("Build missing artifact A", key="build_a") if (df_a.empty and artifact_a == "full") else False
    build_b = st.button("Build missing artifact B", key="build_b") if (df_b.empty and artifact_b == "full") else False

if build_a:
    with st.status("Building artifact for Season A...", expanded=True) as status:
        try:
            get_or_build_defense_clusters(season_a, force_refresh=True, log_fn=st.write)
            invalidate_cache()
            status.update(label="Done", state="complete")
            st.rerun()
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(str(e))
if build_b:
    with st.status("Building artifact for Season B...", expanded=True) as status:
        try:
            get_or_build_defense_clusters(season_b, force_refresh=True, log_fn=st.write)
            invalidate_cache()
            status.update(label="Done", state="complete")
            st.rerun()
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(str(e))

if df_a.empty or df_b.empty:
    st.warning("Load both Cluster A and B data (or build missing artifacts).")
    st.stop()

# ── Feature list ─────────────────────────────────────────────────────────
if use_intersection:
    common = list(set(feat_a) & set(feat_b))
    common.sort()
    feat_compare = common
else:
    all_f = list(set(feat_a) | set(feat_b))
    all_f.sort()
    feat_compare = all_f

if not feat_compare:
    st.warning("No features to compare (empty intersection).")
    st.stop()

# ── Centroid z-scores (within each dataset) ───────────────────────────────
mask_a = df_a["CLUSTER"] == cluster_a
mask_b = df_b["CLUSTER"] == cluster_b
centroid_a = df_a.loc[mask_a, feat_compare].mean()
centroid_b = df_b.loc[mask_b, feat_compare].mean()
# League stats only for features present in each frame
mu_a = league_means_a.reindex(feat_compare)
sd_a = league_stds_a.reindex(feat_compare)
mu_b = league_means_b.reindex(feat_compare)
sd_b = league_stds_b.reindex(feat_compare)
z_a = ((centroid_a - mu_a) / sd_a).fillna(0)
z_b = ((centroid_b - mu_b) / sd_b).fillna(0)

# Top N by max(|z_a|, |z_b|)
combined = pd.DataFrame({"z_a": z_a, "z_b": z_b})
combined["max_abs"] = combined[["z_a", "z_b"]].abs().max(axis=1)
top_n = min(12, len(feat_compare))
top_feats = combined.nlargest(top_n, "max_abs").index.tolist()

# ── Side-by-side bar chart ───────────────────────────────────────────────
st.subheader("Defining features comparison")

units_mode = st.selectbox(
    "Units",
    ["Z-score (σ)", "Percentile", "Band (percentile)"],
    index=0,
    key="compare_units",
)

fig = go.Figure()
z_a_vals = [z_a[f] for f in top_feats]
z_b_vals = [z_b[f] for f in top_feats]
hover_a = [hover_line(f, z_a[f]) for f in top_feats]
hover_b = [hover_line(f, z_b[f]) for f in top_feats]
text_a = [format_feature_value(z_a[f], units_mode) for f in top_feats]
text_b = [format_feature_value(z_b[f], units_mode) for f in top_feats]

fig.add_trace(go.Bar(
    name=f"{season_a} (A)",
    x=z_a_vals,
    y=top_feats,
    orientation="h",
    text=text_a,
    textposition="outside",
    hovertext=hover_a,
    hoverinfo="text",
))
fig.add_trace(go.Bar(
    name=f"{season_b} (B)",
    x=z_b_vals,
    y=top_feats,
    orientation="h",
    text=text_b,
    textposition="outside",
    hovertext=hover_b,
    hoverinfo="text",
))
fig.update_layout(
    barmode="group",
    height=max(400, 36 * len(top_feats)),
    yaxis=dict(autorange="reversed"),
    xaxis_title="Z-score vs league mean (per season)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# ── Summary bullets (largest |z_A - z_B|) ────────────────────────────────
diff = (z_a - z_b).abs().sort_values(ascending=False)
st.subheader("Biggest differences")
bullets = []
for f in diff.head(5).index:
    if f not in z_a.index or f not in z_b.index:
        continue
    za, zb = float(z_a[f]), float(z_b[f])
    direction = "A higher" if za > zb else "B higher"
    bullets.append(f"**{f}**: {direction} (A: {za:+.2f}σ, B: {zb:+.2f}σ)")
if bullets:
    for b in bullets[:3]:
        st.markdown(f"- {b}")
else:
    st.caption("No strong differences.")

# ── Team membership tables ───────────────────────────────────────────────
st.subheader("Team membership")

col_ta, col_tb = st.columns(2)
with col_ta:
    st.markdown(f"**Cluster A — {season_a} {artifact_a}**")
    teams_a = df_a.loc[mask_a, ["TEAM_NAME", "TEAM_ABBREVIATION"] if "TEAM_ABBREVIATION" in df_a.columns else ["TEAM_NAME"]]
    st.dataframe(teams_a.reset_index(drop=True), use_container_width=True, hide_index=True)
    st.caption(f"Last updated: {last_updated(season_a, artifact_a) or 'N/A'}")
with col_tb:
    st.markdown(f"**Cluster B — {season_b} {artifact_b}**")
    teams_b = df_b.loc[mask_b, ["TEAM_NAME", "TEAM_ABBREVIATION"] if "TEAM_ABBREVIATION" in df_b.columns else ["TEAM_NAME"]]
    st.dataframe(teams_b.reset_index(drop=True), use_container_width=True, hide_index=True)
    st.caption(f"Last updated: {last_updated(season_b, artifact_b) or 'N/A'}")

# ── Diagnostics compare ──────────────────────────────────────────────────
with st.expander("Diagnostics comparison"):
    diag_a = load_diagnostics(season_a, artifact_a)
    diag_b = load_diagnostics(season_b, artifact_b)
    rows = []
    for label, d in [("A", diag_a), ("B", diag_b)]:
        if not d:
            rows.append({"": label, "chosen_k": "-", "silhouette": "-", "pca_variance": "-", "residualized": "-"})
        else:
            k_eval = d.get("k_evaluation", [])
            sil = None
            if k_eval and d.get("chosen_k") is not None:
                for r in k_eval:
                    if r.get("k") == d["chosen_k"]:
                        sil = r.get("silhouette")
                        break
            ev = d.get("pca_explained_variance", [])
            pca_var = f"{sum(ev[:2])*100:.1f}%" if len(ev) >= 2 else "-"
            rows.append({
                "": label,
                "chosen_k": d.get("chosen_k", "-"),
                "silhouette": f"{sil:.3f}" if sil is not None else "-",
                "pca_variance": pca_var,
                "residualized": str(d.get("residualized", "-")),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
