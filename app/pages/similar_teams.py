"""
Similar Teams — find most similar team-seasons across the last 10 seasons.

Pick a Season + Artifact + Team; returns top N most similar (season, team) by
cosine similarity in standardized feature space (z-scores). Optional "why similar"
panel shows top features with smallest absolute difference.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from data_helpers.cluster_loader import (
    available_seasons,
    build_team_season_index,
    FEATURE_LABEL_MAP,
    list_available_artifacts,
)

st.title("Similar Teams")

seasons = available_seasons()
if not seasons:
    st.warning("No cluster artifacts found. Run the backfill or build a season from Defense Clusters — Viz.")
    st.stop()

include_latest = st.sidebar.checkbox(
    "Include current season 'Latest' artifact",
    value=False,
    help="When checked, current season may include a rolling/latest artifact in the index.",
)
_seasons_tuple = tuple(seasons)
index_df, feat_cols, log_info = build_team_season_index(
    include_latest_current=include_latest,
    _seasons=_seasons_tuple,
)

if index_df.empty:
    st.warning("No team-season index could be built. Ensure full-season artifacts exist.")
    st.stop()

st.sidebar.caption(f"Index: {len(index_df)} rows, {len(feat_cols)} features")

# ── Query: season, artifact, team ────────────────────────────────────────
with st.sidebar:
    st.header("Query")
    query_season = st.selectbox("Season", seasons, key="sim_season")
    artifacts = list_available_artifacts(query_season) or ["full"]
    query_artifact = st.selectbox("Artifact", artifacts, key="sim_artifact")
    # Teams available for this (season, artifact)
    query_options = index_df[(index_df["season"] == query_season) & (index_df["artifact_key"] == query_artifact)]
    team_names = sorted(query_options["team_name"].unique())
    if not team_names:
        st.warning("No teams for this season/artifact in index.")
        st.stop()
    query_team = st.selectbox("Team", team_names, key="sim_team")
    top_n = st.slider("Top N similar", min_value=5, max_value=20, value=10)

# Query row
query_mask = (
    (index_df["season"] == query_season)
    & (index_df["artifact_key"] == query_artifact)
    & (index_df["team_name"] == query_team)
)
query_row = index_df.loc[query_mask]
if query_row.empty:
    st.warning("Query team-season not found in index.")
    st.stop()
query_row = query_row.iloc[0]
q_vec = np.array(query_row["feature_vector"], dtype=float).reshape(1, -1)

# All vectors
X = np.array(index_df["feature_vector"].tolist(), dtype=float)
sim = cosine_similarity(q_vec, X)[0]
# Exclude self (and any duplicate same season/artifact/team)
for i in range(len(index_df)):
    r = index_df.iloc[i]
    if r["season"] == query_season and r["artifact_key"] == query_artifact and r["team_name"] == query_team:
        sim[i] = -2
order = np.argsort(sim)[::-1]

# Build results table
results = []
for idx in order[: top_n + 5]:
    if sim[idx] < -1:
        continue
    if len(results) >= top_n:
        break
    r = index_df.iloc[idx]
    results.append({
        "rank": len(results) + 1,
        "similarity": round(float(sim[idx]), 4),
        "season": r["season"],
        "team": r["team_name"],
        "cluster": r.get("cluster_label", f"Cluster {r['cluster_id']}"),
    })
res_df = pd.DataFrame(results)

st.subheader(f"Top {top_n} similar to **{query_team}** ({query_season}, {query_artifact})")
st.dataframe(res_df, use_container_width=True, hide_index=True)

# ── Why similar: top 5 features by smallest |z_q - z_o| ───────────────────
st.subheader("Why similar")
# Compare query to the most similar other row (first in order after excluding self)
if results:
    # First non-self index in order
    best_idx = next((i for i in order if sim[i] > -1), None)
    if best_idx is not None:
        o_vec = np.array(index_df.iloc[best_idx]["feature_vector"])
        diff = np.abs(q_vec.ravel() - o_vec)
        feat_order = np.argsort(diff)
        best_row = index_df.iloc[best_idx]
        st.caption(f"Features with smallest difference vs **{best_row['team_name']}** ({best_row['season']}):")
        for j in range(min(5, len(feat_cols))):
            k = feat_order[j]
            f = feat_cols[k]
            pair = FEATURE_LABEL_MAP.get(f)
            short = pair[0] if pair else f
            st.markdown(f"- **{f}** ({short}): Δz = {diff[k]:.2f}")
else:
    st.caption("No similar team-seasons to compare.")

# Optional: show features used
with st.expander("Index info"):
    st.json(log_info)
