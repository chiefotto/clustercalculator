PROMPT 5 — Rolling Window Cluster Analyzer page (build & save windowed artifacts)

You are adding a Streamlit page: app/pages/rolling_cluster_analyzer.py that can generate clusters for time windows (rolling/segment), save them as separate artifacts, and visualize them.

Window specs supported:
- Full (key: full)
- Last N games (key: last{N})
- Month (key: month_{MonthName})
- Season segment (key: segment_{PreAllStar|PostAllStar})

Requirements:
1) Pipeline API:
   Update run_pipeline signature to accept window_spec:
     run_pipeline(season: str, artifact_key: str, window_params: dict, ...)

   It must:
   - Pass window params to nba_api endpoints if supported
   - If not supported, raise UnsupportedFilterError with a clear message
   - Save to:
     data_cache/defense_clusters_{season}_{artifact_key}.parquet
     data_cache/defense_clusters_{season}_{artifact_key}_diagnostics.json
   - Record WINDOW_TYPE, WINDOW_VALUE, ARTIFACT_KEY columns in parquet metadata

2) Loader:
   - list_available_artifacts(season) must include windowed keys
   - load_defense_clusters(season, artifact_key)
   - get_or_build_defense_clusters(season, artifact_key, force_refresh=False) (optional; default is manual build)

3) Page UI:
   Sidebar controls:
   - Season dropdown
   - Window type dropdown
   - Window value selector (N games, month list, segment list)
   - “Build/Update window clusters now” button
   - Show last updated + pipeline flags + chosen_k + silhouette

   Main:
   - Same 4 charts as defense_cluster_viz page, driven by the selected window artifact
   - Units dropdown + cluster labels + PCA axis explanation should work here too (reuse helpers)

4) Safety:
   - Do not auto-build on page load
   - Button triggers build; use st.status() progress updates
   - Disable button while running

Deliverables:
- New page file
- Pipeline + loader extensions for windowed artifacts
- Clear error message when filters aren’t supported by the endpoint