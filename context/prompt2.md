PROMPT 2 — Upgrade visualization page to browse any season artifact + units toggle + labels + PCA axis explanation

You are enhancing the Streamlit visualization page for defensive clusters to support:
- Selecting any season from the last 10 seasons (precomputed artifacts)
- Selecting “Artifact mode”: Full (historical) vs Latest (for current season)
- Units dropdown: Z-score / Percentile / Band(percentile)
- Cluster labeling (human-readable)
- PCA axis explanation (PC1/PC2 interpreted via loadings)

Assumptions:
- Full-season artifacts exist in data_cache:
  defense_clusters_{season}_full.parquet
  defense_clusters_{season}_full_diagnostics.json
- For the current season, a “latest” artifact may exist (rolling/asof). If not, fall back to full.

Tasks:
1) Update app/data_helpers/cluster_loader.py
   - Add list_available_artifacts(season) -> list of keys (e.g., ["full", "last12", "asof_YYYY-MM-DD"])
   - Add resolve_latest_artifact(season) -> best artifact key (prefer rolling/asof > full)
   - Add load_defense_clusters(season, artifact_key="full")
   - Add load_diagnostics(season, artifact_key="full")
   - Add last_updated(season, artifact_key)
   - Keep st.cache_data TTL 1 hour; add invalidate_cache()

2) Add shared display helpers (cached) in cluster_loader.py:
   - z_to_percentile(z): norm.cdf(z)*100
   - percentile_to_band(p): Very Low/Low/Typical/High/Very High using percentile bins
   - format_display(z, units_mode): returns display_text
   - Ensure hover always shows: “{pct}th pct ({band}) · {z:+.2f}σ”

3) Update app/pages/defense_cluster_viz.py:
   Sidebar:
   - Season dropdown (available seasons from loader scan)
   - Artifact dropdown: for past seasons only show Full; for current season show Full + Latest
   - Units dropdown: ["Z-score (σ)", "Percentile", "Band (percentile)"]
   - Show last updated timestamp and pipeline version
   - Keep “Update clusters now” button for current season (and for selected artifact only)

   Main page:
   - PCA Scatter: axis labels include human-readable subtitles from diagnostics PCA loadings
   - Cluster defining features bar chart: bar length uses z-score, bar text uses selected units
   - Distribution plot: hover uses pct+band+z formatting
   - Team drilldown: z-score profile bars + text units + hover formatting

4) Cluster labeling:
   - Derive labels from cluster defining features (top abs(z)):
     take 2 strong positives and 1 strong negative using a FEATURE_LABEL_MAP
   - Show labels in dropdowns and headings: “2 — Perimeter-heavy + Paint-stingy”

5) PCA axis explanation:
   - Read PCA loadings + feature names from diagnostics JSON
   - Under scatter, add expander showing top + and - loadings for PC1 and PC2
   - Axis titles: “PC1 — {subtitle}” and “PC2 — {subtitle}”
   - Subtitle can be generated from top contributors; fall back to “Top features: A, B, C”

Deliverables:
- Updated loader + page implementing the UX
- No pipeline recompute except when user clicks Update
- Must handle missing features across seasons by using intersection