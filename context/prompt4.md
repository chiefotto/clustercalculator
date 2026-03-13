PROMPT 4 — Similar Teams Across 10 Seasons (team-season similarity search)

You are implementing a “Similar Teams” feature that finds the most similar team-seasons across the last 10 seasons using the defensive style feature vectors from saved artifacts.

Objective:
Create a page: app/pages/similar_teams.py that lets users:
- pick a Season + Artifact + Team
- return top N most similar (season, team) entries across all available seasons

Logic requirements:
1) Build a unified “team-season index” dataframe from artifacts:
   - Load all available full-season artifacts for last 10 seasons (and latest for current if desired)
   - Keep rows as (season, artifact_key, team_id, team_name, cluster_id, cluster_label, feature_vector)

2) Feature space alignment:
   - Use intersection of features across all loaded artifacts (global intersection) OR
   - Use pairwise intersection at query time (more flexible but slower).
   Choose global intersection for simplicity and stability; log which features are used.

3) Vector standardization:
   - Use z-scores already computed per season artifact (season-normalized).
   - Similarity should be computed in standardized feature space (z-space).
   - For each row, build a vector of z-values over the chosen feature list.
   - Handle NaNs by dropping features that are NaN for either vector or by imputing 0 (league average). Prefer imputing 0 if NaNs are rare.

4) Similarity metric:
   - Cosine similarity between vectors.
   - Exclude the query team-season itself from results.

5) Output:
   - Table: rank, similarity, season, team, cluster label, key matching features (optional)
   - Add a mini “why similar” panel: show top 5 features with smallest absolute difference

6) Performance:
   - Cache the global index dataframe with st.cache_data (TTL 1 hour)
   - Rebuild only when artifacts change (use invalidate_cache)

Deliverables:
- New page app/pages/similar_teams.py
- Helper in cluster_loader.py: build_team_season_index(artifact_keys=...) cached
- Adds option to include/exclude current season “latest” artifacts