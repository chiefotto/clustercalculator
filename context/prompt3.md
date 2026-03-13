PROMPT 3 — Add Compare Clusters page (cluster vs cluster across seasons, last 10 seasons)

You are adding a new Streamlit page: app/pages/compare_clusters.py

Goal:
Allow users to compare cluster signatures across seasons:
- Select Season A + Artifact A (Full or Latest where applicable) + Cluster A
- Select Season B + Artifact B + Cluster B
- Compare defining features side-by-side using season-normalized z-scores

Requirements:
1) Data loading:
   - Use cluster_loader to load both artifacts and both diagnostics
   - Feature set option:
     - “Intersection only” (default): only features present in both
     - “Union”: include all; missing as NaN and excluded from calculations where needed

2) Standardization for comparison:
   - Compute z-scores within EACH dataset using that dataset’s league mean/std:
     z = (feature - league_mean) / league_std
   - Compute cluster centroid z-profile for each selected cluster:
     centroid_z = (cluster_mean - league_mean) / league_std

3) Main visuals:
   A) Side-by-side horizontal bar chart (Plotly):
      - Choose a shared feature list: top N features by max(|z_A|, |z_B|)
      - Bars for A and B for each feature
      - Hover shows pct+band+z for A and B (use shared formatter helpers)

   B) Summary text:
      - Auto-generate 2–3 bullet summary of biggest differences:
        features with largest |z_A - z_B|

   C) Team membership tables:
      - List teams in Cluster A and Cluster B with counts
      - Show metadata: season, cluster label, last updated

   D) Diagnostics compare:
      - chosen_k, silhouette, PCA variance, residualized flags for each artifact

4) UX:
   - Left controls for both selections
   - If an artifact is missing, show “Build missing artifact” button
     (calls pipeline and saves, then reloads)

5) Registration:
   - Register the page in app/entry.py (or whatever your routing uses)

Deliverables:
- New page file
- Uses existing loader/pipeline patterns
- Handles multi-season and different feature sets gracefully