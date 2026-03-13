```text
You are implementing UX improvements on the Streamlit page `app/pages/defense_cluster_viz.py` for the defensive clustering app.

Goal
Add:
1) A “Units” dropdown that controls how feature values are displayed:
   - Z-score (analytic)
   - Percentile (intuitive)
   - Band (percentile) (digestible)
2) Hover/tooltips should ALWAYS show: percentile + z-score together, regardless of selected Units.
   Example tooltip line:
   OPP_FG3A: 84th pct (High) · +1.00σ
3) Add cluster labeling (human-readable cluster names derived from defining features)
4) Add PCA axis labeling/explanation (interpret PC1/PC2 via loadings)

Constraints
- Do NOT change the underlying clustering math.
- Keep the canonical parquet schema stable; it’s OK to add new metadata columns if needed.
- Keep fast performance: compute once, cache results with st.cache_data, and only recompute when “Update clusters now” is pressed.
- Must work across seasons and artifacts with possibly different feature sets (use intersection of available features for display).

--------------------------------------------------------------------
A) “Units” dropdown + display logic

1) Add sidebar control:
   st.selectbox(
     "Units",
     ["Z-score (σ)", "Percentile", "Band (percentile)"],
     index=0
   )

2) Keep core data in z-score space.
   - You already compute z-scores vs league mean/std for:
     a) cluster defining features (cluster z vs league mean)
     b) team drilldown z profile
     c) distribution chart uses raw feature values (optional) or z-values (depending current design)
   - Do NOT replace z in calculations. Only change what the UI renders as labels / hover.

3) Implement these helper functions in a shared location (prefer `app/data_helpers/cluster_loader.py` so other pages can reuse). Cache them.

   a) z_to_percentile(z):
      - percentile = scipy.stats.norm.cdf(z) * 100
      - round to integer for display (or 1 decimal)
      - clamp [0,100]

   b) percentile_to_band(p):
      Use percentile-based bands to avoid “dense neutral seasons”:
      - p < 10  => "Very Low"
      - 10–25   => "Low"
      - 25–75   => "Typical"
      - 75–90   => "High"
      - >= 90   => "Very High"

   c) format_feature_display(z, units_mode):
      - Always compute:
        p = z_to_percentile(z)
        band = percentile_to_band(p)
      - If units_mode == "Z-score (σ)":
          return display_text = f"{z:+.2f}σ"
          return display_value = z  (if you use numeric labels)
        If units_mode == "Percentile":
          return display_text = f"{p:.0f}th pct"
        If units_mode == "Band (percentile)":
          return display_text = band

      IMPORTANT: even when units_mode is Percentile or Band, keep axis scale in z-score for bar charts that are “z-score vs league mean”. Only the text changes.

4) Hover template requirement (always shows both):
   For any chart where a feature value appears in hover or labels, include:
   f"{FEATURE}: {p:.0f}th pct ({band}) · {z:+.2f}σ"

Implementation detail:
- For Plotly, pass `customdata` containing z, percentile, band (and raw if needed).
- Set hovertemplate using those fields.
- For bar charts, set text=display_text from format_feature_display.

5) Color logic (consistent across units)
Use a color bucket based on percentile band (NOT z thresholds):
- Very Low / Low / Typical / High / Very High
This keeps colors stable across units.
Add a small legend explaining colors correspond to percentile bands.

--------------------------------------------------------------------
B) Cluster labeling (human-readable cluster names)

Problem
“Cluster 0/1/2/3” is not memorable. Create a readable label like:
- “Perimeter Pressure / High 3PA Allowed”
- “Low Pressure / Disciplined Fouling”
etc.

Logic
1) Use the existing cluster defining features computation:
   - cluster_mean_z = (cluster_mean - league_mean) / league_std
   - For each cluster, sort features by abs(z) descending.
   - Take:
     - top_pos = top 2 positive z features (z > +0.5)
     - top_neg = top 1 negative z feature (z < -0.5)
   - Create a label using a mapping from feature names to short phrases.

2) Add FEATURE_LABEL_MAP (short phrases):
   Example:
   OPP_FG3A -> "High 3PA allowed"
   CONTESTED_SHOTS_3PT_PER100 -> "High 3PT contests"
   OPP_PTS_PAINT -> "Paint vulnerable"
   OPP_TOV_PCT -> "Turnover pressure"
   DEFLECTIONS_PER100 -> "Active hands"
   OPP_FTA_RATE -> "Foul-prone"
   OPP_PTS_FB -> "Transition vulnerable"
   DEF_BOXOUTS_PER100 -> "Strong boxouts"
   etc.

3) Label format:
   - If you have 2 strong positives:
       "{phrase(top_pos1)} + {phrase(top_pos2)}"
   - If you have a strong negative:
       append " / Low {phrase(top_neg)}" or use a “stingy” word if appropriate.
   Keep it short (<= ~45 chars) so it fits UI.

4) Store cluster labels:
   - Compute at runtime from the loaded artifact (preferred, no schema changes).
   - Optionally store in diagnostics JSON for consistency across pages.

5) UI integration:
   - In the cluster selector dropdown, show:
       "0 — Turnover pressure + Active hands"
   - In headers:
       "Cluster 0 — Turnover pressure + Active hands"
   - Still keep numeric cluster ID for stability.

--------------------------------------------------------------------
C) PCA axis labeling/explanation (interpret PC1/PC2)

Problem
Scatter axes “PC1/PC2” are opaque.

Logic
1) After running PCA in the pipeline, save enough info to interpret axes:
   Option A (preferred): save PCA loadings + feature names into diagnostics JSON:
     - pca_components: list of components (each is list[float])
     - pca_feature_names: list[str]
     - explained_variance_ratio: list[float]
   Option B: recompute PCA on load (not preferred; avoid drift).

2) On viz page, create an expander section under the PCA scatter:
   “What do PC1 and PC2 represent?”

3) Interpret each PC:
   - Take loadings for PCi (component vector).
   - Rank features by absolute loading.
   - Display:
     - Top 5 positive contributors (highest +loading)
     - Top 5 negative contributors (most -loading)
   Also display explained variance for PC1 and PC2.

4) Optional: generate human-readable axis subtitles:
   - Use top contributors to assemble:
     PC1: “(Turnover pressure ↔ Low pressure)” (example)
   Implement a small heuristic:
     - If OPP_TOV_PCT/DEFLECTIONS high loadings => “Pressure”
     - If OPP_FG3A/CONTESTED_3PT high => “Perimeter”
     - If OPP_PTS_PAINT/BLK/CONTESTED_2PT high => “Paint”
     - If OPP_FTA_RATE high => “Fouling”
     - If OPP_PTS_FB high => “Transition”
   If heuristic fails, fall back to “PC1 (top features: A, B, C)”.

5) Scatter axis labels:
   - Instead of plain “PC1”, show:
     “PC1 — {subtitle}”
     “PC2 — {subtitle}”
   Keep the raw PC name in parentheses if needed.

6) Hover content for scatter points:
   - Keep current team hover with PC1, PC2 plus a few feature highlights.
   - Add the percentile+band+z formatting for those feature highlights.

--------------------------------------------------------------------
D) Where to apply “Units” and tooltip formatting

Apply unit-toggle labels + consistent hover to:
1) Cluster Defining Features bar chart
   - Bar length stays z-score
   - Bar text label uses selected units
   - Hover shows: pct (band) + z-score

2) Team Drilldown bar chart
   - Bar length stays z-score
   - Bar text label uses selected units
   - Hover shows: pct (band) + z-score

3) PCA Scatter hover metrics
   - Hover shows PC1/PC2 plus selected key feature metrics, each formatted:
     “84th pct (High) · +1.00σ”
   - If a metric is missing in this season’s feature set, omit it gracefully.

4) Distribution chart
   - If it plots raw values, the unit toggle only affects hover labels (not axis).
   - If it plots z-scores by cluster, apply same behavior as above.

--------------------------------------------------------------------
E) Deliverables

1) Update `app/data_helpers/cluster_loader.py`:
   - Add cached helpers:
     z_to_percentile, percentile_to_band, format_feature_display
     compute_display_columns(df, feature_cols, units_mode) -> adds columns or returns a formatter

2) Update `app/pages/defense_cluster_viz.py`:
   - Add Units dropdown
   - Thread units_mode into all charts
   - Implement Plotly hovertemplates using customdata with z/pct/band
   - Add “PCA axis explanation” expander using diagnostics PCA loadings
   - Add cluster labels and show them in selectors/headers

3) Update diagnostics writing (if not already):
   - Ensure PCA loadings + feature names are saved in diagnostics JSON so axis interpretation is consistent.

Testing checklist
- Switching Units updates labels instantly without recomputing pipeline.
- Hover always shows “percentile (band) · z-score”.
- Cluster selector shows cluster labels.
- PCA expander shows top positive/negative contributors for PC1/PC2.
- Works for seasons where feature set differs: uses intersection + handles missing.
```
