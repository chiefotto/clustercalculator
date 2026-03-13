PROMPT 1 — Backfill last 10 seasons (full-season artifacts) + skip-if-exists + report

You are implementing a “historical backfill” runner for the NBA defensive clustering pipeline.

Current codebase (already exists):
- app/data_helpers/defense_clusters_pipeline.py has run_pipeline(season, ...) which builds clusters, saves:
  - data_cache/defense_clusters_{season}_full.parquet
  - data_cache/defense_clusters_{season}_full_diagnostics.json
- app/data_helpers/cluster_loader.py has canonical path helpers and caching (loaders).

Task:
Create a new runnable entrypoint to backfill the last 10 seasons of FULL-SEASON artifacts.

Requirements:
1) Add a new script file:
   - scripts/backfill_defense_clusters.py (or app/tools/backfill_defense_clusters.py)
   It must be executable from the repo root.

2) Backfill range:
   - Determine “current season” from config or hardcode for now; then generate last 10 seasons strings like:
     2016-17, 2017-18, …, 2025-26
   - Default: backfill the 10 most recent completed seasons + optionally current season full to-date (but name it full only if it’s actually full; for now, treat it as full artifact generation from endpoints).

3) Skip-if-exists:
   - For each season, check if canonical parquet exists:
     data_cache/defense_clusters_{season}_full.parquet
   - If exists and --force not passed, skip and record “skipped”.

4) Robustness:
   - Use the existing retry logic in pipeline.
   - Catch exceptions per-season; do not abort entire run.
   - Continue to next season; record failures.

5) Output report:
   - At end, print a summary table:
     season | status (built/skipped/failed) | chosen_k | silhouette | updated_at | path
   - Save the same summary to:
     data_cache/backfill_defense_clusters_report_{timestamp}.json

6) CLI options:
   - --force (rebuild even if files exist)
   - --start-season / --end-season (optional)
   - --seasons N (default 10)
   - --dry-run (show what would be built, don’t run pipeline)
   - --verbose

Deliverables:
- New script with main() + argparse
- Uses pipeline run_pipeline() as the single source of truth
- Uses cluster_loader canonical_path helpers for file existence checks
- Uses diagnostics JSON to capture silhouette/k and include them in the report