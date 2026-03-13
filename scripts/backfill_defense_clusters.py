"""
Backfill last N seasons of full-season defense cluster artifacts.

Run from repo root:
    python scripts/backfill_defense_clusters.py
    python scripts/backfill_defense_clusters.py --seasons 5 --dry-run
    python scripts/backfill_defense_clusters.py --force --verbose

Uses app.data_helpers.defense_clusters_pipeline.run_pipeline() and
cluster_loader path helpers. Skips seasons where the canonical parquet
already exists unless --force is passed. Writes a JSON report to
data_cache/backfill_defense_clusters_report_{timestamp}.json.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Run from repo root; app is the package root for data_helpers
ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

# After path fix, these use app/data_cache (pipeline and loader use __file__ relative to app)
from data_helpers.cluster_loader import (
    canonical_diagnostics_path_for_season,
    canonical_path_for_season,
)
from data_helpers.defense_clusters_pipeline import run_pipeline

DATA_CACHE = APP / "data_cache"

CURRENT_SEASON = "2025-26"


def _season_range(start_year: int, end_year: int) -> list[str]:
    """Return season strings from start_year to end_year inclusive (e.g. 2016-17 .. 2025-26)."""
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year, end_year + 1)]


def _last_n_seasons(n: int, current_season: str = CURRENT_SEASON) -> list[str]:
    """Last n seasons ending with current_season. current_season is e.g. 2025-26."""
    start_y = int(current_season[:4])
    end_y = int(current_season[:4])
    start_y = end_y - n + 1
    return _season_range(start_y, end_y)


def _normalize_season(s: str) -> str:
    """Normalize season string: 2024-2025 -> 2024-25, 2024-25 unchanged."""
    s = s.strip()
    if len(s) == 9 and s[4] == "-" and s[5:].isdigit():
        return f"{s[:4]}-{s[-2:]}"
    return s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill full-season defense cluster artifacts for the last N seasons.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if parquet already exists.",
    )
    parser.add_argument(
        "--start-season",
        type=str,
        default=None,
        help="Override start season (e.g. 2024-25; 2024-2025 is normalized to 2024-25).",
    )
    parser.add_argument(
        "--end-season",
        type=str,
        default=None,
        help="Override end season (e.g. 2024-25).",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        default=10,
        metavar="N",
        help="Number of seasons to backfill (default 10).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be built; do not run pipeline.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit pipeline logs.",
    )
    parser.add_argument(
        "--delay-between-seasons",
        type=float,
        default=10.0,
        metavar="SEC",
        help="Seconds to wait after each season before the next (default 10).",
    )
    args = parser.parse_args()

    if args.start_season:
        args.start_season = _normalize_season(args.start_season)
    if args.end_season:
        args.end_season = _normalize_season(args.end_season)

    if args.start_season and args.end_season:
        start_y = int(args.start_season[:4])
        end_y = int(args.end_season[:4])
        seasons = _season_range(start_y, end_y)
    elif args.start_season or args.end_season:
        # One-sided: use --seasons from the given bound
        anchor = args.end_season or args.start_season
        y = int(anchor[:4])
        if args.end_season:
            seasons = _season_range(max(2000, y - args.seasons + 1), y)
        else:
            seasons = _season_range(y, min(2030, y + args.seasons - 1))
    else:
        seasons = _last_n_seasons(args.seasons, CURRENT_SEASON)

    report_rows: list[dict] = []

    def _log(msg: str) -> None:
        if args.verbose:
            print(msg)

    for season in seasons:
        canon_path = canonical_path_for_season(season)
        diag_path = canonical_diagnostics_path_for_season(season)
        exists = canon_path.exists()

        if args.dry_run:
            action = "would build" if (args.force or not exists) else "would skip"
            print(f"  {season}: {action} -> {canon_path}")
            report_rows.append({
                "season": season,
                "status": "would_skip" if exists and not args.force else "would_build",
                "path": str(canon_path),
            })
            continue

        if exists and not args.force:
            # Load diagnostics for report
            chosen_k = None
            silhouette = None
            updated_at = None
            if diag_path.exists():
                try:
                    diag = json.loads(diag_path.read_text())
                    chosen_k = diag.get("chosen_k")
                    k_eval = diag.get("k_evaluation", [])
                    if k_eval and chosen_k is not None:
                        for row in k_eval:
                            if row.get("k") == chosen_k:
                                silhouette = row.get("silhouette")
                                break
                    updated_at = diag.get("updated_at")
                except Exception:
                    pass
            report_rows.append({
                "season": season,
                "status": "skipped",
                "chosen_k": chosen_k,
                "silhouette": silhouette,
                "updated_at": updated_at,
                "path": str(canon_path),
            })
            _log(f"  {season}: skipped (exists)")
            if season != seasons[-1] and args.delay_between_seasons > 0:
                if args.verbose:
                    print(f"  waiting {args.delay_between_seasons}s before next season ...")
                time.sleep(args.delay_between_seasons)
            continue

        try:
            run_pipeline(season, log_fn=_log if args.verbose else None)
            # Reload diagnostics after build
            chosen_k = None
            silhouette = None
            updated_at = None
            if diag_path.exists():
                try:
                    diag = json.loads(diag_path.read_text())
                    chosen_k = diag.get("chosen_k")
                    k_eval = diag.get("k_evaluation", [])
                    if k_eval and chosen_k is not None:
                        for row in k_eval:
                            if row.get("k") == chosen_k:
                                silhouette = row.get("silhouette")
                                break
                    updated_at = diag.get("updated_at")
                except Exception:
                    pass
            report_rows.append({
                "season": season,
                "status": "built",
                "chosen_k": chosen_k,
                "silhouette": silhouette,
                "updated_at": updated_at,
                "path": str(canon_path),
            })
            if season != seasons[-1] and args.delay_between_seasons > 0:
                if args.verbose:
                    print(f"  waiting {args.delay_between_seasons}s before next season ...")
                time.sleep(args.delay_between_seasons)
        except Exception as e:
            report_rows.append({
                "season": season,
                "status": "failed",
                "error": str(e),
                "path": str(canon_path),
            })
            if args.verbose:
                print(f"  {season}: failed: {e}")
            # Continue to next season

    # Summary table
    print("\nBackfill summary")
    print("-" * 80)
    print(f"{'season':<12} {'status':<10} {'chosen_k':<10} {'silhouette':<12} {'updated_at':<26} path")
    print("-" * 80)
    for r in report_rows:
        k = r.get("chosen_k")
        sil = r.get("silhouette")
        sil_str = f"{sil:.3f}" if sil is not None else ""
        ts = (r.get("updated_at") or "")[:25]
        print(f"{r['season']:<12} {r['status']:<10} {str(k):<10} {sil_str:<12} {ts:<26} {r.get('path', '')}")
    print("-" * 80)

    # Save report
    if not args.dry_run and report_rows:
        DATA_CACHE.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = DATA_CACHE / f"backfill_defense_clusters_report_{timestamp}.json"
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "options": {
                "force": args.force,
                "seasons_arg": args.seasons,
                "dry_run": args.dry_run,
            },
            "rows": report_rows,
        }
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    main()
