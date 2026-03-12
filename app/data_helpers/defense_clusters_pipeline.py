"""
Defense Clusters Pipeline  (v2 -- style-first)
================================================
Why the previous version collapsed to K=2
------------------------------------------
The original feature matrix was dominated by a single "quality" axis: metrics
like DEF_RATING, OPP_EFG_PCT, and OPP_PTS all measure *how good* a defense is
rather than *how* it defends.  KMeans splits the strongest variance axis first,
so it carved the league into "elite vs everyone else" -- a real but
uninteresting split for matchup analysis.

Silhouette peaked at K=2 because the quality axis was so dominant that every
additional cluster just subdivided noise on that axis.

Fix applied in v2
-----------------
1. **Style-first features** -- remove pure quality metrics (DEF_RATING, OPP_PTS,
   OPP_EFG_PCT, OPP_FG_PCT) from the clustering matrix.  Keep only *how*
   descriptors: shot-diet frequencies, foul rates, turnover pressure, hustle
   activity, paint/transition allowance.
2. **Residualize** remaining features against DEF_RATING so that any residual
   correlation with quality is removed.  After residualization each feature
   captures "style *given* the team's overall defensive strength."
3. **PCA before K selection** -- cluster on PCA-reduced space (retaining ~85 %
   variance).  If PC1 still correlates with quality, drop it automatically.
4. **K constrained to >= 4** so the algorithm can never collapse back to a
   binary split.

Usage
-----
    # CLI  (inside app/):
    python -m data_helpers.defense_clusters_pipeline

    # Python / Streamlit:
    from data_helpers.defense_clusters_pipeline import run_pipeline
    result_df = run_pipeline("2025-26")
"""

import json
import time
import warnings
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from nba_api.stats.endpoints import (
    leaguedashteamstats,
    leaguehustlestatsteam,
    leaguedashoppptshot,
)

warnings.filterwarnings("ignore", category=FutureWarning)

PIPELINE_VERSION = "2.0.0"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CACHE = BASE_DIR / "data_cache"
API_DELAY = 1.0
API_TIMEOUT = 60
MAX_RETRIES = 3


# ── Feature maps ─────────────────────────────────────────────────────────
#
# STYLE features describe *how* a team defends (shot diet, pressure,
# fouling, hustle, transition).  QUALITY metrics (DEF_RATING, OPP_PTS,
# OPP_EFG_PCT, OPP_FG_PCT) are excluded from clustering and used only as
# an anchor for residualization.

STYLE_STAT_FEATURES: dict[str, tuple[str, str]] = {
    # --- Opponent base (Per100) -----------------------------------------
    "OPP_FG3A":           ("opponent", "OPP_FG3A"),
    "OPP_FG3_PCT":        ("opponent", "OPP_FG3_PCT"),
    "OPP_FTA":            ("opponent", "OPP_FTA"),
    "OPP_OREB":           ("opponent", "OPP_OREB"),
    "OPP_AST":            ("opponent", "OPP_AST"),
    "OPP_TOV":            ("opponent", "OPP_TOV"),
    # --- Defense dashboard (Per100) -------------------------------------
    "STL":                ("defense", "STL"),
    "BLK":                ("defense", "BLK"),
    "DREB_PCT":           ("defense", "DREB_PCT"),
    # --- Four Factors ---------------------------------------------------
    "OPP_FTA_RATE":       ("four_factors", "OPP_FTA_RATE"),
    "OPP_TOV_PCT":        ("four_factors", "OPP_TOV_PCT"),
    "OPP_OREB_PCT":       ("four_factors", "OPP_OREB_PCT"),
    # --- Misc (Per100) --------------------------------------------------
    "OPP_PTS_OFF_TOV":    ("misc", "OPP_PTS_OFF_TOV"),
    "OPP_PTS_2ND_CHANCE": ("misc", "OPP_PTS_2ND_CHANCE"),
    "OPP_PTS_FB":         ("misc", "OPP_PTS_FB"),
    "OPP_PTS_PAINT":      ("misc", "OPP_PTS_PAINT"),
}

QUALITY_ANCHOR: tuple[str, str] = ("advanced", "DEF_RATING")

HUSTLE_TO_PER100 = [
    "DEFLECTIONS",
    "CONTESTED_SHOTS_2PT",
    "CONTESTED_SHOTS_3PT",
    "CHARGES_DRAWN",
    "DEF_LOOSE_BALLS_RECOVERED",
    "DEF_BOXOUTS",
]

OPP_SHOT_STYLE: dict[str, str] = {
    "OPP_SHOT_FG3A_FREQ": "FG3A_FREQUENCY",
}

# Interpretability ranking -- kept first when correlation-pruning
INTERPRETABILITY_ORDER = [
    "OPP_PTS_PAINT", "OPP_PTS_FB", "OPP_FG3_PCT", "OPP_TOV_PCT",
    "DEFLECTIONS_PER100", "OPP_OREB_PCT", "OPP_FTA_RATE", "BLK", "STL",
    "CHARGES_DRAWN_PER100", "DEF_LOOSE_BALLS_RECOVERED_PER100",
    "OPP_PTS_OFF_TOV", "OPP_PTS_2ND_CHANCE",
    "CONTESTED_SHOTS_3PT_PER100", "CONTESTED_SHOTS_2PT_PER100",
    "DEF_BOXOUTS_PER100", "DREB_PCT", "OPP_AST", "OPP_FG3A",
    "OPP_FTA", "OPP_OREB", "OPP_TOV", "OPP_SHOT_FG3A_FREQ",
]


# ═══════════════════════════════════════════════════════════════════════════
#  PART 1 -- DATA COMPILATION  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════

def _pause():
    time.sleep(API_DELAY)


def _retry(fn):
    """Call *fn()* with retries on timeout / connection errors.

    JSONDecodeError is treated as non-retryable: it means the NBA API
    returned an empty / non-JSON body (typically because no data exists
    for the requested season).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = fn()
            _pause()
            return result
        except JSONDecodeError:
            raise ValueError(
                "The NBA API returned an empty response -- this usually "
                "means data for the requested season is not available yet."
            )
        except Exception as exc:
            print(f"    [retry {attempt}/{MAX_RETRIES}] {type(exc).__name__}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(API_DELAY * attempt * 2)
    raise RuntimeError("unreachable")


def fetch_team_stats(measure_type: str, season: str) -> pd.DataFrame:
    return _retry(lambda: leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense=measure_type,
        per_mode_detailed="Per100Possessions",
        season=season,
        season_type_all_star="Regular Season",
        timeout=API_TIMEOUT,
    ).get_data_frames()[0])


def fetch_hustle(season: str) -> pd.DataFrame:
    return _retry(lambda: leaguehustlestatsteam.LeagueHustleStatsTeam(
        per_mode_time="PerGame",
        season=season,
        season_type_all_star="Regular Season",
        timeout=API_TIMEOUT,
    ).get_data_frames()[0])


def fetch_opp_shot_profile(season: str) -> pd.DataFrame:
    return _retry(lambda: leaguedashoppptshot.LeagueDashOppPtShot(
        league_id="00",
        per_mode_simple="PerGame",
        season=season,
        season_type_all_star="Regular Season",
        timeout=API_TIMEOUT,
    ).get_data_frames()[0])


def compile_raw_data(season: str, log_fn=None) -> dict[str, pd.DataFrame]:
    def _log(msg):
        if log_fn:
            log_fn(msg)
        print(f"[pipeline] {msg}")

    _log(f"Fetching data for {season} ...")
    raw: dict[str, pd.DataFrame] = {}

    for measure in ["Opponent", "Defense", "Four Factors", "Advanced", "Misc"]:
        key = measure.lower().replace(" ", "_")
        _log(f"  > LeagueDashTeamStats({measure})")
        raw[key] = fetch_team_stats(measure, season)

    _log("  > LeagueHustleStatsTeam")
    raw["hustle"] = fetch_hustle(season)

    _log("  > LeagueDashOppPtShot")
    raw["opp_shots"] = fetch_opp_shot_profile(season)

    return raw


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 -- STYLE-FIRST FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_table(
    raw: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Cherry-pick *style* features from each source by TEAM_ID.

    Returns
    -------
    meta_df      : team metadata (TEAM_ID, TEAM_NAME, GP, W, L)
    style_df     : style features indexed by TEAM_ID  (numeric, no NaN)
    quality_anchor : DEF_RATING series indexed by TEAM_ID
    """
    pace = raw["advanced"].set_index("TEAM_ID")["PACE"]
    team_ids = raw["opponent"]["TEAM_ID"].values
    result = pd.DataFrame({"TEAM_ID": team_ids})

    meta = raw["opponent"][["TEAM_ID", "TEAM_NAME", "GP", "W", "L"]].copy()

    # -- Style stat features (explicit mapping, no giant merges) --
    for feat_name, (src_key, col_name) in STYLE_STAT_FEATURES.items():
        src = raw[src_key]
        if col_name in src.columns:
            result[feat_name] = result["TEAM_ID"].map(
                src.set_index("TEAM_ID")[col_name]
            )

    # -- Hustle PerGame -> Per100 via PACE --
    hustle = raw["hustle"]
    for col in HUSTLE_TO_PER100:
        if col in hustle.columns:
            pg = hustle.set_index("TEAM_ID")[col]
            result[f"{col}_PER100"] = result["TEAM_ID"].map((pg / pace) * 100)

    # -- Opponent shot-profile rates (already frequencies) --
    opp_shots = raw["opp_shots"]
    for new_name, col in OPP_SHOT_STYLE.items():
        if col in opp_shots.columns:
            result[new_name] = result["TEAM_ID"].map(
                opp_shots.set_index("TEAM_ID")[col]
            )

    # -- Quality anchor (stored separately, NOT clustered on) --
    anchor_src_key, anchor_col = QUALITY_ANCHOR
    quality_anchor = raw[anchor_src_key].set_index("TEAM_ID")[anchor_col]
    quality_anchor = quality_anchor.reindex(team_ids)

    # Finalize style feature matrix
    style_df = result.set_index("TEAM_ID")
    style_df = style_df.select_dtypes(include=[np.number])
    style_df = style_df.dropna(axis=1, how="all")
    style_df = style_df.fillna(style_df.median())

    return meta, style_df, quality_anchor


def residualize_features(
    features_df: pd.DataFrame,
    anchor: pd.Series,
) -> pd.DataFrame:
    """
    For each feature *f*, fit  ``f = a + b * anchor + residual``  via OLS
    and replace *f* with its residual.

    This strips out variance explained by the quality anchor (DEF_RATING)
    so that the remaining signal is purely about *defensive style*.
    """
    result = features_df.copy()
    x = anchor.reindex(features_df.index).values
    X_design = np.column_stack([np.ones_like(x), x])

    for col in result.columns:
        y = result[col].values
        coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        result[col] = y - X_design @ coef

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3 -- CORRELATION PRUNING  (unchanged logic)
# ═══════════════════════════════════════════════════════════════════════════

def prune_correlated(
    df: pd.DataFrame, threshold: float = 0.85,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop one of each pair with |r| > *threshold*.
    Keeps the more-interpretable column (lower rank in INTERPRETABILITY_ORDER).
    """
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    def _rank(col: str) -> int:
        try:
            return INTERPRETABILITY_ORDER.index(col)
        except ValueError:
            return 999

    to_drop: set[str] = set()
    for col in upper.columns:
        for other in upper.index[upper[col] > threshold].tolist():
            if other in to_drop or col in to_drop:
                continue
            if _rank(col) <= _rank(other):
                to_drop.add(other)
            else:
                to_drop.add(col)

    dropped = sorted(to_drop)
    return df.drop(columns=dropped), dropped


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4 -- PCA REDUCTION + QUALITY CHECK
# ═══════════════════════════════════════════════════════════════════════════

def pca_reduce(
    X_scaled: np.ndarray,
    variance_target: float = 0.85,
    min_components: int = 2,
) -> tuple[np.ndarray, PCA, int]:
    """
    Fit PCA and select enough components to explain *variance_target*.
    Returns (X_pca_all_components, fitted_pca, n_components_to_keep).
    """
    pca = PCA(random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = max(min_components, int(np.searchsorted(cumvar, variance_target)) + 1)
    n_keep = min(n_keep, X_scaled.shape[1])

    return X_pca, pca, n_keep


def check_pc1_quality(
    X_pca: np.ndarray,
    anchor_values: np.ndarray,
    threshold: float = 0.6,
) -> tuple[bool, float]:
    """
    Return (should_drop, correlation) for PC1 vs quality anchor.
    If |corr| > threshold the first component is still mostly "quality".
    """
    r = float(np.corrcoef(X_pca[:, 0], anchor_values)[0, 1])
    return abs(r) > threshold, r


# ═══════════════════════════════════════════════════════════════════════════
#  PART 5 -- K SELECTION  (constrained K >= 4)
# ═══════════════════════════════════════════════════════════════════════════

def find_optimal_k(
    X: np.ndarray,
    k_range: tuple[int, int] = (4, 8),
    min_cluster_size: int = 3,
) -> tuple[int, pd.DataFrame]:
    """
    Evaluate candidate *K* values with inertia, silhouette, and cluster-size
    diagnostics.  K is never less than *k_range[0]* (default 4).

    Selection rule
    --------------
    1. Discard Ks where any cluster has fewer than *min_cluster_size* teams.
    2. Among remaining Ks, find the best silhouette score.
    3. If multiple Ks are within 0.02 of the best, prefer the smaller K for
       stability and interpretability.
    """
    rows = []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=50, random_state=42)
        labels = km.fit_predict(X)
        sizes = pd.Series(labels).value_counts()
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X, labels)),
            "min_size": int(sizes.min()),
            "max_size": int(sizes.max()),
            "balance": round(float(sizes.max() / sizes.min()), 2),
        })

    results = pd.DataFrame(rows)

    valid = results[results["min_size"] >= min_cluster_size]
    if valid.empty:
        valid = results

    best_sil = valid["silhouette"].max()
    close = valid[valid["silhouette"] >= best_sil - 0.02]
    best_k = int(close["k"].min())

    return best_k, results


def train_kmeans(X: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(X)
    return km, labels


# ═══════════════════════════════════════════════════════════════════════════
#  PART 6 -- PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    season: str = "2025-26",
    *,
    do_residualize: bool = True,
    pca_variance_target: float = 0.85,
    drop_pc1_threshold: float = 0.6,
    k_range: tuple[int, int] = (4, 8),
    min_cluster_size: int = 3,
    log_fn=None,
) -> pd.DataFrame:
    """
    Full end-to-end pipeline.

    Parameters
    ----------
    do_residualize      : regress out DEF_RATING from every style feature
    pca_variance_target : cumulative variance to retain in PCA
    drop_pc1_threshold  : if |corr(PC1, DEF_RATING)| > this, exclude PC1
    k_range             : (min_k, max_k) -- min_k should be >= 4
    min_cluster_size    : discard K candidates that produce tiny clusters
    log_fn              : optional callable(str) for progress updates (Streamlit)
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)
        print(f"[pipeline] {msg}")

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1 -- Compile raw data ------------------------------------------------
    raw = compile_raw_data(season, log_fn=log_fn)

    # 2 -- Build style-first features + extract quality anchor -------------
    _log("Engineering style features ...")
    meta_df, style_df, quality_anchor = build_feature_table(raw)
    features_before = style_df.columns.tolist()
    _log(f"  {len(features_before)} style features before pruning")

    # 3 -- Residualize against quality anchor ------------------------------
    if do_residualize:
        _log("Residualizing features against DEF_RATING ...")
        style_df = residualize_features(style_df, quality_anchor)

    # 4 -- Correlation pruning ---------------------------------------------
    style_df, dropped = prune_correlated(style_df, threshold=0.85)
    features_after = style_df.columns.tolist()
    _log(f"  Dropped {len(dropped)} correlated: {dropped}")
    _log(f"  {len(features_after)} features after pruning")

    # 5 -- Standardize -----------------------------------------------------
    _log("Standardizing ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(style_df)

    # 6 -- PCA reduction ---------------------------------------------------
    _log("Running PCA ...")
    X_pca_full, pca_obj, n_keep = pca_reduce(
        X_scaled, variance_target=pca_variance_target,
    )
    cumvar = np.cumsum(pca_obj.explained_variance_ratio_)
    _log(f"  Retaining {n_keep} PCs (explains "
         f"{cumvar[n_keep - 1]:.1%} variance)")

    # 7 -- Check PC1 for quality contamination -----------------------------
    anchor_vals = quality_anchor.reindex(style_df.index).values
    should_drop, pc1_corr = check_pc1_quality(
        X_pca_full, anchor_vals, threshold=drop_pc1_threshold,
    )
    _log(f"  corr(PC1, DEF_RATING) = {pc1_corr:+.3f}"
         f"  -> {'DROP PC1' if should_drop else 'keep PC1'}")

    # Build clustering space
    if should_drop:
        cluster_start = 1
        X_cluster = X_pca_full[:, 1:n_keep]
    else:
        cluster_start = 0
        X_cluster = X_pca_full[:, :n_keep]

    # 8 -- K selection (constrained >= 4) ----------------------------------
    _log("Evaluating K ...")
    best_k, k_results = find_optimal_k(
        X_cluster, k_range=k_range, min_cluster_size=min_cluster_size,
    )
    _log(f"  Selected K = {best_k}")
    _log(f"\n{k_results.to_string(index=False)}")

    # 9 -- Final KMeans ----------------------------------------------------
    _log(f"Training KMeans (K={best_k}, n_init=50) ...")
    km, labels = train_kmeans(X_cluster, best_k)

    # 10 -- PCA coords for visualization (always first two full PCs) -------
    pca_1 = X_pca_full[:, 0]
    pca_2 = X_pca_full[:, 1]

    # 11 -- Assemble output DataFrame --------------------------------------
    _log("Assembling output ...")
    output = meta_df.copy()
    output = output.merge(style_df.reset_index(), on="TEAM_ID", how="left")

    output["DEF_RATING"] = output["TEAM_ID"].map(
        quality_anchor.to_dict()
    )
    output["CLUSTER"] = labels
    output["PCA_1"] = pca_1
    output["PCA_2"] = pca_2

    # Store additional PC columns used for clustering
    for i in range(min(n_keep, X_pca_full.shape[1])):
        output[f"_PC{i + 1}"] = X_pca_full[:, i]

    output["SEASON"] = season
    output["K_USED"] = best_k
    output["CHOSEN_K"] = best_k
    output["PIPELINE_VERSION"] = PIPELINE_VERSION
    output["UPDATED_AT"] = timestamp
    output["PCA_VARIANCE_TARGET"] = pca_variance_target
    output["DROPPED_PC1"] = should_drop
    output["RESIDUALIZED"] = do_residualize

    # Merge in abbreviation from nba_teams.parquet
    teams_path = DATA_CACHE / "nba_teams.parquet"
    if teams_path.exists():
        abbr_map = pd.read_parquet(teams_path).set_index("id")["abbreviation"].to_dict()
        output["TEAM_ABBREVIATION"] = output["TEAM_ID"].map(abbr_map)

    # 12 -- Save canonical Parquet (season-scoped) --------------------------
    canon_path = DATA_CACHE / f"defense_clusters_{season}_full.parquet"
    output.to_parquet(canon_path, index=False)
    _log(f"  Saved -> {canon_path}")

    # Also keep the legacy non-_full path so older loader code still works
    legacy_path = DATA_CACHE / f"defense_clusters_{season}.parquet"
    output.to_parquet(legacy_path, index=False)

    # 13 -- Diagnostics JSON sidecar ---------------------------------------
    diagnostics = {
        "pipeline_version": PIPELINE_VERSION,
        "updated_at": timestamp,
        "season": season,
        "residualized": do_residualize,
        "pca_variance_target": pca_variance_target,
        "pca_components_kept": int(n_keep),
        "pca_explained_variance": pca_obj.explained_variance_ratio_.tolist(),
        "pca_components": pca_obj.components_.tolist(),
        "pca_feature_names": features_after,
        "pc1_quality_corr": round(pc1_corr, 4),
        "dropped_pc1": should_drop,
        "drop_pc1_threshold": drop_pc1_threshold,
        "chosen_k": best_k,
        "k_evaluation": k_results.to_dict("records"),
        "features_before_pruning": features_before,
        "features_after_pruning": features_after,
        "dropped_features": dropped,
    }
    diag_path = DATA_CACHE / f"defense_clusters_{season}_full_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2))
    _log(f"  Diagnostics -> {diag_path}")

    # Legacy diagnostics path
    legacy_diag = DATA_CACHE / f"defense_clusters_{season}_diagnostics.json"
    legacy_diag.write_text(json.dumps(diagnostics, indent=2))

    # 14 -- Backward-compatible legacy files -------------------------------
    compat = output[["TEAM_ID", "CLUSTER"]].rename(columns={"CLUSTER": "cluster"})
    compat.to_parquet(DATA_CACHE / "jan21clusters.parquet", index=False)

    if "TEAM_ABBREVIATION" in output.columns:
        compat_m = output[
            ["TEAM_ID", "CLUSTER", "TEAM_NAME", "TEAM_ABBREVIATION"]
        ].rename(columns={
            "CLUSTER": "cluster",
            "TEAM_NAME": "full_name",
            "TEAM_ABBREVIATION": "abbreviation",
        })
        compat_m.to_parquet(
            DATA_CACHE / "clusters_and_teams_jan21.parquet", index=False,
        )

    # 15 -- Print centroid deviations for CLI readability -------------------
    feat_cols = features_after
    centroid = output.groupby("CLUSTER")[feat_cols].mean()
    league_mu = output[feat_cols].mean()
    league_sd = output[feat_cols].std()
    _log("\nCluster centroid deviations (z-score):")
    _log(((centroid - league_mu) / league_sd).round(2).to_string())

    return output


# ═══════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run defense clusters pipeline")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--no-residualize", action="store_true")
    parser.add_argument("--pca-var", type=float, default=0.85)
    parser.add_argument("--drop-pc1-thresh", type=float, default=0.6)
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=8)
    args = parser.parse_args()

    run_pipeline(
        args.season,
        do_residualize=not args.no_residualize,
        pca_variance_target=args.pca_var,
        drop_pc1_threshold=args.drop_pc1_thresh,
        k_range=(args.k_min, args.k_max),
    )
