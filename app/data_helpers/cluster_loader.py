"""
Shared loader for the canonical defense-clusters Parquet.

Every Streamlit page that needs clustered defensive data should import
from here rather than reading parquet files directly.  The loader
enforces column guardrails, caches reads, and provides helpers for
diagnostics, last-updated metadata, and multi-season discovery.

Multi-season artifact layout
-----------------------------
Each season produces isolated files -- no cross-season overwrite:

    data_cache/defense_clusters_{season}_full.parquet
    data_cache/defense_clusters_{season}_full_diagnostics.json

Legacy files (defense_clusters_{season}.parquet) are also written by the
pipeline for backward compatibility but are not the canonical source.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CACHE = BASE_DIR / "data_cache"

# Columns that are metadata / not style-features for clustering.
_META = {
    "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION",
    "GP", "W", "L", "W_PCT",
    "CLUSTER", "PCA_1", "PCA_2", "SEASON",
    "ARTIFACT_KEY", "WINDOW_TYPE", "WINDOW_VALUE",
    "K_USED", "CHOSEN_K",
    "DEF_RATING",
    "PIPELINE_VERSION", "UPDATED_AT",
    "PCA_VARIANCE_TARGET", "DROPPED_PC1", "RESIDUALIZED",
}

_PC_PREFIX = "_PC"

_SEASON_RE = re.compile(r"defense_clusters_(\d{4}-\d{2})")
_ARTIFACT_KEY_RE = re.compile(r"defense_clusters_\d{4}-\d{2}_([^.]+)\.parquet")


# ── Path helpers ──────────────────────────────────────────────────────────

def canonical_path_for_season(season: str) -> Path:
    """Return the canonical parquet path for *season* (full artifact)."""
    return DATA_CACHE / f"defense_clusters_{season}_full.parquet"


def canonical_diagnostics_path_for_season(season: str) -> Path:
    """Return the canonical diagnostics JSON path for *season* (full artifact)."""
    return DATA_CACHE / f"defense_clusters_{season}_full_diagnostics.json"


def parquet_path(season: str, artifact_key: str) -> Path:
    """Return the parquet path for *season* and *artifact_key*."""
    return DATA_CACHE / f"defense_clusters_{season}_{artifact_key}.parquet"


def diagnostics_path(season: str, artifact_key: str) -> Path:
    """Return the diagnostics JSON path for *season* and *artifact_key*."""
    return DATA_CACHE / f"defense_clusters_{season}_{artifact_key}_diagnostics.json"


def parse_season_from_filename(filename: str) -> str | None:
    """
    Extract the season string from a cluster artifact filename.

    >>> parse_season_from_filename("defense_clusters_2025-26_full.parquet")
    '2025-26'
    >>> parse_season_from_filename("defense_clusters_2024-25.parquet")
    '2024-25'
    >>> parse_season_from_filename("defense_clusters_2024-25_full_diagnostics.json")
    '2024-25'
    """
    m = _SEASON_RE.search(str(filename))
    return m.group(1) if m else None


def _resolve_parquet_path(season: str) -> Path | None:
    """
    Return the best existing parquet path for *season*.
    Prefers the canonical ``_full`` file, falls back to legacy.
    """
    canon = canonical_path_for_season(season)
    if canon.exists():
        return canon
    legacy = DATA_CACHE / f"defense_clusters_{season}.parquet"
    if legacy.exists():
        return legacy
    return None


def _resolve_diagnostics_path(season: str) -> Path | None:
    canon = canonical_diagnostics_path_for_season(season)
    if canon.exists():
        return canon
    legacy = DATA_CACHE / f"defense_clusters_{season}_diagnostics.json"
    if legacy.exists():
        return legacy
    return None


# ── Season discovery ──────────────────────────────────────────────────────

def available_seasons() -> list[str]:
    """
    Scan data_cache for all season-scoped cluster parquets.
    Checks both canonical (``_full``) and legacy filenames; deduplicates.
    """
    seen: set[str] = set()
    for p in DATA_CACHE.glob("defense_clusters_*.parquet"):
        s = parse_season_from_filename(p.name)
        if s:
            seen.add(s)
    return sorted(seen)


# ── Artifact discovery ───────────────────────────────────────────────────

def list_available_artifacts(season: str) -> list[str]:
    """
    List artifact keys for *season* (e.g. ["full", "last12", "asof_2025-01-15"]).
    Scans data_cache for defense_clusters_{season}_*.parquet and extracts the key.
    Legacy defense_clusters_{season}.parquet is treated as "full".
    """
    keys: set[str] = set()
    prefix = f"defense_clusters_{season}"
    for p in DATA_CACHE.glob(f"{prefix}*.parquet"):
        if p.name == f"{prefix}.parquet":
            keys.add("full")
        else:
            m = _ARTIFACT_KEY_RE.match(p.name)
            if m:
                keys.add(m.group(1))
    return sorted(keys)


def resolve_latest_artifact(season: str) -> str:
    """
    Return the best artifact key for *season*: prefer rolling/asof > full.
    If none exist, returns "full" (caller should check existence).
    """
    available = list_available_artifacts(season)
    if not available:
        return "full"
    # Prefer keys that look like asof_* or last*
    for k in available:
        if k.startswith("asof_") or k.startswith("last"):
            return k
    return available[0] if available else "full"


# ── Core loader ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_defense_clusters(
    season: str = "2025-26",
    artifact_key: str = "full",
) -> pd.DataFrame:
    """
    Load the defense-clusters parquet for *season* and *artifact_key*.

    Resolution for artifact_key="full": tries canonical _full path, then legacy.
    Guardrails: requires TEAM_ID, TEAM_NAME, CLUSTER.
    """
    if artifact_key == "full":
        path = _resolve_parquet_path(season)
    else:
        path = parquet_path(season, artifact_key)
        if not path.exists():
            path = None
    if path is None:
        st.error(
            f"No cluster file for season **{season}** artifact **{artifact_key}**.  "
            "Use **Update clusters now** to generate it."
        )
        return pd.DataFrame()

    df = pd.read_parquet(path)

    required = ["TEAM_ID", "TEAM_NAME", "CLUSTER"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Parquet is missing required columns: {missing}")
        return pd.DataFrame()

    return df


def invalidate_cache():
    """Clear the cached cluster load so the next call reads fresh data."""
    load_defense_clusters.clear()


def load_diagnostics(season: str, artifact_key: str = "full") -> dict | None:
    """Load the sidecar diagnostics JSON for *season* and *artifact_key*."""
    if artifact_key == "full":
        path = _resolve_diagnostics_path(season)
    else:
        path = diagnostics_path(season, artifact_key)
        if not path.exists():
            path = None
    if path is not None:
        return json.loads(path.read_text())
    return None


def last_updated(season: str, artifact_key: str = "full") -> str | None:
    """Return the ISO timestamp of the last pipeline run for this artifact, or None."""
    diag = load_diagnostics(season, artifact_key)
    if diag:
        return diag.get("updated_at")
    if artifact_key == "full":
        path = _resolve_parquet_path(season)
    else:
        path = parquet_path(season, artifact_key)
    if path is not None and path.exists():
        ts = datetime.fromtimestamp(path.stat().st_mtime)
        return ts.isoformat()
    return None


# ── Get-or-build ─────────────────────────────────────────────────────────

def get_or_build_defense_clusters(
    season: str,
    *,
    force_refresh: bool = False,
    log_fn=None,
) -> pd.DataFrame:
    """
    Return the cluster DataFrame for *season*.

    If the canonical file is missing **or** *force_refresh* is True the
    pipeline is executed first, the parquet is saved, and the cache is
    invalidated before loading.
    """
    path = _resolve_parquet_path(season)

    if path is None or force_refresh:
        from data_helpers.defense_clusters_pipeline import run_pipeline
        run_pipeline(season, log_fn=log_fn)
        invalidate_cache()

    return load_defense_clusters(season)


# ── Column helpers ────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return only the engineered style-feature columns (no metadata, no _PC*)."""
    return [
        c for c in df.columns
        if c not in _META
        and not c.startswith(_PC_PREFIX)
        and df[c].dtype in (np.float64, np.int64, np.float32)
    ]


def get_meta_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c in _META]
    return df[cols].copy()


# ── Team-season index (for similar-teams search) ───────────────────────────

@st.cache_data(ttl=3600)
def build_team_season_index(
    include_latest_current: bool = False,
    _seasons: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Build a unified (season, artifact_key, team, cluster, feature_vector) index
    from full-season artifacts for the last 10 seasons.

    Returns (index_df, feature_columns_used, log_info).
    Uses global intersection of features across all loaded artifacts.
    NaN in feature vectors is imputed as 0 (league average).
    """
    seasons = list(_seasons) if _seasons else sorted(available_seasons())
    if not seasons:
        return pd.DataFrame(), [], {"message": "No seasons available"}

    # Prefer full for all; for current season optionally add latest
    current = seasons[-1] if seasons else None
    rows: list[dict] = []
    all_feature_sets: list[set[str]] = []

    for season in seasons:
        keys = ["full"]
        if include_latest_current and season == current:
            latest = resolve_latest_artifact(season)
            if latest != "full" and parquet_path(season, latest).exists():
                keys.append(latest)
        for artifact_key in keys:
            path = parquet_path(season, artifact_key) if artifact_key != "full" else _resolve_parquet_path(season)
            if path is None or not path.exists():
                continue
            df = pd.read_parquet(path)
            if df.empty or "CLUSTER" not in df.columns:
                continue
            feat = get_feature_columns(df)
            all_feature_sets.append(set(feat))
            league_mu = df[feat].mean()
            league_sd = df[feat].std().replace(0, np.nan)
            cluster_labels = compute_cluster_labels(df, league_mu, league_sd)
            for _, row in df.iterrows():
                cid = int(row["CLUSTER"])
                z_vec = (row[feat] - league_mu) / league_sd
                z_vec = z_vec.fillna(0)
                rows.append({
                    "season": season,
                    "artifact_key": artifact_key,
                    "team_id": row["TEAM_ID"],
                    "team_name": row["TEAM_NAME"],
                    "cluster_id": cid,
                    "cluster_label": cluster_labels.get(cid, ""),
                    "feature_vector": z_vec.values.tolist(),
                    "feature_names": feat,
                })
    if not rows:
        return pd.DataFrame(), [], {"message": "No rows loaded"}

    # Global intersection of features
    common = set(all_feature_sets[0])
    for s in all_feature_sets[1:]:
        common &= s
    common = sorted(common)

    # Rebuild vectors using only common features (and align order)
    index_rows = []
    for r in rows:
        names = r["feature_names"]
        vec = r["feature_vector"]
        z_map = dict(zip(names, vec))
        aligned = [float(z_map.get(f, 0)) for f in common]
        index_rows.append({
            "season": r["season"],
            "artifact_key": r["artifact_key"],
            "team_id": r["team_id"],
            "team_name": r["team_name"],
            "cluster_id": r["cluster_id"],
            "cluster_label": r.get("cluster_label", ""),
            "feature_vector": aligned,
        })
    index_df = pd.DataFrame(index_rows)
    log_info = {"features_used": len(common), "seasons": len(seasons), "rows": len(index_df)}
    return index_df, common, log_info


# ── Cached aggregates ────────────────────────────────────────────────────

@st.cache_data
def compute_cluster_means(df: pd.DataFrame) -> pd.DataFrame:
    feat = get_feature_columns(df)
    return df.groupby("CLUSTER")[feat].mean()


@st.cache_data
def compute_league_means(df: pd.DataFrame) -> pd.Series:
    feat = get_feature_columns(df)
    return df[feat].mean()


@st.cache_data
def compute_league_stds(df: pd.DataFrame) -> pd.Series:
    feat = get_feature_columns(df)
    return df[feat].std()


# ── Display helpers (Units toggle) ────────────────────────────────────────

def z_to_percentile(z: float | np.ndarray) -> float | np.ndarray:
    """Convert z-score(s) to percentile(s) in [0, 100]."""
    return np.clip(norm.cdf(z) * 100, 0, 100)


def percentile_to_band(p: float) -> str:
    if p < 10:
        return "Very Low"
    if p < 25:
        return "Low"
    if p < 75:
        return "Typical"
    if p < 90:
        return "High"
    return "Very High"


_BAND_COLOR = {
    "Very Low": "#2471A3",
    "Low":      "#5DADE2",
    "Typical":  "#85929E",
    "High":     "#E67E22",
    "Very High": "#C0392B",
}


def band_color(band: str) -> str:
    return _BAND_COLOR.get(band, "#85929E")


def format_feature_value(z: float, units_mode: str) -> str:
    """Return a display string for *z* given the selected *units_mode*."""
    p = float(z_to_percentile(z))
    b = percentile_to_band(p)
    if units_mode == "Z-score (σ)":
        return f"{z:+.2f}σ"
    if units_mode == "Percentile":
        return f"{p:.0f}th pct"
    return b


def hover_line(feature: str, z: float) -> str:
    """Rich hover line that always shows percentile + band + z."""
    p = float(z_to_percentile(z))
    b = percentile_to_band(p)
    return f"{feature}: {p:.0f}th pct ({b}) · {z:+.2f}σ"


# ── Feature label map (human-readable short phrases) ──────────────────────

FEATURE_LABEL_MAP: dict[str, tuple[str, str]] = {
    # (positive_phrase, negative_phrase)
    "OPP_FG3A":                       ("High 3PA allowed",       "Low 3PA allowed"),
    "OPP_FG3_PCT":                    ("Poor perimeter D",       "Tight perimeter D"),
    "OPP_FTA":                        ("Foul-prone",             "Disciplined fouling"),
    "OPP_OREB":                       ("Gives up OREBs",         "Limits OREBs"),
    "OPP_AST":                        ("Allows ball movement",   "Disrupts passing"),
    "OPP_TOV":                        ("Forces turnovers",       "Low turnover pressure"),
    "STL":                            ("Active steals",          "Low steal rate"),
    "BLK":                            ("Shot-blocking",          "Low block rate"),
    "DREB_PCT":                       ("Strong rebounding",      "Weak rebounding"),
    "OPP_FTA_RATE":                   ("Foul-prone",             "Disciplined fouling"),
    "OPP_TOV_PCT":                    ("Turnover pressure",      "Low pressure"),
    "OPP_OREB_PCT":                   ("Gives up OREBs",         "Limits OREBs"),
    "OPP_PTS_OFF_TOV":                ("Pts off TOs allowed",    "Limits pts off TOs"),
    "OPP_PTS_2ND_CHANCE":             ("2nd-chance vulnerable",  "Limits 2nd-chance"),
    "OPP_PTS_FB":                     ("Transition vulnerable",  "Limits transition"),
    "OPP_PTS_PAINT":                  ("Paint vulnerable",       "Paint stingy"),
    "DEFLECTIONS_PER100":             ("Active hands",           "Passive hands"),
    "CONTESTED_SHOTS_2PT_PER100":     ("Contests 2PT shots",     "Low 2PT contests"),
    "CONTESTED_SHOTS_3PT_PER100":     ("Contests 3PT shots",     "Low 3PT contests"),
    "CHARGES_DRAWN_PER100":           ("Draws charges",          "Few charges"),
    "DEF_LOOSE_BALLS_RECOVERED_PER100": ("Recovers loose balls", "Loses loose balls"),
    "DEF_BOXOUTS_PER100":             ("Strong boxouts",         "Weak boxouts"),
    "OPP_SHOT_FG3A_FREQ":            ("High 3PA frequency",     "Low 3PA frequency"),
}


def _feature_phrase(feature: str, z: float) -> str:
    """Return a short human-readable phrase for *feature* given deviation *z*."""
    pair = FEATURE_LABEL_MAP.get(feature)
    if pair:
        return pair[0] if z > 0 else pair[1]
    direction = "High" if z > 0 else "Low"
    short = feature.replace("_PER100", "").replace("OPP_", "").replace("_", " ").title()
    return f"{direction} {short}"


@st.cache_data
def compute_cluster_labels(
    df: pd.DataFrame,
    _league_means: pd.Series,
    _league_stds: pd.Series,
) -> dict[int, str]:
    """
    Generate a human-readable label for each cluster based on its
    strongest deviations from the league mean.
    """
    feature_cols = get_feature_columns(df)
    labels: dict[int, str] = {}
    for cid in sorted(df["CLUSTER"].unique()):
        mask = df["CLUSTER"] == cid
        cluster_mean = df.loc[mask, feature_cols].mean()
        z = ((cluster_mean - _league_means) / _league_stds).dropna()

        pos = z[z > 0.5].sort_values(ascending=False)
        neg = z[z < -0.5].sort_values()

        parts: list[str] = []
        for feat in pos.head(2).index:
            parts.append(_feature_phrase(feat, pos[feat]))
        if not neg.empty:
            parts.append(_feature_phrase(neg.index[0], neg.iloc[0]))

        label = " + ".join(parts[:2])
        if len(parts) > 2:
            label += f" / {parts[2]}"
        if not label:
            label = "League average"
        labels[cid] = label[:50]

    return labels


def cluster_display_name(cid: int, labels: dict[int, str]) -> str:
    return f"{cid} — {labels.get(cid, '')}"


# ── PCA axis interpretation ──────────────────────────────────────────────

_THEME_KEYWORDS: dict[str, list[str]] = {
    "Pressure":   ["OPP_TOV_PCT", "DEFLECTIONS_PER100", "OPP_TOV", "STL"],
    "Perimeter":  ["OPP_FG3A", "CONTESTED_SHOTS_3PT_PER100", "OPP_FG3_PCT",
                   "OPP_SHOT_FG3A_FREQ"],
    "Paint":      ["OPP_PTS_PAINT", "BLK", "CONTESTED_SHOTS_2PT_PER100"],
    "Fouling":    ["OPP_FTA_RATE", "OPP_FTA"],
    "Transition": ["OPP_PTS_FB"],
    "Rebounding": ["OPP_OREB_PCT", "DREB_PCT", "DEF_BOXOUTS_PER100", "OPP_OREB"],
    "Hustle":     ["DEFLECTIONS_PER100", "CHARGES_DRAWN_PER100",
                   "DEF_LOOSE_BALLS_RECOVERED_PER100"],
}


def interpret_pc_axis(
    loadings: list[float],
    feature_names: list[str],
    top_n: int = 5,
) -> tuple[str, list[tuple[str, float]], list[tuple[str, float]]]:
    """
    Interpret a single PCA component.

    Returns (subtitle, top_positive, top_negative) where each list is
    [(feature_name, loading), ...].
    """
    pairs = sorted(zip(feature_names, loadings), key=lambda x: x[1])
    top_neg = [(f, round(v, 3)) for f, v in pairs[:top_n]]
    top_pos = [(f, round(v, 3)) for f, v in pairs[-top_n:]][::-1]

    def _theme(items: list[tuple[str, float]]) -> str:
        feats = {f for f, _ in items[:3]}
        best_theme, best_count = "", 0
        for theme, keywords in _THEME_KEYWORDS.items():
            count = len(feats & set(keywords))
            if count > best_count:
                best_theme, best_count = theme, count
        if best_count >= 1:
            return best_theme
        short = items[0][0].replace("_PER100", "").replace("OPP_", "")
        return short.replace("_", " ").title()

    pos_theme = _theme(top_pos)
    neg_theme = _theme(top_neg)
    subtitle = f"{pos_theme} ↔ {neg_theme}"

    return subtitle, top_pos, top_neg
