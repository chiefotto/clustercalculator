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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CACHE = BASE_DIR / "data_cache"

# Columns that are metadata / not style-features for clustering.
_META = {
    "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION",
    "GP", "W", "L", "W_PCT",
    "CLUSTER", "PCA_1", "PCA_2", "SEASON",
    "K_USED", "CHOSEN_K",
    "DEF_RATING",
    "PIPELINE_VERSION", "UPDATED_AT",
    "PCA_VARIANCE_TARGET", "DROPPED_PC1", "RESIDUALIZED",
}

_PC_PREFIX = "_PC"

_SEASON_RE = re.compile(r"defense_clusters_(\d{4}-\d{2})")


# ── Path helpers ──────────────────────────────────────────────────────────

def canonical_path_for_season(season: str) -> Path:
    """Return the canonical parquet path for *season*."""
    return DATA_CACHE / f"defense_clusters_{season}_full.parquet"


def canonical_diagnostics_path_for_season(season: str) -> Path:
    """Return the canonical diagnostics JSON path for *season*."""
    return DATA_CACHE / f"defense_clusters_{season}_full_diagnostics.json"


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


# ── Core loader ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_defense_clusters(season: str = "2025-26") -> pd.DataFrame:
    """
    Load the defense-clusters parquet for *season*.

    Resolution order: canonical ``_full`` path first, then legacy.
    Guardrails: requires TEAM_ID, TEAM_NAME, CLUSTER.
    """
    path = _resolve_parquet_path(season)
    if path is None:
        st.error(
            f"No cluster file for season **{season}**.  "
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


# ── Diagnostics ──────────────────────────────────────────────────────────

def load_diagnostics(season: str) -> dict | None:
    """Load the sidecar diagnostics JSON produced by the pipeline."""
    path = _resolve_diagnostics_path(season)
    if path is not None:
        return json.loads(path.read_text())
    return None


def last_updated(season: str) -> str | None:
    """Return the ISO timestamp of the last pipeline run, or None."""
    diag = load_diagnostics(season)
    if diag:
        return diag.get("updated_at")
    path = _resolve_parquet_path(season)
    if path is not None:
        ts = datetime.fromtimestamp(path.stat().st_mtime)
        return ts.isoformat()
    return None
