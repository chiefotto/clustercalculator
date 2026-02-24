import pandas as pd
import streamlit as st
from util import DATA_CACHE
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# parent.parent because: app/pages/home.py → app/

DATA_CACHE = BASE_DIR / "data_cache"

# DVP file uses short abbreviations; map to nba_teams.parquet (long) abbreviations
DVP_ABBR_TO_NBA = {
    "NY": "NYK",
    "PHO": "PHX",
    "GS": "GSW",
    "SA": "SAS",
    "NO": "NOP",
}


def read_dvp_file():
    dvp_file = pd.read_csv(DATA_CACHE / "nba_dvp - Sheet1.csv")
    return dvp_file


def split_columns_into_value_and_rank(dvp_file: pd.DataFrame) -> pd.DataFrame:
    """
    For each column, split cell values on whitespace (2+ spaces).
    Keep the first part as the column value; put the second part in a new column named '{col}_rank'.
    """
    df = dvp_file.copy()
    for col in df.columns:
        # Split on 2+ spaces, at most 2 parts (value, rank)
        split = df[col].astype(str).str.split(r"\s{2,}", n=1, expand=True, regex=True)
        if split.shape[1] == 2:
            df[col] = split[0]
            df[f"{col}_rank"] = split[1]
    return df


def merge_dvp_with_teams(
    dvp_df: pd.DataFrame,
    team_col: str = "Sort: Team",
) -> pd.DataFrame:
    """
    Merge DVP table with nba_teams.parquet to add TEAM_ID and TEAM_NAME.
    Converts DVP short abbreviations (NY, PHO, GS, SA, NO) to long form first, then joins on
    nba_teams.abbreviation.
    """
    df = dvp_df.copy()
    df[team_col] = df[team_col].replace(DVP_ABBR_TO_NBA)
    nba_teams = pd.read_parquet(DATA_CACHE / "nba_teams.parquet")
    merged = df.merge(
        nba_teams[["id", "full_name", "abbreviation"]],
        left_on=team_col,
        right_on="abbreviation",
        how="left",
    )
    merged = merged.drop(columns=["abbreviation"])
    merged = merged.rename(columns={
        "id": "TEAM_ID",
        "full_name": "TEAM_NAME",
    })
    return merged

# Stat value columns (numeric) used for league avg and def_factor (exclude Position, Team)
DVP_STAT_COLUMNS = [
    "Sort: PTS", "Sort: FG%", "Sort: FT%", "Sort: 3PM",
    "Sort: REB", "Sort: AST", "Sort: STL", "Sort: BLK", "Sort: TO",
]

# Player position (e.g. PG-SG) -> DVP group (planning: pg-sg->G, sg-sf->G-F, sf-pf->F, sf-pf-c->F-C, c->C)
# Normalize by lowercasing and sorting parts so "SG-PG" -> "pg-sg" -> G
_POSITION_PARTS_TO_GROUP = {
    "c": "C",
    "pg-sg": "G",
    "sg-sf": "G-F",
    "sf-pf": "F",
    "c-pf-sf": "F-C",  # sf-pf-c normalized
    "c-sf-pf": "F-C",
}
GROUP_TO_DVP_POSITIONS = {
    "G": ["PG", "SG"],
    "G-F": ["SG", "SF"],
    "F": ["SF", "PF"],
    "F-C": ["SF", "PF", "C"],
    "C": ["C"],
}
POSITION_COL = "Sort: Position"
# DVP stat column -> display name (and optional player log column for avg)
DVP_STAT_TO_LABEL = {
    "Sort: PTS": "PTS",
    "Sort: FG%": "FG%",
    "Sort: FT%": "FT%",
    "Sort: 3PM": "3PM",
    "Sort: REB": "REB",
    "Sort: AST": "AST",
    "Sort: STL": "STL",
    "Sort: BLK": "BLK",
    "Sort: TO": "TOV",
}


def _player_position_to_dvp_positions(position_str: str) -> list[str]:
    """Convert roster position (e.g. 'PG-SG', 'G') to list of DVP positions [PG, SG]."""
    if not position_str or pd.isna(position_str):
        return []
    raw = str(position_str).strip().upper()
    if raw in ("G", "F", "C", "G-F", "F-C"):
        group = {"G": "G", "F": "F", "C": "C", "G-F": "G-F", "F-C": "F-C"}[raw]
        return GROUP_TO_DVP_POSITIONS.get(group, [])
    parts = sorted([p.strip().lower() for p in raw.replace(" ", "-").split("-") if p.strip()])
    key = "-".join(parts)
    group = _POSITION_PARTS_TO_GROUP.get(key)
    if group:
        return GROUP_TO_DVP_POSITIONS.get(group, [])
    if raw in ("PG", "SG", "SF", "PF", "C"):
        return [raw]
    return []


def get_opponent_dvp_for_player(
    dvp_df: pd.DataFrame,
    opp_team_id: int,
    player_position_str: str,
) -> pd.DataFrame:
    """
    Filter DVP to opponent team and player's position(s). For multi-position, average def_factors.
    Returns a dataframe with columns: stat, def_factor (and optionally raw def_factor columns).
    """
    opp_dvp = filter_dvp_by_team(dvp_df, opp_team_id)
    if opp_dvp.empty:
        return pd.DataFrame()
    dvp_positions = _player_position_to_dvp_positions(player_position_str)
    if not dvp_positions:
        return pd.DataFrame()
    subset = opp_dvp.loc[opp_dvp[POSITION_COL].isin(dvp_positions)]
    if subset.empty:
        return pd.DataFrame()
    def_cols = [c for c in subset.columns if c.endswith("_def_factor")]
    if not def_cols:
        return pd.DataFrame()
    # Average def_factor per stat across the position rows
    means = subset[def_cols].mean()
    rows = []
    for col in def_cols:
        stat_col = col.replace("_def_factor", "")
        stat_label = DVP_STAT_TO_LABEL.get(stat_col, stat_col)
        rows.append({"stat": stat_label, "def_factor": round(means[col], 4)})
    out = pd.DataFrame(rows)
    out = out.sort_values("def_factor", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)  # 1 = best matchup (highest def_factor)
    return out


def filter_dvp_by_team(dvp_df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Return DVP rows for the given TEAM_ID (one row per position matchup)."""
    return dvp_df.loc[dvp_df["TEAM_ID"] == team_id].copy()


def add_league_def_factors(dvp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add league average and def_factor for each stat column.
    League avg is over the full table. def_factor = value / league_avg;
    < 1 means this defense allows less than league (better). Lower rank = better defense.
    """
    df = dvp_df.copy()
    for col in DVP_STAT_COLUMNS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        league_avg = series.mean()
        if league_avg and league_avg != 0:
            df[f"{col}_def_factor"] = series / league_avg
    return df


def merged_df():
    df = read_dvp_file()
    df_1 = split_columns_into_value_and_rank(df)
    return merge_dvp_with_teams(df_1)


@st.cache_data(ttl=60 * 60)
def get_dvp_with_def_factors() -> pd.DataFrame:
    """Cached full DVP table with def_factors. Use this to avoid recomputing when switching games."""
    df = merged_df()
    return add_league_def_factors(df)


def _def_factor_to_pct(def_factor: float) -> str:
    """Format def_factor as % above/below league. 1.0 = average (shown as +0%)."""
    pct = (def_factor - 1.0) * 100
    if pct >= 0:
        return f"+{pct:.0f}%"
    return f"{pct:.0f}%"


def _def_factor_col_to_stat_label(col: str) -> str:
    """e.g. 'Sort: PTS_def_factor' -> 'PTS', 'Sort: FG%_def_factor' -> 'FG%'."""
    return col.replace("Sort: ", "").replace("_def_factor", "").strip()


def get_dvp_target_avoid(
    team_dvp: pd.DataFrame,
    top_n: int = 6,
    bottom_n: int = 6,
    min_edge: float = 0.02,
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    """
    From a team's DVP table, return (position, stat) combos to target vs avoid.
    Each cell is one (position, stat) with a def_factor. Higher = target, lower = avoid.
    min_edge: ignore def_factors within (1 - min_edge, 1 + min_edge) so league-avg is excluded.
    Returns (target_list, avoid_list), each list of (position, stat_label, def_factor).
    """
    if team_dvp.empty:
        return [], []
    def_cols = [c for c in team_dvp.columns if c.endswith("_def_factor")]
    if not def_cols:
        return [], []
    pos_col = POSITION_COL
    if pos_col not in team_dvp.columns:
        return [], []

    rows: list[tuple[str, str, float]] = []
    for _, row in team_dvp.iterrows():
        pos = row[pos_col]
        for col in def_cols:
            val = row[col]
            if pd.isna(val):
                continue
            v = float(val)
            rows.append((pos, _def_factor_col_to_stat_label(col), v))

    rows.sort(key=lambda x: x[2], reverse=True)
    target = [r for r in rows if r[2] > 1.0 + min_edge][:top_n]
    avoid = [r for r in rows if r[2] < 1.0 - min_edge][-bottom_n:]
    avoid.reverse()  # worst (lowest) first for readability
    return target, avoid










