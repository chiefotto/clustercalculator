### makes request to get player logs and caches it
### should i download or make a request every time
### figure out how to cache the entire table once a 
### day, then use this file to 

from pathlib import Path
import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import playergamelogs
from util import read_pqt


teams_df = read_pqt("data_cache/nba_teams.parquet")

abbrev_to_id = dict(zip(teams_df["abbreviation"], teams_df["id"]))
abbrev_to_full_name = dict(zip(teams_df["abbreviation"], teams_df["full_name"]))
# test = read_pqt("data_cache/league_player_logs.parquet")


# print("test", test.columns)



def add_opp_team_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TEAM_ABBR"] = df["MATCHUP"].str.split().str[0]
    df["OPP_TEAM_ABBR"] = df["MATCHUP"].str.split().str[-1]

    df["OPP_TEAM_ID"] = df["OPP_TEAM_ABBR"].map(abbrev_to_id)
    df["OPP_TEAM_NAME"] = df["OPP_TEAM_ABBR"].map(abbrev_to_full_name)
    team_first = df["MATCHUP"].str.split().str[0]
    # if you have TEAM_ABBREVIATION column already:
    if "TEAM_ABBREVIATION" in df.columns:
        bad = df[team_first != df["TEAM_ABBREVIATION"]]
    
    if len(bad):
        st.warning(f"Found {len(bad)} rows where MATCHUP doesn't start with TEAM_ABBREVIATION")
        st.dataframe(bad[["TEAM_ABBREVIATION", "MATCHUP"]].head(20))

    return df


CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LEAGUE_LOGS_PATH = CACHE_DIR / "league_player_logs.parquet"

# --- Your endpoint fetch (you implement) ---
def fetch_league_player_logs() -> pd.DataFrame:
    """
    Return a DataFrame with at least: GAME_ID, PLAYER_ID
    plus whatever other columns you need.
    """

    raw = playergamelogs.PlayerGameLogs(
    season_nullable="2025-26",
    season_type_nullable="Regular Season"
    ).get_data_frames()[0]
    return raw



@st.cache_data(show_spinner=False)
def load_league_logs() -> pd.DataFrame:
    if LEAGUE_LOGS_PATH.exists():
        return pd.read_parquet(LEAGUE_LOGS_PATH)
    return pd.DataFrame()

def upsert_league_logs(dedupe_cols=("GAME_ID", "PLAYER_ID")) -> dict:
    """
    Fetches latest league logs, merges with disk, keeps only new rows.
    Returns stats for UI.
    """
    existing = pd.read_parquet(LEAGUE_LOGS_PATH) if LEAGUE_LOGS_PATH.exists() else pd.DataFrame()

    fresh = fetch_league_player_logs()

    if existing.empty:
        merged = fresh.drop_duplicates(list(dedupe_cols))
        merged.to_parquet(LEAGUE_LOGS_PATH, index=False)
        load_league_logs.clear()
        return {"added": len(merged), "total": len(merged), "was_empty": True}

    # Only keep rows that don't already exist
    existing_keys = set(map(tuple, existing[list(dedupe_cols)].astype(str).values))
    fresh_keys = fresh[list(dedupe_cols)].astype(str).apply(tuple, axis=1)

    new_rows_mask = ~fresh_keys.isin(existing_keys)
    new_rows = fresh.loc[new_rows_mask].copy()

    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = merged.drop_duplicates(list(dedupe_cols), keep="last")

    merged = add_opp_team_id(merged)  # ✅ backfill for all rows


    merged.to_parquet(LEAGUE_LOGS_PATH, index=False)

    # Clear cached load so UI reads updated file
    load_league_logs.clear()

    return {"added": len(new_rows), "total": len(merged), "was_empty": False}


existing = pd.read_parquet(LEAGUE_LOGS_PATH)
print("Has OPP_TEAM_ID?", "OPP_TEAM_ID" in existing.columns)
print("Null OPP_TEAM_ID count:", existing["OPP_TEAM_ID"].isna().sum() if "OPP_TEAM_ID" in existing.columns else "n/a")
