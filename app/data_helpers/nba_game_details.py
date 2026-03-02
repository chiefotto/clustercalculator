import streamlit as st
import pandas as pd
from nba_api.stats.endpoints import CommonTeamRoster

from data_util import general_player_logs
from data_helpers.nba_dvp import (
    get_dvp_with_def_factors,
    filter_dvp_by_team,
    get_dvp_target_avoid,
    _def_factor_to_pct,
)


@st.cache_data(ttl=60 * 30)
def get_team_roster_df(team_id: int, season: str = "2025-26") -> pd.DataFrame:
    """Return raw team roster dataframe from nba_api for a given team/season."""
    df = CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
    if df.empty:
        return pd.DataFrame()
    return df


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from candidates that exists in df, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


@st.cache_data(ttl=60 * 30)
def get_game_logs_roster_for_teams(
    home_team_id: int,
    away_team_id: int,
) -> pd.DataFrame:
    """
    Build a roster-like table from the league player game logs, limited to the
    two teams in the selected game.

    Game logs (parquet) are the authoritative source for which players to show.
    Positions are appended later from team roster data.

    Only include players who have logged at least 3 games in the logs for the
    selected teams.
    """
    logs = general_player_logs()

    team_col = _first_existing_column(logs, ["TEAM_ID", "TeamID", "team_id"])
    player_id_col = _first_existing_column(
        logs, ["PLAYER_ID", "PlayerID", "player_id"]
    )
    player_name_col = _first_existing_column(
        logs, ["PLAYER_NAME", "PLAYER", "player_name"]
    )
    team_name_col = _first_existing_column(
        logs, ["TEAM_NAME", "TeamName", "team_name"]
    )

    # If required columns are missing, bail out gracefully.
    if not all([team_col, player_id_col, player_name_col, team_name_col]):
        return pd.DataFrame()

    # Filter down to the two teams for this game.
    mask = logs[team_col].isin([home_team_id, away_team_id])
    team_logs = logs.loc[mask].copy()

    # Require a minimum of 3 games in the logs for each player.
    games_per_player = (
        team_logs.groupby(player_id_col, observed=True)[player_id_col]
        .count()
        .rename("games_played")
    )
    valid_player_ids = games_per_player[games_per_player >= 3].index

    team_logs = team_logs[team_logs[player_id_col].isin(valid_player_ids)]
    if team_logs.empty:
        return pd.DataFrame()

    subset = (
        team_logs.loc[
            :,
            [player_id_col, player_name_col, team_col, team_name_col],
        ]
        .drop_duplicates(subset=[player_id_col, team_col])
        .copy()
    )

    subset = subset.rename(
        columns={
            player_id_col: "PLAYER_ID",
            player_name_col: "PLAYER",
            team_col: "TeamID",
            team_name_col: "TEAM_NAME",
        }
    )

    return subset


@st.cache_data(ttl=60 * 30)
def build_full_game_roster(
    home_team_id: int,
    away_team_id: int,
    season: str = "2025-26",
) -> pd.DataFrame:
    """
    Combine game-log-derived roster with team roster endpoint to append positions.

    - Game logs determine *who* appears (driven by daily/season logs).
    - Team roster endpoint is used only to map PLAYER_ID → POSITION.
    """
    logs_roster = get_game_logs_roster_for_teams(home_team_id, away_team_id)
    if logs_roster.empty:
        return pd.DataFrame()

    home_roster_df = get_team_roster_df(home_team_id, season=season)
    away_roster_df = get_team_roster_df(away_team_id, season=season)

    roster_df = pd.concat([home_roster_df, away_roster_df], ignore_index=True)

    if roster_df.empty:
        # No roster data available; return logs roster without positions.
        logs_roster = logs_roster.copy()
        logs_roster["POSITION"] = ""
        return logs_roster[["TeamID", "TEAM_NAME", "PLAYER", "PLAYER_ID", "POSITION"]]

    # Columns in the roster dataframe
    roster_player_id_col = _first_existing_column(
        roster_df, ["PLAYER_ID", "PlayerID", "player_id"]
    )
    roster_team_id_col = _first_existing_column(
        roster_df, ["TeamID", "TEAM_ID", "team_id"]
    )
    pos_col = _first_existing_column(roster_df, ["POSITION", "Position"])

    if not roster_player_id_col or not roster_team_id_col:
        # Can't reliably map players to current team rosters.
        logs_roster = logs_roster.copy()
        if "POSITION" not in logs_roster.columns:
            logs_roster["POSITION"] = ""
        return logs_roster[["TeamID", "TEAM_NAME", "PLAYER", "PLAYER_ID", "POSITION"]]

    # Keep only players who are currently on the roster for the given team.
    current_roster_keys = (
        roster_df[[roster_player_id_col, roster_team_id_col]]
        .drop_duplicates()
        .rename(
            columns={
                roster_player_id_col: "PLAYER_ID",
                roster_team_id_col: "TeamID",
            }
        )
    )

    logs_roster = logs_roster.merge(
        current_roster_keys, on=["PLAYER_ID", "TeamID"], how="inner"
    )
    if logs_roster.empty:
        return pd.DataFrame()

    cols = [roster_player_id_col]
    if pos_col:
        cols.append(pos_col)

    roster_positions = roster_df[cols].copy()
    rename_map: dict[str, str] = {roster_player_id_col: "PLAYER_ID"}
    if pos_col:
        rename_map[pos_col] = "POSITION"

    roster_positions = roster_positions.rename(columns=rename_map)
    roster_positions = roster_positions.drop_duplicates(subset=["PLAYER_ID"])

    merged = logs_roster.merge(
        roster_positions[["PLAYER_ID", "POSITION"]],
        on="PLAYER_ID",
        how="left",
    )

    if "POSITION" not in merged.columns:
        merged["POSITION"] = ""

    # Keep both ID and name so we can show the name but still stash the ID.
    merged = merged[["TeamID", "TEAM_NAME", "PLAYER", "PLAYER_ID", "POSITION"]]

    # Sort a bit for nicer display (team first, then name)
    merged = merged.sort_values(["TEAM_NAME", "PLAYER"])

    return merged


def get_dvp_for_game():
    """Cached DVP table with def_factors, and home/away subsets."""
    dvp_table = get_dvp_with_def_factors()
    home_team_id = st.session_state.get("home_team_id")
    away_team_id = st.session_state.get("away_team_id")
    home_team_dvp = (
        filter_dvp_by_team(dvp_table, home_team_id)
        if home_team_id is not None
        else pd.DataFrame()
    )
    away_team_dvp = (
        filter_dvp_by_team(dvp_table, away_team_id)
        if away_team_id is not None
        else pd.DataFrame()
    )
    return dvp_table, home_team_dvp, away_team_dvp


def game_details_page():
    st.title("DVP")



    if "home_team_id" not in st.session_state:
        st.warning("Select a game first.")
        return

    home_team_id = st.session_state["home_team_id"]
    away_team_id = st.session_state["away_team_id"]

    _, home_team_dvp, away_team_dvp = get_dvp_for_game()
    # Tables kept for later: player boost/decrease vs def_factor (not displayed here)

    st.subheader(st.session_state["matchup_label"])

    home_team_name = st.session_state.get("home_team_name", "Home")
    away_team_name = st.session_state.get("away_team_name", "Away")

    def _show_target_avoid(team_dvp: pd.DataFrame, team_label: str):
        target, avoid = get_dvp_target_avoid(team_dvp, top_n=6, bottom_n=6)
        if not target and not avoid:
            return
        st.caption(
            f"Vs {team_label} — % vs league avg (target = weak D, avoid = strong D)"
        )
        left, right = st.columns(2)
        with left:
            if target:
                st.markdown("**Target**")
                grid = st.columns(3)
                for i, (pos, stat, df) in enumerate(target):
                    with grid[i % 3]:
                        st.metric(label=f"{pos} {stat}", value=_def_factor_to_pct(df))
        with right:
            if avoid:
                st.markdown("**Avoid**")
                grid = st.columns(3)
                for i, (pos, stat, df) in enumerate(avoid):
                    with grid[i % 3]:
                        st.metric(label=f"{pos} {stat}", value=_def_factor_to_pct(df))

    if not home_team_dvp.empty:
        _show_target_avoid(home_team_dvp, home_team_name)
    if not away_team_dvp.empty:
        _show_target_avoid(away_team_dvp, away_team_name)

    full_game_roster = build_full_game_roster(home_team_id, away_team_id)
    if full_game_roster.empty:
        st.info(
            "No player game logs available yet for this matchup to build a roster."
        )
        return

    # Show team name in the table instead of the numeric team ID.
    display_cols = ["TEAM_NAME", "PLAYER", "PLAYER_ID", "POSITION"]

    event = st.dataframe(
        full_game_roster[display_cols],
        width="content",
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
    )

    sel = event.selection.get("rows", [])
    if sel:
        row = full_game_roster.iloc[sel[0]]
        st.session_state.player_id = int(row["PLAYER_ID"])
        st.session_state.player_name = row["PLAYER"]
        st.session_state.player_team_id = int(row["TeamID"])
        st.session_state.player_position = row.get(
            "POSITION", row.get("Position", "")
        )

        st.switch_page("pages/player_view.py")