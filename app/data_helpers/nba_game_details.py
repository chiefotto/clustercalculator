from datetime import time
import streamlit as st
from nba_api.stats.endpoints import CommonTeamRoster
import pandas as pd
from data_helpers.nba_dvp import (
    get_dvp_with_def_factors,
    filter_dvp_by_team,
    get_dvp_target_avoid,
    _def_factor_to_pct,
)


@st.cache_data(ttl=60*30)
def get_team_roster_player_ids(team_id: int, season: str = "2025-26") -> list[int]:
    df = CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
    if df.empty:
        return []
    return df


def get_dvp_for_game():
    """Cached DVP table with def_factors, and home/away subsets. Tables kept for later boost/decrease calcs."""
    dvp_table = get_dvp_with_def_factors()
    home_team_id = st.session_state.get("home_team_id")
    away_team_id = st.session_state.get("away_team_id")
    home_team_dvp = filter_dvp_by_team(dvp_table, home_team_id) if home_team_id is not None else pd.DataFrame()
    away_team_dvp = filter_dvp_by_team(dvp_table, away_team_id) if away_team_id is not None else pd.DataFrame()
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
        st.caption(f"Vs {team_label} — % vs league avg (target = weak D, avoid = strong D)")
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

    home_roster_df = get_team_roster_player_ids(home_team_id)
    away_roster_df = get_team_roster_player_ids(away_team_id)


    full_game_roster = pd.concat([home_roster_df, away_roster_df], ignore_index=True)
    display_cols = [ 'TeamID','PLAYER', 'PLAYER_ID', 'POSITION']





    event = st.dataframe(
        full_game_roster[display_cols],
        width='content',
        hide_index=True,
        selection_mode='single-row',
        on_select="rerun"
    )
    # event

    sel = event.selection.get('rows', [])
    if sel:
        row = full_game_roster.iloc[sel[0]]
        st.session_state.player_id = int(row['PLAYER_ID'])
        st.session_state.player_name = (row['PLAYER'])
        st.session_state.player_team_id = (row['TeamID'])
        st.session_state.player_position = row.get('POSITION', row.get('Position', ''))

        st.switch_page("pages/player_view.py")

    
    

