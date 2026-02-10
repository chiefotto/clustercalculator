from datetime import time
import streamlit as st
from nba_api.stats.endpoints import CommonTeamRoster
import pandas as pd

@st.cache_data(ttl=60*30)
def get_team_roster_player_ids(team_id: int, season: str = "2025-26") -> list[int]:
    df = CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
    if df.empty:
        return []
    return df
    # return df["PLAYER_ID"].dropna().astype(int).tolist()


 ### need to get df from ^ that request, add 
# _roster_cache: dict[tuple, list[int]] = {}
# def get_team_roster_player_ids_cached(team_id: int, season: str = "2025-26", sleep_s: float = 0.6, verbose: bool = True) -> list[int]:
#     key = (team_id, season)
#     if key in _roster_cache:
#         return _roster_cache[key]

#     try:
#         ids = get_team_roster_player_ids(team_id=team_id, season=season)
#         _roster_cache[key] = ids
#         if verbose:
#             print(f"Roster cached: team {team_id} -> {len(ids)} players")
#         time.sleep(sleep_s)
#         return ids
#     except Exception as e:
#         if verbose:
#             print(f"❌ Roster error team {team_id}: {e}")
#         _roster_cache[key] = []
#         time.sleep(sleep_s)
#         return []



def game_details_page():
    st.title("Game Details")

    if "home_team_id" not in st.session_state:
        st.warning("Select a game first.")
        return

    st.subheader(st.session_state["matchup_label"])
    # st.write("Home:", st.session_state["home_team_id"])
    # st.write("Away:", st.session_state["away_team_id"])

    home_team_id = st.session_state['home_team_id']
    away_team_id = st.session_state['away_team_id']
    home_team_name = st.session_state['home_team_name']
    home_roster_df = get_team_roster_player_ids(home_team_id)
    away_roster_df = get_team_roster_player_ids(away_team_id)
    # keep_cols = ['TeamID','PLAYER', 'PLAYER_ID']
    # home_roster_df = home_roster_df
    # away_roster_df = away_roster_df[keep_cols]

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
        


        st.switch_page("pages/player.py")

    
    



    # st.write("Roster for ", st.session_state.home_team_name)
    # st.dataframe(home_roster_df[['PLAYER', 'PLAYER_ID']], width="content")
    # st.write("Roster for ", st.session_state.away_team_name)

    # st.dataframe(away_roster_df[['PLAYER','PLAYER_ID']], width='content')

    # if st.button("⬅ Back"):
    #     st.switch_page("pages/home.py")

game_details_page()