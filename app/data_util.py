import pandas as pd
import streamlit as st
from util import read_pqt
from pathlib import Path


# def filter_player_vs_cluster(
#     player_logs: pd.DataFrame,
#     player_ids: list,
#     # opponent_team_ids: list,
#     min_minutes: float = 0.0,
# ) -> pd.DataFrame:
#     df = player_logs[
#         (player_logs["PLAYER_ID"].isin(player_ids)) 
#         # & (player_logs["OPP_TEAM_ID"].isin(opponent_team_ids))
#     ]
#     if min_minutes > 0:
#         df = df[df["MIN"] >= min_minutes]
#     return df.sort_values("GAME_DATE")



BASE = Path(__file__).resolve().parent
DATA = BASE / "data_cache" / "league_player_logs.parquet"

@st.cache_data(show_spinner=False)
def general_player_logs():
    # return read_pqt("data_cache/league_player_logs.parquet")
    return pd.read_parquet(DATA)



def filter_player_logs(df, player_id):
    player_df = df.loc[df.PLAYER_ID == player_id]
    return player_df


def filter_player_vs_cluster(df, opp_team_ids):
    mask = df["OPP_TEAM_ID"].isin(opp_team_ids)
    return df.loc[mask]
