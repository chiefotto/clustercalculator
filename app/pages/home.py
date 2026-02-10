import streamlit as st
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import ScheduleLeagueV2
from nba_api.stats.endpoints import playergamelogs

BASE_DIR = Path(__file__).resolve().parent.parent
# parent.parent because: app/pages/home.py → app/

DATA_CACHE = BASE_DIR / "data_cache"


## make request and returrns dataframe
## seperate function for cleaning and formatting table
## another function to filter to todays date
@st.cache_data(ttl=60*30) 
def inital_sched_req ()-> pd.DataFrame:
    sched_req = ScheduleLeagueV2(league_id="00", season="2025-26")
    full_schedule_raw = sched_req.get_data_frames()[0]
    cleaned_schedule_raw = full_schedule_raw[['leagueId', 'seasonYear', 'gameId', 'gameDateEst', 'gameTimeEst', 'weekNumber', 'monthNum', 'homeTeam_teamId', 'awayTeam_teamId']]
    return cleaned_schedule_raw 

# intial_sched_pull = inital_sched_req()

def format_raw_sched (raw_sched:pd.DataFrame)->pd.DataFrame:
    raw_sched['gameDateEst'] = pd.to_datetime(raw_sched['gameDateEst']).dt.date
    raw_sched = pd.DataFrame(raw_sched)
    return raw_sched

# formatted_sched = format_raw_sched(intial_sched_pull)

def filter_for_todays_games(formatted_sched:pd.DataFrame)->pd.DataFrame:
    todays_date = pd.Timestamp.today().date()
    todays_schedule = formatted_sched.loc[formatted_sched["gameDateEst"] == todays_date].copy(deep=True)
    return todays_schedule

# todays_games = filter_for_todays_games(formatted_sched)



def add_clusters_to_table(schedule: pd.DataFrame) -> pd.DataFrame:
    clusters_df = pd.read_parquet(DATA_CACHE / "clusters_and_teams_jan21.parquet")[
        ["TEAM_ID", "cluster", "full_name", "abbreviation"]
    ].copy()

    # normalize types
    clusters_df["TEAM_ID"] = clusters_df["TEAM_ID"].astype(int)

    schedule = schedule.copy()
    schedule["homeTeam_teamId"] = schedule["homeTeam_teamId"].astype(int)
    schedule["awayTeam_teamId"] = schedule["awayTeam_teamId"].astype(int)

    cluster_map = clusters_df.set_index("TEAM_ID")["cluster"].to_dict()
    name_map    = clusters_df.set_index("TEAM_ID")["full_name"].to_dict()
    abbr_map    = clusters_df.set_index("TEAM_ID")["abbreviation"].to_dict()

    schedule.loc[:, "home_team_cluster"] = schedule["homeTeam_teamId"].map(cluster_map)
    schedule.loc[:, "away_team_cluster"] = schedule["awayTeam_teamId"].map(cluster_map)

    schedule.loc[:, "home_team_name"] = schedule["homeTeam_teamId"].map(name_map)
    schedule.loc[:, "away_team_name"] = schedule["awayTeam_teamId"].map(name_map)

    schedule.loc[:, "home_team_abbr"] = schedule["homeTeam_teamId"].map(abbr_map)
    schedule.loc[:, "away_team_abbr"] = schedule["awayTeam_teamId"].map(abbr_map)

    return schedule

# full_card = add_clusters_to_table(todays_games)






def create_daily_sched_table ():
    intial_sched_pull = inital_sched_req()
    formatted_sched = format_raw_sched(intial_sched_pull)
    todays_games = filter_for_todays_games(formatted_sched)
    full_card = add_clusters_to_table(todays_games)
    return full_card





def home_page():
    full_card = create_daily_sched_table()
    pretty_df = full_card[[
        "away_team_name", "away_team_cluster","awayTeam_teamId",
        "home_team_name", "home_team_cluster","homeTeam_teamId"
    ]].copy()

    # # render your dataframe
    event = st.dataframe(
        pretty_df,
        width='content',
        selection_mode="single-row",
        on_select="rerun",
    )
   
    

    sel = event.selection.get("rows", [])
    if sel:
        row = pretty_df.iloc[sel[0]]

        st.session_state.home_team_id = int(row["homeTeam_teamId"])
        st.session_state.away_team_id = int(row["awayTeam_teamId"])
        st.session_state.home_team_name = row["home_team_name"]
        st.session_state.away_team_name = row["away_team_name"]
        st.session_state.matchup_label = (
            f'{row["away_team_name"]} @ {row["home_team_name"]}'
        )

        # In home.py
        st.switch_page("pages/game_details.py")
        
home_page()









