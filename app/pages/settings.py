import streamlit as st
from util import merged_team_clusters
from player_logs import load_league_logs, upsert_league_logs

logs = load_league_logs()
cluster_df = merged_team_clusters()

st.write("Rows on disk:", len(logs))

if st.button("Refresh league logs"):
    stats = upsert_league_logs()
    st.success(f"Added {stats['added']} new rows. Total now {stats['total']}.")
    st.rerun()  # rerun so UI uses fresh cached data


st.dataframe(cluster_df)