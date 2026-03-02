import streamlit as st

home = st.Page("pages/home.py", title="Home", default=True)
details = st.Page("pages/game_details.py", title="Game Details")
player_view = st.Page("pages/player_view.py", title="Player View")
settings = st.Page("pages/settings.py", title="Settings")
game_logs = st.Page("pages/game_logs.py", title="Game Logs")
dvp = st.Page("pages/dvp.py", title="DVP")
clustersAndDvp = st.Page("pages/clusters.py", title="Clusters")
cluster_viz = st.Page("pages/defense_cluster_viz.py", title="Defense Clusters — Viz")

pg = st.navigation([
    home, details, player_view, settings,
    game_logs, dvp, clustersAndDvp, cluster_viz,
])
pg.run()
