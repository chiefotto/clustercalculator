import streamlit as st

home = st.Page("pages/home.py", title="Home", default=True)
details = st.Page("pages/game_details.py", title="Game Details")
player_view = st.Page("pages/player_view.py", title="Player View")
settings = st.Page("pages/settings.py", title="Settings")
game_logs = st.Page("pages/game_logs.py", title="Game Logs")
dvp = st.Page("pages/dvp.py", title="DVP")
clustersAndDvp = st.Page("pages/clusters.py", title="Clusters")
cluster_viz = st.Page("pages/defense_cluster_viz.py", title="Defense Clusters — Viz")
compare_clusters = st.Page("pages/compare_clusters.py", title="Compare Clusters")
similar_teams = st.Page("pages/similar_teams.py", title="Similar Teams")
rolling_analyzer = st.Page("pages/rolling_cluster_analyzer.py", title="Rolling Cluster Analyzer")

pg = st.navigation([
    home, details, player_view, settings,
    game_logs, dvp, clustersAndDvp, cluster_viz, compare_clusters, similar_teams, rolling_analyzer,
])
pg.run()
