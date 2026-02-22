import streamlit as st
from util import read_cluster_file, merged_team_clusters,basic_stats,prob_to_american,prob_over_line
from data_util import filter_player_vs_cluster, filter_player_logs,general_player_logs

def player_details_page():
        

    if "player_team_id" not in st.session_state:
            st.warning("Select a player first.")
            return
    

    player_team_id = st.session_state.player_team_id
    home_team_id = st.session_state.home_team_id
    away_team_id = st.session_state.away_team_id
    player_id = st.session_state.player_id
    player_name = st.session_state.player_name
    opp_team_id = away_team_id if player_team_id == home_team_id else home_team_id
    keep_player_columns = ['PLAYER_NAME','MIN','PTS','REB', 'AST','TOV', 'STL', 'BLK', 'PF','FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'PLUS_MINUS','WL','BLKA', 'PFD','TEAM_NAME','GAME_DATE', 'OPP_TEAM_NAME']



            

    cluster_df, cluster_map = read_cluster_file() ##cluster file i upload
    player_opp_cluster = int(
        cluster_df.loc[
            cluster_df.TEAM_ID == opp_team_id,
            "cluster"
        ].iloc[0]
    )

    all_player_logs = general_player_logs()



    opponent_team_ids = cluster_map[player_opp_cluster]

    player_df_1 = filter_player_logs(all_player_logs,player_id=player_id)

    player_vs_cluster_df_1 = filter_player_vs_cluster(player_df_1,opponent_team_ids)

    # if len(player_vs_cluster_df) < 3:
    #     st.caption("⚠️ Note: Small sample size for this cluster.")
  

    clusters = merged_team_clusters()

    player_opp_cluster = clusters.loc[clusters['cluster']== player_opp_cluster]

    
    player_df = player_df_1[keep_player_columns]

    



   
    
    
    st.write("Player Name", player_name, 'game logs')


    st.dataframe(player_df)


    st.write("player vs cluster:", player_opp_cluster)
    # pyg.walk(player_vs_cluster_df, env="streamlit")

    player_vs_cluster_df = player_vs_cluster_df_1[keep_player_columns]
    st.dataframe(player_vs_cluster_df)


    numeric_cols = player_vs_cluster_df.select_dtypes(include="number").columns.tolist()
    col = st.selectbox("Stat column", numeric_cols, index=numeric_cols.index("PTS") if "PTS" in numeric_cols else 0)
    line = st.number_input("Line (e.g. 5.5)", value=5.5, step=0.5)
    cluster_game_ids = set(player_vs_cluster_df_1["GAME_ID"].astype(str))
    season_excluding_cluster = player_df_1[~player_df_1["GAME_ID"].astype(str).isin(cluster_game_ids)]


    y_ex = season_excluding_cluster[col]

    if len(player_vs_cluster_df) < 1:
        st.warning("not enough data for cluster")
        return
    
    
    s = player_vs_cluster_df[col]
    cluster_stats = basic_stats(s)


    
    gen_stats_ex = basic_stats(y_ex)

    # Calculate the difference in means
    mean_delta = cluster_stats['mean'] - gen_stats_ex['mean']
    delta_pct = (mean_delta / gen_stats_ex['mean']) if gen_stats_ex['mean'] != 0 else 0

    # Calculate the hit rate difference
   


  
    
    st.write(gen_stats_ex, 'gen stats')





    st.json(cluster_stats)


    p = prob_over_line(s, line)
    odds = prob_to_american(p)
    gen_p = prob_over_line(y_ex, line)
    hit_rate_delta = p - gen_p

    if odds is None:
        st.warning("Not enough data / invalid probability.")
    else:
        st.write(f"**P({col} ≥ {line}) = {p:.1%}**")
        st.write(f"**Fair American odds:** {odds:+d}")

    
    st.divider()
    st.subheader(f"Value Analysis: {col} @ {line}")

    m1, m2, m3,m4 = st.columns(4)

    with m1:
    # Display the raw average vs cluster
        st.metric(
            label=f"Avg {col} vs Cluster", 
            value=f"{cluster_stats['mean']:.1f}", 
            delta=f"{mean_delta:+.1f} ({delta_pct:+.1%})" # Shows +5.0 (+15.2%)
        )

    with m2:
        # Use delta_pct to explain the "Value"
        if delta_pct > 0.1:
            st.write("### ✅ Value Found")
            st.write(f"Player performs **{delta_pct:.1%} better** against this cluster than their season average.")
        elif delta_pct < -0.1:
            st.write("### ❌ Bad Matchup")
            st.write(f"Player performs **{abs(delta_pct):.1%} worse** against this cluster.")
        else:
            st.write("### ⚖️ Neutral")

    with m3:
        # Show the probability of hitting the line
        st.metric(label="Hit Rate", value=f"{p:.1%}")

    with m4: 
        st.metric(label="Hit rate delta ", value= f"{hit_rate_delta:.1}")










player_details_page()



### issues arise when player data is small for a cluster
### need to calculate season odds vs cluster odds and compute 
### a value there