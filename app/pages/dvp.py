import pandas as pd
import streamlit as st


from data_helpers.nba_dvp import read_dvp_file, split_columns_into_value_and_rank, merge_dvp_with_teams

df = read_dvp_file()

df_1 = split_columns_into_value_and_rank(df)

merged = merge_dvp_with_teams(df_1)


st.dataframe(df_1, use_container_width=True)     

st.write("DVP with teams")

st.dataframe(merged, use_container_width=True)