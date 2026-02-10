import pandas as pd
import streamlit as st

from data_util import general_player_logs


df = general_player_logs()

st.dataframe(df, use_container_width=True)
