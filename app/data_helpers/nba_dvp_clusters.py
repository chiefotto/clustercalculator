from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
# parent.parent because: app/pages/home.py → app/

DATA_CACHE = BASE_DIR / "data_cache"



def read_pqt(path):
    file_path = pd.read_parquet(path)
    return file_path


def read_cluster_file():
    # cluster_file = pd.read_parquet('../data/jan21clusters.parquet')
    path = (DATA_CACHE/'jan21clusters.parquet')
    cluster_file= read_pqt(path)

    cluster_df = cluster_file[['TEAM_ID', 'cluster']]

    cluster_map = cluster_df.groupby("cluster")["TEAM_ID"].apply(list).to_dict()

    return cluster_df, cluster_map



def clusters_frontend():

    cluster_df, cluster_map = read_cluster_file()


    st.dataframe(cluster_df)
    st.write(cluster_map)
