### this is a utility file 
### should read in csv of 
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
# parent.parent because: app/pages/home.py → app/

DATA_CACHE = BASE_DIR / "data_cache"


def read_pqt(path):
    file_path = pd.read_parquet(path)
    return file_path


@st.cache_data
def read_cluster_file():
    # cluster_file = pd.read_parquet('../data/jan21clusters.parquet')
    path = (DATA_CACHE/'jan21clusters.parquet')
    cluster_file= read_pqt(path)

    cluster_df = cluster_file[['TEAM_ID', 'cluster']]

    cluster_map = cluster_df.groupby("cluster")["TEAM_ID"].apply(list).to_dict()

    return cluster_df, cluster_map

@st.cache_data
def merged_team_clusters():
    cluster_df, _ = read_cluster_file()
    nba_teams = read_pqt("data_cache/nba_teams.parquet")

    merged = cluster_df.merge(
        nba_teams[["id", "full_name", "abbreviation"]],
        left_on="TEAM_ID",
        right_on="id",
        how="left"
    )

    # optional: clean column names
    merged = merged.drop(columns=["id"])
    merged = merged.rename(columns={
        "full_name": "TEAM_NAME",
        "abbreviation": "TEAM_ABBR"
    })

    return merged




def basic_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}

    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }

def prob_over_line(s: pd.Series, line: float) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float((s >= line).mean())

def prob_to_american(p: float) -> int | None:
    if not (0 < p < 1):
        return None
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    else:
        return int(round(100 * (1 - p) / p))








