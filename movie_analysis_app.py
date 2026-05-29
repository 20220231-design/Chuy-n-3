# =============================================================================
# PHAN TICH DU LIEU PHIM: RATING, DOANH THU, THE LOAI
# Dataset: TMDB 5000 Movies
# =============================================================================

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "venv_packages"))

for folder in ["images", "results", "data"]:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# IMPORT THU VIEN
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import json
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import streamlit as st

# ============================================================
# CAU HINH TRANG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Phân Tích Dữ Liệu Phim",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E50914;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #E50914;
        border-bottom: 2px solid #E50914;
        padding-bottom: 6px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PHAN 1: TAI VA LAM SACH DU LIEU
# ============================================================

@st.cache_data(show_spinner=False)
def tai_du_lieu(duong_dan: str) -> pd.DataFrame:
    """Tai file CSV va tra ve DataFrame tho."""
    return pd.read_csv(duong_dan)


def phan_tich_genres(genres_str: str) -> list:
    """Chuyen chuoi JSON cua genres thanh danh sach ten the loai."""
    try:
        data = json.loads(genres_str)
        return [item["name"] for item in data]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


def lam_sach_du_lieu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lam sach du lieu phim:
    - Bo ban ghi thieu release_date, runtime
    - Parse genres tu JSON sang list
    - Trich xuat nam tu release_date
    - Loc vote_average hop le (> 0)
    """
    df = df.copy()

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    df["release_year"] = df["release_date"].dt.year.astype(int)

    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df = df.dropna(subset=["runtime"])
    df = df[df["runtime"] > 0]

    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df = df.dropna(subset=["vote_average"])
    df = df[df["vote_average"] > 0]

    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)

    df["genres_list"] = df["genres"].apply(phan_tich_genres)
    df["genres_str"] = df["genres_list"].apply(
        lambda x: ", ".join(x) if x else "Không xác định"
    )

    return df.reset_index(drop=True)


def tao_df_tai_chinh(df: pd.DataFrame) -> pd.DataFrame:
    """Lay subset phim co budget va revenue > 0, tinh them ROI."""
    df_tc = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    df_tc["roi"] = (df_tc["revenue"] - df_tc["budget"]) / df_tc["budget"]
    return df_tc


# ============================================================
# PHAN 2: PHAN TICH THE LOAI
# ============================================================

def phan_tich_the_loai(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Dem so phim theo tung the loai, tra ve top_n pho bien nhat."""
    tat_ca = df["genres_list"].explode()
    tat_ca = tat_ca[tat_ca.notna() & (tat_ca != "")]
    return tat_ca.value_counts().head(top_n)





# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
