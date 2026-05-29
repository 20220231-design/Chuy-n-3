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
# PHAN 3: TRUC QUAN HOA
# ============================================================

def luu_hinh(fig_mpl, ten_file: str):
    """Luu matplotlib figure ra thu muc images/ duoi dang PNG."""
    duong_dan = os.path.join("images", ten_file)
    fig_mpl.savefig(duong_dan, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig_mpl)


def ve_genre_popular(dem_genres: pd.Series) -> go.Figure:
    """Bieu do cot ngang: Top 10 the loai pho bien nhat."""
    fig = px.bar(
        x=dem_genres.values,
        y=dem_genres.index,
        orientation="h",
        color=dem_genres.values,
        color_continuous_scale="Reds",
        labels={"x": "Số lượng phim", "y": "Thể loại"},
        title="Top 10 Thể Loại Phim Phổ Biến Nhất",
        text=dem_genres.values,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      coloraxis_showscale=False, height=450)

    fig_mpl, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(dem_genres)))
    bars = ax.barh(dem_genres.index[::-1], dem_genres.values[::-1], color=colors[::-1])
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Số lượng phim", fontsize=11)
    ax.set_title("Top 10 Thể Loại Phim Phổ Biến Nhất", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    luu_hinh(fig_mpl, "genre_popular.png")

    return fig


def ve_top_doanh_thu(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Bieu do cot: Top 10 phim doanh thu cao nhat."""
    df_top = df.nlargest(top_n, "revenue")[["title", "revenue", "release_year"]].copy()
    df_top["revenue_ty"] = df_top["revenue"] / 1e9

    fig = px.bar(
        df_top,
        x="revenue_ty",
        y="title",
        orientation="h",
        color="revenue_ty",
        color_continuous_scale="OrRd",
        text=df_top["revenue_ty"].apply(lambda x: f"${x:.2f}B"),
        labels={"revenue_ty": "Doanh thu (tỷ USD)", "title": "Phim"},
        title="Top 10 Phim Doanh Thu Cao Nhất",
        hover_data={"release_year": True},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      coloraxis_showscale=False, height=450)

    fig_mpl, ax = plt.subplots(figsize=(11, 6))
    titles_short = [t[:30] + "..." if len(t) > 30 else t for t in df_top["title"]]
    colors = plt.cm.OrRd(np.linspace(0.4, 0.9, top_n))
    bars = ax.barh(titles_short[::-1], df_top["revenue_ty"].values[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="$%.2fB", padding=3, fontsize=9)
    ax.set_xlabel("Doanh thu (tỷ USD)", fontsize=11)
    ax.set_title("Top 10 Phim Doanh Thu Cao Nhất", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    luu_hinh(fig_mpl, "top_revenue.png")

    return fig


def ve_top_rating(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Bieu do cot: Top 10 phim rating cao nhat (vote_count >= 100)."""
    df_filter = df[df["vote_count"] >= 100] if "vote_count" in df.columns else df
    df_top = df_filter.nlargest(top_n, "vote_average")[
        ["title", "vote_average", "release_year"]
    ].copy()

    fig = px.bar(
        df_top,
        x="vote_average",
        y="title",
        orientation="h",
        color="vote_average",
        color_continuous_scale="Greens",
        text=df_top["vote_average"].apply(lambda x: f"{x:.1f}"),
        labels={"vote_average": "Điểm đánh giá", "title": "Phim"},
        title="Top 10 Phim Rating Cao Nhất",
        range_color=[6, 10],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      coloraxis_showscale=False, height=450,
                      xaxis=dict(range=[0, 11]))

    fig_mpl, ax = plt.subplots(figsize=(11, 6))
    titles_short = [t[:30] + "..." if len(t) > 30 else t for t in df_top["title"]]
    colors = plt.cm.Greens(np.linspace(0.5, 0.9, top_n))
    bars = ax.barh(titles_short[::-1], df_top["vote_average"].values[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xlabel("Điểm đánh giá (0-10)", fontsize=11)
    ax.set_title("Top 10 Phim Rating Cao Nhất", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    luu_hinh(fig_mpl, "top_rating.png")

    return fig


def ve_top_popularity(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Bieu do cot: Top 10 phim popularity cao nhat."""
    df_top = df.nlargest(top_n, "popularity")[["title", "popularity", "release_year"]].copy()

    fig = px.bar(
        df_top,
        x="popularity",
        y="title",
        orientation="h",
        color="popularity",
        color_continuous_scale="Blues",
        text=df_top["popularity"].apply(lambda x: f"{x:.1f}"),
        labels={"popularity": "Độ phổ biến", "title": "Phim"},
        title="Top 10 Phim Phổ Biến Nhất (Popularity)",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      coloraxis_showscale=False, height=450)

    fig_mpl, ax = plt.subplots(figsize=(11, 6))
    titles_short = [t[:30] + "..." if len(t) > 30 else t for t in df_top["title"]]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
    bars = ax.barh(titles_short[::-1], df_top["popularity"].values[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xlabel("Độ phổ biến", fontsize=11)
    ax.set_title("Top 10 Phim Phổ Biến Nhất", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    luu_hinh(fig_mpl, "top_popularity.png")

    return fig


def ve_scatter_rating_revenue(df_tc: pd.DataFrame) -> go.Figure:
    """Scatter plot: Moi quan he giua vote_average va revenue."""
    fig = px.scatter(
        df_tc,
        x="vote_average",
        y="revenue",
        color="budget",
        size="popularity",
        hover_name="title",
        hover_data={"release_year": True, "genres_str": True},
        color_continuous_scale="Viridis",
        labels={
            "vote_average": "Điểm đánh giá (Rating)",
            "revenue": "Doanh thu (USD)",
            "budget": "Ngân sách (USD)",
        },
        title="Mối Quan Hệ Giữa Rating và Doanh Thu",
        log_y=True,
        opacity=0.75,
    )
    fig.update_layout(height=500)

    # Duong xu huong bang numpy polyfit
    x_vals = df_tc["vote_average"].values
    y_vals = np.log10(df_tc["revenue"].values + 1)
    if len(x_vals) > 2:
        coef = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = 10 ** np.polyval(coef, x_line)
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Xu hướng",
        ))

    fig_mpl, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        df_tc["vote_average"], df_tc["revenue"] / 1e6,
        c=df_tc["budget"] / 1e6, cmap="viridis",
        alpha=0.6, s=20, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Ngân sách (triệu USD)")
    ax.set_yscale("log")
    ax.set_xlabel("Điểm đánh giá (Rating)", fontsize=11)
    ax.set_ylabel("Doanh thu (triệu USD, thang log)", fontsize=11)
    ax.set_title("Mối Quan Hệ Giữa Rating và Doanh Thu", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    luu_hinh(fig_mpl, "rating_vs_revenue.png")

    return fig


def ve_phan_phoi_rating(df: pd.DataFrame) -> go.Figure:
    """Histogram phan phoi diem danh gia (vote_average)."""
    fig = px.histogram(
        df,
        x="vote_average",
        nbins=40,
        color_discrete_sequence=["#E50914"],
        labels={"vote_average": "Điểm đánh giá", "count": "Số phim"},
        title="Phân Phối Điểm Đánh Giá (Rating)",
    )
    fig.add_vline(x=df["vote_average"].mean(), line_dash="dash",
                  line_color="gold",
                  annotation_text=f"TB: {df['vote_average'].mean():.2f}")
    fig.update_layout(height=400)
    return fig


def ve_revenue_theo_nam(df_tc: pd.DataFrame) -> go.Figure:
    """Bieu do duong: Tong doanh thu va so phim theo nam."""
    theo_nam = df_tc.groupby("release_year").agg(
        tong_revenue=("revenue", "sum"),
        so_phim=("title", "count"),
    ).reset_index()
    theo_nam["revenue_ty"] = theo_nam["tong_revenue"] / 1e9

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=theo_nam["release_year"], y=theo_nam["revenue_ty"],
        name="Tổng doanh thu (tỷ USD)", marker_color="#E50914", opacity=0.75,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=theo_nam["release_year"], y=theo_nam["so_phim"],
        name="Số phim", mode="lines+markers",
        line=dict(color="gold", width=2),
    ), secondary_y=True)
    fig.update_layout(
        title="Doanh Thu và Số Phim Theo Năm",
        height=420,
        xaxis_title="Năm",
        legend=dict(x=0.01, y=0.99),
    )
    fig.update_yaxes(title_text="Tổng doanh thu (tỷ USD)", secondary_y=False)
    fig.update_yaxes(title_text="Số phim", secondary_y=True)
    return fig


def ve_heatmap_tuong_quan(df_tc: pd.DataFrame) -> go.Figure:
    """Heatmap ma tran tuong quan giua cac bien so."""
    cols = ["budget", "revenue", "vote_average", "runtime", "popularity"]
    cols_co = [c for c in cols if c in df_tc.columns]
    corr = df_tc[cols_co].corr()

    labels_vn = {
        "budget": "Ngân sách",
        "revenue": "Doanh thu",
        "vote_average": "Rating",
        "runtime": "Thời lượng",
        "popularity": "Độ phổ biến",
    }
    idx = [labels_vn.get(c, c) for c in corr.index]

    fig = px.imshow(
        corr.values,
        x=idx, y=idx,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        title="Ma Trận Tương Quan Giữa Các Biến",
        aspect="auto",
    )
    fig.update_layout(height=450)

    fig_mpl, ax = plt.subplots(figsize=(8, 6))
    corr_plot = df_tc[cols_co].corr()
    corr_plot.index = idx
    corr_plot.columns = idx
    sns.heatmap(corr_plot, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 10})
    ax.set_title("Ma Trận Tương Quan", fontsize=13, fontweight="bold")
    plt.tight_layout()
    luu_hinh(fig_mpl, "correlation_heatmap.png")

    return fig


def ve_pie_genres(dem_genres: pd.Series) -> go.Figure:
    """Biểu đồ tròn tỷ lệ thể loại phim (Top 10)."""
    fig = px.pie(
        values=dem_genres.values,
        names=dem_genres.index,
        title="Tỷ Lệ Thể Loại Phim (Top 10)",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=450)
    return fig


def ve_xu_the_the_loai_theo_nam(df: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """Biểu đồ đường xu thế số lượng phim theo thể loại qua các năm."""
    top_genres = phan_tich_the_loai(df, top_n).index.tolist()

    rows = []
    for _, row in df.iterrows():
        for g in row["genres_list"]:
            if g in top_genres:
                rows.append({"release_year": row["release_year"], "genre": g})

    if not rows:
        return go.Figure()

    df_long = pd.DataFrame(rows)
    df_group = (
        df_long.groupby(["release_year", "genre"])
        .size()
        .reset_index(name="so_phim")
    )
    df_group = df_group[df_group["release_year"] >= 1990]

    fig = px.line(
        df_group,
        x="release_year",
        y="so_phim",
        color="genre",
        markers=True,
        labels={"release_year": "Năm", "so_phim": "Số lượng phim", "genre": "Thể loại"},
        title=f"Xu Thế Số Lượng Phim Theo Thể Loại Qua Các Năm (Top {top_n})",
    )
    fig.update_layout(height=450, legend_title_text="Thể loại")
    return fig


def ve_doanh_thu_trung_binh_theo_the_loai(df_tc: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Biểu đồ cột + đường doanh thu trung bình theo thể loại."""
    rows = []
    for _, row in df_tc.iterrows():
        for g in row["genres_list"]:
            if g:
                rows.append({"genre": g, "revenue": row["revenue"]})

    if not rows:
        return go.Figure()

    df_long = pd.DataFrame(rows)
    df_group = (
        df_long.groupby("genre")["revenue"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "dt_tb", "sum": "dt_tong", "count": "so_phim"})
        .nlargest(top_n, "dt_tb")
    )
    df_group["dt_tb_ty"] = df_group["dt_tb"] / 1e6

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df_group["genre"],
        y=df_group["dt_tb_ty"],
        name="Doanh thu TB (triệu USD)",
        marker_color="#E50914",
        opacity=0.8,
        text=df_group["dt_tb_ty"].apply(lambda x: f"${x:.0f}M"),
        textposition="outside",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df_group["genre"],
        y=df_group["so_phim"],
        name="Số phim",
        mode="lines+markers",
        line=dict(color="gold", width=2),
        marker=dict(size=8),
    ), secondary_y=True)
    fig.update_layout(
        title="Doanh Thu Trung Bình Theo Thể Loại Phim",
        height=450,
        xaxis_title="Thể loại",
        legend=dict(x=0.01, y=0.99),
    )
    fig.update_yaxes(title_text="Doanh thu TB (triệu USD)", secondary_y=False)
    fig.update_yaxes(title_text="Số phim", secondary_y=True)
    return fig


def ve_xu_the_doanh_thu_du_bao(df_tc: pd.DataFrame, so_nam_du_bao: int = 5) -> go.Figure:
    """Biểu đồ đường xu thế doanh thu theo năm kèm dự báo tương lai."""
    theo_nam = (
        df_tc.groupby("release_year")["revenue"]
        .sum()
        .reset_index()
    )
    theo_nam["revenue_ty"] = theo_nam["revenue"] / 1e9
    theo_nam = theo_nam[theo_nam["release_year"] >= 1990]

    # Hồi quy tuyến tính để dự báo
    X_fit = theo_nam["release_year"].values.reshape(-1, 1)
    y_fit = theo_nam["revenue_ty"].values
    model = LinearRegression()
    model.fit(X_fit, y_fit)

    nam_max = int(theo_nam["release_year"].max())
    nam_du_bao = np.arange(nam_max + 1, nam_max + so_nam_du_bao + 1)
    y_du_bao = model.predict(nam_du_bao.reshape(-1, 1))

    # Đường xu thế toàn bộ
    nam_all = np.arange(int(theo_nam["release_year"].min()), nam_max + 1)
    y_trend = model.predict(nam_all.reshape(-1, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theo_nam["release_year"],
        y=theo_nam["revenue_ty"],
        mode="lines+markers",
        name="Doanh thu thực tế",
        line=dict(color="#E50914", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=nam_all,
        y=y_trend,
        mode="lines",
        name="Đường xu thế (Linear)",
        line=dict(color="orange", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=nam_du_bao,
        y=y_du_bao,
        mode="lines+markers",
        name=f"Dự báo {so_nam_du_bao} năm tới",
        line=dict(color="royalblue", width=2, dash="dot"),
        marker=dict(size=8, symbol="diamond"),
    ))
    fig.update_layout(
        title="Xu Thế Doanh Thu Theo Năm và Dự Báo Tương Lai",
        xaxis_title="Năm",
        yaxis_title="Tổng doanh thu (tỷ USD)",
        height=450,
        legend=dict(x=0.01, y=0.99),
    )
    return fig



def ve_xu_the_the_loai_doanh_thu_nam(df_tc: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """Biểu đồ đường doanh thu trung bình theo thể loại qua các năm."""
    top_genres = phan_tich_the_loai(df_tc, top_n).index.tolist()

    rows = []
    for _, row in df_tc.iterrows():
        for g in row["genres_list"]:
            if g in top_genres:
                rows.append({
                    "release_year": row["release_year"],
                    "genre": g,
                    "revenue": row["revenue"],
                })

    if not rows:
        return go.Figure()

    df_long = pd.DataFrame(rows)
    df_group = (
        df_long.groupby(["release_year", "genre"])["revenue"]
        .mean()
        .reset_index()
    )
    df_group = df_group[df_group["release_year"] >= 1990]
    df_group["revenue_trieu"] = df_group["revenue"] / 1e6

    fig = px.line(
        df_group,
        x="release_year",
        y="revenue_trieu",
        color="genre",
        markers=True,
        labels={
            "release_year": "Năm",
            "revenue_trieu": "Doanh thu TB (triệu USD)",
            "genre": "Thể loại",
        },
        title=f"Xu Thế Doanh Thu Trung Bình Theo Thể Loại Qua Các Năm (Top {top_n})",
    )
    fig.update_layout(height=450, legend_title_text="Thể loại")
    return fig


# ============================================================
# PHAN 4: MO HINH DU DOAN
# ============================================================

def chuan_bi_features(df_tc: pd.DataFrame, target: str) -> tuple:
    """Chuan bi feature matrix X va target vector y cho mo hinh."""
    top_g = phan_tich_the_loai(df_tc, 10).index.tolist()

    for g in top_g:
        col_name = f"genre_{g.replace(' ', '_')}"
        df_tc[col_name] = df_tc["genres_list"].apply(lambda lst: 1 if g in lst else 0)

    genre_cols = [f"genre_{g.replace(' ', '_')}" for g in top_g]
    base_features = ["budget", "runtime", "popularity", "release_year"]
    if "vote_count" in df_tc.columns:
        base_features.append("vote_count")

    if target == "revenue":
        feature_cols = base_features + ["vote_average"] + genre_cols
    else:
        feature_cols = base_features + ["revenue"] + genre_cols

    feature_cols = [c for c in feature_cols if c in df_tc.columns]
    X = df_tc[feature_cols].fillna(df_tc[feature_cols].median(numeric_only=True))
    y = df_tc[target]

    return X, y, feature_cols


def huan_luyen_mo_hinh(X, y):
    """
    Huan luyen LinearRegression va RandomForestRegressor.
    Tra ve dict ket qua danh gia (MAE, RMSE, R2) va model da train.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ket_qua = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    ket_qua["LinearRegression"] = {
        "MAE": mean_absolute_error(y_test, y_pred_lr),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        "R2": r2_score(y_test, y_pred_lr),
        "y_test": y_test.values,
        "y_pred": y_pred_lr,
        "model": lr,
    }

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    ket_qua["RandomForest"] = {
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "R2": r2_score(y_test, y_pred_rf),
        "y_test": y_test.values,
        "y_pred": y_pred_rf,
        "model": rf,
        "feature_importance": pd.Series(
            rf.feature_importances_, index=X.columns
        ).sort_values(ascending=False),
    }

    return ket_qua


def ve_ket_qua_mo_hinh(ket_qua: dict, target_label: str, ten_file_prefix: str) -> go.Figure:
    """Ve bieu do Actual vs Predicted cho ca 2 mo hinh."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Linear Regression", "Random Forest"),
    )
    colors = {"LinearRegression": "#636EFA", "RandomForest": "#EF553B"}

    for i, (ten, res) in enumerate(ket_qua.items(), 1):
        y_test, y_pred = res["y_test"], res["y_pred"]
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred,
            mode="markers",
            marker=dict(color=colors[ten], size=5, opacity=0.6),
            name=ten,
        ), row=1, col=i)
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines",
            line=dict(color="black", dash="dash", width=1.5),
            showlegend=False,
        ), row=1, col=i)

    fig.update_layout(
        title=f"Thực tế vs Dự đoán - {target_label}",
        height=420,
        xaxis_title="Thực tế", yaxis_title="Dự đoán",
        xaxis2_title="Thực tế", yaxis2_title="Dự đoán",
    )

    fig_mpl, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (ten, res) in zip(axes, ket_qua.items()):
        ax.scatter(res["y_test"], res["y_pred"], alpha=0.5, s=15,
                   color=colors[ten], edgecolors="none")
        mn = min(res["y_test"].min(), res["y_pred"].min())
        mx = max(res["y_test"].max(), res["y_pred"].max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.5)
        ax.set_xlabel("Thực tế", fontsize=10)
        ax.set_ylabel("Dự đoán", fontsize=10)
        ax.set_title(f"{ten}\nR2={res['R2']:.3f}", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle(f"Thực tế vs Dự đoán - {target_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    luu_hinh(fig_mpl, f"prediction_{ten_file_prefix}.png")

    return fig


def ve_feature_importance(ket_qua_rf: dict, top_n: int = 10) -> go.Figure:
    """Bieu do feature importance cua RandomForest."""
    fi = ket_qua_rf["feature_importance"].head(top_n)
    fig = px.bar(
        x=fi.values,
        y=fi.index,
        orientation="h",
        color=fi.values,
        color_continuous_scale="Oranges",
        labels={"x": "Mức độ quan trọng", "y": "Đặc trưng"},
        title="Độ Quan Trọng Của Đặc Trưng (Random Forest)",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      coloraxis_showscale=False, height=380)
    return fig
# ============================================================
# PHAN 5: LUU KET QUA
# ============================================================

def luu_ket_qua(df, df_tc, ket_qua_rev, ket_qua_rat, dem_genres):
    """Luu cac bang thong ke va ket qua mo hinh ra thu muc results/."""
    stats = df[["budget", "revenue", "vote_average", "runtime", "popularity"]].describe()
    stats.to_csv("results/thong_ke_tong_quat.csv")

    df[["title", "release_year", "revenue", "vote_average", "popularity",
        "genres_str"]].nlargest(50, "revenue").to_csv("results/top50_doanh_thu.csv", index=False)

    df[["title", "release_year", "revenue", "vote_average", "popularity",
        "genres_str"]].nlargest(50, "vote_average").to_csv("results/top50_rating.csv", index=False)

    dem_genres.reset_index().rename(
        columns={"genres_list": "the_loai", "count": "so_phim"}
    ).to_csv("results/genre_popular.csv", index=False)

    rows = []
    for ten, res in ket_qua_rev.items():
        rows.append({"Mo_hinh": ten, "Target": "Revenue",
                     "MAE": res["MAE"], "RMSE": res["RMSE"], "R2": res["R2"]})
    for ten, res in ket_qua_rat.items():
        rows.append({"Mo_hinh": ten, "Target": "Vote_Average",
                     "MAE": res["MAE"], "RMSE": res["RMSE"], "R2": res["R2"]})
    df_models = pd.DataFrame(rows)
    df_models.to_csv("results/ket_qua_mo_hinh.csv", index=False)

    with open("results/bao_cao_tom_tat.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("BAO CAO PHAN TICH DU LIEU PHIM - TMDB 5000\n")
        f.write(f"Ngay tao: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Tong so phim (sau lam sach): {len(df)}\n")
        f.write(f"Phim co du lieu tai chinh: {len(df_tc)}\n")
        f.write(f"Khoang nam: {df['release_year'].min()} - {df['release_year'].max()}\n\n")
        f.write("--- THONG KE CHINH ---\n")
        f.write(stats.to_string() + "\n\n")
        f.write("--- TOP 10 THE LOAI PHO BIEN ---\n")
        for g, cnt in dem_genres.items():
            f.write(f"  {g}: {cnt} phim\n")
        f.write("\n--- KET QUA MO HINH DU DOAN ---\n")
        f.write(df_models.to_string(index=False) + "\n")

    return df_models


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
