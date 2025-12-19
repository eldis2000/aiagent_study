import sqlite3
import pandas as pd
import streamlit as st
import folium
import requests
from pathlib import Path
import streamlit.components.v1 as components

# =================================================
# 기본 설정
# =================================================
st.set_page_config(page_title="서울 상권 · 지하철 · 인구 · CCTV", layout="wide")
st.title("🗺️ 서울 상권 · 지하철 · 실시간 인구 밀집 지도 (DB 기반)")

#DB_PATH = "ap_agent.db"
#DB_PATH = Path(r"C:\database\sqlite\ai_agent")
DB_PATH = "ai_agent"

SEOUL_API_KEY = st.secrets["SEOUL_API_KEY"]
SEOUL_API_BASE = "http://openapi.seoul.go.kr:8088"

# =================================================
# 색상
# =================================================
SUBWAY_COLORS = {1: "#0052A4", 2: "#00A84D"}

CONGEST_COLOR = {
    "여유": "#22c55e",
    "보통": "#3b82f6",
    "약간 붐빔": "#f59e0b",
    "붐빔": "#ef4444",
}

# =================================================
# 실시간 인구 기준 지역
# =================================================
AREA_COORDS = {
    "강남역": (37.498095, 127.027610),
    "홍대입구역": (37.556748, 126.923643),
    "명동": (37.563757, 126.985302),
    "잠실역": (37.513262, 127.100159),
    "광화문": (37.571622, 126.976815),
}

# =================================================
# DB 유틸
# =================================================
@st.cache_data(show_spinner=False)
def query_df(sql, params=None):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(sql, conn, params=params)

# =================================================
# 실시간 인구 API
# =================================================
@st.cache_data(ttl=300)
def fetch_live_population(area_nm):
    url = f"{SEOUL_API_BASE}/{SEOUL_API_KEY}/json/citydata_ppltn/1/1/{area_nm}"
    try:
        p = requests.get(url, timeout=5).json()["SeoulRtd.citydata_ppltn"][0]
        mn, mx = int(p["AREA_PPLTN_MIN"]), int(p["AREA_PPLTN_MAX"])
        return {
            "area": p["AREA_NM"],
            "min": mn,
            "max": mx,
            "count": (mn + mx) // 2,
            "level": p["AREA_CONGEST_LVL"],
            "time": p["PPLTN_TIME"],
        }
    except:
        return None

# =================================================
# 사이드바
# =================================================
st.sidebar.header("🔍 상권 검색")
keyword = st.sidebar.text_input("상호명 검색")

selected_areas = st.sidebar.multiselect(
    "🧍 실시간 인구 밀집 지역",
    list(AREA_COORDS.keys()),
    default=["강남역", "홍대입구역", "명동", "잠실역", "광화문"],
)

submit = st.sidebar.button("검색")

# =================================================
# 지도 생성
# =================================================
if submit and keyword.strip():

    # ==========================
    # 🏪 상권 검색
    # ==========================
    commerce_df = query_df(
        """
        SELECT id, name, branch_name, industry_name, lat, lng
        FROM commerce
        WHERE name LIKE ?
        LIMIT 300
        """,
        (f"%{keyword}%",)
    )

    if commerce_df.empty:
        st.warning("검색 결과가 없습니다.")
        st.stop()

    center_lat = commerce_df["lat"].mean()
    center_lng = commerce_df["lng"].mean()

    # ==========================
    # 지도 생성 (타일 즉시)
    # tiles="OpenStreetMap"      # 기본
    # tiles="CartoDB dark_matter" # 다크모드
    # tiles="Stamen Toner"       # 흑백
    # ==========================
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=False
    )

    # ==========================
    # 🚇 지하철 (1·2호선)
    # ==========================
    subway_fg = folium.FeatureGroup(name="🚇 지하철 (1·2호선)", show=False)

    subway_df = query_df(
        """
        SELECT line, station_name, lat, lng, seq
        FROM subway_station
        WHERE line IN (1,2)
        ORDER BY line, seq
        """
    )

    for line in [1, 2]:
        line_df = subway_df[subway_df["line"] == line]
        coords = line_df[["lat", "lng"]].values.tolist()
        if len(coords) > 1:
            folium.PolyLine(
                coords,
                color=SUBWAY_COLORS[line],
                weight=4,
                opacity=0.8
            ).add_to(subway_fg)

    subway_fg.add_to(m)

    # ==========================
    # 🧍 실시간 인구 밀집
    # ==========================
    pop_fg = folium.FeatureGroup(name="🧍 실시간 인구 밀집", show=False)

    for area in selected_areas:
        info = fetch_live_population(area)
        if info:
            lat, lng = AREA_COORDS[area]
            color = CONGEST_COLOR.get(info["level"], "#999")

            folium.Circle(
                [lat, lng],
                radius=max(600, info["count"] / 2),
                color=color,
                fill=True,
                fill_opacity=0.3,
                tooltip=f"{area} · {info['level']} · {info['min']:,}~{info['max']:,}명"
            ).add_to(pop_fg)

    pop_fg.add_to(m)

    # ==========================
    # 📹 CCTV (아이콘)
    # ==========================
    cctv_fg = folium.FeatureGroup(name="📹 CCTV", show=False)

    cctv_df = query_df(
        """
        SELECT id, location, lat, lng
        FROM trans_cctv_location
        LIMIT 800
        """
    )

    for _, r in cctv_df.iterrows():
        folium.Marker(
            [r["lat"], r["lng"]],
            icon=folium.Icon(
                icon="video-camera",
                prefix="fa",
                color="purple"
            ),
            tooltip=r["location"]
        ).add_to(cctv_fg)

    cctv_fg.add_to(m)

    # ==========================
    # 🏪 상권 (아이콘)
    # ==========================
    store_fg = folium.FeatureGroup(name="🏪 상권", show=True)

    for _, r in commerce_df.iterrows():
        name = r["name"]
        if r["branch_name"]:
            name += f" ({r['branch_name']})"

        folium.Marker(
            [r["lat"], r["lng"]],
            icon=folium.Icon(
                icon="shopping-cart",
                prefix="fa",
                color="red"
            ),
            tooltip=name
        ).add_to(store_fg)

    store_fg.add_to(m)

    # ==========================
    # 레이어 컨트롤
    # ==========================
    folium.LayerControl(collapsed=False).add_to(m)

    components.html(m._repr_html_(), height=780, scrolling=False)

else:
    st.info("👈 좌측에서 상호명을 입력하고 검색하세요.")
