import requests
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import html

BASE_URL = "https://www.joongang.co.kr"

CATEGORIES = {
    "사회": "/society",
    "정치": "/politics",
    "경제": "/money",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# ---------------------------
# 뉴스 수집 함수
# ---------------------------
@st.cache_data(ttl=60 * 10)
def fetch_news(category_path, days):
    url = BASE_URL + category_path
    res = requests.get(url, headers=HEADERS, timeout=10)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "lxml")
    story_list = soup.select_one("#story_list")

    if not story_list:
        return []

    cutoff = datetime.now() - timedelta(days=days)
    articles = []

    for card in story_list.select("li.card"):
        headline_tag = card.select_one("h2.headline a")
        date_tag = card.select_one("p.date")
        desc_tag = card.select_one("p.description")
        print(desc_tag)

        if not headline_tag or not date_tag:
            continue

        # 날짜 변환
        try:
            published_at = datetime.strptime(
                date_tag.get_text(strip=True),
                "%Y.%m.%d %H:%M"
            )
        except ValueError:
            continue

        # 최근 N일 필터
        if published_at < cutoff:
            continue

        articles.append({
            "title": headline_tag.get_text(strip=True),
            "link": headline_tag["href"],
            "date": published_at.strftime("%Y-%m-%d %H:%M"),
            "summary": desc_tag.get_text(strip=True) if desc_tag else ""
        })

    return articles


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="중앙일보 뉴스 대시보드",
    layout="wide"
)

st.title("🗞️ 중앙일보 뉴스 대시보드")

# ===== Sidebar =====
st.sidebar.header("🛠️ 뉴스 설정")

selected_categories = st.sidebar.multiselect(
    "카테고리 선택",
    options=list(CATEGORIES.keys()),
    default=["사회", "정치", "경제"]
)

days = st.sidebar.slider(
    "수집 기간 (최근 N일)",
    min_value=1,
    max_value=14,
    value=10
)

# ===================

if not selected_categories:
    st.warning("사이드바에서 최소 1개 이상의 카테고리를 선택하세요.")
    st.stop()

# ---------------------------
# 뉴스 출력
# ---------------------------
for category in selected_categories:
    st.subheader(f"📌 {category}")

    with st.spinner(f"{category} 뉴스 수집 중..."):
        articles = fetch_news(CATEGORIES[category], days)

    if not articles:
        st.info("해당 기간의 기사가 없습니다.")
        continue

    for art in articles:
        title = html.escape(art["title"])
        summary = art["summary"]
        link = html.escape(art["link"])
        st.markdown(
            f"""
            <div style="
                border:1px solid #e5e7eb;
                border-radius:14px;
                padding:18px;
                margin-bottom:16px;
                background-color:#ffffff;
            ">
                <h3 style="margin:0 0 6px 0; line-height:1.4;color:#111827;">
                    {title}
                </h3>
                <div style="color:#6b7280; font-size:0.85em; margin-bottom:10px;">
                    🕒 {art['date']}
                </div>
                <div style="margin-top:12px;">
                    <a href="{link}" target="_blank">🔗 기사 원문 보기</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
