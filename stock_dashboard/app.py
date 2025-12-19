import requests
from io import StringIO
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from bs4 import BeautifulSoup

# =====================================================
# App Config
# =====================================================
st.set_page_config(
    page_title="📈 주식 비교 & 10일 예측 대시보드",
    layout="wide"
)

st.title("📈 주식 비교 & 10일 예측 대시보드")
st.caption("미국 주식: Stooq · 한국 주식: Naver Finance · 예측: 이동평균 + 신뢰구간")

# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("⚙️ 설정")

market = st.sidebar.radio(
    "시장 선택",
    ["미국 주식", "한국 주식 (KOSPI/KOSDAQ)"]
)

period = st.sidebar.selectbox(
    "기간 선택",
    ["1주", "1달", "1년"]
)

period_days = {"1주": 7, "1달": 30, "1년": 365}
view_days = period_days[period]

tickers_input = st.sidebar.text_input(
    "종목 입력 (콤마로 구분, 최대 3개)",
    placeholder="예: TSLA,AAPL 또는 005930,035720"
)

show_predict = st.sidebar.checkbox("🔮 10일 후 예측 표시", value=True)
run_btn = st.sidebar.button("📊 그래프 그리기")

# =====================================================
# Data Fetch - US (Stooq)
# =====================================================
@st.cache_data(ttl=300)
def fetch_us_stock(ticker: str, days: int) -> pd.DataFrame:
    symbol = ticker.lower() + ".us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return pd.DataFrame()

    df = pd.read_csv(StringIO(r.text))
    if df.empty:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df.tail(days)

# =====================================================
# Data Fetch - KR (Naver)
# =====================================================
@st.cache_data(ttl=300)
def fetch_kr_stock(code: str, days: int) -> pd.DataFrame:
    url = f"https://finance.naver.com/item/sise_day.nhn?code={code}"
    headers = {"User-Agent": "Mozilla/5.0"}

    dfs = []
    page = 1

    while len(dfs) < days and page <= 15:
        r = requests.get(url + f"&page={page}", headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", class_="type2")
        if table is None:
            break

        df = pd.read_html(str(table))[0].dropna()
        dfs.append(df)
        page += 1

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs)
    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.sort_values("날짜")
    df = df.rename(columns={"날짜": "Date", "종가": "Close"})
    return df[["Date", "Close"]].tail(days)

# =====================================================
# Prediction - Moving Average + Confidence Band
# =====================================================
def predict_next_10_days_ma(df_source: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    이동평균 기반 10영업일 예측 + 신뢰구간
    """
    if len(df_source) < window + 5:
        return pd.DataFrame()

    prices = df_source["Close"]
    ma = prices.rolling(window=window).mean().dropna()

    # 평균 변화량
    deltas = ma.diff().dropna()
    avg_delta = deltas.mean()

    last_ma = ma.iloc[-1]
    preds = []

    for _ in range(10):
        last_ma += avg_delta
        preds.append(last_ma)

    future_dates = pd.bdate_range(
        start=df_source["Date"].iloc[-1] + pd.Timedelta(days=1),
        periods=10
    )

    pred_df = pd.DataFrame({
        "Date": future_dates,
        "Pred": preds
    })

    # --- 신뢰구간 계산 ---
    returns = prices.pct_change().dropna()
    volatility = returns.std()

    horizon = np.arange(1, 11)
    pred_df["Upper"] = pred_df["Pred"] * (1 + volatility * np.sqrt(horizon))
    pred_df["Lower"] = pred_df["Pred"] * (1 - volatility * np.sqrt(horizon))

    return pred_df

# =====================================================
# Chart
# =====================================================
def draw_comparison_chart(data_bundle: dict, show_predict: bool, title: str):
    fig = go.Figure()

    for label, bundle in data_bundle.items():
        df_view = bundle["view"]
        df_pred_source = bundle["pred"]

        # 실제 데이터
        fig.add_trace(go.Scatter(
            x=df_view["Date"],
            y=df_view["Close"],
            mode="lines+markers",
            name=label
        ))

        # 예측 + 신뢰구간
        if show_predict:
            pred_df = predict_next_10_days_ma(df_pred_source)
            pred_df = pred_df[pred_df["Date"] > df_view["Date"].max()]

            if not pred_df.empty:
                # 상단 밴드
                fig.add_trace(go.Scatter(
                    x=pred_df["Date"],
                    y=pred_df["Upper"],
                    line=dict(width=0),
                    showlegend=False
                ))

                # 하단 밴드
                fig.add_trace(go.Scatter(
                    x=pred_df["Date"],
                    y=pred_df["Lower"],
                    fill="tonexty",
                    fillcolor="rgba(100,150,255,0.2)",
                    line=dict(width=0),
                    showlegend=False
                ))

                # 예측선
                fig.add_trace(go.Scatter(
                    x=pred_df["Date"],
                    y=pred_df["Pred"],
                    mode="lines+markers",
                    line=dict(dash="dot"),
                    marker=dict(symbol="circle-open"),
                    name=f"{label} (예측)"
                ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Action
# =====================================================
if run_btn:
    if not tickers_input.strip():
        st.warning("종목을 입력하세요.")
    else:
        tickers = [t.strip() for t in tickers_input.split(",")][:3]
        data_bundle = {}

        with st.spinner("데이터 불러오는 중..."):
            for t in tickers:
                if market == "미국 주식":
                    df_view = fetch_us_stock(t, view_days)
                    df_pred = fetch_us_stock(t, max(view_days, 60))
                    label = t.upper()
                else:
                    df_view = fetch_kr_stock(t, view_days)
                    df_pred = fetch_kr_stock(t, max(view_days, 60))
                    label = f"{t} (KR)"

                if not df_view.empty:
                    data_bundle[label] = {
                        "view": df_view,
                        "pred": df_pred
                    }
                else:
                    st.warning(f"{t} 데이터 없음")

        if data_bundle:
            draw_comparison_chart(
                data_bundle,
                show_predict,
                f"{market} · {period} 비교 + 10일 예측"
            )
        else:
            st.error("표시할 데이터가 없습니다.")

# =====================================================
# Footer
# =====================================================
st.caption("⚠️ 예측은 참고용이며 투자 판단의 책임은 사용자에게 있습니다.")
