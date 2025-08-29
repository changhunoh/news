# streamlit_app.py
import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

import streamlit as st

# 1) Streamlit secrets를 환경변수로 주입 (이미 값 있으면 덮어쓰지 않음)
for k, v in st.secrets.items():
    os.environ.setdefault(k, str(v))

from .mcp_server import inquery_stock_info  # ← 점(.) 제거

import asyncio, datetime as dt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from .mcp_server import inquery_stock_info

st.set_page_config(page_title="멀티 종목 차트", page_icon="📈", layout="centered")

@st.cache_data(show_spinner=False)
def _to_df(resp, symbol):
    items = resp.get("output") or resp.get("output1") or []
    if isinstance(items, dict):
        items = [items]
    rows = []
    for x in items:
        date = x.get("stck_bsop_date") or x.get("trd_dd") or x.get("stck_bsop_dt")
        rows.append({
            "date": pd.to_datetime(date),
            "open": float(x.get("stck_oprc") or 0),
            "high": float(x.get("stck_hgpr") or 0),
            "low": float(x.get("stck_lwpr") or 0),
            "close": float(x.get("stck_clpr") or 0),
            "volume": float(x.get("acml_vol") or 0),
            "symbol": symbol,
        })
    return pd.DataFrame(rows).sort_values("date")

async def _fetch(symbols, start, end):
    coros = [inquery_stock_info(s, start, end) for s in symbols]
    results = await asyncio.gather(*coros, return_exceptions=True)
    dfs = []
    for s, r in zip(symbols, results):
        if isinstance(r, Exception):
            st.warning(f"{s} 조회 실패: {r}")
        else:
            dfs.append(_to_df(r, s))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

st.title("📈 멀티 종목 차트")

symbols_input = st.text_input("종목코드(쉼표로 구분)", "005930,000660,035420")
col1, col2 = st.columns(2)
with col1:
    end = st.date_input("종료일", dt.date.today())
with col2:
    start = st.date_input("시작일", dt.date.today() - dt.timedelta(days=90))

chart_type = st.radio("차트 타입", ["라인(종가)", "캔들(OHLC)"], horizontal=True)

if st.button("차트 그리기"):
    syms = [s.strip() for s in symbols_input.split(",") if s.strip()]
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    with st.spinner("조회 중..."):
        df = asyncio.run(_fetch(syms, start_s, end_s))
    if df.empty:
        st.error("데이터가 없습니다.")
    else:
        if chart_type == "라인(종가)":
            fig = go.Figure()
            for sym, g in df.groupby("symbol"):
                fig.add_trace(go.Scatter(x=g["date"], y=g["close"], mode="lines", name=sym))
            fig.update_layout(title="종가 라인차트", xaxis_title="날짜", yaxis_title="가격")
            st.plotly_chart(fig, use_container_width=True)
        else:  # 캔들
            # 심볼별 탭으로 캔들 표시
            tabs = st.tabs(list(df["symbol"].unique()))
            for tab, sym in zip(tabs, df["symbol"].unique()):
                with tab:
                    g = df[df["symbol"] == sym]
                    fig = go.Figure(data=[go.Candlestick(
                        x=g["date"], open=g["open"], high=g["high"], low=g["low"], close=g["close"]
                    )])
                    fig.update_layout(title=f"{sym} 캔들차트", xaxis_title="날짜", yaxis_title="가격")
                    st.plotly_chart(fig, use_container_width=True)





