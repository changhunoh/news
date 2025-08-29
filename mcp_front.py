# streamlit_app.py
import os, sys
from pathlib import Path

import streamlit as st

# 1) Streamlit secrets를 환경변수로 주입 (secrets 우선 적용)
for k, v in st.secrets.items():
    os.environ[k] = str(v)  # setdefault가 아니라 확실히 덮어쓰기

# 2) 이제 envs(.env 로드 + 상수/로거) 를 임포트
sys.path.append(str(Path(__file__).parent.resolve()))
import envs  # noqa: E402

from mcp_server import inquery_stock_info  # noqa: E402

import asyncio, datetime as dt
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="멀티 종목 차트", page_icon="📈", layout="centered")

@st.cache_data(show_spinner=False)
def _to_df(resp: dict, symbol: str) -> pd.DataFrame:
    items = resp.get("output") or resp.get("output1") or []
    if isinstance(items, dict):
        items = [items]
    rows = []
    for x in items:
        date = x.get("stck_bsop_date") or x.get("trd_dd") or x.get("stck_bsop_dt")
        rows.append({
            "date": pd.to_datetime(date, format="%Y%m%d", errors="coerce"),
            "open": float(x.get("stck_oprc") or 0),
            "high": float(x.get("stck_hgpr") or 0),
            "low": float(x.get("stck_lwpr") or 0),
            "close": float(x.get("stck_clpr") or 0),
            "volume": float(x.get("acml_vol") or 0),
            "symbol": symbol,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("date").reset_index(drop=True)

async def _fetch(symbols, start, end):
    coros = [inquery_stock_info(s, start, end) for s in symbols]
    results = await asyncio.gather(*coros, return_exceptions=True)
    dfs = []
    for s, r in zip(symbols, results):
        if isinstance(r, Exception):
            st.warning(f"{s} 조회 실패: {r}")
        else:
            df_s = _to_df(r, s)
            if not df_s.empty:
                dfs.append(df_s)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

st.title("📈 멀티 종목 차트")

symbols_input = st.text_input("종목코드(쉼표로 구분)", "005930,000660,035420")
col1, col2 = st.columns(2)
with col1:
    end_date = st.date_input("종료일", dt.date.today())
with col2:
    start_date = st.date_input("시작일", dt.date.today() - dt.timedelta(days=90))

chart_type = st.radio("차트 타입", ["라인(종가)", "캔들(OHLC)"], horizontal=True)

# 버튼 핸들러
if st.button("차트 그리기"):
    syms = [s.strip() for s in symbols_input.split(",") if s.strip()]
    if not syms:
        st.error("종목코드를 입력하세요."); st.stop()
    if start_date > end_date:
        st.error("시작일이 종료일보다 늦습니다."); st.stop()

    start_s = start_date.strftime("%Y%m%d")
    end_s = end_date.strftime("%Y%m%d")

    with st.spinner("조회 중..."):
        df = run_async(_fetch(syms, start_s, end_s))

    if df.empty:
        st.info("조회 결과가 없습니다. 날짜/종목코드를 확인해 주세요.")
    else:
        if chart_type == "라인(종가)":
            fig = go.Figure()
            for sym, g in df.groupby("symbol"):
                fig.add_trace(go.Scatter(x=g["date"], y=g["close"], mode="lines", name=sym))
            fig.update_layout(title="종가 라인차트", xaxis_title="날짜", yaxis_title="가격")
            st.plotly_chart(fig, use_container_width=True)
        else:
            tabs = st.tabs(list(df["symbol"].unique()))
            for tab, sym in zip(tabs, df["symbol"].unique()):
                with tab:
                    g = df[df["symbol"] == sym]
                    fig = go.Figure(data=[go.Candlestick(
                        x=g["date"], open=g["open"], high=g["high"], low=g["low"], close=g["close"]
                    )])
                    fig.update_layout(title=f"{sym} 캔들차트", xaxis_title="날짜", yaxis_title="가격")
                    st.plotly_chart(fig, use_container_width=True)
