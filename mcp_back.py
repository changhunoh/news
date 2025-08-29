# mcp_multi_plot.py
import asyncio, datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

# MCP 도구 임포트 (네가 쓰는 구조 기준)
from mcp_server import inquery_stock_info   # 일별 시세 조회


def _normalize_kis_output(resp, symbol):
    """KIS 일별 응답을 표준 컬럼으로 정리"""
    items = resp.get("output") or resp.get("output1") or []
    if isinstance(items, dict):
        items = [items]
    rows = []
    for x in items:
        # KIS 표준 키 이름들에 방어적으로 대응
        date = x.get("stck_bsop_date") or x.get("trd_dd") or x.get("stck_bsop_dt")
        open_ = x.get("stck_oprc") or x.get("opnprc")
        high = x.get("stck_hgpr") or x.get("hgpr")
        low = x.get("stck_lwpr") or x.get("lwpr")
        close = x.get("stck_clpr") or x.get("clpr")
        vol = x.get("acml_vol") or x.get("trdvol")
        if date and close:
            rows.append({
                "date": pd.to_datetime(date),
                "open": float(open_ or 0),
                "high": float(high or 0),
                "low": float(low or 0),
                "close": float(close),
                "volume": float(vol or 0),
                "symbol": symbol,
            })
    df = pd.DataFrame(rows).sort_values("date")
    return df

async def fetch_one(symbol: str, start: str, end: str) -> pd.DataFrame:
    resp = await inquery_stock_info(symbol, start, end)
    return _normalize_kis_output(resp, symbol)

async def fetch_all(symbols, start, end):
    tasks = [fetch_one(s, start, end) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    dfs = []
    for s, r in zip(symbols, results):
        if isinstance(r, Exception):
            print(f"[WARN] {s} 조회 실패: {r}")
        elif r.empty:
            print(f"[WARN] {s} 데이터 없음")
        else:
            dfs.append(r)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def plot_close(df: pd.DataFrame):
    plt.figure(figsize=(10,5))
    for sym, g in df.groupby("symbol"):
        plt.plot(g["date"], g["close"], label=sym)
    plt.title("종가 추이")
    plt.xlabel("날짜")
    plt.ylabel("종가")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 예시: 최근 3개월, 국내 종목 3개
    symbols = ["005930","000660","035420"]  # 삼성전자, SK하이닉스, NAVER
    end = dt.datetime.now().strftime("%Y%m%d")
    start = (dt.datetime.now() - dt.timedelta(days=90)).strftime("%Y%m%d")

    df = asyncio.run(fetch_all(symbols, start, end))
    if df.empty:
        print("데이터가 없습니다.")
    else:
        print(df.head())

        plot_close(df)

