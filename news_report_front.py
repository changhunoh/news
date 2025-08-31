# app.py
import os
from typing import List, Dict, Any, Optional
import streamlit as st

from news_report_service import NewsReportService

st.set_page_config(page_title="우리 연금술사 연금리포트", page_icon="📰", layout="centered")

# -----------------------------
# st.secrets → os.environ 주입
# -----------------------------
def _prime_env_from_secrets() -> None:
    try:
        if hasattr(st, "secrets") and st.secrets:
            for k, v in st.secrets.items():
                if k == "gcp_service_account":
                    continue
                os.environ.setdefault(k, str(v))
    except Exception:
        pass
_prime_env_from_secrets()

# -----------------------------
# helpers
# -----------------------------
def _parse_stocks(csv_text: str) -> List[str]:
    return [s.strip() for s in csv_text.split(",") if s.strip()]

def _fmt_link(md: Dict[str, Any]) -> str:
    title = md.get("title") or md.get("headline") or md.get("doc_title") or md.get("doc_id") or ""
    url = md.get("url") or md.get("link") or md.get("source_url") or ""
    if url and title: return f"[{title}]({url})"
    if url:           return f"[원문 링크]({url})"
    return title or "(링크/제목 없음)"

# -----------------------------
# Service 인스턴스 (캐시)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_service() -> Optional[NewsReportService]:
    try:
        return NewsReportService()
    except Exception as e:
        st.warning(f"서비스 초기화 실패: {e}")
        return None


# ======================
# CSS: 사이드바 hover 시에만 보이도록
# ======================
st.markdown(
    """
    <style>
    /* 사이드바 전체를 왼쪽으로 숨김 */
    [data-testid="stSidebar"] {
        transform: translateX(-250px);
        transition: all 0.3s;
        opacity: 0.2;  /* 살짝만 보이게 */
    }
    /* 마우스를 올리면 원위치 */
    [data-testid="stSidebar"]:hover {
        transform: translateX(0);
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# UI
# -----------------------------
st.title("💰 우리연금술사 종합리포트")
st.markdown(
    "<p style='font-size:20px; color:gray;'>✨ 창훈님을 위한 연금술사의 연금리포트 ✨</p>",
    unsafe_allow_html=True
)
#st.caption("우리 연금술사가 창훈님을 위해 제작한 퇴직연금 종합 리포트에요")

with st.sidebar:
    st.subheader("실행 설정")
    stocks_text = st.text_input("종목(콤마로 구분)", value="삼성전자,SK하이닉스,LG에너지솔루션", help="예: 삼성전자,우리금융지주 / 또는 AAPL,NVDA")
    #template = st.text_input("질문 템플릿 (옵션)", value="{stock} 관련해서 종목의 가격에 중요한 뉴스는?")
    #top_k = st.number_input("top_k", min_value=1, max_value=20, value=5, step=1)
    #use_rerank = st.toggle("리랭크 사용 (현재는 top_k 자르기)", value=False)
    #rerank_top_k = st.number_input("rerank_top_k", min_value=1, max_value=50, value=5, step=1)
    #max_workers = st.slider("동시 처리 쓰레드", min_value=1, max_value=10, value=5)
    run_btn = st.button("🚀 실행", type="primary")

st.divider()
# st.markdown("""
# **필수 Secrets:** `GOOGLE_CLOUD_PROJECT`, `QDRANT_URL`, `QDRANT_API_KEY`  
# (옵션) `GOOGLE_CLOUD_LOCATION`, `COLLECTION_NAME`, `EMBED_MODEL_NAME`, `GENAI_MODEL_NAME`, `EMBED_DIM`, `DEFAULT_TOP_K`, `RERANK_TOP_K`, `[gcp_service_account]`
# """)

if run_btn:
    stocks = _parse_stocks(stocks_text)
    if not stocks:
        st.error("종목을 1개 이상 입력해 주세요.")
        st.stop()

    svc = get_service()
    if svc is None:
        st.error("서비스 초기화 실패. Secrets 설정을 확인해 주세요.")
        st.stop()

    # 런타임 설정 반영
    #svc.top_k = int(top_k)
    #svc.use_rerank = bool(use_rerank)
    #svc.rerank_top_k = int(rerank_top_k)

    # (선택) 디버깅: 각 종목의 보유 문서 수
    cols = st.columns(len(stocks))
    for i, s in enumerate(stocks):
        with cols[i]:
            try:
                c = svc.count_by_stock(s)
                st.caption(f"`{s}` 문서 수: **{c}**")
            except Exception:
                pass

    with st.spinner("리포트를 생성하는중..."):
        try:
            #base_template = (template or None)
            result = svc.answer_5_stocks_and_reduce(
                stocks=stocks,
                template=f"{stocks} 관련해서 종목의 가격에 영향을 미치는 중요한 뉴스는?",
                max_workers=int(5),
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    # 최종 리포트
    st.subheader("📌 최종 리포트")
    final_report = (result.get("final_report") or "").strip()
    if final_report:
        st.markdown(final_report)
    else:
        st.info("최종 리포트를 생성하지 못했습니다.")

    st.divider()

    # 종목별 결과
    st.subheader("🔎 종목별 요약보기")
    for r in result.get("results", []):
        stock = r.get("stock", "")
        with st.expander(f"[{stock}] 요약 보기", expanded=False):
            ans = (r.get("answer") or "").strip()
            if ans:
                st.markdown(ans)
            else:
                st.write("관련된 정보를 찾을 수 없습니다.")

            # 소스 문서
            src_docs = r.get("source_documents") or []
            if src_docs:
                st.markdown("**근거 기사**")
                for i, d in enumerate(src_docs[:10], start=1):
                    md = d.get("metadata") or {}
                    score = d.get("score")
                    dist_mode = d.get("distance_mode") or ""
                    link = _fmt_link(md)
                    # 루트 스키마: doc_id/stock/chunk_idx/… 노출
                    extra = []
                    if "stock" in md: extra.append(f"stock=`{md.get('stock')}`")
                    if "doc_id" in md: extra.append(f"doc_id=`{md.get('doc_id')}`")
                    if "chunk_idx" in md: extra.append(f"chunk=`{md.get('chunk_idx')}`")
                    meta_line = " • ".join(extra)
                    #st.markdown(f"- {i}. {link}  \n  - {meta_line} • score(raw): `{score}` • mode: `{dist_mode}`")
                    st.markdown(f"- {i}. {link}")
            else:
                st.write("소스 문서 없음")
























