# app.py
import os
from typing import List, Dict, Any, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue
import streamlit as st

# 서비스 코드 import (같은 리포에 news_rag_service.py가 있어야 합니다)
from news_report_service import NewsReportService

st.set_page_config(page_title="우리 연금술사 • News RAG 테스트", page_icon="📰", layout="centered")


# -----------------------------
# st.secrets → os.environ 주입
# -----------------------------
def _prime_env_from_secrets() -> None:
    try:
        if hasattr(st, "secrets") and st.secrets:
            for k, v in st.secrets.items():
                # 서비스 계정(dict)은 여기서 env로 넣지 않고 NewsQnAService 쪽에서 직접 사용
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
    title = (
        md.get("title")
        or md.get("headline")
        or md.get("doc_title")
        or md.get("source")
        or ""
    )
    url = md.get("url") or md.get("link") or md.get("source_url") or ""
    if url and title:
        return f"[{title}]({url})"
    elif url:
        return f"[원문 링크]({url})"
    return title or "(메타데이터 없음)"


# -----------------------------
# Service 인스턴스 (캐시)
# -----------------------------
# 벡터DB 페이로드 확인 필요시 활성화
# def sidebar_qdrant_raw_payload_browser(svc):
#     """
#     Qdrant 컬렉션에서 payload 원본 그대로를 페이지 단위로 조회/표시/다운로드.
#     - 서버 필터/인덱스 불필요 (scroll only)
#     - offset 기반 페이지 이동
#     - 클라이언트 측 표시 개수 제한 및 다운로드(JSON / NDJSON)
#     """
#     st.sidebar.subheader("🧾 Qdrant Raw Payload Browser")

#     col_name = getattr(svc, "collection", "stock_news")
#     st.sidebar.caption(f"Collection: `{col_name}`")

#     page_size = st.sidebar.number_input("페이지 크기", min_value=5, max_value=500, value=30, step=5)
#     show_max = st.sidebar.number_input("표시할 개수(상위)", min_value=1, max_value=200, value=20, step=1)
#     as_list_view = st.sidebar.toggle("한 번에 JSON 배열로 보기", value=False)

#     # 상태 저장
#     if "raw_points" not in st.session_state: st.session_state["raw_points"] = []
#     if "raw_offset" not in st.session_state: st.session_state["raw_offset"] = None
#     if "raw_next" not in st.session_state: st.session_state["raw_next"] = None

#     def _scroll(limit_val: int, offset_val=None):
#         # 필터 없이 payload만 조회
#         return svc.qc.scroll(
#             collection_name=col_name,
#             limit=int(limit_val),
#             with_payload=True,
#             with_vectors=False,
#             offset=offset_val,
#         )

#     # 버튼들
#     c1, c2, c3 = st.sidebar.columns(3)
#     if c1.button("⏮ 처음부터"):
#         st.session_state["raw_offset"] = None
#         pts, nxt = _scroll(page_size, None)
#         st.session_state["raw_points"] = pts
#         st.session_state["raw_next"] = nxt
#     if c2.button("🔄 새로고침"):
#         pts, nxt = _scroll(page_size, st.session_state.get("raw_offset"))
#         st.session_state["raw_points"] = pts
#         st.session_state["raw_next"] = nxt
#     if c3.button("⏭ 다음 페이지"):
#         pts, nxt = _scroll(page_size, st.session_state.get("raw_next"))
#         st.session_state["raw_points"] = pts
#         st.session_state["raw_offset"] = st.session_state.get("raw_next")
#         st.session_state["raw_next"] = nxt

#     # 초회 자동 로드
#     if not st.session_state["raw_points"]:
#         pts, nxt = _scroll(page_size, None)
#         st.session_state["raw_points"] = pts
#         st.session_state["raw_next"] = nxt

#     points = st.session_state["raw_points"]
#     next_off = st.session_state.get("raw_next")
#     st.sidebar.caption(f"현재 페이지 개수: {len(points)}  |  next_offset: `{next_off}`")

#     # payload 원본 목록
#     payloads: List[Dict[str, Any]] = []
#     for p in points:
#         payloads.append(p.payload or {})

#     # 표시
#     to_show = payloads[: int(show_max)]
#     st.sidebar.markdown(f"**표시 중: {len(to_show)}건 (총 {len(payloads)}건 중)**")

#     if as_list_view:
#         # JSON 배열로 한 번에 보기
#         st.sidebar.json(to_show)
#     else:
#         # 개별 payload 원본을 펼침/축소로 보기
#         for i, pl in enumerate(to_show, start=1):
#             with st.sidebar.expander(f"payload #{i}", expanded=False):
#                 st.json(pl)

#     # 다운로드 (JSON / NDJSON)
#     json_data = json.dumps(to_show, ensure_ascii=False, indent=2)
#     ndjson_data = "\n".join(json.dumps(obj, ensure_ascii=False) for obj in to_show)

#     st.sidebar.download_button(
#         "⬇️ Download (JSON 배열)",
#         data=json_data.encode("utf-8"),
#         file_name=f"{col_name}_payloads.json",
#         mime="application/json",
#         use_container_width=True,
#     )
#     st.sidebar.download_button(
#         "⬇️ Download (NDJSON)",
#         data=ndjson_data.encode("utf-8"),
#         file_name=f"{col_name}_payloads.ndjson",
#         mime="application/x-ndjson",
#         use_container_width=True,
#     )

# -----------------------------
# UI
# -----------------------------
st.title("📰 우리 연금술사 • News RAG 간단 테스트")
st.caption("Qdrant + Vertex AI (Gemini/Embedding) 기반 • 5개 종목 병렬(Map) → 최종 통합(Reduce)")

with st.sidebar:
    st.subheader("실행 설정")
    stocks_text = st.text_input(
        "종목(콤마로 구분)",
        value="AAPL,NVDA,TSLA,MSFT,AMZN",
        help="예: AAPL,NVDA,TSLA,MSFT,AMZN",
    )
    template = st.text_input(
        "질문 템플릿 (옵션)",
        value="{stock} 관련해서 종목의 가격에 중요한 뉴스는?",
        help="비우면 서비스 기본 템플릿 사용"
    )
    top_k = st.number_input("top_k", min_value=1, max_value=20, value=5, step=1)
    use_rerank = st.toggle("리랭크 사용 (현재는 top_k 자르기)", value=False)
    rerank_top_k = st.number_input("rerank_top_k", min_value=1, max_value=50, value=5, step=1)
    max_workers = st.slider("동시 처리 쓰레드", min_value=1, max_value=10, value=5)
    run_btn = st.button("🚀 실행", type="primary")

st.divider()
st.markdown(
    """
**주의:** Streamlit Cloud에서 실행하려면 `Secrets`에 아래 값들을 설정해 주세요.  
`GOOGLE_CLOUD_PROJECT`, `QDRANT_URL`, `QDRANT_API_KEY`, (선택) `GOOGLE_CLOUD_LOCATION`, `COLLECTION_NAME`, `EMBED_MODEL_NAME`, `GENAI_MODEL_NAME`, `EMBED_DIM`, `DEFAULT_TOP_K`, `RERANK_TOP_K`, 그리고 `[gcp_service_account]` 서비스 계정.
"""
)

if run_btn:
    stocks = _parse_stocks(stocks_text)
    if not stocks:
        st.error("종목을 1개 이상 입력해 주세요.")
        st.stop()

    svc =  NewsReportService()
    """
    벡터디비 페이로드 확인시에만 활성화
    if svc is not None:
        sidebar_qdrant_raw_payload_browser(svc)
    """
    if svc is None:
        st.error("서비스를 초기화할 수 없습니다. 좌측의 Secrets 설정을 확인해 주세요.")
        st.stop()

    # 실행 전 런타임 설정 반영
    svc.top_k = int(top_k)
    svc.use_rerank = bool(use_rerank)
    svc.rerank_top_k = int(rerank_top_k)

    with st.spinner("분석 중..."):
        try:
            base_template = (template or None)
            result = svc.answer_5_stocks_and_reduce(
                stocks=stocks,
                template=base_template,
                max_workers=int(max_workers),
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
    st.subheader("🔎 종목별 부분답 & 소스")
    for r in result.get("results", []):
        stock = r.get("stock", "")
        with st.expander(f"[{stock}] 부분답 보기", expanded=False):
            ans = (r.get("answer") or "").strip()
            if ans:
                st.markdown(ans)
            else:
                st.write("응답 없음")

            # 소스 문서
            src_docs = r.get("source_documents") or []
            if src_docs:
                st.markdown("**참고 소스(상위 몇 건)**")
                for i, d in enumerate(src_docs[:10], start=1):
                    md = d.get("metadata") or {}
                    score = d.get("score")
                    distance_mode = d.get("distance_mode") or ""
                    link = _fmt_link(md)
                    st.markdown(f"- {i}. {link}  \n  - score(raw): `{score}` • distance_mode: `{distance_mode}`")
            else:
                st.write("소스 문서 없음")










