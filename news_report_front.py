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
    if svc is not None:
        sidebar_qdrant_metadata_tools(svc)
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

def sidebar_qdrant_metadata_tools(svc):
    """
    Qdrant 메타데이터를 간단히 훑어보는 사이드바 도구.
    - 샘플 payload 확인
    - stock 필터 스크롤
    - payload 인덱스/스키마 확인
    - stock 분포 집계
    """
    st.sidebar.subheader("🧭 Qdrant 메타데이터 탐색")

    # 기본 옵션
    col_name = getattr(svc, "collection", "stock_news")
    st.sidebar.caption(f"Collection: `{col_name}`")

    stock_filter = st.sidebar.text_input("stock 필터(옵션, metadata.stock)", value="")
    limit = st.sidebar.number_input("가져올 샘플 개수", 5, 500, 20, step=5)
    show_raw = st.sidebar.toggle("Raw payload 보이기", value=False)

    # =============== 인덱스/스키마 보기 ===============
    if st.sidebar.button("📑 인덱스/스키마 보기"):
        try:
            info = svc.qc.get_collection(col_name)
            # payload_schema가 있으면 보여주고, 없으면 전체 info를 json으로 노출
            payload_schema = getattr(info, "payload_schema", None)
            st.sidebar.markdown("**Payload schema**")
            if payload_schema:
                st.sidebar.json(payload_schema)
            else:
                st.sidebar.json(info.dict() if hasattr(info, "dict") else str(info))
        except Exception as e:
            st.sidebar.error(f"스키마 조회 실패: {e}")

    # =============== 샘플 payload 조회 ===============
    if st.sidebar.button("🔍 샘플 payload 보기"):
        try:
            q_filter = None
            if stock_filter.strip():
                q_filter = Filter(
                    must=[FieldCondition(key="metadata.stock", match=MatchValue(value=stock_filter.strip()))]
                )

            # scroll로 샘플 가져오기
            points, _ = svc.qc.scroll(
                collection_name=col_name,
                limit=int(limit),
                with_payload=True,
                with_vectors=False,
                scroll_filter=q_filter,   # qdrant_client>=1.6: scroll_filter, 구버전은 filter
            )

            def _row(p):
                payload = p.payload or {}
                md = payload.get("metadata") or {}
                # 흔히 보는 필드만 안전하게 노출 (없으면 빈 문자열)
                return {
                    "id": str(getattr(p, "id", "")),
                    "metadata.stock": (md.get("stock") if isinstance(md, dict) else ""),
                    "metadata.title": (md.get("title") if isinstance(md, dict) else ""),
                    "metadata.url": (md.get("url") if isinstance(md, dict) else ""),
                }

            rows = [_row(p) for p in points]
            st.sidebar.markdown(f"**샘플 {len(rows)}건**")
            if rows:
                st.sidebar.dataframe(rows, use_container_width=True)
            else:
                st.sidebar.info("결과가 없습니다.")

            if show_raw:
                st.sidebar.markdown("**Raw payload (최대 3건)**")
                for p in points[:3]:
                    st.sidebar.json(p.payload or {})
        except Exception as e:
            # 인덱스 없는 필터 에러 등 → 안내
            st.sidebar.error(f"샘플 조회 실패: {e}\n\n필요시 `metadata.stock`에 keyword 인덱스를 생성하세요.")

    # =============== stock 분포 집계 ===============
    if st.sidebar.button("📊 stock 분포 집계(샘플 기반)"):
        try:
            # 필터 없이 넓게 가져와서 샘플 분포 파악
            raw_limit = int(max(limit, 100))
            points, _ = svc.qc.scroll(
                collection_name=col_name,
                limit=raw_limit,
                with_payload=True,
                with_vectors=False,
            )
            from collections import Counter
            cnt = Counter()
            for p in points:
                payload = p.payload or {}
                md = payload.get("metadata") or {}
                val = None
                if isinstance(md, dict):
                    val = md.get("stock")
                # 문자열 또는 리스트 모두 대응
                if isinstance(val, str):
                    cnt[val] += 1
                elif isinstance(val, list):
                    for v in val:
                        cnt[v] += 1

            top = [{"stock": k, "count": v} for k, v in cnt.most_common(30)]
            if top:
                st.sidebar.dataframe(top, use_container_width=True)
            else:
                st.sidebar.info("샘플에서 stock 값을 찾지 못했습니다.")
        except Exception as e:
            st.sidebar.error(f"집계 실패: {e}")


