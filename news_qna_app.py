# news_qna_app.py
import os, re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional
import streamlit as st

# ------------------------
# 기본 설정
# ------------------------
st.set_page_config(page_title="우리 연금술사", page_icon="🧙‍♂️", layout="centered")
TZ = ZoneInfo(os.getenv("APP_TZ", "Asia/Seoul"))

def fmt_ts(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%Y-%m-%d %H:%M")

def _escape_html(s: Optional[str]) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _linkify(s: str) -> str:
    return re.sub(r"(https?://[\w\-\./%#\?=&:+,~]+)", r'<a href="\1" target="_blank">\1</a>', s or "")

# 환경/컬렉션 정보 표시 (사이드바)
st.sidebar.write("COLLECTION_NAME =", os.getenv("COLLECTION_NAME"))
st.sidebar.write("EMBED_MODEL_NAME =", os.getenv("EMBED_MODEL_NAME"))
st.sidebar.write("EMBED_DIM =", os.getenv("EMBED_DIM"))

from qdrant_client import QdrantClient
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
col = os.getenv("COLLECTION_NAME","stock_news")
info = client.get_collection(col)
st.sidebar.write("Qdrant vector_size =", info.config.params.vectors.size)
cnt = client.count(col, exact=True).count
st.sidebar.write("Qdrant points =", cnt)

# ------------------------
# 백엔드 서비스(선택)
# ------------------------
try:
    from news_qna_service import NewsQnAService
except Exception:
    NewsQnAService = None

@st.cache_resource
def get_service():
    if NewsQnAService is None:
        return None
    try:
        return NewsQnAService(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            collection=os.getenv("COLLECTION_NAME", "stock_news"),
        )
    except Exception as e:
        st.error(f"[서비스 초기화 실패] {e}")
        return None

svc = get_service()

# ------------------------
# 상태
# ------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "대화를 새로 시작합니다. 무엇이 궁금하신가요?",
        "ts": fmt_ts(datetime.now(TZ))
    }]

# ------------------------
# 추출 함수들
# ------------------------
def _extract_text(d: dict) -> str:
    # retrieve 결과 형태를 기준으로 content 우선
    if not isinstance(d, dict):
        return str(d)
    txt = d.get("content")
    if txt:
        return txt
    # 혹시 service가 payload 형태를 그대로 넘길 때 대비
    md = d.get("metadata") or {}
    if isinstance(md.get("doc"), dict):
        return md["doc"].get("content") or md["doc"].get("text") or ""
    if isinstance(md.get("doc"), str):
        return md["doc"]
    return md.get("content") or md.get("text") or ""

def _extract_title_url(d: dict):
    md = (d.get("metadata") or {}) if isinstance(d, dict) else {}
    title = md.get("title") or md.get("path") or md.get("source") or md.get("file_name")
    url = md.get("url") or md.get("link")
    return title, url

def _extract_score_str(d: dict) -> Optional[str]:
    if not isinstance(d, dict):
        return None
    sim = d.get("score", None)
    dist = d.get("distance", None)
    mode = (d.get("distance_mode") or "").lower()
    try:
        if sim is not None:
            return f"{float(sim):.4f}"
        if dist is not None:
            if "cosine" in mode:
                return f"{1.0 - float(dist):.4f}"
            return f"dist={float(dist):.4f}"
    except Exception:
        pass
    # metadata 안쪽일 수도 있음
    md = d.get("metadata") or {}
    for k in ("score","similarity"):
        if k in md:
            try: return f"{float(md[k]):.4f}"
            except: pass
    for k in ("distance",):
        if k in md:
            try:
                return f"{1.0 - float(md[k]):.4f}" if "cosine" in mode else f"dist={float(md[k]):.4f}"
            except:
                pass
    return None

# ------------------------
# 메시지 렌더
# ------------------------
def render_messages(msgs, placeholder):
    html = []
    for m in msgs:
        if m["role"] == "user":
            html.append(
                f"<div style='text-align:right; margin:6px;'>"
                f"<span style='background:#0b62e6; color:white; padding:8px 12px; border-radius:12px;'>{_linkify(_escape_html(m['content']))}</span>"
                f"</div>"
            )
        else:
            html.append(
                f"<div style='text-align:left; margin:6px;'>"
                f"<span style='background:#f1f1f1; padding:8px 12px; border-radius:12px;'>{_linkify(_escape_html(m['content']))}</span>"
                f"<div style='font-size:11px; color:gray;'>{m['ts']}</div>"
            )
            # 🔎 근거칩
            for j, src in enumerate(m.get("sources", []), 1):
                title, url = _extract_title_url(src)
                score_s = _extract_score_str(src)
                label = f"#{j} {title or f'문서 {j}'}"  # ← 따옴표 오타 수정
                if score_s:
                    label += f" ({score_s})"
                if url:
                    label = f"<a href='{url}' target='_blank'>{label}</a>"
                html.append(f"<div style='font-size:12px; color:#0b62e6; margin-left:12px;'>📎 {label}</div>")

    placeholder.markdown("\n".join(html), unsafe_allow_html=True)

# ------------------------
# UI 헤더 & 플레이스홀더
# ------------------------
st.title("🧙‍♂️ 우리 연금술사")
messages_ph = st.empty()
debug = st.sidebar.toggle("🔍 RAG 디버그 보기", value=True)

# ------------------------
# 답변 생성
# ------------------------
def run_answer(question: str):
    # 1) 사용자 메시지 즉시 반영
    st.session_state["messages"].append({
        "role": "user", "content": question, "ts": fmt_ts(datetime.now(TZ))
    })
    render_messages(st.session_state["messages"], messages_ph)

    # 2) 서비스 호출
    sources = []
    result: Dict[str, Any] = {}
    if svc:
        try:
            result = svc.answer(question) or {}
            # 다양한 키 대응
            ans = (
                result.get("answer")
                or result.get("output_text")
                or result.get("output")
                or result.get("content")
                or ""
            ).strip()
            sources = (
                result.get("source_documents")
                or result.get("sources")
                or result.get("docs")
                or []
            )
            if not ans:
                # 근거는 있는데 답변이 비면 안전 Fallback
                if sources:
                    ans = "관련 원문 요약에 실패했습니다. 원문 일부:\n\n" + (_extract_text(sources[0])[:400])
                else:
                    ans = "관련 정보를 찾을 수 없습니다."
        except Exception as e:
            ans = f"오류 발생: {e}"
            result = {"error": str(e)}
            sources = []
    else:
        ans = f"데모 응답: '{question}'에 대한 분석 결과는 준비 중입니다."
        result = {"answer": ans, "source_documents": []}
        sources = []

    # 3) 디버그 패널
    if debug:
        with st.expander("RAG raw result"):
            try:
                st.write("result keys:", list(result.keys()))
            except Exception:
                st.write("result:", result)
            st.write("num sources:", len(sources))
            for i, d in enumerate(sources, 1):
                title, url = _extract_title_url(d)
                score_s = _extract_score_str(d)
                st.markdown(f"**#{i} {title}** | score={score_s}")
                if url: st.markdown(f"[원문]({url})")
                st.code(_extract_text(d)[:600])

    # 4) 어시스턴트 메시지 반영
    st.session_state["messages"].append({
        "role": "assistant",
        "content": ans,
        "sources": sources,
        "ts": fmt_ts(datetime.now(TZ))
    })
    render_messages(st.session_state["messages"], messages_ph)

# ---- 폼 (제출 먼저 처리 → 마지막에 렌더) ----
with st.form("chat_form", clear_on_submit=True):
    user_q = st.text_input("질문을 입력하세요", "")
    submitted = st.form_submit_button("전송")

if submitted and user_q.strip():
    # 즉시 반영 구조 → rerun 호출하지 않음
    run_answer(user_q)

# 최초/마지막 안전 렌더
render_messages(st.session_state["messages"], messages_ph)

# ------------------------
# 사이드바: 바로 붙여넣기용 덤프
# ------------------------
with st.sidebar.expander("🧩 debug dump (붙여넣어 주시면 돼요)"):
    q = st.text_input("테스트 질의", "삼성전자 전망", key="dump_q")
    if st.button("answer() 호출", key="dump_btn"):
        try:
            res = svc.answer(q) if svc else {}
        except Exception as e:
            res = {"error": str(e)}
        st.write("keys:", list(res.keys()) if isinstance(res, dict) else type(res))
        srcs = (res.get("source_documents") or res.get("sources") or res.get("docs") or []) if isinstance(res, dict) else []
        st.write("num sources:", len(srcs))
        if srcs:
            s0 = srcs[0]
            st.write("source[0] keys:", list(s0.keys()) if isinstance(s0, dict) else type(s0))
            md = (s0.get("metadata") or {}) if isinstance(s0, dict) else {}
            st.write("metadata keys:", list(md.keys()))
            txt = (s0.get("content") or s0.get("page_content") or s0.get("text")
                   or (s0.get("metadata") or {}).get("content") or "")
            st.code((txt[:600] + (" …" if len(txt) > 600 else "")))
