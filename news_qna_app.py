# news_qna_app.py
import os, re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional
import streamlit as st

st.set_page_config(page_title="우리 연금술사", page_icon="🧙‍♂️", layout="centered")
TZ = ZoneInfo(os.getenv("APP_TZ", "Asia/Seoul"))

def fmt_ts(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%Y-%m-%d %H:%M")

def _escape_html(s: Optional[str]) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _linkify(s: str) -> str:
    return re.sub(r"(https?://[\w\-\./%#\?=&:+,~]+)", r'<a href="\1" target="_blank">\1</a>', s or "")

# (선택) 백엔드
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

# 상태
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "대화를 새로 시작합니다. 무엇이 궁금하신가요?",
        "ts": fmt_ts(datetime.now(TZ))
    }]

# 메시지 렌더
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
                md = src.get("metadata", {}) if isinstance(src, dict) else {}
                title = md.get("title") or f"문서 {j}"
                url = md.get("url")
                score = md.get("score", 0.0)
                label = f"#{j} {title} ({score:.2f})"
                if url:
                    label = f"<a href='{url}' target='_blank'>{label}</a>"
                html.append(f"<div style='font-size:12px; color:#0b62e6; margin-left:12px;'>📎 {label}</div>")

    placeholder.markdown("\n".join(html), unsafe_allow_html=True)

# 메시지 영역 placeholder (중요!)
st.title("🧙‍♂️ 우리 연금술사")
messages_ph = st.empty()

debug = st.sidebar.toggle("🔍 RAG 디버그 보기", value=True)

# 답변 생성
def run_answer(question: str):
    # 사용자 메시지
    st.session_state["messages"].append({
        "role": "user", "content": question, "ts": fmt_ts(datetime.now(TZ))
    })
    render_messages(st.session_state["messages"], messages_ph)

    # 답변 생성
    sources = []
    raw_result = {}
    if svc:
        try:
            result = svc.answer(question) or {}
            ans = raw_result.get("answer") or raw_result.get("content") or "답변을 가져오지 못했습니다."
            # 다양한 키 호환
            sources = (
            raw_result.get("source_documents")
            or raw_result.get("sources")
            or raw_result.get("docs")
            or []
            )
        except Exception as e:
            ans = f"오류 발생: {e}"
    else:
        ans = f"데모 응답: '{question}'에 대한 분석 결과는 준비 중입니다."
        raw_result = {"answer": ans, "source_documents": []}
    # 디버그 패널
    if debug:
        with st.expander("🔎 RAG 디버그 (원시 결과/컨텍스트/스코어)"):
            st.write("raw_result keys:", list(raw_result.keys()))
            st.write("num sources:", len(sources))
            for i, d in enumerate(sources, 1):
                if isinstance(d, dict):
                    md = d.get("metadata", {})
                    score = md.get("score", d.get("score", None))
                    title = md.get("title") or md.get("path") or md.get("source") or f"문서 {i}"
                    url = md.get("url")
                    st.markdown(f"**#{i} {title}**  | score={score}")
                    if url: st.markdown(f"[원문]({url})")
                    txt = d.get("content") or d.get("page_content") or ""
                    st.code((txt[:800] + (" …" if len(txt)>800 else "")))
                else:
                    st.write(d)

    # 어시스턴트 메시지 (근거 포함)
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
    run_answer(user_q)
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# 마지막 안전 렌더 (최초 로드/새로고침용)
render_messages(st.session_state["messages"], messages_ph)
