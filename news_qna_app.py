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
def render_messages(msgs: List[Dict[str,Any]], placeholder):
    html = []
    for m in msgs:
        if m["role"] == "user":
            html.append(
                f"<div style='text-align:right; margin:6px;'>"
                f"<span style='background:#0b62e6; color:white; padding:8px 12px; border-radius:12px; display:inline-block;'>{_linkify(_escape_html(m['content']))}</span>"
                f"</div>"
            )
        else:
            html.append(
                f"<div style='text-align:left; margin:6px;'>"
                f"<span style='background:#f1f1f1; padding:8px 12px; border-radius:12px; display:inline-block;'>{_linkify(_escape_html(m['content']))}</span>"
                f"<div style='font-size:11px; color:gray;'>{m['ts']}</div>"
                f"</div>"
            )
    placeholder.markdown("\n".join(html), unsafe_allow_html=True)

# 메시지 영역 placeholder (중요!)
st.title("🧙‍♂️ 우리 연금술사")
messages_ph = st.empty()

# 답변 생성
def run_answer(question: str):
    # 1) 사용자 메시지 추가 & 즉시 렌더
    st.session_state["messages"].append({
        "role": "user", "content": question, "ts": fmt_ts(datetime.now(TZ))
    })
    render_messages(st.session_state["messages"], messages_ph)

    # 2) 응답 생성 (백엔드 or 데모)
    if svc:
        try:
            result = svc.answer(question) or {}
            ans = result.get("answer") or result.get("content") or "답변을 가져오지 못했습니다."
        except Exception as e:
            ans = f"오류 발생: {e}"
    else:
        ans = f"데모 응답: '{question}'에 대한 분석 결과는 준비 중입니다."

    # 3) 봇 메시지 추가 & 다시 렌더
    st.session_state["messages"].append({
        "role": "assistant", "content": ans, "ts": fmt_ts(datetime.now(TZ))
    })
    render_messages(st.session_state["messages"], messages_ph)

# ---- 폼 (제출 먼저 처리 → 마지막에 렌더) ----
with st.form("chat_form", clear_on_submit=True):
    user_q = st.text_input("질문을 입력하세요", "")
    submitted = st.form_submit_button("전송")

if submitted and user_q.strip():
    run_answer(user_q)

# 마지막 안전 렌더 (최초 로드/새로고침용)
render_messages(st.session_state["messages"], messages_ph)
