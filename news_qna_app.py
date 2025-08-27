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

# ------------------------
# 백엔드 서비스 (선택)
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
# 상태 초기화
# ------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "대화를 새로 시작합니다. 무엇이 궁금하신가요?",
        "ts": fmt_ts(datetime.now(TZ))
    }]

# ------------------------
# 메시지 출력 함수
# ------------------------
def render_messages(msgs: List[Dict[str,Any]]):
    for m in msgs:
        if m["role"] == "user":
            st.markdown(
                f"<div style='text-align:right; margin:6px;'>"
                f"<span style='background:#0b62e6; color:white; padding:8px 12px; border-radius:12px;'>{m['content']}</span>"
                f"</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align:left; margin:6px;'>"
                f"<span style='background:#f1f1f1; padding:8px 12px; border-radius:12px;'>{m['content']}</span>"
                f"<div style='font-size:11px; color:gray;'>{m['ts']}</div>"
                f"</div>", unsafe_allow_html=True
            )

# ------------------------
# QnA 실행
# ------------------------
def run_answer(question: str):
    st.session_state["messages"].append({
        "role": "user", "content": question, "ts": fmt_ts(datetime.now(TZ))
    })

    # 답변 생성
    if svc:
        try:
            ans = svc.answer(question).get("answer", "답변을 가져오지 못했습니다.")
        except Exception as e:
            ans = f"오류 발생: {e}"
    else:
        ans = f"데모 응답: '{question}'에 대한 분석 결과는 준비 중입니다."

    st.session_state["messages"].append({
        "role": "assistant", "content": ans, "ts": fmt_ts(datetime.now(TZ))
    })

# ------------------------
# UI
# ------------------------
st.title("🧙‍♂️ 우리 연금술사")
render_messages(st.session_state["messages"])

with st.form("chat_form", clear_on_submit=True):
    user_q = st.text_input("질문을 입력하세요", "")
    submitted = st.form_submit_button("전송")
    if submitted and user_q.strip():
        run_answer(user_q)
