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

# ------------------------
# 아바타 설정
# ------------------------
ASSISTANT_AVATAR_URL = os.getenv("ASSISTANT_AVATAR_URL", "")
USER_AVATAR_URL      = os.getenv("USER_AVATAR_URL", "")
ASSISTANT_EMOJI      = "🧙‍♂️"
USER_EMOJI           = "🤴"

def _avatar_html(role: str) -> str:
    if role == "assistant":
        if ASSISTANT_AVATAR_URL:
            return f"<div class='avatar'><img src='{ASSISTANT_AVATAR_URL}'/></div>"
        return f"<div class='avatar emoji'>{ASSISTANT_EMOJI}</div>"
    else:
        if USER_AVATAR_URL:
            return f"<div class='avatar'><img src='{USER_AVATAR_URL}'/></div>"
        return f"<div class='avatar emoji'>{USER_EMOJI}</div>"

# ------------------------
# CSS 스타일
# ------------------------
st.markdown("""
<style>
/* 전체 레이아웃 */
.main {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

/* 채팅 메시지 */
.chat-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
    align-items: flex-start;
}

.bot-row {
    justify-content: flex-start;
}

.user-row {
    justify-content: flex-end;
}

/* 아바타 */
.avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    overflow: hidden;
    border: 1px solid #e5e7eb;
    background: #fff;
    flex: 0 0 40px;
}

.avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

.avatar.emoji {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}

/* 말풍선 */
.bubble {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 18px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: keep-all;
    overflow-wrap: break-word;
}

.bubble.bot {
    background: #f6f8fb;
    color: #111;
    border: 1px solid #eef2f7;
    box-shadow: 0 2px 8px rgba(15,23,42,.08);
}

.bubble.user {
    background: #0b62e6;
    color: #fff;
    border: 0;
    box-shadow: 0 4px 12px rgba(11,98,230,.2);
}

/* 타임스탬프 */
.time {
    font-size: 11px;
    color: #6b7280;
    margin-top: 4px;
}

/* 타이핑 버블 */
.typing-bubble {
    position: relative;
    display: inline-flex;
    gap: 6px;
    align-items: center;
    background: #f6f8fb;
    color: #111;
    border: 1px solid #eef2f7;
    border-radius: 18px;
    padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(15,23,42,.08);
}

.typing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #a8b3c8;
    display: inline-block;
    animation: typingDot 1.2s infinite ease-in-out;
}

.typing-dot:nth-child(2) { animation-delay: .15s; }
.typing-dot:nth-child(3) { animation-delay: .3s; }

@keyframes typingDot {
    0%,80%,100% { transform: translateY(0); opacity: .5 }
    40% { transform: translateY(-4px); opacity: 1 }
}



/* 스트림릿 기본 스타일 제거 */
div[data-testid="stTextInput"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

div[data-testid="stTextInput"] input {
    border: 0 !important;
    flex: 1;
    padding: 12px 16px !important;
    font-size: 15px !important;
    background: transparent !important;
    border-radius: 16px !important;
}

div[data-testid="stTextInput"] input:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* 버튼 스타일 */
.stButton > button {
    border-radius: 50% !important;
    background: #0b62e6 !important;
    color: #fff !important;
    font-size: 18px !important;
    font-weight: 700;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: none !important;
    cursor: pointer;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #094fc0 !important;
    transform: scale(1.05);
}

/* 헤더 */
h1 {
    text-align: center;
    margin-bottom: 30px;
    color: #1f2937;
}

/* 채팅 영역 여백 */
.chat-area {
    margin-bottom: 20px;
}

/* 반응형 */
@media (max-width: 768px) {
    .bubble {
        max-width: 85%;
    }
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# 백엔드 서비스
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
# 세션 상태
# ------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "안녕하세요! 뉴스 Q&A 도우미입니다. 무엇이 궁금하신가요?",
        "ts": fmt_ts(datetime.now(TZ))
    }]

for k, v in {
    "is_generating": False,
    "to_process": False,
    "queued_q": "",
    "pending_idx": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------
# 메시지 렌더러
# ------------------------
def render_messages(msgs, placeholder):
    html_parts = []
    for m in msgs:
        role = m.get("role","assistant")
        ts   = _escape_html(m.get("ts",""))
        if role=="assistant":
            if m.get("pending"):
                html_parts.append(
                    "<div class='chat-row bot-row'>"
                    f"{_avatar_html('assistant')}"
                    "<div><div class='typing-bubble'>"
                    "<span class='typing-dot'></span>"
                    "<span class='typing-dot'></span>"
                    "<span class='typing-dot'></span>"
                    "</div>"
                    f"<div class='time'>{ts}</div></div></div>"
                )
            else:
                text=_linkify(_escape_html(m.get("content","")))
                html_parts.append(
                    "<div class='chat-row bot-row'>"
                    f"{_avatar_html('assistant')}"
                    f"<div><div class='bubble bot'>{text}</div>"
                    f"<div class='time'>{ts}</div></div></div>"
                )
        else: # user
            text=_linkify(_escape_html(m.get("content","")))
            html_parts.append(
                "<div class='chat-row user-row'>"
                f"<div><div class='bubble user'>{text}</div>"
                f"<div class='time' style='text-align:right'>{ts}</div></div>"
                f"{_avatar_html('user')}"
                "</div>"
            )
    placeholder.markdown("\n".join(html_parts), unsafe_allow_html=True)

# ------------------------
# 메인 UI
# ------------------------
st.markdown('<div class="main">', unsafe_allow_html=True)

# 헤더
st.title("🧙‍♂️ 우리 연금술사")

# 채팅 영역
st.markdown('<div class="chat-area">', unsafe_allow_html=True)
messages_ph = st.empty()
render_messages(st.session_state["messages"], messages_ph)
st.markdown('</div>', unsafe_allow_html=True)

# 입력창
if not st.session_state.get("is_generating", False):
    col1, col2 = st.columns([1, 0.15])
    
    with col1:
        user_q = st.text_input(
            "질문을 입력하세요...",
            key="user_input",
            label_visibility="collapsed",
            placeholder="예) 삼성전자 전망 알려줘"
        )
    
    with col2:
        clicked = st.button(
            "➤",
            key="send_button",
            use_container_width=True,
            disabled=st.session_state.get("is_generating", False)
        )

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------
# 메시지 처리
# ------------------------
final_q = (st.session_state.get("user_input", "") or "").strip()

if clicked and final_q and not st.session_state.get("is_generating", False):
    now = fmt_ts(datetime.now(TZ))
    st.session_state["messages"].append({"role": "user", "content": final_q, "ts": now})
    st.session_state["messages"].append({
        "role": "assistant", "content": "", "ts": now, "pending": True
    })
    st.session_state["pending_idx"] = len(st.session_state["messages"]) - 1
    st.session_state["queued_q"] = final_q
    st.session_state["is_generating"] = True
    st.session_state["to_process"] = True
    st.rerun()

if st.session_state.get("to_process", False):
    final_q = st.session_state.get("queued_q", "")
    pending_idx = st.session_state.get("pending_idx")
    sources, ans, result = [], "관련 정보를 찾을 수 없습니다.", {}
    try:
        if svc:
            result = svc.answer(final_q) or {}
            ans = (
                result.get("answer") or result.get("output_text") or
                result.get("output") or result.get("content") or ""
            ).strip() or ans
            sources = (
                result.get("source_documents") or
                result.get("sources") or
                result.get("docs") or []
            )
        else:
            ans = f"데모 응답: '{final_q}'에 대한 분석 결과는 준비 중입니다."
    except Exception as e:
        ans = f"오류 발생: {e}"

    st.session_state["messages"][pending_idx] = {
        "role": "assistant",
        "content": ans,
        "sources": sources,
        "ts": fmt_ts(datetime.now(TZ))
    }
    st.session_state["is_generating"] = False
    st.session_state["to_process"] = False
    st.session_state["queued_q"] = ""
    st.session_state["pending_idx"] = None
    st.rerun()
