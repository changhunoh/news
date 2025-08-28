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
    max-width: 900px;
    margin: 0 auto;
    padding: 24px;
}

/* 채팅 메시지 */
.chat-row {
    display: flex;
    margin: 20px 0;
    align-items: flex-start;
}

.bot-row {
    justify-content: flex-start;
    gap: 12px;
}

.user-row {
    justify-content: flex-end;
    gap: 8px;
}

/* 아바타 */
.avatar {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    overflow: hidden;
    border: 2px solid #ffffff;
    background: #fff;
    flex: 0 0 42px;
    box-shadow: 
        0 4px 12px rgba(0, 0, 0, 0.15),
        0 2px 4px rgba(0, 0, 0, 0.1);
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
    background: #f8fafc;
    color: #64748b;
    border: 2px solid #e2e8f0;
}

/* 말풍선 */
.bubble {
    max-width: 85%;
    padding: 14px 18px;
    border-radius: 20px;
    line-height: 2.0;
    white-space: pre-wrap;
    word-break: keep-all;
    overflow-wrap: break-word;
    position: relative;
    font-size: 15px;
}

/* 끊기면 안 되는 덩어리 전용 */
.no-break {
  white-space: nowrap;     /* 핵심! 한 줄로 유지 */
}

/* assistant색상 그라데이션 등 */

.bubble.bot {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    color: #1f2937;
    border: 1px solid #e2e8f0;
    box-shadow: 
        0 4px 12px rgba(0, 0, 0, 0.08),
        0 2px 4px rgba(0, 0, 0, 0.04);
}

.bubble.user {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: #fff;
    border: 0;
    box-shadow: 
        0 6px 16px rgba(59, 130, 246, 0.3),
        0 4px 8px rgba(59, 130, 246, 0.2);
}

/* 타임스탬프 */
.time {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 6px;
    font-weight: 500;
}

/* 타이핑 버블 */
.typing-bubble {
    position: relative;
    display: inline-flex;
    gap: 6px;
    align-items: center;
    background: #ffffff;
    color: #111;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 12px 16px;
    box-shadow: 
        0 2px 4px rgba(0, 0, 0, 0.1),
        0 4px 8px rgba(0, 0, 0, 0.06),
        0 8px 16px rgba(0, 0, 0, 0.04);
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
    border: 2px solid #e2e8f0 !important;
    flex: 1;
    padding: 14px 18px !important;
    font-size: 16px !important;
    background: #ffffff !important;
    border-radius: 20px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.3s ease;
}

div[data-testid="stTextInput"] input:focus {
    outline: none !important;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1) !important;
    border-color: #3b82f6 !important;
    transform: translateY(-1px);
}

/* 버튼 스타일 */
.stButton > button {
    border-radius: 50% !important;
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    color: #fff !important;
    font-size: 20px !important;
    font-weight: 700;
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: none !important;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
    transform: scale(1.1);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
}

/* 헤더 */
h1 {
    text-align: center;
    margin-bottom: 40px;
    color: #1e293b;
    font-size: 2.2rem;
    font-weight: 700;
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
        "content": """안녕하세요! 뉴스 Q&A 도우미입니다. 무엇이 궁금하신가요?

💡 예시 질문:
• 삼성전자 주가 전망이 어떻게 되나요?
• 최근 AI 관련 뉴스는 어떤 것들이 있나요?
• 반도체 시장 동향에 대해 알려주세요
• 특정 기업의 실적 발표 내용을 요약해주세요""",
        "ts": fmt_ts(datetime.now(TZ))
    }]

for k, v in {
    "is_generating": False,
    "to_process": False,
    "queued_q": "",
    "pending_idx": None,
    "input_key": 0,
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
col1, col2 = st.columns([1, 0.15])

with col1:
    user_q = st.text_input(
        "질문을 입력하세요...",
        key=f"user_input_{st.session_state.get('input_key', 0)}",
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
current_input_key = f"user_input_{st.session_state.get('input_key', 0)}"
final_q = (st.session_state.get(current_input_key, "") or "").strip()

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
    st.session_state["input_key"] = st.session_state.get("input_key", 0) + 1
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
