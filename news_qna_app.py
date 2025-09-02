# news_qna_app.py
import os, re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional
import streamlit as st
import time  # ← 스트리밍 효과 구현 용도 추가

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
    padding: 12px 24px 20px 24px;
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
    flex-direction: row-reverse; /* 아바타가 오른쪽으로 */
    justify-content: flex-start; /* 또는 제거해도 OK (기본값) */
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
    max-width: none;
    width: auto;
    padding: 16px 20px;
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
    margin-bottom: 0 0 24px 0;
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
.stApp {
    background: linear-gradient(180deg, #e0f7ff 0%, #ffffff 100%);
}
.main {
    max-width: 900px;
    width: 100%;
    margin: 0 auto;          /* 가운데 정렬 핵심 */
    padding: 12px 24px 20px 24px;
    background: transparent !important;
}
.thinking-text {
    font-weight: 600;
    margin-right: 6px;
    color: #334155;
}

/* 채팅창 메시지 영역 */
.chat-area {
    max-width: 700px;        /* 채팅창 폭 제한 */
    margin: 0 auto 20px auto; /* 가로 가운데 + 아래쪽 여백 */
}
/* 헤더 분리 추가 */
/* 헤더(제목) 전용 래퍼 */
.header-wrap {
  width: 100%;
  display: flex;
  justify-content: center;     /* 가로 중앙 */
  padding: 48px 0 12px;        /* 위여백 넉넉히 */
}

/* 제목 스타일 (h1 대신 커스텀) */
.app-title {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  font-size: 2.4rem;
  font-weight: 800;
  color: #1e293b;
  letter-spacing: -0.02em;
}

/* 제목과 채팅 사이 분리선(옵션) */
.section-sep {
  width: 100%;
  max-width: 900px;
  height: 1px;
  background: linear-gradient(90deg, rgba(30,41,59,0) 0%, rgba(148,163,184,.45) 50%, rgba(30,41,59,0) 100%);
  margin: 12px auto 28px;
}

/* 채팅 영역 전용 래퍼 */
.chat-wrap {
  width: 100%;
  max-width: 720px;            /* 채팅 폭 고정 */
  margin: 0 auto;              /* 가로 중앙 */
}

/* (기존) .main 은 레이아웃용 껍데기만 유지 */
.main {
  max-width: 900px;
  width: 100%;
  margin: 0 auto;
  padding: 0 24px 24px;        /* 헤더 padding은 header-wrap이 담당 */
  background: transparent !important;}
.stForm .stButton > button,
.stForm button[type="submit"] {
  border-radius: 50% !important;                   /* 원하면 50%로 동그랗게 */
  background: linear-gradient(135deg,#3b82f6, #1d4ed8) !important;
  color: #fff !important;
  border: none !important;
  box-shadow: 0 6px 16px rgba(59,130,246,.35) !important;
  font-weight: 700 !important;
}

.stForm .stButton > button:hover,
.stForm button[type="submit"]:hover {
  background: linear-gradient(135deg,#2563eb, #1e40af) !important;
  transform: translateY(-1px) scale(1.02);
  box-shadow: 0 10px 22px rgba(59,130,246,.45) !important;
}

.stForm .stButton > button:focus,
.stForm button[type="submit"]:focus {
  outline: none !important;
  box-shadow: 0 0 0 4px rgba(59,130,246,.18) !important;
}

/* 폼 제출 버튼을 파란색 채움으로 (Streamlit 버전별 모두 커버) */
.stForm .stFormSubmitButton > button,
.stForm [data-testid="baseButton-secondary"],
.stForm [data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
  color: #fff !important;
  border: none !important;
  box-shadow: 0 6px 16px rgba(59,130,246,.35) !important;
  font-weight: 700 !important;
}

/* 호버/포커스 상태 */
.stForm .stFormSubmitButton > button:hover,
.stForm [data-testid="baseButton-secondary"]:hover,
.stForm [data-testid="baseButton-primary"]:hover {
  background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
  transform: translateY(-1px) scale(1.02);
  box-shadow: 0 10px 22px rgba(59,130,246,.45) !important;
}
.stForm .stFormSubmitButton > button:focus,
.stForm [data-testid="baseButton-secondary"]:focus,
.stForm [data-testid="baseButton-primary"]:focus {
  outline: none !important;
  box-shadow: 0 0 0 4px rgba(59,130,246,.18) !important;
}

/* (선택) 동그란 액션버튼 스타일을 원하면 아래도 추가 */
.stForm .stFormSubmitButton > button,
.stForm [data-testid^="baseButton"] {
  height: 48px !important;
  width: 48px !important;
  border-radius: 50% !important;   /* 원형 */
  padding: 0 !important;
  font-size: 20px !important;
}
/* 폼 제출 버튼: 모든 버전 커버 */
.stForm .stFormSubmitButton button,
.stForm [data-testid="baseButton-secondary"],
.stForm [data-testid="baseButton-primary"],
.stForm [data-testid^="baseButton-"] {
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
  color: #fff !important;
  border: none !important;
  box-shadow: 0 6px 16px rgba(59,130,246,.35) !important;
  font-weight: 700 !important;
}

/* 호버/포커스 */
.stForm .stFormSubmitButton button:hover,
.stForm [data-testid^="baseButton-"]:hover {
  background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
  transform: translateY(-1px) scale(1.02);
  box-shadow: 0 10px 22px rgba(59,130,246,.45) !important;
}
.stForm .stFormSubmitButton button:focus,
.stForm [data-testid^="baseButton-"]:focus {
  outline: none !important;
  box-shadow: 0 0 0 4px rgba(59,130,246,.18) !important;
}

/* (원형 액션 버튼 유지) */
.stForm .stFormSubmitButton button,
.stForm [data-testid^="baseButton-"] {
  height: 48px !important;
  width: 48px !important;
  border-radius: 50% !important;
  padding: 0 !important;
  font-size: 20px !important;
}
.app-subtitle {
        font-size: 18px;   /* 제목보다 작은 글씨 */
        font-weight: normal;
        color: #666666;    /* 회색 톤, 필요하면 바꾸기 */
        margin-top: -8px;  /* 제목과 간격 줄이기 */
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
        "content": """안녕하세요! 연금술사입니다. 
퇴직연금 운용 상품에 대해 궁금한 점을 모두 물어봐주세요.

💡 예시 질문:
• 삼성전자 주가 전망이 어떻게 되나요?
• 최근 AI 관련 종목은 어떤 것들이 있나요?
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
            # 생각 중 텍스트 추가 전
            # if m.get("pending"):
            #     html_parts.append(
            #         "<div class='chat-row bot-row'>"
            #         f"{_avatar_html('assistant')}"
            #         "<div><div class='typing-bubble'>"
            #         "<span class='typing-dot'></span>"
            #         "<span class='typing-dot'></span>"
            #         "<span class='typing-dot'></span>"
            #         "</div>"
            #         f"<div class='time'>{ts}</div></div></div>"
            #     )
            
            # else:
            #     text=_linkify(_escape_html(m.get("content","")))
            #     html_parts.append(
            #         "<div class='chat-row bot-row'>"
            #         f"{_avatar_html('assistant')}"
            #         f"<div><div class='bubble bot'>{text}</div>"
            #         f"<div class='time'>{ts}</div></div></div>"
            #     )

            #if role=="assistant":
            # 생각 중 텍스트 추가
                if m.get("pending"):
                    html_parts.append(
                        "<div class='chat-row bot-row'>"
                        f"{_avatar_html('assistant')}"
                        "<div><div class='typing-bubble'>"
                        "<span class='thinking-text'>생각중</span>"
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

# --- 헤더(제목) ---
st.markdown('<div class="header-wrap">', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle-top">우리 연금술사</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-title">🔮 <span>연금술사의 수정구</span></div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# (옵션) 제목과 채팅 사이 분리선
st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

# --- 메인 래퍼 시작 ---
st.markdown('<div class="main">', unsafe_allow_html=True)

# --- 채팅 영역 ---
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
st.markdown('<div class="chat-area">', unsafe_allow_html=True)
messages_ph = st.empty()
render_messages(st.session_state["messages"], messages_ph)
st.markdown('</div>', unsafe_allow_html=True)  # .chat-area 닫기

# 입력창 (중앙 고정은 .chat-wrap이 담당)
# col1, col2 = st.columns([1, 0.15])
# with col1:
#     user_q = st.text_input(
#         "질문을 입력하세요...",
#         key=f"user_input_{st.session_state.get('input_key', 0)}",
#         label_visibility="collapsed",
#         placeholder="예) 삼성전자 전망 알려줘"
#     )
# with col2:
#     clicked = st.button(
#         "➤",
#         key="send_button",
#         use_container_width=True,
#         disabled=st.session_state.get("is_generating", False)
#     )

# st.markdown('</div>', unsafe_allow_html=True)  # .chat-wrap 닫기
# st.markdown('</div>', unsafe_allow_html=True)  # .main 닫기

# 입력창 (Enter 전송 가능: st.form 사용)
with st.form("ask_form", clear_on_submit=True):
    col1, col2 = st.columns([1, 0.15])
    with col1:
        user_q = st.text_input(
            "질문을 입력하세요...",
            key="user_input",                   # input_key 불필요
            label_visibility="collapsed",
            placeholder="예) 삼성전자 전망 알려줘"
        )
    with col2:
        submitted = st.form_submit_button(
            "➤",
            use_container_width=True,
            disabled=st.session_state.get("is_generating", False)
        )

# ↓↓↓ 이 두 줄 반드시 복구 (폼 바로 아래에 위치)
st.markdown('</div>', unsafe_allow_html=True)  # .chat-wrap 닫기
st.markdown('</div>', unsafe_allow_html=True)  # .main 닫기

# 헤더 분리 전
# ------------------------
# 메인 UI
# ------------------------
# st.markdown('<div class="main">', unsafe_allow_html=True)

# # 헤더
# st.title("🧙‍♂️ 우리 연금술사")

# # 채팅 영역
# st.markdown('<div class="chat-area">', unsafe_allow_html=True)
# messages_ph = st.empty()
# render_messages(st.session_state["messages"], messages_ph)
# st.markdown('</div>', unsafe_allow_html=True)

# # 입력창
# col1, col2 = st.columns([1, 0.15])

# with col1:
#     user_q = st.text_input(
#         "질문을 입력하세요...",
#         key=f"user_input_{st.session_state.get('input_key', 0)}",
#         label_visibility="collapsed",
#         placeholder="예) 삼성전자 전망 알려줘"
#     )

# with col2:
#     clicked = st.button(
#         "➤",
#         key="send_button",
#         use_container_width=True,
#         disabled=st.session_state.get("is_generating", False)
#     )


# with col1:
#     user_q = st.text_input(
#         "질문을 입력하세요...",
#         key=f"user_input_{st.session_state.get('input_key', 0)}",
#         label_visibility="collapsed",
#         placeholder="예) 삼성전자 전망 알려줘"
#     )

# with col2:
#     clicked = st.button(
#         "➤",
#         key="send_button",
#         use_container_width=True,
#         disabled=st.session_state.get("is_generating", False)
#     )

# if clicked and user_q:
#     # 사용자 메시지 표시
#     st.chat_message("user").write(user_q)

#     # 어시스턴트 메시지 + 스트리밍 출력
#     assistant_box = st.chat_message("assistant")
#     stream = service.answer_stream(user_q)   # ← 제너레이터 호출
#     assistant_box.write_stream(stream)       # ← 스트리밍 출력

#st.markdown('</div>', unsafe_allow_html=True)
# ------------------------
# 메시지 처리
# ------------------------
#current_input_key = f"user_input_{st.session_state.get('input_key', 0)}"
# final_q = (st.session_state.get(current_input_key, "") or "").strip()

# if clicked and final_q and not st.session_state.get("is_generating", False):
#     now = fmt_ts(datetime.now(TZ))
#     st.session_state["messages"].append({"role": "user", "content": final_q, "ts": now})
#     st.session_state["messages"].append({
#         "role": "assistant", "content": "", "ts": now, "pending": True
#     })
#     st.session_state["pending_idx"] = len(st.session_state["messages"]) - 1
#     st.session_state["queued_q"] = final_q
#     st.session_state["is_generating"] = True
#     st.session_state["to_process"] = True
#     st.session_state["input_key"] = st.session_state.get("input_key", 0) + 1
#     st.rerun()

# 메시지 처리
final_q = (user_q or "").strip()
if submitted and final_q and not st.session_state.get("is_generating", False):
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
    
# stream 효과 구현 용도 제거
# if st.session_state.get("to_process", False):
#     final_q = st.session_state.get("queued_q", "")
#     pending_idx = st.session_state.get("pending_idx")
#     sources, ans, result = [], "관련 정보를 찾을 수 없습니다.", {}
#     try:
#         if svc:
#             result = svc.answer(final_q) or {}
#             ans = (
#                 result.get("answer") or result.get("output_text") or
#                 result.get("output") or result.get("content") or ""
#             ).strip() or ans
#             sources = (
#                 result.get("source_documents") or
#                 result.get("sources") or
#                 result.get("docs") or []
#             )
#         else:
#             ans = f"데모 응답: '{final_q}'에 대한 분석 결과는 준비 중입니다."
#     except Exception as e:
#         ans = f"오류 발생: {e}"

# 스트리밍 함수 추가 (생각 중 추가 전)
# if st.session_state.get("to_process", False):
#     final_q = st.session_state.get("queued_q", "")
#     pending_idx = st.session_state.get("pending_idx")
#     sources = []

#     try:
#         if svc and hasattr(svc, "answer_stream"):
#             # 1) 타이핑 버블을 실제 스트림 메시지로 전환
#             st.session_state["messages"][pending_idx]["pending"] = False
#             st.session_state["messages"][pending_idx]["content"] = ""
#             st.session_state["messages"][pending_idx]["ts"] = fmt_ts(datetime.now(TZ))
#             render_messages(st.session_state["messages"], messages_ph)

#             # 2) 백엔드 스트림 소비
#             stream = svc.answer_stream(final_q)  # ← 핵심: 스트림 제너레이터 받기
#             buf = []
#             for chunk in stream:                 # ← 핵심: 제너레이터를 for로 '소비'
#                 if not isinstance(chunk, str):
#                     continue
#                 buf.append(chunk)
#                 st.session_state["messages"][pending_idx]["content"] = "".join(buf)
#                 render_messages(st.session_state["messages"], messages_ph)
#                 time.sleep(0.3)  # 프레임 드랍 방지, 체감 타자 효과

#             # 3) 스트림 완료 후, 근거 문서 부착(선택)
#             try:
#                 if hasattr(svc, "retrieve_only"):
#                     sources = svc.retrieve_only(final_q, top_k=5) or []
#             except Exception:
#                 sources = []
#             st.session_state["messages"][pending_idx]["sources"] = sources
#             st.session_state["messages"][pending_idx]["ts"] = fmt_ts(datetime.now(TZ))

#         else:
#             # 스트리밍 미지원/서비스 없음: 폴백
#             ans = f"데모 응답: '{final_q}'에 대한 분석 결과는 준비 중입니다."
#             st.session_state["messages"][pending_idx] = {
#                 "role": "assistant", "content": ans, "sources": [],
#                 "ts": fmt_ts(datetime.now(TZ))
#             }

#     except Exception as e:
#         st.session_state["messages"][pending_idx] = {
#             "role": "assistant", "content": f"오류 발생: {e}", "sources": [],
#             "ts": fmt_ts(datetime.now(TZ))
#         }

#     # 4) 상태 정리 및 최종 리렌더
#     st.session_state["is_generating"] = False
#     st.session_state["to_process"] = False
#     st.session_state["queued_q"] = ""
#     st.session_state["pending_idx"] = None
#     render_messages(st.session_state["messages"], messages_ph)
#     st.rerun()

# 생각 중 추가 + 스트리밍
if st.session_state.get("to_process", False):
    final_q = st.session_state.get("queued_q", "")
    pending_idx = st.session_state.get("pending_idx")
    sources = []

    try:
        if svc and hasattr(svc, "answer_stream"):
            # 1) 먼저 '생각중 …' 볼 수 있도록 pending 그대로 렌더
            st.session_state["messages"][pending_idx]["pending"] = True
            st.session_state["messages"][pending_idx]["content"] = ""
            st.session_state["messages"][pending_idx]["ts"] = fmt_ts(datetime.now(TZ))
            render_messages(st.session_state["messages"], messages_ph)

            # 2) 백엔드 스트림 소비
            stream = svc.answer_stream(final_q)
            buf = []
            got_first_chunk = False

            for chunk in stream:
                if not isinstance(chunk, str):
                    continue
                buf.append(chunk)

                # 첫 청크를 받는 순간 -> pending 해제하고 내용 표시 시작
                if not got_first_chunk:
                    got_first_chunk = True
                    st.session_state["messages"][pending_idx]["pending"] = False

                st.session_state["messages"][pending_idx]["content"] = "".join(buf)
                render_messages(st.session_state["messages"], messages_ph)
                time.sleep(0.1)  # 타자 효과

            # 3) 스트림 완료 후 근거 문서(선택)
            try:
                if hasattr(svc, "retrieve_only"):
                    sources = svc.retrieve_only(final_q, top_k=5) or []
            except Exception:
                sources = []
            st.session_state["messages"][pending_idx]["sources"] = sources
            st.session_state["messages"][pending_idx]["ts"] = fmt_ts(datetime.now(TZ))

            # 첫 청크가 하나도 오지 않았다면(에러/빈 응답) → 대체 메시지
            if not got_first_chunk:
                st.session_state["messages"][pending_idx]["pending"] = False
                st.session_state["messages"][pending_idx]["content"] = (
                    "관련 정보를 찾을 수 없습니다."
                )
                render_messages(st.session_state["messages"], messages_ph)

        else:
            # 스트리밍 미지원/서비스 없음: 폴백
            ans = f"데모 응답: '{final_q}'에 대한 분석 결과는 준비 중입니다."
            st.session_state["messages"][pending_idx] = {
                "role": "assistant", "content": ans, "sources": [],
                "ts": fmt_ts(datetime.now(TZ))
            }

    except Exception as e:
        st.session_state["messages"][pending_idx] = {
            "role": "assistant", "content": f"오류 발생: {e}", "sources": [],
            "ts": fmt_ts(datetime.now(TZ))
        }

    # 상태 정리 및 최종 리렌더
    st.session_state["is_generating"] = False
    st.session_state["to_process"] = False
    st.session_state["queued_q"] = ""
    st.session_state["pending_idx"] = None
    render_messages(st.session_state["messages"], messages_ph)
    st.rerun()



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
