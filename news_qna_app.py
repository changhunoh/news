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
ASSISTANT_AVATAR_URL = os.getenv("ASSISTANT_AVATAR_URL", "")  # 예: https://.../wizard.png
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
# CSS (말풍선+아바타+타이핑 버블)
# ------------------------
st.markdown("""
<style>
.chat-row{ display:flex; gap:10px; margin:10px 0; align-items:flex-start; }
.bot-row { justify-content:flex-start; }
.user-row{ justify-content:flex-end;  }

/* 아바타 */
.avatar{ width:40px; height:40px; border-radius:999px; overflow:hidden;
         border:1px solid #e5e7eb; background:#fff; flex:0 0 40px; }
.avatar img{ width:100%; height:100%; object-fit:cover; display:block; }
.avatar.emoji{ display:flex; align-items:center; justify-content:center; font-size:22px; }

/* 말풍선 */
.bubble{ max-width: clamp(260px, 65vw, 720px);
         padding:10px 14px; border-radius:16px; line-height:1.6;
         white-space:pre-wrap; word-break:keep-all; overflow-wrap:break-word; }
.bubble.bot  { background:#f6f8fb; color:#111;
               border:1px solid #eef2f7;
               box-shadow:0 6px 16px rgba(15,23,42,.12), inset 0 1px 0 rgba(255,255,255,.65);}
.bubble.user { background:#0b62e6; color:#fff; border:0;
               box-shadow: 0 10px 24px rgba(11,98,230,.28); }

/* 타임스탬프 */
.time{ font-size:11px; color:#6b7280; margin-top:4px; }

/* 타이핑 버블 */
.typing-bubble{
  position:relative;
  display:inline-flex; gap:6px; align-items:center;
  background:#f6f8fb; color:#111;
  border:1px solid #eef2f7; border-radius:16px; padding:10px 12px;
  box-shadow:0 6px 16px rgba(15,23,42,.12), inset 0 1px 0 rgba(255,255,255,.65);
}
.typing-dot{
  width:8px; height:8px; border-radius:50%; background:#a8b3c8; display:inline-block;
  animation: typingDot 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2){ animation-delay:.15s; }
.typing-dot:nth-child(3){ animation-delay:.3s; }
@keyframes typingDot{ 0%,80%,100%{transform:translateY(0);opacity:.5} 40%{transform:translateY(-4px);opacity:1} }

/*채팅창*/
.chat-dock{
  position: fixed;
  bottom: 16px; left: 50%; transform: translateX(-50%);
  width: 92%; max-width: 720px; z-index: 100;
}
.dock-wrap{
  display: flex; gap: 8px; align-items: center;
  background: #fff; border-radius: 999px;
  padding: 8px; border: 1px solid #e5e7eb;
  box-shadow: 0 6px 18px rgba(0,0,0,.08);
}
#chat_input {
  border:0 !important;
  flex: 1;
  padding: 12px 16px !important;
  font-size: 15px !important;
  background: transparent !important;
}
#chat_input:focus { outline:none !important; }
button[kind="secondaryFormSubmit"] {
  border-radius: 999px !important;
  background:#0b62e6 !important; color:#fff !important;
  font-size:18px !important; font-weight:700;
  width:42px; height:42px;
  display:flex; align-items:center; justify-content:center;
}
button[kind="secondaryFormSubmit"]:hover {
  background:#094fc0 !important;
}
/* text_input 기본 wrapper 제거 */
div[data-testid="stTextInput"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
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
        "content": "대화를 새로 시작합니다. 무엇이 궁금하신가요?",
        "ts": fmt_ts(datetime.now(TZ))
    }]

for k, v in {
    "chat_input": "",
    "is_generating": False,
    "to_process": False,     # 전송 직후 처리 플래그
    "queued_q": "",          # 전송된 질문 보관
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
# 헤더 + 메시지 영역
# ------------------------
st.title("🧙‍♂️ 우리 연금술사")
messages_ph = st.empty()

# 기존 메시지 표시
render_messages(st.session_state["messages"], messages_ph)

# ------------------------
# 입력 폼
# ------------------------
clicked = False
if not st.session_state.get("is_generating", False):
    st.markdown('<div class="chat-dock"><div class="dock-wrap">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 0.14])

    user_q = c1.text_input(
        "질문을 입력하세요...",
        key="chat_input",
        label_visibility="collapsed",
        placeholder="예) 삼성전자 전망 알려줘"
    )

    clicked = c2.button(
        "➤",
        use_container_width=True,
        disabled=st.session_state.get("is_generating", False)
    )
    st.markdown('</div></div>', unsafe_allow_html=True)

final_q = (st.session_state.get("chat_input", "") or "").strip()
if clicked and final_q and not st.session_state.get("is_generating", False):
    now = fmt_ts(datetime.now(TZ))
    st.session_state["messages"].append({"role": "user", "content": final_q, "ts": now})
    st.session_state["messages"].append({
        "role": "assistant", "content": "", "ts": now, "pending": True
    })
    st.session_state["pending_idx"] = len(st.session_state["messages"]) - 1
    st.session_state["queued_q"] = final_q
    st.session_state["chat_input"] = ""
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
