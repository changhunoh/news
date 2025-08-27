# app.py
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st

# ──────────────────────────────────────────────────────────────────
# 외부 RAG 서비스 (질문에 주신 코드가 들어있는 파일)
# ──────────────────────────────────────────────────────────────────
from news_qna_service import NewsQnAService

# ──────────────────────────────────────────────────────────────────
# 기본 셋업
# ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="우리 연금술사", page_icon="📰", layout="centered")
TZ = ZoneInfo(os.getenv("APP_TZ", "Asia/Seoul"))

def ts_now():
    return (
        datetime.now(TZ)
        .strftime("%Y년 %m월 %d일 %p %I:%M")
        .replace("AM", "오전")
        .replace("PM", "오후")
    )

# 라이트 모드 강제 (다크에서 색 깨짐 방지)
st.markdown('<meta name="color-scheme" content="light">', unsafe_allow_html=True)

# 헤더(겹침 방지: 레이아웃 내부에 넣기)
with st.container():
    st.markdown(
        """
        <div style="
            display:flex; align-items:center; gap:10px;
            padding:10px 6px 4px 6px; margin-bottom:4px;">
          <span style="font-size:22px;">🧙‍♂️</span>
          <div style="font-weight:700; font-size:22px;">우리 연금술사</div>
        </div>
        <hr style="margin:0 0 8px 0; border:0; border-top:1px solid #eee;">
        """,
        unsafe_allow_html=True,
    )

# 말풍선 색/여백 커스터마이즈
st.markdown(
    """
    <style>
      .stChatMessage {padding-top: 4px; padding-bottom: 4px;}
      /* assistant bubble */
      .stChatMessage[data-testid="stChatMessage"]:has(img[alt="assistant-avatar"]) .stMarkdown p {
        background:#F4F6F9; color:#111; border-radius:16px; 
        padding:12px 16px; margin:6px 0; border-top-left-radius:6px;
        box-shadow: 0 1px 2px rgba(0,0,0,.06);
      }
      /* user bubble */
      .stChatMessage[data-testid="stChatMessage"]:has(img[alt="user-avatar"]) .stMarkdown p {
        background:#0b46ff; color:#fff; border-radius:16px; 
        padding:12px 16px; margin:6px 0; border-top-right-radius:6px;
        box-shadow: 0 1px 2px rgba(0,0,0,.08);
      }
      /* timestamp */
      .bubble-ts {
        font-size:11px; color:#8b8b8b; margin-top:2px;
      }
      /* chat_input 박스 넓이/여백 */
      .stChatInput { padding-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────
# 세션 상태
# ──────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "대화를 새로 시작합니다. 무엇이 궁금하신가요?", "ts": ts_now()}
    ]

if "rag" not in st.session_state:
    try:
        # 환경변수/Streamlit secrets는 NewsQnAService 내부에서 사용
        st.session_state.rag = NewsQnAService()
    except Exception as e:
        st.session_state.rag = None
        st.warning(f"RAG 초기화 오류: {e}\n\nDemo 모드로 동작합니다.")

rag = st.session_state.rag

# 아바타(이모지 or 이미지 URL 사용 가능)
ASSISTANT_AVATAR = "🧙‍♂️"
USER_AVATAR = "🧑‍💼"

st.markdown(
    """
    <style>
      .bubble { padding:12px 16px; border-radius:16px; margin:6px 0; max-width:80%;
                box-shadow:0 1px 2px rgba(0,0,0,.06); word-wrap:break-word; font-size:15px; }
      .bubble.assistant { background:#F4F6F9; color:#111; border-top-left-radius:6px; }
      .bubble.user      { background:#0b46ff; color:#fff; border-top-right-radius:6px; }
      .bubble-ts { font-size:11px; color:#8b8b8b; margin-top:2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

def render_assistant(text, ts=None, sources=None):
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown(f'<div class="bubble assistant">{text}</div>', unsafe_allow_html=True)
        if ts: st.markdown(f'<div class="bubble-ts">{ts}</div>', unsafe_allow_html=True)
        if sources:
            with st.expander("📰 근거 보기"):
                for i, d in enumerate(sources, 1):
                    meta = d.get("metadata", {})
                    title = meta.get("title") or meta.get("news_title") or meta.get("file_name") or "문서"
                    url = meta.get("url") or meta.get("link")
                    score = d.get("score")
                    st.markdown(f"**{i}. {title}**  \n- score: `{score:.4f}`" if score is not None else f"**{i}. {title}**")
                    if url: st.markdown(f"- 링크: {url}")
                    if meta:
                        keep = {k: meta[k] for k in list(meta)[:6]}
                        st.code(keep, language="json")

def render_user(text, ts=None):
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(f'<div class="bubble user">{text}</div>', unsafe_allow_html=True)
        if ts: st.markdown(f'<div class="bubble-ts">{ts}</div>', unsafe_allow_html=True)

# 메시지 출력
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        render_assistant(msg["content"], ts=msg.get("ts"), sources=msg.get("sources"))
    else:
        render_user(msg["content"], ts=msg.get("ts"))

# 입력 처리
prompt = st.chat_input("질문을 입력하세요…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "ts": ts_now()})
    with st.spinner("답변 생성 중…"):
        if rag is not None:
            result = rag.answer(prompt)
            answer = result.get("answer", "관련된 정보를 찾을 수 없습니다.")
            sources = result.get("source_documents", [])
        else:
            answer = "데모 모드 응답입니다."
            sources = []
    st.session_state.messages.append({"role": "assistant", "content": answer, "ts": ts_now(), "sources": sources})
    st.rerun()
