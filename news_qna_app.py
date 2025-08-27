# app.py
import os, re
from typing import List, Dict, Any, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st

# =========================
# 페이지 설정
# =========================
st.set_page_config(page_title="우리 연금술사", page_icon="📰", layout="centered")

# =========================
# ENV from st.secrets → os.environ
# =========================
def _prime_env_from_secrets():
    try:
        if hasattr(st, 'secrets') and st.secrets:
            for k, v in st.secrets.items():
                os.environ.setdefault(k, str(v))
        else:
            st.warning("No secrets found in st.secrets. Ensure secrets are properly configured.")
    except FileNotFoundError:
        st.error("Secrets file not found. Please check your Streamlit configuration.")
    except Exception as e:
        st.error(f"Error loading secrets: {e}")

_prime_env_from_secrets()

# =========================
# 기본 유틸
# =========================
TZ = ZoneInfo(os.getenv("APP_TZ", "Asia/Seoul"))

def format_timestamp(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%Y년 %m월 %d일 %p %I:%M").replace("AM", "오전").replace("PM", "오후")

def _escape_html(s: Optional[str]) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _linkify(s: str) -> str:
    return re.sub(r"(https?://[\w\-\./%#\?=&:+,~]+)", r'<a href="\1" target="_blank">\1</a>', s or "")

def _build_messages_html(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for i, m in enumerate(messages):
        role = m.get("role", "assistant")
        row  = "user-row" if role == "user" else "bot-row"
        bub  = "user-bubble" if role == "user" else "bot-bubble"
        text_raw = m.get("content", "") or ""
        ts   = _escape_html(m.get("ts", ""))

        # 생성 중(typing) 말풍선
        if m.get("pending"):
            bubble = (
                '<div class="typing-bubble">'
                '<span class="typing-dot"></span>'
                '<span class="typing-dot"></span>'
                '<span class="typing-dot"></span>'
                '</div>'
            )
            parts.append(
                f'<div class="chat-row bot-row">{bubble}</div>'
                f'<div class="timestamp ts-left">{ts}</div>'
            )
        else:
            text = _linkify(_escape_html(text_raw))
            parts.append(
                f'<div class="chat-row {row}"><div class="chat-bubble {bub}">{text}</div></div>'
                f'<div class="timestamp {"ts-right" if role=="user" else "ts-left"}">{ts}</div>'
            )
            # 소스칩 (assistant에만)
            if role == "assistant":
                srcs = m.get("sources") or []
                if srcs:
                    chips = []
                    for j, d in enumerate(srcs, 1):
                        md = (d.get("metadata") or {}) if isinstance(d, dict) else {}
                        title = md.get("title") or md.get("path") or md.get("source") or f"문서 {j}"
                        url   = md.get("url")
                        try:
                            score = float(d.get("score", 0.0) or 0.0)
                        except:
                            score = 0.0
                        label = f"#{j} {title} · {score:.3f}"
                        if url:
                            link_html = f'<a href="{url}" target="_blank">{label}</a>'
                        else:
                            link_html = label
                        chips.append(f'<span class="source-chip">{link_html}</span>')
                    parts.append(f'<div class="src-row">{"".join(chips)}</div>')

    return (
        '<div class="screen-shell">'
        '<div class="screen-body" id="screen-body">'
        + "".join(parts) +
        '<div class="screen-spacer"></div>'
        '<div id="end-anchor"></div>'
        '</div></div>'
        '<script>(function(){'
        ' try {'
        '   var end = document.getElementById("end-anchor");'
        '   if (end) end.scrollIntoView({behavior:"instant", block:"end"});'
        ' } catch(e){}'
        '})();</script>'
    )

# =========================
# CSS / JS (생략 — 기존 그대로 유지)
# =========================
# (여기 CSS 블록과 fit() JS 블록 붙여 넣기)

# =========================
# 백엔드 서비스 (NewsQnAService) / Vertex 초기화
# =========================
# (여기 부분은 기존 코드 그대로 유지)

# =========================
# 세션 상태 초기화
# =========================
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "안녕하세요! ✅ 연금/주식 뉴스를 근거로 QnA 도와드려요. 무엇이든 물어보세요.",
        "sources": [],
        "ts": format_timestamp(datetime.now(TZ))
    }]

if "_preset" not in st.session_state:
    st.session_state["_preset"] = None

# =========================
# 헤더/프리셋
# =========================
head_l, head_r = st.columns([1.5, 0.16])
with head_l:
    st.markdown('<div class="chat-header"><div class="chat-title">🧙‍♂️ 우리 연금술사</div></div>', unsafe_allow_html=True)
with head_r:
    if st.button("🔄", help="대화 초기화", use_container_width=True):
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": "대화를 새로 시작합니다. 무엇이 궁금하신가요?",
            "sources": [],
            "ts": format_timestamp(datetime.now(TZ))
        }]
        st.session_state["_preset"] = None
        st.rerun()

cols = st.columns(3)
for i, label in enumerate(["우리금융지주 전망?", "호텔신라 실적 포인트?", "배당주 포트 제안"]):
    with cols[i]:
        if st.button(label, use_container_width=True):
            st.session_state["_preset"] = label
st.divider()

# =========================
# 메시지 영역 (단일 블록) + 입력 Dock
# =========================
ph_messages = st.empty()
ph_messages.markdown(
    _build_messages_html(st.session_state["messages"]),
    unsafe_allow_html=True
)

# Dock (폼)
st.markdown('<div class="chat-dock"><div class="dock-wrap">', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([1, 0.18])
    user_q = c1.text_input("질문을 입력하세요...", key="custom_input", label_visibility="collapsed")
    submitted = c2.form_submit_button("➤", use_container_width=True, type="primary")
st.markdown('</div></div>', unsafe_allow_html=True)

# =========================
# 제출 처리
# =========================
def run_answer(question: str):
    now = format_timestamp(datetime.now(TZ))
    st.session_state["messages"].append({"role":"user","content":question,"sources":[], "ts":now})

    # pending 버블 + 즉시 렌더
    now_p = format_timestamp(datetime.now(TZ))
    st.session_state["messages"].append({"role":"assistant","content":"", "sources":[], "ts":now_p, "pending": True})
    ph_messages.markdown(_build_messages_html(st.session_state["messages"]), unsafe_allow_html=True)

    # 생성
    with st.spinner("검색/생성 중…"):
        main = {}
        if svc is None:
            st.warning("백엔드 서비스가 초기화되지 않았습니다.")
        else:
            try:
                main = svc.answer(question) or {}
            except Exception as e:
                st.error(f"svc.answer 오류: {e}")
                main = {}
        main_sources = main.get("source_documents", []) or []
        answer = generate_with_context(question, main_sources)

    # pending 교체
    st.session_state["messages"][-1] = {
        "role": "assistant",
        "content": answer,
        "sources": main_sources,
        "ts": format_timestamp(datetime.now(TZ))
    }
    ph_messages.markdown(_build_messages_html(st.session_state["messages"]), unsafe_allow_html=True)

# =========================
# 실행
# =========================
if submitted and user_q:
    run_answer(user_q)
elif st.session_state["_preset"]:
    run_answer(st.session_state["_preset"])
    st.session_state["_preset"] = None
