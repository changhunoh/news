# app.py
import os, re
import streamlit as st
from typing import List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

# =========================
# Page & Secrets
# =========================
st.set_page_config(page_title="나의 퇴직연금 챗봇", page_icon="📰", layout="centered")

def _prime_env_from_secrets():
    try:
        for k, v in st.secrets.items():
            os.environ.setdefault(k, str(v))
    except Exception:
        pass
_prime_env_from_secrets()

TZ = ZoneInfo(os.getenv("APP_TZ", "Asia/Seoul"))

# =========================
# Backend
# =========================
from news_qna_service import NewsQnAService

@st.cache_resource
def get_service() -> NewsQnAService:
    return NewsQnAService(
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_key=os.getenv("QDRANT_API_KEY"),
        collection=os.getenv("COLLECTION_NAME", "stock_news"),
        embed_model_name=os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001"),
        gen_model_name=os.getenv("GENAI_MODEL_NAME", "gemini-2.5-pro"),
        embed_dim=int(os.getenv("EMBED_DIM", "3072")),
        top_k=int(os.getenv("DEFAULT_TOP_K", "6")),
        use_rerank=False,
    )
svc = get_service()

# =========================
# State
# =========================
# 메시지 스키마:
# {"role": "user"|"assistant", "content": str, "sources": List[dict], "ts": "YYYY-MM-DD HH:MM"}
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "안녕하세요! 📈 연금/주식 뉴스를 근거로 QnA 도와드려요.\n무엇이든 물어보세요.",
            "sources": [],
            "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),
        }
    ]

# =========================
# CSS (폰 프레임 + 버블 + 칩 + 아바타 + 타임스탬프)
# =========================
st.markdown("""
<style>
html, body, .block-container { background: #f6f8fb !important; }
[data-testid="stSidebar"], header[tabindex="0"] { display: none !important; }

/* 폰 프레임 */
.phone-wrap { display:flex; justify-content:center; margin:20px auto; max-width:560px; }
.phone {
  width:100%; background:#fff; border-radius:26px; overflow:hidden;
  border:1px solid #e7ebf3; box-shadow:0 10px 25px rgba(23,30,60,0.08);
  display:flex; flex-direction:column; height:80vh;
}
.phone-header {
  padding:14px; text-align:center; font-weight:800; font-size:20px;
  border-bottom:1px solid #eef2f7; color:#1f2a44;
  background:linear-gradient(180deg,#ffffff 0%,#fafcff 100%);
}
.phone-body { flex:1; overflow-y:auto; padding:16px; }

/* 채팅 행(아바타+버블) */
.chat-row { display:flex; align-items:flex-end; margin:10px 0; }
.row-left { flex-direction:row; }
.row-right { flex-direction:row-reverse; }

.avatar {
  width:34px; height:34px; min-width:34px;
  border-radius:999px; display:grid; place-items:center;
  font-size:18px; font-weight:700; color:#1f2a44; background:#eef2ff;
  border:1px solid #dee7ff; box-shadow:0 2px 6px rgba(23,30,60,0.06);
  margin:0 8px;
}
.avatar.user { background:#dbe7ff; color:#0b62e6; }
.avatar.bot  { background:#eef9ff; color:#0b62e6; }

.chat-bubble {
  max-width: 78%; padding:12px 14px; border-radius:18px;
  line-height:1.55; font-size:15px; word-break:break-word; position:relative;
}
.user-bubble {
  background:#0b62e6; color:white; border-bottom-right-radius:4px;
  box-shadow:0 6px 16px rgba(11,98,230,0.18);
}
.assistant-bubble {
  background:#f4f9ff; color:#1f2a44; border-bottom-left-radius:4px;
  border:1px solid #e1efff; box-shadow:0 6px 16px rgba(23,30,60,0.06);
}

/* 타임스탬프 */
.timestamp {
  font-size:11px; color:#6b7280; margin:2px 4px;
}
.ts-left { text-align:left; }
.ts-right { text-align:right; }

/* 출처칩 */
.source-chip {
  display:inline-block; padding:3px 10px; border-radius:999px;
  background:#eef4ff; color:#1757ff; font-weight:800; font-size:12px;
  border:1px solid #dce7ff; margin:6px 6px 0 0;
}
.source-chip a { color:#1757ff; text-decoration:none; }
.source-chip a:hover { text-decoration:underline; }
.src-row { margin-left:42px; margin-top:4px; } /* 아바타 폭만큼 들여쓰기 */
.small { font-size:12px; color:#64748b; }
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def _render_sources_inline(sources: List[Dict[str, Any]]):
    """답변 버블 바로 아래에 출처칩 + 스니펫(expander)"""
    if not sources:
        return
    st.markdown('<div class="src-row">', unsafe_allow_html=True)
    chips = []
    for i, d in enumerate(sources, start=1):
        meta = d.get("metadata", {}) or {}
        title = meta.get("title") or meta.get("path") or meta.get("source") or f"문서 {i}"
        url = meta.get("url")
        score = float(d.get("score", 0.0))
        label = f"#{i} {title} · {score:.3f}"
        if url:
            chip = f'<span class="source-chip"><a href="{url}" target="_blank" rel="noopener">{label}</a></span>'
        else:
            chip = f'<span class="source-chip">{label}</span>'
        chips.append(chip)
    st.markdown("".join(chips), unsafe_allow_html=True)

    with st.expander("원문 스니펫 보기"):
        for i, d in enumerate(sources, start=1):
            snippet = re.sub(r"\s+", " ", (d.get("content") or ""))[:500]
            meta = d.get("metadata", {}) or {}
            title = meta.get("title") or f"문서 {i}"
            url = meta.get("url")
            link = f" — [원문]({url})" if url else ""
            st.markdown(f"**#{i} {title}**{link}\n\n{snippet}{'…' if len(snippet)==500 else ''}")
            st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def _row(role: str, text: str, ts: str, sources: List[Dict[str, Any]] | None = None):
    """한 줄(아바타+버블+타임스탬프)을 렌더"""
    left = (role == "assistant")
    row_class = "row-left" if left else "row-right"
    bubble_class = "assistant-bubble" if left else "user-bubble"
    avatar_class = "bot" if left else "user"
    avatar_emoji = "🤖" if left else "🧑"
    ts_class = "ts-left" if left else "ts-right"

    st.markdown(f'<div class="chat-row {row_class}">', unsafe_allow_html=True)
    st.markdown(f'<div class="avatar {avatar_class}">{avatar_emoji}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble {bubble_class}">{text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="timestamp {ts_class}">{ts}</div>', unsafe_allow_html=True)

    if left and sources:
        _render_sources_inline(sources)

# =========================
# Header
# =========================
st.markdown('<div class="phone-wrap"><div class="phone">', unsafe_allow_html=True)
st.markdown('<div class="phone-header">💬 나의 퇴직연금 챗봇</div>', unsafe_allow_html=True)
st.markdown('<div class="phone-body">', unsafe_allow_html=True)

# =========================
# Render history
# =========================
for m in st.session_state.messages:
    _row(m["role"], m["content"], m.get("ts",""), m.get("sources"))

# =========================
# Input & Answer
# =========================
prompt = st.chat_input("질문을 입력하세요… (예: 우리금융지주 전망은?)")

if prompt:
    now_ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")

    # 1) 사용자 메시지 즉시 렌더 + 저장
    st.session_state.messages.append({
        "role":"user", "content":prompt, "sources": [], "ts": now_ts
    })
    _row("user", prompt, now_ts, None)

    # 2) 답변 생성
    with st.spinner("검색/생성 중…"):
        result = svc.answer(prompt)
    answer = (result.get("answer") or "").strip()
    sources: List[Dict[str, Any]] = result.get("source_documents", []) or []
    if not answer:
        answer = "답변 생성 중 문제가 발생했습니다. 질문을 더 구체적으로 입력해 주세요."

    # 3) 답변 렌더 + 저장
    now_ts2 = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    _row("assistant", answer, now_ts2, sources)
    st.session_state.messages.append({
        "role":"assistant", "content":answer, "sources":sources, "ts": now_ts2
    })

# =========================
# Tail
# =========================
st.markdown('</div></div></div>', unsafe_allow_html=True)
