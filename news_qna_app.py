import os, io, re
from typing import List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import streamlit as st

# 페이지 설정
st.set_page_config(page_title="우리 연금술사", page_icon="📰", layout="centered")

# 환경 변수 설정
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

TZ = ZoneInfo(os.getenv("APP_TZ", "Asia/Seoul"))

# CSS (스크롤바 스타일 추가)
st.markdown("""
<style>
/* 기존 CSS 유지, 스크롤바 스타일 추가 */
.screen-body::-webkit-scrollbar {
  width: 8px;
}
.screen-body::-webkit-scrollbar-track {
  background: #f0f4ff;
  border-radius: 8px;
}
.screen-body::-webkit-scrollbar-thumb {
  background: #c0c7d6;
  border-radius: 8px;
}
.screen-body::-webkit-scrollbar-thumb:hover {
  background: #a0a7b6;
}
.screen-body {
  scrollbar-width: thin;
  scrollbar-color: #c0c7d6 #f0f4ff;
}
</style>
""", unsafe_allow_html=True)

# 백엔드 서비스
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
        top_k=int(os.getenv("DEFAULT_TOP_K", "8")),
        use_rerank=False,
    )
svc = get_service()
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))

# Vertex AI 초기화
_vertex_inited = False
_embed_model = None
_gen_model = None

def _ensure_vertex_init():
    global _vertex_inited
    if not _vertex_inited:
        try:
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            if not project:
                st.error("GOOGLE_CLOUD_PROJECT is not set in environment variables.")
                return
            import vertexai
            vertexai.init(project=project, location=location)
            _vertex_inited = True
        except Exception as e:
            st.error(f"Failed to initialize Vertex AI: {e}")

# 세션 상태
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "안녕하세요! ✅ 연금/주식 뉴스를 근거로 QnA 도와드려요. 무엇이든 물어보세요.",
        "sources": [], "ts": format_timestamp(datetime.now(TZ))
    }]
if "temp_docs" not in st.session_state:
    st.session_state.temp_docs = []
if "_preset" not in st.session_state:
    st.session_state._preset = None

# 유틸리티 함수
def format_timestamp(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%Y년 %m월 %d일 %p %I:%M").replace("AM", "오전").replace("PM", "오후")

def _md(html: str): st.markdown(html, unsafe_allow_html=True)
def _escape_html(s: str) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def _linkify(s: str) -> str:
    return re.sub(r"(https?://[\\w\\-\\./%#\\?=&:+,~]+)", r'<a href="\\1" target="_blank">\\1</a>', s)
def _render_message(text: str, sender: str, ts: str):
    row = "user-row" if sender=="user" else "bot-row"
    bub = "user-bubble" if sender=="user" else "bot-bubble"
    safe = _linkify(_escape_html(text or ""))
    _md(f'<div class="chat-row {row}"><div class="chat-bubble {bub}">{safe}</div></div>')
    _md(f'<div class="timestamp {"ts-right" if sender=="user" else "ts-left"}">{ts}</div>')
def _render_sources_inline(sources: List[Dict[str,Any]]):
    if not sources: return
    chips = []
    for i, d in enumerate(sources, 1):
        m = d.get("metadata", {}) or {}
        title = m.get("title") or m.get("path") or m.get("source") or f"문서 {i}"
        url = m.get("url")
        score = float(d.get("score", 0.0))
        label = f"#{i} {title} · {score:.3f}"
        chip_html = f'<span class="source-chip"><a href="{url}" target="_blank">{label}</a></span>' if url else f'<span class="source-chip">{label}</span>'
        chips.append(chip_html)
    _md(f'<div class="src-row">{"".join(chips)}</div>')
def _copy_button(text: str, key: str):
    from streamlit.components.v1 import html as st_html
    safe = (text or "").replace("\\","\\\\").replace("`","\\`")
    st_html(f"""
<div class="action-bar">
  <button class="action-btn" id="copy-{key}" data-text="{safe}">📋 복사</button>
  <span class="small" id="copied-{key}" style="display:none;">복사됨!</span>
</div>
<script>
(function(){{
  const b=document.getElementById("copy-{key}"), t=document.getElementById("copied-{key}");
  if(!b) return;
  b.onclick=async()=>{{
    try{{ await navigator.clipboard.writeText(b.getAttribute("data-text")); t.style.display="inline-block"; setTimeout(()=>t.style.display="none",1200); }}
    catch(e){{ const ta=document.createElement('textarea'); ta.value=b.getAttribute("data-text"); document.body.appendChild(ta); ta.select(); document.execCommand('copy'); document.body.removeChild(ta); t.style.display="inline-block"; setTimeout(()=>t.style.display="none",1200); }}
  }};
}})();
</script>
""", height=30)

# 파일 업로드 처리
def _read_text_from_file(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    try:
        if name.endswith((".txt", ".md", ".csv")):
            return data.decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
            except ImportError:
                st.error("pypdf 모듈이 설치되지 않았습니다. PDF 파일을 처리할 수 없습니다.")
                return ""
        elif name.endswith(".docx"):
            try:
                import docx
                d = docx.Document(io.BytesIO(data))
                return "\n".join(p.text for p in d.paragraphs)
            except ImportError:
                st.error("python-docx 모듈이 설치되지 않았습니다. DOCX 파일을 처리할 수 없습니다.")
                return ""
        else:
            st.warning(f"지원되지 않는 파일 형식: {name}")
            return ""
    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {e}")
        return ""

# UI 렌더링
c1, c2 = st.columns([1.5, 0.16])
with c1: _md('<div class="chat-header"><div class="chat-title">🧙‍♂️ 우리 연금술사</div></div>')
with c2:
    if st.button("🔄", help="대화 초기화", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "대화를 새로 시작합니다. 무엇이 궁금하신가요?",
            "sources": [],
            "ts": format_timestamp(datetime.now(TZ))
        }]
        st.session_state._preset = None
        st.session_state.temp_docs = []
        st.rerun()

# 프리셋 & 업로더
cols = st.columns(3)
for i, label in enumerate(["우리금융지주 전망?", "호텔신라 실적 포인트?", "배당주 포트 제안"]):
    with cols[i]:
        if st.button(label, use_container_width=True):
            st.session_state._preset = label
st.divider()
uploaded_files = st.file_uploader("문서 업로드 (PDF, TXT, DOCX)", accept_multiple_files=True, type=["pdf", "txt", "md", "docx"])
if uploaded_files:
    added = add_uploaded_to_temp_index(uploaded_files)
    st.success(f"{added}개의 문서 조각이 임시 인덱스에 추가되었습니다.")
st.divider()

# 메시지 렌더링
st.markdown('<div class="screen-body">', unsafe_allow_html=True)
for i, m in enumerate(st.session_state.messages):
    _render_message(m["content"], m["role"], m.get("ts",""))
    if m["role"] == "assistant":
        _copy_button(m["content"], key=f"msg-{i}")
        if m.get("sources"):
            _render_sources_inline(m["sources"])

# 입력바
st.markdown('<div class="chat-dock"><div class="dock-wrap">', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([1, 0.18])
    user_q = c1.text_input("질문을 입력하세요...", key="custom_input", label_visibility="collapsed")
    submitted = c2.form_submit_button("➤", use_container_width=True)
st.markdown('</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 제출 처리
if submitted and user_q:
    run_answer(user_q)
elif st.session_state._preset:
    run_answer(st.session_state._preset)
    st.session_state._preset = None
