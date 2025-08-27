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
        if hasattr(st, "secrets") and st.secrets:
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
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

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
            continue

        # 일반 버블
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
                    url = md.get("url")
                    try:
                        score = float(d.get("score", 0.0) or 0.0)
                    except Exception:
                        score = 0.0
                    label = f"#{j} {title} · {score:.3f}"
                    link_html = f'<a href="{url}" target="_blank">{label}</a>' if url else label
                    chips.append(f'<span class="source-chip">{link_html}</span>')
                parts.append(f'<div class="src-row">{"".join(chips)}</div>')

    # 본문 + 스크롤 앵커/스페이서
    return (
        '<div class="screen-shell">'
        '<div class="screen-body" id="screen-body">'
        + "".join(parts) +
        '<div class="screen-spacer"></div>'
        '<div id="end-anchor"></div>'
        '</div></div>'
        '<script>(function(){'
        ' document.addEventListener("click", function(ev){'
        '   var b = ev.target.closest(".copy-btn"); if(!b) return;'
        '   var txt = b.getAttribute("data-text") || "";'
        '   var ta = document.createElement("textarea"); ta.value = txt;'
        '   document.body.appendChild(ta); ta.select(); try{document.execCommand("copy");}catch(e){};'
        '   document.body.removeChild(ta);'
        ' }, true);'
        ' try {'
        '   var end = document.getElementById("end-anchor");'
        '   if (end) end.scrollIntoView({behavior:"instant", block:"end"});'
        ' } catch(e){}'
        '})();</script>'
    )

# =========================
# CSS
# =========================
st.markdown("""
<style>
:root{
  color-scheme: light !important;
  --brand:#0b62e6; --bezel:#0b0e17; --screen:#ffffff;
  --line:#e6ebf4; --chip:#eef4ff; --text:#1f2a44;
  --dock-h: 140px; /* 입력 Dock 전체 높이(그림자 포함) */
}
html, body, [data-testid="stAppViewContainer"]{ height: 100%; }
html, body, [data-testid="stAppViewContainer"], section.main, .stMain, [data-testid="stSidebar"]{
  background: radial-gradient(1200px 700px at 50% 0, #f0f4ff 0%, #f6f8fb 45%, #eef1f6 100%) !important;
  color: var(--text) !important;
}
.block-container > :first-child{
  position: relative !important;
  height: clamp(620px, 82vh, 860px);
  background: var(--screen) !important;
  border: 1px solid var(--line) !important;
  border-radius: 30px !important;
  padding: 12px 14px 14px !important;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.65);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.block-container > :first-child > div{
  display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0;
}
.screen-shell{ position: relative; display:flex; flex-direction:column; flex:1 1 auto; min-height:0; }
.block-container > :first-child .element-container:has(.screen-shell){ height: 100%; }
.screen-body{
  flex: 1 1 auto; min-height: 0; overflow-y: auto; touch-action: pan-y; -webkit-overflow-scrolling: touch;
  padding: 8px 10px 12px; scrollbar-width: thin; scrollbar-color: #c0c7d6 #f0f4ff;
}
.screen-body::-webkit-scrollbar{ width:8px; }
.screen-body::-webkit-scrollbar-track{ background:#f0f4ff; border-radius:8px; }
.screen-body::-webkit-scrollbar-thumb{ background:#c0c7d6; border-radius:8px; }
.screen-body::-webkit-scrollbar-thumb:hover{ background:#a0a7b6; }
.screen-body{ overscroll-behavior: contain; }
.screen-spacer{ flex: 0 0 var(--dock-h); height: var(--dock-h); }
.stChatInputContainer{ display:none !important; }
a{ color: var(--brand) !important; }
hr{ border:0; border-top:1px solid var(--line) !important; }
button, .stButton > button, .stDownloadButton > button{
  background: var(--chip) !important; border:1px solid #dce7ff !important; color:var(--brand) !important;
  border-radius:999px !important; font-weight:700 !important; padding:8px 14px !important;
  min-height:auto !important; line-height:1.1 !important;
}
.st-expander, .st-expander div[role="button"]{
  background:#fff !important; border:1px solid var(--line) !important; color:var(--text) !important;
}
.chat-header{ display:flex; align-items:center; justify-content:space-between; margin:8px 6px 12px; }
.chat-title{ font-size:20px; font-weight:900; color:var(--text); letter-spacing:.2px; }
.reset-btn > button{
  width:38px; height:38px; border-radius:999px !important;
  background:var(--chip) !important; color:var(--brand) !important;
  border:1px solid #dce7ff !important; box-shadow:0 4px 12px rgba(23,87,255,.08);
}
.chat-row{ display:flex; margin:12px 0; align-items:flex-end; }
.user-row{ justify-content:flex-end; }
.bot-row{ justify-content:flex-start; align-items:flex-start !important; }
.chat-bubble{
  max-width:86%; padding:14px 16px; border-radius:18px; line-height:1.65; font-size:16px;
  background:#ffffff; color:var(--text); border:1px solid var(--line); border-bottom-left-radius:8px;
  box-shadow:0 10px 22px rgba(15,23,42,.08); white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word;
}
.bot-row .chat-bubble{ position:relative; margin-left:54px; margin-top:2px; }
.bot-row .chat-bubble::before{
  content:"🧙‍♂️"; position:absolute; left:-54px; top:0; width:42px; height:42px; border-radius:999px;
  background:#fff; border:1px solid var(--line); display:flex; align-items:center; justify-content:center; font-size:20px;
  box-shadow:0 6px 14px rgba(15,23,42,.08);
}
.user-bubble{
  background:var(--brand) !important; color:#fff !important; border:0 !important;
  border-bottom-right-radius:8px; box-shadow:0 10px 28px rgba(11,98,230,.26);
  font-weight:700; letter-spacing:.2px; padding:16px 18px;
}
.timestamp{ font-size:12px; color:#6b7280; margin:4px 6px; }
.ts-left{ text-align:left; } .ts-right{ text-align:right; }
.action-bar{ display:flex; gap:8px; margin:6px 6px 0; }
.action-btn{ font-size:12px; padding:6px 10px; border-radius:10px; border:1px solid #dce7ff; background:#eef4ff; color:var(--brand); }
.source-chip{
  display:inline-block; padding:4px 10px; border-radius:999px; background:#eef4ff; color:var(--brand);
  font-weight:800; font-size:12px; border:1px solid #dce7ff; margin:6px 6px 0 0;
}
.source-chip a{ color:var(--brand); text-decoration:none; }
.source-chip a:hover{ text-decoration:underline; }
.chat-dock{
  position:absolute !important; left:50% !important; bottom:16px !important; transform:translateX(-50%);
  width:92%; max-width:370px; z-index:30; filter: drop-shadow(0 10px 20px rgba(15,23,42,.18));
}
.chat-dock .dock-wrap{
  display:flex; gap:8px; align-items:center; background:#fff; border-radius:999px; padding:8px;
  border:1px solid #e6ebf4; box-shadow:0 8px 24px rgba(15,23,42,.10);
}
.chat-dock .stTextInput > div > div{ background:transparent !important; border:0 !important; padding:0 !important; }
.chat-dock input{ height:44px !important; padding:0 12px !important; font-size:15px !important; background:#ffffff !important; color:#1f2a44 !important; }
.chat-dock .send-btn > button{
  width:40px; height:40px; border-radius:999px !important; background:#e6efff !important; color:#0b62e6 !important;
  border:0 !important; box-shadow:inset 0 0 0 1px #d8e6ff; font-weight:800;
}
@media (max-width: 480px){
  .block-container > :first-child{ height: clamp(560px, 86vh, 820px); }
  .block-container{ max-width: 94vw; }
}
[data-testid="stHeader"]{ background:transparent !important; border:0 !important; }
.chat-dock:empty, .chat-dock .dock-wrap:empty{ display:none !important; }
.chat-dock .dock-wrap > *:not(form){ display:none !important; }
.typing-bubble{
  max-width:86%; padding:14px 16px; border-radius:18px; background:#ffffff; color:var(--text);
  border:1px solid var(--line); border-bottom-left-radius:8px; box-shadow:0 10px 22px rgba(15,23,42,.08);
  display:inline-flex; gap:6px; align-items:center;
}
.typing-dot{
  width:8px; height:8px; border-radius:50%; background:#a8b3c8; display:inline-block;
  animation: typingDot 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2){ animation-delay: .15s; }
.typing-dot:nth-child(3){ animation-delay: .3s; }
@keyframes typingDot{
  0%, 80%, 100% { transform: translateY(0); opacity:.5; }
  40% { transform: translateY(-4px); opacity:1; }
}
</style>
""", unsafe_allow_html=True)

# =========================
# JS: 높이 보정
# =========================
st.markdown("""
<script>
(function(){
  function fit(){
    const card = document.querySelector('.block-container > :first-child');
    const body = document.getElementById('screen-body') || document.querySelector('.screen-body');
    const dock = document.querySelector('.chat-dock');
    if(!card || !body) return;
    const cardRect = card.getBoundingClientRect();
    const bodyRect = body.getBoundingClientRect();
    const topInside = bodyRect.top - cardRect.top;
    const dockH = (dock ? dock.offsetHeight : 0) + 16; // 아래 여유
    const targetH = card.clientHeight - topInside - dockH;
    if (targetH > 120) {
      body.style.height = targetH + 'px';
      body.style.overflowY = 'auto';
    }
  }
  window.addEventListener('load', fit);
  window.addEventListener('resize', fit);
  const ro = new ResizeObserver(fit);
  ro.observe(document.body);
  setTimeout(fit, 50); setTimeout(fit, 200); setTimeout(fit, 600);
})();
</script>
""", unsafe_allow_html=True)

# =========================
# 백엔드 서비스 (선택)
# =========================
try:
    from news_qna_service import NewsQnAService
except Exception as e:
    NewsQnAService = None
    st.error(f"[임포트 오류] news_qna_service 모듈을 불러오지 못했습니다: {e}")

@st.cache_resource
def get_service():
    if NewsQnAService is None:
        return None
    try:
        return NewsQnAService(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_key=os.getenv("QDRANT_API_KEY"),
            collection=os.getenv("COLLECTION_NAME", "stock_news"),
            embed_model_name=os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001"),
            gen_model_name=os.getenv("GENAI_MODEL_NAME", "gemini-2.5-pro"),
            embed_dim=int(os.getenv("EMBED_DIM", "3072")),
            top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
            use_rerank=False,
        )
    except Exception as e:
        st.error(f"NewsQnAService 초기화 실패: {e}")
        return None

svc = get_service()

# =========================
# Vertex AI (생성모델만)
# =========================
_vertex_inited = False
_gen_model = None

def _ensure_vertex_init() -> bool:
    global _vertex_inited
    if _vertex_inited:
        return True
    try:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            st.error("GOOGLE_CLOUD_PROJECT is not set in environment variables.")
            return False
        import vertexai
        vertexai.init(project=project, location=location)
        _vertex_inited = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize Vertex AI: {e}")
        return False

def _get_gen_model():
    global _gen_model
    if _gen_model is None:
        if not _ensure_vertex_init():
            return None
        try:
            from vertexai.generative_models import GenerativeModel
            _gen_model = GenerativeModel(os.getenv("GENAI_MODEL_NAME", "gemini-2.5-pro"))
        except Exception as e:
            st.error(f"생성 모델 로딩 실패: {e}")
            return None
    return _gen_model

def generate_with_context(question: str, main_sources: List[Dict[str, Any]]) -> str:
    def snip(t, n=1800): 
        return re.sub(r"\s+", " ", t or "")[:n]
    ctx = "\n\n".join([snip(d.get("content", "")) for d in main_sources])[:10000]
    sys = (
        "당신은 주식/연금 뉴스를 바탕으로 답하는 분석가입니다. "
        "컨텍스트 근거로 한국어로 정확하게 답하세요. "
        "근거가 부족하면 추정하지 말고 '관련된 정보를 찾을 수 없습니다.'라고 답하세요. "
        "핵심은 **굵게** 강조하세요."
    )
    prompt = f"{sys}\n\n[컨텍스트]\n{ctx}\n\n[질문]\n{question}"

    model = _get_gen_model()
    if model is None:
        return "생성 모델 초기화에 실패했습니다. 환경 변수와 Vertex 설정을 확인해 주세요."
    try:
        from vertexai.generative_models import GenerationConfig
        resp = model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=0.2)
        )
        return (getattr(resp, "text", None) or "").strip() or "관련된 정보를 찾을 수 없습니다."
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

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
# 메시지 영역 + placeholder
# =========================
ph_messages = st.empty()
ph_messages.markdown(_build_messages_html(st.session_state["messages"]), unsafe_allow_html=True)

# =========================
# 입력 Dock (폼)
# =========================
st.markdown('<div class="chat-dock"><div class="dock-wrap">', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([1, 0.18])
    user_q = c1.text_input("질문을 입력하세요...", key="custom_input", label_visibility="collapsed")
    submitted = c2.form_submit_button("➤", use_container_width=True, type="primary")
st.markdown('</div></div>', unsafe_allow_html=True)

# =========================
# 제출 처리 로직
# =========================
def run_answer(question: str):
    # 사용자 메시지 추가
    now = format_timestamp(datetime.now(TZ))
    st.session_state["messages"].append({"role": "user", "content": question, "sources": [], "ts": now})

    # pending 버블 + 즉시 렌더
    now_p = format_timestamp(datetime.now(TZ))
    st.session_state["messages"].append({"role": "assistant", "content": "", "sources": [], "ts": now_p, "pending": True})
    ph_messages.markdown(_build_messages_html(st.session_state["messages"]), unsafe_allow_html=True)

    # 생성
    with st.spinner("검색/생성 중…"):
        main: Dict[str, Any] = {}
        if svc is None:
            st.warning("백엔드 서비스가 초기화되지 않았습니다. news_qna_service 모듈/환경변수를 확인해 주세요.")
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
# 실행 트리거
# =========================
if submitted and user_q:
    run_answer(user_q)
elif st.session_state.get("_preset"):
    run_answer(st.session_state["_preset"])
    st.session_state["_preset"] = None
