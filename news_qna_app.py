# app.py
import os, re
from typing import List, Dict, Any, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st

# =========================
# 페이지 설정 (최초 Streamlit 호출 전/초기에!)
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
    # 과도 이스케이프 수정 (\w, \? 등)
    return re.sub(r"(https?://[\w\-\./%#\?=&:+,~]+)", r'<a href="\1" target="_blank">\1</a>', s or "")

def _render_messages_block(messages: List[Dict[str, Any]]):
    # 메시지들을 **하나의 HTML 블록**으로 만들어 한 번만 렌더
    # (Streamlit이 element-container로 쪼개지 못하게 -> 내부 스크롤 정상 동작)
    parts = []
    for i, m in enumerate(messages):
        role = m.get("role", "assistant")
        row = "user-row" if role == "user" else "bot-row"
        bub = "user-bubble" if role == "user" else "bot-bubble"
        text_raw = m.get("content", "") or ""
        text = _linkify(_escape_html(text_raw))
        ts = _escape_html(m.get("ts", ""))

        # 말풍선 + 타임스탬프 + 복사 버튼
        parts.append(
            f'<div class="chat-row {row}"><div class="chat-bubble {bub}">{text}</div></div>'
            f'<div class="timestamp {"ts-right" if role=="user" else "ts-left"}">{ts}</div>'
            f'<div class="action-bar"><button class="action-btn copy-btn" '
            f'data-text="{_escape_html(text_raw)}">📋 복사</button></div>'
        )

        # 소스칩(assistant에만 표시)
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
                    if url:
                        chips.append(f'<span class="source-chip"><a href="{url}" target="_blank">{label}</a></span>')
                    else:
                        chips.append(f'<span class="source-chip">{label}</span>')
                parts.append(f'<div class="src-row">{"".join(chips)}</div>')

    html = (
        '<div class="screen-shell">'
        '<div class="screen-body">'
        + "".join(parts) +
        '</div></div>'
        # 복사 버튼 이벤트 위임(문서에 한 번만)
        '<script>(function(){'
        ' document.addEventListener("click", function(ev){'
        '   var b = ev.target.closest(".copy-btn"); if(!b) return;'
        '   var txt = b.getAttribute("data-text") || "";'
        '   var ta = document.createElement("textarea"); ta.value = txt;'
        '   document.body.appendChild(ta); ta.select(); try{document.execCommand("copy");}catch(e){};'
        '   document.body.removeChild(ta);'
        ' }, true);'
        '})();</script>'
    )
    st.markdown(html, unsafe_allow_html=True)

# =========================
# CSS
# =========================
st.markdown("""
<style>
:root{
  color-scheme: light !important;
  --brand:#0b62e6; --bezel:#0b0e17; --screen:#ffffff;
  --line:#e6ebf4; --chip:#eef4ff; --text:#1f2a44;
}
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
  overflow: hidden; /* 바깥은 숨기고, 내부에서 스크롤 */
}

/* 내부 스크롤 구조 */
.screen-shell{
  position: relative;
  height: 100%;
  display: flex;
  flex-direction: column;
}
/* :has 지원 브라우저에서 부모 element-container 높이 보장 */
.block-container > :first-child .element-container:has(.screen-shell){
  height: 100%;
}

.screen-body{
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  overflow-y: auto;            /* 여기서 스크롤 생성 */
  padding: 8px 10px 120px;
  padding-bottom: calc(120px + env(safe-area-inset-bottom, 0px));
  scroll-padding-bottom: 120px;
  scrollbar-width: thin; 
  scrollbar-color: #c0c7d6 #f0f4ff;
}
.screen-body::-webkit-scrollbar{ width:8px; }
.screen-body::-webkit-scrollbar-track{ background:#f0f4ff; border-radius:8px; }
.screen-body::-webkit-scrollbar-thumb{ background:#c0c7d6; border-radius:8px; }
.screen-body::-webkit-scrollbar-thumb:hover{ background:#a0a7b6; }
.screen-body{ overscroll-behavior: contain; }

.stChatInputContainer{ display:none !important; }
a{ color: var(--brand) !important; }
hr{ border:0; border-top:1px solid var(--line) !important; }
button, .stButton > button, .stDownloadButton > button{
  background: var(--chip) !important; border:1px solid #dce7ff !important; color:var(--brand) !important;
  border-radius:999px !important; font-weight:700 !important; padding:8px 14px !important; min-height:auto !important; line-height:1.1 !important;
}
.st-expander, .st-expander div[role="button"]{ background:#fff !important; border:1px solid var(--line) !important; color:var(--text) !important; }
.chat-header{ display:flex; align-items:center; justify-content:space-between; margin:8px 6px 12px; }
.chat-title{ font-size:20px; font-weight:900; color:var(--text); letter-spacing:.2px; }
.reset-btn > button{ width:38px; height:38px; border-radius:999px !important; background:var(--chip) !important; color:var(--brand) !important; border:1px solid #dce7ff !important; box-shadow:0 4px 12px rgba(23,87,255,.08); }
.chat-row{ display:flex; margin:12px 0; align-items:flex-end; }
.user-row{ justify-content:flex-end; }
.bot-row{ justify-content:flex-start; align-items:flex-start !important; }
.chat-bubble{
  max-width:86%; padding:14px 16px; border-radius:18px; line-height:1.65; font-size:16px; background:#ffffff; color:var(--text);
  border:1px solid var(--line); border-bottom-left-radius:8px; box-shadow:0 10px 22px rgba(15,23,42,.08); white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word;
}
.bot-row .chat-bubble{ position:relative; margin-left:54px; margin-top:2px; }
.bot-row .chat-bubble::before{
  content:"🧙‍♂️"; position:absolute; left:-54px; top:0; width:42px; height:42px; border-radius:999px; background:#fff; border:1px solid var(--line);
  display:flex; align-items:center; justify-content:center; font-size:20px; box-shadow:0 6px 14px rgba(15,23,42,.08);
}
.user-bubble{
  background:var(--brand) !important; color:#fff !important; border:0 !important; border-bottom-right-radius:8px;
  box-shadow:0 10px 28px rgba(11,98,230,.26); font-weight:700; letter-spacing:.2px; padding:16px 18px;
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

/* 입력 Dock: 절대 고정 (같은 큰 컨테이너 기준) */
.chat-dock{
  position:absolute !important; left:50% !important; bottom:16px !important; transform:translateX(-50%);
  width:92%; max-width:370px; z-index:20; filter: drop-shadow(0 10px 20px rgba(15,23,42,.18));
}
.chat-dock .dock-wrap{
  display:flex; gap:8px; align-items:center; background:#fff; border-radius:999px; padding:8px; border:1px solid #e6ebf4; box-shadow:0 8px 24px rgba(15,23,42,.10);
}
.chat-dock .stTextInput > div > div{ background:transparent !important; border:0 !important; padding:0 !important; }
.chat-dock input{ height:44px !important; padding:0 12px !important; font-size:15px !important; }
.chat-dock .send-btn > button{
  width:40px; height:40px; border-radius:999px !important; background:#e6efff !important; color:#0b62e6 !important; border:0 !important; box-shadow:inset 0 0 0 1px #d8e6ff; font-weight:800;
}

@media (max-width: 480px){
  .block-container > :first-child{ height: clamp(560px, 86vh, 820px); }
  .block-container{ max-width: 94vw; }
}
[data-testid="stHeader"]{ background:transparent !important; border:0 !important; }
.chat-dock:empty, .chat-dock .dock-wrap:empty{ display:none !important; }
.chat-dock .dock-wrap > *:not(form){ display:none !important; }
.chat-dock input{ background:#ffffff !important; color:#1f2a44 !important; }
</style>
""", unsafe_allow_html=True)

# :has 미지원 브라우저 폴백 (부모 element-container 높이 100%)
st.markdown("""
<script>
(function(){
  document.querySelectorAll('.screen-shell').forEach(function(shell){
    var parent = shell.closest('.element-container') || shell.parentElement;
    if (parent && (getComputedStyle(parent).height === 'auto' || !parent.style.height)) {
      parent.style.height = '100%';
    }
  });
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
            top_k=int(os.getenv("DEFAULT_TOP_K", "8")),
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

def generate_with_context(question: str, main_sources: List[Dict[str,Any]]) -> str:
    def snip(t, n=1800): return re.sub(r"\s+"," ",t or "")[:n]
    ctx = "\n\n".join([snip(d.get("content","")) for d in main_sources])[:10000]
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
        resp = model.generate_content(prompt, generation_config=GenerationConfig(temperature=0.2, max_output_tokens=1024))
        return (getattr(resp, "text", None) or "").strip() or "관련된 정보를 찾을 수 없습니다."
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

# =========================
# 세션 상태
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "안녕하세요! ✅ 연금/주식 뉴스를 근거로 QnA 도와드려요. 무엇이든 물어보세요.",
        "sources": [],
        "ts": format_timestamp(datetime.now(TZ))
    }]
if "_preset" not in st.session_state:
    st.session_state._preset = None

# =========================
# 헤더/프리셋
# =========================
head_l, head_r = st.columns([1.5, 0.16])
with head_l:
    st.markdown('<div class="chat-header"><div class="chat-title">🧙‍♂️ 우리 연금술사</div></div>', unsafe_allow_html=True)
with head_r:
    if st.button("🔄", help="대화 초기화", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "대화를 새로 시작합니다. 무엇이 궁금하신가요?",
            "sources": [],
            "ts": format_timestamp(datetime.now(TZ))
        }]
        st.session_state._preset = None
        st.rerun()

cols = st.columns(3)
for i, label in enumerate(["우리금융지주 전망?", "호텔신라 실적 포인트?", "배당주 포트 제안"]):
    with cols[i]:
        if st.button(label, use_container_width=True):
            st.session_state._preset = label
st.divider()

# =========================
# 메시지 영역 (단일 블록 렌더) + 입력 Dock
# =========================
_render_messages_block(st.session_state.messages)

# Dock (폼)
st.markdown('<div class="chat-dock"><div class="dock-wrap">', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([1, 0.18])
    user_q = c1.text_input("질문을 입력하세요...", key="custom_input", label_visibility="collapsed")
    submitted = c2.form_submit_button("➤", use_container_width=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# =========================
# 제출 처리
# =========================
def run_answer(question: str):
    now = format_timestamp(datetime.now(TZ))
    st.session_state.messages.append({"role":"user","content":question,"sources":[], "ts":now})

    with st.spinner("검색/생성 중…"):
        main = {}
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

    now2 = format_timestamp(datetime.now(TZ))
    st.session_state.messages.append({"role":"assistant","content":answer,"sources":main_sources,"ts":now2})
    # Streamlit은 submit 후 전체 재실행하므로, 위에 있는 단일 블록 렌더가 최신 messages를 표시함.

if 'submitted' in locals() and submitted and user_q:
    run_answer(user_q)
elif st.session_state._preset:
    run_answer(st.session_state._preset)
    st.session_state._preset = None
