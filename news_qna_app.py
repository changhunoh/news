# app.py
import os, io, re
from typing import List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import streamlit as st

# =========================
# Page config & Secrets → ENV
# =========================
st.set_page_config(page_title="우리 연금술사", page_icon="📰", layout="centered")

def _prime_env_from_secrets():
    try:
        for k, v in st.secrets.items():
            os.environ.setdefault(k, str(v))
    except Exception:
        pass
_prime_env_from_secrets()

TZ = ZoneInfo(os.getenv("APP_TZ", "Asia/Seoul"))

# 라이트 강제(브라우저/컨테이너 전역)
st.markdown("""
<style>
:root { color-scheme: light !important; }

/* 배경/글자 라이트 고정 */
html, body,
[data-testid="stAppViewContainer"], section.main, .stMain, .stBlock, .block-container,
[data-testid="stHeader"], [data-testid="stSidebar"] {
  background: #f6f8fb !important; color: #1f2a44 !important;
}

/* 고정 폭(폰 느낌) + 반응형 */
.block-container{ max-width:560px; margin-inline:auto; padding-top:10px; }
@media (max-width:640px){ .block-container{ max-width:94vw; } }

/* 버튼/칩/익스팬더 라이트 스킨 */
button, .stButton>button, .stDownloadButton>button {
  background:#eef4ff !important; border:1px solid #dce7ff !important;
  color:#1757ff !important; border-radius:999px !important; font-weight:700 !important;
  padding:8px 14px !important; min-height:auto !important; line-height:1.1 !important;
}
.st-expander, .st-expander div[role="button"]{
  background:#ffffff !important; border:1px solid #e6ebf4 !important; color:#1f2a44 !important;
}
a { color:#0b62e6 !important; }
hr{ border:0; border-top:1px solid #e6ebf4 !important; }

/* 입력창 */
.stChatInputContainer textarea, textarea, input, .stTextInput>div>div>input {
  background:#ffffff !important; color:#1f2a44 !important; border:1px solid #e6ebf4 !important;
  border-radius:12px !important; padding:12px !important; font-size:15px !important;
}

/* 헤더 - 단순화: 기본 렌더링 보장 */
.chat-header{ 
  display:flex; 
  align-items:center; 
  justify-content:space-between; 
  margin: 4px 2px 12px; 
}
.chat-title{ 
  font-size:20px; 
  font-weight:900; 
  color:#1f2a44; 
  white-space: nowrap;  /* 줄바꿈 방지, 필요 시 제거 */
  overflow: hidden;  /* 넘침 방지 */
  text-overflow: ellipsis;  /* 넘침 시 ... 표시 */
}
.reset-btn>button{
  width:38px; height:38px; border-radius:999px !important;
  background:#eef4ff !important; color:#1757ff !important; border:1px solid #dce7ff !important;
  box-shadow:0 4px 12px rgba(23,87,255,0.08);
}

/* 채팅 버블 */
.chat-row{ display:flex; margin:10px 0; }
.user-row{ justify-content:flex-end; }
.bot-row{ justify-content:flex-start; }
.chat-bubble{
  max-width:82%; padding:12px 14px; border-radius:18px; line-height:1.6; font-size:15px;
  background:#ffffff; color:#1f2a44; border:1px solid #e6ebf4; border-bottom-left-radius:6px;
  box-shadow:0 8px 20px rgba(15,23,42,0.08);
  white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word;
}
.user-bubble{
  background:#0b62e6 !important; color:#ffffff !important; border:0 !important;
  border-bottom-right-radius:6px; box-shadow:0 6px 18px rgba(11,98,230,0.18);
}

/* 타임스탬프 / 액션바 / 출처 칩 */
.timestamp{ font-size:12px; color:#6b7280; margin:4px 6px; }
.ts-left{text-align:left;} .ts-right{text-align:right;}
.action-bar{ display:flex; gap:8px; margin:6px 6px 0; }
.action-btn{
  font-size:12px; padding:6px 10px; border-radius:10px;
  border:1px solid #dce7ff; background:#eef4ff; color:#1757ff;
}
.source-chip{
  display:inline-block; padding:4px 10px; border-radius:999px;
  background:#eef4ff; color:#1757ff; font-weight:800; font-size:12px;
  border:1px solid #dce7ff; margin:6px 6px 0 0;
}
.source-chip a{ color:#1757ff; text-decoration:none; }
.source-chip a:hover{ text-decoration:underline; }
.src-row{ margin:4px 6px 0; }
</style>
""", unsafe_allow_html=True)
# =========================
# Backend service
# =========================
from news_qna_service import NewsQnAService  # 리포에 존재해야 함

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

# =========================
# Lazy Vertex (업로드 임시 인덱스/생성 시 사용)
# =========================
_vertex_inited = False
_embed_model = None
_gen_model = None

def _ensure_vertex_init():
    global _vertex_inited
    if not _vertex_inited:
        try:
            import vertexai
            vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                          location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
        except Exception:
            pass
        _vertex_inited = True

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _ensure_vertex_init()
        from vertexai.language_models import TextEmbeddingModel
        _embed_model = TextEmbeddingModel.from_pretrained(os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001"))
    return _embed_model

def _get_gen_model():
    global _gen_model
    if _gen_model is None:
        _ensure_vertex_init()
        from vertexai.generative_models import GenerativeModel
        _gen_model = GenerativeModel(os.getenv("GENAI_MODEL_NAME", "gemini-2.5-pro"))
    return _gen_model

# =========================
# Session state
# =========================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = [{
        "role": "assistant",
        "content": "안녕하세요! ✅ 연금/주식 뉴스를 근거로 QnA 도와드려요. 무엇이든 물어보세요.",
        "sources": [], "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    }]
if "temp_docs" not in st.session_state:
    st.session_state.temp_docs: List[Dict[str, Any]] = []
if "_preset" not in st.session_state:
    st.session_state._preset = None

# =========================
# Small utils
# =========================
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
    chips=[]
    for i, d in enumerate(sources, 1):
        m = d.get("metadata", {}) or {}
        title = m.get("title") or m.get("path") or m.get("source") or f"문서 {i}"
        url = m.get("url")
        score = float(d.get("score", 0.0))
        label = f"#{i} {title} · {score:.3f}"
        if url:
            chip_html = f'<span class="source-chip"><a href="{url}" target="_blank">{label}</a></span>'
        else:
            chip_html = f'<span class="source-chip">{label}</span>'
        chips.append(chip_html)
    _md(f'<div class="src-row">{"".join(chips)}</div>')
def _copy_button(text: str, key: str):
    from streamlit.components.v1 import html as st_html
    safe=(text or "").replace("\\","\\\\").replace("`","\\`")
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

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))

# =========================
# Upload → Temp Index (세션)
# =========================
def _read_text_from_file(uploaded) -> str:
    name=uploaded.name.lower(); data=uploaded.read()
    try:
        if name.endswith((".txt",".md",".csv")): return data.decode("utf-8",errors="ignore")
        if name.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
            except Exception: return ""
        if name.endswith(".docx"):
            try:
                import docx; d=docx.Document(io.BytesIO(data))
                return "\n".join(p.text for p in d.paragraphs)
            except Exception: return ""
    except Exception: return ""
    return ""

def _chunk(text: str, size=1200, overlap=150):
    out,s,n=[],0,len(text or "")
    while s<n:
        e=min(s+size,n); out.append(text[s:e])
        if e==n: break
        s=max(e-overlap,s+1)
    return out

def _embed_texts(texts: List[str]) -> List[np.ndarray]:
    from vertexai.language_models import TextEmbeddingInput
    model=_get_embed_model()
    embs=model.get_embeddings([TextEmbeddingInput(text=t,task_type="RETRIEVAL_DOCUMENT") for t in texts],
                              output_dimensionality=EMBED_DIM)
    return [np.array(e.values,dtype=np.float32) for e in embs]

def add_uploaded_to_temp_index(files) -> int:
    if not files: return 0
    added=0
    for f in files:
        raw=_read_text_from_file(f)
        if not raw: continue
        chunks=_chunk(raw); vecs=_embed_texts(chunks)
        for i,(seg,vec) in enumerate(zip(chunks,vecs)):
            st.session_state.temp_docs.append({
                "id": f"{f.name}:{i}",
                "content": seg,
                "metadata": {"title": f.name, "source": "upload"},
                "emb": vec
            }); added+=1
    return added

def search_temp_index(query: str, top_k=5) -> List[Dict[str,Any]]:
    if not st.session_state.temp_docs: return []
    from vertexai.language_models import TextEmbeddingInput
    model=_get_embed_model()
    qv=np.array(model.get_embeddings([TextEmbeddingInput(text=query,task_type="RETRIEVAL_QUERY")],
                                     output_dimensionality=EMBED_DIM)[0].values, dtype=np.float32)
    scored=[(_cosine(qv,d["emb"]),d) for d in st.session_state.temp_docs]
    scored.sort(key=lambda x:x[0], reverse=True)
    out=[]
    for s,d in scored[:top_k]:
        out.append({"id":d["id"],"content":d["content"],"metadata":d["metadata"],"score":float(s)})
    return out

# =========================
# Generation with merged context
# =========================
def generate_with_context(question: str,
                          main_sources: List[Dict[str,Any]],
                          extra_sources: List[Dict[str,Any]]) -> str:
    def snip(t,n=1800): return re.sub(r"\s+"," ",t or "")[:n]
    merged=(extra_sources or []) + (main_sources or [])
    ctx="\n\n".join([snip(d.get("content","")) for d in merged])[:10000]
    sys=(
        "당신은 주식/연금 뉴스를 바탕으로 답하는 분석가입니다. "
        "컨텍스트 근거로 한국어로 정확하게 답하세요. "
        "근거가 부족하면 추정하지 말고 '관련된 정보를 찾을 수 없습니다.'라고 답하세요. "
        "핵심은 **굵게** 강조하세요."
    )
    prompt=f"{sys}\n\n[컨텍스트]\n{ctx}\n\n[질문]\n{question}"
    try:
        model=_get_gen_model()
        resp=model.generate_content(prompt, generation_config={"temperature":0.2, "max_output_tokens":1024})
        return (resp.text or "").strip()
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

# =========================
# Header (제목 + 우측 회전 초기화)
# =========================
c1, c2 = st.columns([1.5, 0.16])
with c1: _md('<div class="chat-header"><div class="chat-title">🧙‍♂️ 우리 연금술사</div></div>')
with c2:
    if st.button("🔄", help="대화 초기화", use_container_width=True):
        st.session_state.messages=[{
            "role":"assistant","content":"대화를 새로 시작합니다. 무엇이 궁금하신가요?",
            "sources":[], "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
        }]
        st.rerun()

# =========================
# Presets & Uploader
# =========================
cols = st.columns(3)
for i, label in enumerate(["우리금융지주 전망?", "호텔신라 실적 포인트?", "배당주 포트 제안"]):
    with cols[i]:
        if st.button(label, use_container_width=True):
            st.session_state._preset = label
st.divider()

# =========================
# Render history
# =========================
for i, m in enumerate(st.session_state.messages):
    _render_message(m["content"], m["role"], m.get("ts",""))
    if m["role"]=="assistant":
        _copy_button(m["content"], key=f"msg-{i}")
        if m.get("sources"): _render_sources_inline(m["sources"])

# =========================
# Answer flow
# =========================
def run_answer(question: str):
    if not question: return
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    st.session_state.messages.append({"role":"user","content":question,"sources":[],"ts":now})
    _render_message(question, "user", now)
    with st.spinner("검색/생성 중…"):
        main = svc.answer(question) or {}
        main_sources = main.get("source_documents", []) or []
        extra = search_temp_index(question, top_k=5)
        answer = generate_with_context(question, main_sources, extra)
        merged_sources = (extra or []) + (main_sources[:5] if main_sources else [])
    now2 = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    st.session_state.messages.append({"role":"assistant","content":answer,"sources":merged_sources,"ts":now2})
    _render_message(answer,"assistant",now2)
    _copy_button(answer, key=f"ans-{len(st.session_state.messages)}")
    _render_sources_inline(merged_sources)

# 입력 + 프리셋 + 다시 생성
q = st.chat_input(placeholder="질문을 입력하세요…", key="chat_input")
if not q: q = st.session_state._preset
if q:
    run_answer(q)
    st.session_state._preset = None

# “다시 생성” 버튼
if len(st.session_state.messages) >= 2:
    last_user = next((m["content"] for m in reversed(st.session_state.messages) if m["role"]=="user"), None)
    if last_user and st.button("🔁 답변 다시 생성", use_container_width=True):
        run_answer(last_user)
