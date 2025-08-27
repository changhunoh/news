# app.py
import os, io, re
from typing import List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import streamlit as st

# ----------------------------
# Page config & Secrets → ENV
# ----------------------------
st.set_page_config(page_title="나의 퇴직연금 챗봇", page_icon="📰", layout="centered")

def _prime_env_from_secrets():
    try:
        for k, v in st.secrets.items():
            os.environ.setdefault(k, str(v))
    except Exception:
        pass

_prime_env_from_secrets()
TZ = ZoneInfo(os.getenv("APP_TZ", "Asia/Seoul"))

# 고정 폭(폰 느낌) + 반응형
st.markdown("""
<style>
.block-container{ max-width:560px; margin-inline:auto; padding-top:10px; }
@media (max-width:640px){ .block-container{ max-width:94vw; } }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Backend service
# ----------------------------
from news_qna_service import NewsQnAService  # 같은 리포에 있어야 합니다.

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

# ----------------------------
# Lazy Vertex models (for temp index + generation)
# ----------------------------
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

# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = [{
        "role": "assistant",
        "content": "안녕하세요! ✅ 연금/주식 뉴스를 근거로 QnA 도와드려요. 무엇이든 물어보세요.",
        "sources": [], "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    }]
if "temp_docs" not in st.session_state:
    st.session_state.temp_docs: List[Dict[str, Any]] = []  # 업로드 임시 인덱스
if "_preset" not in st.session_state:
    st.session_state._preset = None

# ----------------------------
# Light Theme (강제 적용)
# ----------------------------
THEME = {
    "bg": "#f6f8fb",
    "text": "#1f2a44",
    "muted": "#5b6785",
    "user_bg": "#0b62e6",
    "user_fg": "#ffffff",
    "bot_bg": "#ffffff",
    "bot_fg": "#1f2a44",
    "bubble_border": "#e6ebf4",
    "chip_bg": "#eef4ff",
    "chip_fg": "#1757ff",
    "chip_border": "#dce7ff",
    "divider": "#e6ebf4",
    "time": "#6b7280",
    "input_bg": "#ffffff",
    "input_fg": "#1f2a44",
}

# ===== CSS (색/래핑/버튼/칩/입력창 보정 + 헤더/리셋 버튼) =====
st.markdown(f"""
<style>
html, body {{ background:{THEME["bg"]} !important; color:{THEME["text"]}; }}
h3, h4, h5, h6{{ color:{THEME["text"]}; }}
.stMarkdown p, .stMarkdown div{{ color:{THEME["text"]}; }}

/* 상단 헤더 */
.chat-header {{
  display:flex; align-items:center; justify-content:space-between;
  margin: 4px 2px 12px;
}}
.chat-title {{ font-size:20px; font-weight:900; color:{THEME["text"]}; }}
.reset-btn > button {{
  width:38px; height:38px; border-radius:999px;
  border:1px solid {THEME["chip_border"]} !important;
  background:{THEME["chip_bg"]} !important;
  color:{THEME["chip_fg"]} !important;
  font-weight:900 !important;
  box-shadow:0 4px 12px rgba(23,87,255,0.08);
}}

/* 추천 질문 칩(버튼) */
div.stButton > button {{
  border-radius:999px !important;
  padding:8px 14px !important;
  font-weight:700 !important;
  font-size:14px !important;
  border:1px solid {THEME["chip_border"]} !important;
  background:{THEME["chip_bg"]} !important;
  color:{THEME["chip_fg"]} !important;
  min-height:auto !important;
}}

/* Expander header */
.streamlit-expanderHeader {{ font-weight:800 !important; color:{THEME["text"]} !important; }}

/* 채팅 레이아웃 */
.chat-row {{ display:flex; margin:10px 0; }}
.user-row {{ justify-content:flex-end; }}
.bot-row  {{ justify-content:flex-start; }}

.chat-bubble {{
  max-width:82%;
  padding:12px 14px;
  border-radius:18px;
  line-height:1.6;
  font-size:15px;
  background:{THEME["bot_bg"]};
  color:{THEME["bot_fg"]};
  border:1px solid {THEME["bubble_border"]};
  border-bottom-left-radius:6px;
  box-shadow:0 8px 20px rgba(15,23,42,0.08);
  /* 텍스트 안전 래핑 */
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
}}
.user-bubble {{
  background:{THEME["user_bg"]} !important;
  color:{THEME["user_fg"]} !important;
  border:0 !important;
  border-bottom-right-radius:6px;
  box-shadow:0 6px 18px rgba(11,98,230,0.18);
}}

.timestamp {{ font-size:12px; color:{THEME["time"]}; margin:4px 6px; }}
.ts-left {{ text-align:left; }} .ts-right{{ text-align:right; }}

/* 액션바(복사 등) */
.action-bar {{ display:flex; gap:8px; margin:6px 6px 0; }}
.action-btn {{
  font-size:12px; padding:6px 10px; border-radius:10px;
  border:1px solid {THEME["chip_border"]};
  background:{THEME["chip_bg"]}; color:{THEME["chip_fg"]};
}}
.action-btn:hover{{ filter:brightness(1.05); }}

/* 출처 칩 */
.source-chip {{
  display:inline-block; padding:4px 10px; border-radius:999px;
  background:{THEME["chip_bg"]}; color:{THEME["chip_fg"]}; font-weight:800; font-size:12px;
  border:1px solid {THEME["chip_border"]}; margin:6px 6px 0 0;
}}
.source-chip a{{ color:{THEME["chip_fg"]}; text-decoration:none; }}
.source-chip a:hover{{ text-decoration:underline; }}
.src-row {{ margin:4px 6px 0; }}

/* 구분선 */
hr {{ border:0; border-top:1px solid {THEME["divider"]}; }}

/* 입력창 가독성 */
.stChatInputContainer textarea {{
  background:{THEME["input_bg"]} !important;
  color:{THEME["input_fg"]} !important;
  border:1px solid {THEME["bubble_border"]} !important;
  border-radius:12px !important;
  padding:12px !important;
  font-size:15px !important;
}}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Small utils (rendering)
# ----------------------------
def _md(html: str):
    st.markdown(html, unsafe_allow_html=True)

def _escape_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _linkify(s: str) -> str:
    pattern = r"(https?://[\w\-\./%#\?=&:+,~]+)"
    return re.sub(pattern, r'<a href="\1" target="_blank">\1</a>', s)

def _render_message(text: str, sender: str, ts: str):
    row = "user-row" if sender == "user" else "bot-row"
    bub = "user-bubble" if sender == "user" else "bot-bubble"
    safe = _linkify(_escape_html(text or ""))
    _md(f'<div class="chat-row {row}"><div class="chat-bubble {bub}">{safe}</div></div>')
    _md(f'<div class="timestamp {"ts-right" if sender=="user" else "ts-left"}">{ts}</div>')

def _render_sources_inline(sources: List[Dict[str,Any]]):
    if not sources:
        return
    chips = []
    for i, d in enumerate(sources, 1):
        meta = d.get("metadata", {}) or {}
        title = meta.get("title") or meta.get("path") or meta.get("source") or f"문서 {i}"
        url   = meta.get("url")
        score = float(d.get("score", 0.0))
        label = f"#{i} {title} · {score:.3f}"
        link = f'<a href="{url}" target="_blank">{label}</a>' if url else label
        chips.append(f'<span class="source-chip">{link}</span>')
    _md(f'<div class="src-row">{"".join(chips)}</div>')

def _copy_button(text: str, key: str):
    from streamlit.components.v1 import html as st_html
    safe = (text or "").replace("\\", "\\\\").replace("`", "\\`")
    st_html(f"""
<div class="action-bar">
  <button class="action-btn" id="copy-{key}" data-text="{safe}">📋 복사</button>
  <span class="small" id="copied-{key}" style="display:none;">복사됨!</span>
</div>
<script>
(function(){{
  const b=document.getElementById("copy-{key}"), t=document.getElementById("copied-{key}");
  if(!b) return;
  b.onclick = async () => {{
    try {{
      await navigator.clipboard.writeText(b.getAttribute("data-text"));
      t.style.display="inline-block"; setTimeout(()=>t.style.display="none",1200);
    }} catch(e) {{
      const ta=document.createElement('textarea'); ta.value=b.getAttribute("data-text");
      document.body.appendChild(ta); ta.select(); document.execCommand('copy'); document.body.removeChild(ta);
      t.style.display="inline-block"; setTimeout(()=>t.style.display="none",1200);
    }}
  }};
}})();
</script>
""", height=30)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))

# ----------------------------
# Upload → Temp Index (세션 전용)
# ----------------------------
def _read_text_from_file(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    try:
        if name.endswith((".txt",".md",".csv")):
            return data.decode("utf-8", errors="ignore")
        if name.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                return ""
        if name.endswith(".docx"):
            try:
                import docx
                doc = docx.Document(io.BytesIO(data))
                return "\n".join(p.text for p in doc.paragraphs)
            except Exception:
                return ""
    except Exception:
        return ""
    return ""

def _chunk(text: str, size=1200, overlap=150):
    out, s, n = [], 0, len(text or "")
    while s < n:
        e = min(s + size, n)
        out.append(text[s:e])
        if e == n: break
        s = max(e - overlap, s + 1)
    return out

def _embed_texts(texts: List[str]) -> List[np.ndarray]:
    model = _get_embed_model()
    from vertexai.language_models import TextEmbeddingInput
    inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in texts]
    embs = model.get_embeddings(inputs, output_dimensionality=EMBED_DIM)
    return [np.array(e.values, dtype=np.float32) for e in embs]

def add_uploaded_to_temp_index(files) -> int:
    if not files: return 0
    added = 0
    for f in files:
        raw = _read_text_from_file(f)
        if not raw: continue
        chunks = _chunk(raw)
        vecs = _embed_texts(chunks)
        for i, (seg, vec) in enumerate(zip(chunks, vecs)):
            st.session_state.temp_docs.append({
                "id": f"{f.name}:{i}",
                "content": seg,
                "metadata": {"title": f.name, "source": "upload"},
                "emb": vec
            })
            added += 1
    return added

def search_temp_index(query: str, top_k=5) -> List[Dict[str,Any]]:
    if not st.session_state.temp_docs: return []
    model = _get_embed_model()
    from vertexai.language_models import TextEmbeddingInput
    qv = np.array(
        model.get_embeddings([TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")],
                             output_dimensionality=EMBED_DIM)[0].values,
        dtype=np.float32
    )
    scored = [(_cosine(qv, d["emb"]), d) for d in st.session_state.temp_docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, d in scored[:top_k]:
        out.append({"id": d["id"], "content": d["content"], "metadata": d["metadata"], "score": float(s)})
    return out

# ----------------------------
# Generation with merged context
# ----------------------------
def generate_with_context(question: str,
                          main_sources: List[Dict[str,Any]],
                          extra_sources: List[Dict[str,Any]]) -> str:
    def snip(t, n=1800):
        t = re.sub(r"\s+", " ", t or "")
        return t[:n]
    merged = (extra_sources or []) + (main_sources or [])
    ctx = "\n\n".join([snip(d.get("content","")) for d in merged])[:10000]

    sys_prompt = (
        "당신은 주식/연금 뉴스를 바탕으로 답하는 분석가입니다. "
        "주어진 컨텍스트에 근거하여 한국어로 간결하고 정확하게 답하세요. "
        "근거가 부족하면 추정하지 말고 '관련된 정보를 찾을 수 없습니다.'라고 답하세요. "
        "핵심은 **굵게** 강조하세요."
    )
    prompt = f"{sys_prompt}\n\n[컨텍스트]\n{ctx}\n\n[질문]\n{question}"

    try:
        model = _get_gen_model()
        resp = model.generate_content(prompt, generation_config={"temperature":0.2, "max_output_tokens":1024})
        return (resp.text or "").strip()
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

# ----------------------------
# Header (제목 + 우측 회전 초기화 버튼)
# ----------------------------
col_t, col_btn = st.columns([1, 0.16])
with col_t:
    st.markdown('<div class="chat-header"><div class="chat-title">💬 나의 퇴직연금 챗봇</div></div>', unsafe_allow_html=True)
with col_btn:
    if st.button("🔄", help="대화 초기화", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "대화를 새로 시작합니다. 무엇이 궁금하신가요?",
            "sources": [], "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
        }]
        st.rerun()

# ----------------------------
# Presets & Uploader
# ----------------------------
preset_cols = st.columns(3)
presets = ["우리금융지주 전망?", "호텔신라 실적 포인트?", "배당주 포트 제안"]
for i, label in enumerate(presets):
    with preset_cols[i]:
        if st.button(label, use_container_width=True):
            st.session_state._preset = label

with st.expander("📎 파일 업로드(임시 인덱스)", expanded=False):
    files = st.file_uploader("txt, md, csv, pdf, docx 지원", type=["txt","md","csv","pdf","docx"], accept_multiple_files=True)
    if st.button("임시 인덱스에 추가"):
        n = add_uploaded_to_temp_index(files or [])
        st.success(f"세그먼트 {n}개 추가됨")
    st.caption(f"세션 보관 중인 세그먼트: {len(st.session_state.temp_docs)}")

st.divider()

# ----------------------------
# Render history
# ----------------------------
for i, m in enumerate(st.session_state.messages):
    _render_message(m["content"], m["role"], m.get("ts", ""))
    if m["role"] == "assistant":
        _copy_button(m["content"], key=f"msg-{i}")
        if m.get("sources"):
            _render_sources_inline(m["sources"])

# ----------------------------
# Answer runner
# ----------------------------
def run_answer(question: str):
    if not question: return
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    st.session_state.messages.append({"role": "user", "content": question, "sources": [], "ts": now})
    _render_message(question, "user", now)

    with st.spinner("검색/생성 중…"):
        main = svc.answer(question) or {}
        main_sources = main.get("source_documents", []) or []
        extra = search_temp_index(question, top_k=5)
        answer = generate_with_context(question, main_sources, extra)
        merged_sources = (extra or []) + (main_sources[:5] if main_sources else [])

    now2 = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": merged_sources, "ts": now2})
    _render_message(answer, "assistant", now2)
    _copy_button(answer, key=f"ans-{len(st.session_state.messages)}")
    _render_sources_inline(merged_sources)

# ----------------------------
# Chat input + 재생성
# ----------------------------
q = st.chat_input(placeholder="질문을 입력하세요…", key="chat_input")
if not q:
    q = st.session_state._preset

if q:
    run_answer(q)
    st.session_state._preset = None

if len(st.session_state.messages) >= 2:
    last_user = None
    for m in reversed(st.session_state.messages):
        if m["role"] == "user":
            last_user = m["content"]
            break
    if last_user:
        if st.button("🔁 답변 다시 생성", use_container_width=True):
            run_answer(last_user)
