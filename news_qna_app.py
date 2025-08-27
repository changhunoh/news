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

# 환경/컬렉션 정보 표시 (사이드바)
st.sidebar.write("COLLECTION_NAME =", os.getenv("COLLECTION_NAME"))
st.sidebar.write("EMBED_MODEL_NAME =", os.getenv("EMBED_MODEL_NAME"))
st.sidebar.write("EMBED_DIM =", os.getenv("EMBED_DIM"))

from qdrant_client import QdrantClient
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
col = os.getenv("COLLECTION_NAME","stock_news")
info = client.get_collection(col)
st.sidebar.write("Qdrant vector_size =", info.config.params.vectors.size)
cnt = client.count(col, exact=True).count
st.sidebar.write("Qdrant points =", cnt)

# ------------------------
# 백엔드 서비스(선택)
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
# 상태
# ------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "안녕하세요. 여러분들의 연금을 풍요롭게 만드는 연금술사입니다. 무엇이 궁금하신가요?",
        "ts": fmt_ts(datetime.now(TZ))
    }]

# ------------------------
# 추출 함수들
# ------------------------
def _extract_text(d: dict) -> str:
    # retrieve 결과 형태를 기준으로 content 우선
    if not isinstance(d, dict):
        return str(d)
    txt = d.get("content")
    if txt:
        return txt
    # 혹시 service가 payload 형태를 그대로 넘길 때 대비
    md = d.get("metadata") or {}
    if isinstance(md.get("doc"), dict):
        return md["doc"].get("content") or md["doc"].get("text") or ""
    if isinstance(md.get("doc"), str):
        return md["doc"]
    return md.get("content") or md.get("text") or ""

def _extract_title_url(d: dict):
    md = (d.get("metadata") or {}) if isinstance(d, dict) else {}
    title = md.get("title") or md.get("path") or md.get("source") or md.get("file_name")
    url = md.get("url") or md.get("link")
    return title, url

def _extract_score_str(d: dict) -> Optional[str]:
    if not isinstance(d, dict):
        return None
    sim = d.get("score", None)
    dist = d.get("distance", None)
    mode = (d.get("distance_mode") or "").lower()
    try:
        if sim is not None:
            return f"{float(sim):.4f}"
        if dist is not None:
            if "cosine" in mode:
                return f"{1.0 - float(dist):.4f}"
            return f"dist={float(dist):.4f}"
    except Exception:
        pass
    # metadata 안쪽일 수도 있음
    md = d.get("metadata") or {}
    for k in ("score","similarity"):
        if k in md:
            try: return f"{float(md[k]):.4f}"
            except: pass
    for k in ("distance",):
        if k in md:
            try:
                return f"{1.0 - float(md[k]):.4f}" if "cosine" in mode else f"dist={float(md[k]):.4f}"
            except:
                pass
    return None
    
# ------------------------
# 아바타 경로
# ------------------------

ASSISTANT_AVATAR_URL = os.getenv("ASSISTANT_AVATAR_URL", "")  # 예: https://...
USER_AVATAR_URL      = os.getenv("USER_AVATAR_URL", "")
ASSISTANT_EMOJI      = "🧙‍♂️"
USER_EMOJI           = "🤴"

# ------------------------
# 메시지 렌더
# ------------------------
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
# 아바타 CSS
# ------------------------

st.markdown("""
<style>
/* 아바타 + 말풍선 기본 레이아웃 */
.chat-row{ display:flex; gap:10px; margin:10px 0; align-items:flex-start; }
.bot-row { justify-content:flex-start; }
.user-row{ justify-content:flex-end;  }

/* 아바타 */
.avatar{ width:40px; height:40px; border-radius:999px; overflow:hidden;
         border:1px solid #e5e7eb; background:#fff; flex:0 0 40px; }
.avatar img{ width:100%; height:100%; object-fit:cover; display:block; }
.avatar.emoji{ display:flex; align-items:center; justify-content:center; font-size:22px; }

/* 말풍선 */
.bubble{ max-width:80%; padding:10px 12px; border-radius:14px; line-height:1.6; }
.bubble.bot  { background:#f5f6f8; color:#111; }
.bubble.user { background:#0b62e6; color:#fff; }

/* 타임스탬프 */
.time{ font-size:11px; color:#6b7280; margin-top:4px; }

.bubble{ position:relative; border-radius:16px; }

.bubble.bot{
  background:#f6f8fb;
  border:1px solid #eef2f7;
  box-shadow: 0 6px 16px rgba(15,23,42,.12), inset 0 1px 0 rgba(255,255,255,.65);
}

.bubble.user{
  background:#0b62e6;
  border:0;
  box-shadow: 0 10px 24px rgba(11,98,230,.28);
}

/* 선택: 아주 미묘한 hover 입체감 */
.bubble:hover{ transform: translateY(-1px); transition: transform .12s ease; }

/* 선택: 꼬리(보톡스 형태) */
.bubble.bot::after{
  content:""; position:absolute; left:-6px; top:10px;
  width:12px; height:12px; background:#f6f8fb;
  border-left:1px solid #eef2f7; border-bottom:1px solid #eef2f7;
  transform: rotate(45deg); border-bottom-left-radius:3px;
}

.bubble.user::after{
  content:""; position:absolute; right:-6px; top:10px;
  width:12px; height:12px; background:#0b62e6;
  box-shadow: 2px 6px 12px rgba(11,98,230,.22);
  transform: rotate(45deg); border-top-right-radius:3px;
}
/* 말풍선 폭을 조금 넓히고, 줄바꿈 규칙을 한국어 친화적으로 조정 */
.bubble{
  display: inline-block;
  /* 화면에 따라 가변 폭: 최소 260px ~ 최대 680px 사이 */
  max-width: clamp(260px, 60vw, 680px);
  /* 내용 줄바꿈 규칙 */
  white-space: pre-wrap;        /* \n 유지 + 일반 줄바꿈 허용 */
  word-break: keep-all;         /* 한국어 단어(조합) 중간 단위로 끊지 않음 */
  overflow-wrap: break-word;    /* 너무 길면 단어 기준으로만 줄바꿈 */
}

/* 유저/봇 공통으로 내부 텍스트에도 동일 규칙 적용(링크 등 인라인 요소 포함) */
.bubble, .bubble *{
  white-space: pre-wrap;
  word-break: keep-all;
  overflow-wrap: break-word;
}

/* 필요시: 아주 긴 URL 같은 비연속 문자열은 anywhere로 최후 보정 */
.bubble a{
  overflow-wrap: anywhere;  /* 링크가 너무 길면 어딘가에서라도 꺾이도록 */
}
</style>
""", unsafe_allow_html=True)

def render_messages(msgs, placeholder):
    html_parts = []
    for m in msgs:
        role = m.get("role","assistant")
        text = _linkify(_escape_html(m.get("content","")))
        ts   = _escape_html(m.get("ts",""))

        if role == "assistant":
            html_parts.append(
                "<div class='chat-row bot-row'>"
                f"{_avatar_html('assistant')}"
                f"<div><div class='bubble bot'>{text}</div>"
                f"<div class='time'>{ts}</div>"
                "</div></div>"
            )
            # 근거칩(있을 때만)
            for j, src in enumerate(m.get("sources", []), 1):
                title, url = _extract_title_url(src)
                score_s = _extract_score_str(src)
                label = f"#{j} {title or f'문서 {j}'}"
                if score_s: label += f" ({score_s})"
                if url: label = f"<a href='{url}' target='_blank'>{label}</a>"
                html_parts.append(
                    "<div class='chat-row bot-row' style='margin-top:-6px;'>"
                    f"<div style='width:40px'></div>"
                    f"<div class='time' style='margin-left:4px;'>📎 {label}</div>"
                    "</div>"
                )
        else:
            # 유저는 오른쪽 정렬: 말풍선 먼저, 아바타는 우측
            html_parts.append(
                "<div class='chat-row user-row'>"
                f"<div><div class='bubble user'>{text}</div>"
                f"<div class='time' style='text-align:right'>{ts}</div>"
                "</div>"
                f"{_avatar_html('user')}"
                "</div>"
            )

    placeholder.markdown("\n".join(html_parts), unsafe_allow_html=True)

# ------------------------
# UI 헤더 & 플레이스홀더
# ------------------------
st.title("🧙‍♂️ 우리 연금술사")
messages_ph = st.empty()
debug = st.sidebar.toggle("🔍 RAG 디버그 보기", value=True)
# ------------------------
# 답변 생성
# ------------------------
def run_answer(question: str):
    # 1) 사용자 메시지 즉시 반영
    st.session_state["messages"].append({
        "role": "user", "content": question, "ts": fmt_ts(datetime.now(TZ))
    })
    render_messages(st.session_state["messages"], messages_ph)

    # 2) 서비스 호출
    sources = []
    result: Dict[str, Any] = {}
    if svc:
        try:
            result = svc.answer(question) or {}
            # 다양한 키 대응
            ans = (
                result.get("answer")
                or result.get("output_text")
                or result.get("output")
                or result.get("content")
                or ""
            ).strip()
            sources = (
                result.get("source_documents")
                or result.get("sources")
                or result.get("docs")
                or []
            )
            if not ans:
                # 근거는 있는데 답변이 비면 안전 Fallback
                if sources:
                    ans = "관련 원문 요약에 실패했습니다. 원문 일부:\n\n" + (_extract_text(sources[0])[:400])
                else:
                    ans = "관련 정보를 찾을 수 없습니다."
        except Exception as e:
            ans = f"오류 발생: {e}"
            result = {"error": str(e)}
            sources = []
    else:
        ans = f"데모 응답: '{question}'에 대한 분석 결과는 준비 중입니다."
        result = {"answer": ans, "source_documents": []}
        sources = []

    # 3) 디버그 패널
    if debug:
        with st.expander("RAG raw result"):
            try:
                st.write("result keys:", list(result.keys()))
            except Exception:
                st.write("result:", result)
            st.write("num sources:", len(sources))
            for i, d in enumerate(sources, 1):
                title, url = _extract_title_url(d)
                score_s = _extract_score_str(d)
                st.markdown(f"**#{i} {title}** | score={score_s}")
                if url: st.markdown(f"[원문]({url})")
                st.code(_extract_text(d)[:600])

    # 4) 어시스턴트 메시지 반영
    st.session_state["messages"].append({
        "role": "assistant",
        "content": ans,
        "sources": sources,
        "ts": fmt_ts(datetime.now(TZ))
    })
    render_messages(st.session_state["messages"], messages_ph)

# ---- 폼 (제출 먼저 처리 → 마지막에 렌더) ----
with st.form("chat_form", clear_on_submit=True):
    user_q = st.text_input("질문을 입력하세요", "")
    submitted = st.form_submit_button("전송")

if submitted and user_q.strip():
    # 즉시 반영 구조 → rerun 호출하지 않음
    run_answer(user_q)

# 최초/마지막 안전 렌더
render_messages(st.session_state["messages"], messages_ph)

# ------------------------
# 사이드바: 바로 붙여넣기용 덤프
# ------------------------
with st.sidebar.expander("🧩 debug dump (붙여넣어 주시면 돼요)"):
    st.write("svc is None? ->", svc is None)

    q = st.text_input("테스트 질의", "호텔신라 주식에 대해 어떻게 생각해?", key="dump_q")

    # 1) svc.answer() 원형 출력
    if st.button("answer() 호출", key="dump_btn"):
        try:
            if svc is None:
                st.warning("svc 가 None 입니다. news_qna_service 임포트/환경변수 확인 필요.")
                res = {}
            else:
                res = svc.answer(q) or {}
            st.write("type(res):", type(res))
            if isinstance(res, dict):
                st.write("keys:", list(res.keys()))
                srcs = (res.get("source_documents") or res.get("sources") or res.get("docs") or [])
                st.write("num sources:", len(srcs))
                if srcs:
                    s0 = srcs[0]
                    st.write("source[0] keys:", list(s0.keys()) if isinstance(s0, dict) else type(s0))
                    md = (s0.get("metadata") or {}) if isinstance(s0, dict) else {}
                    st.write("metadata keys:", list(md.keys()))
                    # 안전 추출
                    txt = (
                        s0.get("content") or s0.get("page_content") or s0.get("text")
                        or (s0.get("metadata") or {}).get("content") or ""
                    )
                    st.code((txt[:600] + (" …" if len(txt) > 600 else "")))
            else:
                st.write("res:", res)
        except Exception as e:
            st.exception(e)

    # 2) svc 우회: Vertex 임베딩 + Qdrant 직접 검색(검색단만 점검)
    # --- Qdrant 직접 검색(LLM 제외) : 로컬과 완전 동일한 파라미터로 ---
if st.button("Qdrant 직접 검색(LLM 제외)", key="raw_search_btn"):
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
        from qdrant_client import models

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if project:
            vertexai.init(project=project, location=location)

        emb_name = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")
        emb_dim  = int(os.getenv("EMBED_DIM", "3072"))
        emb_norm = os.getenv("EMBED_NORMALIZE", "false").lower() == "true"
        top_k    = int(os.getenv("DEFAULT_TOP_K", "100"))  # 로컬과 동일하게

        model = TextEmbeddingModel.from_pretrained(emb_name)
        inputs = [TextEmbeddingInput(text=q, task_type="RETRIEVAL_QUERY")]
        qv = model.get_embeddings(inputs, output_dimensionality=emb_dim)[0].values

        # 선택: 적재를 정규화했다면 질의도 동일하게 정규화
        if emb_norm:
            import math
            n = math.sqrt(sum(x*x for x in qv)) or 1.0
            qv = [x / n for x in qv]

        # 검색 파라미터도 통일
        search_params = models.SearchParams(
            hnsw_ef=int(os.getenv("QDRANT_HNSW_EF", "128")),
            exact=os.getenv("QDRANT_EXACT", "false").lower() == "true",
        )

        hits = client.search(
            collection_name=col,
            query_vector=qv,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            search_params=search_params,
        )

        # 컬렉션 distance 모드 (cosine/dot/euclid)
        try:
            params = info.config.params
            dist_mode = str(params.vectors.distance).lower()
        except Exception:
            dist_mode = "unknown"

        st.write("검색 결과 개수:", len(hits))

        for i, h in enumerate(hits[:5], 1):
            payload = h.payload or {}

            # ✅ 당신 DB 스키마: { "doc": ..., "metadata": {...} }
            doc = payload.get("doc")
            if isinstance(doc, dict):
                text = doc.get("content") or doc.get("text") or doc.get("page_content") or ""
            elif isinstance(doc, str):
                text = doc
            else:
                text = payload.get("content") or payload.get("text") or ""

            # Qdrant score는 보통 distance → cosine이면 sim = 1 - dist
            dist = float(getattr(h, "score", 0.0))
            sim = (1.0 - dist) if "cosine" in dist_mode else None

            title = (
                (payload.get("metadata") or {}).get("title")
                or payload.get("title") or payload.get("path")
                or payload.get("source") or payload.get("file_name")
                or f"문서 {i}"
            )
            url = (payload.get("metadata") or {}).get("url") or payload.get("url") or payload.get("link")

            head = f"**#{i} {title}**"
            if sim is not None:
                head += f"  | sim={sim:.4f} (dist={dist:.4f}, {dist_mode})"
            else:
                head += f"  | dist={dist:.4f} ({dist_mode})"
            st.markdown(head)
            if url: st.markdown(f"[원문]({url})")
            st.code((text[:600] + (" …" if len(text) > 600 else "")))
    except Exception as e:
        st.exception(e)
