# app.py
import os
import re
import streamlit as st
from typing import List, Dict, Any

# =========================
# Page / Secrets → ENV priming
# =========================
st.set_page_config(
    page_title="나의 퇴직연금관리",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def _prime_env_from_secrets():
    """Streamlit secrets를 os.environ에 주입 (서비스에서 os.getenv 사용 가능)"""
    try:
        for k, v in st.secrets.items():
            os.environ.setdefault(k, str(v))
    except Exception:
        pass

_prime_env_from_secrets()

# =========================
# Backend Service
# =========================
from news_qna_service import NewsQnAService  # 같은 리포지토리에 두세요

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
        embed_dim=int(os.getenv("EMBED_DIM", "3072")),         # Qdrant 컬렉션과 일치
        top_k=int(os.getenv("DEFAULT_TOP_K", "10")),
        use_rerank=False,  # 필요 시 True로 바꾸고 서비스 내부 rerank 구현
    )

svc = get_service()

# =========================
# Router / State
# =========================
if "page" not in st.session_state:
    st.session_state.page = "index"   # index, article
if "question" not in st.session_state:
    st.session_state.question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_docs" not in st.session_state:
    st.session_state.last_docs: List[Dict[str, Any]] = []
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0
if "summary_date" not in st.session_state:
    st.session_state.summary_date = "오늘의 요약"

def go(page: str):
    st.session_state.page = page
    st.rerun()

# =========================
# CSS (폰 프레임 + 스타일)
# =========================
CSS = """
/* 전체 */
html, body, .block-container { background: #f6f8fb !important; }
section.main > div { padding-top: 10px; }
/* 사이드바/헤더 숨김 */
[data-testid="stSidebar"], header[tabindex="0"] { display: none !important; }

/* 폰 프레임 */
.phone-wrap { display:flex; justify-content:center; margin:20px auto 40px; max-width:560px; }
.phone {
  width:100%; background:#fff; border-radius:26px; overflow:hidden;
  border:1px solid #e7ebf3; box-shadow:0 10px 25px rgba(23,30,60,0.08);
}
.phone-header {
  position:relative; padding:18px 24px;
  background:linear-gradient(180deg,#ffffff 0%,#fafcff 100%);
  border-bottom:1px solid #eef2f7;
}
.phone-title { text-align:center; font-weight:800; font-size:22px; color:#1f2a44; }
.header-close {
  position:absolute; right:12px; top:10px; width:36px; height:36px; border-radius:999px;
  background:#fff; border:1px solid #e7ebf3; box-shadow:0 2px 8px rgba(23,30,60,0.06);
  display:grid; place-items:center; font-size:20px; color:#4b5568;
}
.phone-body { padding:16px; }

/* 카드 공통 */
.card {
  background:#fff; border:1px solid #e7ebf3; border-radius:20px;
  padding:18px; box-shadow:0 6px 18px rgba(23,30,60,0.06); margin:14px 8px;
}
.sec-title { font-size:18px; font-weight:800; color:#1765f0; margin:0 0 10px; }

/* 요약 카드 */
.summary-top {
  position:relative; background:#f4f9ff; border:1px solid #e1efff; border-radius:16px; padding:14px;
}
.summary-bg {
  position:absolute; inset:0;
  background: radial-gradient(80% 60% at 80% 20%, rgba(22,160,255,0.12), rgba(22,160,255,0) 60%),
              radial-gradient(70% 60% at 20% 80%, rgba(22,160,255,0.08), rgba(22,160,255,0) 60%);
  border-radius:16px; pointer-events:none;
}
.badge { display:inline-block; padding:3px 8px; border-radius:999px; background:#e7f1ff; color:#0b62e6; font-weight:800; font-size:13px; }
.summary-text { font-size:16px; font-weight:700; color:#1f2a44; line-height:1.5; margin-top:8px; }

/* 링크형 버튼 */
.link-btn > button {
  background: transparent !important;
  border: none !important;
  color: #0b62e6 !important;
  text-decoration: underline !important;
  font-weight: 700 !important;
  padding: 0 !important;
  box-shadow: none !important;
  min-height: auto !important;
}
.link-btn { margin-top: 8px; }

/* 마켓 */
.market { margin-top:10px; border-top:1px solid #e7ebf3; padding-top:10px; }
.mrow { display:grid; grid-template-columns:1fr auto auto; gap:10px; font-size:15px; padding:5px 0; }
.mname { color:#23304d; font-weight:600; }
.mval,.mchg { font-weight:700; }
.negative { color:#e03131; }
.positive { color:#0ca678; }

/* 평가금액 */
.amount { font-size:38px; font-weight:900; color:#131b2f; letter-spacing:0.5px; }
.unit { font-size:18px; font-weight:800; color:#3f4b6a; margin-left:6px; }

/* 수익률 */
.return-grid { display:grid; grid-template-columns:auto 1fr; gap:10px 24px; align-items:center; }
.label-col span, .value-col span { display:block; padding:6px 0; font-size:16px; }
.label-col span { color:#1765f0; font-weight:800; }
.value-col span { color:#1f2a44; font-weight:700; }

/* 기사 화면 */
.article-title { font-size:22px; font-weight:900; color:#0f1a31; margin:6px 0 4px; }
.article-meta { color:#5b6785; font-weight:700; margin-bottom:12px; }
.article-p { color:#26324d; font-size:16px; line-height:1.85; margin:12px 0; }
.share-btn {
  display:block; width:100%; padding:14px 18px; text-align:center;
  background:linear-gradient(180deg,#1d68ff 0%, #0052f5 100%);
  color:#fff; border-radius:12px; font-weight:900; font-size:18px; border:0;
  box-shadow:0 6px 16px rgba(0,82,245,0.28);
}
.footer-space { height:16px; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# =========================
# UI Components
# =========================
def header(title: str, back_to: str | None = None):
    st.markdown('<div class="phone-wrap"><div class="phone">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="phone-header">
      <div class="phone-title">{title}</div>
      <div class="header-close">×</div>
    </div>
    <div class="phone-body">
    """, unsafe_allow_html=True)
    if back_to is not None:
        cols = st.columns([1,6,1])
        with cols[2]:
            if st.button("닫기", use_container_width=True):
                go(back_to)

def tail():
    st.markdown('<div class="footer-space"></div></div></div>', unsafe_allow_html=True)

def summary_section():
    st.markdown(f"""
    <div class="card">
      <div class="sec-title">오늘의 요약</div>
      <div class="summary-top">
        <div class="summary-bg"></div>
        <span class="badge">AI 요약</span>
    """, unsafe_allow_html=True)

    q = st.text_input("질문을 입력하세요 (예: 우리금융지주 전망은?)",
                      value=st.session_state.question, label_visibility="collapsed")
    cols = st.columns([3,1])
    with cols[1]:
        run = st.button("분석", use_container_width=True)
    with cols[0]:
        st.caption("Qdrant + Gemini 임베딩으로 검색 후 답변을 생성합니다.")

    if run and q.strip():
        st.session_state.question = q
        with st.spinner("검색/생성 중..."):
            data = svc.answer(q)
        st.session_state.last_answer = data.get("answer", "")
        st.session_state.last_docs = data.get("source_documents", [])
        st.session_state.selected_idx = 0

    st.markdown(
        f"""<div class="summary-text">{st.session_state.last_answer or "질문을 입력하고 ‘분석’을 눌러보세요."}</div>""",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)  # .summary-top 닫기

    # 원문 보기
    col = st.container()
    with col:
        if st.button("원문 보기", key="go_article", help="최상위 문서 보기", type="secondary"):
            st.session_state.selected_idx = 0
            go("article")
    st.markdown("</div>", unsafe_allow_html=True)  # .card 닫기

    # 샘플 박스 (원하면 실제 데이터로 교체)
    market_rows = [("코스피", "3,200", "-12.34 (10%)"), ("KODEX 200 TR", "1,234", "-12.34 (10%)")]
    st.markdown(f"""
    <div class="card" style="margin-top:10px;">
      <div style="color:#303a53; font-size:15px; line-height:1.6; margin-bottom:10px;">
        연금 관련 이슈/뉴스를 요약해 보여줍니다.
      </div>
      <div class="market">
        {"".join([
          f'<div class="mrow">'
          f'  <div class="mname">{name}</div>'
          f'  <div class="mval negative">{val}</div>'
          f'  <div class="mchg negative">{chg}</div>'
          f'</div>'
          for (name, val, chg) in market_rows
        ])}
      </div>
    </div>
    """, unsafe_allow_html=True)

def eval_box():
    st.markdown(f"""
    <div class="card">
      <div class="sec-title" style="color:#3b82f6;">평가금액</div>
      <div class="amount">12,345,678 <span class="unit">원</span></div>
    </div>
    """, unsafe_allow_html=True)

def return_box():
    st.markdown(f"""
    <div class="card">
      <div class="sec-title" style="color:#3b82f6;">투자수익률 현황</div>
      <div class="return-grid">
        <div class="label-col">
          <span>원금</span>
          <span>누적수익</span>
          <span>누적수익률</span>
        </div>
        <div class="value-col">
          <span>10,000,000 원</span>
          <span>2,345,678 원</span>
          <span class="positive">+ 23.5%</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def article_section():
    docs = st.session_state.last_docs or []
    if not docs:
        st.info("먼저 질문을 입력하고 ‘분석’을 눌러 문서를 검색하세요.")
        return

    idx = max(0, min(st.session_state.get("selected_idx", 0), len(docs)-1))
    d = docs[idx] if isinstance(docs[idx], dict) else {}
    meta = d.get("metadata", {})
    title = meta.get("title") or meta.get("path") or meta.get("source") or f"문서 {idx+1}"
    src = meta.get("url") or meta.get("source") or ""
    score = d.get("score", 0.0)
    content = d.get("content", "")
    content_html = re.sub(r"\n", "<br/>", content)

    st.markdown(f"""
    <div class="card">
      <div class="article-title">{title}</div>
      <div class="article-meta">유사도 {score:.3f} {("· " + src) if src else ""}</div>
      <div class="article-p">{content_html}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("이전 문서"):
            st.session_state.selected_idx = max(0, idx-1)
            st.rerun()
    with c2:
        if st.button("다음 문서"):
            st.session_state.selected_idx = min(len(docs)-1, idx+1)
            st.rerun()

    if st.button("공유하기", use_container_width=True):
        st.toast("공유 기능은 프로토타입에서만 제공됩니다.", icon="ℹ️")

# =========================
# Pages
# =========================
def render_index():
    header("나의 퇴직연금관리", back_to=None)
    summary_section()
    eval_box()
    return_box()
    tail()

def render_article():
    header("원문 기사", back_to="index")
    article_section()
    tail()

# =========================
# Router Switch
# =========================
page = st.session_state.page
if page == "index":
    render_index()
elif page == "article":
    render_article()
else:
    render_index()
