import os, threading
from typing import List, Dict, Any, Optional, TypedDict
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from qdrant_client import QdrantClient
from google.oauth2 import service_account
import streamlit as st  # Streamlit secrets 사용 시


class RAGState(TypedDict):
    question: str
    documents: List[Dict[str, Any]]
    answer: Optional[str]


class NewsQnAService:
    """Qdrant + Gemini 기반 RAG 서비스"""
    _thread_local = threading.local()

    def __init__(
        self,
        project: str | None = None,
        location: str = "us-central1",
        qdrant_url: str | None = None,
        qdrant_key: str | None = None,
        collection: str = "stock_news",
        embed_model_name: str = "gemini-embedding-001",
        gen_model_name: str = "gemini-2.5-pro",
        embed_dim: int = 3072,
        top_k: int = 10,
        rerank_top_k: int = 5,
        use_rerank: bool = False,
    ):
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not self.project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT required")

        # 🔐 Streamlit secrets → 서비스계정 크레덴셜
        sa_info = None
        try:
            sa_info = st.secrets.get("gcp_service_account", None)
        except Exception:
            sa_info = None

        creds = None
        if sa_info:
            creds = service_account.Credentials.from_service_account_info(
                dict(sa_info),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        # ✅ Vertex 초기화 (credentials 포함)
        vertexai.init(project=self.project, location=self.location, credentials=creds)

        # Qdrant 설정
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_key = qdrant_key or os.getenv("QDRANT_API_KEY")
        if not (self.qdrant_url and self.qdrant_key):
            raise RuntimeError("QDRANT_URL / QDRANT_API_KEY required")

        self.collection = collection or os.getenv("COLLECTION_NAME", "stock_news")
        self.embed_model_name = embed_model_name or os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")
        self.gen_model_name = gen_model_name or os.getenv("GENAI_MODEL_NAME", "gemini-2.5-pro")
        self.embed_dim = int(embed_dim or int(os.getenv("EMBED_DIM", "3072")))
        self.top_k = int(top_k or int(os.getenv("DEFAULT_TOP_K", "10")))
        self.rerank_top_k = int(rerank_top_k or int(os.getenv("RERANK_TOP_K", "5")))
        self.use_rerank = use_rerank

        self.qc = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_key)

        # 모델은 스레드 로컬 캐싱
        self._ensure_models()

    # ---------- internals ----------
    def _ensure_models(self):
        if not hasattr(self._thread_local, "embed_model") or getattr(self._thread_local, "embed_name", None) != self.embed_model_name:
            self._thread_local.embed_model = TextEmbeddingModel.from_pretrained(self.embed_model_name)
            self._thread_local.embed_name = self.embed_model_name
        if not hasattr(self._thread_local, "gen_model") or getattr(self._thread_local, "gen_name", None) != self.gen_model_name:
            self._thread_local.gen_model = GenerativeModel(self.gen_model_name)
            self._thread_local.gen_name = self.gen_model_name

    @property
    def embed_model(self) -> TextEmbeddingModel:
        self._ensure_models()
        return self._thread_local.embed_model

    @property
    def gen_model(self) -> GenerativeModel:
        self._ensure_models()
        return self._thread_local.gen_model

    def _embed_query(self, text: str) -> list[float]:
        inp = [TextEmbeddingInput(text=text or "", task_type="RETRIEVAL_QUERY")]
        return self.embed_model.get_embeddings(inp, output_dimensionality=self.embed_dim)[0].values

    def _approx_token_len(self, s: str) -> int:
        # 대략 1 token ~= 4 chars 가정
        return max(1, len(s) // 4)

    def _clip_docs(self, docs: List[Dict[str, Any]], per_doc_chars: int = 1200, max_total_chars: int = 12000) -> List[Dict[str, Any]]:
        clipped: List[Dict[str, Any]] = []
        total = 0
        for d in docs:
            txt = (d.get("content") or "").strip()
            if not txt:
                continue
            if len(txt) > per_doc_chars:
                txt = txt[:per_doc_chars] + "…"
            if total + len(txt) > max_total_chars:
                remain = max_total_chars - total
                if remain <= 0:
                    break
                txt = txt[:remain] + "…"
            nd = dict(d)
            nd["content"] = txt
            clipped.append(nd)
            total += len(txt)
        return clipped

    def _make_prompt(self, question: str, ctx: str) -> str:
        return f"""
당신은 주식시장과 연금에 정통한 전문 애널리스트입니다.
아래 컨텍스트를 근거로 **한국어** 답변을 작성하세요.

[작성 지침]
1) (현황 요약) → (원인/맥락) → (전망/조언) 의 3단락 이상
2) 중요 포인트는 **굵게**, 핵심 수치는 `코드블록` 스타일
3) ▸, ✔, ✦ 등의 불릿으로 가독성 향상
4) 마지막에 --- 넣고 근거 기사 한 줄 요약
5) 모호/근거없음 → "관련된 정보를 찾을 수 없습니다."라고 명시

[컨텍스트]
{ctx}

[질문]
{question}
""".strip()

    # ---------- RAG steps ----------
    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        qv = self._embed_query(question)
        hits = self.qc.search(
            collection_name=self.collection,
            query_vector=qv,
            limit=self.rerank_top_k if self.use_rerank else self.top_k,
            with_payload=True,
            with_vectors=False,
        )
        docs: List[Dict[str, Any]] = []
        for h in hits:
            payload = h.payload or {}
            docs.append({
                "id": str(h.id),
                "content": payload.get("text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "text"},
                "score": float(getattr(h, "score", 1.0)),
            })
        return docs

    def rerank(self, question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # TODO: 필요 시 cross-encoder/Vertex Ranking 연결
        return (docs or [])[: self.top_k]

    def generate(self, question: str, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return "관련된 정보를 찾을 수 없습니다."

        # 1) 컨텍스트 클리핑 (문서당/전체 길이 제한)
        docs_small = self._clip_docs(docs, per_doc_chars=1200, max_total_chars=12000)
        ctx = "\n\n".join(d["content"] for d in docs_small)
        prompt = self._make_prompt(question, ctx)

        # 2) 토큰 예산 보호: 프롬프트 6k 토큰(≈24k chars) 이하
        MAX_PROMPT_TOKENS = 6000
        MAX_PROMPT_CHARS = MAX_PROMPT_TOKENS * 4
        if len(prompt) > MAX_PROMPT_CHARS:
            over = len(prompt) - MAX_PROMPT_CHARS
            keep = max(0, len(ctx) - over - 1000)  # 지침 여유
            ctx = (ctx[:keep] + "…") if keep > 0 else ""
            prompt = self._make_prompt(question, ctx)

        gcfg = {"temperature": 0.2, "max_output_tokens": 800}

        def _call_model(p: str) -> str:
            resp = self.gen_model.generate_content(p, generation_config=gcfg)
            # 1) 일반 경로
            text = getattr(resp, "text", None)
            if text:
                return text.strip()
            # 2) candidates → parts 경로
            cands = getattr(resp, "candidates", None)
            if cands and getattr(cands[0], "content", None):
                parts = getattr(cands[0].content, "parts", None)
                if parts and hasattr(parts[0], "text"):
                    return (parts[0].text or "").strip()
            # 3) 실패
            raise RuntimeError("empty_text")

        # 3) 1차 호출
        try:
            return _call_model(prompt)
        except Exception:
            # 4) 재시도: 더 짧은 컨텍스트
            docs_tiny = self._clip_docs(docs, per_doc_chars=600, max_total_chars=6000)
            ctx2 = "\n\n".join(d["content"] for d in docs_tiny)
            prompt2 = self._make_prompt(question, ctx2)
            try:
                return _call_model(prompt2)
            except Exception:
                # 5) 최종 실패
                return (
                    "관련된 정보를 찾을 수 없습니다.\n\n"
                    "사유: 모델 안전/길이 제한으로 응답을 생성하지 못했습니다. "
                    "질문을 더 구체적으로 주시거나(예: 기간/종목/이슈), 컨텍스트를 줄여 다시 시도해 주세요."
                )

    # ---------- public APIs ----------
    def answer(self, question: str) -> Dict[str, Any]:
        docs = self.retrieve(question)
        docs = self.rerank(question, docs) if self.use_rerank else docs[: self.top_k]
        ans = self.generate(question, docs)
        return {"answer": ans, "source_documents": docs}

    def retrieve_only(self, question: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        prev_top_k, self.top_k = self.top_k, (top_k or self.top_k)
        try:
            docs = self.retrieve(question)
            return docs[: (top_k or self.top_k)]
        finally:
            self.top_k = prev_top_k
