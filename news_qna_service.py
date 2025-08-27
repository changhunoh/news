import os, re, threading
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
        top_k: int = 5,
        rerank_top_k: int = 5,
        use_rerank: bool = False,
    ):
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not self.project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT required")

        # 🔐 Streamlit secrets에서 서비스계정 읽어 credentials 생성
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

        # ✅ credentials까지 명시해서 초기화

        vertexai.init(project=self.project, location=self.location,credentials=creds)

        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_key = qdrant_key or os.getenv("QDRANT_API_KEY")
        if not (self.qdrant_url and self.qdrant_key):
            raise RuntimeError("QDRANT_URL / QDRANT_API_KEY required")

        self.collection = collection or os.getenv("COLLECTION_NAME", "stock_news")
        self.embed_model_name = embed_model_name or os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")
        self.gen_model_name = gen_model_name or os.getenv("GENAI_MODEL_NAME", "gemini-2.5-pro")
        self.embed_dim = int(embed_dim or int(os.getenv("EMBED_DIM", "3072")))
        self.top_k = int(top_k or int(os.getenv("DEFAULT_TOP_K", "5")))
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

    # ---------- RAG steps ----------
    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        qv = self._embed_query(question)
        hits = self.qc.search(
            collection_name=self.collection,
            query_vector=qv,
            limit=self.top_k if not self.use_rerank else self.rerank_top_k,
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
        # 자리만들기: 필요시 Vertex Ranking이나 cross-encoder 붙이기
        # 지금은 그대로 top_k 상위만 리턴
        return (docs or [])[: self.top_k]

    def generate(self, question: str, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return "관련된 정보를 찾을 수 없습니다."
        ctx = "\n\n".join(d["content"] for d in docs)
        prompt = f"""
      당신은 주식시장과 연금에 정통한 전문 애널리스트입니다.  
      아래 컨텍스트를 근거로 한국어 답변을 작성하세요.  
          당신은 주식시장과 연금에 정통한 전문 애널리스트입니다.  
          아래 컨텍스트를 근거로 한국어 답변을 작성하세요.  
      
        [작성 지침]  
        1. 답변은 **3단락 이상**으로 구성하세요.  
           - (1) 현황 요약  
           - (2) 원인/맥락 분석  
           - (3) 향후 전망 및 투자자 조언  
        2. **중요 포인트는 굵게**, 핵심 수치는 `코드블록 스타일`로 표시하세요.  
        3. 답변 중간에는 ▸, ✔, ✦ 같은 불릿 아이콘을 활용해 시각적으로 보기 좋게 정리하세요.  
        4. 마지막에 `---` 구분선을 넣고, 근거 기사 한 줄 요약을 첨부하세요.  
        5. 모호하거나 근거 없는 내용은 쓰지 말고 "관련된 정보를 찾을 수 없습니다."라고 답하세요.

        [답변예시]
        📊 현황 요약

        최근 엔비디아 주가가 30일 종가 기준 100일 이동평균선 아래로 하락하면서 기술적 지표상 부정적인 신호가 발생했습니다. 특히 변동성이 비트코인보다도 2배 이상 높아졌다는 점이 투자자 불안 심리를 키우고 있습니다.

        🔍 원인 및 배경
        
        ▸ 최근 2주간 빅테크 전반의 약세가 동반되며 엔비디아 주가에 부담이 되었고,
        ▸ 금리 인하 시 수혜주에 대한 관심이 분산된 것도 추가적인 하락 압력으로 작용했습니다.
        그러나 일부 전문가들은 이번 조정이 장기 성장성에는 큰 영향을 주지 않을 것이라 강조하고 있습니다.

        📈 향후 전망 & 투자자 조언

        ✔ 단기적으로는 추가 하락 가능성을 염두에 두어야 하며, 보수적 투자자라면 관망이 유리합니다.
        ✔ 반면, 장기적 관점에서는 매수 기회로 작용할 수 있다는 점에서 공격적 투자자에게는 긍정적일 수 있습니다.
        ✦ 따라서 리스크 관리와 분할 매수 전략이 균형 잡힌 접근법이 될 것입니다.

        📰 근거 기사: 엔비디아 주가가 30일 종가 기준 100일 이동평균선 아래로 떨어지며 기술적 부정 신호 발생
        
        [컨텍스트]
        {ctx}
        
        [질문]
        {question}
        """
        try:
            resp = self.gen_model.generate_content(prompt, generation_config={"temperature": 0.2, "max_output_tokens": 1200})
            return (resp.text or "").strip()
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {e}"

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
