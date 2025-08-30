import os, re, threading
from typing import List, Dict, Any, Optional, TypedDict
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from qdrant_client import QdrantClient
from google.oauth2 import service_account
from collections.abc import Generator
from vertexai.generative_models import Candidate
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
        gen_model_name: str = "gemini-2.5-flash-lite",
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
        self.gen_model_name = gen_model_name or os.getenv("GENAI_MODEL_NAME", "gemini-2.5-flash-lite")
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
    def _extract_text_from_payload(self, payload: dict) -> str:
        """
        payload["doc"]가 문자열이거나, dict(예: {"content": "...", "text": "...", ...})일 수 있으니 모두 커버
        """
        if not isinstance(payload, dict):
            return ""
        doc = payload.get("doc")
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            # 흔한 텍스트 키들 우선순위
            return doc.get("content") or doc.get("text") or doc.get("page_content") or ""
        return ""
    
    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        qv = self._embed_query(question)
        hits = self.qc.search(
            collection_name=self.collection,
            query_vector=qv,
            limit=self.top_k if not self.use_rerank else self.rerank_top_k,
            with_payload=True,
            with_vectors=False,
        )
    
        # (선택) distance 모드 파악
        dist_mode = getattr(self, "_dist_mode", None)
        if dist_mode is None:
            try:
                info = self.qc.get_collection(self.collection)
                params = getattr(info.config, "params", None) or getattr(info, "config", None)
                vectors = getattr(params, "vectors", None)
                dist_mode = str(getattr(vectors, "distance", "")).lower() if vectors else ""
            except Exception:
                dist_mode = ""
            self._dist_mode = dist_mode
    
        docs: List[Dict[str, Any]] = []
        for h in hits:
            payload = h.payload or {}
            text = self._extract_text_from_payload(payload)
    
            # 메타데이터: payload["metadata"] 최우선, 없으면 payload에서 doc 제외
            md = {}
            if isinstance(payload.get("metadata"), dict):
                md = dict(payload["metadata"])
            else:
                md = {k: v for k, v in payload.items() if k not in ("doc", "metadata")}
    
            raw = getattr(h, "score", None)  # Qdrant는 보통 distance를 score로 반환
            distance = float(raw) if raw is not None else None
            similarity = None
            if distance is not None and "cosine" in dist_mode:
                similarity = distance
    
            docs.append({
                "id": str(getattr(h, "id", "")),
                "content": text,            # ✅ 이제 doc 기반 본문
                "metadata": md,             # ✅ metadata 그대로
                "score": similarity if similarity is not None else (float(raw) if raw is not None else None),
                "distance": distance,
                "distance_mode": dist_mode,
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
      아래 컨텍스트를 근거로 사용자의 질문 의도에 맞는 한국어 답변을 충실하게 작성하세요.
      답변을 작성 시 아래 지침을 반드시 지켜주세요.
      
        [작성 지침]  
        1. 답변은 **3단락 이상**으로 구성하세요.  
        2. **중요 포인트는 굵게**, 핵심 수치는 `코드블록 스타일`로 표시하세요.  
        3. 답변 중간에는 ▸, ✔, ✦ 같은 불릿 아이콘을 활용해 시각적으로 보기 좋게 정리하세요.  
        4. 마지막에 `---` 구분선을 넣고, 근거 기사 한 줄 요약을 첨부하세요.  
        5. 모호하거나 근거 없는 내용은 쓰지 말고 "관련된 정보를 찾을 수 없습니다."라고 답하세요.
        
        [컨텍스트]
        {ctx}
        
        [질문]
        {question}
        """
        try:
            resp = self.gen_model.generate_content(prompt, generation_config={"temperature": 0.2})
            return (resp.text or "").strip()
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {e}"

    def generate_stream(self, question: str, docs: List[Dict[str, Any]]) -> Generator[str, None, None]:
        """Gemini 스트리밍 제너레이터: chunk 문자열을 yield"""
        if not docs:
            yield "관련된 정보를 찾을 수 없습니다."
            return

        ctx = "\n\n".join(d["content"] for d in docs)
        prompt = f"""
      당신은 주식시장과 연금에 정통한 전문 애널리스트입니다.  
      아래 컨텍스트를 근거로 사용자의 질문 의도에 맞는 한국어 답변을 충실하게 작성하세요.
      답변을 작성 시 아래 지침을 반드시 지켜주세요.
      
        [작성 지침]  
        1. 답변은 **3단락 이상**으로 구성하세요.  
        2. **중요 포인트는 굵게**, 핵심 수치는 `코드블록 스타일`로 표시하세요.  
        3. 답변 중간에는 ▸, ✔, ✦ 같은 불릿 아이콘을 활용해 시각적으로 보기 좋게 정리하세요.  
        4. 마지막에 `---` 구분선을 넣고, 근거 기사 한 줄 요약을 첨부하세요.  
        5. 모호하거나 근거 없는 내용은 쓰지 말고 "관련된 정보를 찾을 수 없습니다."라고 답하세요.

        [컨텍스트]
        {ctx}
        
        [질문]
        {question}
        """

        try:
            responses = self.gen_model.generate_content(
                prompt,
                stream=True,
                generation_config={"temperature": 0.2},
            )

            # 일부 SDK 버전에서 응답이 후보/파츠로 나뉘거나 response.text에 델타가 들어옵니다.
            for res in responses:
                # 1) 가장 간단: 델타 텍스트가 있으면 그대로
                if getattr(res, "text", None):
                    yield res.text
                    continue

                # 2) 후보/파츠에서 텍스트 축출
                chunk = []
                for cand in (getattr(res, "candidates", None) or []):
                    if isinstance(cand, Candidate) and getattr(cand, "content", None):
                        for part in getattr(cand.content, "parts", []) or []:
                            t = getattr(part, "text", None)
                            if t:
                                chunk.append(t)
                if chunk:
                    yield "".join(chunk)

        except Exception as e:
            yield f"\n\n(스트리밍 중 오류가 발생했습니다: {e})"

    # ---------- public APIs ----------
    def answer(self, question: str) -> Dict[str, Any]:
        docs = self.retrieve(question)
        docs = self.rerank(question, docs) if self.use_rerank else docs[: self.top_k]
        ans = self.generate(question, docs)
        return {"answer": ans, "source_documents": docs}

    def answer_stream(self, question: str):
        """retrieve → (옵션) rerank → stream 생성기를 반환"""
        docs = self.retrieve(question)
        docs = self.rerank(question, docs) if self.use_rerank else docs[: self.top_k]
        return self.generate_stream(question, docs)
    

    def retrieve_only(self, question: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        prev_top_k, self.top_k = self.top_k, (top_k or self.top_k)
        try:
            docs = self.retrieve(question)
            return docs[: (top_k or self.top_k)]
        finally:
            self.top_k = prev_top_k

    # ---------- 진단함수 ----------
    def diagnose(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
    
        # 1) 컬렉션/벡터 구성
        try:
            col = self.qc.get_collection(self.collection)
            cfg = getattr(col, "config", None) or col
            vectors = getattr(cfg, "vectors", None)
            if hasattr(vectors, "size"):
                info["vector_size"] = vectors.size
                info["distance"] = str(getattr(vectors, "distance", ""))
                info["named_vectors"] = None
            elif isinstance(vectors, dict):
                info["named_vectors"] = {k: {"size": v.size, "distance": str(v.distance)} for k, v in vectors.items()}
            else:
                info["vectors_raw"] = str(vectors)
        except Exception as e:
            info["collection_error"] = f"{e}"
    
        # 2) 포인트 존재여부 + payload 키 미리보기
        try:
            scrolled = self.qc.scroll(
                collection_name=self.collection,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            pts = getattr(scrolled, "points", None)
            if pts and len(pts) > 0:
                p = pts[0]
                payload = p.payload or {}
                info["has_points"] = True
                info["sample_payload_keys"] = list(payload.keys())
                # 본문 후보 키 몇 개 미리보기
                preview_keys = [k for k in ("doc", "page_content", "content", "text") if k in payload]
                preview = {}
                for k in preview_keys[:3]:
                    v = payload.get(k)
                    preview[k] = (v[:80] + "...") if isinstance(v, str) and len(v) > 80 else v
                info["payload_preview"] = preview
            else:
                info["has_points"] = False
        except Exception as e:
            info["scroll_error"] = f"{e}"
    
        # 3) 쿼리 임베딩 차원
        info["embed_dim_config"] = self.embed_dim
    
        # 4) 환경 변수/엔드포인트
        info["qdrant_url"] = self.qdrant_url
        info["collection"] = self.collection
        info["vector_name"] = getattr(self, "vector_name", None)
    
        return info

