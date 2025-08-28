# news_report_service.py
# ------------------------------------------------------------
# Qdrant + Vertex AI (Gemini / Embedding) 기반 RAG 서비스
#   - stock으로 먼저 서버-사이드 필터 → 필터된 서브셋에서 벡터검색
#   - 단일 질문 + 5개 종목 병렬(Map) → 최종 통합(Reduce)
# ------------------------------------------------------------

import os, threading, re
from typing import List, Dict, Any, Optional, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel, SafetySetting, HarmCategory, HarmBlockThreshold, GenerationConfig

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, PayloadSchemaType
)

# Streamlit이 없을 수도 있으니 안전 import
try:
    import streamlit as st
except Exception:
    class _DummySt:
        secrets = {}
    st = _DummySt()

# 서비스계정 import (환경에 따라 없을 수 있음)
try:
    from google.oauth2 import service_account
except Exception:
    service_account = None


class RAGState(TypedDict):
    question: str
    documents: List[Dict[str, Any]]
    answer: Optional[str]


class NewsReportService:
    """루트 payload 스키마(text/stock/...) 기준. stock pre-filter → 벡터검색."""
    _thread_local = threading.local()

    def __init__(
        self,
        project: Optional[str] = None,
        location: str = "us-central1",
        qdrant_url: Optional[str] = None,
        qdrant_key: Optional[str] = None,
        collection: str = "stock_news",
        embed_model_name: str = "gemini-embedding-001",
        gen_model_name: str = "gemini-2.5-pro", #최종 리포트 모델
        rag_model_name: str = "gemini-1.5-pro", # RAG 모델
        embed_dim: int = 3072,
        top_k: int = 1,
        rerank_top_k: int = 1,
        use_rerank: bool = False,
    ):
        # ---- GCP & Vertex init ----

        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not self.project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT required")
        sa_info = None
        try:
            sa_info = getattr(st, "secrets", {}).get("gcp_service_account", None)  # type: ignore[attr-defined]
        except Exception:
            sa_info = None

        creds = None
        if sa_info and service_account is not None:
            creds = service_account.Credentials.from_service_account_info(
                dict(sa_info),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        vertexai.init(project=self.project, location=self.location, credentials=creds)

        # ---- Qdrant ----
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_key = qdrant_key or os.getenv("QDRANT_API_KEY")
        if not (self.qdrant_url and self.qdrant_key):
            raise RuntimeError("QDRANT_URL / QDRANT_API_KEY required")

        self.collection = collection or os.getenv("COLLECTION_NAME", "stock_news")
        self.embed_model_name = embed_model_name or os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")
        self.gen_model_name = gen_model_name or os.getenv("GENAI_MODEL_NAME", "gemini-2.5-pro")
        self.rag_model_name = rag_model_name or os.getenv("RAG_MODEL_NAME",'gemini-1.5-pro')
        self.embed_dim = int(embed_dim or int(os.getenv("EMBED_DIM", "3072")))
        self.top_k = int(top_k or int(os.getenv("DEFAULT_TOP_K", "1")))
        self.rerank_top_k = int(rerank_top_k or int(os.getenv("RERANK_TOP_K", "1")))
        self.use_rerank = use_rerank
        print(gen_model_name)
        print(rag_model_name)

        self._dist_mode: Optional[str] = None

        # 프로세스 전역 클라이언트 + 스레드-로컬 클라이언트
        self.qc = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_key)

        self.gen_model_name = gen_model_name
        self.rag_model_name = rag_model_name

        # 모델 핸들 준비
        self._ensure_models()

        # 필터를 쓰려면 인덱스가 필요 → 한 번 보장
        self._ensure_stock_index()

    # ----------------- 내부 유틸 -----------------
    def _ensure_models(self):
        if (not hasattr(self._thread_local, "embed_model")
            or getattr(self._thread_local, "embed_name", None) != self.embed_model_name):
            self._thread_local.embed_model = TextEmbeddingModel.from_pretrained(self.embed_model_name)
            self._thread_local.embed_name = self.embed_model_name

        if not hasattr(self._thread_local, "rag_model") or getattr(self._thread_local, "rag_model_name", None) != self.rag_model_name:
            self._thread_local.rag_model = GenerativeModel(self.rag_model_name)
            self._thread_local.rag_model_name = self.rag_model_name

        if (not hasattr(self._thread_local, "gen_model")
            or getattr(self._thread_local, "gen_name", None) != self.gen_model_name):
            self._thread_local.gen_model = GenerativeModel(self.gen_model_name)
            self._thread_local.gen_name = self.gen_model_name

    def _tl_qc(self) -> QdrantClient:
        if not hasattr(self._thread_local, "qc"):
            self._thread_local.qc = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_key)
        return self._thread_local.qc

    @property
    def embed_model(self) -> TextEmbeddingModel:
        self._ensure_models()
        return self._thread_local.embed_model

    @property
    def gen_model(self) -> GenerativeModel:
        self._ensure_models()
        return self._thread_local.gen_model

    @property
    def rag_model(self) -> GenerativeModel:
        self._ensure_models()
        return self._thread_local.rag_model

    def _ensure_stock_index(self) -> None:
        """루트 'stock'에 keyword 인덱스 보장(없으면 생성)."""
        try:
            self.qc.create_payload_index(
                collection_name=self.collection,
                field_name="stock",
                field_schema=PayloadSchemaType.KEYWORD,  # 또는 "keyword"
                wait=True,
            )
        except Exception:
            # 이미 있거나 권한 문제면 조용히 패스
            pass

    # ----------------- 임베딩 & 텍스트 -----------------
    def _embed_query(self, text: str) -> List[float]:
        inp = [TextEmbeddingInput(text=text or "", task_type="RETRIEVAL_QUERY")]
        return self.embed_model.get_embeddings(inp, output_dimensionality=self.embed_dim)[0].values

    @staticmethod
    def _extract_text_from_payload(payload: dict) -> str:
        """현재 스키마: 본문은 payload['text'] (metadata 래퍼 없음)"""
        if not isinstance(payload, dict):
            return ""
        if isinstance(payload.get("text"), str):
            return payload["text"]
        # 호환(이전 스키마)
        doc = payload.get("doc")
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            return doc.get("content") or doc.get("text") or doc.get("page_content") or ""
        return ""

    # ----------------- Retrieve (stock pre-filter → vector search) -----------------
    def retrieve(self, question: str, stock: Optional[str] = None) -> List[Dict[str, Any]]:
        qv = self._embed_query(question)
        want = self.rerank_top_k if self.use_rerank else self.top_k

        q_filter = None
        if stock:
            self._ensure_stock_index()
            q_filter = Filter(must=[FieldCondition(key="stock", match=MatchValue(value=str(stock)))])

        # 필터된 서브셋에서 벡터검색
        try:
            hits = self._tl_qc().search(
                collection_name=self.collection,
                query_vector=qv,
                limit=want,
                with_payload=True,
                with_vectors=False,
                query_filter=q_filter,
            )
        except Exception as e:
            # 인덱스 이슈 등 → 한 번 더 인덱스 보장 후 재시도
            if stock and ("Index required" in str(e) or "Bad request" in str(e)):
                self._ensure_stock_index()
                hits = self._tl_qc().search(
                    collection_name=self.collection,
                    query_vector=qv,
                    limit=want,
                    with_payload=True,
                    with_vectors=False,
                    query_filter=q_filter,
                )
            else:
                raise

        # distance 모드 캐시
        if self._dist_mode is None:
            try:
                info = self._tl_qc().get_collection(self.collection)
                params = getattr(info.config, "params", None) or getattr(info, "config", None)
                vectors = getattr(params, "vectors", None)
                self._dist_mode = str(getattr(vectors, "distance", "")).lower() if vectors else ""
            except Exception:
                self._dist_mode = ""

        # hits → docs
        docs: List[Dict[str, Any]] = []
        for h in hits:
            payload = h.payload or {}
            text = self._extract_text_from_payload(payload)
            md = {k: v for k, v in payload.items() if k != "text"}  # 루트 payload 전체를 메타데이터로
            raw = getattr(h, "score", None)
            distance = float(raw) if raw is not None else None
            docs.append({
                "id": str(getattr(h, "id", "")),
                "content": text,
                "metadata": md,
                "score": distance,
                "distance": distance,
                "distance_mode": self._dist_mode,
            })
        return docs

    # ----------------- (선택) 리랭크 -----------------
    def rerank(self, question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return (docs or [])[: self.top_k]

    def _gen_config(self, temperature: float = 0.2) -> GenerationConfig:
        # 필요 시 여기서 top_p/top_k 도 조절 가능
        return GenerationConfig(temperature=temperature)


    
    # ----------------- Generate -----------------
    def generate(self, question: str, docs: List[Dict[str, Any]], stock: Optional[str] = None) -> str:
        if not docs:
            return "관련된 정보를 찾을 수 없습니다."
        def _trunc(s: str, limit=1600):
            s = s or ""
            return s if len(s) <= limit else s[:limit] + "..."
        ctx = "\n\n---\n\n".join(_trunc(d["content"]) for d in docs[:5])

        prompt = f"""
당신은 주식시장과 연금에 정통한 전문 애널리스트입니다.
아래 컨텍스트를 바탕으로 {stock} 종목의 가격 결정에 중요한 핵심정보를 요약하세요.
문단 2~3개, 전체 350~450단어 내, 수치는 `백틱`으로 표시할 것.

[작성 지침]
1) 답변은 3단락 이상
2) 근거 없는 내용은 쓰지 말 것(모호하면 '관련된 정보를 찾을 수 없습니다.')

[대상 종목]
{stock}

[컨텍스트 발췌]
{ctx}

[질문]
{question}
"""
        try:
            # rag_model (1.5 pro) 사용
            resp = self._thread_local.rag_model.generate_content(
                prompt,
                generation_config={"temperature": 0.0},
                safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
                
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {e}"

    # ----------------- Public APIs -----------------
    def answer(self, question: str, stock: Optional[str] = None) -> Dict[str, Any]:
        docs = self.retrieve(question, stock=stock)
        docs = self.rerank(question, docs) if self.use_rerank else docs[: self.top_k]
        ans = self.generate(question, docs, stock=stock)
        return {"answer": ans, "source_documents": docs}

    def retrieve_only(self, question: str, top_k: Optional[int] = None, stock: Optional[str] = None) -> List[Dict[str, Any]]:
        prev_top_k, self.top_k = self.top_k, (top_k or self.top_k)
        try:
            docs = self.retrieve(question, stock=stock)
            return docs[: (top_k or self.top_k)]
        finally:
            self.top_k = prev_top_k

    # ----------------- 다중 종목 Map → Reduce -----------------
    def _stock_question(self, stock: str, template: Optional[str] = None) -> str:
        template = template or "{stock} 관련해서 종목의 가격에 중요한 뉴스는?"
        return template.format(stock=stock)

    def answer_for_stock(self, stock: str, template: Optional[str] = None) -> Dict[str, Any]:
        q = self._stock_question(stock, template)
        res = self.answer(q, stock=stock)
        return {
            "stock": stock,
            "question": q,
            "answer": res["answer"],
            "source_documents": res.get("source_documents", []),
        }

    def answer_multi_stocks(self, stocks: List[str], template: Optional[str] = None, max_workers: int = 5) -> List[Dict[str, Any]]:
        results: List[Optional[Dict[str, Any]]] = [None] * len(stocks)
        def _one(i: int, s: str) -> tuple[int, Dict[str, Any]]:
            return i, self.answer_for_stock(s, template=template)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_one, i, s) for i, s in enumerate(stocks)]
            for fut in as_completed(futs):
                i, r = fut.result()
                results[i] = r
        return [r for r in results if r is not None]

    def _reduce_across_stocks(self, base_template: str, per_stock_results: List[Dict[str, Any]]) -> str:
        def _fmt_sources(docs: List[Dict[str, Any]]) -> List[str]:
            out = []
            for d in docs[:3]:
                md = d.get("metadata", {}) or {}
                title = md.get("title") or md.get("headline") or md.get("doc_title") or md.get("doc_id") or ""
                url = md.get("url") or md.get("link") or md.get("source_url") or ""
                if title and url: out.append(f"{title} — {url}")
                elif title:       out.append(title)
                elif url:         out.append(url)
            return out
    
        lines, source_lines = [], []
        for r in per_stock_results:
            stock = r["stock"]
            ans = (r.get("answer") or "").strip()
            lines.append(f"### [{stock}] 부분답\n{ans}\n")
            for s in _fmt_sources(r.get("source_documents", [])):
                source_lines.append(f"[{stock}] {s}")
    
        # 순서 보존 dedup
        seen, dedup = set(), []
        for s in source_lines:
            if s not in seen:
                seen.add(s); dedup.append(s)
    
        # ✅ f-string 안에 백슬래시가 들어가던 join을 미리 계산
        parts_joined   = "\n\n".join(lines)
        sources_joined = "\n".join(dedup[:12])
    
        prompt = f"""
    당신은 증권사 리서치센터장입니다.
    아래 각 종목의 부분 답변을 취합하여, 공통 질의({base_template})에 대한 **종합 리포트**를 작성하세요.
    
    [요구사항]
    1) 종목별 핵심 뉴스와 가격 영향 경로를 비교 정리(긍/부정, 단기/중기)
    2) 공통 테마(금리, 환율, 공급망, 규제 등) 식별 및 교차영향 설명
    3) 종목별 리스크/촉발요인, 모니터링 지표 제시
    4) 결론: 포트폴리오 관점 제언(오버웨이트/뉴트럴/언더웨이트 등 사용 가능)
    5) 수치는 `백틱`으로, 핵심 포인트는 **굵게**, 불릿 적절 활용
    6) 모호하면 '관련된 정보를 찾을 수 없습니다.'라고 분명히 표기
    
    [종목별 부분답 모음]
    {parts_joined}
    
    [근거 기사/자료 후보]
    {sources_joined}
    """
        try:
            resp = self._thread_local.rag_model.generate_content(
                prompt, 
                generation_config={"temperature": 0.25},
                safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            }
            )
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            return f"최종 통합 생성 오류: {e}"

    def answer_5_stocks_and_reduce(self, stocks: List[str], template: Optional[str] = None, max_workers: int = 5) -> Dict[str, Any]:
        template = template or "{stock} 관련해서 종목의 가격에 중요한 뉴스는?"
        per_stock = self.answer_multi_stocks(stocks, template=template, max_workers=max_workers)
        final = self._reduce_across_stocks(template, per_stock)
        return {
            "base_template": template,
            "stocks": stocks,
            "results": per_stock,
            "final_report": final,
        }

    # ---- 진단용: 해당 종목 문서 수 ----
    def count_by_stock(self, stock: str) -> int:
        self._ensure_stock_index()
        try:
            res = self.qc.count(
                collection_name=self.collection,
                count_filter=Filter(must=[FieldCondition(key="stock", match=MatchValue(value=str(stock)))]),
                exact=True,
            )
            return int(getattr(res, "count", 0))
        except Exception:
            return 0







