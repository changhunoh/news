# ------------------------------------------------------------
# Qdrant + Vertex AI (Gemini / Embedding) 기반 RAG 서비스
#   - 단일 질문(answer)
#   - 5개 종목 병렬(Map) → 최종 통합(Reduce)
# ------------------------------------------------------------
# 수정 사항:
# - retrieve 메서드에서 stock이 주어진 경우, 먼저 stock 값 매칭으로 포인트를 선별(scroll 사용)한 후,
#   클라이언트 측에서 쿼리 벡터와의 유사도 계산(검색)을 수행하도록 구조 변경.
# - 이는 서버 측 필터 인덱스 의존성을 제거하고, 사용자의 요구(먼저 선별 후 검색)에 맞춤.
# - stock 필드가 list일 수 있음을 고려해 MatchAny 필터 사용 (string 필드에도 호환 가능하도록 테스트 필요, 하지만 Qdrant에서 array 필드에 MatchAny가 적합).
# - 클라이언트 측 유사도 계산을 위해 numpy 사용 (환경에 포함됨).
# - scroll 페이징 처리로 모든 매칭 포인트 가져옴 (종목당 문서 수가 많지 않다고 가정, e.g., <10k).
# - distance_mode에 따라 cosine/dot/euclid 지원 (euclid는 negative distance로 similarity처럼 취급).
# - 기존 에러 핸들링 제거, 항상 이 구조 사용.
# - 기타: import 추가 (numpy), hits 변환 로직 docs 변환과 일치시킴.
# ------------------------------------------------------------

import os, re, threading
from typing import List, Dict, Any, Optional, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

import numpy as np  # 클라이언트 측 유사도 계산용

# Streamlit 환경에서도/아니어도 동작하도록 안전 import
try:
    import streamlit as st
except Exception:  # streamlit 미설치 환경
    class _DummySt:
        secrets = {}
    st = _DummySt()

try:
    from google.oauth2 import service_account
except Exception:
    service_account = None  # 외부환경용


class RAGState(TypedDict):
    question: str
    documents: List[Dict[str, Any]]
    answer: Optional[str]


class NewsReportService:
    """Qdrant + Gemini 기반 RAG 서비스 (단일/다중 종목 대응)"""
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
        # ---- GCP 설정 ----
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not self.project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT required")

        # 서비스계정(st.secrets) → Credentials
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

        # ---- Qdrant 설정 ----
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

        # distance 모드 캐시
        self._dist_mode: Optional[str] = None

        # 프로세스 단일 Qdrant 클라 (가벼운 작업용)
        self.qc = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_key)

        # 모델 & 스레드-로컬 캐시 준비
        self._ensure_models()

    # ----------------- 내부 유틸 -----------------
    def _ensure_models(self):
        """스레드-로컬 모델 핸들 캐시"""
        if (not hasattr(self._thread_local, "embed_model")
            or getattr(self._thread_local, "embed_name", None) != self.embed_model_name):
            self._thread_local.embed_model = TextEmbeddingModel.from_pretrained(self.embed_model_name)
            self._thread_local.embed_name = self.embed_model_name

        if (not hasattr(self._thread_local, "gen_model")
            or getattr(self._thread_local, "gen_name", None) != self.gen_model_name):
            self._thread_local.gen_model = GenerativeModel(self.gen_model_name)
            self._thread_local.gen_name = self.gen_model_name

    def _tl_qc(self) -> QdrantClient:
        """스레드-로컬 Qdrant 클라이언트 (병렬 안전)"""
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

    # ----------------- 임베딩 -----------------
    def _embed_query(self, text: str) -> List[float]:
        inp = [TextEmbeddingInput(text=text or "", task_type="RETRIEVAL_QUERY")]
        return self.embed_model.get_embeddings(inp, output_dimensionality=self.embed_dim)[0].values

    # ----------------- Payload 텍스트 추출 -----------------
    @staticmethod
    def _extract_text_from_payload(payload: dict) -> str:
        """
        현재 스키마 기준:
          - 본문은 payload["text"] (string)
          - 'metadata' 래퍼 없음
        """
        if not isinstance(payload, dict):
            return ""
        if isinstance(payload.get("text"), str):
            return payload["text"]
        # 혹시 과거 스키마도 들어올 수 있으니 백업 경로 유지
        doc = payload.get("doc")
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            return doc.get("content") or doc.get("text") or doc.get("page_content") or ""
        return ""

    def _payload_matches_stock_root(self, payload: dict, stock: str) -> bool:
        if not payload or not stock:
            return False
        v = payload.get("stock")
        if isinstance(v, str):
            return v.upper() == stock.upper()
        if isinstance(v, list):
            return stock.upper() in {str(x).upper() for x in v}
        return False

    # ----------------- Retrieve -----------------
    def retrieve(self, question: str, stock: Optional[str] = None) -> List[Dict[str, Any]]:
        qv = self._embed_query(question)

        # distance 모드 캐시 (클라이언트 계산에 필요)
        if self._dist_mode is None:
            try:
                info = self._tl_qc().get_collection(self.collection)
                params = getattr(info.config, "params", None) or getattr(info, "config", None)
                vectors = getattr(params, "vectors", None)
                self._dist_mode = str(getattr(vectors, "distance", "")).lower() if vectors else "cosine"
            except Exception:
                self._dist_mode = "cosine"  # 기본 폴백

        if stock:
            # stock 필드가 list일 수 있으므로 MatchAny 사용 (string에도 적용 가능하도록)
            q_filter = Filter(must=[FieldCondition(key="stock", match=MatchAny(any=[stock]))])

            # 먼저 stock 매칭 포인트 선별 (scroll with filter, 페이징 처리)
            all_points = []
            offset = None
            while True:
                result = self._tl_qc().scroll(
                    collection_name=self.collection,
                    scroll_filter=q_filter,
                    limit=500,  # 배치 크기 (조정 가능, 종목당 총 문서 수에 따라)
                    with_payload=True,
                    with_vectors=True,
                    offset=offset,
                )
                points, next_offset = result
                all_points.extend(points)
                if next_offset is None:
                    break
                offset = next_offset

            # 선별된 포인트 중 벡터 검색 (클라이언트 측 유사도 계산)
            if all_points:
                valid_points = [p for p in all_points if p.vector is not None]
                if not valid_points:
                    hits = []
                else:
                    vectors = np.array([p.vector for p in valid_points])
                    qv_np = np.array(qv)

                    if self._dist_mode == "cosine":
                        vec_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                        qv_norm = np.linalg.norm(qv_np)
                        vectors = vectors / np.maximum(vec_norms, 1e-9)  # 0-norm 방지
                        qv_np = qv_np / max(qv_norm, 1e-9)
                        scores = np.dot(vectors, qv_np)
                    elif self._dist_mode == "dot":
                        scores = np.dot(vectors, qv_np)
                    elif self._dist_mode == "euclid":
                        scores = -np.sqrt(np.sum((vectors - qv_np) ** 2, axis=1))
                    else:
                        raise ValueError(f"Unsupported distance mode: {self._dist_mode}")

                    # 정렬 (cosine/dot: 내림차순, euclid: 오름차순(negative이므로 내림차순 효과))
                    is_descending = self._dist_mode in ("cosine", "dot")
                    sorted_indices = np.argsort(scores)[::-1 if is_descending else 1]
                    limit = self.rerank_top_k if self.use_rerank else self.top_k
                    top_indices = sorted_indices[:limit]

                    hits = []
                    for idx in top_indices:
                        p = valid_points[idx]
                        hit = type('Hit', (), {})()  # 간단 mock hit 객체
                        hit.id = p.id
                        hit.payload = p.payload
                        hit.score = float(scores[idx])
                        hits.append(hit)
            else:
                hits = []
        else:
            # stock 없음: 기존 벡터 검색 (필터 없이)
            hits = self._tl_qc().search(
                collection_name=self.collection,
                query_vector=qv,
                limit=self.top_k if not self.use_rerank else self.rerank_top_k,
                with_payload=True,
                with_vectors=False,
            )

        # hits → docs 변환
        docs: List[Dict[str, Any]] = []
        for h in hits:
            payload = h.payload or {}
            text = self._extract_text_from_payload(payload)
            md = {k: v for k, v in payload.items() if k != "text"}

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
        # TODO: Vertex Ranking / Cross-Encoder 붙일 수 있음. 지금은 top_k 자르기만.
        return (docs or [])[: self.top_k]

    # ----------------- Generate -----------------
    def generate(self, question: str, docs: List[Dict[str, Any]], stock: Optional[str] = None) -> str:
        if not docs:
            return "관련된 정보를 찾을 수 없습니다."

        # 컨텍스트 길이 관리(상위 5개 발췌)
        def _trunc(s: str, limit=1600):
            s = s or ""
            return s if len(s) <= limit else s[:limit] + "..."

        ctx = "\n\n---\n\n".join(_trunc(d["content"]) for d in docs[:5])

        prompt = f"""
당신은 주식시장과 연금에 정통한 전문 애널리스트입니다.
아래 컨텍스트를 바탕으로 {stock or "대상"} 종목의 가격 결정에 중요한 핵심정보를 요약하세요.

[작성 지침]
1) 답변은 3단락 이상
 (1) 현황 요약
 (2) 원인/맥락
 (3) 향후 전망 및 투자자 조언
2) 근거 없는 내용은 쓰지 말 것(모호하면 '관련된 정보를 찾을 수 없습니다.')

[대상 종목]
{stock or "N/A"}

[컨텍스트 발췌]
{ctx}

[질문]
{question}
"""
        try:
            resp = self.gen_model.generate_content(
                prompt,
                generation_config={"temperature": 0.2},
            )
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {e}"

    # ----------------- Public: 단일 질문 -----------------
    def answer(self, question: str, stock: Optional[str] = None) -> Dict[str, Any]:
        docs = self.retrieve(question, stock=stock)
        docs = self.rerank(question, docs) if self.use_rerank else docs[: self.top_k]
        ans = self.generate(question, docs, stock=stock)  # ← stock 전달
        return {"answer": ans, "source_documents": docs}

    def retrieve_only(self, question: str, top_k: Optional[int] = None, stock: Optional[str] = None) -> List[Dict[str, Any]]:
        prev_top_k, self.top_k = self.top_k, (top_k or self.top_k)
        try:
            docs = self.retrieve(question, stock=stock)
            return docs[: (top_k or self.top_k)]
        finally:
            self.top_k = prev_top_k

    # ======================================================
    # ===============  다중 종목 Map → Reduce  =============
    # ======================================================
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

    def answer_multi_stocks(
        self,
        stocks: List[str],
        template: Optional[str] = None,
        max_workers: int = 5,
    ) -> List[Dict[str, Any]]:
        """여러 종목 동시 처리(Map) — 입력 순서 보존"""
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
        # 간단 소스 요약(제목/URL) 모으기
        def _fmt_sources(docs: List[Dict[str, Any]]) -> List[str]:
            out = []
            for d in docs[:3]:
                md = d.get("metadata", {}) or {}
                title = md.get("title") or md.get("headline") or md.get("doc_title") or ""
                url = md.get("url") or md.get("link") or md.get("source_url") or ""
                if title and url:
                    out.append(f"{title} — {url}")
                elif title:
                    out.append(title)
                elif url:
                    out.append(url)
            return out

        lines = []
        source_lines = []
        for r in per_stock_results:
            stock = r["stock"]
            ans = (r.get("answer") or "").strip()
            lines.append(f"### [{stock}] 부분답\n{ans}\n")
            for s in _fmt_sources(r.get("source_documents", [])):
                source_lines.append(f"[{stock}] {s}")

        joined_parts = "\n\n".join(lines)
        # 순서 보존 중복 제거
        seen, dedup = set(), []
        for s in source_lines:
            if s not in seen:
                seen.add(s)
                dedup.append(s)
        joined_sources = "\n".join(dedup[:12])

        prompt = f"""
당신은 증권사 리서치센터장입니다.
아래 각 종목의 부분 답변을 취합하여, 공통 질의("{base_template}")에 대한 **종합 리포트**를 작성하세요.

[요구사항]
1) 종목별 핵심 뉴스와 가격 영향 경로를 비교 정리(긍/부정, 단기/중기)
2) 공통 테마(금리, 환율, 공급망, 규제 등) 식별 및 교차영향 설명
3) 종목별 리스크/촉발요인, 모니터링 지표 제시
4) 결론: 포트폴리오 관점 제언(오버웨이트/뉴트럴/언더웨이트 등 사용 가능)
5) 수치는 `백틱`으로, 핵심 포인트는 **굵게**, 불릿 적절 활용
6) 모호하면 '관련된 정보를 찾을 수 없습니다.'라고 분명히 표기

[종목별 부분답 모음]
{joined_parts}

[근거 기사/자료 후보]
{joined_sources}
"""
        try:
            resp = self.gen_model.generate_content(
                prompt,
                generation_config={"temperature": 0.0},
            )
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            return f"최종 통합 생성 오류: {e}"

    def answer_5_stocks_and_reduce(
        self,
        stocks: List[str],   # 5개 권장(3~8개도 동작)
        template: Optional[str] = None,  # 기본: "{stock} 관련해서 종목의 가격에 중요한 뉴스는?"
        max_workers: int = 5,
    ) -> Dict[str, Any]:
        template = template or "{stock} 관련해서 종목의 가격에 중요한 뉴스는?"
        per_stock = self.answer_multi_stocks(stocks, template=template, max_workers=max_workers)
        final = self._reduce_across_stocks(template, per_stock)
        return {
            "base_template": template,
            "stocks": stocks,
            "results": per_stock,   # [{stock, question, answer, source_documents}, ...]
            "final_report": final,  # 종합 리포트
        }


# ------------------------------------------------------------
# 간단 실행 예시 (환경변수 세팅 필요)
#   - GOOGLE_CLOUD_PROJECT
#   - (옵션) GOOGLE_CLOUD_LOCATION
#   - QDRANT_URL, QDRANT_API_KEY
#   - COLLECTION_NAME (기본: stock_news)
#   - EMBED_MODEL_NAME, GENAI_MODEL_NAME, EMBED_DIM, DEFAULT_TOP_K, RERANK_TOP_K
# ------------------------------------------------------------
if __name__ == "__main__":
    # 예: 5개 종목 동시 처리 후 통합
    stocks = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]  # 원하는 티커/심볼 리스트로 교체
    svc = NewsReportService(top_k=5, use_rerank=False)

    result = svc.answer_5_stocks_and_reduce(stocks)
    print("=== [FINAL REPORT] ===\n")
    print((result.get("final_report") or "")[:4000])  # 길면 앞부분만 출력

    # 종목별 부분답 미리보기
    for r in result.get("results", []):
        print(f"\n--- [{r.get('stock','')}] ---")
        print((r.get("answer") or "")[:1200])
