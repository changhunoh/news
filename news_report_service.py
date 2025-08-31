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
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
# sending mail
import markdown
import mailing

load_dotenv()
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
    #_thread_local = threading.local()

    def __init__(
        self,
        project: Optional[str] = None,
        location: str = "us-central1",
        qdrant_url: Optional[str] = None,
        qdrant_key: Optional[str] = None,
        collection: str = "stock_news",
        embed_model_name: str = "gemini-embedding-001",
        gen_model_name: str = "gemini-2.5-pro", #최종 리포트 모델
        rag_model_name: str = "gemini-2.5-flash-lite", # RAG 모델
        embed_dim: int = 3072,
        top_k: int = 30,
        rerank_top_k: int = 20,
        use_rerank: bool = False,
    ):
        # ---- GCP & Vertex init (보안 관련 설정) ----

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

        # ---- Qdrant (백터 DB 설정) ----
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_key = qdrant_key or os.getenv("QDRANT_API_KEY")
        if not (self.qdrant_url and self.qdrant_key):
            raise RuntimeError("QDRANT_URL / QDRANT_API_KEY required")
        # ---- Model Config  ----
        self.collection = collection or os.getenv("COLLECTION_NAME", "stock_news")
        self.embed_model_name = embed_model_name or os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")
        self.gen_model_name = gen_model_name or os.getenv("GENAI_MODEL_NAME", "gemini-2.5-pro")
        self.rag_model_name = rag_model_name or os.getenv("RAG_MODEL_NAME",'gemini-2.5-flash-lite')
        self.embed_dim = int(embed_dim or int(os.getenv("EMBED_DIM", "3072")))
        self.top_k = int(top_k or int(os.getenv("DEFAULT_TOP_K", "30")))
        self.rerank_top_k = int(rerank_top_k or int(os.getenv("RERANK_TOP_K", "20")))
        self.use_rerank = use_rerank
        
        self._dist_mode: Optional[str] = None

        # Qdrant 벡터DB 
        self.qc = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_key)
        # self.gen_model_name = gen_model_name
        # self.rag_model_name = rag_model_name

        # 모델 핸들 준비
        # self._ensure_models()
    
        # 모델 공유 핸들
        self.embed_model = TextEmbeddingModel.from_pretrained(self.embed_model_name)
        self.rag_model   = GenerativeModel(self.rag_model_name)
        self.gen_model   = GenerativeModel(self.gen_model_name)
        # 필터를 쓰려면 인덱스가 필요 → 한 번 보장
        self._ensure_stock_index()

    # ----------------- 내부 유틸 -----------------
    # def _ensure_models(self):
    #     if (not hasattr(self._thread_local, "embed_model")
    #         or getattr(self._thread_local, "embed_name", None) != self.embed_model_name):
    #         self._thread_local.embed_model = TextEmbeddingModel.from_pretrained(self.embed_model_name)
    #         self._thread_local.embed_name = self.embed_model_name

    #     if not hasattr(self._thread_local, "rag_model") or getattr(self._thread_local, "rag_model_name", None) != self.rag_model_name:
    #         self._thread_local.rag_model = GenerativeModel(self.rag_model_name)
    #         self._thread_local.rag_model_name = self.rag_model_name

    #     if (not hasattr(self._thread_local, "gen_model")
    #         or getattr(self._thread_local, "gen_name", None) != self.gen_model_name):
    #         self._thread_local.gen_model = GenerativeModel(self.gen_model_name)
    #         self._thread_local.gen_name = self.gen_model_name

    # def _tl_qc(self) -> QdrantClient:
    #     if not hasattr(self._thread_local, "qc"):
    #         self._thread_local.qc = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_key)
    #     return m if m is not None else self._embed_model_shared

    # @property
    # def embed_model(self) -> TextEmbeddingModel:
    #     self._ensure_models()
    #     m = getattr(self._thread_local, "embed_model", None)
    #     return self._thread_local.embed_model

    # @property
    # def rag_model(self) -> GenerativeModel:
    #     self._ensure_models()
    #     m = getattr(self._thread_local, "rag_model", None)
    #     return m if m is not None else self._rag_model_shared
    
    # @property
    # def gen_model(self) -> GenerativeModel:
    #     self._ensure_models()
    #     m = getattr(self._thread_local, "gen_model", None)
    #     return m if m is not None else self._gen_model_shared

    today = datetime.now().strftime("%Y-%m-%d")
    
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
            hits = self.qc.search(
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
                hits = self.qc.search(
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
                info = self.qc.get_collection(self.collection)
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
        #return (docs or [])[: self.top_k]
        return (docs or [])[: self.rerank_top_k]
    
    def _safe_settings(self):
        return [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                          threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                          threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                          threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                          threshold=HarmBlockThreshold.BLOCK_NONE),
        ]    
    def _gen_config(self, temperature: float = 0.2) -> GenerationConfig:
        # 필요 시 여기서 top_p/top_k 도 조절 가능
        return GenerationConfig(temperature=temperature)

    def _extract_text(self, resp) -> str:
        # vertexai SDK 응답을 최대한 안전하게 텍스트로 변환
        try:
            if getattr(resp, "text", None):
                return resp.text.strip()
            cands = getattr(resp, "candidates", None) or []
            if not cands:
                return ""
            first = cands[0]
            content = getattr(first, "content", None)
            parts = getattr(content, "parts", None) or []
            out = []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    out.append(t)
            return "\n".join(out).strip()
        except Exception:
            return ""
    
    # ----------------- Generate (Rag 기능 수행) -----------------
    def generate(self, question: str, docs: List[Dict[str, Any]], stock: Optional[str] = None) -> str:
        #self._ensure_models()
        if not docs:
            return "관련된 정보를 찾을 수 없습니다."
        #def _trunc(s: str, limit=1600):
            #s = s or ""
            #return s if len(s) <= limit else s[:limit] + "..."
        #ctx = "\n\n---\n\n".join(_trunc(d["content"]) for d in docs[:10])
        ctx = "\n\n---\n\n".join(
            f"{i+1}. {d['content']}" 
            for i,d in enumerate(docs))

        prompt = f"""
당신은 주식시장과 연금에 정통한 전문 애널리스트입니다.
아래 컨텍스트를 바탕으로 {stock} 종목의 가격 결정에 중요한 뉴스 중 
10개를 골라 핵심정보를 요약하고, 종목의 전망에 대한 😊긍정 또는 😥부정을 판단해주세요.
중요 정보의 유실이 발생하지 않도록 유의해주세요.

[작성 지침]
1) 답변은 3단락 이상
2) 근거 없는 내용은 쓰지 말 것(모호하면 '관련된 정보를 찾을 수 없습니다.')
3) 😊긍정 또는 😥부정에 대한 판단 후 핵심정보 요약 제공

[대상 종목]
{stock}

[컨텍스트 발췌]
{ctx}

[질문]
{question}
"""
        try:
            # rag_model (2.5 flahs light) 사용
            resp = self.rag_model.generate_content(
                prompt,
                generation_config=self._gen_config(temperature=0.0),
                safety_settings=self._safe_settings(),
            )
            text = self._extract_text(resp)
            if not text:
                return "답변 생성에 실패했습니다. 안전필터 또는 토큰 한도에 의해 응답이 비었습니다."
            return text
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

    def _reduce_across_stocks(self, template , per_stock_results: List[Dict[str, Any]]) -> str:
        #self._ensure_models()
        if not per_stock_results:
            return "종목별 결과가 비어있습니다."
        
        def _fmt_sources(docs: List[Dict[str, Any]]) -> List[str]:
            out = []
            for d in (docs or [])[:3]:
                md = d.get("metadata", {}) or {}
                title = md.get("title") or md.get("headline") or md.get("doc_title") or md.get("doc_id") or ""
                url = md.get("url") or md.get("link") or md.get("source_url") or ""
                if title and url: out.append(f"{title} — {url}")
                elif title:       out.append(title)
                elif url:         out.append(url)
            return out
    
        def _hard_trunc(s: str, limit=1200):
            s = (s or "").strip()
            return s if len(s) <= limit else s[:limit] + "..."
    
        lines, source_lines = [], []
        for r in per_stock_results:
            stock = r.get("stock","")
            ans = r.get("answer") or ""
            ans = _hard_trunc(ans,1400) if ans else "관련된 정보를 찾을 수 없습니다."
            lines.append(f"### [{stock}] 부분답\n{ans}\n")
            for s in _fmt_sources(r.get("source_documents", [])):
                source_lines.append(f"[{stock}] {s}")
    
        # 순서 보존 dedup
        seen, dedup = set(), []
        for s in source_lines:
            if s not in seen:
                seen.add(s); dedup.append(s)
    
        # ✅ f-string 안에 백슬래시가 들어가던 join을 미리 계산
        parts_joined   = "\n\n".join(lines[:5])
        sources_joined = "\n".join(dedup[:12])
        print(parts_joined)
        print(sources_joined)
    
        prompt = f"""
    당신은 증권사 리서치센터장입니다.
    아래 각 종목의 부분 답변을 취합하여 **종합 리포트**를 작성하세요.
    종합리포트 작성 시 역할 설명은 필요 없으며, 답변 안에 자기소개는 포함하지 마세요.
    오늘이 {self.today}임을 고려하여 리포트를 생성해주세요.
    종합리포트 생성 시 아래 요구사항을 지켜주세요.
    
    [요구사항]
    0) 리포트 전체 내용에 대한 불릿을 붙인 개조식의 세 줄 내용 요약 제공
    1) 종목별 핵심 뉴스와 가격 영향 경로를 비교 정리(긍/부정, 단기/중기)
    2) 공통 테마(금리, 환율, 공급망, 규제 등) 식별 및 교차영향 설명
    3) 종목별 리스크/촉발요인, 모니터링 지표 제시
    4) 결론: 포트폴리오 관점 제언(오버웨이트/뉴트럴/언더웨이트 등 사용 가능)
    5) 수치는 `백틱`으로, 핵심 포인트는 **굵게**, 불릿 적절 활용
    6) 모호하면 '관련된 정보를 찾을 수 없습니다.'라고 분명히 표기
    7) 줄바꿈은 '<br>' 같은 HTML 태그 대신 실제 줄바꿈(엔터, 개행)으로 표시할 것
    8) 중요 포인트 앞에는 📌, 긍정 요인에는 📈, 리스크 요인에는 ⚠️ 같은 이모지를 붙이세요.
    9) 리포트 생성 시 표 형식은 사용하지 말 것
    
    아래는 당신이 생성할 리포트의 예시입니다.
    
    [답변 예시]

🔎 핵심만 콕콕

엔비디아의 2분기 실적이 예상치를 상회했지만, 주가는 하락했습니다.
핵심 동력인 데이터센터 부문의 지표가 기대를 밑돌았는데요.
엔비디아는 추가 자사주 매입 계획과 함께, 높은 3분기 실적 전망을 내놨습니다.

**엔비디아, 2분기 실적 발표**

💸 분기 사상 최대 매출: 
엔비디아는 올해 2분기(5월~7월) 기준, 매출 467억 4천만 달러, 조정 주당순이익(EPS)은 1.05달러를 기록했다고 밝혔습니다. 
전년 동기 대비 각각 56%, 59% 증가한 수치로, 분기 기준 사상 최대 실적인데요. 
엔비디아의 이번 실적은 AI 붐의 지속 여부를 가늠할 수 있는 핵심 지표로, 이번 주 뉴욕증시의 최대 분수령으로 꼽혔습니다.

주당순이익(EPS, Earnings Per Share): 
한 주당 기업이 벌어들인 순이익을 나타내는 지표입니다. 
회사가 벌어들인 이익을 주식 수로 나눈 값이죠. 
이 수치가 높을수록 회사가 벌어들인 이익이 많다는 뜻이고, 투자자들은 EPS를 참고해 기업의 수익성과 주식의 투자 가치를 가늠할 수 있습니다.

😮 시장 예상치 웃돌았다: 
월가의 엔비디아 실적 예상치는 매출 460억 6천만 달러와 주당순이익 1.01달러였습니다. 
이번에 발표한 실적이 예상치를 살짝 상회한 건데요. 
엔비디아는 AI 수요뿐만 아니라, 게임 부문이나 자율주행 등 전반적인 반도체 수요가 회복세에 접어들며 전 사업 부문이 고른 성장을 보였다고 발표했습니다.


📉 반응은 의외로 냉정?: 
하지만, 기대 이상의 실적에도 엔비디아의 주가는 시간외거래에서 3% 넘게 하락했습니다. 
지난 1년간 주가가 두 배 이상 상승한 상황에서, '어닝 서프라이즈' 이상의 모멘텀을 기대했던 투자자들이 많았기 때문인데요. 
일부 투자자들이 실적 발표 직후 차익 실현에 나선 것도 하락 요인으로 작용했습니다. 
그만큼 현재 엔비디아의 주가에 현재 기업가치에 대한 평가는 물론, 향후 성장성에 대한 기대가 크게 반영돼 있었다는 의미죠.


**매출 대부분은 데이터센터에서**

💾 AI 시대의 중심, 데이터센터: 2분기 전체 매출의 88%를 차지한 데이터센터 부문의 매출은 411억 달러를 기록했습니다. 
전년 동기 대비 56%, 1분기 대비 5% 증가한 수치인데요. 
데이터센터 부문은 생성형 AI, 초거대 언어모델, 클라우드 인프라 수요에 대응하는 고성능 칩 공급을 담당합니다.

🏢 여전히 폭발적인 성장세: 
엔비디아 데이터센터 부문의 주요 고객은 메타, 아마존, 마이크로소프트 등 '하이퍼스케일러'(초대형 클라우드 기업)입니다. 
이 기업들은 자사 AI 서비스 경쟁력을 높이기 위해 지속적으로 그래픽처리장치(GPU) 인프라에 투자하고 있는데요. 
대형 클라우드 기업뿐 아니라, 정부기관 및 일반 기업의 AI 시스템 구축 수요도 눈에 띄게 증가하고 있습니다. 
이에 따라 엔비디아는 다양한 규모의 고객군을 대상으로 맞춤형 AI 인프라 제품을 확장에 나섰죠.

그래픽처리장치(GPU): 
컴퓨터가 화면에 복잡한 그래픽이나 영상을 빠르게 처리하도록 돕는 전용 칩입니다. 
원래는 3D 게임이나 그래픽 작업에 주로 쓰였지만, 현재는 인공지능·데이터 분석 등 대규모 연산에도 활용됩니다.

😔 일부 전망치에 못 미쳐: 
데이터센터 부문의 성장세 자체는 이어졌지만, 이날 발표한 2분기 매출(약 411억 달러)은 시장 예상치(약 413억 달러)보다는 소폭 낮은 수치였습니다. 
2억 달러 차이라 해도 시장의 기대 수준이 높았던 만큼 시장을 실망하게 한 것이죠. 
특히 1분기에는 해당 부문에서 예상을 크게 웃도는 실적을 냈던 터라, 폭발적 성장이 없는 것에 대한 아쉬움이 더해졌습니다.

**엔비디아, 앞으로는 어떨까**

🤷 3분기 실적 전망은?: 
엔비디아는 다음 분기 매출 전망치를 540억 달러(±2%)로 제시했습니다. 
이는 월가 평균 예상치(약 531억 달러)를 웃도는 수준인데요. 
회사 측은 해당 예상치에 중국을 대상으로 한 H20 칩 매출은 포함돼 있지 않다고 밝혔습니다. 
엔비디아는 실적 발표 후 컨퍼런스 콜에서 향후 중국향 수출이 재개될 경우 20억~50억 달러 규모의 추가 매출이 반영될 수 있다는 분석을 내놨죠.

🗣️ 젠슨 황의 한마디: 
엔비디아 젠슨 황 CEO는 실적 발표 자리에서 "AI의 대전환은 아직 초기 단계"라고 말하며, 앞으로의 기술 주도권 경쟁에서 차세대 GPU 칩인 '블랙웰'이 핵심 역할을 하게 될 것이라 말했습니다. 
단순히 칩을 제조하는 수준을 넘어서, 전체 AI 컴퓨팅 인프라를 설계·공급하는 플랫폼 기업으로 전환 중이라는 점도 강조했죠. 
이를 통해 고객사가 더 쉽게 엔비디아의 인프라를 통합하고, 다양한 모델을 효율적으로 구동할 수 있도록 돕겠다는 겁니다.

🇨🇳 중국 시장은 여전히 변수: 
젠슨 황 CEO는 중국 시장의 성장성을 강조하며 수출 규제 완화를 촉구했지만, 미·중 갈등과 정책 리스크는 여전히 큰 변수입니다. 
최근 중국 정부에서는 H20 칩 구매 자제를 권고했고, 이에 따라 엔비디아가 H20 생산을 중단하기도 했는데요. 
기술력과 수요는 충분하지만, 지정학적 리스크로 실적 전망엔 불확실성이 남아 있습니다.
    
    
    [보유 종목별 요약]
    {parts_joined}
    """
        print(f'gen_ai {prompt}')
        try:
            resp = self.gen_model.generate_content(
                prompt, 
                generation_config=self._gen_config(temperature=0.25),
                safety_settings=self._safe_settings())
            text = self._extract_text(resp)
            text = text.replace("<br>", "\n").strip()
            text = text.replace("format:", "").strip()
            return text if text else "최종 통합 생성에 실패했습니다. 안전필터 또는 토큰 한도에 의해 응답이 비었습니다."
        except Exception as e:
            return f"최종 통합 생성 오류: {e}"

    def answer_5_stocks_and_reduce(self, stocks: List[str], template: Optional[str] = None, max_workers: int = 5) -> Dict[str, Any]:
        template = template or "{stock} 관련해서 종목의 가격에 중요한 뉴스는?"
        per_stock = self.answer_multi_stocks(stocks, template=template, max_workers=max_workers)
        final = self._reduce_across_stocks(template, per_stock)

        #html 테스트
        mailing.send_mail("fanxy0730@gmail.com", final)

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

"""

if __name__ == "__main__":
    # ✅ 환경변수 필요: GOOGLE_CLOUD_PROJECT, QDRANT_URL, QDRANT_API_KEY
    # 또는 Streamlit secrets (gcp_service_account)로 초기화 가능

    try:
        service = NewsReportService()
    except Exception as e:
        print(f"[Init Error] 서비스 초기화 실패: {e}")
        exit(1)

    # 테스트용 종목 (원하는 티커/심볼로 교체 가능)
    test_stocks = ["삼성전자", "현대차", "카카오", "네이버", "LG에너지솔루션"]

    # 기본 질의 템플릿
    template = "{stock} 관련해서 종목의 가격에 중요한 뉴스는?"

    print(">>> 5개 종목 병렬 질의 & 최종 리포트 생성 테스트")
    result = service.answer_5_stocks_and_reduce(test_stocks, template=template, max_workers=5)

    # 종목별 결과 출력
    for r in result["results"]:
        print("=" * 80)
        print(f"[{r['stock']}] 질문: {r['question']}")
        print(f"부분답:\n{r['answer'][:500]}...\n")  # 앞부분만 표시

    print("=" * 80)
    print(">>> 최종 통합 리포트:")
    print(result["final_report"])

"""





















