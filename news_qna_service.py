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
from dotenv import load_dotenv

load_dotenv()

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
    
    # def _extract_text_from_payload(self, payload: dict) -> str:
    #     """
    #     payload["doc"]가 문자열이거나, dict(예: {"content": "...", "text": "...", ...})일 수 있으니 모두 커버
    #     """
    #     if not isinstance(payload, dict):
    #         return ""
    #     doc = payload.get("doc")
    #     if isinstance(doc, str):
    #         return doc
    #     if isinstance(doc, dict):
    #         # 흔한 텍스트 키들 우선순위
    #         return doc.get("content") or doc.get("text") or doc.get("page_content") or ""
    #     return ""

    def _extract_text_from_payload(self, payload: dict) -> str:
        """
        (text, title, link) 추출:
        1) payload["doc"] (str/dict)
        2) payload["metadata"] (dict)
        3) payload 상위 키
        우선순위로 안전하게 가져온다.
        """
        if not isinstance(payload, dict):
            return "","",""
        text = ""
        title = ""
        link = ""
        
        doc = payload.get("doc")

        if isinstance(doc, str):
            return doc
        elif isinstance(doc, dict):
            text  = doc.get("text") or doc.get("content") or doc.get("page_content") or ""
            title = doc.get("title") or title
            link  = doc.get("link")  or doc.get("url") or link

        # 2) metadata
        md = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        if not text:
            text = md.get("text") or md.get("content") or md.get("page_content") or ""
        if not title:
            title = md.get("title") or md.get("headline") or md.get("subject") or ""
        if not link:
            link = md.get("link") or md.get("url") or ""

        # 3) payload 상위 보강
        if not title:
            title = payload.get("title") or title
        if not link:
            link = payload.get("link") or payload.get("url") or link
        if not text:
            text = payload.get("text") or payload.get("content") or payload.get("page_content") or text

        return text, title, link
    # title, link 추가 전
    # def retrieve(self, question: str) -> List[Dict[str, Any]]:
    #     qv = self._embed_query(question)
    #     hits = self.qc.search(
    #         collection_name=self.collection,
    #         query_vector=qv,
    #         limit=self.top_k if not self.use_rerank else self.rerank_top_k,
    #         with_payload=True,
    #         with_vectors=False,
    #     )
    
    #     # (선택) distance 모드 파악
    #     dist_mode = getattr(self, "_dist_mode", None)
    #     if dist_mode is None:
    #         try:
    #             info = self.qc.get_collection(self.collection)
    #             params = getattr(info.config, "params", None) or getattr(info, "config", None)
    #             vectors = getattr(params, "vectors", None)
    #             dist_mode = str(getattr(vectors, "distance", "")).lower() if vectors else ""
    #         except Exception:
    #             dist_mode = ""
    #         self._dist_mode = dist_mode
    
    #     docs: List[Dict[str, Any]] = []
    #     for h in hits:
    #         payload = h.payload or {}
    #         text = self._extract_text_from_payload(payload)
    
    #         # 메타데이터: payload["metadata"] 최우선, 없으면 payload에서 doc 제외
    #         md = {}
    #         if isinstance(payload.get("metadata"), dict):
    #             md = dict(payload["metadata"])
    #         else:
    #             md = {k: v for k, v in payload.items() if k not in ("doc", "metadata")}
    
    #         raw = getattr(h, "score", None)  # Qdrant는 보통 distance를 score로 반환
    #         distance = float(raw) if raw is not None else None
    #         similarity = None
    #         if distance is not None and "cosine" in dist_mode:
    #             similarity = distance
    
    #         docs.append({
    #             "id": str(getattr(h, "id", "")),
    #             "content": text,            # ✅ 이제 doc 기반 본문
    #             #"metadata": md,             # ✅ metadata 그대로
    #             "title": title,
    #             "link": link,
    #             "score": similarity if similarity is not None else (float(raw) if raw is not None else None),
    #             "distance": distance,
    #             "distance_mode": dist_mode,
    #         })
    #     return docs
    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        qv = self._embed_query(question)
        hits = self.qc.search(
            collection_name=self.collection,
            query_vector=qv,
            limit=self.top_k if not self.use_rerank else self.rerank_top_k,
            with_payload=True,
            with_vectors=False,
        )

        # 거리/스코어 모드 파악
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

            # ✅ 여기서 통합 추출
            text, title, link = self._extract_text_from_payload(payload)

            # metadata 구성
            if isinstance(payload.get("metadata"), dict):
                md = dict(payload["metadata"])
            else:
                md = {k: v for k, v in payload.items() if k not in ("doc", "metadata")}

            raw_score = getattr(h, "score", None)
            score = float(raw_score) if raw_score is not None else None

            # ✅ Qdrant의 score는 "클수록 더 유사"가 되도록 정의됨.
            #    cosine/dot은 score를 그대로 similarity로 쓰는 게 일반적.
            #    euclid일 땐 관례상 -distance가 score인 경우가 많아 별도 가공 없이 표기만 하거나 None 처리.
            similarity = None
            if score is not None:
                if "cosine" in dist_mode or "dot" in dist_mode:
                    similarity = score
                else:
                    similarity = None  # 필요하면 -score 등으로 환산 정책 결정

            docs.append({
                "id": str(getattr(h, "id", "")),
                "content": text,        # ✅ 이제 빈 값이 아님 (metadata까지 커버)
                "title": title,
                "link": link,
                "metadata": md,
                "score": similarity if similarity is not None else score,
                "distance": score,      # 혼동 방지를 원하면 이 필드명은 빼거나 'raw_score'로 변경 권장
                "distance_mode": dist_mode,
            })

        return docs

    def rerank(self, question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 자리만들기: 필요시 Vertex Ranking이나 cross-encoder 붙이기
        # 지금은 그대로 top_k 상위만 리턴
        return (docs or [])[: self.top_k]

 # 5. 모호하거나 근거 없는 내용은 쓰지 말고 "관련된 정보를 찾을 수 없습니다."라고 답하세요.
    def generate(self, question: str, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return "관련된 정보를 찾을 수 없습니다."
        # ctx = "\n\n".join(d["content"] for d in docs)
        ctx = "\n\n".join(f"""제목: {d["title"]}
본문: {d["content"]}
url: {d["link"]}""" for d in docs)
        
        prompt = f"""
      당신은 주식시장과 연금에 정통한 전문 애널리스트입니다. 
      당신에게 주식 종목과 관련된 뉴스기사가 제공됩니다. 
      아래 뉴스기사를 근거로 사용자의 질문 의도에 맞는 한국어 답변을 충실하게 작성하세요.
      답변을 작성 시 아래 지침을 반드시 지켜주세요.
      
        [작성 지침]  
        1. 답변은 **3단락 이상**으로 구성하세요.  
        2. **중요 포인트는 굵게**, 핵심 수치는 `코드블록 스타일`로 표시하세요.  
        3. 답변 중간에는 ▸, ✔, ✦ 같은 불릿 아이콘을 활용해 시각적으로 보기 좋게 정리하세요.  
        4. 마지막에 `---` 구분선을 넣고, 제목과 url을 첨부해주세요
        5. 답변에 적절한 이모지를 사용해주세요.
        
        [뉴스기사]
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

        #ctx = "\n\n".join(d["content"] for d in docs)
        ctx = "\n\n".join(f"""제목: {d["title"]}
본문: {d["content"]}
url: {d["link"]}""" for d in docs)
        prompt = f"""
      당신은 주식시장과 연금에 정통한 전문 애널리스트입니다.
      당신에게 주식 종목과 관련된 뉴스기사가 제공됩니다. 
      아래 뉴스기사를 근거로 사용자의 질문 의도에 맞는 한국어 답변을 충실하게 작성하세요.
      
        [작성 지침]  
        1. 답변은 **3단락 이상**으로 구성하세요.  
        2. **중요 포인트는 굵게**, 핵심 수치는 `코드블록 스타일`로 표시하세요.  
        3. 답변 중간에는 ▸, ✔, ✦ 같은 불릿 아이콘을 활용해 시각적으로 보기 좋게 정리하세요.  
        4. 마지막에 `---` 구분선을 넣고, 제목과 url을 첨부해주세요.
        5. 만약 제목과 url이 비어 있다면 별도로 답변에 포함시켜서는 안됩니다.
        5. 답변에 적절한 이모지를 사용하여 시각적으로 보기 좋게 정리하세요.
        6. 답변을 할 때 내용에 맞는 적절한 소제목을 붙여 주세요.
        
        아래는 당신이 수행해야할 업무의 생성 예시입니다.

### 실적 부진에도 긍정적인 주가 전망 📈

최근 삼성전자는 시장 전망치를 밑도는 `4조 6000억원`의 2분기 잠정 실적을 발표하며 부진한 모습을 보였습니다. 
하지만 놀랍게도 주가는 큰 타격을 받지 않았으며, 오히려 증권가에서는 긍정적인 전망을 쏟아내고 있습니다. 
다수의 애널리스트들은 **삼성전자가 '실적 바닥'을 통과하고 있으며, 이제 본격적인 회복 국면에 진입했다**고 평가합니다. 
이러한 기대감을 바탕으로 KB증권, 신한투자증권 등은 목표 주가를 `9만원`으로 상향 조정했으며, 키움증권 역시 목표가를 `8만 9000원`으로 높여 잡는 등 '8만전자'를 넘어선 상승 가능성에 무게를 싣고 있습니다.

### 미래 성장을 이끌 핵심 동력: HBM과 파운드리 🤖

애널리스트들이 삼성전자의 미래를 밝게 보는 가장 큰 이유는 바로 **반도체 사업부의 경쟁력 회복에 대한 강한 믿음** 때문입니다. 
특히 인공지능(AI) 시대의 핵심 부품으로 떠오른 고대역폭 메모리(HBM)와 파운드리(반도체 위탁생산) 부문이 성장을 견인할 것으로 보입니다.

▸ **HBM 경쟁력 회복**: 현재 유통되는 HBM3E의 다음 세대인 **HBM4의 샘플 공급을 앞두고 있으며, 품질 또한 기대 이상의 모습**을 보이는 것으로 파악됩니다. 이는 AI 반도체 시장에서 다시 리더십을 회복할 수 있다는 중요한 신호입니다.
✔ **파운드리 사업 반등**: 최근 **테슬라와의 파운드리 계약을 확보한 것은 삼성전자의 서브 5나노 공정 기술력을 입증**하는 중요한 성과입니다. 향후 2나노 공정 개선을 통해 추가적인 대형 고객사를 확보할 가능성이 커지면서 파운드리 부문의 적자 축소와 실적 개선이 기대됩니다.
✦ **견조한 외국인 매수세와 주주환원**: 이재용 회장의 사법 리스크 해소와 함께 외국인 투자자들의 매수세가 꾸준히 유입되고 있습니다. 여기에 **자사주 추가 매입 및 소각 등 주주환원 정책에 대한 기대감** 역시 주가에 긍정적인 요인으로 작용하고 있습니다.

### 과거부터 이어진 '저평가' 매력과 리스크 요인 🧐

삼성전자의 '저평가' 매력은 어제오늘의 이야기가 아닙니다. 
기사에 따르면, 주가가 사상 최고치를 경신하던 2016년에도 자산운용사들은 **"연간 영업이익이 `40조원`에 달하지만 시가총액은 `260조원`에 불과하다"**며 해외 경쟁사 대비 항상 저평가되어 왔다고 분석했습니다. 
당시 주가수익비율(PER)이 10배 미만이었던 점을 고려하면, 현재의 실적 회복 기대감은 주가 상승 여력이 충분하다는 근거가 됩니다.

물론 리스크 요인도 존재합니다. 
스마트폰의 두뇌 역할을 하는 AP(애플리케이션 프로세서)를 퀄컴에 전적으로 의존하게 되면서 **가격 협상력이 떨어져 수익성 확보에 어려움을 겪을 수 있다**는 점은 우려스러운 부분입니다. 
또한, 애널리스트들은 삼성전자의 주가가 실적과 거의 완벽하게 동행하기 때문에, 만약 기대했던 반도체 부문의 실적 개선이 지연될 경우 언제든 주가가 하락할 수 있다는 점을 항상 유념해야 한다고 조언합니다. ⚠️

---
**[근거 기사]**
- 삼성전자 ‘기회의 순간 5’ [스페셜리포트] (url: https://n.news.naver.com/mnews/article/024/0000098907?sid=101)
- Chip giants accelerate efforts to develop HBM alternatives (url: https://n.news.naver.com/mnews/article/009/0005506084?sid=104)
- [PRNewswire] 풀무원, 신속 반응형 공급망 구축을 위해 키넥시스와 파트... (url: https://n.news.naver.com/mnews/article/001/0010073574?sid=104)

        
[뉴스기사]
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

"""
if __name__  == "__main__":
    newsqa = NewsQnAService()
    doc_res = newsqa.retrieve_only("삼성전자 주가 전망은?")
    
    print("문서검색 결과")
    print(doc_res)
    print("---")
    
    print("정답 결과")
    result_stream = newsqa.answer_stream("삼성전자 주가 전망은?")
    
    # 제너레이터 객체를 반복하여 텍스트 청크를 순서대로 출력
    for chunk in result_stream:
        print(chunk, end="") # end=""를 사용해 줄바꿈 없이 이어붙임
    print() # 마지막에 줄바꿈 추가
"""
