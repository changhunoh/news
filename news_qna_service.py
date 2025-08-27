import os, threading
from typing import List, Dict, Any, Optional, TypedDict
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from qdrant_client import QdrantClient
from google.oauth2 import service_account
import streamlit as st  # Streamlit secrets 사용 시
from dotenv import load_dotenv


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

load_dotenv()

# ==== Config ====
GCP_PROJECT      = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION     = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
CHROMA_DB_DIR    = os.getenv("CHROMA_DB_DIR", "./chroma_store")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "stock_news")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")
EMBED_DIM        = int(os.getenv("EMBED_DIM", "3072"))
GENAI_MODEL_NAME = os.getenv("GENAI_MODEL_NAME", "gemini-2.5-flash-lite")
DEFAULT_TOP_K    = int(os.getenv("DEFAULT_TOP_K", "10"))
RERANK_TOP_K     = int(os.getenv("RERANK_TOP_K", "5"))
QDRANT_URL       = os.getenv("QDRANT_URL")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY")
USE_QDRANT       = bool(QDRANT_URL and QDRANT_API_KEY)

# ---------------------------
# Thread-local models
# ---------------------------
_thread_local = threading.local()

def _get_embed_model_thread_local(model_name: str = EMBED_MODEL_NAME):
    if not hasattr(_thread_local, "embed_model") or getattr(_thread_local, "embed_model_name", None) != model_name:
        _thread_local.embed_model = TextEmbeddingModel.from_pretrained(model_name)
        _thread_local.embed_model_name = model_name
    return _thread_local.embed_model

def _get_genai_model_thread_local(model_name: str = GENAI_MODEL_NAME):
    if not hasattr(_thread_local, "genai_model") or getattr(_thread_local, "genai_model_name", None) != model_name:
        _thread_local.genai_model = GenerativeModel(model_name)
        _thread_local.genai_model_name = model_name
    return _thread_local.genai_model

# ---------------------------
# VectorDB 연결
# ---------------------------
qc = None
collection = None

if USE_QDRANT:
    qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(f"[INFO] Qdrant connected: {QDRANT_URL}, collection='{COLLECTION_NAME}'")
else:
    from chromadb import PersistentClient
    try:
        chroma_client = PersistentClient(path=CHROMA_DB_DIR)
        collection = chroma_client.get_collection(COLLECTION_NAME)
        print(f"[INFO] ChromaDB collection '{COLLECTION_NAME}' loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load ChromaDB collection: {e}")
        collection = None

# ---------------------------
# RAG State 정의
# ---------------------------
class RAGState(TypedDict):
    question: str
    documents: List[Dict[str, Any]]
    answer: Optional[str]

# ---------------------------
# LangGraph Nodes
# ---------------------------
def retrieve(state: RAGState) -> RAGState:
    question = state["question"]
    print(f"[STEP] Retrieving documents for query: '{question}'")

    # 1) 질의 임베딩
    embed_model = _get_embed_model_thread_local()
    inputs = [TextEmbeddingInput(text=question, task_type="RETRIEVAL_QUERY")]
    qvec = embed_model.get_embeddings(inputs, output_dimensionality=EMBED_DIM)[0].values

    docs: List[Dict[str, Any]] = []

    if USE_QDRANT and qc is not None:
        hits = qc.search(
            collection_name=COLLECTION_NAME,
            query_vector=qvec,
            limit=DEFAULT_TOP_K,
            with_payload=True,
            with_vectors=False,
        )
        for h in hits:
            payload = h.payload or {}
            docs.append({
                "id": str(h.id),
                "content": payload.get("text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "text"},
                "score": float(getattr(h, "score", 1.0)),
            })
    else:
        if collection is None:
            print("[WARN] collection is None")
            state["documents"] = []
            return state
        res = collection.query(
            query_embeddings=[qvec],
            n_results=DEFAULT_TOP_K,
            include=["metadatas", "documents", "distances"],
        )
        if res.get("ids"):
            for i in range(len(res["ids"][0])):
                docs.append({
                    "id": res["ids"][0][i],
                    "content": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "score": 1.0 - res["distances"][0][i],
                })

    state["documents"] = docs
    return state

def rerank_with_vertex(state: RAGState) -> RAGState:
    documents = state["documents"]
    question = state["question"]

    if not documents:
        return state
    print("[STEP] Reranking documents with Vertex AI Ranking API (stub).")

    try:
        # TODO: Vertex AI Ranking API 정식 호출 (현재 placeholder)
        # 일단은 단순 score 내림차순 정렬
        ranked_docs = sorted(documents, key=lambda d: d.get("score", 0.0), reverse=True)
        state["documents"] = ranked_docs[:RERANK_TOP_K]
    except Exception as e:
        print(f"[ERROR] Rerank failed: {e}")
        state["documents"] = documents[:DEFAULT_TOP_K]

    return state

def generate(state: RAGState) -> RAGState:
    question = state["question"]
    documents = state["documents"]

    if not documents:
        state["answer"] = "관련된 정보를 찾을 수 없습니다."
        return state

    print("[STEP] Generating answer with Gemini.")

    context = "\n\n".join([doc["content"] for doc in documents])
    prompt = f"""
당신은 주식시장과 연금에 정통한 우리은행 소속 애널리스트입니다.
아래 뉴스 기사 내용을 참고하여 질문에 답변하세요.
답변이 불가능하면 "관련된 정보를 찾을 수 없습니다."라고 말하세요.
중요 포인트는 **강조**해주세요.
답변 마지막에 참고 기사 제목들을 나열하세요.

[뉴스 기사]
{context}
---
[질문]
{question}
"""

    genai_model = _get_genai_model_thread_local()

    try:
        response = genai_model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 1500},
        )
        answer = (response.text or "").strip()
        state["answer"] = answer
    except Exception as e:
        print(f"[ERROR] Failed to generate answer: {e}")
        state["answer"] = "답변 생성 중 오류가 발생했습니다."

    return state

# ---------------------------
# LangGraph Workflow
# ---------------------------
def build_rag_graph():
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank_with_vertex)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

rag_graph = build_rag_graph()

# ---------------------------
# Public APIs
# ---------------------------
def get_rag_response(query: str) -> Dict[str, Any]:
    if (not USE_QDRANT or qc is None) and (collection is None):
        return {
            "answer": "백엔드 초기화 실패: Qdrant/Chroma 연결이 없습니다.",
            "source_documents": [],
        }

    initial_state = RAGState(question=query, documents=[], answer=None)
    final_state = rag_graph.invoke(initial_state)

    return {
        "answer": final_state.get("answer", "답변을 찾을 수 없습니다."),
        "source_documents": final_state.get("documents", []),
    }

def retrieve_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    embed_model = _get_embed_model_thread_local()
    inputs = [TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")]
    qvec = embed_model.get_embeddings(inputs, output_dimensionality=EMBED_DIM)[0].values

    docs: List[Dict[str, Any]] = []

    if USE_QDRANT and qc is not None:
        hits = qc.search(
            collection_name=COLLECTION_NAME,
            query_vector=qvec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        for h in hits:
            payload = h.payload or {}
            docs.append({
                "id": str(h.id),
                "content": payload.get("text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "text"},
                "score": float(getattr(h, "score", 1.0)),
            })
    else:
        if collection is None:
            return []
        res = collection.query(
            query_embeddings=[qvec],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        if res.get("ids"):
            for i in range(len(res["ids"][0])):
                docs.append({
                    "id": res["ids"][0][i],
                    "content": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "score": 1.0 - res["distances"][0][i],
                })

    return docs

# ---------------------------
# Local test
# ---------------------------
if __name__ == "__main__":
    print("[INFO] news_qna_service.py test run")
    q = "호텔신라 주식 전망은?"
    docs = retrieve_documents(q, top_k=3)
    print(f"Retrieved {len(docs)} docs.")
    resp = get_rag_response(q)
    print("Answer:", resp["answer"][:300])
    print("Sources:", len(resp["source_documents"]))
