import os
import re
import numpy as np
import threading
from typing import List, Dict, Any, Optional, TypedDict
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb import PersistentClient
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
print("라이브러리 임포트 성공")

# 환경변수 로드
load_dotenv()

#==== Config ====

GCP_PROJECT     = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION    = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
CHROMA_DB_DIR   = os.getenv("CHROMA_DB_DIR", "./chroma_store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "stock_news")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")
EMBED_DIM       = int(os.getenv("EMBED_DIM", "768"))
GENAI_MODEL_NAME= os.getenv("GENAI_MODEL_NAME", "gemini-2.5-flash-lite")
DEFAULT_TOP_K   = int(os.getenv("DEFAULT_TOP_K", 20))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K",20))

#==== Config ====
_thread_local = threading.local()

# 임베딩 모델 로드
def _get_embed_model_thread_local(model_name:str = EMBED_MODEL_NAME):
    if not hasattr(_thread_local, "embed_model") or getattr(_thread_local, "model_name", None) != model_name:
        from vertexai.language_models import TextEmbeddingModel
        _thread_local.embed_model = TextEmbeddingModel.from_pretrained(model_name)
        _thread_local.model_name = model_name
    return _thread_local.embed_model

# LLM 모델 로드
def _get_genai_model_thread_local(model_name: str = GENAI_MODEL_NAME):
    if not hasattr(_thread_local, "genai_model") or getattr(_thread_local, "genai_model_name", None) != model_name:
        from vertexai.generative_models import GenerativeModel
        _thread_local.genai_model = GenerativeModel(model_name)
        _thread_local.genai_model_name = model_name
    return _thread_local.genai_model

# ===== Connect Chroma =====

try:
    chroma_client = PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"[INFO] ChromaDB collection '{COLLECTION_NAME}' loaded.")
except Exception as e:
    print(f"[ERROR] Failed to load ChromaDB collection: {e}")
    collection = None
    
#RAG state for Langgraph
class RAGState(TypedDict):
    question: str
    documents: List[Dict[str,Any]]
    answer: Optional[str]

# ==== 1) LangGraph Nodes ====
def retrieve(state: RAGState) -> RAGState:
    """
    질문을 임베딩하고 ChromaDB에서 관련 문서를 검색합니다.
    """
    if collection is None:
        return state
    
    question = state["question"]
    print(f"[STEP] Retrieving documents for query: '{question}'")

    #쿼리 임베딩
    embed_model = _get_embed_model_thread_local()
    inputs = [TextEmbeddingInput(text=question, task_type="RETRIEVAL_QUERY")]
    embed_vec = embed_model.get_embeddings(inputs, output_dimensionality=EMBED_DIM)[0].values

    #ChromaDB 검색
    search_results = collection.query(
        query_embeddings=[embed_vec],
        n_results=DEFAULT_TOP_K,
        include=['embeddings', 'metadatas', 'documents', 'distances']
    )
    #결과 정리
    documents = []
    if search_results.get("ids"):
        for i in range(len(search_results["ids"][0])):
            documents.append({
                "id":search_results["ids"][0][i],
                "content":search_results["documents"][0][i],
                "metadata":search_results["metadatas"][0][i],
                "score":1.0 - search_results["distances"][0][i] # cosine distance -> similarity score
            })
    #RAGState에 저장
    state["documents"] = documents
    return state

def rerank_with_api(state: RAGState) -> RAGState:
    documents = state["documents"]
    question = state["question"]

    if not documents:
        return state
    print("[STEP] Reranking documents with Vertex AI Ranking API.")

    #Rerank API 호출을 위한 입력 데이터 구성
    rank_records = []
    for i, doc in enumerate(documents):
        rank_records.append({
            "id": str(i), # 고유 ID
            "content": doc['content']
        })
    try:
        # Vertex AI Ranking API 호출
        ranking_client = aiplatform.MatchingEngineIndexServiceAsyncClient(
            client_options={"api_endpoint": f"{GCP_LOCATION}-aiplatform.googleapis.com"}
        )

        # API 호출
        response = ranking_client.rank(
            project=GCP_PROJECT,
            location=GCP_LOCATION,
            query=question,
            records=rank_records,
        )
        # API 응답을 기반으로 문서 재정렬
        ranked_docs = sorted(documents, 
                         key=lambda doc: next(r.score for r in response.records if r.id == str(documents.index(doc))), 
                         reverse=True)

        # 상위 k개만 선택
        state["documents"] = ranked_docs[:RERANK_TOP_K]
    
    except Exception as e:
        print(f"[ERROR] Vertex AI Ranking API failed: {e}")
        # 오류 발생 시 기존 문서 유지
        state["documents"] = documents[:DEFAULT_TOP_K]
    
    return state

def generate(state: RAGState) -> RAGState:
    """
    검색된 문서를 바탕으로 질문에 답변을 생성합니다.
    """
    question = state['question']
    documents=state['documents']

    if not documents:
        state["answer"] = "관련된 정보를 찾을 수 없습니다."
        print("[STEP] No documents found, returning fallback answer.")
        return state
    
    print("[STEP] Generating answer based on retrieved documents.")

    #프롬프트 구성
    context = '\n\n'.join([doc['content'] for doc in documents])
    prompt = f"""
    당신은 주식시장에 대해 날카로운 통찰력을 가진 애널리스트입니다.
    아래 뉴스 기사 내용을 참고하여 사용자의 질문에 답변해주세요.
    만약 주어진 정보만으로 답변이 불가능하다면, "관련된 정보를 찾을 수 없습니다."라고 답변하세요.
    답변 중 중요한 부분은 마크다운을 형식을 사용하여 **강조**해주세요.

    ---
    뉴스 기사 내용:
    {context}
    ---
    사용자 질문:
    {question}
    """

    #Gemini 모델 활용 답변 생성
    genai_model = _get_genai_model_thread_local()
    
    try:
        response = genai_model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 1024}
        )
        answer = response.text
        # 답변에 불필요한 공백이나 형식 제거
        answer = re.sub(r'^\s+|\s+$', '', answer)
        state["answer"] = answer
    
    except Exception as e:
        print(f"[ERROR] Failed to generate answer: {e}")
        state["answer"] = "답변 생성 중 오류가 발생했습니다."

    return state

# ==== 2) LangGraph 구성 ====

def build_rag_graph():
    """RAG 워크플로를 정의하고 컴파일합니다."""
    workflow = StateGraph(RAGState)
    
    # 노드 추가
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank",rerank_with_api)
    workflow.add_node("generate", generate)

    # 흐름 정의
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# 컴파일된 그래프를 전역 변수로 저장

rag_graph = build_rag_graph()

def get_rag_response(query: str) -> Dict[str, Any]:
    """
    RAG 워크플로를 수행하여 답변과 출처 문서를 반환합니다.
    """
    if collection is None:
        return {
            "answer": "서비스 준비 중: ChromaDB 컬렉션이 로드되지 않았습니다.",
            "source_documents": []
        }

    initial_state = RAGState(question=query, documents=[], answer=None)
    final_state = rag_graph.invoke(initial_state)

    return {
        "answer": final_state.get("answer", "답변을 찾을 수 없습니다."),
        "source_documents": final_state.get("documents", [])
    }

# 이 함수는 테스트용으로 추가
def retrieve_documents(query: str, top_k: int) -> List[Dict[str, Any]]:
    if collection is None:
        return []

    embed_model = _get_embed_model_thread_local()
    inputs = [TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")]
    embed_vec = embed_model.get_embeddings(inputs, output_dimensionality=EMBED_DIM)[0].values

    search_results = collection.query(
        query_embeddings=[embed_vec],
        n_results=top_k,
        include=['embeddings', 'metadatas', 'documents', 'distances']
    )

    documents = []
    if search_results.get("ids"):
        for i in range(len(search_results["ids"][0])):
            documents.append({
                "id": search_results["ids"][0][i],
                "content": search_results["documents"][0][i],
                "metadata": search_results["metadatas"][0][i],
                "score": 1.0 - search_results["distances"][0][i]
            })
    return documents

if __name__ == "__main__":
    print("[INFO] news_qna.py 테스트를 시작합니다.")

    #예시 질문
    test_query = "삼성전자의 주가 전망에 대해 알려주세요."
    print("\n=== 문서 검색 기능 테스트 ===")
    try:
        # retrieve_documents 함수는 내부에서 스레드 로컬 모델을 사용하므로, 
        # 직접 호출해도 문제 없습니다.
        retrieved_docs = retrieve_documents(test_query, top_k=5)
        
        if not retrieved_docs:
            print("  [WARN] 검색된 문서가 없습니다. ChromaDB에 문서가 있는지 확인하세요.")
        else:
            print(f"  [SUCCESS] {len(retrieved_docs)}개의 문서가 검색되었습니다.")
            for i, doc in enumerate(retrieved_docs):
                print(f"  - 문서 #{i+1}, 유사도: {doc['score']:.4f}")
                print(f"    내용: {doc['content'][:100]}...")
                
    except Exception as e:
        print(f"  [FAIL] 문서 검색 중 오류 발생: {e}")
        
    # 2. 답변 생성 기능 테스트
    print("\n=== 답변 생성 기능 테스트 ===")
    try:
        # get_rag_response 함수를 호출하여 전체 워크플로 테스트
        response = get_rag_response(test_query)
        
        answer = response.get("answer")
        source_docs = response.get("source_documents", [])
        
        if "오류" in answer or not answer:
            print(f"  [FAIL] 답변 생성 실패. 답변: '{answer}'")
        else:
            print(f"  [SUCCESS] 답변 생성 완료.")
            print(f"    답변: {answer}")
            print(f"    참고 문서 수: {len(source_docs)}")

    except Exception as e:
        print(f"  [FAIL] 답변 생성 중 오류 발생: {e}")
        
    print("\n[INFO] 테스트 완료.")





