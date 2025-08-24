import streamlit as st
from typing import List, Dict, Any, Optional

# rag_core.py에서 RAG 로직을 임포트
from news_qna import get_rag_response, collection

#==== 디자인 및 색상 설정 ====
# 이미지에 사용된 색상을 기반으로 변수 정의
PRIMARY_BLUE = "#007BFF"  # 강조색 (버튼, 하이라이트)
SECONDARY_BLUE = "#E6F0FF" # 파란색 카드 배경
LIGHT_GRAY = "#f0f2f6"     # 전체 배경색
CARD_BG = "#FFFFFF"        # 카드 배경색
TEXT_DARK = "#333333"      # 주요 텍스트
TEXT_GRAY = "#888888"      # 보조 텍스트

st.set_page_config(
    page_title="은행 뉴스 Q&A",
    page_icon="🏦",
    layout="wide"
)

# 사용자 정의 CSS를 사용하여 이미지와 유사한 스타일 구현
st.markdown(f"""
    <style>
    /* 전체 페이지 배경색 */
    .stApp {{
        background-color: {LIGHT_GRAY};
    }}

    /* 헤더 */
    .header-container {{
        background-color: {CARD_BG};
        padding: 1rem 1.5rem;
        border-bottom-left-radius: 15px;
        border-bottom-right-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
    }}
    .header-text {{
        font-size: 1.5rem;
        font-weight: bold;
        color: {TEXT_DARK};
    }}
    .header-icons {{
        font-size: 1.2rem;
        color: {TEXT_GRAY};
    }}

    /* 카드 스타일 (질문, 계좌 정보 등) */
    .card-container {{
        background-color: {CARD_BG};
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
    }}
    .blue-card-container {{
        background-color: {SECONDARY_BLUE};
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
    }}
    
    /* 텍스트 스타일 */
    .card-title {{
        font-size: 1rem;
        font-weight: bold;
        color: {TEXT_DARK};
    }}
    .card-content {{
        font-size: 1.2rem;
        font-weight: bold;
        color: {PRIMARY_BLUE};
    }}
    .card-meta {{
        font-size: 0.8rem;
        color: {TEXT_GRAY};
    }}
    .highlighted-text {{
        font-weight: bold;
        color: {PRIMARY_BLUE};
    }}
    
    /* 채팅 메시지 스타일 */
    .stChatMessage.st-emotion-cache-1c7c9rp {{ /* 사용자 메시지 배경 */
        background-color: #E6F0FF;
        border-top-left-radius: 15px;
        border-bottom-left-radius: 15px;
        border-top-right-radius: 15px;
        border-bottom-right-radius: 5px;
        border-left: 5px solid {PRIMARY_BLUE};
    }}
    .stChatMessage.st-emotion-cache-7ym5gk {{ /* 어시스턴트 메시지 배경 */
        background-color: #f0f2f6;
        border-top-left-radius: 15px;
        border-bottom-left-radius: 5px;
        border-top-right-radius: 15px;
        border-bottom-right-radius: 15px;
        border-left: 5px solid {TEXT_GRAY};
    }}

    </style>
""", unsafe_allow_html=True)

#==== 헤더 구현 ====
with st.container():
    st.markdown("""
        <div class="header-container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.5rem;">👤</span>
                    <span class="header-text" style="margin-left: 0.5rem;">김우리님</span>
                </div>
                <div style="display: flex; gap: 1rem;">
                    <span class="header-icons">💬</span>
                    <span class="header-icons">🔔</span>
                    <span class="header-icons">☰</span>
                </div>
            </div>
            <p style="color: {TEXT_DARK}; font-size: 1.2rem; margin-top: 1rem;">
                우리 <span class="highlighted-text">WON인증서 3일후</span><br>유효기간 만료됩니다.
                <span style="font-size: 1.2rem; display: block; margin-top: 0.5rem;">인증서를 갱신해주세요 →</span>
            </p>
        </div>
    """.format(TEXT_DARK=TEXT_DARK, highlighted_text=PRIMARY_BLUE), unsafe_allow_html=True)

st.header("뉴스 기사 Q&A", divider='blue')

# ==== Q&A 서비스 로직 ====

# 채팅 UI 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("참고 문서 보기"):
                for idx, source in enumerate(message["sources"]):
                    st.markdown(f"**문서 #{idx+1}** (유사도: {source['score']:.2f})")
                    if 'title' in source['metadata']:
                        st.markdown(f"**제목:** {source['metadata']['title']}")
                    st.markdown(f"**내용:** {source['content'][:200]}...")
                    st.markdown("---")

# 사용자 입력 처리
if prompt := st.chat_input("궁금한 점을 질문해 주세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if collection is None:
            st.error("서비스를 시작할 수 없습니다. ChromaDB 컬렉션을 확인해주세요.")
            st.stop()

        with st.spinner("답변을 생성하고 있습니다..."):
            response = get_rag_response(prompt)
            
            answer = response["answer"]
            sources = response["source_documents"]

            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": sources
            })
            
            if sources:
                with st.expander("참고 문서 보기"):
                    for idx, source in enumerate(sources):
                        st.markdown(f"**문서 #{idx+1}** (유사도: {source['score']:.2f})")
                        if 'title' in source['metadata']:
                            st.markdown(f"**제목:** {source['metadata']['title']}")
                        st.markdown(f"**내용:** {source['content'][:200]}...")
                        st.markdown("---")