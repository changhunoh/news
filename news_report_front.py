# app.py
import os
from typing import List, Dict, Any, Optional
import streamlit as st

from news_report_service import NewsReportService

st.set_page_config(page_title="연금술사의 비밀서재📘", page_icon="📰", layout="centered")

SAMPLE_FINAL_REPORT =  """
### 금융 섹터 주요 종목 종합 분석 리포트

#### 1. 종목별 핵심 뉴스와 가격 영향 경로 비교

📌 **우리금융지주: 비핵심자산 매각을 통한 자본적정성 강화**
*   **핵심 뉴스**: 경기도 안성 연수원 매각 추진. 최소 매각 희망가 `400억원`. 이는 약 `2조원` 규모의 전체 유휴 부동산 매각 계획의 일환.
*   **가격 영향 경로**:
    *   📈 **긍정적 (중기)**: 유형자산 매각 이익은 **자본비율(CET1)을 직접적으로 개선**하는 효과. 재무 건전성 강화는 향후 M&A 등 성장 동력 확보를 위한 실탄으로 활용 가능. 금융당국의 조건부 승인 사항 이행이라는 점도 불확실성 해소 요인.

📌 **삼성화재: 디지털 채널 기반 여행보험의 폭발적 성장**
*   **핵심 뉴스**: 해외여행보험 가입자 및 원수보험료 급증(전월 대비 각 `37.6%`, `37.1%` 증가). 디지털 플랫폼을 통한 젊은 층(`2030` 비중 `52.6%`) 유입 성공.
*   **가격 영향 경로**:
    *   📈 **긍정적 (단기/중기)**: 리오프닝에 따른 **견조한 실적 성장세**를 직접적으로 증명. 특히, 디지털 채널 경쟁력과 상품 혁신(연간 보험, 선물하기 등)은 지속 가능한 성장 모델을 제시하며 **수익성 및 시장 지배력 강화**에 기여.

📌 **삼성생명: 뚜렷한 모멘텀 부재 속 상대적 강세**
*   **핵심 뉴스**: 시장 보합세 속에서 `2.33%` 상승하며 상대적 강세 시현. 외국인 투자자 동향이 주요 변수로 작용.
*   **가격 영향 경로**:
    *   📈 **긍정적 (단기)**: 뚜렷한 개별 호재 없이 시장 대비 강한 수급이 유입되고 있다는 점은 긍정적.
    *   ⚠️ **중립적 (중기)**: 다만, **펀더멘털 개선을 동반한 상승이 아니므로** 추세 지속 여부는 불확실. 금리 등 거시 변수와 외국인 수급에 따라 변동성 확대 가능.

📌 **카카오뱅크: 스톡옵션 행사에 따른 오버행(Overhang) 우려**
*   **핵심 뉴스**: 주식매수선택권(스톡옵션) 행사에 따른 보통주 추가 상장.
*   **가격 영향 경로**:
    *   ⚠️ **부정적 (단기)**: 신규 주식 유통에 따른 **공급 물량 부담(오버행)**은 기존 주주 지분가치 희석 우려를 낳으며 주가에 단기 하방 압력으로 작용. 임직원 보상이라는 장기적 긍정 효과보다 단기적 수급 부담이 더 크게 부각될 가능성.

📌 **메리츠금융지주: 주주환원 정책의 핵심, 감액배당에 대한 과세**
*   **핵심 뉴스**: 정부의 감액배당(자본준비금 활용 배당)에 대한 대주주 배당소득세 과세 방침 발표.
*   **가격 영향 경로**:
    *   ⚠️ **부정적 (중기)**: 메리츠금융의 **핵심 투자 포인트였던 고배당 및 주주환원 정책의 매력도 저하** 우려. 과거와 같은 파격적인 감액배당이 어려워질 경우, 배당 정책의 변화가 불가피하며 이는 기업가치 평가에 부정적 영향을 미칠 수 있음.

#### 2. 공통 테마 식별 및 교차영향 설명

📌 **규제 및 세제 변화의 직접적 영향**
*   금융 산업은 본질적으로 규제 산업임을 재확인. **메리츠금융지주**는 감액배당 과세라는 세제 변화에 직접적인 타격을 입으며, 이는 회사의 핵심 전략 수정으로 이어질 수 있는 중대한 변수. **우리금융지주** 역시 금융당국의 조건부 승인(유휴 부동산 매각)을 이행하는 과정에서 자산 매각이 진행되는 등 규제 환경이 경영 전략에 미치는 영향이 지대함. **카카오뱅크** 또한 잠재적인 플랫폼 규제 리스크에 상시 노출되어 있음.

📌 **금리 및 거시경제 환경의 차별적 영향**
*   보고서에 직접 언급되진 않았으나, 금리는 금융주 전반을 관통하는 핵심 테마.
*   **은행주(우리금융지주, 카카오뱅크)**는 금리 상승 시 순이자마진(NIM) 개선 기대감이 존재.
*   **보험주(삼성생명, 삼성화재)**는 금리 상승이 신규 투자자산의 운용수익률을 높여 긍정적이나, 동시에 보유 채권 평가손실을 유발할 수 있어 영향이 복합적. 특히 **삼성생명**과 같은 생명보험사는 자산 듀레이션이 길어 금리 변동에 더 민감.

📌 **디지털 전환(Digital Transformation) 성과**
*   **삼성화재**는 디지털 채널을 통한 성공적인 고객 확보 사례를 명확히 보여줌. 이는 전통 금융사가 신성장 동력을 어떻게 창출하는지 보여주는 모범 사례. 반면, 디지털 네이티브인 **카카오뱅크**는 성장성과 별개로 수급 이슈에 발목이 잡힌 상황으로, 기술력 외 전통적 금융 분석 지표의 중요성도 부각됨.

#### 3. 종목별 리스크/촉발요인 및 모니터링 지표

**우리금융지주**
*   ⚠️ **리스크**: 부동산 경기 침체로 인한 자산 매각 지연 또는 매각가 하락.
*   📈 **촉발요인**: 성공적인 자산 매각 후 확보 자금을 활용한 비은행 M&A 추진 발표.
*   📌 **모니터링 지표**: 분기별 자본비율(CET1) 추이, 부동산 매각 진행 경과 및 최종 매각가.

**삼성화재**
*   ⚠️ **리스크**: 경기 둔화에 따른 해외여행 수요 감소, 디지털 보험 시장 경쟁 심화.
*   📈 **촉발요인**: 여행보험 외 다른 보종(자동차, 장기 등)에서의 디지털 채널 성과 가시화.
*   📌 **모니터링 지표**: 월별 해외 출국자 수, 여행보험 원수보험료 증감률, 경쟁사 디지털 상품 출시 동향.

**삼성생명**
*   ⚠️ **리스크**: 금리 급등 시 보유 채권 평가손실 확대, 뚜렷한 성장 모멘텀 부재.
*   📈 **촉발요인**: 신계약 CSM(계약서비스마진)의 예상 상회 성장, 시장 기대치를 뛰어넘는 주주환원 정책 발표.
*   📌 **모니터링 지표**: 국고채 10년물 금리 추이, 외국인 순매수 동향, 분기별 신계약 CSM 규모.

**카카오뱅크**
*   ⚠️ **리스크**: 잔여 스톡옵션 물량의 추가 행사 가능성, 플랫폼 규제 강화 우려.
*   📈 **촉발요인**: 대출 성장률 회복 및 연체율의 안정적 관리, 수급 부담을 상쇄할 만한 혁신적 신규 서비스 출시.
*   📌 **모니터링 지표**: 스톡옵션 행사 가능 물량, 분기별 여신 성장률 및 연체율, 월간활성이용자수(MAU).

**메리츠금융지주**
*   ⚠️ **리스크**: **주주환원 정책의 불확실성 증대**, 감액배당 축소에 따른 투자 매력도 감소.
*   📈 **촉발요인**: 불확실성을 해소할 수 있는 새로운 중장기 주주환원 정책(자사주 매입/소각 강화 등) 발표.
*   📌 **모니터링 지표**: 차기 배당 정책 관련 공시, 경영진의 주주환원 관련 커뮤니케이션.

#### 4. 결론: 포트폴리오 관점 제언

금융 섹터 내에서도 개별 종목의 모멘텀이 뚜렷하게 갈리고 있습니다. 규제, 수급, 실적 등 각기 다른 동인에 의해 주가 향방이 결정되는 국면입니다.

*   **Overweight (비중확대): 삼성화재, 우리금융지주**
    *   **삼성화재**는 리오프닝 국면에서 가장 확실한 실적 개선 스토리를 보여주고 있으며, 디지털 전환 성과가 가시화되고 있어 지속적인 관심이 유효합니다.
    *   **우리금융지주**는 자산 매각을 통한 펀더멘털(자본비율) 개선이라는 명확한 목표를 가지고 있으며, 현재 주가 수준은 이러한 개선 여력을 충분히 반영하지 못하고 있다고 판단됩니다.

*   **Neutral (중립): 삼성생명**
    *   업종 대표주로서 안정성은 갖추고 있으나, 현재 시점에서 주가 상승을 이끌 뚜렷한 촉발요인이 부족합니다. 금리 환경 변화와 수급을 확인하며 접근할 필요가 있습니다.

*   **Underweight (비중축소): 메리츠금융지주, 카카오뱅크**
    *   **메리츠금융지주**는 핵심 투자 논리였던 주주환원 정책에 대한 불확실성이 해소되기 전까지 보수적인 접근이 필요합니다. 정책 변화의 폭과 그에 따른 시장 반응을 지켜봐야 합니다.
    *   **카카오뱅크**는 단기적인 스톡옵션 오버행 이슈가 주가에 부담으로 작용하고 있습니다. 성장성에 대한 기대감만으로 단기 수급 악화를 극복하기는 어려울 수 있습니다.

결론적으로, 현재 포트폴리오 전략은 **펀더멘털 개선 가시성이 높은 종목(우리금융, 삼성화재)을 중심으로 비중을 확대**하고, **정책 및 수급 관련 불확실성에 노출된 종목(메리츠, 카카오뱅크)은 비중을 줄여 리스크를 관리**하는 것이 바람직합니다.
"""


# -----------------------------
# st.secrets → os.environ 주입
# -----------------------------
def _prime_env_from_secrets() -> None:
    try:
        if hasattr(st, "secrets") and st.secrets:
            for k, v in st.secrets.items():
                if k == "gcp_service_account":
                    continue
                os.environ.setdefault(k, str(v))
    except Exception:
        pass
_prime_env_from_secrets()

# -----------------------------
# helpers
# -----------------------------
def _parse_stocks(csv_text: str) -> List[str]:
    return [s.strip() for s in csv_text.split(",") if s.strip()]

def _fmt_link(md: Dict[str, Any]) -> str:
    title = md.get("title") or md.get("headline") or md.get("doc_title") or md.get("doc_id") or ""
    url = md.get("url") or md.get("link") or md.get("source_url") or ""
    if url and title: return f"[{title}]({url})"
    if url:           return f"[원문 링크]({url})"
    return title or "(링크/제목 없음)"

# -----------------------------
# Service 인스턴스 (캐시)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_service() -> Optional[NewsReportService]:
    try:
        return NewsReportService()
    except Exception as e:
        st.warning(f"서비스 초기화 실패: {e}")
        return None


# ======================
# CSS: 사이드바 hover 시에만 보이도록
# ======================
st.markdown(
    """
    <style>
    /* 사이드바 전체를 왼쪽으로 숨김 */
    [data-testid="stSidebar"] {
        transform: translateX(-250px);
        transition: all 0.3s;
        opacity: 0.2;  /* 살짝만 보이게 */
    }
    /* 마우스를 올리면 원위치 */
    [data-testid="stSidebar"]:hover {
        transform: translateX(0);
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# UI
# -----------------------------
st.title("📘 연금술사의 비밀서재")
st.markdown(
    "<p style='font-size:20px; color:gray;'>✨ 창훈님을 위한 연금술사의 연금리포트 ✨</p>",
    unsafe_allow_html=True
)
#st.caption("우리 연금술사가 창훈님을 위해 제작한 퇴직연금 종합 리포트에요")

with st.sidebar:
    st.subheader("실행 설정")
    stocks_text = st.text_input("종목(콤마로 구분)", value="우리금융지주,삼성화재,삼성생명,카카오뱅크,메리츠금융지주", help="예: 삼성전자,우리금융지주 / 또는 AAPL,NVDA")
    #template = st.text_input("질문 템플릿 (옵션)", value="{stock} 관련해서 종목의 가격에 중요한 뉴스는?")
    #top_k = st.number_input("top_k", min_value=1, max_value=20, value=5, step=1)
    #use_rerank = st.toggle("리랭크 사용 (현재는 top_k 자르기)", value=False)
    #rerank_top_k = st.number_input("rerank_top_k", min_value=1, max_value=50, value=5, step=1)
    #max_workers = st.slider("동시 처리 쓰레드", min_value=1, max_value=10, value=5)
    run_btn = st.button("🚀 실행", type="primary")

st.divider()
# st.markdown("""
# **필수 Secrets:** `GOOGLE_CLOUD_PROJECT`, `QDRANT_URL`, `QDRANT_API_KEY`  
# (옵션) `GOOGLE_CLOUD_LOCATION`, `COLLECTION_NAME`, `EMBED_MODEL_NAME`, `GENAI_MODEL_NAME`, `EMBED_DIM`, `DEFAULT_TOP_K`, `RERANK_TOP_K`, `[gcp_service_account]`
# """)

if run_btn:
    stocks = _parse_stocks(stocks_text)
    if not stocks:
        st.error("종목을 1개 이상 입력해 주세요.")
        st.stop()

    svc = get_service()
    if svc is None:
        st.error("서비스 초기화 실패. Secrets 설정을 확인해 주세요.")
        st.stop()

    # 런타임 설정 반영
    #svc.top_k = int(top_k)
    #svc.use_rerank = bool(use_rerank)
    #svc.rerank_top_k = int(rerank_top_k)

    # (선택) 디버깅: 각 종목의 보유 문서 수
    cols = st.columns(len(stocks))
    for i, s in enumerate(stocks):
        with cols[i]:
            try:
                c = svc.count_by_stock(s)
                st.caption(f"`{s}` 문서 수: **{c}**")
            except Exception:
                pass

    with st.spinner("리포트를 생성하는중..."):
        try:
            #base_template = (template or None)
            result = svc.answer_5_stocks_and_reduce(
                stocks=stocks,
                template=f"{stocks} 관련해서 종목의 가격에 영향을 미치는 중요한 뉴스는?",
                max_workers=int(5),
            )
        except Exception as e:
            st.exception(e)
            st.stop()
    # 최종 리포트
    st.subheader("📌 최종 리포트")
    final_report = (result.get("final_report") or "").strip()
    if final_report:
        st.markdown(final_report)
    else:
        st.info("최종 리포트를 생성하지 못했습니다.")

    st.divider()

    # 종목별 결과
    st.subheader("🔎 종목별 요약보기")
    for r in result.get("results", []):
        stock = r.get("stock", "")
        with st.expander(f"[{stock}] 요약 보기", expanded=False):
            ans = (r.get("answer") or "").strip()
            if ans:
                st.markdown(ans)
            else:
                st.write("관련된 정보를 찾을 수 없습니다.")

            # 소스 문서
            src_docs = r.get("source_documents") or []
            if src_docs:
                st.markdown("**근거 기사**")
                for i, d in enumerate(src_docs[:10], start=1):
                    md = d.get("metadata") or {}
                    score = d.get("score")
                    dist_mode = d.get("distance_mode") or ""
                    link = _fmt_link(md)
                    # 루트 스키마: doc_id/stock/chunk_idx/… 노출
                    extra = []
                    if "stock" in md: extra.append(f"stock=`{md.get('stock')}`")
                    if "doc_id" in md: extra.append(f"doc_id=`{md.get('doc_id')}`")
                    if "chunk_idx" in md: extra.append(f"chunk=`{md.get('chunk_idx')}`")
                    meta_line = " • ".join(extra)
                    #st.markdown(f"- {i}. {link}  \n  - {meta_line} • score(raw): `{score}` • mode: `{dist_mode}`")
                    st.markdown(f"- {i}. {link}")
            else:
                st.write("소스 문서 없음")
#최종리포트 예시 추가
else:
    st.subheader("📌 최종 리포트")
    st.markdown(SAMPLE_FINAL_REPORT)

    st.divider()
    st.subheader("🔎 종목별 요약보기")




























