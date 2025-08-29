import logging
from dotenv import load_dotenv
import sys

# 로깅 설정: 반드시 stderr로 출력
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("slm_mcp-server")
# Global strings for API endpoints and paths
DOMAIN = "https://openapi.koreainvestment.com:9443"
VIRTUAL_DOMAIN = "https://openapivts.koreainvestment.com:29443"  # 모의투자

# API paths
STOCK_PRICE_PATH = "/uapi/domestic-stock/v1/quotations/inquire-price"  # 현재가조회
BALANCE_PATH = "/uapi/domestic-stock/v1/trading/inquire-balance"  # 잔고조회
TOKEN_PATH = "/oauth2/tokenP"  # 토큰발급
HASHKEY_PATH = "/uapi/hashkey"  # 해시키발급
ORDER_PATH = "/uapi/domestic-stock/v1/trading/order-cash"  # 현금주문
ORDER_LIST_PATH = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"  # 일별주문체결조회
ORDER_DETAIL_PATH = "/uapi/domestic-stock/v1/trading/inquire-ccnl"  # 주문체결내역조회
STOCK_INFO_PATH = "/uapi/domestic-stock/v1/quotations/inquire-daily-price"  # 일별주가조회
STOCK_HISTORY_PATH = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"  # 주식일별주가조회
STOCK_ASK_PATH = "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn"  # 주식호가조회

# 해외주식 API 경로
OVERSEAS_STOCK_PRICE_PATH = "/uapi/overseas-price/v1/quotations/price"
OVERSEAS_ORDER_PATH = "/uapi/overseas-stock/v1/trading/order"
OVERSEAS_BALANCE_PATH = "/uapi/overseas-stock/v1/trading/inquire-balance"
OVERSEAS_ORDER_LIST_PATH = "/uapi/overseas-stock/v1/trading/inquire-daily-ccld"

# Headers and other constants
CONTENT_TYPE = "application/json"
AUTH_TYPE = "Bearer"

# Market codes for overseas stock
MARKET_CODES = {
    "NASD": "나스닥",
    "NYSE": "뉴욕",
    "AMEX": "아멕스",
    "SEHK": "홍콩",
    "SHAA": "중국상해",
    "SZAA": "중국심천",
    "TKSE": "일본",
    "HASE": "베트남 하노이",
    "VNSE": "베트남 호치민"
}