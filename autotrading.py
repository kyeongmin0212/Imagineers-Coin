import os
import sys
import json
import time
import asyncio
import logging
import threading
import hashlib
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from dotenv import load_dotenv
import pyupbit
from openai import OpenAI
import ta
import telegram
from telegram import Bot
from filelock import FileLock
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

# Windows 콘솔에서 유니코드 지원
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 로깅 설정
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('trading_bot.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
# httpx/telegram 라이브러리가 URL에 봇 토큰을 포함하여 로깅하는 것을 방지
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 환경 변수 로드 (OneDrive 외부 경로에서 로드)
_ENV_PATH = os.path.join(os.path.expanduser("~"), ".env_autocoin")
if not os.path.exists(_ENV_PATH):
    raise FileNotFoundError(f".env 파일을 찾을 수 없습니다: {_ENV_PATH}\n{_ENV_PATH} 경로에 .env 파일을 생성하세요.")
load_dotenv(_ENV_PATH)

# 상수 정의
COINS_FILE = "purchased_coins.json"
TRADE_HISTORY_FILE = "trade_history.json"
POSITIONS_FILE = "positions.json"
CACHE_DIR = "api_cache"
MIN_KRW_BALANCE = 5000
# AI 시장 위험도(risk_level): 이 값 이상이면 신규 매수 금지(기존 >=7과 동일)
MARKET_RISK_HIGH_THRESHOLD = int(os.getenv("MARKET_RISK_HIGH_THRESHOLD", "8"))
MAX_COINS = 5
DCA_RATIO = 0.05
TRADING_FEE = 0.0005  # 업비트 매수/매도 각 0.05%
API_CALL_LIMIT = 30
ORDER_LIMIT = 10
CACHE_EXPIRY_SECONDS = {
    'default': 1200,
    'market_trend': 1200,
    'price_prediction': 1200,
    'opportunities': 1200
}

# 메인 사이클 sleep 중에도 미실현 손실이 -stop_loss% 이하이면 시장가 손절 (초 단위 폴링)
EMERGENCY_STOP_POLL_SECONDS = int(os.getenv("EMERGENCY_STOP_POLL_SECONDS", "10"))

# 캐시 디렉토리 생성
os.makedirs(CACHE_DIR, exist_ok=True)

# API 키 설정
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")

logger.info("API 키 확인 중...")
missing_keys = []
for key, value in [
    ("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
    ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
    ("OPENAI_API_KEY", OPENAI_API_KEY),
    ("UPBIT_ACCESS_KEY", UPBIT_ACCESS_KEY),
    ("UPBIT_SECRET_KEY", UPBIT_SECRET_KEY)
]:
    if not value:
        missing_keys.append(key)
if missing_keys:
    logger.error(f"누락된 API 키: {', '.join(missing_keys)}")
    raise ValueError("환경 변수 파일(.env)에 하나 이상의 API 키가 누락되었습니다.")

# 업비트 클라이언트 초기화
upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
logger.info("업비트 클라이언트 초기화 완료")

# JSON 직렬화를 위한 유틸리티 함수
def serialize_numpy(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def extract_executed_price(order_info, executed_amount, fallback_price):
    """업비트 주문 응답에서 정확한 체결 단가를 추출한다.

    시장가 매수 주문의 경우 order_info["price"]는 총 KRW 금액이고,
    avg_price 필드도 간헐적으로 총 금액이 들어오는 버그가 있다.
    현재가(fallback_price) 대비 5배 이상 차이나면 총 금액으로 간주하고 보정한다.
    """
    if not order_info or executed_amount <= 0:
        return fallback_price

    # 1) trades 배열에서 가중 평균 단가 계산 (가장 정확)
    trades = order_info.get("trades")
    if trades and isinstance(trades, list):
        total_funds = sum(float(t.get("funds", 0)) for t in trades)
        total_vol = sum(float(t.get("volume", 0)) for t in trades)
        if total_vol > 0 and total_funds > 0:
            unit_price = total_funds / total_vol
            if fallback_price and abs(unit_price / fallback_price - 1) < 5:
                return unit_price

    # 2) avg_price 필드
    avg_price_str = order_info.get("avg_price")
    if avg_price_str:
        avg_price_val = float(avg_price_str)
        if avg_price_val > 0:
            # 현재가 대비 5배 이상이면 총 금액이 들어온 것으로 판단
            if fallback_price and fallback_price > 0 and avg_price_val / fallback_price > 5:
                corrected = avg_price_val / executed_amount
                logger.warning(
                    f"avg_price 보정: {avg_price_val:,.2f} -> {corrected:,.2f} "
                    f"(현재가={fallback_price:,.2f}, 수량={executed_amount:.8f})"
                )
                return corrected
            return avg_price_val

    # 3) price 필드 (시장가 매수: 총 KRW, 지정가: 주문 단가)
    price_str = order_info.get("price")
    if price_str:
        price_val = float(price_str)
        if price_val > 0:
            if fallback_price and fallback_price > 0 and price_val / fallback_price > 5:
                return price_val / executed_amount
            return price_val

    return fallback_price


def verify_price_with_balance(ticker, price):
    """매수 체결 후 업비트 잔고의 avg_buy_price와 대조하여 검증.
    차이가 1% 이상이면 업비트 값으로 교체한다.
    """
    try:
        balances = upbit.get_balances()
        if not balances:
            return price
        currency = ticker.replace("KRW-", "")
        for b in balances:
            if b.get('currency') == currency:
                real_avg = float(b.get('avg_buy_price', 0))
                if real_avg > 0:
                    diff_pct = abs(price - real_avg) / real_avg
                    if diff_pct > 0.01:
                        logger.warning(
                            f"{ticker} 체결가 보정: {price:,.2f} -> {real_avg:,.2f} "
                            f"(업비트 avg_buy_price 기준, 차이 {diff_pct:.1%})"
                        )
                        return real_avg
                break
    except Exception as e:
        logger.warning(f"{ticker} 잔고 검증 실패: {e}")
    return price


async def resolve_executed_price(trading_history, ticker, side, order_info, executed_amount, fallback_price, reason, *, verify_balance=True, entry_price=None):
    """체결가 추출 → 잔고 검증 → 거래 기록을 한 번에 수행한다.
    Returns: 최종 확정된 체결 단가
    """
    price = extract_executed_price(order_info, executed_amount, fallback_price)
    if verify_balance:
        price = verify_price_with_balance(ticker, price)
    kwargs = {}
    if entry_price is not None:
        kwargs['entry_price'] = entry_price
    await trading_history.add_trade(ticker, side, price, executed_amount, reason, **kwargs)
    return price


def safe_get_current_price(ticker):
    """pyupbit.get_current_price 안전 래퍼 — 실패 시 1초 후 1회 재시도, 그래도 실패하면 None."""
    for attempt in range(2):
        try:
            price = pyupbit.get_current_price(ticker)
            if price and price > 0:
                return price
        except Exception:
            pass
        if attempt == 0:
            time.sleep(1)
    logger.warning(f"{ticker} 현재 가격 가져오기 실패 (2회 시도)")
    return None


# AI 응답 검증 함수
def validate_market_analysis(data: dict) -> dict:
    """analyze_market() GPT-4 응답 스키마 검증. 실패 시 ValueError 발생."""
    if not isinstance(data, dict):
        raise ValueError(f"응답이 dict가 아닙니다: {type(data)}")
    # risk_level: 1~10 정수
    risk_level = data.get("risk_level")
    try:
        risk_level = int(risk_level)
    except (TypeError, ValueError):
        raise ValueError(f"risk_level이 숫자가 아닙니다: {risk_level!r}")
    if not (1 <= risk_level <= 10):
        raise ValueError(f"risk_level 범위 오류 (1~10): {risk_level}")
    # buy_ratio / sell_ratio: "0%"~"100%" 또는 0~100 숫자
    for field in ("buy_ratio", "sell_ratio"):
        val = data.get(field)
        if val is None:
            raise ValueError(f"필드 누락: {field}")
        try:
            float(str(val).replace("%", ""))
        except ValueError:
            raise ValueError(f"{field} 값 파싱 실패: {val!r}")
    # key_indicators: 리스트
    if not isinstance(data.get("key_indicators"), list):
        raise ValueError(f"key_indicators가 리스트가 아닙니다: {data.get('key_indicators')!r}")
    # news_sentiment: 선택적 필드 (positive/negative/neutral)
    ns = data.get("news_sentiment")
    if ns is not None and ns not in ("positive", "negative", "neutral"):
        data["news_sentiment"] = "neutral"
    # news_impact: 선택적 필드 (high/medium/low)
    ni = data.get("news_impact")
    if ni is not None and ni not in ("high", "medium", "low"):
        data["news_impact"] = "low"
    return data

def validate_opportunities(data: dict) -> list:
    """scan_for_opportunities() GPT-4 응답 스키마 검증. 실패 시 ValueError 발생."""
    if not isinstance(data, dict):
        raise ValueError(f"응답이 dict가 아닙니다: {type(data)}")
    recommendations = data.get("recommendations")
    if not isinstance(recommendations, list):
        raise ValueError(f"recommendations가 리스트가 아닙니다: {recommendations!r}")
    validated = []
    for i, item in enumerate(recommendations):
        if not isinstance(item, dict):
            logger.warning(f"추천 항목 {i} 형식 오류, 건너뜀: {item!r}")
            continue
        ticker = item.get("ticker")
        if not isinstance(ticker, str) or not ticker.startswith("KRW-"):
            logger.warning(f"추천 항목 {i} ticker 오류, 건너뜀: {ticker!r}")
            continue
        score = item.get("score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            logger.warning(f"추천 항목 {i} score 파싱 실패, 건너뜀: {score!r}")
            continue
        if not (0 <= score <= 100):
            logger.warning(f"추천 항목 {i} score 범위 오류 (0~100), 건너뜀: {score}")
            continue
        if not isinstance(item.get("reason"), str):
            logger.warning(f"추천 항목 {i} reason 누락 또는 형식 오류, 건너뜀")
            continue
        validated.append({**item, "score": score})
    return validated

# Rate Limiter
class RateLimiter:
    def __init__(self):
        self.order_lock = threading.Lock()
        self.api_lock = threading.Lock()
        self.order_count = 0
        self.api_count = 0
        self.last_reset = time.time()

    def _reset_if_needed(self):
        current_time = time.time()
        if current_time - self.last_reset >= 1:
            with self.order_lock:
                self.order_count = 0
            with self.api_lock:
                self.api_count = 0
            self.last_reset = current_time

    def can_make_order(self):
        self._reset_if_needed()
        with self.order_lock:
            if self.order_count < ORDER_LIMIT:
                self.order_count += 1
                return True
            return False

    def can_make_api_call(self, max_wait=10):
        self._reset_if_needed()
        start_time = time.time()
        with self.api_lock:
            while self.api_count >= API_CALL_LIMIT:
                if time.time() - start_time > max_wait:
                    logger.warning("API 호출 대기 시간 초과")
                    return False
                time.sleep(0.1)
                self._reset_if_needed()
            self.api_count += 1
            return True

rate_limiter = RateLimiter()

# 캐시 데코레이터
def cache_result(expiry_seconds=14400):
    def decorator(func):
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        async def async_wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
            lock_file = f"{cache_file}.lock"

            with FileLock(lock_file):
                try:
                    if os.path.exists(cache_file):
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                            if time.time() - cache_data.get('timestamp', 0) < expiry_seconds:
                                logger.info(f"{func.__name__} 캐시 결과 사용")
                                return cache_data['result']
                except Exception as e:
                    logger.error(f"{func.__name__} 캐시 읽기 실패: {e}")

                result = await func(*args, **kwargs)

                try:
                    cache_data = {
                        'timestamp': time.time(),
                        'result': json.loads(json.dumps(result, default=serialize_numpy))
                    }
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False)
                    logger.info(f"{func.__name__} 캐시 저장 완료")
                except Exception as e:
                    logger.error(f"{func.__name__} 캐시 저장 실패: {e}")

                return result

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def sync_wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
            lock_file = f"{cache_file}.lock"

            with FileLock(lock_file):
                try:
                    if os.path.exists(cache_file):
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                            if time.time() - cache_data.get('timestamp', 0) < expiry_seconds:
                                logger.info(f"{func.__name__} 캐시 결과 사용")
                                return cache_data['result']
                except Exception as e:
                    logger.error(f"{func.__name__} 캐시 읽기 실패: {e}")

                result = func(*args, **kwargs)

                try:
                    cache_data = {
                        'timestamp': time.time(),
                        'result': json.loads(json.dumps(result, default=serialize_numpy))
                    }
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False)
                    logger.info(f"{func.__name__} 캐시 저장 완료")
                except Exception as e:
                    logger.error(f"{func.__name__} 캐시 저장 실패: {e}")

                return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def invalidate_cache(func_name, ticker):
    cache_key = f"{func_name}_{hashlib.md5(ticker.encode()).hexdigest()}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        logger.info(f"{func_name} 캐시 무효화: {ticker}")

class TradingError(Exception):
    pass

@cache_result(expiry_seconds=CACHE_EXPIRY_SECONDS['default'])
def get_fear_and_greed():
    while not rate_limiter.can_make_api_call():
        time.sleep(0.1)
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info("공포와 탐욕 지수 가져오기 성공")
        return int(data['data'][0]['value'])
    except Exception as e:
        logger.error(f"공포와 탐욕 지수 가져오기 실패: {e}")
        return 50

def get_google_news_rss(max_results=10):
    """Google News RSS에서 암호화폐/금융 관련 뉴스를 가져옵니다."""
    keywords = ["bitcoin crypto", "cryptocurrency regulation", "비트코인", "암호화폐 규제"]
    all_articles = []
    for keyword in keywords:
        try:
            url = f"https://news.google.com/rss/search?q={quote_plus(keyword)}&hl=ko&gl=KR&ceid=KR:ko"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            channel = root.find("channel")
            if channel is None:
                continue
            for item in channel.findall("item")[:5]:
                title = item.findtext("title", "")
                pub_date = item.findtext("pubDate", "")
                source = item.findtext("source", "")
                all_articles.append({
                    "title": title,
                    "source": source,
                    "published_at": pub_date,
                    "keyword": keyword,
                })
        except Exception as e:
            logger.warning(f"Google News RSS 가져오기 실패 (키워드: {keyword}): {e}")
            continue
    # 중복 제거 (제목 기준)
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        if article["title"] not in seen_titles:
            seen_titles.add(article["title"])
            unique_articles.append(article)
    logger.info(f"Google News RSS 뉴스 {len(unique_articles)}건 가져오기 성공")
    return unique_articles[:max_results]


def collect_news_data():
    """Google News RSS에서 뉴스를 수집하고 요약합니다."""
    google_news = get_google_news_rss(max_results=20)

    return {
        "crypto_news": [],
        "google_news": google_news,
        "sentiment_summary": {"positive": 0, "negative": 0, "neutral": 0},
        "total_articles": len(google_news),
    }


class FileManager:
    @staticmethod
    def load_json(filename, default=None):
        lock_file = f"{filename}.lock"
        with FileLock(lock_file):
            try:
                if os.path.exists(filename):
                    with open(filename, "r", encoding='utf-8') as f:
                        return json.load(f)
                return default if default is not None else {}
            except Exception as e:
                logger.error(f"{filename} 파일 로드 실패: {e}")
                return default if default is not None else {}

    @staticmethod
    def save_json(filename, data):
        lock_file = f"{filename}.lock"
        with FileLock(lock_file):
            try:
                with open(filename, "w", encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x.tolist() if isinstance(x, np.ndarray) else str(x))
                logger.info(f"{filename} 파일 저장 완료")
            except Exception as e:
                logger.error(f"{filename} 파일 저장 실패: {e}")

async def async_send_telegram_message(message, max_retries=3):
    for attempt in range(max_retries):
        try:
            max_length = 4096
            if len(message) > max_length:
                message = message[:max_length - 3] + "..."
            logger.info(f"텔레그램 메시지 전송 시도 {attempt + 1}/{max_retries}: {message}")
            if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
                logger.error("텔레그램 토큰 또는 채팅 ID 누락")
                return False
            bot = Bot(token=TELEGRAM_TOKEN)
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            logger.info("텔레그램 메시지 전송 성공")
            return True
        except telegram.error.TelegramError as e:
            logger.error(f"텔레그램 메시지 전송 실패: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"텔레그램 메시지 전송 중 예상치 못한 오류: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    logger.error("텔레그램 메시지 전송 실패: 최대 재시도 횟수 초과")
    return False

class TradingHistory:
    def __init__(self):
        self.history = FileManager.load_json(TRADE_HISTORY_FILE, {"trades": [], "performance": {}, "blacklist": {}})
        logger.info("거래 내역 초기화 완료")

    def add_to_blacklist(self, ticker, cooldown_hours=4):
        """손절 후 코인을 일정 시간 블랙리스트에 추가"""
        expiry = (datetime.now() + timedelta(hours=cooldown_hours)).isoformat()
        self.history["blacklist"][ticker] = expiry
        FileManager.save_json(TRADE_HISTORY_FILE, self.history)
        logger.info(f"{ticker} 블랙리스트 추가: {cooldown_hours}시간")

    def is_blacklisted(self, ticker):
        blacklist = self.history.get("blacklist", {})  # 안전하게 접근
        logger.info(f"블랙리스트 확인: {blacklist}")
        if ticker in blacklist:
            expiry = datetime.fromisoformat(blacklist[ticker])
            if datetime.now() < expiry:
                logger.info(f"{ticker} 블랙리스트에 있음, 만료: {expiry}")
                return True
            else:
                del blacklist[ticker]
                FileManager.save_json(TRADE_HISTORY_FILE, self.history)
                logger.info(f"{ticker} 블랙리스트 만료")
        return False
    
    async def analyze_loss(self, ticker, trade):
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=30)
            if df is None or len(df) < 14:
                logger.warning(f"{ticker} 손실 분석 데이터 부족")
                return None
            df = MarketAnalyzer().add_technical_indicators(df)
            trade_time = datetime.fromisoformat(trade['timestamp'])
            df_index = df.index.get_indexer([trade_time], method='nearest')[0]
            indicators = {
                'rsi': float(df['rsi'].iloc[df_index]) if 'rsi' in df else None,
                'macd_diff': float(df['macd_diff'].iloc[df_index]) if 'macd_diff' in df else None,
                'price_change_7d': float(((df['close'].iloc[df_index] - df['close'].iloc[df_index-7]) / df['close'].iloc[df_index-7] * 100)) if df_index >= 7 else None,
                'fear_greed': get_fear_and_greed()
            }
            analysis = {
                "ticker": ticker,
                "trade": trade,
                "indicators": indicators,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"{ticker} 손실 분석 완료: {json.dumps(analysis, default=serialize_numpy)}")
            await async_send_telegram_message(
                f"📉 손실 분석: {ticker}\n"
                f"RSI: {indicators['rsi']:.2f}\n"
                f"MACD 차이: {indicators['macd_diff']:.2f}\n"
                f"7일 가격 변화: {indicators['price_change_7d']:.2f}%\n"
                f"공포/탐욕 지수: {indicators['fear_greed']}"
            )
            return analysis
        except Exception as e:
            logger.error(f"{ticker} 손실 분석 실패: {e}")
            return None

    async def add_trade(self, ticker, action, price, amount, reason, entry_price=None):
        trade = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "action": action,
            "price": float(price),
            "amount": float(amount),
            "reason": reason
        }
        if action == "sell" and entry_price is not None:
            trade["entry_price"] = float(entry_price)
        self.history["trades"].append(trade)
        FileManager.save_json(TRADE_HISTORY_FILE, self.history)
        action_text = "매수" if action == "buy" else "매도"
        message = (
            f"🔄 거래 실행: {ticker}\n"
            f"유형: {action_text}\n"
            f"가격: {price:,.2f}원\n"
            f"수량: {amount:.8f}개\n"
            f"사유: {reason}"
        )
        await async_send_telegram_message(message)
        logger.info(f"{ticker} {action_text} 거래 기록 추가: 가격={price:,.2f}원, 수량={amount:.8f}개")

    def calculate_performance(self, ticker):
        while not rate_limiter.can_make_api_call():
            time.sleep(0.1)
        try:
            trades = [t for t in self.history["trades"] if t["ticker"] == ticker]
            if not trades:
                logger.info(f"{ticker} 거래 내역 없음")
                return None
            total_invested = sum(t["amount"] * t["price"] for t in trades if t["action"] == "buy")
            total_returned = sum(t["amount"] * t["price"] for t in trades if t["action"] == "sell")
            current_price = safe_get_current_price(ticker)
            if current_price is None:
                logger.error(f"{ticker} 현재 가격 가져오기 실패")
                return None
            current_holdings = sum(t["amount"] for t in trades if t["action"] == "buy") - \
                             sum(t["amount"] for t in trades if t["action"] == "sell")
            current_value = current_holdings * current_price if current_holdings > 0 else 0
            performance = {
                "total_invested": float(total_invested),
                "total_returned": float(total_returned),
                "current_holdings": float(current_holdings),
                "current_value": float(current_value),
                "total_value": float(total_returned + current_value),
                "roi": float(((total_returned + current_value - total_invested) / total_invested * 100)) \
                    if total_invested > 0 else 0
            }
            logger.info(f"{ticker} 성과 계산: ROI={performance['roi']:.2f}%")
            return performance
        except Exception as e:
            logger.error(f"{ticker} 성과 계산 실패: {e}")
            return None

class StopLossManager:
    def __init__(self, default_stop_loss=4):
        self.default_stop_loss = default_stop_loss
        self.positions = FileManager.load_json(POSITIONS_FILE, {})
        logger.info("손절/익절 관리자 초기화 완료")

    def add_position(self, ticker, entry_price, amount, stop_loss=None):
        position = {
            "entry_price": float(entry_price),
            "amount": float(amount),
            "stop_loss": float(stop_loss or self.default_stop_loss),
            "timestamp": datetime.now().isoformat(),
            "stop_loss_order_uuid": None,
            "highest_price": float(entry_price),
            "trailing_stop_price": None,
        }
        self.positions[ticker] = position
        FileManager.save_json(POSITIONS_FILE, self.positions)
        logger.info(f"{ticker} 포지션 추가: 진입가={entry_price:,.2f}원, 수량={amount:.8f}개")
        self._place_limit_orders(ticker, entry_price, amount, position)

    def _place_limit_orders(self, ticker, entry_price, amount, position):
        try:
            stop_loss_price = entry_price * (1 - position["stop_loss"] / 100)
            if rate_limiter.can_make_order():
                sl_order = upbit.sell_limit_order(ticker, stop_loss_price, amount)
                if sl_order and "uuid" in sl_order:
                    position["stop_loss_order_uuid"] = sl_order["uuid"]
                    logger.info(f"{ticker} 손절 주문 설정: {stop_loss_price:,.2f}원")
            FileManager.save_json(POSITIONS_FILE, self.positions)
        except Exception as e:
            logger.error(f"{ticker} 지정가 주문 설정 실패: {e}")

    def update_position(self, ticker, new_avg_price=None, amount=None, stop_loss=None):
        if ticker in self.positions:
            position = self.positions[ticker]
            if new_avg_price is not None:
                position["entry_price"] = float(new_avg_price)
            if amount is not None:
                position["amount"] = float(amount)
            if stop_loss is not None:
                position["stop_loss"] = float(stop_loss)
            # highest_price가 entry_price의 3배 이상이면 비정상 → 현재가로 보정
            entry = position.get("entry_price", 0)
            highest = position.get("highest_price", 0)
            if entry > 0 and highest > entry * 3:
                current_price = safe_get_current_price(ticker)
                position["highest_price"] = float(current_price) if current_price else float(entry)
                logger.warning(f"{ticker} highest_price 비정상 보정: {highest:,.2f} → {position['highest_price']:,.2f}")
            position["updated_at"] = datetime.now().isoformat()
            if position.get("stop_loss_order_uuid"):
                try:
                    upbit.cancel_order(position["stop_loss_order_uuid"])
                    logger.info(f"{ticker} 기존 손절 주문 취소")
                except Exception as e:
                    logger.error(f"{ticker} 주문 취소 실패: {e}")
            self._place_limit_orders(ticker, position["entry_price"], position["amount"], position)
            FileManager.save_json(POSITIONS_FILE, self.positions)
            logger.info(f"{ticker} 포지션 업데이트: 평균가={new_avg_price:,.2f}원, 수량={amount:.8f}개")

    def cancel_limit_orders_only(self, ticker):
        """긴급 시장가 손절 전에 걸어둔 지정가 손절만 취소하고 포지션은 유지."""
        if ticker not in self.positions:
            return
        position = self.positions[ticker]
        if position.get("stop_loss_order_uuid"):
            try:
                upbit.cancel_order(position["stop_loss_order_uuid"])
                logger.info(f"{ticker} 손절 지정가 취소 (긴급 손절 직전)")
            except Exception as e:
                logger.error(f"{ticker} 지정가 취소 실패: {e}")
            position["stop_loss_order_uuid"] = None
        FileManager.save_json(POSITIONS_FILE, self.positions)

    def remove_position(self, ticker):
        if ticker in self.positions:
            position = self.positions[ticker]
            if position.get("stop_loss_order_uuid"):
                try:
                    upbit.cancel_order(position["stop_loss_order_uuid"])
                    logger.info(f"{ticker} 손절 주문 취소")
                except Exception as e:
                    logger.error(f"{ticker} 주문 취소 실패: {e}")
            del self.positions[ticker]
            FileManager.save_json(POSITIONS_FILE, self.positions)
            logger.info(f"{ticker} 포지션 제거")

    def update_highest_price(self, ticker, current_price):
        """최고가를 항상 업데이트 (수익률과 무관)."""
        if ticker not in self.positions:
            return
        position = self.positions[ticker]
        entry_price = float(position["entry_price"])
        highest_price = float(position.get("highest_price") or entry_price)
        if current_price > highest_price:
            position["highest_price"] = float(current_price)
            FileManager.save_json(POSITIONS_FILE, self.positions)

    def update_trailing_stop(self, ticker, current_price):
        """수익이 발생하면 트레일링 스탑 가격을 상향 조정 (낮아지지 않음)."""
        if ticker not in self.positions:
            return False
        position = self.positions[ticker]
        entry_price = float(position["entry_price"])
        profit_pct = (current_price - entry_price) / entry_price * 100

        highest_price = float(position.get("highest_price") or entry_price)

        if profit_pct < 2:
            return False  # 수익 2% 미만은 트레일링 스탑 미적용

        # 횡보장 트레일 비율 (고점 대비 하락%)
        if profit_pct >= 30:
            trail_pct = 4.0
        elif profit_pct >= 20:
            trail_pct = 3.5
        elif profit_pct >= 15:
            trail_pct = 3.0
        elif profit_pct >= 10:
            trail_pct = 2.5
        elif profit_pct >= 5:
            trail_pct = 2.0
        else:
            trail_pct = 1.5

        new_trailing_stop = highest_price * (1 - trail_pct / 100)
        old_trailing_stop = position.get("trailing_stop_price")

            # 장세별 트레일링 스탑 추천
            # 횡보장 — 타이트하게, 빠르게 먹고 나오기
            # 수익 구간	고점 대비 하락 허용
            # 2~5%	-1.5%
            # 5~10%	-2%
            # 10~15%	-2.5%
            # 15~20%	-3%
            # 20~30%	-3.5%
            # 30%+	-4%
            # 트레일링 시작: 수익 2%부터
            # 특징: 3~5% 수익에서 자주 매도됨. 큰 수익 못 내지만 손실도 적음


            # 상승장 — 넉넉하게, 큰 파도 타기
            # 수익 구간	고점 대비 하락 허용
            # 5~10%	-4%
            # 10~15%	-5%
            # 15~20%	-6%
            # 20~30%	-7%
            # 30~50%	-8%
            # 50%+	-10%
            # 트레일링 시작: 수익 5%부터
            # 특징: 흔들림에 안 털림. 30% 수익에서 7% 빠져도 23% 남음

            
            # 하락장 — 최대한 빨리 탈출
            # 수익 구간	고점 대비 하락 허용
            # 1~3%	-1%
            # 3~5%	-1.5%
            # 5~10%	-2%
            # 10%+	-2.5%
            # 트레일링 시작: 수익 1%부터
            # 손절도 타이트하게: -4% → **-3%**로 줄이는 게 좋음
            # 특징: 조금이라도 수익 나면 바로 확보. 하락장에서 수익 나는 것 자체가 행운


            
        if old_trailing_stop is None or new_trailing_stop > old_trailing_stop:
            position["trailing_stop_price"] = float(new_trailing_stop)
            FileManager.save_json(POSITIONS_FILE, self.positions)
            logger.info(
                f"{ticker} 트레일링 스탑 업데이트: {new_trailing_stop:,.2f}원 "
                f"(고점={highest_price:,.2f}원, 트레일={trail_pct}%, 수익={profit_pct:.2f}%)"
            )
            return True
        return False

    async def check_limit_orders(self):
        for ticker, position in list(self.positions.items()):
            try:
                uuid = position.get("stop_loss_order_uuid")
                if uuid and rate_limiter.can_make_api_call():
                    order_info = upbit.get_order(uuid)
                    if order_info and order_info["state"] == "done":
                        executed_amount = float(order_info["executed_volume"])
                        ref_price = float(position.get('entry_price', 0)) or safe_get_current_price(ticker) or 0
                        executed_price = await resolve_executed_price(TradingHistory(), ticker, "sell", order_info, executed_amount, ref_price, "stop_loss 실행", verify_balance=False, entry_price=position.get('entry_price'))
                        self.remove_position(ticker)
                        purchased = FileManager.load_json(COINS_FILE, {"coins": []})
                        if ticker in purchased["coins"]:
                            purchased["coins"].remove(ticker)
                            FileManager.save_json(COINS_FILE, purchased)
                        TradingHistory().add_to_blacklist(ticker)
                        invalidate_cache("scan_for_opportunities", ticker)
                        message = (
                            f"🔴 손절 실행: {ticker}\n"
                            f"체결가: {executed_price:,.2f}원\n"
                            f"수량: {executed_amount:.8f}개\n"
                            f"총 금액: {executed_amount * executed_price:,.2f}원"
                        )
                        await async_send_telegram_message(message)
            except Exception as e:
                logger.error(f"{ticker} 지정가 주문 확인 실패: {e}")

class MarketAnalyzer:
    def __init__(self):
        os.makedirs("lstm_models", exist_ok=True)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = self._build_lstm_model()
        logger.info("시장 분석기 초기화 완료")

    def _build_lstm_model(self, n_features=4):
        model = Sequential([
            Input(shape=(60, n_features)),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        logger.info(f"LSTM 모델 생성 완료 (피처 {n_features}개)")
        return model

    def _prepare_features(self, df):
        """종가 + 거래량 + RSI + MACD 4개 피처 준비"""
        feat_df = df.copy()
        feat_df['rsi'] = RSIIndicator(close=feat_df['close'], window=14).rsi()
        macd_ind = MACD(close=feat_df['close'])
        feat_df['macd_diff'] = macd_ind.macd_diff()
        feat_df = feat_df.dropna()
        features = feat_df[['close', 'volume', 'rsi', 'macd_diff']].values
        return features, feat_df

    @cache_result(expiry_seconds=CACHE_EXPIRY_SECONDS['price_prediction'])
    def predict_next_price(self, ticker):
        while not rate_limiter.can_make_api_call():
            time.sleep(0.1)
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=365)
            if df is None or len(df) < 90:
                logger.warning(f"{ticker} 데이터 부족: {len(df) if df is not None else 0}일")
                return None
            features, feat_df = self._prepare_features(df)
            if len(features) < 60:
                logger.warning(f"{ticker} 피처 데이터 부족: {len(features)}일")
                return None
            # 훈련/검증 분할 기준점 (스케일러 누출 방지)
            scaler_train_size = int(len(features) * 0.8)
            # 훈련 데이터로만 스케일러 fit (데이터 누출 방지)
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            feature_scaler.fit(features[:scaler_train_size])
            scaled_features = feature_scaler.transform(features)
            # 종가 스케일러 (역변환용) — 훈련 데이터로만 fit
            close_prices = features[:, 0].reshape(-1, 1)
            close_scaler = MinMaxScaler(feature_range=(0, 1))
            close_scaler.fit(close_prices[:scaler_train_size])
            last_60_days = scaled_features[-60:].reshape((1, 60, features.shape[1]))
            model_file_keras = f"lstm_models/lstm_model_{ticker}.keras"
            model_file_h5 = f"lstm_models/lstm_model_{ticker}.h5"
            # .keras 우선, .h5 폴백
            if os.path.exists(model_file_keras):
                model_file = model_file_keras
            elif os.path.exists(model_file_h5):
                model_file = model_file_h5
            else:
                model_file = model_file_keras
            model_age_hours = (
                (time.time() - os.path.getmtime(model_file)) / 3600
                if os.path.exists(model_file) else float("inf")
            )
            loaded = False
            if model_age_hours < 24:
                try:
                    self.model = tf.keras.models.load_model(model_file, compile=False)
                    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
                    logger.info(f"{ticker} 기존 모델 로드 (경과: {model_age_hours:.1f}h)")
                    loaded = True
                except Exception as load_err:
                    logger.warning(f"{ticker} 모델 로드 실패 (재훈련): {load_err}")
                    os.remove(model_file)
            if not loaded:
                self.model = self._build_lstm_model(n_features=features.shape[1])
                X = []
                for i in range(60, len(scaled_features)):
                    X.append(scaled_features[i-60:i])
                X = np.array(X)
                if len(X) == 0:
                    logger.warning(f"{ticker} 유효한 시퀀스 없음")
                    return None
                y = scaled_features[60:, 0]  # 종가만 타겟
                train_size = int(len(X) * 0.8)
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]
                history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=0)
                logger.info(f"{ticker} LSTM 재훈련 완료 — 손실: {history.history['loss'][-1]:.6f}, 검증: {history.history['val_loss'][-1]:.6f}")
                save_path = model_file_keras
                self.model.save(save_path, include_optimizer=False)
                if os.path.exists(model_file_h5):
                    os.remove(model_file_h5)
                logger.info(f"{ticker} 모델 저장: {save_path}")
            predicted_scaled = self.model.predict(last_60_days, verbose=0)
            predicted_price = float(close_scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0])
            actual_price = feat_df['close'].iloc[-1]
            prediction_error = abs(predicted_price - actual_price) / actual_price * 100
            direction = "상승" if predicted_price > actual_price else "하락"
            logger.info(
                f"{ticker} LSTM 예측: {predicted_price:,.2f}원, 실제가: {actual_price:,.2f}원, "
                f"오차: {prediction_error:.2f}%, 방향: {direction}"
            )
            return predicted_price
        except Exception as e:
            logger.error(f"{ticker} 가격 예측 실패: {e}")
            return None

    def add_technical_indicators(self, df):
        try:
            df = df.copy()
            min_length = 26
            if len(df) < min_length:
                # 신규 상장 코인 등에서 발생 — 호출자가 fallback 처리하거나 지표 없이 진행
                logger.debug(f"데이터 길이 {len(df)}가 최소 요구 길이 {min_length}보다 짧음 (지표 계산 생략)")
                return df
            indicator_bb = BollingerBands(close=df['close'])
            df['bb_bbm'] = indicator_bb.bollinger_mavg()
            df['bb_bbh'] = indicator_bb.bollinger_hband()
            df['bb_bbl'] = indicator_bb.bollinger_lband()
            df['rsi'] = RSIIndicator(close=df['close']).rsi()
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
            df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
            df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
            logger.debug("기술적 지표 추가 완료")
            return df
        except Exception as e:
            logger.error(f"기술적 지표 추가 실패: {e}")
            return df

    def analyze_order_book(self, ticker):
        while not rate_limiter.can_make_api_call():
            time.sleep(0.1)
        try:
            orderbook = pyupbit.get_orderbook(ticker)
            if not orderbook or not isinstance(orderbook, (dict, list)):
                logger.warning(f"{ticker} 오더북 데이터 없음 (응답: {type(orderbook).__name__})")
                return None
            # 단일 티커는 dict, 복수는 list[dict] — pyupbit 반환 형식 통일
            if isinstance(orderbook, list):
                ob = orderbook[0] if orderbook else None
            else:
                ob = orderbook
            if not ob or not isinstance(ob, dict) or not ob.get('orderbook_units'):
                logger.warning(f"{ticker} 오더북 데이터 없음")
                return None
            bids = ob['orderbook_units']
            current_price = safe_get_current_price(ticker)
            if not current_price:
                logger.error(f"{ticker} 현재 가격 가져오기 실패")
                return None
            buy_volume_analysis = {}
            for bid in bids:
                price = float(bid['bid_price'])
                volume = float(bid['bid_size'])
                price_diff_percent = ((current_price - price) / current_price) * 100
                if 1 <= price_diff_percent <= 3:
                    buy_volume_analysis[price] = volume
            if buy_volume_analysis:
                optimal_buy_price = max(buy_volume_analysis, key=buy_volume_analysis.get)
                logger.info(f"{ticker} 최적 매수가격: {optimal_buy_price:,.2f}원")
                return {
                    'current_price': float(current_price),
                    'optimal_buy_price': float(optimal_buy_price),
                    'buy_volume': float(buy_volume_analysis[optimal_buy_price])
                }
            logger.warning(f"{ticker} 적절한 매수 구간 없음")
            return None
        except Exception as e:
            logger.error(f"{ticker} 오더북 분석 실패: {e}")
            return None

    @cache_result(expiry_seconds=CACHE_EXPIRY_SECONDS['market_trend'])
    def analyze_market_trend(self):
        while not rate_limiter.can_make_api_call():
            time.sleep(0.1)
        try:
            top_coins = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
            trend_data = {}
            for coin in top_coins:
                df = pyupbit.get_ohlcv(coin, interval="day", count=7)
                if df is not None and not df.empty:
                    trend_data[coin] = {
                        "price_change": float(((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100)),
                        "volume_change": float(((df['volume'].iloc[-1] - df['volume'].iloc[0]) / df['volume'].iloc[0] * 100))
                    }
            logger.info("시장 트렌드 분석 완료")
            return trend_data
        except Exception as e:
            logger.error(f"시장 트렌드 분석 실패: {e}")
            return {}

    @cache_result(expiry_seconds=CACHE_EXPIRY_SECONDS['market_trend'])
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
    def ai_analyze_market(self):
        while not rate_limiter.can_make_api_call():
            time.sleep(0.1)
        try:
            fear_greed = get_fear_and_greed()
            news_data = collect_news_data()
            market_data = {
                "fear_greed_index": fear_greed,
                "market_trend": self.analyze_market_trend(),
                "news_data": {
                    "google_news": news_data["google_news"],
                    "total_articles": news_data["total_articles"],
                },
                "timestamp": datetime.now().isoformat()
            }
            system_prompt = """당신은 암호화폐 시장 분석가입니다. 주어진 시장 데이터와 뉴스 데이터를 종합 분석하고, 전반적인 매매 전략을 아래 JSON 형식으로 반환하세요. 반드시 JSON 형식으로만 응답하며, 추가 텍스트는 포함시키지 마세요.

분석 시 다음을 반드시 고려하세요:
1. Fear & Greed Index와 가격/거래량 추세 (기존 지표)
2. Google News 뉴스 제목을 분석하여 시장 감성 판단
3. 규제/정책 관련 뉴스 (SEC, 각국 정부 규제, 법안 등)
4. 해킹/보안 사고 뉴스
5. ETF/기관투자 관련 뉴스
6. 거시경제 뉴스 (금리, 인플레이션, 주요 경제지표)

뉴스에서 시장에 큰 영향을 줄 수 있는 이벤트가 감지되면 risk_level에 적극 반영하세요.
- 부정적 뉴스(규제 강화, 해킹, 대규모 매도 등): risk_level 상향
- 긍정적 뉴스(ETF 승인, 기관 매수, 규제 완화 등): risk_level 하향

예시:
{"risk_level": 7, "buy_ratio": "30%", "sell_ratio": "50%", "news_sentiment": "negative", "news_impact": "high", "key_indicators": ["거래량 증가", "RSI 과매도", "SEC 규제 강화 뉴스", "거래소 해킹 보도"]}

규칙:
- risk_level: 1~10 사이의 정수
- buy_ratio/sell_ratio: 퍼센트 문자열
- news_sentiment: "positive", "negative", "neutral" 중 하나
- news_impact: "high", "medium", "low" 중 하나
- key_indicators: 문자열 배열 (뉴스 기반 지표도 포함)"""
            user_content = json.dumps(market_data, default=serialize_numpy)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            result = None
            for attempt in range(2):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        response_format={"type": "json_object"},
                        messages=messages
                    )
                    raw = response.choices[0].message.content
                    logger.info(f"OpenAI 원시 응답: {raw}")
                    result = json.loads(raw)
                    break
                except json.JSONDecodeError:
                    if attempt == 0:
                        logger.warning(f"OpenAI 응답 JSON 파싱 실패 (재시도 1/1), 응답: {response.choices[0].message.content}")
                        messages.append({"role": "assistant", "content": response.choices[0].message.content})
                        messages.append({"role": "user", "content": "위 응답이 올바른 JSON이 아닙니다. 반드시 JSON 형식으로만 다시 응답해주세요."})
                    else:
                        logger.error(f"OpenAI 응답 JSON 파싱 최종 실패: 응답: {response.choices[0].message.content}")
                        return None
                except Exception as api_err:
                    raise
            result = validate_market_analysis(result)
            logger.info(f"AI 시장 분석 완료 (뉴스 {news_data['total_articles']}건 반영)")
            return result
        except ValueError as e:
            logger.error(f"AI 시장 분석 응답 검증 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"AI 시장 분석 실패: {e}")
            return None

    @cache_result(expiry_seconds=CACHE_EXPIRY_SECONDS['opportunities'])
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
    async def scan_for_opportunities(self):
        while not rate_limiter.can_make_api_call():
            await asyncio.sleep(0.1)
        try:
            for attempt in range(3):
                try:
                    tickers = pyupbit.get_tickers(fiat="KRW")
                    tickers = [t for t in tickers if t != "KRW-KRW" and t.startswith("KRW-")]
                    break
                except Exception as e:
                    logger.warning(f"티커 가져오기 실패, 시도 {attempt + 1}/3: {e}")
                    if attempt == 2:
                        raise e
                    await asyncio.sleep(2)
            logger.info(f"{len(tickers)}개 티커 데이터 가져오는 중...")
            market_data = {}
            sample_tickers = tickers

            def _fetch_ohlcv_with_fallback(ticker):
                # 일봉 우선, 26개 미만(신규 상장 등)이면 4시간봉으로 대체해 지표 계산 가능하도록 함
                df = pyupbit.get_ohlcv(ticker, interval="day", count=30)
                if df is None or df.empty or len(df) < 26:
                    df_h4 = pyupbit.get_ohlcv(ticker, interval="minute240", count=180)
                    if df_h4 is not None and not df_h4.empty and len(df_h4) >= 26:
                        return ticker, df_h4
                return ticker, df

            with ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(_fetch_ohlcv_with_fallback, sample_tickers))
                for ticker, df in results:
                    if df is not None and not df.empty:
                        coin_data = {
                            "price_change": float(((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100)) if len(df) > 1 else 0,
                            "volume_change": float(((df['volume'].iloc[-1] - df['volume'].iloc[0]) / df['volume'].iloc[0] * 100)) if len(df) > 1 else 0,
                            "current_price": float(df['close'].iloc[-1])
                        }
                        # 기술적 지표 추가 (OpenAI가 정확한 분석 가능하도록)
                        try:
                            df_ind = self.add_technical_indicators(df)
                            if 'rsi' in df_ind.columns:
                                coin_data["rsi"] = round(float(df_ind['rsi'].iloc[-1]), 2) if not pd.isna(df_ind['rsi'].iloc[-1]) else None
                            if 'macd' in df_ind.columns and 'macd_signal' in df_ind.columns:
                                coin_data["macd"] = round(float(df_ind['macd'].iloc[-1]), 4) if not pd.isna(df_ind['macd'].iloc[-1]) else None
                                coin_data["macd_signal"] = round(float(df_ind['macd_signal'].iloc[-1]), 4) if not pd.isna(df_ind['macd_signal'].iloc[-1]) else None
                            if 'bb_bbl' in df_ind.columns and 'bb_bbh' in df_ind.columns:
                                coin_data["bb_lower"] = round(float(df_ind['bb_bbl'].iloc[-1]), 2) if not pd.isna(df_ind['bb_bbl'].iloc[-1]) else None
                                coin_data["bb_upper"] = round(float(df_ind['bb_bbh'].iloc[-1]), 2) if not pd.isna(df_ind['bb_bbh'].iloc[-1]) else None
                        except Exception:
                            pass
                        market_data[ticker] = coin_data
            fear_greed = get_fear_and_greed()
            market_summary = {
                "fear_greed_index": fear_greed,
                "tickers": market_data,
                "timestamp": datetime.now().isoformat()
            }
            system_prompt = """당신은 암호화폐 시장 분석가입니다. 주어진 시장 데이터(RSI, MACD, 볼린저밴드 등 기술적 지표 포함)를 분석하여 가격 상승 전 조기 매수 기회를 추천하세요.
                        우선 조건: RSI 30~40, 현재가가 볼린저밴드 하단(bb_lower) 근처, MACD가 시그널선 위로 교차 직전(macd > macd_signal 직전), 최근 상승률 20% 미만.
                        각 코인의 실제 지표 수치를 reason에 반드시 포함하여 구체적으로 분석하세요. 동일한 이유를 복붙하지 마세요.
                        반드시 아래 JSON 형식으로만 응답하며, 추가 텍스트는 포함시키지 마세요:
                        {
                          "recommendations": [
                            {"ticker": "코인 심볼", "score": 0-100 사이 값, "reason": "해당 코인의 실제 지표 수치 기반 추천 이유"},
                            ...
                          ]
                        }
                        score는 숫자 형식이어야 합니다. 최대 5개의 코인을 추천하세요. reason은 반드시 한국어로 작성하세요."""
            user_content = json.dumps(market_summary, default=serialize_numpy)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            ai_recommendations = None
            for attempt in range(2):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        response_format={"type": "json_object"},
                        messages=messages
                    )
                    raw = response.choices[0].message.content
                    logger.info(f"OpenAI 원시 응답: {raw}")
                    ai_recommendations = json.loads(raw)
                    break
                except json.JSONDecodeError:
                    if attempt == 0:
                        logger.warning(f"OpenAI 응답 JSON 파싱 실패 (재시도 1/1), 응답: {response.choices[0].message.content}")
                        messages.append({"role": "assistant", "content": response.choices[0].message.content})
                        messages.append({"role": "user", "content": "위 응답이 올바른 JSON이 아닙니다. 반드시 JSON 형식으로만 다시 응답해주세요."})
                    else:
                        logger.error(f"OpenAI 응답 JSON 파싱 최종 실패: 응답: {response.choices[0].message.content}")
                        return []
                except Exception as api_err:
                    raise
            if ai_recommendations is None:
                return []
            opportunities = validate_opportunities(ai_recommendations)
            logger.info(f"AI가 {len(opportunities)}개의 거래 기회 발견")
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            return opportunities[:5]
        except ValueError as e:
            logger.error(f"AI 거래 기회 응답 검증 실패: {e}")
            return []
        except Exception as e:
            logger.error(f"거래 기회 탐색 실패: {e}")
            return []

    @cache_result(expiry_seconds=CACHE_EXPIRY_SECONDS['default'])
    def evaluate_coin(self, ticker):
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=30)
            if df is None or df.empty or len(df) < 14:
                # 일봉 부족 시 4시간봉으로 대체 (6개 = 1일 → 180개 ≈ 30일)
                df = pyupbit.get_ohlcv(ticker, interval="minute240", count=180)
                if df is None or df.empty or len(df) < 14:
                    logger.warning(f"{ticker} 데이터 부족: 일봉/4시간봉 모두 부족")
                    return None
                logger.info(f"{ticker} 일봉 부족 → 4시간봉({len(df)}개)으로 대체")
            df = self.add_technical_indicators(df)
            if len(df) < 26:
                logger.warning(f"{ticker} 데이터 길이 {len(df)}가 분석에 부족")
                return None
            score = 0
            volatility = self.get_volatility_factor(ticker)
            last_rsi = df['rsi'].iloc[-1] if 'rsi' in df else float('nan')
            if pd.notna(last_rsi):
                if 30 <= last_rsi <= 40:
                    score += 40 / volatility
                elif last_rsi < 30:
                    score += 20 / volatility
                elif last_rsi > 70:
                    score -= 20
            if 'macd_diff' in df and len(df['macd_diff']) >= 2:
                if df['macd_diff'].iloc[-2] < 0 and df['macd_diff'].iloc[-1] > 0:
                    score += 20
                elif df['macd_diff'].iloc[-1] < 0 and df['macd_diff'].iloc[-1] > df['macd_diff'].iloc[-2]:
                    score += 15
            if 'bb_bbl' in df and df['close'].iloc[-1] < df['bb_bbm'].iloc[-1]:
                score += 25
            if 'volume_sma' in df and df['volume'].iloc[-1] > df['volume_sma'].iloc[-1] * 1.5:
                score += 10
            df_7d = pyupbit.get_ohlcv(ticker, interval="day", count=7)
            if df_7d is not None and len(df_7d) >= 2:
                price_change = ((df_7d['close'].iloc[-1] - df_7d['close'].iloc[0]) / df_7d['close'].iloc[0]) * 100
                if price_change < -10:
                    logger.info(f"{ticker} 급격한 하락으로 매수 불가")
                    return None
                if price_change < 20:
                    score += 15
            df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
            if 'ema_9' in df and df['ema_9'].iloc[-1] > df['ema_20'].iloc[-1]:
                score += 10
            else:
                score -= 10
            if score > 30:
                result = {
                    'ticker': ticker,
                    'score': score,
                    'price': float(df['close'].iloc[-1]),
                    'rsi': float(last_rsi),
                    'volume_ratio': float(df['volume'].iloc[-1] / df['volume_sma'].iloc[-1]) if 'volume_sma' in df and not pd.isna(df['volume_sma'].iloc[-1]) else 0,
                    'price_change_7d': float(price_change) if 'price_change' in locals() else 0
                }
                logger.info(f"{ticker} 평가 완료: 점수={score}, RSI={last_rsi:.2f}, 7일 상승률={result['price_change_7d']:.2f}%")
                return result
            return None
        except Exception as e:
            logger.error(f"{ticker} 코인 평가 실패: {e}")
            return None

    def test_lstm_prediction(self, ticker):
        predicted_price = self.predict_next_price(ticker)
        if predicted_price:
            logger.info(f"{ticker} 테스트 예측 완료: {predicted_price:,.2f}원")
            return predicted_price
        return None

    def monitor_prediction_accuracy(self, ticker):
        df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
        if df is None or len(df) < 2:
            logger.warning(f"{ticker} 모니터링 데이터 부족")
            return
        actual_price = df['close'].iloc[-1]
        predicted_price = self.predict_next_price(ticker)
        if predicted_price:
            error = abs(predicted_price - actual_price) / actual_price * 100
            logger.info(f"{ticker} 예측 정확도: 실제가={actual_price:,.2f}원, 예측가={predicted_price:,.2f}원, 오차={error:.2f}%")

    def backtest_lstm(self, ticker, days=30):
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=days + 90)
            if df is None or len(df) < 90:
                logger.warning(f"{ticker} 백테스팅 데이터 부족: {len(df) if df is not None else 0}일")
                return None
            features, feat_df = self._prepare_features(df)
            if len(features) < 60:
                logger.warning(f"{ticker} 백테스팅 피처 부족")
                return None
            errors = []
            for i in range(60, len(features) - 1):
                temp_features = features[:i]
                feature_scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = feature_scaler.fit_transform(temp_features)
                close_scaler = MinMaxScaler(feature_range=(0, 1))
                close_scaler.fit_transform(temp_features[:, 0].reshape(-1, 1))
                X = [scaled[j-60:j] for j in range(60, len(scaled))]
                X = np.array(X)
                y = scaled[60:, 0]
                self.model = self._build_lstm_model(n_features=features.shape[1])
                self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                last_60 = scaled[-60:].reshape((1, 60, features.shape[1]))
                predicted_scaled = self.model.predict(last_60, verbose=0)
                predicted_price = close_scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0]
                actual_price = features[i, 0]
                error = abs(predicted_price - actual_price) / actual_price * 100
                errors.append(error)
                logger.info(f"{ticker} 백테스팅 [{i-59}/{len(features)-60}]: 예측가={predicted_price:,.2f}원, 실제가={actual_price:,.2f}원, 오차={error:.2f}%")
            avg_error = sum(errors) / len(errors) if errors else 0
            logger.info(f"{ticker} 백테스팅 완료: 평균 예측 오차={avg_error:.2f}%")
            return avg_error
        except Exception as e:
            logger.error(f"{ticker} 백테스팅 실패: {e}")
            return None

    def get_volatility_factor(self, ticker):
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=30)
            if df is None or len(df) < 14:
                return 1.0
            df = self.add_technical_indicators(df)
            atr = df['atr'].iloc[-1]
            avg_price = df['close'].mean()
            volatility = atr / avg_price if avg_price > 0 else 0
            raw = min(1.5, max(0.5, volatility * 10))
            return 2.0 - raw  # 반전: 변동성 높을수록 비율 감소
        except Exception as e:
            logger.error(f"{ticker} 변동성 계산 실패: {e}")
            return 1.0
        
    def ai_sell_decision(self, ticker, current_price, profit_pct, signals):
        """LSTM+지표 시그널 2개 이상일 때 GPT-4 최종 매도 판단. (True, reason) 반환."""
        try:
            prompt = {
                "ticker": ticker,
                "current_price": current_price,
                "profit_percent": round(profit_pct, 2),
                "bearish_signals": signals,
            }
            system_content = (
                "당신은 암호화폐 트레이딩 전문가입니다. "
                "보유 종목의 기술적 지표와 현재 수익률을 보고 지금 매도해야 하는지 판단하세요. "
                "반드시 아래 JSON 형식으로만 응답하세요:\n"
                '{"sell": true/false, "reason": "판단 이유 (한국어)"}'
            )
            user_content = json.dumps(prompt, ensure_ascii=False, default=serialize_numpy)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            for attempt in range(2):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        response_format={"type": "json_object"},
                        messages=messages
                    )
                    raw = response.choices[0].message.content
                    result = json.loads(raw)
                    should_sell = bool(result.get("sell", False))
                    reason = result.get("reason", "")
                    logger.info(f"{ticker} GPT-4 매도 판단: {'매도' if should_sell else '보유'} — {reason}")
                    return should_sell, reason
                except json.JSONDecodeError:
                    if attempt == 0:
                        logger.warning(f"{ticker} GPT-4 매도 판단 JSON 파싱 실패, 재시도")
                        messages.append({"role": "assistant", "content": response.choices[0].message.content})
                        messages.append({"role": "user", "content": "위 응답이 올바른 JSON이 아닙니다. 반드시 JSON 형식으로만 다시 응답해주세요."})
                    else:
                        logger.error(f"{ticker} GPT-4 매도 판단 JSON 파싱 최종 실패")
                        return False, ""
                except Exception as api_err:
                    raise
            return False, ""
        except Exception as e:
            logger.error(f"{ticker} GPT-4 매도 판단 실패: {e}")
            return False, ""

    def retrain_on_loss(self, ticker, trade):
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=365)
            if df is None or len(df) < 90:
                logger.warning(f"{ticker} 재학습 데이터 부족")
                return
            features, feat_df = self._prepare_features(df)
            if len(features) < 60:
                logger.warning(f"{ticker} 재학습 피처 데이터 부족")
                return
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = feature_scaler.fit_transform(features)
            X = [scaled_features[i-60:i] for i in range(60, len(scaled_features))]
            X = np.array(X)
            y = scaled_features[60:, 0]
            trade_time = datetime.fromisoformat(trade['timestamp'])
            df_index = feat_df.index.get_indexer([trade_time], method='nearest')[0]
            if df_index >= 60:
                X_new = scaled_features[df_index-60:df_index].reshape((1, 60, features.shape[1]))
                y_new = scaled_features[df_index, 0]
                self.model.fit(X_new, np.array([y_new]), epochs=5, verbose=0)
                logger.info(f"{ticker} 손실 데이터로 재학습 완료")
                self.model.save(f"lstm_models/lstm_model_{ticker}.keras", include_optimizer=False)
        except Exception as e:
            logger.error(f"{ticker} 손실 재학습 실패: {e}")

class TradingBot:
    def __init__(self):
        self.trading_history = TradingHistory()
        self.stop_loss_manager = StopLossManager()
        self.market_analyzer = MarketAnalyzer()
        self.purchased_coins = FileManager.load_json(COINS_FILE, {"coins": [], "last_analysis": {}})
        self.last_opportunity_scan = 0
        self.opportunities = []
        self.buy_ratios = {
            'BTC': 0.5,
            'ETH': 0.5,
            'ALT': 0.4,
            'HIGH_CONFIDENCE': 0.7
        }
        self.last_portfolio_error_time = 0
        self._sell_lock = asyncio.Lock()
        self._buy_lock = asyncio.Lock()
        self.sync_with_upbit_account()
        logger.info("트레이딩 봇 초기화 완료")

    async def _get_actual_sell_price(self, order_uuid, fallback_price):
        """시장가 매도 주문의 실제 체결 평균가를 조회"""
        try:
            for _ in range(5):
                await asyncio.sleep(0.5)
                order_info = upbit.get_order(order_uuid)
                if order_info and order_info.get("trades"):
                    total_funds = sum(float(t["funds"]) for t in order_info["trades"])
                    total_volume = sum(float(t["volume"]) for t in order_info["trades"])
                    if total_volume > 0:
                        actual_price = total_funds / total_volume
                        logger.info(f"실제 체결가: {actual_price:,.2f}원 (호가: {fallback_price:,.2f}원)")
                        return actual_price
            logger.warning(f"체결 정보 조회 실패, 호가 사용: {fallback_price:,.2f}원")
            return fallback_price
        except Exception as e:
            logger.error(f"체결가 조회 실패: {e}")
            return fallback_price

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def sync_with_upbit_account(self):
        try:
            balances = upbit.get_balances()
            if not balances:
                logger.error("Upbit 계정 정보를 가져오지 못했습니다.")
                return
            current_coins = set()
            for balance in balances:
                currency = balance['currency']
                unit_currency = balance.get('unit_currency', 'KRW')
                ticker = f"{unit_currency}-{currency}"
                total_amount = float(balance['balance']) + float(balance.get('locked', 0))
                if total_amount > 0 and ticker.startswith('KRW-') and ticker != 'KRW-KRW':
                    current_coins.add(ticker)
                    if ticker not in self.purchased_coins["coins"]:
                        self.purchased_coins["coins"].append(ticker)
                        logger.info(f"{ticker} 코인을 Upbit 계정에서 발견하여 추가: {total_amount:.8f}개")
                    # 항상 업비트 실제 평균 매수가로 동기화 (DCA 후 불일치 방지)
                    avg_price = float(balance.get('avg_buy_price', 0))
                    if avg_price == 0:
                        current_price = safe_get_current_price(ticker)
                        avg_price = current_price if current_price else 0
                    if ticker not in self.stop_loss_manager.positions:
                        self.stop_loss_manager.add_position(ticker, avg_price, total_amount)
                        logger.info(f"{ticker} 포지션 추가: 매수가={avg_price:,.2f}원, 수량={total_amount:.8f}개")
                    else:
                        self.stop_loss_manager.update_position(ticker, new_avg_price=avg_price, amount=total_amount)
                        logger.info(f"{ticker} 포지션 동기화: 평균가={avg_price:,.2f}원, 수량={total_amount:.8f}개")
            # cleanup: purchased_coins와 positions 양쪽을 모두 검사 (불일치한 stale 데이터까지 제거)
            tracked_tickers = set(self.purchased_coins["coins"]) | set(self.stop_loss_manager.positions.keys())
            for ticker in tracked_tickers:
                if ticker not in current_coins:
                    if ticker in self.purchased_coins["coins"]:
                        self.purchased_coins["coins"].remove(ticker)
                    if ticker in self.stop_loss_manager.positions:
                        self.stop_loss_manager.remove_position(ticker)
                    logger.info(f"{ticker} 코인을 Upbit 계정에서 제거")
            FileManager.save_json(COINS_FILE, self.purchased_coins)
            FileManager.save_json(POSITIONS_FILE, self.stop_loss_manager.positions)
            logger.info("Upbit 계정 동기화 완료")
        except Exception as e:
            logger.error(f"Upbit 계정 동기화 실패: {e}", exc_info=True)
            raise

    async def find_trading_opportunities(self):
        self.opportunities = await self.market_analyzer.scan_for_opportunities()
        logger.info(f"{len(self.opportunities)}개의 조기 매수 기회 발견")

    async def execute_additional_purchase(self, ticker):
        try:
            if ticker not in self.purchased_coins["coins"]:
                logger.info(f"{ticker} 보유 중이 아니므로 추가 매수 불가")
                return False

            evaluation = self.market_analyzer.evaluate_coin(ticker)
            if not evaluation:
                logger.warning(f"{ticker} 추가매수 점수 미달")
                return False

            score = evaluation["score"]
            current_price = evaluation["price"]
            price_change_7d = evaluation["price_change_7d"]
            position = self.stop_loss_manager.positions.get(ticker)
            if not position or not current_price:
                logger.warning(f"{ticker} 포지션 또는 가격 정보 없음")
                return False

            profit_percent = ((current_price * (1 - TRADING_FEE) - position["entry_price"] * (1 + TRADING_FEE)) / (position["entry_price"] * (1 + TRADING_FEE))) * 100
            krw_balance = upbit.get_balance("KRW")
            if krw_balance is None or krw_balance < MIN_KRW_BALANCE:
                logger.info(f"KRW 잔고 부족: {krw_balance if krw_balance else '없음'}원")
                return False

            budget = None
            reason = None
            volatility_factor = self.market_analyzer.get_volatility_factor(ticker)
            
            if score >= 75 and price_change_7d > 0:
                buy_ratio = self.buy_ratios["HIGH_CONFIDENCE"] * volatility_factor
                budget = krw_balance * buy_ratio
                reason = f"불타기: 점수 {score}, 7일 상승률 {price_change_7d:.2f}%"
                logger.info(f"{ticker} 불타기 조건 충족: 점수={score}, 예산={budget:,.2f}원")
            elif 50 <= score < 75 and profit_percent <= -3 * volatility_factor:
                if profit_percent < -10:
                    logger.info(f"{ticker} 손실률 {profit_percent:.2f}%로 물타기 중단")
                    return False
                buy_ratio = DCA_RATIO * volatility_factor
                budget = krw_balance * buy_ratio
                reason = f"물타기: 점수 {score}, 손실률 {profit_percent:.2f}%"
                logger.info(f"{ticker} 물타기 조건 충족: 점수={score}, 예산={budget:,.2f}원")
            else:
                logger.info(f"{ticker} 추가 매수 조건 미충족: 점수={score}, 손실률={profit_percent:.2f}%")
                return False

            while not rate_limiter.can_make_order():
                await asyncio.sleep(0.1)

            buy_info = self.market_analyzer.analyze_order_book(ticker)
            optimal_price = buy_info['optimal_buy_price'] if buy_info and 'optimal_buy_price' in buy_info else current_price
            amount = budget / optimal_price
            buy_order = upbit.buy_limit_order(ticker, optimal_price, amount)
            if buy_order and "uuid" in buy_order:
                order_uuid = buy_order["uuid"]
                order_info = None
                for _ in range(10):
                    order_info = upbit.get_order(order_uuid)
                    if order_info and order_info["state"] in ["done", "cancel"]:
                        break
                    await asyncio.sleep(1)
                if order_info is None:
                    logger.error(f"{ticker} 주문 정보 가져오기 실패: UUID={order_uuid}")
                    upbit.cancel_order(order_uuid)
                    logger.info(f"{ticker} 주문 취소: UUID={order_uuid}")
                    return False
                executed_amount = float(order_info.get("executed_volume", 0))
                if executed_amount > 0 and order_info["state"] == "done":
                    executed_price = await resolve_executed_price(self.trading_history, ticker, "buy", order_info, executed_amount, optimal_price, reason)
                    total_amount = position["amount"] + executed_amount
                    total_cost = (position["amount"] * position["entry_price"]) + (executed_amount * executed_price)
                    new_avg_price = total_cost / total_amount if total_amount > 0 else executed_price
                    self.stop_loss_manager.update_position(ticker, new_avg_price=new_avg_price, amount=total_amount)
                    logger.info(f"{ticker} 추가 매수 완료: 가격={executed_price:,.2f}원, 수량={executed_amount:.8f}개")
                    message = (
                        f"🔥 추가 매수 실행: {ticker}\n"
                        f"가격: {executed_price:,.2f}원\n"
                        f"수량: {executed_amount:.8f}개\n"
                        f"총 금액: {executed_amount * executed_price:,.2f}원\n"
                        f"사유: {reason}"
                    )
                    await async_send_telegram_message(message)
                    self.sync_with_upbit_account()
                    return True
                else:
                    logger.warning(f"{ticker} 추가 매수 미체결: 상태={order_info['state']}")
                    upbit.cancel_order(order_uuid)
                    logger.info(f"{ticker} 미체결 주문 취소: UUID={order_uuid}")
                    message = (
                        f"⚠️ 추가 매수 미체결 및 취소: {ticker}\n"
                        f"주문 UUID: {order_uuid}\n"
                        f"상태: {order_info['state']}"
                    )
                    await async_send_telegram_message(message)
                    return False
            else:
                logger.error(f"{ticker} 추가 매수 주문 실패: {buy_order or '응답 없음'}")
                return False
        except Exception as e:
            logger.error(f"{ticker} 추가 매수 실행 실패: {e}")
            return False

    async def execute_trading_strategy(self):
        try:
            if len(self.purchased_coins["coins"]) >= MAX_COINS:
                logger.info("최대 5개 코인 보유 중, 신규 매수 건너뜀")
                return
            krw_balance = upbit.get_balance("KRW")
            if krw_balance is None or krw_balance < MIN_KRW_BALANCE:
                logger.info(f"KRW 잔고 부족: {krw_balance if krw_balance else '없음'}원")
                return
            market_analysis = self.market_analyzer.ai_analyze_market()
            if market_analysis is None:
                logger.info("시장 분석 실패, 거래 중단")
                return
            try:
                risk_level = int(market_analysis.get("risk_level", 5))
            except (ValueError, TypeError) as e:
                logger.error(f"위험 수준 형식 오류: {market_analysis.get('risk_level')}, 기본값 5 사용")
                risk_level = 5
            logger.info(f"시장 위험 수준: {risk_level}, KRW 잔고: {krw_balance:,.2f}원")
            if risk_level >= MARKET_RISK_HIGH_THRESHOLD:
                logger.info(
                    f"시장 위험으로 신규 매수 중단: 위험 수준 {risk_level} >= {MARKET_RISK_HIGH_THRESHOLD}"
                )
                return
            # 위험 수준별 투자 비율 조절 (threshold 미만 구간에서 단계적 축소)
            if risk_level >= MARKET_RISK_HIGH_THRESHOLD - 1:
                risk_ratio = 0.5
            elif risk_level >= MARKET_RISK_HIGH_THRESHOLD - 2:
                risk_ratio = 0.7
            else:
                risk_ratio = 1.0
            if risk_ratio < 1.0:
                logger.info(f"시장 위험 수준 {risk_level}: 투자 비율 {risk_ratio:.0%}로 축소")
            logger.info(f"{len(self.opportunities)}개의 거래 기회 처리 중")
            max_opportunities = min(2, len(self.opportunities))
            # 업비트 실잔고 기준 보유 코인 목록 (사이클 간 중복 매수 방지)
            try:
                live_balances = upbit.get_balances() or []
                live_holdings = {
                    f"KRW-{b['currency']}"
                    for b in live_balances
                    if b.get('currency') != 'KRW' and float(b.get('balance', 0)) + float(b.get('locked', 0)) > 0
                }
            except Exception:
                live_holdings = set(self.purchased_coins["coins"])
            for opp in self.opportunities[:max_opportunities]:
                ticker = opp["ticker"]
                if ticker in live_holdings:
                    logger.info(f"{ticker} 업비트 실잔고 보유 중, 매수 건너뜀")
                    continue
                if self.trading_history.is_blacklisted(ticker):
                    logger.info(f"{ticker} 블랙리스트로 인해 매수 건너뜀: 점수={opp['score']}, 이유={opp['reason']}")
                    continue

                coin_type = 'BTC' if ticker == 'KRW-BTC' else 'ETH' if ticker == 'KRW-ETH' else 'ALT'
                buy_ratio = self.buy_ratios.get(coin_type, self.buy_ratios['ALT'])
                if opp['score'] >= 85:
                    buy_ratio = self.buy_ratios['HIGH_CONFIDENCE']
                volatility_factor = self.market_analyzer.get_volatility_factor(ticker)
                buy_ratio *= volatility_factor
                predicted_price = self.market_analyzer.predict_next_price(ticker)
                current_price = safe_get_current_price(ticker)
                buy_info = self.market_analyzer.analyze_order_book(ticker)
                if not buy_info or 'optimal_buy_price' not in buy_info:
                    logger.warning(f"{ticker} 최적 매수가격 계산 실패, 현재가 사용")
                    if not current_price:
                        logger.error(f"{ticker} 현재 가격 없음, 건너뜀")
                        continue
                    buy_info = {
                        'current_price': float(current_price),
                        'optimal_buy_price': float(current_price),
                        'buy_volume': 0
                    }
                budget = krw_balance * buy_ratio * risk_ratio / max_opportunities
                optimal_price = buy_info['optimal_buy_price']
                try:
                    score = int(opp['score'])
                except (ValueError, TypeError) as e:
                    logger.error(f"{ticker} 점수 형식 오류: {opp['score']}, 건너뜀")
                    continue
                logger.info(f"{ticker} 평가: 예측가={predicted_price or '없음'}, 현재가={current_price:,.2f}원, 점수={score}, 최적가={optimal_price:,.2f}원, 투자비율={buy_ratio:.2%}")
                # LSTM은 데이터 수집용으로만 사용 (매수 판단에 미반영)
                # lstm_bullish = bool(predicted_price and current_price and predicted_price > current_price * 1.01)
                # 점수 85이상만 진입
                if len(self.purchased_coins["coins"]) >= MAX_COINS:
                    logger.info(f"최대 {MAX_COINS}개 보유 도달, 추가 매수 중단")
                    break
                if score >= 85:
                    reason = f"AI 조기 매수 추천 (점수: {score}, 예측가: {predicted_price or '없음'})"
                    await self.execute_buy(ticker, reason, budget=budget)
                else:
                    logger.info(f"{ticker} 건너뜀: 점수 미달({score} < 85)")
                krw_balance = upbit.get_balance("KRW")
                if krw_balance is None or krw_balance < MIN_KRW_BALANCE:
                    logger.info(f"{ticker} 처리 후 KRW 잔고 부족: {krw_balance if krw_balance else '없음'}원")
                    break
        except Exception as e:
            logger.error(f"매수 전략 실행 실패: {e}", exc_info=True)

    async def _check_bearish_signals(self, ticker, current_price):
        """기술적 지표 기반 매도 시그널 1차 필터. 감지된 시그널 목록 반환."""
        signals = []
        try:
            # LSTM은 데이터 수집용으로만 사용 (매도 판단에 미반영)
            predicted_price = self.market_analyzer.predict_next_price(ticker)
            # if predicted_price and predicted_price < current_price * 0.99:
            #     signals.append(f"LSTM하락예측({predicted_price:,.0f}원<현재가)")

            df = pyupbit.get_ohlcv(ticker, interval="day", count=30)
            if df is not None and len(df) >= 26:
                df = self.market_analyzer.add_technical_indicators(df)

                last_rsi = df['rsi'].iloc[-1] if 'rsi' in df and not pd.isna(df['rsi'].iloc[-1]) else None
                if last_rsi is not None and last_rsi > 65:
                    signals.append(f"RSI과매수({last_rsi:.1f})")

                if 'macd_diff' in df and len(df['macd_diff'].dropna()) >= 2:
                    prev_diff = df['macd_diff'].iloc[-2]
                    last_diff = df['macd_diff'].iloc[-1]
                    if not pd.isna(prev_diff) and not pd.isna(last_diff):
                        if prev_diff > 0 and last_diff < 0:
                            signals.append("MACD데스크로스")
                        elif last_diff < 0 and last_diff < prev_diff:
                            signals.append("MACD음전환가속")

                if 'bb_bbh' in df and not pd.isna(df['bb_bbh'].iloc[-1]):
                    if current_price > df['bb_bbh'].iloc[-1]:
                        signals.append("BB상단돌파")

                if 'ema_9' in df and 'ema_20' in df and len(df) >= 2:
                    prev_ema9 = df['ema_9'].iloc[-2]
                    prev_ema20 = df['ema_20'].iloc[-2]
                    last_ema9 = df['ema_9'].iloc[-1]
                    last_ema20 = df['ema_20'].iloc[-1]
                    if not pd.isna(prev_ema9) and not pd.isna(prev_ema20) and not pd.isna(last_ema9) and not pd.isna(last_ema20):
                        if prev_ema9 > prev_ema20 and last_ema9 < last_ema20:
                            signals.append("EMA데드크로스")
        except Exception as e:
            logger.error(f"{ticker} 매도 시그널 체크 실패: {e}")
        return signals

    async def _execute_profit_sell(self, ticker, position, current_price, reason):
        """수익 중 시장가 매도 (트레일링 스탑 / AI 판단 공통 처리)."""
        while not rate_limiter.can_make_order():
            await asyncio.sleep(0.1)
        async with self._sell_lock:
            position = self.stop_loss_manager.positions.get(ticker)
            if not position:
                logger.info(f"{ticker} 매도 스킵: 이미 청산됨")
                return False
            current_price = safe_get_current_price(ticker) or current_price
            amount = position["amount"]
            profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
            try:
                sell_result = upbit.sell_market_order(ticker, amount)
                if sell_result and "uuid" in sell_result:
                    actual_price = await self._get_actual_sell_price(sell_result["uuid"], current_price)
                    profit_pct = ((actual_price * (1 - TRADING_FEE) - position['entry_price'] * (1 + TRADING_FEE)) / (position['entry_price'] * (1 + TRADING_FEE))) * 100
                    await self.trading_history.add_trade(ticker, "sell", actual_price, amount, reason, entry_price=position['entry_price'])
                    self.stop_loss_manager.remove_position(ticker)
                    if ticker in self.purchased_coins["coins"]:
                        self.purchased_coins["coins"].remove(ticker)
                        FileManager.save_json(COINS_FILE, self.purchased_coins)
                    message = (
                        f"🟢 익절 실행: {ticker}\n"
                        f"진입가: {position['entry_price']:,.2f}원\n"
                        f"체결가: {actual_price:,.2f}원\n"
                        f"수익률: {profit_pct:.2f}% (수수료 포함)\n"
                        f"거래 금액: {amount * actual_price:,.2f}원\n"
                        f"사유: {reason}"
                    )
                    await async_send_telegram_message(message)
                    logger.info(f"{ticker} 익절 실행 완료: {reason}")
                    self.sync_with_upbit_account()
                    return True
                else:
                    logger.error(f"{ticker} 익절 주문 실패: {sell_result or '응답 없음'}")
                    return False
            except Exception as e:
                logger.error(f"{ticker} 익절 실행 실패: {e}")
                return False

    async def _check_stop_and_trailing(self, ticker, emergency=False):
        """손절 + 트레일링 스탑 + 최고가/트레일링 업데이트 공통 로직.

        Returns:
            "sold" — 손절 또는 트레일링 스탑으로 매도 완료
            "hold" — 매도 조건 미충족, 보유 유지
            None   — 포지션/가격 정보 없음 (건너뜀)
        """
        position = self.stop_loss_manager.positions.get(ticker)
        if not position:
            return None
        current_price = safe_get_current_price(ticker)
        if not current_price:
            return None

        entry_price = float(position["entry_price"])
        profit_percent = ((current_price * (1 - TRADING_FEE) - entry_price * (1 + TRADING_FEE)) / (entry_price * (1 + TRADING_FEE))) * 100
        sl_pct = float(position.get("stop_loss", self.stop_loss_manager.default_stop_loss))
        prefix = "긴급 " if emergency else ""

        # 1. 손절
        if profit_percent <= -sl_pct:
            log_fn = logger.warning if emergency else logger.info
            log_fn(f"{prefix}손절 조건: {ticker} 수익률 {profit_percent:.2f}% <= -{sl_pct}%")
            if emergency:
                self.stop_loss_manager.cancel_limit_orders_only(ticker)
                position = self.stop_loss_manager.positions.get(ticker)
                if not position:
                    return "sold"
            await self._execute_stop_loss(ticker, position, current_price, emergency=emergency)
            return "sold"

        # 2. 트레일링 스탑 체크 (이미 설정된 경우 수익률과 무관하게 체크)
        trailing_stop_price = position.get("trailing_stop_price")
        if trailing_stop_price and current_price <= trailing_stop_price:
            reason = (
                f"{prefix}트레일링 스탑 발동 (수익={profit_percent:.2f}%, "
                f"스탑={trailing_stop_price:,.2f}원)"
            )
            log_fn = logger.warning if emergency else logger.info
            log_fn(f"{ticker} {reason}")
            self.stop_loss_manager.cancel_limit_orders_only(ticker)
            position = self.stop_loss_manager.positions.get(ticker)
            if position:
                await self._execute_profit_sell(ticker, position, current_price, reason)
            return "sold"

        # 3. 최고가 항상 업데이트 (수익/손실 무관)
        self.stop_loss_manager.update_highest_price(ticker, current_price)

        # 4. 수익 중이면 트레일링 스탑 업데이트
        if profit_percent > 0:
            self.stop_loss_manager.update_trailing_stop(ticker, current_price)

        return "hold"

    async def execute_sell_strategy(self):
        """손절 + 트레일링 스탑 + LSTM/기술지표 → GPT-4 AI 매도 통합 처리."""
        try:
            if not self.purchased_coins["coins"]:
                logger.info("보유 코인 없음, 매도 전략 건너뜀")
                return
            for ticker in self.purchased_coins["coins"][:]:
                if not rate_limiter.can_make_api_call():
                    await asyncio.sleep(0.1)
                    continue
                try:
                    # 손절/트레일링 스탑 공통 체크
                    result = await self._check_stop_and_trailing(ticker)
                    if result != "hold":
                        continue

                    # 수익 중인 경우: AI 매도 판단
                    position = self.stop_loss_manager.positions.get(ticker)
                    if not position:
                        continue
                    current_price = safe_get_current_price(ticker)
                    if not current_price:
                        continue
                    entry_price = float(position["entry_price"])
                    profit_percent = ((current_price * (1 - TRADING_FEE) - entry_price * (1 + TRADING_FEE)) / (entry_price * (1 + TRADING_FEE))) * 100
                    sl_pct = float(position.get("stop_loss", self.stop_loss_manager.default_stop_loss))

                    # AI 매도 판단 비활성화 — 트레일링 스탑이 익절 전담
                    # if profit_percent > 0:
                    #     signals = await self._check_bearish_signals(ticker, current_price)
                    #     if len(signals) >= 2:
                    #         logger.info(f"{ticker} 매도 시그널 {len(signals)}개 감지: {signals} → GPT-4 판단 요청")
                    #         should_sell, ai_reason = self.market_analyzer.ai_sell_decision(
                    #             ticker, current_price, profit_percent, signals
                    #         )
                    #         if should_sell:
                    #             reason = f"AI 매도 판단: {ai_reason} (시그널: {', '.join(signals)})"
                    #             self.stop_loss_manager.cancel_limit_orders_only(ticker)
                    #             position = self.stop_loss_manager.positions.get(ticker)
                    #             if position:
                    #                 await self._execute_profit_sell(ticker, position, current_price, reason)
                    #         else:
                    #             logger.info(f"{ticker} GPT-4 보유 유지 판단: {ai_reason}")

                    if profit_percent > 0:
                        logger.info(f"{ticker} 수익 중: {profit_percent:.2f}% (트레일링 스탑 감시 중)")
                    else:
                        logger.info(f"{ticker} 매도 조건 미충족: 수익률={profit_percent:.2f}% (손절 -{sl_pct}%)")
                except Exception as e:
                    logger.error(f"{ticker} 매도 전략 실행 실패: {e}")
        except Exception as e:
            logger.error(f"매도 전략 전체 실행 실패: {e}", exc_info=True)

    async def _execute_stop_loss(self, ticker, position, current_price, emergency=False):
        while not rate_limiter.can_make_order():
            await asyncio.sleep(0.1)
        async with self._sell_lock:
            position = self.stop_loss_manager.positions.get(ticker)
            if not position:
                logger.info(f"{ticker} 손절 스킵: 이미 청산됨")
                return False
            amount = position["amount"]
            loss_pct = ((current_price - position['entry_price']) / position['entry_price'] * 100)
            reason_text = (
                f"긴급 손절 (손실: {loss_pct:.2f}%)"
                if emergency else f"손절 (손실: {loss_pct:.2f}%)"
            )
            try:
                sell_result = upbit.sell_market_order(ticker, amount)
                if sell_result and "uuid" in sell_result:
                    actual_price = await self._get_actual_sell_price(sell_result["uuid"], current_price)
                    loss_pct = ((actual_price * (1 - TRADING_FEE) - position['entry_price'] * (1 + TRADING_FEE)) / (position['entry_price'] * (1 + TRADING_FEE))) * 100
                    await self.trading_history.add_trade(
                        ticker, "sell", actual_price, amount,
                        reason_text, entry_price=position['entry_price']
                    )
                    self.stop_loss_manager.remove_position(ticker)
                    if ticker in self.purchased_coins["coins"]:
                        self.purchased_coins["coins"].remove(ticker)
                        FileManager.save_json(COINS_FILE, self.purchased_coins)
                    self.trading_history.add_to_blacklist(ticker)
                    invalidate_cache("scan_for_opportunities", ticker)
                    title = "🔴 긴급 손절 실행" if emergency else "🔴 손절 실행"
                    message = (
                        f"{title}: {ticker}\n"
                        f"진입가: {position['entry_price']:,.2f}원\n"
                        f"체결가: {actual_price:,.2f}원\n"
                        f"손실률: {loss_pct:.2f}% (수수료 포함)\n"
                        f"거래 금액: {amount * actual_price:,.2f}원"
                    )
                    await async_send_telegram_message(message)
                    logger.info(f"{ticker} {'긴급 ' if emergency else ''}손절 실행 완료")
                    self.sync_with_upbit_account()
                    return True
                else:
                    logger.error(f"{ticker} 손절 주문 실패: {sell_result or '응답 없음'}")
                    return False
            except Exception as e:
                logger.error(f"{ticker} 손절 실행 실패: {e}")
                return False

    async def check_pending_orders(self):
        try:
            for ticker, position in list(self.stop_loss_manager.positions.items()):
                if position.get('status') != 'pending' or 'order_uuid' not in position:
                    continue
                order_uuid = position['order_uuid']
                order_info = upbit.get_order(order_uuid)
                if not order_info:
                    logger.warning(f"{ticker} 주문 정보 가져오기 실패, UUID: {order_uuid}")
                    continue
                if order_info['state'] == 'done':
                    logger.info(f"{ticker} 주문 체결됨")
                    executed_amount = float(order_info['executed_volume'])
                    current_ref = safe_get_current_price(ticker) or 0
                    reason = position.get('reason', 'AI 추천')
                    avg_price = await resolve_executed_price(self.trading_history, ticker, "buy", order_info, executed_amount, current_ref, reason)
                    position['status'] = 'active'
                    position['entry_price'] = avg_price
                    position['amount'] = executed_amount
                    self.stop_loss_manager.positions[ticker] = position
                    FileManager.save_json(POSITIONS_FILE, self.stop_loss_manager.positions)
                    if ticker not in self.purchased_coins["coins"]:
                        self.purchased_coins["coins"].append(ticker)
                        FileManager.save_json(COINS_FILE, self.purchased_coins)
                    message = (
                        f"✅ 매수 주문 체결: {ticker}\n"
                        f"체결가: {avg_price:,.2f}원\n"
                        f"수량: {executed_amount:.8f}개\n"
                        f"총 금액: {executed_amount * avg_price:,.2f}원\n"
                    )
                    await async_send_telegram_message(message)
                    self.sync_with_upbit_account()
                elif order_info['state'] == 'wait':
                    logger.info(f"{ticker} 주문 대기 중")
                elif order_info['state'] == 'cancel':
                    logger.info(f"{ticker} 주문 취소됨")
                    del self.stop_loss_manager.positions[ticker]
                    FileManager.save_json(POSITIONS_FILE, self.stop_loss_manager.positions)
        except Exception as e:
            logger.error(f"대기 주문 확인 오류: {e}")

    async def execute_buy(self, ticker, reason, budget=None):
        if ticker in self.purchased_coins["coins"]:
            logger.info(f"{ticker} 이미 보유 중, 건너뜀")
            return False
        while not rate_limiter.can_make_order():
            await asyncio.sleep(0.1)
        async with self._buy_lock:
            # lock 획득 후 업비트 실잔고로 재확인 (사이클 간 중복 매수 방지)
            try:
                live_bal = upbit.get_balance(ticker.replace("KRW-", "")) or 0
            except Exception:
                live_bal = 0
            if live_bal > 0 or ticker in self.purchased_coins["coins"]:
                logger.info(f"{ticker} 이미 보유 중 (실잔고 재확인), 건너뜀")
                return False
            try:
                current_price = safe_get_current_price(ticker)
                if not current_price:
                    logger.error(f"{ticker} 현재 가격 가져오기 실패")
                    return False
                if budget is None:
                    krw_balance = upbit.get_balance("KRW")
                    if krw_balance is None:
                        logger.error("KRW 잔고 가져오기 실패, API 키 또는 네트워크 확인")
                        return False
                    coin_type = 'BTC' if ticker == 'KRW-BTC' else 'ETH' if ticker == 'KRW-ETH' else 'ALT'
                    buy_ratio = self.buy_ratios.get(coin_type, self.buy_ratios['ALT'])
                    volatility_factor = self.market_analyzer.get_volatility_factor(ticker)
                    buy_ratio *= volatility_factor
                    budget = krw_balance * buy_ratio
                if budget < MIN_KRW_BALANCE:
                    logger.error(f"예산 부족: {budget:,.2f}원 < {MIN_KRW_BALANCE}원")
                    return False
                amount = budget / current_price
                buy_result = upbit.buy_market_order(ticker, budget)
                if buy_result and "uuid" in buy_result:
                    order_uuid = buy_result["uuid"]
                    order_info = upbit.get_order(order_uuid)
                    for _ in range(10):
                        if order_info and order_info["state"] in ["done", "cancel"]:
                            break
                        await asyncio.sleep(1)
                        order_info = upbit.get_order(order_uuid)
                    executed_amount = float(order_info.get("executed_volume", 0)) if order_info else 0
                    order_state = order_info["state"] if order_info else "unknown"
                    if executed_amount > 0 and order_state in ["done", "cancel"]:
                        executed_price = await resolve_executed_price(self.trading_history, ticker, "buy", order_info, executed_amount, current_price, reason)
                        self.stop_loss_manager.add_position(ticker, executed_price, executed_amount)
                        if ticker not in self.purchased_coins["coins"]:
                            self.purchased_coins["coins"].append(ticker)
                            FileManager.save_json(COINS_FILE, self.purchased_coins)
                        state_text = "부분 체결" if order_state == "cancel" else "체결 완료"
                        logger.info(f"{ticker} 매수 {state_text}: 가격={executed_price:,.2f}원, 수량={executed_amount:.8f}개")
                        if order_state == "cancel":
                            logger.warning(f"{ticker} 주문이 부분 체결 후 취소됨 — 체결된 수량만 포지션에 반영")
                        message = (
                            f"🟢 매수 {state_text}: {ticker}\n"
                            f"가격: {executed_price:,.2f}원\n"
                            f"수량: {executed_amount:.8f}개\n"
                            f"총 금액: {executed_amount * executed_price:,.2f}원\n"
                            f"사유: {reason}"
                        )
                        await async_send_telegram_message(message)
                        self.sync_with_upbit_account()
                        return True
                    else:
                        logger.error(f"{ticker} 매수 미체결 또는 취소: 상태={order_info['state'] if order_info else '정보 없음'}")
                        if order_info and order_info["state"] == "wait":
                            upbit.cancel_order(order_uuid)
                            logger.info(f"{ticker} 미체결 주문 취소: UUID={order_uuid}")
                            message = (
                                f"⚠️ 매수 미체결 및 취소: {ticker}\n"
                                f"주문 UUID: {order_uuid}\n"
                                f"상태: {order_info['state']}"
                            )
                            await async_send_telegram_message(message)
                        return False
                else:
                    logger.error(f"{ticker} 매수 주문 실패: {buy_result or '응답 없음'}")
                    return False
            except Exception as e:
                logger.error(f"{ticker} 매수 실행 실패: {e}", exc_info=True)
                return False

    async def log_portfolio_status(self, force=False):
        # 2시간(7200초)마다 한 번만 텔레그램 전송 (force=True면 무조건 전송)
        now = time.time()
        if not force and now - getattr(self, "_last_portfolio_msg_time", 0) < 60:
            return
        self._last_portfolio_msg_time = now
        try:
            balances = upbit.get_balances() or []
            balance_map = {
                f"KRW-{b['currency']}": b
                for b in balances
                if b.get('currency') != 'KRW'
            }
            krw_balance = upbit.get_balance("KRW") or 0
            total_coin_value = 0.0
            total_invested = 0.0
            coin_lines = []
            if not self.purchased_coins["coins"]:
                coin_lines.append("없음")
            else:
                try:
                    price_map = pyupbit.get_current_price(self.purchased_coins["coins"]) or {}
                    if not isinstance(price_map, dict):
                        price_map = {}
                except Exception as e:
                    logger.warning(f"일괄 가격 조회 실패: {e}")
                    price_map = {}
                # None 값 제거
                price_map = {k: v for k, v in price_map.items() if v}
                # 일괄 조회 실패 시 개별 조회 fallback (safe_get_current_price 사용)
                if not price_map:
                    logger.info("일괄 가격 조회 결과 비어있음, 개별 조회 시도")
                    for ticker in self.purchased_coins["coins"]:
                        p = safe_get_current_price(ticker)
                        if p:
                            price_map[ticker] = p
                # 여전히 누락되거나 None인 코인이 있으면 개별 보완
                for ticker in self.purchased_coins["coins"]:
                    if not price_map.get(ticker):
                        p = safe_get_current_price(ticker)
                        if p:
                            price_map[ticker] = p
                for ticker in self.purchased_coins["coins"]:
                    b = balance_map.get(ticker)
                    amount = (float(b['balance']) + float(b.get('locked', 0))) if b else 0
                    if amount > 0:
                        current_price = price_map.get(ticker) if isinstance(price_map, dict) else price_map
                        if current_price:
                            value = amount * current_price
                            total_coin_value += value
                            avg_price = float(b.get('avg_buy_price', 0)) if b else 0
                            if avg_price == 0:
                                avg_price = self.stop_loss_manager.positions.get(ticker, {}).get('entry_price', current_price)
                            invested = amount * avg_price
                            total_invested += invested
                            roi = ((current_price * (1 - TRADING_FEE) - avg_price * (1 + TRADING_FEE)) / (avg_price * (1 + TRADING_FEE)) * 100) if avg_price > 0 else 0
                            profit_krw = (current_price * (1 - TRADING_FEE) * amount) - (avg_price * (1 + TRADING_FEE) * amount)
                            coin_name = ticker.replace("KRW-", "")
                            coin_lines.append(f"• {coin_name}: {roi:+.2f}% ({profit_krw:+,.0f}원)")
                        else:
                            coin_lines.append(f"• {ticker.replace('KRW-','')}: 가격 정보 없음")
                    else:
                        coin_lines.append(f"• {ticker.replace('KRW-','')}: 잔고 없음")
            total_sell_value = total_coin_value * (1 - TRADING_FEE)  # 매도 시 수수료
            total_buy_cost = total_invested * (1 + TRADING_FEE)    # 매수 시 수수료
            total_roi = ((total_sell_value - total_buy_cost) / total_buy_cost * 100) if total_buy_cost > 0 else 0
            total_profit_krw = total_sell_value - total_buy_cost
            portfolio_message = (
                f"📊 {datetime.now().strftime('%m/%d %H:%M')}\n"
                + "\n".join(coin_lines) + "\n"
                f"─────────────────\n"
                f"총 수익률: {total_roi:+.2f}% ({total_profit_krw:+,.0f}원)"
            )
            await async_send_telegram_message(portfolio_message)
            logger.info(f"포트폴리오 상태: KRW 잔고={krw_balance:,.2f}원, 총 투자={total_invested:,.2f}원, 코인 평가액={total_coin_value:,.2f}원, 수익률={total_roi:.2f}%")
        except Exception as e:
            logger.error(f"포트폴리오 상태 기록 오류: {e}", exc_info=True)
            current_time = time.time()
            if current_time - self.last_portfolio_error_time > 14400:
                await async_send_telegram_message(f"⚠️ 포트폴리오 상태 조회 실패: {str(e)[:100]}...")
                self.last_portfolio_error_time = current_time

    async def emergency_stop_loss_loop(self, poll_seconds: float):
        """메인 사이클 대기와 무관하게 손절/트레일링 스탑을 폴링."""
        logger.info(f"긴급 손절 감시 시작 (폴링 {poll_seconds}초, 포지션 stop_loss%% 기준)")
        while True:
            try:
                await asyncio.sleep(poll_seconds)
                for ticker in list(self.purchased_coins.get("coins") or []):
                    if not rate_limiter.can_make_api_call():
                        await asyncio.sleep(0.2)
                        continue
                    await self._check_stop_and_trailing(ticker, emergency=True)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"긴급 손절 루프 오류: {e}", exc_info=True)

    async def run_trading_cycle(self):
        try:
            logger.info("트레이딩 사이클 시작...")
            await asyncio.gather(
                self.check_pending_orders(),
                self.stop_loss_manager.check_limit_orders(),
                return_exceptions=True
            )
            await self.find_trading_opportunities()
            await self.execute_sell_strategy()
            await self.execute_trading_strategy()
            await self.log_portfolio_status()
            logger.info("트레이딩 사이클 완료")
        except Exception as e:
            logger.error(f"트레이딩 사이클 실패: {e}", exc_info=True)
            await async_send_telegram_message(f"⚠️ 트레이딩 사이클 오류: {str(e)[:100]}...")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @cache_result(expiry_seconds=CACHE_EXPIRY_SECONDS['default'])
    def analyze_coin(self, ticker):
        while not rate_limiter.can_make_api_call():
            time.sleep(0.1)
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=30)
            if df is None or df.empty:
                logger.error(f"{ticker} OHLCV 데이터 가져오기 실패")
                return None
            if len(df) < 14:
                logger.warning(f"{ticker} 데이터 부족: {len(df)}일, 최소 14일 필요")
                return None
            df = self.market_analyzer.add_technical_indicators(df)
            logger.info(f"{ticker} 코인 분석 완료: {len(df)}일 데이터")
            return df
        except Exception as e:
            logger.error(f"{ticker} 코인 분석 실패: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    CYCLE_INTERVAL = int(os.getenv("CYCLE_INTERVAL", 1200))  #작동 시간 설정 60 = 1분, 3600= 1시간

    async def main():
        bot = TradingBot()
        asyncio.create_task(bot.emergency_stop_loss_loop(EMERGENCY_STOP_POLL_SECONDS))
        try:
            logger.info("초기 포트폴리오 상태 조회 중...")
            await bot.log_portfolio_status(force=True)
            while True:
                start_time = time.time()
                await bot.run_trading_cycle()
                elapsed = time.time() - start_time
                remaining = max(0, CYCLE_INTERVAL - elapsed)
                logger.info(f"다음 사이클까지 {remaining:.0f}초 대기")
                if remaining == 0:
                    logger.warning("트레이딩 사이클 실행 시간이 1시간을 초과했습니다. 즉시 다음 사이클 시작.")
                await asyncio.sleep(remaining)
        except asyncio.CancelledError:
            logger.info("프로그램 종료 요청 수신, 리소스 정리 중...")
            raise
        except Exception as e:
            logger.error(f"메인 루프 오류: {str(e)}", exc_info=True)
            await async_send_telegram_message(f"⚠️ 메인 루프 오류: {str(e)[:100]}...")
            logger.info("60초 후 재시도...")
            await asyncio.sleep(60)
            raise

    logger.info("트레이딩 봇 시작...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("사용자에 의한 종료 요청. 프로그램 정상 종료")
    except Exception as e:
        logger.error(f"프로그램 실행 실패: {str(e)}", exc_info=True)
        try:
            asyncio.run(
                async_send_telegram_message(f"🚨 프로그램 비정상 종료: {str(e)[:100]}...")
            )
        except Exception:
            pass