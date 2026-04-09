"""
trade_history.json의 가격을 업비트 실제 체결 내역과 대조하여 자동 보정하는 스크립트.
실행: python fix_trade_prices.py
"""
import os
import sys
import json
import time
import jwt
import uuid as uuid_mod
import hashlib
from urllib.parse import urlencode
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

_ENV_PATH = os.path.join(os.path.expanduser("~"), ".env_autocoin")
load_dotenv(_ENV_PATH)

ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_HISTORY_FILE = os.path.join(BASE_DIR, "trade_history.json")


def upbit_request(endpoint, params=None):
    """업비트 인증 API 호출"""
    if params:
        query_string = urlencode(params)
        m = hashlib.sha512()
        m.update(query_string.encode())
        query_hash = m.hexdigest()
        payload = {
            'access_key': ACCESS_KEY,
            'nonce': str(uuid_mod.uuid4()),
            'query_hash': query_hash,
            'query_hash_alg': 'SHA512',
        }
    else:
        payload = {
            'access_key': ACCESS_KEY,
            'nonce': str(uuid_mod.uuid4()),
        }
    jwt_token = jwt.encode(payload, SECRET_KEY)
    headers = {'Authorization': f'Bearer {jwt_token}'}
    url = f'https://api.upbit.com/v1{endpoint}'
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_all_closed_orders(market, start_date):
    """특정 마켓의 완료된 주문 전체 조회"""
    all_orders = []
    page = 1
    while True:
        params = {
            'market': market,
            'state': 'done',
            'limit': 100,
            'page': page,
            'order_by': 'asc',
        }
        orders = upbit_request('/orders/closed', params)
        if not orders:
            break
        all_orders.extend(orders)
        if len(orders) < 100:
            break
        page += 1
        time.sleep(0.1)
    return all_orders


def get_order_detail(uuid):
    """개별 주문 상세 조회 (trades 포함)"""
    params = {'uuid': uuid}
    return upbit_request('/order', params)


def calc_avg_price_from_order(order):
    """주문에서 실제 평균 체결 단가 계산"""
    # trades 배열이 있으면 가중 평균
    trades = order.get('trades', [])
    if trades:
        total_funds = sum(float(t.get('funds', 0)) for t in trades)
        total_vol = sum(float(t.get('volume', 0)) for t in trades)
        if total_vol > 0 and total_funds > 0:
            return round(total_funds / total_vol, 8)

    # paid_fee + executed_volume 기반 계산
    executed_vol = float(order.get('executed_volume', 0))
    if executed_vol <= 0:
        return None

    # 시장가 매수: price = 총 KRW
    # 지정가: price = 단가
    side = order.get('side', '')
    ord_type = order.get('ord_type', '')
    price = float(order.get('price', 0)) if order.get('price') else 0

    if ord_type == 'price':  # 시장가 매수 (KRW 금액 지정)
        if price > 0:
            return round(price / executed_vol, 8)
    elif ord_type == 'market':  # 시장가 매도
        # trades에서 계산해야 하지만, 없으면 paid_fee 기반 추정
        paid_fee = float(order.get('paid_fee', 0))
        # 매도 수익 = executed_volume * avg_price, fee = 0.05%
        # 역산 불가능하므로 None 반환
        return None
    else:  # limit 주문
        if price > 0:
            return price

    return None


def match_order_to_trade(order, trade):
    """업비트 주문과 trade_history 기록이 매칭되는지 확인"""
    executed_vol = float(order.get('executed_volume', 0))
    trade_amount = float(trade['amount'])

    # 수량 매칭 (1% 오차 허용)
    if abs(executed_vol - trade_amount) / max(trade_amount, 0.0001) > 0.01:
        return False

    # 시간 매칭 (5분 이내)
    try:
        order_time = datetime.fromisoformat(order['created_at'].replace('+09:00', '').replace('T', 'T'))
        trade_time = datetime.fromisoformat(trade['timestamp'])
        if abs((order_time - trade_time).total_seconds()) > 300:
            return False
    except Exception:
        pass

    # side 매칭
    order_side = order.get('side', '')
    if trade['action'] == 'buy' and order_side != 'bid':
        return False
    if trade['action'] == 'sell' and order_side != 'ask':
        return False

    return True


def main():
    # trade_history.json 로드
    with open(TRADE_HISTORY_FILE, 'r', encoding='utf-8') as f:
        history = json.load(f)

    trades = history['trades']
    print(f"총 {len(trades)}건의 거래 기록 확인\n")

    # 거래에 등장하는 모든 마켓 수집
    markets = set(t['ticker'] for t in trades)
    print(f"마켓: {', '.join(sorted(markets))}\n")

    # 마켓별 업비트 체결 내역 조회
    all_orders = {}
    for market in sorted(markets):
        print(f"{market} 주문 내역 조회 중...")
        orders = get_all_closed_orders(market, None)
        all_orders[market] = orders
        print(f"  → {len(orders)}건")
        time.sleep(0.2)

    print()

    # 각 거래 기록에 대해 매칭되는 주문 찾기
    fixed_count = 0
    for i, trade in enumerate(trades):
        ticker = trade['ticker']
        orders = all_orders.get(ticker, [])

        matched_order = None
        for order in orders:
            if match_order_to_trade(order, trade):
                matched_order = order
                break

        if not matched_order:
            print(f"[{i+1}] {trade['action'].upper()} {ticker} {trade['price']}원 × {trade['amount']} → 매칭 주문 없음 (건너뜀)")
            continue

        # 상세 조회 (trades 배열 포함)
        try:
            detail = get_order_detail(matched_order['uuid'])
            time.sleep(0.1)
        except Exception as e:
            print(f"[{i+1}] {ticker} 상세 조회 실패: {e}")
            detail = matched_order

        # 실제 체결 단가 계산
        real_price = calc_avg_price_from_order(detail)
        if real_price is None:
            # trades 배열 없이 시장가 매도인 경우, 개별 체결 내역에서 역산
            executed_vol = float(detail.get('executed_volume', 0))
            paid_fee = float(detail.get('paid_fee', 0))
            # 매도 총액 역산: trades_count 기반은 불가, locked 필드 활용
            remaining_vol = float(detail.get('remaining_volume', 0))
            if detail.get('ord_type') == 'market' and executed_vol > 0:
                # 시장가 매도: locked = 매도 수량, 수익금 = ?
                # trades 배열이 없으면 정확한 단가 계산 불가
                print(f"[{i+1}] {trade['action'].upper()} {ticker} → 시장가 매도, trades 정보 없어 기존 가격 유지")
                continue
            else:
                print(f"[{i+1}] {trade['action'].upper()} {ticker} → 단가 계산 불가, 기존 가격 유지")
                continue

        old_price = trade['price']
        if abs(old_price - real_price) / max(real_price, 0.0001) < 0.001:
            print(f"[{i+1}] {trade['action'].upper()} {ticker} {old_price}원 → 정확함 ✓")
            continue

        # 가격 수정
        trade['price'] = real_price
        fixed_count += 1
        print(f"[{i+1}] {trade['action'].upper()} {ticker} {old_price}원 → {real_price}원 ★ 수정됨")

    print(f"\n총 {fixed_count}건 수정됨")

    if fixed_count > 0:
        # 백업 생성
        backup_path = TRADE_HISTORY_FILE + '.bak'
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
        print(f"백업 저장: {backup_path}")

        # 저장
        with open(TRADE_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
        print(f"trade_history.json 저장 완료")
    else:
        print("수정할 항목 없음")


if __name__ == '__main__':
    main()
