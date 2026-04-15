"""
업비트 트레이딩 봇 대시보드 서버
실행: python dashboard_server.py
접속: http://localhost:8080
"""
import os
import json
import glob
import time
import requests
from flask import Flask, jsonify, send_file
from datetime import datetime
from dotenv import load_dotenv

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'api_cache')

# 업비트 인증 (KRW 잔고 조회용)
_ENV_PATH = os.path.join(os.path.expanduser("~"), ".env_autocoin")
load_dotenv(_ENV_PATH)

try:
    import pyupbit
    _upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
except Exception:
    _upbit = None


def load_json(filename):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def smart_round(v):
    """저가 코인은 소수점 유지, 고가 코인은 정수 반올림"""
    if v is None:
        return None
    if v < 1:
        return round(v, 4)
    if v < 10:
        return round(v, 2)
    if v < 1000:
        return round(v, 1)
    return round(v)


def get_upbit_prices(tickers):
    if not tickers:
        return {}
    try:
        markets = ','.join(tickers)
        resp = requests.get(
            f"https://api.upbit.com/v1/ticker?markets={markets}",
            timeout=5
        )
        return {item['market']: item['trade_price'] for item in resp.json()}
    except Exception:
        return {}


def get_krw_balance():
    """업비트 KRW 잔고 조회"""
    try:
        if _upbit:
            bal = _upbit.get_balance("KRW")
            return round(bal) if bal is not None else None
    except Exception:
        pass
    return None


def get_upbit_holdings():
    """업비트 실제 보유 코인 조회 (KRW 제외).
    반환: {ticker: {'amount': float, 'avg_buy_price': float}}
    """
    if not _upbit:
        return None
    try:
        balances = _upbit.get_balances()
        if not balances or not isinstance(balances, list):
            return None
        holdings = {}
        for b in balances:
            try:
                currency = b.get('currency')
                if not currency or currency == 'KRW':
                    continue
                unit = b.get('unit_currency', 'KRW')
                ticker = f"{unit}-{currency}"
                if not ticker.startswith('KRW-'):
                    continue
                amount = float(b.get('balance', 0)) + float(b.get('locked', 0))
                if amount <= 0:
                    continue
                avg_price = float(b.get('avg_buy_price', 0) or 0)
                holdings[ticker] = {'amount': amount, 'avg_buy_price': avg_price}
            except (TypeError, ValueError):
                continue
        return holdings
    except Exception:
        return None


def get_latest_risk_level():
    """가장 최근 AI 시장 분석 캐시에서 risk_level 읽기"""
    try:
        pattern = os.path.join(CACHE_DIR, 'ai_analyze_market_*.json')
        files = [f for f in glob.glob(pattern) if not f.endswith('.lock')]
        if not files:
            return None
        latest = max(files, key=os.path.getmtime)
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = data.get('result', {})
        if isinstance(result, dict) and 'risk_level' in result:
            return int(result['risk_level'])
    except Exception:
        pass
    return None


def get_bot_last_activity():
    """trading_bot.log 마지막 수정 시간 기준 경과 분 반환"""
    log_path = os.path.join(BASE_DIR, 'trading_bot.log')
    if os.path.exists(log_path):
        minutes_ago = int((time.time() - os.path.getmtime(log_path)) / 60)
        return minutes_ago
    return None


def pair_trades(trades):
    """매수/매도 쌍 매칭하여 완결 거래 생성 (DCA 평단가 기준)"""
    completed = []
    # 종목별 누적 포지션 (DCA 반영)
    positions = {}  # ticker -> {total_cost, total_amount, first_buy_date}

    for trade in trades:
        ticker = trade['ticker']
        action = trade['action']

        if action == 'buy':
            if ticker not in positions:
                positions[ticker] = {'total_cost': 0, 'total_amount': 0, 'first_buy_date': trade['timestamp']}
            pos = positions[ticker]
            pos['total_cost'] += trade['price'] * trade['amount']
            pos['total_amount'] += trade['amount']

        elif action == 'sell':
            pos = positions.get(ticker)
            if pos and pos['total_amount'] > 0:
                # 매도 기록에 실제 진입가가 있으면 우선 사용 (업비트 동기화된 정확한 평단가)
                if trade.get('entry_price'):
                    avg_buy_price = float(trade['entry_price'])
                else:
                    avg_buy_price = pos['total_cost'] / pos['total_amount']
                used_amount = min(pos['total_amount'], trade['amount'])
                buy_cost = avg_buy_price * used_amount
                sell_revenue = trade['price'] * used_amount
                pnl = sell_revenue - buy_cost
                pnl_pct = (trade['price'] - avg_buy_price) / avg_buy_price * 100
                fee = (buy_cost + sell_revenue) * 0.0005

                # 보유 기간 계산
                try:
                    buy_dt = datetime.fromisoformat(pos['first_buy_date'])
                    sell_dt = datetime.fromisoformat(trade['timestamp'])
                    holding_hours = int((sell_dt - buy_dt).total_seconds() / 3600)
                except Exception:
                    holding_hours = None

                completed.append({
                    'ticker': ticker,
                    'coin': ticker.replace('KRW-', ''),
                    'buy_date': pos['first_buy_date'],
                    'buy_price': smart_round(avg_buy_price),
                    'sell_date': trade['timestamp'],
                    'sell_price': smart_round(trade['price']),
                    'amount': used_amount,
                    'pnl_krw': round(pnl),
                    'pnl_pct': round(pnl_pct, 2),
                    'fee': round(fee),
                    'net_pnl_krw': round(pnl - fee),
                    'holding_hours': holding_hours,
                    'reason': trade.get('reason', ''),
                })

                # 매도된 만큼 포지션 차감
                pos['total_amount'] -= used_amount
                pos['total_cost'] -= avg_buy_price * used_amount
                if pos['total_amount'] <= 0:
                    del positions[ticker]

    return completed


@app.route('/api/data')
def get_data():
    positions_raw = load_json('positions.json') or {}
    history = load_json('trade_history.json') or {'trades': []}
    trades = history.get('trades', [])

    # 업비트 실제 잔고와 동기화 (사용자가 앱에서 직접 매매한 경우 반영)
    upbit_holdings = get_upbit_holdings()
    if upbit_holdings is not None:
        # 1) 업비트에 없는 코인은 positions에서 제거 (수동 매도 반영)
        for ticker in list(positions_raw.keys()):
            if ticker not in upbit_holdings:
                positions_raw.pop(ticker, None)
        # 2) 업비트 보유 코인의 수량/평단가를 실제 값으로 갱신, 신규 발견 시 추가
        for ticker, h in upbit_holdings.items():
            avg_price = h['avg_buy_price'] or 0
            amount = h['amount']
            if ticker in positions_raw:
                pos = positions_raw[ticker]
                pos['amount'] = amount
                if avg_price > 0:
                    pos['entry_price'] = avg_price
            else:
                # 봇이 모르는 수동 매수 코인 — 최소 필드만 채워 추가
                positions_raw[ticker] = {
                    'entry_price': avg_price,
                    'amount': amount,
                    'stop_loss': 5,
                    'take_profit': 10,
                    'timestamp': '',
                }

    # 현재가 조회
    tickers = list(positions_raw.keys())
    prices = get_upbit_prices(tickers)

    # 포지션별 DCA 횟수 계산 (첫 매수 제외)
    buy_counts = {}
    for t in trades:
        if t['action'] == 'buy':
            buy_counts[t['ticker']] = buy_counts.get(t['ticker'], 0) + 1

    # 포지션 정보 계산
    positions = []
    total_invested = 0
    total_current_value = 0

    for ticker, pos in positions_raw.items():
        cur_price = prices.get(ticker, 0)
        entry = pos['entry_price']
        amount = pos['amount']
        invested = entry * amount
        cur_value = cur_price * amount if cur_price else invested
        pnl_pct = (cur_price - entry) / entry * 100 if cur_price and entry else 0
        pnl_krw = cur_value - invested

        total_invested += invested
        total_current_value += cur_value

        trailing_stop_price = pos.get('trailing_stop_price')
        highest_price = pos.get('highest_price')
        dca_count = max(0, buy_counts.get(ticker, 1) - 1)

        positions.append({
            'ticker': ticker,
            'coin': ticker.replace('KRW-', ''),
            'entry_price': smart_round(entry),
            'current_price': smart_round(cur_price),
            'amount': round(amount, 6),
            'invested': round(invested),
            'current_value': round(cur_value),
            'pnl_pct': round(pnl_pct, 2),
            'pnl_krw': round(pnl_krw),
            'stop_loss': pos.get('stop_loss', 5),
            'take_profit': pos.get('take_profit', 10),
            'timestamp': pos.get('timestamp', ''),
            'highest_price': smart_round(highest_price) if highest_price else None,
            'trailing_stop_price': smart_round(trailing_stop_price) if trailing_stop_price else None,
            'updated_at': pos.get('updated_at', ''),
            'dca_count': dca_count,
        })

    # 완결 거래 계산
    completed = pair_trades(trades)
    completed_sorted = sorted(completed, key=lambda x: x['sell_date'], reverse=True)

    # 성과 지표
    total_realized_pnl = sum(t['pnl_krw'] for t in completed)
    total_fees = sum(t['fee'] for t in completed)
    net_realized_pnl = total_realized_pnl - total_fees
    wins = sum(1 for t in completed if t['pnl_krw'] > 0)
    win_rate = wins / len(completed) * 100 if completed else 0
    avg_pnl_pct = sum(t['pnl_pct'] for t in completed) / len(completed) if completed else 0
    unrealized_pnl = round(total_current_value - total_invested)

    # 원금 고정 300,000원
    krw_bal = get_krw_balance() or 0
    estimated_initial = 300000
    # 총 수익률 = (현재 총자산 - 원금) / 원금 * 100  ← 실시간 시세 반영
    current_total_asset = krw_bal + total_current_value
    total_return_pct = round((current_total_asset - estimated_initial) / estimated_initial * 100, 2) if estimated_initial > 0 else 0

    chart_points = []
    running = estimated_initial
    for ct in sorted(completed, key=lambda x: x['sell_date']):
        running += ct['pnl_krw'] - ct['fee']
        chart_points.append({
            'date': ct['sell_date'][:16].replace('T', ' '),
            'capital': round(running),
            'ticker': ct['coin'],
            'pnl_pct': ct['pnl_pct'],
        })

    # 매수/매도 내역 (raw) — 가격 소수점 보정
    raw_trades = []
    for t in sorted(trades, key=lambda x: x['timestamp'], reverse=True):
        rt = dict(t)
        rt['price'] = smart_round(t['price'])
        raw_trades.append(rt)

    # 블랙리스트 정보
    blacklist_raw = history.get('blacklist', {})
    now = datetime.now()
    blacklist = []
    for ticker, expiry_str in blacklist_raw.items():
        try:
            expiry = datetime.fromisoformat(expiry_str)
            if expiry > now:
                remaining_min = int((expiry - now).total_seconds() / 60)
                blacklist.append({
                    'ticker': ticker,
                    'coin': ticker.replace('KRW-', ''),
                    'expiry': expiry_str[:16].replace('T', ' '),
                    'remaining_min': remaining_min,
                })
        except Exception:
            pass

    return jsonify({
        'positions': positions,
        'completed_trades': completed_sorted,
        'raw_trades': raw_trades,
        'metrics': {
            'total_completed_trades': len(completed),
            'total_raw_trades': len(trades),
            'win_rate': round(win_rate, 1),
            'total_realized_pnl': total_realized_pnl,
            'net_realized_pnl': net_realized_pnl,
            'total_fees': total_fees,
            'avg_pnl_pct': round(avg_pnl_pct, 2),
            'total_return_pct': total_return_pct,
            'estimated_initial': estimated_initial,
            'unrealized_pnl': unrealized_pnl,
            'total_invested': round(total_invested),
            'total_current_value': round(total_current_value),
            'open_positions': len(positions),
            'krw_balance': get_krw_balance(),
            'risk_level': get_latest_risk_level(),
        },
        'blacklist': blacklist,
        'chart': chart_points,
        'bot_last_activity_min': get_bot_last_activity(),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })


@app.route('/api/tunnel-url')
def tunnel_url():
    url_file = os.path.join(BASE_DIR, 'tunnel_url.txt')
    try:
        with open(url_file, 'r') as f:
            return jsonify({'url': f.read().strip()})
    except FileNotFoundError:
        return jsonify({'url': None})


@app.route('/')
def index():
    return send_file(os.path.join(BASE_DIR, 'dashboard.html'))


if __name__ == '__main__':
    print("=" * 50)
    print("  업비트 트레이딩 봇 대시보드 v2")
    print("  http://localhost:8080 에서 접속하세요")
    print(f"  Python: {__import__('sys').version[:10]}")
    print(f"  파일: {__import__('os').path.abspath(__file__)}")
    print("=" * 50)
    app.run(debug=False, port=8080)
