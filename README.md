# Imagineers-Coin — LSTM 기반 암호화폐 자동매매 봇

업비트 시세를 수집해 **LSTM으로 다음 종가를 예측**하고, 기술지표·시장 심리와 함께
판단해 매수·매도를 자동 실행하는 개인 프로젝트입니다.
거래 내역은 텔레그램으로 실시간 통보되고, 웹 대시보드로 손익을 확인할 수 있습니다.

`2026.04 ~` · Python · 단일 파일 2,300줄 규모

<br>

## 시스템 구조

```
데이터 수집       PyUpbit          최근 60일 OHLCV · 이상치/결측 제거
      ↓
전처리            MinMaxScaler     30일 시퀀스 생성
      ↓
가격 예측         LSTM 3층 + Dense  다음날 종가 예측
      ↓
              ┌─ 기술지표 6종     Bollinger · RSI · MACD · ADX · ATR · EMA
매매 판단  ────┼─ 시장 심리        공포탐욕지수 API
              ├─ 뉴스             Google News RSS
              └─ AI 분석          OpenAI 기반 시장 상황 판단
      ↓
주문 실행         오더북 분석으로 최적 매수호가 산출
      ↓
포지션 관리       손절 -3.5% · 익절 · 트레일링 스탑
      ↓
알림 / 모니터링   텔레그램 봇 · 웹 대시보드 (Cloudflare Tunnel로 외부 공유)
```

<br>

## 주요 구현

### 예측 모델
- **LSTM 3층 + Dense 출력**, 30일 시퀀스로 다음날 종가 예측
- MinMaxScaler 정규화, Dropout · EarlyStopping
- Grid Search로 learning rate · batch size 튜닝
- `backtest_lstm()` — 과거 구간 예측 성능 검증
- `monitor_prediction_accuracy()` — 예측값과 실제값 괴리 추적
- `retrain_on_loss()` — **손실 발생 시 해당 종목 모델 재학습**

### 매매 로직
- 예측가와 현재가 비교 + 기술지표 6종 조합으로 매수 판단
- **오더북 분석** — 현재가 대비 1~3% 아래 매수 호가 중 거래량이 가장 큰 지점을 최적 매수가로 선정
- **트레일링 스탑** — 고점 갱신을 추적하며 손절선을 따라 올림
- 손절 -3.5%, 익절, 시장 위험도에 따른 신규 매수 차단
- 손실 종목 **블랙리스트** 처리 (쿨다운 24시간)

### 안정성 설계
| 문제 | 대응 |
|---|---|
| 업비트 API 호출 제한 (분당 30회) | `RateLimiter` 클래스로 주문·조회 호출 분리 관리 |
| 여러 스레드의 JSON 동시 접근 | `FileLock` 기반 `FileManager` |
| 일시적 API 실패 | `tenacity` 지수 백오프 재시도 (최대 3회) |
| 반복 조회 비용 | 파일 캐시 데코레이터 (`cache_result`), 손절 후 무효화 |

### 모니터링
- **텔레그램 봇** — 매수·매도 시점, 사유, 오류를 실시간 통보
- **웹 대시보드** (`dashboard.html` + `dashboard_server.py`)
  - 총 자산 · 수익률 · 보유 포지션 · 시장 위험도
  - 완결 거래 테이블 (매수가 · 매도가 · 보유기간 · 손익 · 수수료 · **매도 사유**)
  - 자산 변화 그래프, 포지션 구성 도넛 차트
  - Cloudflare Tunnel로 외부 접속 지원

<br>

## 실거래 결과

가상 백테스트가 아니라 **실제 계좌로 운용한 결과**입니다.

| 항목 | 값 |
|---|---|
| 기간 | 2026.04.01 ~ 05.27 |
| 총 거래 | 238건 (매수 123 · 매도 115) |
| 원금 | 300,000원 |
| 승률 | 56 % |
| 실현 손익 | **-25,806원** (수수료 6,132원 별도) |

**수익을 내지 못했습니다.** 손절이 대부분(-3.4 ~ -3.6%)이고 익절이 적었습니다.
원인을 분석한 결과는 아래와 같습니다.

<br>

## 한계와 개선점

직접 코드를 검토하며 정리한 문제들입니다.

**1. 손절 반응 지연**
현재가 폴링 주기가 10초입니다. 업비트 API가 분당 30회 제한이라 보유 종목이
5개면 10초가 한계입니다. 급락 시 손절이 늦어 -3.4% 목표가 실제로는
-3.5~-3.6%에 체결됐습니다.
→ 웹소켓으로 전환하면 실시간 대응이 가능하지만, 기존 폴링 구조와 상태 관리가
   충돌할 위험이 있어 트레이드오프를 검토 중입니다.

**2. 매매 로직 중복**
- 손절 실행이 3곳(`execute_sell_strategy` / `check_positions` / `emergency_stop_loss_loop`)
- 익절 2곳, 매수 3곳에 유사 코드가 흩어져 있습니다
- 같은 사이클에서 손절 체크가 2번 수행됩니다
→ 공통 실행 경로로 통합이 필요합니다

**3. 예측 신뢰도**
고변동성 구간에서 LSTM 예측이 실제 방향과 어긋나는 경우가 많았습니다.
단일 모델 의존을 줄이고 기술지표 가중치를 높이는 방향을 검토했습니다.

<br>

## 기술 스택

`Python` `TensorFlow/Keras` `LSTM` `pandas` `NumPy` `scikit-learn`
`PyUpbit` `OpenAI API` `Telegram Bot API` `ta` `tenacity` `filelock`

<br>

## 실행

```bash
pip install -r requirements.txt
python autotrading.py
```

환경변수(`~/.env_autocoin`)에 다음 값이 필요합니다.

```
UPBIT_ACCESS_KEY · UPBIT_SECRET_KEY
OPENAI_API_KEY
TELEGRAM_TOKEN · TELEGRAM_CHAT_ID
```

대시보드는 `python dashboard_server.py` 실행 후 `http://localhost:8080`.

> 개인 거래 데이터(`trade_history.json`, `positions.json`), 로그, 학습된 모델은
> `.gitignore`로 제외했습니다.

<br>

## 참고

투자 판단을 자동화한 개인 학습용 프로젝트입니다. 실제 자금이 투입되며
손실이 발생할 수 있으므로, 그대로 사용하는 것을 권하지 않습니다.
