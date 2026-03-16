---
applyTo: "quantbox/execution/**/*.py,quantbox/core/**/*.py"
---
# API Testing Skill — QuantBox Binance API 검증 규칙

> **목적:** API 관련 코드 변경 시 반드시 실제 소액 주문으로 검증하는 절차.
> Binance 공식 문서가 부정확한 사례가 확인됨 — mock 테스트만으로 커밋 금지.

---

## 1. 핵심 규칙

**API 변경 시 반드시 실제 소액 주문으로 full lifecycle 검증 후 커밋.**

| 단계 | 설명 |
|------|------|
| 1. Unit test | mock 기반 pytest — 로직 검증 (필수지만 충분하지 않음) |
| 2. **Live micro-test** | 실제 DOGEUSDT 50개 ($5 미만) 주문 → 체결 → 취소/청산 |
| 3. Commit | 2단계 통과 후에만 커밋 |

**위반 선례 (2026-03-17, Incident #8):**
- Binance 공식 문서에 regular API (`POST /fapi/v1/order`)가 `TRAILING_STOP_MARKET` 지원한다고 명시
- `binance-futures-connector-python` 소스도 동일하게 나열
- 22개 mock 테스트 전부 통과
- **라이브 API → `-4120: Order type not supported for this endpoint`**
- 원인: Binance가 이 주문 타입을 Algo API 전용으로 이전했지만 문서 미업데이트

---

## 2. 테스트 심볼 & 수량

```python
SYM = 'DOGEUSDT'  # 저가, 유동성 충분, step_size=1
QTY = 50          # ≈ $5 (leverage=5 → margin=$1)
```

다른 심볼이 필요한 경우 (예: 고가 심볼 테스트):
- `BTCUSDT` qty=0.001 (≈ $100 notional, margin $3-5)
- 반드시 leverage=5 이하로 설정

---

## 3. Full Lifecycle 테스트 템플릿

```python
import os, time
from dotenv import load_dotenv
load_dotenv()
from binance.client import Client

c = Client(os.environ['BINANCE_API_KEY'], os.environ['BINANCE_API_SECRET'])
SYM, QTY = 'DOGEUSDT', 50

# 1. Leverage 설정
c.futures_change_leverage(symbol=SYM, leverage=5)

# 2. 포지션 오픈
entry = c.futures_create_order(
    symbol=SYM, side='BUY', type='MARKET',
    quantity=QTY, positionSide='LONG'
)
time.sleep(1)
od = c.futures_get_order(symbol=SYM, orderId=entry['orderId'])
ep = float(od['avgPrice'])
print(f'Entry: ${ep:.6f}')

# 3. 테스트할 주문 배치 (예: SL, TP, Trail)
# ... 여기에 테스트 대상 주문 ...

# 4. 주문 상태 확인
time.sleep(2)
# ... 상태 조회 ...

# 5. 주문 취소
# ... 취소 ...

# 6. 포지션 청산
c.futures_create_order(
    symbol=SYM, side='SELL', type='MARKET',
    quantity=QTY, positionSide='LONG'
)
print('✅ Lifecycle complete')
```

---

## 4. Algo API vs Regular API 구분

| 기능 | Regular API | Algo API |
|------|------------|----------|
| Endpoint | `POST /fapi/v1/order` | `POST /fapi/v1/algoOrder` |
| MARKET / LIMIT | ✅ | ❌ |
| STOP_MARKET | ✅ (하지만 Algo도 가능) | ✅ |
| TAKE_PROFIT_MARKET | ✅ (하지만 Algo도 가능) | ✅ |
| **TRAILING_STOP_MARKET** | **❌ (-4120)** | **✅ (유일)** |
| 주문 ID 타입 | `orderId` (int) | `algoId` (int, 4000000xxx) |
| 취소 | `DELETE /fapi/v1/order` | `DELETE /fapi/v1/algoOrder` |
| 조회 | `GET /fapi/v1/order` | `GET /fapi/v1/algoOrder` |
| 상태 전이 | NEW → FILLED / CANCELED | NEW → WORKING → TRIGGERED → FINISHED |

> ⚠️ Algo API로 배치한 주문은 Regular API로 취소/조회 불가. 반대도 마찬가지.
> `algoId`와 `orderId`는 완전히 별개 ID 체계.

---

## 5. 알려진 Binance 문서 오류 (2026-03 기준)

| 문서 내용 | 실제 동작 | 발견일 |
|-----------|----------|--------|
| Regular API response example에 `TRAILING_STOP_MARKET` 포함 | `-4120` 거부 | 2026-03-17 |
| Algo API `algoStatus: "CANCELLED"` (British) | 실제 `"CANCELED"` (American) 반환 | 2026-03-17 |

> **원칙: 문서보다 실제 API 응답이 source of truth.**
