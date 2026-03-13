# QuantBox 전략 레퍼런스

> **목적:** 활성 전략 3개 + 리스크 관리 게이트 6개를 한눈에 파악하기 위한 레퍼런스.  
> **Source of Truth:** `quantbox/strategy/vol_straddle/config.py`

---

## 전략 구조 한 눈에 보기

```
신호 감지 ──→ 게이트 통과? ──→ 포지션 진입 ──→ 청산
   │                │
   ├─ D_mom8: VR squeeze + momentum 방향   → LONG or SHORT (단방향)
   ├─ S_d8:  VR durational squeeze         → LONG + SHORT 동시 (양방향)
   └─ MA Cluster: SMA 집중 + breakout      → LONG or SHORT (단방향)
```

**Universe:** 60 symbols (Binance USDT-M perpetuals, `symbol_whitelist`)  
**운영 모드:** Hedge Mode (LONG/SHORT 독립 포지션) · 최대 10 동시 슬롯

---

## 전략 1: D_mom8 (VR Squeeze + 방향성)

**개념:** ATR 비율(VR)이 낮으면 변동성이 압축(squeeze)된 상태. 여기서 8-bar 모멘텀 방향으로 진입.

### 진입 조건

| 단계 | 조건 |
|------|------|
| ① VR 계산 | `VR = ATR(8) / ATR(56)` — 단기/장기 변동성 비율 |
| ② Squeeze 감지 | `VR < 0.50` (threshold=0.5) |
| ③ 방향 결정 | 8-bar 모멘텀 > 0 → **LONG** / < 0 → **SHORT** |
| ④ 멀티 TF | 15m · 1h · 4h 각각 독립 신호 발생 |

### 청산 조건 (우선순위 순)

| 조건 | 기준 |
|------|------|
| 🔴 Hard SL | 진입가 ±**1.0%** |
| 🟡 Trail Stop | 고점 대비 **0.2%** 하락 (Trail 활성화: +**1.0%** 수익 도달 시) |
| 🟢 Hard TP | 진입가 ±**20%** (사실상 미도달 — trail이 먼저 청산) |
| ⏱ Max Hold | **192 bars** (15m 기준 48h) |

### 성능 (백테스트, 60 syms · 6yr · flat SL/TP 기준)

| N | WR | PF | EV |
|---|----|----|-----|
| 2,868 | 62.6% | 1.971 | +39 bp |

**L/N = 2.1** (leverage 21x / 10 slots) → 잔고 $1,000 기준 슬롯당 notional $2,100

---

## 전략 2: S_d8 (Straddle — 오늘부터 실제 작동)

**개념:** 변동성이 장시간 압축(squeeze 3+ bars)된 후 → 방향 모름 → LONG + SHORT **동시 진입**.  
한쪽이 터지면 SL, 반대쪽이 수익. 방향 예측 불필요.

> ✅ **2026-03-13 G6 버그 수정 (commit 58e0842):** 기존에 BTC Trend Gate가 LONG leg를 차단해  
> Straddle이 단 한 번도 실거래에서 오픈되지 못했음. 오늘부터 정상 작동.

### 진입 조건

| 단계 | 조건 |
|------|------|
| ① VR Squeeze | `VR < 0.57` |
| ② 지속 기간 | 연속 **3 bars 이상** squeeze 유지 (`s_squeeze_dur=3`) |
| ③ 동시 진입 | **LONG + SHORT** 동시 오픈 (같은 심볼, 각각 독립 포지션) |
| ④ 멀티 TF | 15m · 1h · 4h 각각 독립 신호 |

### 청산 조건

| 조건 | LONG leg | SHORT leg |
|------|----------|-----------|
| 🔴 Hard SL | 진입가 -**1.5%** | 진입가 +**1.5%** |
| 🟡 Trail Stop | 동일 (`d_trail_activate=1.0%, d_trail_distance=0.2%`) | 동일 |
| 🟢 Hard TP | ±**20%** | ±**20%** |

*한쪽이 SL 청산 → 반대쪽 포지션은 계속 운영*

### 성능 (백테스트, 60 syms · 6yr · flat SL/TP 기준)

| N | WR | PF | EV |
|---|----|----|-----|
| 7,706 | 48.4% | **4.755** | +41 bp |

**L/N = 3.0** (leverage 30x / 10 slots) → 슬롯당 LONG $3,000 + SHORT $3,000  
(각 leg가 독립 슬롯 소비 — 스트래들 1쌍 = 슬롯 2개)

---

## 전략 3: MA Cluster (SMA 클러스터)

**개념:** 4개 SMA(20/50/100/200)가 서로 3% 이내로 집중 → 균형점. 여기서 벗어날 때(breakout + momentum) 진입.

### 진입 조건

| 단계 | 조건 |
|------|------|
| ① SMA 집중 | SMA 20/50/100/200 가격 편차 < **3%** |
| ② Breakout | 현재가가 SMA 다발 위/아래로 이탈 |
| ③ 방향 | 모멘텀 > 1% → **LONG** / < -1% → **SHORT** |
| ④ 4h TF 전용 | 15m/1h 시그널은 없음. 4h 바 기준으로만 발생 |

### 청산 조건

| 조건 | 기준 |
|------|------|
| 🔴 Hard SL | 진입가 ±**2.0%** |
| 🟡 Trail Stop | 고점 대비 **0.5%** 하락 (Trail 활성화: +**4.0%** 수익 도달 시) |
| 🟢 Hard TP | ±**8.0%** |

*MA Cluster는 trail 활성화 임계가 높음(4%) — 더 큰 수익 후 보호*

### 성능

백테스트 데이터 별도 (r60 기준 D_mom8에 포함). 라이브에서 활성화되어 있으나 신호 빈도 낮음.

**L/N = 2.1** (leverage 21x / 10 slots)

---

## 리스크 관리 게이트 6개

### 진입 게이트 (신호를 막는 필터)

| # | 이름 | 조건 | 막는 것 |
|---|------|------|---------|
| **G4** | Dir Diversity | 이미 LONG이 과반이면 새 LONG 차단 | 방향 쏠림 방지 |
| **G5** | TBR z-score | Taker Buy 비율 z-score < 0.5 (현재 **OFF**) | 수동 호가 압박 없는 진입 |
| **G6** | BTC Trend Gate | BTC < MA50(15m 기준 12.5h)이면 **D_mom8 LONG만** 차단 | 약세장 롱 필터 |

> ⚠️ **G6는 S_d8(Straddle)에는 적용 안 됨** — 양방향 중립 전략이므로 방향 필터 대상 아님

### 포지션 사이징 조절

| # | 이름 | 조건 | 효과 |
|---|------|------|------|
| **G3** | SoftTight DD Scale | HWM 대비 DD가 2/4/6% 도달 시 | 슬롯당 자본 80%/60%/40%로 축소 |

### 세션 중단 트리거

| # | 이름 | 조건 | 효과 |
|---|------|------|------|
| **G1** | 일일 손실 한도 | 하루 손실 > 잔고 **5%** | 당일 신규 진입 차단 |
| **G2** | 세션 DD Halt | HWM 대비 낙폭 > **30%** | 봇 완전 중단 (r48 P99 기준) |

---

## 사이징 공식

```
슬롯당 자본 = 잔고 / 10
Notional  = 슬롯당 자본 × leverage

VR Squeeze (D_mom8 / S_d8): leverage = 30x → Notional = 잔고/10 × 30  (L/N = 3.0)
MA Cluster:                  leverage = 21x → Notional = 잔고/10 × 21  (L/N = 2.1)

예) 잔고 $1,000 기준:
  D_mom8 슬롯 1개: 마진 $100, Notional $3,000
  MA Cluster 슬롯 1개: 마진 $100, Notional $2,100
  10슬롯 전부 열면: 총 마진 = 잔고 (100% 사용)
```

> SoftTight 발동 시: `슬롯당 자본 × scale (0.8 / 0.6 / 0.4)`

---

## 최근 주요 변경사항

| 날짜 | 변경 | 효과 |
|------|------|------|
| **2026-03-13** | **Straddle G6 버그 수정** (commit `58e0842`) | S_d8이 처음으로 실거래 오픈 가능 |
| **2026-03-13** | **d_trail_activate 0.5%→1.0%** (r69, commit `a061e07`) | OOS 2023/24/25 PF +38%/+23%/+40% |
| 2026-03-13 | TBR gate 재검증 (r71) | trail=1.0% 환경에서도 OFF 유지 (2024/25 악화 확인) |

---

## 현재 라이브 세션 현황

| 항목 | 값 |
|------|-----|
| 잔고 | $1,405.95 |
| HWM | $1,431.04 |
| Session DD | 1.8% |
| WR (8 trades) | 62% |
| 활성 포지션 | 4개 |
| 배포 커밋 | `58e0842` (2026-03-13 15:25) |

---

*Source of truth: `quantbox/strategy/vol_straddle/config.py` · 이 문서는 config 변경 시 동기화*

---

## TradingStrategy Protocol (v14.24.0)

**개념:** 모든 거래 전략이 준수하는 타입 안전 인터페이스. `@runtime_checkable` Protocol.  
**위치:** `quantbox/strategy/protocol.py`  
**테스트:** `tests/test_strategy_protocol.py` (32개)

### 14개 멤버

| 멤버 | 종류 | 목적 |
|------|------|------|
| `n_active` | property | 활성 포지션 수 |
| `config` | property | 전략 설정 객체 |
| `on_bar()` | method | 바 클로즈 시 신호 처리 |
| `update_position()` | method | 포지션 상태 업데이트 |
| `update_market_data()` | method | 15m 시장 데이터 갱신 |
| `update_htf_data()` | method | 고타임프레임(1h/4h) 데이터 갱신 |
| `open_position()` | method | 포지션 오픈 |
| `close_position()` | method | 포지션 청산 |
| `get_positions()` | method | 전체 포지션 조회 |
| `get_position()` | method | 단일 포지션 조회 |
| `has_position()` | method | 포지션 보유 여부 |
| `get_pnl()` | method | 미실현 PnL 조회 |
| `to_state_dict()` | method | 상태 직렬화 (JSON) |
| `from_state_dict()` | method | 상태 복원 (크래시 복구) |

### 사용 예시

```python
from quantbox.strategy.protocol import TradingStrategy

def process_strategy(strategy: TradingStrategy) -> None:
    print(f"Active positions: {strategy.n_active}")
    # isinstance() 지원 (runtime_checkable)
    assert isinstance(strategy, TradingStrategy)
```

### 현재 구현체

| 전략 | 클래스 | Protocol 준수 |
|------|--------|--------------|
| VR Squeeze (D_mom8 + S_d8) | `VolStraddleStrategy` | ✅ v14.24.0 |
| MA Cluster | `VolStraddleStrategy` | ✅ v14.24.0 |
| RubberBand (레거시) | `RubberBandStrategy` | ✅ v14.24.0 |
