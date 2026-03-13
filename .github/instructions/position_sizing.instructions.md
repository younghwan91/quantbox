---
applyTo: "quantbox/execution/**/*.py,quantbox/strategy/**/*.py,quantbox/**/*config*.py"
---
# Position Sizing Skill — QuantBox Slot Margin & Capital Validation

> **목적:** AI 에이전트가 포지션 사이징 계산을 정확히 검증할 수 있도록.
> 잘못된 사이징은 실제 자금 손실로 이어짐. 코드를 먼저 읽고, 추측하지 말 것.

---

## 1. VR Squeeze 프로덕션 사이징 공식

### 1.1 기본 공식 (코드 source: `quantbox/execution/positions.py`)
```python
# dynamic_capital_enabled=False (프로덕션 기본값)
per_slot_capital = balance / max_concurrent  # balance / 10
notional = per_slot_capital * base_leverage  # per_slot × 30
```

| 파라미터 | 값 | 출처 |
|----------|-----|------|
| `base_leverage` | 30x | `config.py` |
| `max_concurrent` | 10 | `config.py` |
| `L/N` (risk per trade) | **3.0** | 30 / 10 |
| `d_trail_activate` | **0.010** (1.0%) | `config.py` (r69에서 0.005→0.010 변경) |
| `d_trail_distance` | 0.002 (0.2%) | `config.py` |
| `d_tp` | 0.20 (**dead param** — trail이 먼저 실행) | `config.py` |

**잔고 $1000일 때 예시:**
- `per_slot_capital = $1000 / 10 = $100`
- `notional = $100 × 30 = $3,000`
- 마진 요구 = `notional / leverage = $3,000 / 30 = $100`

### 1.2 MA Cluster 사이징 (`quantbox/execution/positions.py`)
```python
# signal_meta에 leverage_override 포함 시
leverage = sig_meta['leverage_override']  # 21x (risk parity 설계)
notional = per_slot_capital * leverage   # $100 × 21 = $2,100
```

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| `leverage_override` | 21x | MA cluster: SL=2%, 12/0.01 × 0.02 ≈ 21 |
| `L/N` (MA) | **2.1** | 21 / 10 |

**위험 동등성 (Risk Parity):** 12× SL_VR(1%) ≈ 21× SL_MA(0.571%)... 실제 설계:
```
VR:  30x × 1% SL = 30% max loss per slot
MA:  21x × 2% SL = 42% max loss per slot  ← 다름! 설계 의도 확인 필요
```

---

## 2. SoftTight DD Scaling

계좌 DD가 임계값을 초과하면 자동으로 사이징 축소:

```python
# 코드: quantbox/execution/positions.py
hwm = self._high_water_mark
dd_pct = (hwm - balance) / hwm

if dd_pct >= cfg.soft_eq_dd3:   eq_scale = cfg.soft_eq_scale3  # 0.40
elif dd_pct >= cfg.soft_eq_dd2: eq_scale = cfg.soft_eq_scale2  # 0.60
elif dd_pct >= cfg.soft_eq_dd1: eq_scale = cfg.soft_eq_scale1  # 0.80
else:                            eq_scale = 1.0

per_slot_capital *= eq_scale
```

| DD 임계 | 스케일 | 결과 (잔고 $1000) |
|---------|--------|-------------------|
| < 2%    | 1.0×   | $100/slot         |
| ≥ 2%    | 0.8×   | $80/slot          |
| ≥ 4%    | 0.6×   | $60/slot          |
| ≥ 6%    | 0.4×   | $40/slot          |

**확인 방법:**
```python
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()
print(f'soft_equity_scale={cfg.soft_equity_scale}')
print(f'DD thresholds: {cfg.soft_eq_dd1}, {cfg.soft_eq_dd2}, {cfg.soft_eq_dd3}')
print(f'Scale factors: {cfg.soft_eq_scale1}, {cfg.soft_eq_scale2}, {cfg.soft_eq_scale3}')
```

---

## 3. 마진 충분성 검증 (Incident #3 교훈)

### 3.1 전체 슬롯 사용 시 마진 합계 계산

반드시 `n_active > 0` 상태를 포함해서 검증:

```python
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()

balance = 1000.0  # 실제 잔고로 교체
max_slots = cfg.max_concurrent  # 10

# SoftTight 없다고 가정 (worst case = no DD)
per_slot = balance / max_slots           # $100
notional_per_slot = per_slot * cfg.base_leverage  # $3000
margin_per_slot = notional_per_slot / cfg.base_leverage  # = per_slot = $100

total_margin_10_slots = margin_per_slot * max_slots
print(f'10 slots 전부 열면: total_margin={total_margin_10_slots:.2f}')
print(f'balance={balance:.2f}')
print(f'충분: {total_margin_10_slots <= balance}')

# dynamic_capital_enabled=True일 때 (현재 비활성, 주의)
if cfg.dynamic_capital_enabled:
    print("WARNING: dynamic_capital_enabled=True → 50% cap 주의")
    # n_active=0일 때: per_slot = balance / max(1, 0+1) = balance
    # → 첫 슬롯이 전체 잔고를 독점!
    # → min(per_slot, balance*0.5) cap 있음
    # → 2번째 슬롯: per_slot = balance / max(1, 1+1) = balance/2 = $500
    # → 3번째: balance/3... 슬롯 10개면 $100씩이어야 하는데 처음에 $500 할당됨
```

### 3.2 슬롯 순차 오픈 시뮬레이션

```python
balance = 1000.0
cfg_leverage = 30.0
cfg_max = 10
remaining_balance = balance

for i in range(cfg_max):
    per_slot = remaining_balance / cfg_max  # 항상 balance/10으로 나눠야 함
    margin = per_slot  # leverage cancel
    remaining_balance -= margin
    print(f'Slot {i+1}: margin=${margin:.2f}, remaining=${remaining_balance:.2f}')
    if remaining_balance < 0:
        print(f"ERROR: Margin insufficient at slot {i+1}!")
        break
```

### 3.3 ExposureGuard 영향 (v14.23.0+)

ExposureGuard는 포지션 사이징 공식 자체를 바꾸지 않음. 대신 총 노출도가 상한을 초과하면 **신규 진입을 차단**:

```python
# ExposureGuard 작동 원리 (execution/exposure.py)
cap_notional = balance * max_exposure_multiplier  # 기본: balance × 35.0
current_notional = sum(pos.notional for pos in open_positions)

if current_notional + new_notional > cap_notional:
    return False  # 진입 차단
```

**현재 설정 확인:**
```python
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()
print(f'max_exposure_multiplier={cfg.max_exposure_multiplier}')  # 35.0
# 이론적 최대 노출도: 10슬롯 × 30x = balance의 30배
# ExposureGuard 상한: 35배 (17% 헤드룸)
```

**실거래 영향:**
- 정상 운용 (10슬롯 × 30x = 30배): 35배 cap을 초과하지 않아 영향 없음
- MA Cluster 혼합 시 (21x): 노출도 감소 → 더 안전
- cap 초과 발생 시: 신규 진입만 차단, 기존 포지션은 그대로 유지

---

## 4. Conviction-Weighted Sizing (CW-Tier)

VR squeeze에만 적용. 신호 강도에 따라 notional 조정:

```python
# 코드: quantbox/execution/positions.py
if vs.cw_enabled and vs.cw_multipliers and strategy_type == 'vr_squeeze':
    idx = min(conviction_score, len(vs.cw_multipliers) - 1)
    cw_mult = vs.cw_multipliers[idx]
    notional *= cw_mult
```

**설정 확인:**
```python
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()
print(f'cw_enabled={cfg.cw_enabled}')
print(f'cw_multipliers={cfg.cw_multipliers}')
```

---

## 5. L/N 값 빠른 계산

백테스트/보고서에서 수익률 계산 시 반드시 L/N 명시:

```python
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()

vr_ln = cfg.base_leverage / cfg.max_concurrent
print(f'VR squeeze L/N = {vr_ln:.1f}')  # 3.0

# MA cluster — leverage_override 읽기
# quantbox/strategy/vol_straddle/signals.py에서 실제 값 확인
```

**compound 수익률 계산:**
```python
# 1x compound (compute_metrics() 기본값, backtest 전용)
equity = initial * (1 + pnl_pct)

# 프로덕션 compound (실거래)
equity = initial * (1 + pnl_pct * LN)  # LN = L/N
```

> ⚠️ `compute_metrics()`는 1x compound만 계산. 프로덕션 수익률로 직접 사용 금지.
> 반드시 L/N 적용 버전으로 변환하거나 명시적으로 "1x compound" 라고 표기.

---

## 6. 실거래 수익률 추정 공식

```python
def estimate_production_pnl(
    pf: float,          # Profit Factor (from backtest)
    n_trades: int,      # Number of trades
    win_rate: float,    # Win rate (fraction, e.g. 0.55)
    avg_win: float,     # Average win per trade (fraction)
    avg_loss: float,    # Average loss per trade (fraction, negative)
    ln: float = 3.0,    # L/N ratio (VR=3.0, MA=2.1)
    years: float = 1.0, # Backtest duration in years
) -> dict:
    ev_per_trade = win_rate * avg_win + (1 - win_rate) * avg_loss
    trades_per_year = n_trades / years
    annual_trades = int(trades_per_year)
    
    # 1x compound (what compute_metrics returns)
    r_1x = (1 + ev_per_trade) ** annual_trades - 1
    
    # L/N compound (production)
    r_ln = (1 + ev_per_trade * ln) ** annual_trades - 1
    
    return {
        'ev_per_trade': ev_per_trade,
        'annual_return_1x': r_1x,
        f'annual_return_LN{ln}': r_ln,
        'PF': pf,
    }
```

---

## 7. 슬롯 마진 체크 (API)

```python
import os; from dotenv import load_dotenv; load_dotenv()
from binance.client import Client
c = Client(os.environ['BINANCE_API_KEY'], os.environ['BINANCE_API_SECRET'])

acct = c.futures_account()
total_margin = float(acct['totalInitialMargin'])
available = float(acct['availableBalance'])
total_wallet = float(acct['totalWalletBalance'])
unrealized = float(acct['totalUnrealizedProfit'])

print(f'Wallet Balance:   ${total_wallet:.2f}')
print(f'Unrealized PnL:   ${unrealized:+.2f}')
print(f'Initial Margin:   ${total_margin:.2f}')
print(f'Available:        ${available:.2f}')
print(f'Margin Usage:     {total_margin/total_wallet*100:.1f}%')

# 열린 포지션 수
positions = [p for p in c.futures_position_information() if abs(float(p['positionAmt'])) > 0]
print(f'\nOpen positions: {len(positions)}/10')
for p in positions:
    print(f"  {p['symbol']} {p['positionSide']}: notional=${abs(float(p['notional'])):.0f}")
```

---

## 8. 백테스트에서 사이징 검증

백테스트 스크립트에서 L/N 하드코딩 금지. 반드시 config에서 읽어야 함:

```python
# ✅ 올바른 방법
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()
VR_LN = cfg.base_leverage / cfg.max_concurrent  # 3.0

# ❌ 금지 (하드코딩)
VR_LN = 2.5  # 백테스트와 실거래 불일치 → 수익률 보고 오류
```

**프로덕션 compound 계산 (backtesting.md와 연동):**
```python
# scaling_r38.py portfolio_sim() 사용 후
from quantbox.research.scaling_r38 import collect_trades, portfolio_sim
trades = collect_trades(...)
metrics = portfolio_sim(trades, ln=VR_LN)  # L/N 전달 필수
```
