---
applyTo: "quantbox/scalper.py,quantbox/execution/**/*.py"
---
# Live Trading Skill — QuantBox Session Management

> **목적:** AI 에이전트가 라이브 세션을 안전하게 시작, 재시작, 종료할 수 있도록.
> 모든 절차는 실제 자금 손실 및 버그 사례에서 도출됨. 순서 임의 변경 금지.

---

## 1. 세션 시작 전 체크리스트 (Pre-Flight)

### 1.1 Config 검증 (반드시 코드를 읽어서 확인)
```bash
python3 -c "
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()
print(f'd_sl={cfg.d_sl} d_tp={cfg.d_tp}')
print(f'd_trail_activate={cfg.d_trail_activate} d_trail_distance={cfg.d_trail_distance}')
print(f'base_leverage={cfg.base_leverage} max_concurrent={cfg.max_concurrent}')
print(f'L/N = {cfg.base_leverage / cfg.max_concurrent:.2f}')
print(f'hedge_mode={cfg.hedge_mode}')
print(f'dynamic_capital_enabled={cfg.dynamic_capital_enabled}')
"
```

**VR Squeeze 프로덕션 기준값 (v14.17.0+):**
- `d_sl=0.01` (1%), `d_tp=0.20` (20% dead cap — trail exits first)
- `d_trail_activate=0.01` (1.0%), `d_trail_distance=0.002` (0.2%)
- `base_leverage=30.0`, `max_concurrent=10` → L/N=3.0
- `s_squeeze_dur=3`, `s_vr_threshold=0.57`
- `hedge_mode=True`, `dynamic_capital_enabled=False`

> ⚠️ `.env` 값이 `config.py` 기본값보다 항상 우선. 반드시 `.env`도 함께 확인.

### 1.2 API 연결 확인
```bash
python3 -c "
import os; from dotenv import load_dotenv; load_dotenv()
from binance.client import Client
c = Client(os.environ['BINANCE_API_KEY'], os.environ['BINANCE_API_SECRET'])
bal = [b for b in c.futures_account_balance() if b['asset'] == 'USDT'][0]
mode = c.futures_get_position_mode()
print(f'Balance: \${float(bal[\"availableBalance\"]):.2f}')
print(f'Hedge mode: {mode[\"dualSidePosition\"]}')  # must be True for vol_straddle
"
```

### 1.3 State 파일 건강 확인
```bash
python3 -c "
import json
state = json.load(open('data/state/scalper_state.json'))
for sym, pos in state.get('positions', {}).items():
    api_sym = pos.get('symbol', sym)
    if ':' in api_sym:
        print(f'CORRUPT: {sym} symbol={api_sym} (has colon!)')
    else:
        print(f'OK: {sym} symbol={api_sym} side={pos.get(\"position_side\")}')
"
```

부패 발견 시: `debugging.instructions.md` Section 1.1 Origin A 참조.

---

## 2. 기존 세션 종료

### 2.1 기존 세션 PID 확인
```bash
ps aux | grep "scalper" | grep -v grep | awk '{print $2, $11, $12, $13, $14}'
```

### 2.2 안전 종료 (Ctrl+C)
```bash
tmux send-keys -t quantbox C-c
sleep 3
# 상태 저장 확인
tail -5 logs/bot/vol_straddle/$(ls -t logs/bot/vol_straddle/ | head -1)
```
`_save_state` 호출 로그 확인 후 종료.

### 2.3 강제 종료 (응급)
```bash
kill <PID>  # 숫자 PID 직접 지정 필수
```
> ⚠️ 강제 종료 후 state 파일 손상 가능성 있음. 재시작 전 1.3 체크 필수.

---

## 3. 신규 세션 시작

```bash
cd /home/user/Documents/git/quantbox
tmux send-keys -t quantbox "python3 -m quantbox.scalper --strategy vol_straddle" Enter
```

**시작 후 즉시 확인 (30초 내):**
```bash
# 최신 로그에서 포지션 복원 확인
LOG=$(ls -t logs/bot/vol_straddle/ | head -1)
grep -E "Restor|Position opened|hedge:" logs/bot/vol_straddle/$LOG | head -10
```

**정상 로그 예시:**
```
[SUIUSDT] Position opened: LONG TF=15m SL=1.0% TP=20.0% ... (hedge:LONG)
Restored 4 positions, 1 trade records
```

**이상 징후:**
- `hedge:` 없이 포지션 오픈 → `hedge_mode=False`인지 확인
- `APIError(code=-4059)` → Binance 계정 hedge mode 미설정 (`_ensure_position_mode` 확인)

---

## 4. 재시작 중 기존 포지션이 있을 때

### 4.1 기존 포지션의 SL/TP 상태 확인
```bash
python3 -c "
import json
state = json.load(open('data/state/scalper_state.json'))
for sym, pos in state.get('positions', {}).items():
    sl = pos.get('server_sl_order_id') or 'NONE'
    tp = pos.get('server_tp_order_id') or 'NONE'
    bars = pos.get('bars_held', 0)
    max_hold = pos.get('max_hold', 192)
    trail = pos.get('trail_active', False)
    peak = pos.get('peak_gain', 0) * 100
    print(f'{sym}: bars={bars}/{max_hold} trail={trail} peak={peak:+.3f}% SL={sl} TP={tp}')
"
```

### 4.2 trail_active=True 포지션 점검

trail exit 조건:
```
trail_stop = peak_gain - d_trail_distance
exit if current_pnl < trail_stop
```

현재 PnL 확인:
```bash
python3 -c "
import os; from dotenv import load_dotenv; load_dotenv()
from binance.client import Client
c = Client(os.environ['BINANCE_API_KEY'], os.environ['BINANCE_API_SECRET'])
for p in [p for p in c.futures_position_information() if abs(float(p['positionAmt'])) > 0]:
    e, m, a = float(p['entryPrice']), float(p['markPrice']), float(p['positionAmt'])
    print(f\"{p['symbol']} {p['positionSide']}: pnl={(m-e)/e*100 if a>0 else (e-m)/e*100:+.3f}%\")
"
```

### 4.3 SL=NONE 포지션이 있을 때

신세션 시작 후 최초 1분 cycle에서 bracket repair가 자동 실행됨.
다음 로그를 반드시 확인:
```bash
grep "Bracket repair\|SERVER STOP\|FAILED" logs/bot/vol_straddle/$(ls -t logs/bot/vol_straddle/ | head -1) | head -10
```

**성공:** `Bracket repair complete: SL=OK TP=OK`
**실패:** `-1022` → state 파일 `symbol` 필드 점검 (1.3 절차)

---

## 5. 포지션 exit 예측

| Exit 종류 | 조건 | 실행 |
|-----------|------|------|
| Trail stop | `current_pnl < peak_gain - trail_distance` | 다음 바 클로즈에 MARKET |
| Hard SL | LONG: `price ≤ entry × (1 - sl_pct)` / SHORT: `price ≥ entry × (1 + sl_pct)` | 서버 STOP 주문 |
| Hard TP | LONG: `price ≥ entry × (1 + tp_pct)` / SHORT: 반대 | 서버 LIMIT 주문 |
| Max hold | `bars_held ≥ max_hold` (기본 192 bars = 48h) | 다음 바에 강제 청산 |

**strategy_type별 SL/trail 파라미터 (state의 `strategy_type` 필드 먼저 읽을 것):**

| strategy_type | sl_pct | trail_activate | trail_distance |
|--------------|--------|---------------|---------------|
| `vr_squeeze` | `cfg.d_sl` = 0.01 (1%) | `cfg.d_trail_activate` = 0.01 | `cfg.d_trail_distance` = 0.002 |
| `straddle` | `cfg.s_sl` = 0.015 (1.5%) | `cfg.s_trail_activate` = 0.015 | `cfg.s_trail_distance` = 0.001 |
| `ma_cluster` | `cfg.ma_cluster_sl` = 0.02 (2%) | `cfg.ma_cluster_trail_activate` = 0.02 | `cfg.ma_cluster_trail_distance` = 0.003 |

> ⚠️ SL 가격을 계산할 때 strategy_type 무시하고 d_sl=1%로 계산하면 틀린다 (Incident 2026-03-13).
> **가장 확실한 방법:** state의 `server_sl_order_id`로 `/fapi/v1/openAlgoOrders`에서 `triggerPrice` 직접 읽기.

---

## 6. 긴급 수동 청산

세션이 다운됐거나 client-side stop이 동작하지 않을 때:

```python
import os; from dotenv import load_dotenv; load_dotenv()
from binance.client import Client
c = Client(os.environ['BINANCE_API_KEY'], os.environ['BINANCE_API_SECRET'])

positions = [p for p in c.futures_position_information() if abs(float(p['positionAmt'])) > 0]
for p in positions:
    sym = p['symbol']
    amt = float(p['positionAmt'])
    side = 'SELL' if amt > 0 else 'BUY'
    pos_side = p['positionSide']
    qty = abs(amt)
    print(f"Closing {sym} {pos_side} qty={qty:.4f}")
    c.futures_create_order(
        symbol=sym, side=side, type='MARKET',
        quantity=qty, positionSide=pos_side,
        reduceOnly=False,
    )
```

> ⚠️ Hedge mode에서 `reduceOnly=True`와 `positionSide` 동시 사용 불가.

---

## 7. 로그 위치 & 진단

```
logs/bot/vol_straddle/scalper_live_YYYYMMDD_HHMMSS.log  # 라이브
logs/bot/vol_straddle/scalper_paper_YYYYMMDD_HHMMSS.log # 페이퍼
data/state/scalper_state.json                           # 포지션 상태
data/state/scalper_state_paper.json                     # 페이퍼 상태
```

**진단 grep 패턴:**
```bash
LOG=logs/bot/vol_straddle/$(ls -t logs/bot/vol_straddle/ | head -1)
grep "ERROR\|WARNING\|APIError"   $LOG | tail -20   # 에러
grep "Bracket repair"             $LOG | tail -10   # SL/TP 상태
grep "trail_stop\|trailing stop"  $LOG | tail -10   # trail exit
grep "CLOSED\|trade"              $LOG | tail -10   # 체결 내역
grep "Summary"                    $LOG | tail -5    # 잔고/WR 요약
```

---

## 8. Git 흐름 (배포 시)

| 단계 | 명령 |
|------|------|
| feature 완료 | `git checkout develop && git merge --squash feature/xxx` |
| research 완료 | `git checkout develop && git merge --no-ff research/xxx` |
| 릴리즈 | `git checkout main && git merge --no-ff develop -m "release: vX.Y.Z ..."` |
| 태그 | `git tag -a vX.Y.Z -m "..."` |
| 배포 | `git push origin main && git push origin vX.Y.Z` |

**main 머지 = 즉시 라이브 배포.** 머지 후 세션 재시작 필수.

---

## 9. DD Halt 설정

DD halt 임계값: `config.py:max_drawdown_pct` + `.env:MAX_DRAWDOWN_PCT` 둘 다 변경 필수. 적정값은 백테스트 session DD P99 기준 (r48 research 참조).
