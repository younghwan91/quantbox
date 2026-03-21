---
name: quantbox-live-ops-agent
description: |
  Autonomous live trading operations agent for QuantBox. Connects to the running tmux session,
  captures real-time logs, queries Binance USDT-M Futures API for ground-truth account state,
  cross-references local state/journal for discrepancies, performs deep strategy performance
  analysis (win/loss causality, exit reason breakdown, regime detection), identifies errors/bugs
  in logs and code, then builds a prioritized remediation plan and executes fixes.
  Triggers: live ops, trading audit, session audit, performance analysis, strategy review,
           discrepancy check, binance verify, account audit, PnL analysis, live debug,
           position audit, trade forensics, desk check, risk check, ops agent
license: MIT
---

# QuantBox Live Operations Agent

> **Primary entry point:** Run `/live-ops` in Claude Code (`.claude/commands/live-ops.md`).
> This reference doc provides the detailed procedures the slash command executes.

## Purpose

Full-stack autonomous diagnostics for a live algorithmic trading session. This skill operates
as a Wall Street crypto quant desk operations agent — it assumes nothing, verifies everything
against exchange ground truth, and produces actionable intelligence.

**Phases:**
1. **Session Capture** — tmux pane + log file collection
2. **Exchange Ground Truth** — Binance API account/position/order snapshot
3. **Discrepancy Detection** — local state vs exchange reality
4. **Strategy Performance Analysis** — journal-driven win/loss causality
5. **Error & Bug Detection** — log pattern + code trace
6. **Remediation Plan** — prioritized fix list with severity/effort
7. **Execute Fixes** — apply safe fixes, flag risky ones for human approval

> **Iron Rule:** Every number comes from logs, state files, journal DB, or Binance API.
> Never guess, estimate, or assume any value.

---

## Phase 1: Session Capture

### 1.1 Tmux Session Discovery

```bash
# Find all quantbox-related tmux sessions
tmux list-sessions 2>/dev/null | grep -i "quant\|scalper\|bot"

# If no session found, check systemd service
systemctl --user status quantbox 2>/dev/null || true
```

### 1.2 Capture Live Terminal Output

```bash
# Capture last 200 lines from the live trading pane
# Try common session/window names
for sess in quantbox scalper bot; do
  tmux capture-pane -t "$sess" -p -S -200 2>/dev/null && break
done
```

**What to extract:**
- Current cycle number and timestamp
- Last signal scan results
- Any visible errors or warnings
- Active position summaries printed by the bot

### 1.3 Log File Collection

```bash
# Identify the active log file (most recently modified)
LOG_DIR="/home/user/Documents/git/quantbox/logs/bot/vol_straddle"
LIVE_LOG=$(ls -t "$LOG_DIR"/scalper_live_*.log 2>/dev/null | head -1)
echo "Active log: $LIVE_LOG"
echo "Age: $(( ($(date +%s) - $(stat -c %Y "$LIVE_LOG")) ))s since last write"
echo "Size: $(stat -c %s "$LIVE_LOG") bytes"
echo "Lines: $(wc -l < "$LIVE_LOG")"

# Extract key sections from tail
echo "=== LAST 3 CYCLE SUMMARIES ==="
grep -E "Cycle #|═══|Summary|balance|HWM|DD=" "$LIVE_LOG" | tail -20

echo "=== LAST 5 SIGNAL SCANS ==="
grep -E "Squeeze monitor:|MA Cluster monitor:|G6 BTC" "$LIVE_LOG" | tail -15

echo "=== LAST 5 ENTRY/EXIT EVENTS ==="
grep -E "ENTRY|EXIT|CLOSE|OPEN|fill|bracket" "$LIVE_LOG" | tail -15
```

### 1.4 Session Health Verdict

| Check | Healthy | Stale | Dead |
|-------|---------|-------|------|
| Log age | < 900s (15 min) | 900-1800s | > 1800s |
| Cycle incrementing | Yes | Stuck same # | N/A |
| Tmux pane responsive | Shows output | Frozen | No session |
| Python process alive | `pgrep -f scalper` | - | No PID |

```bash
# Quick process check
pgrep -af "quantbox.scalper" || echo "NO SCALPER PROCESS RUNNING"
```

---

## Phase 2: Exchange Ground Truth (Binance API)

### 2.1 Account Balance Snapshot

```python
import json, os, time, hmac, hashlib, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE = "https://fapi.binance.com"
API_KEY = os.environ["BINANCE_API_KEY"]
API_SECRET = os.environ["BINANCE_API_SECRET"]

def signed_get(path, params=None):
    params = params or {}
    params["timestamp"] = int(time.time() * 1000)
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    r = requests.get(f"{BASE}{path}?{qs}&signature={sig}",
                     headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
    r.raise_for_status()
    return r.json()

# Account balance
acct = signed_get("/fapi/v2/account")
wallet = float(acct["totalWalletBalance"])
available = float(acct["availableBalance"])
margin = float(acct["totalInitialMargin"])
upnl = float(acct["totalUnrealizedProfit"])
maint_margin = float(acct["totalMaintMargin"])

print(f"Wallet Balance:    ${wallet:.2f}")
print(f"Available Balance: ${available:.2f}")
print(f"Initial Margin:    ${margin:.2f}")
print(f"Maint Margin:      ${maint_margin:.2f}")
print(f"Unrealized PnL:    ${upnl:+.2f}")
print(f"Margin Ratio:      {maint_margin / wallet * 100:.2f}%" if wallet > 0 else "N/A")
```

### 2.2 Exchange Positions (Ground Truth)

```python
# All positions with non-zero amount
positions_raw = signed_get("/fapi/v2/positionRisk")
exchange_positions = {}
for p in positions_raw:
    amt = float(p["positionAmt"])
    if amt == 0:
        continue
    sym = p["symbol"]
    side = p["positionSide"]  # LONG or SHORT (hedge mode)
    key = f"{sym}:{side}"
    exchange_positions[key] = {
        "symbol": sym,
        "side": side,
        "amount": amt,
        "entry_price": float(p["entryPrice"]),
        "mark_price": float(p["markPrice"]),
        "unrealized_pnl": float(p["unRealizedProfit"]),
        "leverage": int(p["leverage"]),
        "notional": abs(float(p["notional"])),
        "liquidation_price": float(p["liquidationPrice"]) if p["liquidationPrice"] != "0" else None,
    }
    pnl_pct = (float(p["markPrice"]) - float(p["entryPrice"])) / float(p["entryPrice"]) * 100
    if side == "SHORT":
        pnl_pct = -pnl_pct
    print(f"  {key}: amt={amt} entry=${float(p['entryPrice']):.4f} "
          f"mark=${float(p['markPrice']):.4f} uPnL=${float(p['unRealizedProfit']):+.2f} "
          f"({pnl_pct:+.2f}%) lev={p['leverage']}x liq=${p['liquidationPrice']}")

print(f"\nTotal exchange positions: {len(exchange_positions)}")
```

### 2.3 Open Orders (Regular + Algo)

```python
# Regular open orders
regular_orders = signed_get("/fapi/v1/openOrders")
print(f"\nRegular open orders: {len(regular_orders)}")
for o in regular_orders:
    print(f"  {o['symbol']} {o['side']} {o['type']} qty={o['origQty']} "
          f"stop={o.get('stopPrice', 'N/A')} status={o['status']}")

# Algo open orders (SL/TP trailing stops)
algo_orders = signed_get("/fapi/v1/openAlgoOrders")
if isinstance(algo_orders, list):
    print(f"\nAlgo open orders: {len(algo_orders)}")
    for o in algo_orders:
        print(f"  algoId={o['algoId']} {o['symbol']} {o['side']} {o['orderType']} "
              f"trigger={o.get('triggerPrice', 'N/A')} qty={o['quantity']} "
              f"status={o['algoStatus']}")
else:
    print(f"\nAlgo orders response: {algo_orders}")
```

### 2.4 Recent Income (Realized PnL from Exchange)

```python
# Last 24h realized PnL from Binance income endpoint
end_ms = int(time.time() * 1000)
start_ms = end_ms - 24 * 3600 * 1000

income = signed_get("/fapi/v1/income", {
    "incomeType": "REALIZED_PNL",
    "startTime": start_ms,
    "endTime": end_ms,
    "limit": 1000,
})

exchange_pnl_24h = sum(float(r["income"]) for r in income)
exchange_trades_24h = len(income)
print(f"\nExchange 24h realized PnL: ${exchange_pnl_24h:+.2f} ({exchange_trades_24h} fills)")

# Also get funding fees
funding = signed_get("/fapi/v1/income", {
    "incomeType": "FUNDING_FEE",
    "startTime": start_ms,
    "endTime": end_ms,
    "limit": 1000,
})
funding_total = sum(float(r["income"]) for r in funding)
print(f"Exchange 24h funding fees: ${funding_total:+.2f} ({len(funding)} payments)")

# Commission fees
commission = signed_get("/fapi/v1/income", {
    "incomeType": "COMMISSION",
    "startTime": start_ms,
    "endTime": end_ms,
    "limit": 1000,
})
commission_total = sum(float(r["income"]) for r in commission)
print(f"Exchange 24h commissions: ${commission_total:+.2f} ({len(commission)} fills)")
```

---

## Phase 3: Discrepancy Detection

### 3.1 Position Discrepancy (State vs Exchange)

```python
import json
from pathlib import Path

# Load local state
state = json.loads(Path("data/state/scalper_state.json").read_text())
local_positions = state.get("positions", {})

# Compare keys
local_keys = set(local_positions.keys())
exchange_keys = set(exchange_positions.keys())

forward_ghosts = local_keys - exchange_keys   # In state, NOT on exchange
reverse_ghosts = exchange_keys - local_keys   # On exchange, NOT in state
matched = local_keys & exchange_keys

print("=== POSITION DISCREPANCY REPORT ===")
print(f"Local positions:    {len(local_keys)}")
print(f"Exchange positions: {len(exchange_keys)}")
print(f"Matched:            {len(matched)}")

if forward_ghosts:
    print(f"\nFORWARD GHOSTS (state has, exchange doesn't): {forward_ghosts}")
    for k in forward_ghosts:
        pos = local_positions[k]
        print(f"  {k}: entry={pos.get('entry_price')} qty=? strategy={pos.get('strategy_type')}")
    print("  ACTION: Verify position was closed. If so, update state. If not, EMERGENCY.")

if reverse_ghosts:
    print(f"\nREVERSE GHOSTS (exchange has, state doesn't): {reverse_ghosts}")
    for k in reverse_ghosts:
        ep = exchange_positions[k]
        print(f"  {k}: amt={ep['amount']} entry={ep['entry_price']} uPnL=${ep['unrealized_pnl']:+.2f}")
    print("  ACTION: Position opened outside bot OR state update failed. Investigate immediately.")

if not forward_ghosts and not reverse_ghosts:
    print("\nNo ghost positions detected.")
```

### 3.2 Quantity Mismatch (Matched Positions)

```python
for k in matched:
    local = local_positions[k]
    exch = exchange_positions[k]
    local_qty = float(local.get("qty", 0) or 0)
    exch_qty = abs(exch["amount"])

    if abs(local_qty - exch_qty) > 1e-8:
        print(f"\nQTY MISMATCH: {k}")
        print(f"  Local qty:    {local_qty}")
        print(f"  Exchange qty: {exch_qty}")
        print(f"  Delta:        {exch_qty - local_qty}")

    # Entry price drift check (should be identical)
    local_entry = float(local.get("entry_price", 0))
    exch_entry = exch["entry_price"]
    if abs(local_entry - exch_entry) / exch_entry > 0.001:  # > 0.1% drift
        print(f"\nENTRY PRICE MISMATCH: {k}")
        print(f"  Local entry:    {local_entry}")
        print(f"  Exchange entry: {exch_entry}")
```

### 3.3 Balance Discrepancy

```python
local_balance = state.get("initial_balance", 0)
hwm = state.get("high_water_mark", 0)
local_dd = (1 - local_balance / hwm) * 100 if hwm > 0 else 0

# Exchange balance (wallet balance = realized balance without uPnL)
balance_delta = abs(wallet - local_balance)

print(f"\n=== BALANCE DISCREPANCY ===")
print(f"Local balance:    ${local_balance:.2f}")
print(f"Exchange wallet:  ${wallet:.2f}")
print(f"Delta:            ${balance_delta:.2f}")
print(f"Local HWM:        ${hwm:.2f}")
print(f"Local DD:         {local_dd:.2f}%")

if balance_delta > 1.0:
    print(f"  WARNING: Balance drift > $1.00 — possible missed trade recording or manual intervention")
```

### 3.4 SL/TP Order Integrity

```python
# Build map of algo orders by ID
algo_by_id = {}
if isinstance(algo_orders, list):
    for o in algo_orders:
        algo_by_id[str(o["algoId"])] = o

issues = []
for pos_key, pos in local_positions.items():
    sl_id = str(pos.get("server_sl_order_id", ""))
    tp_id = str(pos.get("server_tp_order_id", ""))

    # Check SL exists on exchange
    if sl_id and sl_id not in algo_by_id:
        issues.append(f"MISSING SL: {pos_key} sl_id={sl_id} not in openAlgoOrders")
    elif not sl_id and pos.get("entry_price"):
        issues.append(f"NO SL RECORDED: {pos_key} has entry but no server_sl_order_id")

    # Check for orphaned algo orders (order exists but position doesn't)
    if sl_id in algo_by_id:
        sl_algo = algo_by_id[sl_id]
        if sl_algo.get("algoStatus") not in ("WORKING", ""):
            issues.append(f"SL NOT WORKING: {pos_key} sl status={sl_algo.get('algoStatus')}")

# Orphaned algo orders (no matching position)
state_algo_ids = set()
for pos in local_positions.values():
    if pos.get("server_sl_order_id"):
        state_algo_ids.add(str(pos["server_sl_order_id"]))
    if pos.get("server_tp_order_id"):
        state_algo_ids.add(str(pos["server_tp_order_id"]))

for oid, o in algo_by_id.items():
    if oid not in state_algo_ids:
        issues.append(f"ORPHAN ALGO ORDER: algoId={oid} sym={o['symbol']} type={o['orderType']} — no matching position in state")

if issues:
    print(f"\n=== SL/TP INTEGRITY ISSUES ({len(issues)}) ===")
    for i in issues:
        print(f"  {i}")
else:
    print("\nSL/TP integrity: OK")
```

---

## Phase 4: Strategy Performance Analysis

### 4.1 Journal-Driven Performance (SQLite)

```python
import sqlite3
from pathlib import Path
from collections import defaultdict

DB = Path("data/journal.sqlite")
if not DB.exists():
    print("WARNING: journal.sqlite not found — falling back to state trade_history")
else:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    # Overall stats
    r = conn.execute("""
        SELECT COUNT(*) as n,
               SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(pnl_usd), 2) as total_pnl,
               ROUND(AVG(pnl_pct), 4) as avg_pnl_pct,
               ROUND(AVG(bars_held), 1) as avg_bars,
               SUM(CASE WHEN pnl_usd > 0 THEN pnl_usd ELSE 0 END) as gross_win,
               ABS(SUM(CASE WHEN pnl_usd < 0 THEN pnl_usd ELSE 0 END)) as gross_loss
        FROM trades
        WHERE ABS(pnl_pct) < 10 AND symbol NOT LIKE '%:%'
    """).fetchone()

    n = r["n"] or 0
    wins = r["wins"] or 0
    gp = r["gross_win"] or 0
    gl = r["gross_loss"] or 0
    pf = gp / gl if gl > 0 else float("inf")

    print(f"\n=== OVERALL JOURNAL PERFORMANCE ===")
    print(f"Trades: {n}  |  WR: {wins/n*100:.1f}%  |  PF: {pf:.2f}  |  Net: ${r['total_pnl']:+.2f}")
    print(f"Avg PnL: {r['avg_pnl_pct']*100:.2f}%  |  Avg Hold: {r['avg_bars']} bars")
```

### 4.2 Per-Strategy Breakdown

```python
    rows = conn.execute("""
        SELECT strategy_type,
               COUNT(*) as n,
               SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(pnl_usd), 2) as total_pnl,
               ROUND(AVG(pnl_pct), 4) as avg_pnl_pct,
               ROUND(AVG(bars_held), 1) as avg_bars,
               SUM(CASE WHEN pnl_usd > 0 THEN pnl_usd ELSE 0 END) as gp,
               ABS(SUM(CASE WHEN pnl_usd < 0 THEN pnl_usd ELSE 0 END)) as gl
        FROM trades
        WHERE ABS(pnl_pct) < 10 AND symbol NOT LIKE '%:%'
        GROUP BY strategy_type
    """).fetchall()

    print(f"\n=== PER-STRATEGY PERFORMANCE ===")
    for r in rows:
        n = r["n"]
        wr = r["wins"] / n * 100 if n else 0
        pf = r["gp"] / r["gl"] if r["gl"] > 0 else float("inf")
        print(f"  {r['strategy_type']:12s}  N={n:3d}  WR={wr:5.1f}%  PF={pf:.2f}  "
              f"Net=${r['total_pnl']:+8.2f}  AvgBars={r['avg_bars']}")
```

### 4.3 Exit Reason Analysis (Win/Loss Causality)

```python
    rows = conn.execute("""
        SELECT exit_reason, strategy_type,
               COUNT(*) as n,
               SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(pnl_usd), 2) as total_pnl,
               ROUND(AVG(pnl_pct), 4) as avg_pnl_pct,
               ROUND(AVG(bars_held), 1) as avg_bars
        FROM trades
        WHERE ABS(pnl_pct) < 10 AND symbol NOT LIKE '%:%'
        GROUP BY exit_reason, strategy_type
        ORDER BY total_pnl ASC
    """).fetchall()

    print(f"\n=== EXIT REASON BREAKDOWN (sorted by PnL, worst first) ===")
    for r in rows:
        n = r["n"]
        wr = r["wins"] / n * 100 if n else 0
        print(f"  {r['exit_reason']:20s} [{r['strategy_type']:10s}]  N={n:3d}  WR={wr:5.1f}%  "
              f"Net=${r['total_pnl']:+8.2f}  AvgPnL={r['avg_pnl_pct']*100:+.2f}%  Bars={r['avg_bars']}")
```

**Causality interpretation guide:**

| Exit Reason | If High Loss | Likely Cause | Investigate |
|-------------|-------------|--------------|-------------|
| `sl_hit` | Normal | SL doing its job | Check if SL% is too wide |
| `trail_stop` | Unusual | Trail locking too little | Check trail_distance, trail_activate |
| `max_hold` | Bad sign | Positions drifting sideways | Check max_hold bars, market regime |
| `signal_exit` | Depends | Signal reversal | Check signal quality in current regime |
| `straddle_rollback` | Expected | Failed straddle pair | Check straddle success rate |
| `dd_scaling_exit` | Emergency | DD protection firing | Check DD thresholds |

### 4.4 Losing Streak & Regime Analysis

```python
    # Consecutive loss detection
    trades = conn.execute("""
        SELECT id, symbol, strategy_type, direction, pnl_usd, exit_reason,
               exit_time, bars_held
        FROM trades
        WHERE ABS(pnl_pct) < 10 AND symbol NOT LIKE '%:%'
        ORDER BY exit_time DESC
        LIMIT 50
    """).fetchall()

    # Find current streak
    streak = 0
    streak_type = None  # "win" or "loss"
    for t in trades:
        if streak_type is None:
            streak_type = "win" if t["pnl_usd"] > 0 else "loss"
        if (t["pnl_usd"] > 0) == (streak_type == "win"):
            streak += 1
        else:
            break

    print(f"\n=== STREAK ANALYSIS ===")
    print(f"Current streak: {streak} consecutive {'wins' if streak_type == 'win' else 'LOSSES'}")

    if streak_type == "loss" and streak >= 3:
        print(f"  WARNING: {streak} consecutive losses — review market regime")
        # Show the losing trades
        for t in trades[:streak]:
            print(f"    {t['exit_time'][:16]} {t['symbol']:12s} {t['strategy_type']:10s} "
                  f"${t['pnl_usd']:+.2f} reason={t['exit_reason']} bars={t['bars_held']}")

    # Per-symbol P&L ranking (identify bleeding symbols)
    sym_rows = conn.execute("""
        SELECT symbol, COUNT(*) as n, ROUND(SUM(pnl_usd), 2) as total_pnl,
               ROUND(100.0 * SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as wr
        FROM trades
        WHERE ABS(pnl_pct) < 10 AND symbol NOT LIKE '%:%'
              AND exit_time >= datetime('now', '-7 days')
        GROUP BY symbol
        ORDER BY total_pnl ASC
        LIMIT 10
    """).fetchall()

    print(f"\n=== WORST PERFORMING SYMBOLS (7d) ===")
    for r in sym_rows:
        tag = "BLEEDING" if r["total_pnl"] < -5 else ""
        print(f"  {r['symbol']:12s}  N={r['n']:2d}  WR={r['wr']:5.1f}%  Net=${r['total_pnl']:+8.2f}  {tag}")
```

### 4.5 Time-of-Day Edge Decay

```python
    rows = conn.execute("""
        SELECT CAST(strftime('%H', exit_time) AS INTEGER) as hour,
               COUNT(*) as n,
               ROUND(SUM(pnl_usd), 2) as total_pnl,
               ROUND(AVG(pnl_pct), 4) as avg_pnl
        FROM trades
        WHERE ABS(pnl_pct) < 10 AND symbol NOT LIKE '%:%'
        GROUP BY hour ORDER BY hour
    """).fetchall()

    print(f"\n=== HOURLY P&L HEATMAP (UTC) ===")
    for r in rows:
        bar = ("+" if r["total_pnl"] >= 0 else "-") * min(int(abs(r["total_pnl"]) / 2), 30)
        print(f"  {r['hour']:02d}:00  N={r['n']:3d}  ${r['total_pnl']:+8.2f}  {bar}")
```

---

## Phase 5: Error & Bug Detection

### 5.1 Log Error Extraction

```bash
LOG_DIR="/home/user/Documents/git/quantbox/logs/bot/vol_straddle"
LIVE_LOG=$(ls -t "$LOG_DIR"/scalper_live_*.log 2>/dev/null | head -1)

echo "=== ERRORS & WARNINGS (last 50) ==="
grep -E "ERROR|CRITICAL|FAIL|Exception|Traceback" "$LIVE_LOG" | tail -50

echo ""
echo "=== API ERROR CODES ==="
grep -oE "\-[0-9]{4}" "$LIVE_LOG" | sort | uniq -c | sort -rn | head -20

echo ""
echo "=== BRACKET ISSUES ==="
grep -E "bracket repair|SL.*CANCEL|orphan|ghost|MISSING" "$LIVE_LOG" | tail -20

echo ""
echo "=== RECONCILIATION EVENTS ==="
grep -iE "reconcil|recon|sync|ghost|counterpart" "$LIVE_LOG" | tail -20

echo ""
echo "=== RATE LIMIT / CONNECTIVITY ==="
grep -iE "rate.?limit|timeout|connection|retry|429|418" "$LIVE_LOG" | tail -10
```

### 5.2 Error Code Classification

| Code | Severity | Meaning | Root Cause | Fix |
|------|----------|---------|------------|-----|
| `-1022` | CRITICAL | Invalid signature | Colon in symbol field (compound key leak) | Fix state: strip `:LONG`/`:SHORT` from `symbol` field |
| `-2021` | HIGH | Order would trigger immediately | `activationPrice` below current mark price | Set `activationPrice=entry×(1±trail_activate)` — future price OK (Binance keeps in NEW). Ensure `callbackRate≥0.5%` |
| `-2027` | MEDIUM | Notional too large | Position size × leverage exceeds limit | Reduce qty or check leverage calc |
| `-2028` | HIGH | Insufficient margin | Leverage changed in close path (Rule #1) | Verify leverage guard |
| `-4120` | HIGH | Order type not supported | Trailing stop on regular API | Must use Algo API endpoint |
| `-2019` | MEDIUM | Margin insufficient | DD scaling not reducing size enough | Check slot sizing formula |
| `-1015` | LOW | Too many orders | Rate limiting | Add order placement cooldown |
| `-4131` | MEDIUM | Algo order not found | Stale algo order ID in state | Re-sync SL/TP from exchange |

### 5.3 Traceback Analysis

```bash
# Extract full Python tracebacks
LOG_DIR="/home/user/Documents/git/quantbox/logs/bot/vol_straddle"
LIVE_LOG=$(ls -t "$LOG_DIR"/scalper_live_*.log 2>/dev/null | head -1)

# Find traceback blocks (Traceback line + subsequent indented lines)
awk '/Traceback \(most recent call last\)/{p=1} p{print} /^[^ ]/ && p && !/Traceback/{p=0}' \
    "$LIVE_LOG" | tail -100
```

**For each traceback, trace to source:**
1. Identify the exception type and message
2. Find the file:line in the traceback
3. Read that code section
4. Check if it matches any known incident rule (#1-#8)
5. If novel, document as potential new incident rule

### 5.4 State File Integrity

```python
import json
from pathlib import Path

state = json.loads(Path("data/state/scalper_state.json").read_text())
issues = []

# Check version
if state.get("version") != "2.0.0":
    issues.append(f"VERSION: expected 2.0.0, got {state.get('version')}")

# Check checksum
stored_checksum = state.get("_checksum")
if stored_checksum:
    import hashlib
    state_copy = {k: v for k, v in state.items() if k != "_checksum"}
    computed = hashlib.sha256(json.dumps(state_copy, sort_keys=True, default=str).encode()).hexdigest()
    if stored_checksum != computed:
        issues.append("CHECKSUM MISMATCH: state file may be corrupted")

# Check position fields
for pos_key, pos in state.get("positions", {}).items():
    sym = pos.get("symbol", "")
    if ":" in sym:
        issues.append(f"CORRUPT SYMBOL: {pos_key} symbol='{sym}' has colon (Rule #1022)")
    if pos.get("position_side") not in ("LONG", "SHORT", None):
        issues.append(f"INVALID SIDE: {pos_key} position_side='{pos.get('position_side')}'")
    if pos.get("entry_price") and not pos.get("server_sl_order_id"):
        issues.append(f"NO SL: {pos_key} has entry but no server_sl_order_id")

# Check balance sanity
bal = state.get("initial_balance", 0)
hwm = state.get("high_water_mark", 0)
if hwm > 0 and bal > hwm * 1.01:
    issues.append(f"BALANCE > HWM: balance=${bal:.2f} > hwm=${hwm:.2f} (should never happen)")
if bal < 0:
    issues.append(f"NEGATIVE BALANCE: ${bal:.2f}")

print(f"State integrity: {len(issues)} issues")
for i in issues:
    print(f"  {i}")
```

---

## Phase 6: Remediation Plan

### 6.1 Issue Severity Classification

After collecting all findings from Phases 1-5, classify each issue:

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| P0 CRITICAL | Active capital risk | Immediate | Ghost position, missing SL, balance mismatch > $10 |
| P1 HIGH | Will cause loss if not fixed | This cycle | API errors repeating, state corruption, orphan orders |
| P2 MEDIUM | Degraded performance | Today | Losing streak > 5, bleeding symbol, stale log |
| P3 LOW | Monitoring/hygiene | This week | Minor balance drift, slow log growth |

### 6.2 Remediation Plan Template

Use emoji indicators for quick visual parsing:
- Session: 🟢 HEALTHY / 🟡 STALE / 🔴 DEAD
- Priority: 🚨 P0 / 🔴 P1 / 🟠 P2 / 🟡 P3
- Strategy: 🟢 profitable / 🟡 marginal / 🔴 bleeding
- Checks: ✅ pass / ❌ fail / ⚠️ warning
- Fixes: 🟢 SAFE / 🟡 CONFIRM / 🔴 FORBIDDEN

```markdown
# 📊 Live Ops Report — {date} {time} UTC

## Executive Summary

| Metric | Value |
|--------|-------|
| Session | 🟢/🟡/🔴 {status} |
| Positions | {N} local / {M} exchange / {ghosts} 👻 ghosts |
| Balance | `${local}` local / `${exchange}` exchange (Δ `${delta}`) |
| 24h PnL | `${realized}` + `${funding}` + `${commission}` = **`${net}` net** |
| Errors | {count} in current log |

## Discrepancy Matrix

| Check | Status | Detail |
|-------|--------|--------|
| Position match | ✅/❌ | ... |
| Forward ghosts | ✅/❌ | ... |
| Reverse ghosts | ✅/❌ | ... |
| Balance delta | ✅/⚠️ | ... |
| SL/TP integrity | ✅/❌ | ... |
| State corruption | ✅/❌ | ... |

## 🚨 P0 — Immediate Action Required
1. ❌ [ISSUE]: {description}
   - Evidence: {log line / API response / state field}
   - Impact: {what happens if unfixed}
   - Fix: {exact command or code change}
   - Verify: {how to confirm fix worked}

## 🔴 P1 — Fix This Cycle
...

## 🟠 P2 — Fix Today
...

## 🟡 P3 — Track This Week
...

## 📈 Strategy Recommendations

### {strategy} — 🟢 KEEP / 🟡 TUNE / 🔴 PAUSE
- **Stats:** {N} trades, {WR}% WR, PF {pf}, net `${pnl}`
- **Key exits:** {dominant exit reason}
- **Action:** {recommendation}

## 🔧 Fix Actions

| # | Issue | Priority | Class | Action |
|---|-------|----------|-------|--------|
| 1 | {desc} | 🚨/🔴/🟠/🟡 | 🟢/🟡/🔴 | {command} |
```

### 6.3 Automated Fix Decision Tree

```
Is the fix SAFE (no capital risk)?
├─ YES: State cleanup, orphan order cancel, log rotation
│   └─ Execute automatically, report result
├─ MAYBE: Config param tweak, SL re-placement
│   └─ Show exact change, ask human to confirm
└─ NO: Emergency close, leverage change, strategy disable
    └─ NEVER auto-execute. Present plan, wait for human
```

---

## Phase 7: Execute Fixes

### 7.1 Safe Auto-Fixes (No Confirmation Needed)

```python
# Fix 1: Clean colon from symbol fields in state
import json
from pathlib import Path

state_path = Path("data/state/scalper_state.json")
state = json.loads(state_path.read_text())
fixed = []

for key, pos in state.get("positions", {}).items():
    sym = pos.get("symbol", "")
    if ":" in sym:
        clean_sym = sym.split(":")[0]
        side = sym.split(":")[1] if ":" in sym else None
        pos["symbol"] = clean_sym
        if side and pos.get("position_side") is None:
            pos["position_side"] = side
        fixed.append(f"{key}: symbol '{sym}' → '{clean_sym}'")

if fixed:
    # Backup first
    import shutil
    shutil.copy(state_path, state_path.with_suffix(".json.pre_fix_backup"))
    state_path.write_text(json.dumps(state, indent=2))
    print(f"Fixed {len(fixed)} symbol fields:")
    for f in fixed:
        print(f"  {f}")
```

### 7.2 Confirmed Fixes (Human Approval Required)

```python
# Fix: Cancel orphaned algo orders
# REQUIRES HUMAN CONFIRMATION — these are live exchange orders
orphan_ids = []  # populated from Phase 3.4

print(f"\nOrphaned algo orders to cancel: {len(orphan_ids)}")
for oid in orphan_ids:
    print(f"  Will cancel algoId={oid}")

# UNCOMMENT AFTER HUMAN APPROVAL:
# for oid in orphan_ids:
#     result = signed_get("/fapi/v1/algo/futures/cancelOrder", {"algoId": oid})
#     print(f"  Cancelled {oid}: {result}")
```

### 7.3 Code Bug Fixes

For bugs found in Phase 5, the agent should:

1. **Identify the exact file:line** from the traceback
2. **Read the surrounding code** to understand context
3. **Check against incident rules #1-#8** for known patterns
4. **Draft a fix** with minimal scope (no refactoring)
5. **Run tests** to verify: `uv run pytest tests/ -v`
6. **Present the diff** for human review before committing

```bash
# After applying code fix, verify
PYTHONPATH=. python3 -m pytest tests/ -v
PYTHONPATH=. python3 -m mypy
PYTHONPATH=. python3 -m ruff check quantbox/ --fix
```

---

## Quick Run: Full Diagnostic One-Liner

```bash
cd /home/user/Documents/git/quantbox && \
echo "=== SESSION ===" && \
tmux capture-pane -t quantbox -p 2>/dev/null | tail -10 && \
echo "=== PROCESS ===" && \
pgrep -af "quantbox.scalper" && \
echo "=== LOG AGE ===" && \
LOG=$(ls -t logs/bot/vol_straddle/scalper_live_*.log 2>/dev/null | head -1) && \
echo "$(($(date +%s) - $(stat -c %Y "$LOG")))s ago" && \
echo "=== ERRORS ===" && \
grep -c "ERROR\|CRITICAL" "$LOG" && \
echo "=== STATE ===" && \
python3 -c "
import json; s=json.load(open('data/state/scalper_state.json'))
print(f'Balance: \${s[\"initial_balance\"]:.2f}  HWM: \${s[\"high_water_mark\"]:.2f}  '
      f'Positions: {len(s[\"positions\"])}  Cycle: {s[\"cycle_count\"]}')
" && \
echo "=== ACCOUNT ===" && \
PYTHONPATH=. python3 scripts/check_account.py && \
echo "=== JOURNAL ===" && \
PYTHONPATH=. python3 scripts/journal.py 2>/dev/null
```

---

## Agent Orchestration Notes

When invoked as a Claude Code agent skill, execute phases in this order:

1. **Parallel Phase 1+2**: Capture tmux/logs AND query Binance API simultaneously
2. **Sequential Phase 3**: Compare results from 1+2 for discrepancies
3. **Parallel Phase 4+5**: Run performance analysis AND error detection simultaneously
4. **Sequential Phase 6**: Build remediation plan from all findings
5. **Phase 7**: Execute safe fixes, present risky fixes for approval

**Agent delegation:**
- Use `oh-my-claudecode:qa-tester` for tmux session interaction
- Use `oh-my-claudecode:analyst` for performance causality analysis
- Use `oh-my-claudecode:debugger` for traceback root-cause analysis
- Use `oh-my-claudecode:tracer` for discrepancy investigation

---

## When to Use This Skill

- **Morning desk check**: Full Phase 1-6 diagnostic before market opens
- **Post-deploy verification**: After code push, verify session restarted correctly
- **Drawdown investigation**: When DD exceeds warning threshold
- **Performance review**: Weekly strategy performance audit
- **Incident response**: When alerts fire or positions look wrong
- **Pre-weekend check**: Verify all SLs in place before low-liquidity hours

## When NOT to Use This Skill

- For backtest analysis → use `quantbox-trade-event-sim`
- For code review before commit → use `quantbox-incident-rules`
- For ghost position recovery only → use `quantbox-reconciliation-ghost`
- For signal development → use `quantbox-signal-development`
