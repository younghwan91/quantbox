#!/usr/bin/env python3
"""세션 로그 9개 트레이드로 트레일링 스톱 A/B 시뮬레이션.

로그에서 추출한 실제 z-score 궤적을 사용.
파라미터: activate_delta=0.3, buffer=0.1 (구현체와 동일)
"""
import re
import sys
from pathlib import Path

LOG = Path(__file__).parent.parent / 'logs' / 'session_20260215_050732.log'

# ─── 1. Parse log ─────────────────────────────────────────────
log = LOG.read_text()

# Close lines: entry_z, exit_z, reason, pnl, pair info
close_pat = re.compile(
    r'Slot (\d+): Trade closed: PnL=\$([-+\d.]+) \(([-+\d.]+)%\), Reason=(\w+),'
    r' held=([\d.]+)h, z: ([-+\d.]+)→([-+\d.]+).*?costs=\[fee=\$([\d.]+),'
    r' fund=\$([\d.]+)'
)

# Monitor lines: slot, pair, z, entry_z
monitor_pat = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*?Slot (\d+): \[MONITOR\] (\S+)'
    r' z=([-+\d.]+) \(entry=([-+\d.]+)\)'
)

# Build trade list from close lines (in chronological order)
trades = []
for m in close_pat.finditer(log):
    slot = int(m.group(1))
    pnl = float(m.group(2))
    pnl_pct = float(m.group(3))
    reason = m.group(4)
    held_h = float(m.group(5))
    entry_z = float(m.group(6))
    exit_z = float(m.group(7))
    fee = float(m.group(8))
    funding = float(m.group(9))
    trades.append({
        'slot': slot, 'pnl': pnl, 'pnl_pct': pnl_pct,
        'reason': reason, 'held_h': held_h,
        'entry_z': entry_z, 'exit_z': exit_z,
        'fee': fee, 'funding': funding,
    })

# Build monitor z-score sequences per slot lifecycle
# Group by (slot, entry_z) to distinguish different trades on same slot
monitor_data = {}
for m in monitor_pat.finditer(log):
    ts, slot, pair, z, entry_z = (
        m.group(1), int(m.group(2)), m.group(3),
        float(m.group(4)), float(m.group(5))
    )
    key = (slot, entry_z)
    if key not in monitor_data:
        monitor_data[key] = {'pair': pair, 'zpath': []}
    monitor_data[key]['zpath'].append(z)

# Match monitor data to trades
for t in trades:
    key = (t['slot'], t['entry_z'])
    if key in monitor_data:
        t['pair'] = monitor_data[key]['pair']
        t['zpath'] = monitor_data[key]['zpath']
    else:
        t['pair'] = '???'
        t['zpath'] = []

# ─── 2. Trailing stop simulation ──────────────────────────────
ACTIVATE_DELTA = 0.3
BUFFER = 0.1

def simulate_trailing_stop(entry_z: float, zpath: list[float],
                           actual_reason: str, actual_exit_z: float) -> dict:
    """Simulate trailing stop on a z-score path.
    
    Returns dict with trailing_triggered, trigger_idx, trigger_z.
    """
    sign = 1 if entry_z > 0 else -1  # Long spread = +, Short spread = -
    best_z = entry_z
    trailing_active = False
    
    for i, z in enumerate(zpath):
        # Update best_z (closest to 0 = most improvement)
        if sign > 0:  # Long: z should decrease
            if z < best_z:
                best_z = z
        else:  # Short: z should increase (toward 0)
            if z > best_z:
                best_z = z
        
        # Check activation
        delta = abs(entry_z) - abs(best_z)  # how much z improved toward 0
        if delta >= ACTIVATE_DELTA:
            trailing_active = True
        
        # Check trailing stop trigger
        if trailing_active:
            if sign > 0:  # Long: trigger if z rises back past entry - buffer
                threshold = entry_z - BUFFER
                if z >= threshold:
                    return {
                        'triggered': True, 'trigger_idx': i,
                        'trigger_z': z, 'best_z': best_z,
                        'trailing_active': True, 'delta_at_best': delta
                    }
            else:  # Short: trigger if z drops back past entry + buffer
                threshold = entry_z + BUFFER
                if z <= threshold:
                    return {
                        'triggered': True, 'trigger_idx': i,
                        'trigger_z': z, 'best_z': best_z,
                        'trailing_active': True, 'delta_at_best': delta
                    }
    
    return {
        'triggered': False, 'trigger_idx': -1,
        'trigger_z': None, 'best_z': best_z,
        'trailing_active': trailing_active,
        'delta_at_best': abs(entry_z) - abs(best_z)
    }

# ─── 3. Run simulation ────────────────────────────────────────
print(f"{'='*78}")
print(f"{'TRAILING STOP SIMULATION ON 9 LIVE TRADES':^78}")
print(f"{'='*78}")
print(f"  activate_delta={ACTIVATE_DELTA}, buffer={BUFFER}")
print(f"  Trades: {len(trades)}")
print()

total_pnl_baseline = 0.0
total_pnl_trailing = 0.0
winners_killed = 0
losers_saved = 0

for i, t in enumerate(trades):
    entry_z = t['entry_z']
    zpath = t['zpath']
    actual_pnl = t['pnl']
    reason = t['reason']
    pair = t['pair']
    
    sim = simulate_trailing_stop(entry_z, zpath, reason, t['exit_z'])
    
    total_pnl_baseline += actual_pnl
    
    # Determine trailing stop outcome
    if sim['triggered']:
        # Trailing stop would have fired before actual exit
        # Estimate PnL at trailing stop point
        # The z at trigger is near entry ± buffer, so PnL ≈ ~0 (minus costs)
        # More precisely: PnL is proportional to (entry_z - trigger_z) / entry_z
        # Rough estimate: nearly breakeven minus costs
        costs = t['fee'] + t['funding']
        trailing_pnl = -costs  # conservative: breakeven minus costs
        
        if reason == 'stop_loss':
            # SAVED a loser
            losers_saved += 1
            saved_amount = abs(actual_pnl) - costs  # how much we saved
            tag = f"✅ SAVED ${saved_amount:+.2f}"
        elif reason == 'mean_reversion':
            # KILLED a winner - bad!
            winners_killed += 1
            lost_profit = actual_pnl + costs  # profit we missed
            tag = f"❌ KILLED (missed ${lost_profit:.2f})"
        else:
            tag = "⚠️ MODIFIED"
        
        total_pnl_trailing += trailing_pnl
    else:
        # Trailing stop didn't fire → same as baseline
        total_pnl_trailing += actual_pnl
        tag = "— no change"
    
    sign_str = "Long" if entry_z > 0 else "Short"
    z_readings = f"{len(zpath)} readings"
    best_str = f"best_z={sim['best_z']:+.3f}" if sim['best_z'] is not None else ""
    delta_str = f"Δ={sim['delta_at_best']:.3f}"
    
    print(f"  Trade {i+1}: {pair:<30} {sign_str:5} entry_z={entry_z:+.3f}")
    print(f"    Baseline: reason={reason:<16} PnL=${actual_pnl:+.2f}")
    print(f"    Monitor:  {z_readings}, {best_str}, {delta_str}, active={sim['trailing_active']}")
    if sim['triggered']:
        print(f"    Trail:    TRIGGERED at reading #{sim['trigger_idx']+1}, z={sim['trigger_z']:+.3f}")
    print(f"    Result:   {tag}")
    print()

# ─── 4. Summary ───────────────────────────────────────────────
print(f"{'─'*78}")
print(f"{'SUMMARY':^78}")
print(f"{'─'*78}")

win_trades = [t for t in trades if t['reason'] == 'mean_reversion']
lose_trades = [t for t in trades if t['reason'] == 'stop_loss']

print(f"  Total trades: {len(trades)} (W={len(win_trades)}, L={len(lose_trades)})")
print()
print(f"  {'Metric':<30} {'Baseline':>12} {'Trailing':>12} {'Diff':>10}")
print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
print(f"  {'Total PnL':<30} ${total_pnl_baseline:>+10.2f} ${total_pnl_trailing:>+10.2f} ${total_pnl_trailing - total_pnl_baseline:>+9.2f}")

base_wr = len(win_trades) / len(trades) * 100
# With trailing: losers_saved become ~breakeven (not wins)
trail_wins = len(win_trades) - winners_killed
trail_losses = len(lose_trades) - losers_saved
trail_neutral = losers_saved + winners_killed  # breakeven-ish
trail_wr = trail_wins / len(trades) * 100

print(f"  {'Win Rate':<30} {base_wr:>11.1f}% {trail_wr:>11.1f}% {trail_wr - base_wr:>+9.1f}%")
print(f"  {'Losers Saved':<30} {'':>12} {losers_saved:>12}")
print(f"  {'Winners Killed':<30} {'':>12} {winners_killed:>12}")
print()

# Verdict
if total_pnl_trailing > total_pnl_baseline and winners_killed == 0:
    verdict = "🟢 TRAILING STOP WINS — safe to deploy"
elif total_pnl_trailing > total_pnl_baseline:
    verdict = f"🟡 NET POSITIVE but killed {winners_killed} winner(s) — review carefully"
else:
    verdict = "🔴 TRAILING STOP LOSES — do NOT deploy"

print(f"  VERDICT: {verdict}")
print(f"{'─'*78}")
