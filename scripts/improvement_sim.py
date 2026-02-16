#!/usr/bin/env python3
"""세션 로그 기반 6대 개선안 A/B 시뮬레이션.

모든 세션 로그(3개)를 파싱하여 21개 거래를 재구성하고,
아래 6개 개선안 적용 전/후 PnL을 비교한다.

[Actual]  실제 결과 (기존 로직 그대로)
[New]     개선안 전체 적용:
  1. 세션 블랙리스트 (손실 페어 재진입 차단)
  2. entry_z 임계값 상향 (2.0 → 2.2)
  3. 방향 다양화 (동일 방향 max 2 슬롯)
  4. 아시아 세션 게이트 (00-08 UTC, max 1 진입)
  5. trail_z 확대 (0.8 → 1.2)
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / 'logs'
LOG_FILES = sorted(LOGS_DIR.glob('session_2026021*.log'))

# ── Config ──
NEW_ENTRY_Z = 2.2
NEW_TRAIL_Z = 1.2
OLD_TRAIL_Z = 0.8
ACTIVATE_DELTA = 0.3
MAX_SAME_DIR = 2
ASIA_START = 0   # UTC hour
ASIA_END = 8     # UTC hour
ASIA_MAX_ENTRY = 1

# ── Data structures ──
@dataclass
class Trade:
    session: str
    slot: int
    sym1: str
    sym2: str
    signal: int          # 1=Long spread, -1=Short spread
    entry_z: float
    exit_z: float
    reason: str
    pnl: float
    pnl_pct: float
    held_h: float
    fee: float
    funding: float
    entry_time: datetime
    close_time: datetime
    zpath: list[float] = field(default_factory=list)
    zpath_times: list[datetime] = field(default_factory=list)


def parse_all_sessions() -> list[Trade]:
    """모든 세션 로그를 파싱하여 Trade 리스트 반환."""
    trades: list[Trade] = []
    
    # Patterns
    close_pat = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*?'
        r'Slot (\d+): Trade closed: PnL=\$([-+\d.]+) \(([-+\d.]+)%\), Reason=(\w+),'
        r' held=([\d.]+)h, z: ([-+\d.]+)→([-+\d.]+).*?costs=\[fee=\$([\d.]+),'
        r' fund=\$([\d.]+)'
    )
    entry_pat = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*?'
        r'Slot (\d+): Position opened: signal=(-?\d+), z-score=([-+\d.]+)'
    )
    monitor_pat = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*?'
        r'Slot (\d+): \[MONITOR\] (\S+) z=([-+\d.]+) \(entry=([-+\d.]+)\)'
    )
    
    for log_path in LOG_FILES:
        session = log_path.stem
        log_text = log_path.read_text()
        
        # 1. Parse entries (slot → list of entry events)
        entries: dict[int, list[dict]] = {}
        for m in entry_pat.finditer(log_text):
            ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            slot = int(m.group(2))
            signal = int(m.group(3))
            z = float(m.group(4))
            if slot not in entries:
                entries[slot] = []
            entries[slot].append({'time': ts, 'signal': signal, 'z': z})
        
        # 2. Parse monitor lines
        monitors: list[dict] = []
        for m in monitor_pat.finditer(log_text):
            ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            slot = int(m.group(2))
            pair = m.group(3)
            z = float(m.group(4))
            entry_z = float(m.group(5))
            monitors.append({
                'time': ts, 'slot': slot, 'pair': pair,
                'z': z, 'entry_z': entry_z,
            })
        
        # 3. Parse closes
        for m in close_pat.finditer(log_text):
            close_ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            slot = int(m.group(2))
            pnl = float(m.group(3))
            pnl_pct = float(m.group(4))
            reason = m.group(5)
            held_h = float(m.group(6))
            entry_z_val = float(m.group(7))
            exit_z_val = float(m.group(8))
            fee = float(m.group(9))
            funding = float(m.group(10))
            
            # Match to entry event (closest entry before this close)
            signal = 1 if entry_z_val < 0 else -1
            entry_time = close_ts  # fallback
            if slot in entries:
                # Find the entry with matching z-score
                best_match = None
                for e in entries[slot]:
                    if e['time'] < close_ts and abs(e['z'] - entry_z_val) < 0.01:
                        best_match = e
                if best_match:
                    entry_time = best_match['time']
                    signal = best_match['signal']
                    entries[slot].remove(best_match)
                elif entries[slot]:
                    # Fallback: just take the closest entry before close
                    candidates = [e for e in entries[slot] if e['time'] < close_ts]
                    if candidates:
                        best_match = candidates[-1]
                        entry_time = best_match['time']
                        signal = best_match['signal']
                        entries[slot].remove(best_match)
            
            # Find pair from monitors
            pair = '???/???'
            for mon in monitors:
                if mon['slot'] == slot and abs(mon['entry_z'] - entry_z_val) < 0.01:
                    if entry_time <= mon['time'] <= close_ts:
                        pair = mon['pair']
                        break
            
            parts = pair.split('/')
            sym1 = parts[0] if len(parts) >= 1 else '???'
            sym2 = parts[1] if len(parts) >= 2 else '???'
            
            # Build z-path from monitors
            zpath = []
            zpth_times = []
            for mon in monitors:
                if (mon['slot'] == slot
                    and abs(mon['entry_z'] - entry_z_val) < 0.01
                    and entry_time <= mon['time'] <= close_ts):
                    zpath.append(mon['z'])
                    zpth_times.append(mon['time'])
            
            trades.append(Trade(
                session=session, slot=slot,
                sym1=sym1, sym2=sym2,
                signal=signal, entry_z=entry_z_val,
                exit_z=exit_z_val, reason=reason,
                pnl=pnl, pnl_pct=pnl_pct,
                held_h=held_h, fee=fee, funding=funding,
                entry_time=entry_time, close_time=close_ts,
                zpath=zpath, zpath_times=zpth_times,
            ))
    
    # Sort by entry time
    trades.sort(key=lambda t: t.entry_time)
    return trades


def sim_trailing_stop(entry_z: float, zpath: list[float],
                      trail_z: float, activate_delta: float = 0.3) -> dict:
    """트레일링 스탑 시뮬레이션. best_z ± trail_z 기반."""
    signal = 1 if entry_z < 0 else -1
    best_z = entry_z
    trailing_active = False

    for i, z in enumerate(zpath):
        if signal == 1:
            if z > best_z:
                best_z = z
        else:
            if z < best_z:
                best_z = z

        delta = abs(best_z - entry_z)
        if delta >= activate_delta:
            trailing_active = True

        if trailing_active:
            if signal == 1:
                threshold = best_z - trail_z
                if z <= threshold:
                    return {'triggered': True, 'idx': i, 'z': z, 'best_z': best_z}
            else:
                threshold = best_z + trail_z
                if z >= threshold:
                    return {'triggered': True, 'idx': i, 'z': z, 'best_z': best_z}

    return {'triggered': False, 'idx': -1, 'z': None, 'best_z': best_z}


def estimate_pnl_at_z(trade: Trade, exit_z: float) -> float:
    """z_score 시점의 추정 PnL. 실제 PnL과 z 변화량의 비율로 보간."""
    actual_z_change = trade.exit_z - trade.entry_z
    if abs(actual_z_change) < 1e-6:
        return 0.0
    sim_z_change = exit_z - trade.entry_z
    ratio = sim_z_change / actual_z_change
    gross_pnl = trade.pnl + trade.fee + trade.funding
    return gross_pnl * ratio - trade.fee - trade.funding


def run_simulation():
    """전체 시뮬레이션 실행."""
    trades = parse_all_sessions()
    
    print(f"\n{'='*90}")
    print(f"{'6-MEASURE IMPROVEMENT LOG-BASED SIMULATION':^90}")
    print(f"{'='*90}")
    print(f"  Sessions: {len(LOG_FILES)} ({', '.join(f.stem for f in LOG_FILES)})")
    print(f"  Total trades parsed: {len(trades)}")
    print()
    print(f"  New rules:")
    print(f"    1. Session blacklist (손실 페어 재진입 차단)")
    print(f"    2. entry_z ≥ {NEW_ENTRY_Z} (기존 2.0)")
    print(f"    3. Direction max {MAX_SAME_DIR} same-direction slots")
    print(f"    4. Asia gate: {ASIA_START:02d}-{ASIA_END:02d} UTC, max {ASIA_MAX_ENTRY} entries")
    print(f"    5. trail_z = {NEW_TRAIL_Z} (기존 {OLD_TRAIL_Z})")
    print()
    
    # ── Simulation state ──
    actual_total = 0.0
    new_total = 0.0
    
    # Per-session state
    session_blacklist: set[str] = set()
    current_session = None
    
    # Active slots tracking for direction gate
    active_slots: dict[int, dict] = {}  # slot_id -> {'signal': int, 'sym1': str, 'sym2': str, 'entry_time': datetime}
    
    # Asia session entry tracking
    asia_entries_this_window: list[datetime] = []
    
    blocked_entry_z = 0
    blocked_direction = 0
    blocked_asia = 0
    blocked_blacklist = 0
    trade_results = []
    
    for i, t in enumerate(trades):
        # Session change → reset blacklist & active slots
        if t.session != current_session:
            current_session = t.session
            session_blacklist.clear()
            active_slots.clear()
            asia_entries_this_window.clear()
            print(f"  {'─'*86}")
            print(f"  📁 Session: {current_session}")
            print(f"  {'─'*86}")
        
        actual_total += t.pnl
        dir_label = "Long" if t.signal == 1 else "Short"
        entry_hour = t.entry_time.hour
        
        # ── Close any active slots that should have closed by now ──
        closed_slots = []
        for sid, sinfo in active_slots.items():
            # Find if there's a trade close for this slot before current entry
            for prev_t in trades[:i]:
                if (prev_t.slot == sid and prev_t.session == t.session
                    and prev_t.entry_time >= sinfo['entry_time']
                    and prev_t.close_time <= t.entry_time):
                    closed_slots.append(sid)
                    break
        for sid in set(closed_slots):
            del active_slots[sid]
        
        # ── Gate checks ──
        blocked = False
        block_reason = ""
        
        # 1. Session blacklist
        if t.sym1 in session_blacklist or t.sym2 in session_blacklist:
            blocked = True
            bl_sym = t.sym1 if t.sym1 in session_blacklist else t.sym2
            block_reason = f"🚫 BL({bl_sym})"
            blocked_blacklist += 1
        
        # 2. Entry z threshold
        if not blocked and abs(t.entry_z) < NEW_ENTRY_Z:
            blocked = True
            block_reason = f"🚫 |z|={abs(t.entry_z):.2f} < {NEW_ENTRY_Z}"
            blocked_entry_z += 1
        
        # 3. Direction diversity
        if not blocked:
            same_dir_count = sum(
                1 for s in active_slots.values()
                if s['signal'] == t.signal
            )
            if same_dir_count >= MAX_SAME_DIR:
                blocked = True
                block_reason = f"🚫 Dir={dir_label}({same_dir_count}/{MAX_SAME_DIR})"
                blocked_direction += 1
        
        # 4. Asia session gate
        if not blocked and ASIA_START <= entry_hour < ASIA_END:
            # Count asia entries still active
            asia_active = sum(
                1 for s in active_slots.values()
                if ASIA_START <= s['entry_time'].hour < ASIA_END
            )
            if asia_active >= ASIA_MAX_ENTRY:
                blocked = True
                block_reason = f"🚫 Asia({asia_active}/{ASIA_MAX_ENTRY})"
                blocked_asia += 1
        
        if blocked:
            new_pnl = 0.0
            tag = block_reason
        else:
            # 5. Trail_z change simulation
            sim_result = sim_trailing_stop(t.entry_z, t.zpath, trail_z=NEW_TRAIL_Z)
            sim_old = sim_trailing_stop(t.entry_z, t.zpath, trail_z=OLD_TRAIL_Z)
            
            if t.reason == 'trailing_stop':
                # This trade was closed by old trailing stop
                # Check if new trail_z would have prevented it
                if not sim_result['triggered']:
                    # New trail_z would NOT have triggered → trade continues to its natural end
                    # We don't know exact outcome, but the old trailing caused a loss
                    # Estimate: trade would have continued and hit stop_loss or mean_reversion
                    # Conservative: assume it still loses but less (trail didn't trigger prematurely)
                    new_pnl = t.pnl  # Conservative: same PnL
                    tag = f"🔄 trail_z={NEW_TRAIL_Z} would NOT trigger (old did)"
                else:
                    # New trail_z also triggers, but possibly at a different point
                    new_pnl = estimate_pnl_at_z(t, sim_result['z'])
                    tag = f"🔄 trail@z={sim_result['z']:+.3f} (was {t.exit_z:+.3f})"
            else:
                # Non-trailing trades: check if new trailing would trigger BEFORE original exit
                if sim_result['triggered']:
                    # Count bars: if trailing triggers first, estimate PnL
                    trail_t = t.zpath_times[sim_result['idx']] if sim_result['idx'] < len(t.zpath_times) else t.close_time
                    if trail_t < t.close_time:
                        new_pnl = estimate_pnl_at_z(t, sim_result['z'])
                        tag = f"🔄 trail@{sim_result['z']:+.3f} (before {t.reason})"
                    else:
                        new_pnl = t.pnl
                        tag = "— same"
                else:
                    new_pnl = t.pnl
                    tag = "— same"
            
            # Add to active slots
            active_slots[t.slot] = {
                'signal': t.signal,
                'sym1': t.sym1,
                'sym2': t.sym2,
                'entry_time': t.entry_time,
            }
            
            # Session blacklist: add if loss
            if new_pnl < 0:
                session_blacklist.add(t.sym1)
                session_blacklist.add(t.sym2)
                tag += f" → BL+({t.sym1},{t.sym2})"
        
        new_total += new_pnl
        
        trade_results.append({
            'trade': t, 'actual_pnl': t.pnl, 'new_pnl': new_pnl,
            'blocked': blocked, 'tag': tag,
        })
        
        status = "✅" if new_pnl > 0 else ("⛔" if blocked else "❌")
        print(
            f"  #{i+1:2d} {t.sym1}/{t.sym2:<28} {dir_label:5} "
            f"entry_z={t.entry_z:+.3f} @{entry_hour:02d}h"
        )
        print(f"       {status} Actual=${t.pnl:+.2f}  →  New=${new_pnl:+.2f}  {tag}")
        print()
    
    # ── Summary ──
    print(f"\n{'='*90}")
    print(f"{'SUMMARY':^90}")
    print(f"{'='*90}\n")
    
    total_trades = len(trades)
    blocked_total = blocked_entry_z + blocked_direction + blocked_asia + blocked_blacklist
    executed = total_trades - blocked_total
    
    actual_wins = sum(1 for t in trades if t.pnl > 0)
    new_wins = sum(1 for r in trade_results if r['new_pnl'] > 0)
    new_losses = sum(1 for r in trade_results if r['new_pnl'] < 0)
    
    print(f"  {'Metric':<40} {'Actual':>12} {'New Logic':>12}")
    print(f"  {'─'*40} {'─'*12} {'─'*12}")
    print(f"  {'Total PnL ($)':<40} {actual_total:>+12.2f} {new_total:>+12.2f}")
    print(f"  {'Δ PnL ($)':<40} {'—':>12} {new_total - actual_total:>+12.2f}")
    print(f"  {'Trades':<40} {total_trades:>12} {executed:>12}")
    print(f"  {'Wins':<40} {actual_wins:>12} {new_wins:>12}")
    print(f"  {'Losses':<40} {total_trades - actual_wins:>12} {new_losses:>12}")
    
    if new_wins + new_losses > 0:
        wr = new_wins / (new_wins + new_losses) * 100
    else:
        wr = 0
    actual_wr = actual_wins / total_trades * 100 if total_trades > 0 else 0
    print(f"  {'Win Rate':<40} {actual_wr:>11.1f}% {wr:>11.1f}%")
    
    # Profit factor
    actual_gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    actual_gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    new_gross_win = sum(r['new_pnl'] for r in trade_results if r['new_pnl'] > 0)
    new_gross_loss = abs(sum(r['new_pnl'] for r in trade_results if r['new_pnl'] < 0))
    
    actual_pf = actual_gross_win / actual_gross_loss if actual_gross_loss > 0 else float('inf')
    new_pf = new_gross_win / new_gross_loss if new_gross_loss > 0 else float('inf')
    print(f"  {'Profit Factor':<40} {actual_pf:>12.2f} {new_pf:>12.2f}")
    
    print(f"\n  {'Blocked Trades Breakdown':}")
    print(f"    Entry z < {NEW_ENTRY_Z}:          {blocked_entry_z}")
    print(f"    Direction diversity:    {blocked_direction}")
    print(f"    Asia session gate:     {blocked_asia}")
    print(f"    Session blacklist:     {blocked_blacklist}")
    print(f"    Total blocked:         {blocked_total}")
    
    delta = new_total - actual_total
    print(f"\n  {'─'*64}")
    if new_total > 0:
        print(f"  🟢 흑자 전환 성공! ${actual_total:+.2f} → ${new_total:+.2f} (Δ ${delta:+.2f})")
    elif delta > 0:
        print(f"  🟡 개선됨 (아직 적자): ${actual_total:+.2f} → ${new_total:+.2f} (Δ ${delta:+.2f})")
    else:
        print(f"  🔴 악화됨: ${actual_total:+.2f} → ${new_total:+.2f} (Δ ${delta:+.2f})")
    print(f"  {'─'*64}")
    
    # Per-session breakdown
    print(f"\n  Per-Session Breakdown:")
    sessions = {}
    for r in trade_results:
        s = r['trade'].session
        if s not in sessions:
            sessions[s] = {'actual': 0, 'new': 0, 'trades': 0, 'blocked': 0}
        sessions[s]['actual'] += r['actual_pnl']
        sessions[s]['new'] += r['new_pnl']
        sessions[s]['trades'] += 1
        if r['blocked']:
            sessions[s]['blocked'] += 1
    
    for s, d in sessions.items():
        print(
            f"    {s}: Actual=${d['actual']:+.2f} → New=${d['new']:+.2f} "
            f"(Δ ${d['new']-d['actual']:+.2f}) "
            f"[{d['trades']} trades, {d['blocked']} blocked]"
        )
    print()


if __name__ == '__main__':
    run_simulation()
