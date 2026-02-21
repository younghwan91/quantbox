#!/usr/bin/env python3
"""
Alpha Search Round 2 — Entry/Exit Parameter Refinement
=======================================================
Round 1 showed that alternative signal generators are alpha-dead.
v12.2's edge comes from its FILTER combination, not the signal itself.

This round keeps v12.2's signal engine and filters, but tests
entry/exit parameter variations that could improve WR AND R:R:

  A: v12.2 Baseline (current production)
  B: Higher Confidence (conf≥3) — stricter entry → fewer bad trades
  C: Wider Target (0.50 retrace vs 0.382) — bigger wins per trade
  D: Faster Stop Decay (0.5 vs 0.3) — cut losers faster
  E: Shorter Hold (12 vs 20 bars) — remove tail losses
  F: Vol Gate (2.0× vol surge required) — only high-conviction entries
  G: Combo Best (conf≥3 + 0.50 target + 0.5 decay + vol gate)
  H: Asymmetric Exit (target 0.50, stop 0.50) — force R:R > 1 by design

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/alpha_search_r2.py
"""

import sys
import time
import json
import subprocess
import numba
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from quantbox.core.settings import get_settings
from quantbox.strategy.rubberband import (
    RubberBandConfig,
    compute_signals_v9,
    apply_mtf_confirmation,
    apply_mtf_leverage,
    compute_atr_pct,
)

CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def _find_cache(symbol: str, interval: str, days: int) -> Path | None:
    pattern = f'{symbol}_{interval}_{days}d_*.npz'
    matches = sorted(CACHE_DIR.glob(pattern), reverse=True)
    return matches[0] if matches else None


def load_15m(symbol: str, days: int) -> dict | None:
    cache = _find_cache(symbol, '15m', days)
    if cache is None or not cache.exists():
        return None
    d = np.load(cache)
    return {k: d[k] for k in ['close', 'high', 'low', 'volume', 'taker_buy']}


def load_4h(symbol: str, days: int) -> np.ndarray | None:
    cache = _find_cache(symbol, '4h', days)
    if cache is None or not cache.exists():
        return None
    d = np.load(cache)
    return d['close']


def gen_signals(
    data: dict, btc: np.ndarray, close_4h, settings,
    # Overrides
    min_confidence: int = None,
    target_retrace: float = None,
    stop_mult: float = None,
    stop_decay: float = None,
    max_hold: int = None,
    atr_multiplier: float = None,
    threshold_floor: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate v12.2 signals with optional parameter overrides."""
    cfg = RubberBandConfig()
    cfg.lookback = settings.rb_lookback
    cfg.threshold = settings.rb_threshold
    cfg.target_retrace = target_retrace if target_retrace is not None else settings.rb_target_retrace
    cfg.stop_mult = stop_mult if stop_mult is not None else settings.rb_stop_mult
    cfg.max_hold = max_hold if max_hold is not None else settings.rb_max_hold
    cfg.stop_decay = stop_decay if stop_decay is not None else settings.rb_stop_decay
    cfg.btc_directional_period = settings.rb_btc_directional_period
    cfg.btc_regime_period = settings.rb_btc_regime_period
    cfg.asset_trend_period = settings.rb_asset_trend_period
    cfg.base_leverage = settings.rb_base_leverage
    cfg.confidence_leverage_step = settings.rb_confidence_leverage_step
    cfg.max_leverage = settings.rb_max_leverage
    cfg.rsi_period = settings.rb_rsi_period
    cfg.vol_period = settings.rb_vol_period
    cfg.stoch_enabled = settings.rb_stoch_enabled
    cfg.stoch_k_period = settings.rb_stoch_k_period
    cfg.stoch_d_period = settings.rb_stoch_d_period
    cfg.macd_enabled = settings.rb_macd_enabled
    cfg.macd_fast = settings.rb_macd_fast
    cfg.macd_slow = settings.rb_macd_slow
    cfg.macd_signal_period = settings.rb_macd_signal_period
    cfg.dynamic_target_enabled = settings.rb_dynamic_target_enabled
    cfg.target_conf3 = settings.rb_target_conf3
    cfg.target_conf4 = settings.rb_target_conf4
    cfg.target_conf5 = settings.rb_target_conf5
    cfg.min_confidence = min_confidence if min_confidence is not None else settings.rb_min_confidence

    _atr_mult = atr_multiplier if atr_multiplier is not None else settings.rb_atr_multiplier
    _floor = threshold_floor if threshold_floor is not None else settings.rb_threshold_floor
    _cap = settings.rb_threshold_cap

    c = data['close']
    h = data['high']
    l = data['low']
    v = data['volume']
    t = data['taker_buy']
    n = min(len(c), len(btc))
    c, h, l, v, t, b = c[:n], h[:n], l[:n], v[:n], t[:n], btc[:n]

    sigs = compute_signals_v9(
        c, h, l, v, t, b,
        cfg.lookback, cfg.threshold, cfg.target_retrace, cfg.stop_mult,
        cfg.max_hold, cfg.stop_decay, cfg.btc_directional_period,
        cfg.btc_regime_period, cfg.asset_trend_period,
        cfg.min_confidence, cfg.base_leverage,
        cfg.confidence_leverage_step, cfg.max_leverage,
        cfg.rsi_period, cfg.vol_period,
        cfg.dynamic_target_enabled,
        cfg.target_conf3, cfg.target_conf4, cfg.target_conf5,
        cfg.stoch_enabled, cfg.stoch_k_period, cfg.stoch_d_period,
        cfg.macd_enabled, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period,
        atr_period=settings.rb_atr_period,
        atr_multiplier=_atr_mult,
        threshold_floor=_floor,
        threshold_cap=_cap,
    )

    if close_4h is not None:
        sigs = apply_mtf_confirmation(
            sigs, close_4h, settings.rb_mtf_sma_period, ratio=16)
        sigs = apply_mtf_leverage(sigs, settings.rb_mtf_leverage_mult)

    return c, sigs


def simulate_portfolio(
    all_close: dict[str, np.ndarray],
    all_signals: dict[str, np.ndarray],
    initial_capital: float = 2095.0,
    max_slots: int = 3,
    cost_per_side: float = 0.0004,
    funding_per_bar: float = 0.01 / 100 / 32,
    # Optional: volume gate filter on entries
    all_volume: dict[str, np.ndarray] = None,
    vol_gate_mult: float = 0.0,  # 0 = disabled
    vol_gate_lookback: int = 96,
) -> dict:
    """Portfolio sim with optional volume gate on entries."""
    symbols = sorted(all_close.keys())
    n_bars = min(len(all_close[s]) for s in symbols)

    balance = initial_capital
    equity_curve = np.zeros(n_bars)
    equity_curve[0] = balance

    # Pre-compute volume SMA if needed
    vol_sma: dict[str, np.ndarray] = {}
    if vol_gate_mult > 0 and all_volume:
        for sym in symbols:
            v = all_volume.get(sym)
            if v is None:
                continue
            vm = np.zeros(len(v))
            s = 0.0
            for bi in range(len(v)):
                s += v[bi]
                if bi >= vol_gate_lookback:
                    s -= v[bi - vol_gate_lookback]
                if bi >= vol_gate_lookback - 1:
                    vm[bi] = s / vol_gate_lookback
            vol_sma[sym] = vm

    positions: dict[str, dict] = {}
    trades: list[dict] = []
    peak = balance
    max_dd = 0.0

    for i in range(1, n_bars):
        # Update existing
        symbols_to_close = []
        for sym, pos in list(positions.items()):
            c_arr = all_close[sym]
            s_arr = all_signals[sym]

            bar_ret = c_arr[i] / c_arr[i - 1] - 1.0
            direction = 1.0 if pos['signal'] > 0 else -1.0
            leverage = abs(pos['signal'])

            pnl = direction * leverage * bar_ret * pos['notional']
            pnl -= abs(leverage) * funding_per_bar * pos['notional']
            pos['unrealized'] += pnl
            pos['bars_held'] += 1

            cur_sig = s_arr[i] if i < len(s_arr) else 0.0
            prv_sig = s_arr[i - 1] if i - 1 < len(s_arr) else 0.0

            if cur_sig == 0.0 and prv_sig != 0.0:
                symbols_to_close.append(sym)
            elif cur_sig != 0.0 and np.sign(cur_sig) != np.sign(pos['signal']):
                symbols_to_close.append(sym)

        for sym in set(symbols_to_close):
            if sym not in positions:
                continue
            pos = positions.pop(sym)
            lev = abs(pos['signal'])
            close_cost = lev * cost_per_side * pos['notional']
            net_pnl = pos['unrealized'] - close_cost
            balance += net_pnl
            pnl_pct = net_pnl / pos['notional'] * 100
            trades.append({
                'symbol': sym, 'signal': pos['signal'],
                'entry_bar': pos['entry_bar'], 'exit_bar': i,
                'bars_held': pos['bars_held'],
                'notional': pos['notional'],
                'pnl_usd': net_pnl, 'pnl_pct': pnl_pct,
            })

        # New entries
        n_active = len(positions)
        if n_active < max_slots:
            candidates = []
            for sym in symbols:
                if sym in positions:
                    continue
                s_arr = all_signals[sym]
                if i >= len(s_arr):
                    continue
                cur_sig = s_arr[i]
                prv_sig = s_arr[i - 1] if i > 0 else 0.0
                if cur_sig != 0.0 and prv_sig == 0.0:
                    # Volume gate check
                    if vol_gate_mult > 0 and sym in vol_sma:
                        v_arr = all_volume.get(sym)
                        vm = vol_sma[sym]
                        if v_arr is not None and i < len(v_arr) and i < len(vm) and vm[i] > 0:
                            if v_arr[i] < vol_gate_mult * vm[i]:
                                continue  # Skip low-volume entry
                    candidates.append((sym, cur_sig))

            candidates.sort(key=lambda x: abs(x[1]), reverse=True)

            for sym, sig in candidates:
                if len(positions) >= max_slots:
                    break
                per_slot = initial_capital / max(1, len(positions) + 1)
                per_slot = min(per_slot, initial_capital * 0.5)
                if per_slot < 10.0:
                    continue

                lev = abs(sig)
                entry_cost = lev * cost_per_side * per_slot
                balance -= entry_cost

                positions[sym] = {
                    'signal': sig,
                    'entry_bar': i,
                    'bars_held': 0,
                    'notional': per_slot,
                    'unrealized': 0.0,
                }

        total_unrealized = sum(p['unrealized'] for p in positions.values())
        equity_curve[i] = balance + total_unrealized

        if equity_curve[i] > peak:
            peak = equity_curve[i]
        dd = (peak - equity_curve[i]) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Stats
    n_trades = len(trades)
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    n_wins = len(wins)
    wr = n_wins / n_trades * 100 if n_trades > 0 else 0.0
    total_return = (equity_curve[-1] / initial_capital - 1.0) * 100
    gp = sum(t['pnl_usd'] for t in wins)
    gl = sum(abs(t['pnl_usd']) for t in losses)
    pf = gp / gl if gl > 0 else 999.0

    eq_rets = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-10)
    eq_rets = eq_rets[~np.isnan(eq_rets)]
    sharpe = float(np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(96 * 365.25)) if len(eq_rets) > 0 and np.std(eq_rets) > 0 else 0.0
    calmar = total_return / (max_dd * 100) if max_dd > 0 else 0.0

    aw = float(np.mean([t['pnl_pct'] for t in wins])) if wins else 0
    al = float(np.mean([abs(t['pnl_pct']) for t in losses])) if losses else 1
    rr = aw / al if al > 0 else 999
    avg_pnl = float(np.mean([t['pnl_pct'] for t in trades])) if trades else 0
    avg_bars = float(np.mean([t['bars_held'] for t in trades])) if trades else 0

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'total_return_pct': round(total_return, 1),
        'final_balance': round(equity_curve[-1], 2),
        'max_dd_pct': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 2),
        'calmar': round(calmar, 1),
        'n_trades': n_trades, 'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2),
        'avg_pnl_pct': round(avg_pnl, 3),
        'avg_win_pct': round(aw, 3),
        'avg_loss_pct': round(al, 3),
        'actual_rr': round(rr, 2),
        'avg_bars_held': round(avg_bars, 1),
    }


def main():
    commit = _get_git_commit()
    settings = get_settings()

    print("=" * 90)
    print("  ALPHA SEARCH ROUND 2 — ENTRY/EXIT PARAMETER REFINEMENT")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Same v12.2 signal engine, different param combos")
    print(f"  Target: WR > 64.2% AND R:R > 0.98 simultaneously")
    print("=" * 90)

    SYMBOLS_50 = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
        'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
        'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
        'ICPUSDT', 'FILUSDT',
        'TRXUSDT', 'SUIUSDT', 'TONUSDT', 'LTCUSDT', 'BCHUSDT',
        'UNIUSDT', 'NEARUSDT', 'APTUSDT', 'AAVEUSDT', 'INJUSDT',
        'ETCUSDT', 'MATICUSDT', 'GRTUSDT', 'STXUSDT', 'FTMUSDT',
        'EGLDUSDT', 'AXSUSDT', 'SANDUSDT', 'THETAUSDT', 'ALGOUSDT',
        'RUNEUSDT', 'DYDXUSDT', 'CRVUSDT', 'LDOUSDT', 'COMPUSDT',
        'MKRUSDT', 'SNXUSDT', 'VETUSDT', 'EOSUSDT', 'APEUSDT',
    ]

    DAYS = 1095
    CAPITAL = 2095.0
    COST = 0.0004
    FUNDING = 0.01 / 100 / 32

    # Variant definitions:
    # (name, desc, overrides_dict, vol_gate_mult)
    variants = [
        ('A-Base', 'v12.2 Baseline (conf≥2, tgt 0.382, stop 0.786, decay 0.3)', {}, 0.0),
        ('B-Conf3', 'Confidence ≥ 3 (stricter entry)', {'min_confidence': 3}, 0.0),
        ('C-Tgt50', 'Target 0.500 retrace (bigger wins)', {'target_retrace': 0.5}, 0.0),
        ('D-Decay5', 'Stop decay 0.5 (cut losers faster)', {'stop_decay': 0.5}, 0.0),
        ('E-Hold12', 'Max hold 12 bars (3h time stop)', {'max_hold': 12}, 0.0),
        ('F-VolGate', 'Volume gate 2.0× (high-vol entries only)', {}, 2.0),
        ('G-Combo', 'conf≥3 + tgt=0.50 + decay=0.5 + vol gate', {
            'min_confidence': 3,
            'target_retrace': 0.5,
            'stop_decay': 0.5,
        }, 2.0),
        ('H-Asym', 'Asymmetric: target=0.50, stop=0.50 (R:R=1.0 by design)', {
            'target_retrace': 0.5,
            'stop_mult': 0.50,
        }, 0.0),
        ('I-Tight', 'Tight ATR: 1.5× (lower threshold, more entries)', {
            'atr_multiplier': 1.5,
            'threshold_floor': 0.006,
        }, 0.0),
        ('J-Best', 'conf≥3 + tgt=0.45 + stop=0.60 + decay=0.4 + hold=16', {
            'min_confidence': 3,
            'target_retrace': 0.45,
            'stop_mult': 0.60,
            'stop_decay': 0.4,
            'max_hold': 16,
        }, 0.0),
    ]

    # Load data
    print(f"\n📊 Loading 50-symbol data ({DAYS}d)...")
    btc_data = load_15m('BTCUSDT', DAYS)
    if btc_data is None:
        print("ERROR: BTC 15m data not found")
        sys.exit(1)
    btc_close = btc_data['close']

    all_data: dict[str, dict] = {}
    all_4h: dict[str, np.ndarray] = {}
    for sym in SYMBOLS_50:
        d = load_15m(sym, DAYS)
        if d is None:
            print(f"  SKIP {sym}")
            continue
        h4 = load_4h(sym, DAYS)
        all_data[sym] = d
        if h4 is not None:
            all_4h[sym] = h4

    available = [s for s in SYMBOLS_50 if s in all_data]
    print(f"  Loaded: {len(available)} symbols")

    # Run each variant
    results: dict[str, dict] = {}

    for name, desc, overrides, vol_gate in variants:
        print(f"\n{'─' * 70}")
        print(f"  ⚡ {name}: {desc}")
        print(f"{'─' * 70}")

        t0 = time.time()

        sym_close: dict[str, np.ndarray] = {}
        sym_sigs: dict[str, np.ndarray] = {}
        sym_vol: dict[str, np.ndarray] = {}

        for sym in available:
            data = all_data[sym]
            close_4h = all_4h.get(sym)
            c, sigs = gen_signals(data, btc_close, close_4h, settings, **overrides)
            sym_close[sym] = c
            sym_sigs[sym] = sigs
            sym_vol[sym] = data['volume'][:len(c)]

        # Left-pad
        max_bars = max(len(v) for v in sym_close.values())
        for sym in list(sym_close.keys()):
            for d, fv in [(sym_close, 'price'), (sym_sigs, 'zero'), (sym_vol, 'zero')]:
                arr = d[sym]
                if len(arr) < max_bars:
                    pad_n = max_bars - len(arr)
                    fill = arr[0] if fv == 'price' else 0.0
                    d[sym] = np.concatenate([np.full(pad_n, fill), arr])

        result = simulate_portfolio(
            sym_close, sym_sigs,
            initial_capital=CAPITAL, max_slots=3,
            cost_per_side=COST, funding_per_bar=FUNDING,
            all_volume=sym_vol if vol_gate > 0 else None,
            vol_gate_mult=vol_gate,
        )
        elapsed = time.time() - t0

        n_days = len(result['equity_curve']) / 96
        trades_per_day = round(result['n_trades'] / n_days, 2)

        wr_ok = '✓' if result['win_rate'] > 64.2 else '✗'
        rr_ok = '✓' if result['actual_rr'] > 0.98 else '✗'
        both = '★' if result['win_rate'] > 64.2 and result['actual_rr'] > 0.98 and name != 'A-Base' else ''

        summary = {
            'description': desc,
            'overrides': overrides,
            'vol_gate': vol_gate,
            'total_return_pct': result['total_return_pct'],
            'final_balance': result['final_balance'],
            'max_dd_pct': result['max_dd_pct'],
            'sharpe': result['sharpe'],
            'calmar': result['calmar'],
            'n_trades': result['n_trades'],
            'trades_per_day': trades_per_day,
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'avg_pnl_pct': result['avg_pnl_pct'],
            'avg_win_pct': result['avg_win_pct'],
            'avg_loss_pct': result['avg_loss_pct'],
            'actual_rr': result['actual_rr'],
            'avg_bars_held': result['avg_bars_held'],
            'elapsed': round(elapsed, 2),
        }
        results[name] = summary

        print(f"  Return: {result['total_return_pct']:+,.1f}%  Sharpe: {result['sharpe']:.2f}  "
              f"DD: {result['max_dd_pct']:.1f}%  Calmar: {result['calmar']:.1f}")
        print(f"  Trades: {result['n_trades']} ({trades_per_day}/day)  "
              f"Bars: {result['avg_bars_held']:.1f}  "
              f"WR: {result['win_rate']:.1f}%[{wr_ok}]  R:R: {result['actual_rr']:.2f}[{rr_ok}]  "
              f"PF: {result['profit_factor']:.2f}  {both} [{elapsed:.1f}s]")

    # Results table
    names = list(results.keys())
    base = results['A-Base']

    print(f"\n\n{'=' * 120}")
    print("  ALPHA SEARCH R2 — COMPARATIVE RESULTS")
    print(f"  Target: WR > {base['win_rate']}% AND R:R > {base['actual_rr']}")
    print(f"{'=' * 120}\n")

    col_w = 11
    print(f"  {'Metric':<18}", end='')
    for n in names:
        label = n[:col_w]
        print(f"  {label:>{col_w}}", end='')
    print()
    print("  " + "-" * (18 + (col_w + 2) * len(names)))

    rows = [
        ('Return', 'total_return_pct', '{:+,.1f}%'),
        ('Sharpe', 'sharpe', '{:.2f}'),
        ('Calmar', 'calmar', '{:.1f}'),
        ('Max DD', 'max_dd_pct', '{:.1f}%'),
        ('Trades', 'n_trades', '{:,d}'),
        ('Trades/Day', 'trades_per_day', '{:.2f}'),
        ('Bars Held', 'avg_bars_held', '{:.1f}'),
        ('★ Win Rate', 'win_rate', '{:.1f}%'),
        ('★ R:R', 'actual_rr', '{:.2f}'),
        ('PF', 'profit_factor', '{:.2f}'),
        ('Avg Win', 'avg_win_pct', '{:+.3f}%'),
        ('Avg Loss', 'avg_loss_pct', '{:.3f}%'),
        ('Avg PnL', 'avg_pnl_pct', '{:+.3f}%'),
    ]

    for label, key, fmt in rows:
        print(f"  {label:<18}", end='')
        for n in names:
            val = results[n][key]
            s = fmt.format(val)
            print(f"  {s:>{col_w}}", end='')
        print()

    # Delta vs Base
    print(f"\n{'─' * 120}")
    print(f"  DELTA vs A-Base (WR={base['win_rate']}%, R:R={base['actual_rr']})")
    print(f"{'─' * 120}\n")

    for n in names[1:]:
        r = results[n]
        d_wr = r['win_rate'] - base['win_rate']
        d_rr = r['actual_rr'] - base['actual_rr']
        d_ret = r['total_return_pct'] - base['total_return_pct']
        d_sh = r['sharpe'] - base['sharpe']
        d_dd = r['max_dd_pct'] - base['max_dd_pct']
        wr_ok = '✓' if d_wr > 0 else '✗'
        rr_ok = '✓' if d_rr > 0 else '✗'
        both = '★ DUAL WIN' if d_wr > 0 and d_rr > 0 else ''
        print(f"  {n:<12} ΔWR={d_wr:>+5.1f}%p[{wr_ok}]  ΔR:R={d_rr:>+5.2f}[{rr_ok}]  "
              f"ΔRet={d_ret:>+8,.1f}%  ΔSharpe={d_sh:>+5.2f}  ΔDD={d_dd:>+5.1f}%  {both}")

    # Highlight winners
    print(f"\n{'=' * 120}")
    print("  VERDICT")
    print(f"{'=' * 120}\n")

    dual_winners = []
    for n in names[1:]:
        r = results[n]
        if r['win_rate'] > base['win_rate'] and r['actual_rr'] > base['actual_rr']:
            dual_winners.append(n)

    if dual_winners:
        print(f"  ★ Dual improvements found: {', '.join(dual_winners)}")
        for n in dual_winners:
            r = results[n]
            print(f"    {n}: WR {r['win_rate']:.1f}% (Δ+{r['win_rate']-base['win_rate']:.1f}), "
                  f"R:R {r['actual_rr']:.2f} (Δ+{r['actual_rr']-base['actual_rr']:.2f}), "
                  f"Sharpe {r['sharpe']:.2f}, Return {r['total_return_pct']:+,.1f}%")
    else:
        print("  No variant achieves BOTH WR and R:R improvement simultaneously.")
        print("  This confirms a structural WR/R:R tradeoff in mean-reversion.")
        print()
        # Show closest candidates
        best_rr = max(names[1:], key=lambda n: results[n]['actual_rr'])
        best_wr = max(names[1:], key=lambda n: results[n]['win_rate'])
        print(f"  Best R:R: {best_rr} → R:R={results[best_rr]['actual_rr']:.2f}, "
              f"WR={results[best_rr]['win_rate']:.1f}%")
        print(f"  Best WR:  {best_wr} → WR={results[best_wr]['win_rate']:.1f}%, "
              f"R:R={results[best_wr]['actual_rr']:.2f}")

    # Save
    output_path = Path(__file__).parent.parent / 'logs' / 'alpha_search_r2_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
    print(f"\n{'=' * 120}")
    print("  DONE")
    print(f"{'=' * 120}")


if __name__ == '__main__':
    main()
