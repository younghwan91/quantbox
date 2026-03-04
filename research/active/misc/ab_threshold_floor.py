#!/usr/bin/env python3
"""
A/B Test: rb_threshold_floor  0.008 (A/현재) vs 0.015 (B/제안)
=================================================================
Ghost SL 분석 결과, move < 1.5%인 신호에서 SL_dist(=move×0.382)가
15분봉 노이즈 범위 내에 들어와 즉각 SL 체결이 반복됨.

THR_FLOOR만 변경 (모든 다른 파라미터 동일):
  A: THR_FLOOR = 0.008  (현재 v12.5.3 프로덕션)
  B: THR_FLOOR = 0.015  (제안, minimum SL_dist=0.574%)

Usage:
    uv run python research/active/ab_threshold_floor.py
"""

import sys
import time
import subprocess
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

# ─── Import from v1253 backbone ────────────────────────────────────
# We import the scanning + simulation functions directly.
# Only THR_FLOOR changes between A and B — everything else is identical.
sys.path.insert(0, str(ROOT / 'research' / 'active'))
import v1253_intrabar_6yr_backtest as v1253

# Override module-level globals we DO want to share
from v1253_intrabar_6yr_backtest import (
    scan_entries_risk_off_only,
    apply_mtf_filter,
    simulate_portfolio,
    load_15m,
    load_4h,
    LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
    ASSET_TREND_PERIOD, MIN_CONFIDENCE, RSI_PERIOD, VOL_PERIOD,
    STOCH_ENABLED, STOCH_K, STOCH_D,
    MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ATR_PERIOD, ATR_MULT, THR_CAP,
    MTF_SMA_PERIOD, DATA_DAYS, INITIAL_CAPITAL,
)
from quantbox.strategy.rubberband import compute_atr_pct

# ─── A/B configs ───────────────────────────────────────────────────
CONFIGS = {
    'A_current  (floor=0.8%)': 0.008,
    'B_proposed (floor=1.5%)': 0.015,
}

SYMBOLS_50 = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
    'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
    'ICPUSDT', 'FILUSDT', 'TRXUSDT', 'SUIUSDT', 'TONUSDT', 'LTCUSDT',
    'BCHUSDT', 'UNIUSDT', 'NEARUSDT', 'APTUSDT', 'AAVEUSDT', 'INJUSDT',
    'ETCUSDT', 'GRTUSDT', 'STXUSDT', 'EGLDUSDT', 'AXSUSDT', 'SANDUSDT',
    'THETAUSDT', 'ALGOUSDT', 'RUNEUSDT', 'DYDXUSDT', 'CRVUSDT', 'LDOUSDT',
    'COMPUSDT', 'SNXUSDT', 'VETUSDT', 'APEUSDT',
    'ENAUSDT', 'TAOUSDT', 'WLDUSDT', 'WIFUSDT',
]


def _get_git_commit() -> str:
    try:
        short = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                        cwd=ROOT, text=True).strip()
        msg = subprocess.check_output(['git', 'log', '-1', '--format=%s'],
                                      cwd=ROOT, text=True).strip()[:60]
        return f"{short} ({msg})"
    except Exception:
        return "unknown"


def load_and_prepare_data() -> tuple[dict, dict, dict, list[str], int]:
    """데이터 로드 및 공통 전처리 (한 번만 수행)."""
    print(f"\n📊 Loading {len(SYMBOLS_50)}-symbol data ({DATA_DAYS}d)...")
    btc_data = load_15m('BTCUSDT')
    if btc_data is None:
        raise RuntimeError("BTC 15m data not found in cache_extended")
    btc_close = btc_data['close']
    print(f"  BTC: {len(btc_close):,} bars ({len(btc_close)/96:.0f} days)")

    all_data: dict[str, dict] = {}
    all_4h: dict[str, np.ndarray] = {}
    for sym in SYMBOLS_50:
        d = load_15m(sym)
        if d is None:
            continue
        all_data[sym] = d
        h4 = load_4h(sym)
        if h4 is not None:
            all_4h[sym] = h4

    available = [s for s in SYMBOLS_50 if s in all_data]
    print(f"  Loaded: {len(available)} symbols")
    return all_data, all_4h, btc_close, available


def scan_for_floor(
    all_data: dict, all_4h: dict, btc_close: np.ndarray,
    available: list[str], thr_floor: float
) -> tuple[dict, dict, dict, dict, dict, dict, dict, int]:
    """지정된 thr_floor로 엔트리 스캔 → 심볼별 배열 반환."""
    sym_dirs:  dict[str, np.ndarray] = {}
    sym_confs: dict[str, np.ndarray] = {}
    sym_moves: dict[str, np.ndarray] = {}
    sym_close: dict[str, np.ndarray] = {}
    sym_high:  dict[str, np.ndarray] = {}
    sym_low:   dict[str, np.ndarray] = {}
    sym_atr:   dict[str, np.ndarray] = {}

    for sym in available:
        d = all_data[sym]
        n = min(len(d['close']), len(btc_close))
        c, h, l, v, t = (d['close'][:n], d['high'][:n], d['low'][:n],
                         d['volume'][:n], d['taker_buy'][:n])
        b = btc_close[:n]

        dirs, confs, moves = scan_entries_risk_off_only(
            c, h, l, v, t, b,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
            ASSET_TREND_PERIOD, MIN_CONFIDENCE, RSI_PERIOD, VOL_PERIOD,
            STOCH_ENABLED, STOCH_K, STOCH_D,
            MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            ATR_PERIOD, ATR_MULT, thr_floor, THR_CAP,
        )

        close_4h = all_4h.get(sym)
        if close_4h is not None:
            dirs, confs, moves = apply_mtf_filter(
                dirs, confs, moves, close_4h, MTF_SMA_PERIOD, ratio=16)

        sym_dirs[sym] = dirs
        sym_confs[sym] = confs
        sym_moves[sym] = moves
        sym_close[sym] = c
        sym_high[sym] = h
        sym_low[sym] = l
        sym_atr[sym] = compute_atr_pct(h, l, c, ATR_PERIOD)

    # Left-pad to uniform length
    max_n = max(len(sym_close[s]) for s in sym_close)
    for sym in list(sym_close.keys()):
        for d_ref, fill in [
            (sym_close, 'first'), (sym_high, 'first'), (sym_low, 'first'),
            (sym_dirs, 'zero'), (sym_confs, 'zero'), (sym_moves, 'zero'),
            (sym_atr, 'nan'),
        ]:
            arr = d_ref[sym]
            if len(arr) < max_n:
                pad = max_n - len(arr)
                fv = arr[0] if fill == 'first' else (np.nan if fill == 'nan' else 0.0)
                d_ref[sym] = np.concatenate([np.full(pad, fv), arr])

    total_entries = sum(np.count_nonzero(sym_dirs[s]) for s in sym_dirs)
    return sym_close, sym_high, sym_low, sym_dirs, sym_confs, sym_moves, sym_atr, total_entries


def print_summary(label: str, thr_floor: float, total_entries: int,
                  result: dict, elapsed: float):
    reasons = '  '.join(f"{k}={v}" for k, v in sorted(result['exit_reasons'].items()))
    width = 80
    print(f"\n{'=' * width}")
    print(f"  {label}   (THR_FLOOR={thr_floor*100:.1f}%)")
    print(f"{'=' * width}")
    print(f"  Entry opportunities: {total_entries:,}")
    print(f"  Return:        {result['total_return_pct']:+,.1f}%")
    print(f"  Final Balance: ${result['final_balance']:,.2f}  (from ${INITIAL_CAPITAL:,.0f})")
    print(f"  Max DD:        {result['max_dd_pct']:.1f}%")
    print(f"  Sharpe:        {result['sharpe']:.2f}")
    print(f"  Calmar:        {result['calmar']:.1f}")
    print(f"  Trades:        {result['n_trades']:,}  ({result['trades_per_day']}/day)")
    print(f"  Avg Bars:      {result['avg_bars_held']:.1f}")
    print(f"  Win Rate:      {result['win_rate']:.1f}%")
    print(f"  R:R:           {result['actual_rr']:.2f}")
    print(f"  Profit Factor: {result['profit_factor']:.2f}")
    print(f"  Avg Win:       {result['avg_win_pct']:+.3f}%")
    print(f"  Avg Loss:      {result['avg_loss_pct']:.3f}%")
    print(f"  Exit Reasons:  {reasons}")
    print(f"  Elapsed:       {elapsed:.1f}s")


def print_headtohead(results: list[tuple[str, float, dict]]):
    width = 80
    (la, fa, ra), (lb, fb, rb) = results
    print(f"\n{'=' * width}")
    print(f"  HEAD-TO-HEAD: A (floor={fa*100:.1f}%) vs B (floor={fb*100:.1f}%)")
    print(f"{'=' * width}")
    print(f"  {'Metric':<22} {'A (current)':>13} {'B (proposed)':>13} {'Delta B-A':>10}")
    print(f"  {'-' * 60}")

    metrics = [
        ('Return %',       'total_return_pct', '.1f'),
        ('Max DD %',       'max_dd_pct',       '.1f'),
        ('Sharpe',         'sharpe',           '.2f'),
        ('Calmar',         'calmar',           '.1f'),
        ('Trades',         'n_trades',         'd'),
        ('Win Rate %',     'win_rate',         '.1f'),
        ('R:R',            'actual_rr',        '.2f'),
        ('Profit Factor',  'profit_factor',    '.2f'),
        ('Avg Win %',      'avg_win_pct',      '.3f'),
        ('Avg Loss %',     'avg_loss_pct',     '.3f'),
    ]
    for label, key, fmt in metrics:
        va = ra[key]
        vb = rb[key]
        delta = vb - va
        if fmt == 'd':
            print(f"  {label:<22} {va:>13,d} {vb:>13,d} {delta:>+10,d}")
        else:
            print(f"  {label:<22} {va:>13{fmt}} {vb:>13{fmt}} {delta:>+10{fmt}}")

    # Verdict
    improved = (rb['sharpe'] > ra['sharpe'] and
                rb['max_dd_pct'] <= ra['max_dd_pct'] * 1.05 and
                rb['calmar'] >= ra['calmar'] * 0.95)
    print(f"\n  VERDICT: {'✅ B is BETTER — apply to production' if improved else '⚠️  B is NOT clearly better — keep current'}")
    if rb['n_trades'] < ra['n_trades']:
        trade_diff = ra['n_trades'] - rb['n_trades']
        pct = trade_diff / ra['n_trades'] * 100
        print(f"  ℹ️  B filters {trade_diff:,} trades ({pct:.1f}% fewer) — expected from higher floor")


def main():
    commit = _get_git_commit()
    print("=" * 80)
    print("  A/B TEST: rb_threshold_floor  0.008 vs 0.015")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Rationale: Ghost SL 7건 중 5건이 move<1.5%, SL_dist<0.58% 신호")
    print(f"  Fixed : THR_FLOOR only — all other params identical")
    print("=" * 80)

    # ── Load data once ──
    all_data, all_4h, btc_close, available = load_and_prepare_data()

    ab_results: list[tuple[str, float, dict]] = []

    for label, thr_floor in CONFIGS.items():
        print(f"\n🔍 Scanning entries  [{label}]...")
        t0 = time.time()
        (sym_close, sym_high, sym_low,
         sym_dirs, sym_confs, sym_moves,
         sym_atr, total_entries) = scan_for_floor(
            all_data, all_4h, btc_close, available, thr_floor
        )
        scan_time = time.time() - t0
        print(f"  Entry opportunities: {total_entries:,}  ({scan_time:.1f}s)")

        print(f"⚡ Simulating portfolio  [{label}]...")
        t0 = time.time()
        result = simulate_portfolio(
            sym_close, sym_high, sym_low,
            sym_dirs, sym_confs, sym_moves,
            sym_atr=sym_atr,
        )
        elapsed = time.time() - t0

        print_summary(label, thr_floor, total_entries, result, elapsed)
        ab_results.append((label, thr_floor, result))

    print_headtohead(ab_results)
    print(f"\n{'=' * 80}")
    print("  A/B TEST COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
