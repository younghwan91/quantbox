#!/usr/bin/env python3
"""
Alpha Search R9 — MA & TOTAL3 PROXY SWEEP (6-YEAR CRASH-ERA)
==============================================================
MOTIVATION: Crowd psychology clusters at popular exchange moving averages.
  Binance defaults: MA(7), MA(25), MA(99)
  Upbit defaults: MA(5), MA(10), MA(20), MA(60), MA(120)
  Universal: MA(200) — the bull/bear divider

  TOTAL3 (TradingView) = Total Crypto Market Cap ex-BTC/ETH.
  We build an equal-weight altcoin proxy from our 48 non-BTC/ETH symbols.
  When this proxy trends down, individual alt long entries should be cautious.

  Expert insight: "total3 라는 차트를 사용하라는 전문가도 있어" — alt-season
  detection through aggregate altcoin index is a widely-used macro filter.

Base: T100_S38_PBT (R8 user pick: R:R=1.66, most regime-stable)
Fixed: M-2SSTSX (2slt + SoftTight + SteepX + dir_diversity + conf≥3)

Test Groups (19 variants + 1 reference):
  A. Asset 4h MA Trend Filter (5) — only enter in direction of popular MA
  B. MA Proximity Confidence Boost (5) — +1 conf near crowd-watched MA
  C. TOTAL3 Proxy Regime (4) — alt market cap index as macro filter
  D. Best Combos (4) — stacked filters

Tested MAs (on 4h chart, matching exchange defaults):
  MA(25)  — Binance default short-term (≈ 4 days)
  MA(99)  — Binance default long-term (≈ 16 days)
  MA(200) — Universal bull/bear line (≈ 33 days)

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/alpha_search_r9.py
"""

import sys
import time
import json
import subprocess
import numpy as np
from dataclasses import dataclass, field, replace, fields
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from quantbox.strategy.rubberband import _compute_sma_inline, compute_atr_pct

from scripts.alpha_search_r7 import (
    R7Config, simulate_r7, scan_entries, apply_mtf_filter,
    load_15m, load_4h,
    LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
    ASSET_TREND_PERIOD, MIN_CONFIDENCE, RSI_PERIOD, VOL_PERIOD,
    STOCH_ENABLED, STOCH_K, STOCH_D,
    MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP, MTF_SMA_PERIOD,
    DATA_DAYS, CACHE_DIR,
)


SYMBOLS_50 = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
    'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
    'ICPUSDT', 'FILUSDT', 'TRXUSDT', 'SUIUSDT', 'TONUSDT', 'LTCUSDT',
    'BCHUSDT', 'UNIUSDT', 'NEARUSDT', 'APTUSDT', 'AAVEUSDT', 'INJUSDT',
    'ETCUSDT', 'MATICUSDT', 'GRTUSDT', 'STXUSDT', 'FTMUSDT', 'EGLDUSDT',
    'AXSUSDT', 'SANDUSDT', 'THETAUSDT', 'ALGOUSDT', 'RUNEUSDT', 'DYDXUSDT',
    'CRVUSDT', 'LDOUSDT', 'COMPUSDT', 'MKRUSDT', 'SNXUSDT', 'VETUSDT',
    'EOSUSDT', 'APEUSDT',
]

# Exchange-default MA periods (computed on 4h chart)
MA_PERIODS_4H = [25, 99, 200]


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


# ═══════════════════════════════════════════════════════════════════
# R9 Variant Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class R9Variant:
    """Describes which MA/TOTAL3 filters to apply on top of T100_S38_PBT."""
    name: str = ''
    desc: str = ''

    # Group A: Asset 4h MA trend filter (additional direction gate)
    ma_trend_periods: list[int] = field(default_factory=list)
    ma_golden_cross: bool = False  # check MA crossover instead of price vs MA

    # Group B: MA proximity confidence boost
    proximity_periods: list[int] = field(default_factory=list)
    proximity_pct: float = 0.02

    # Group C: TOTAL3 proxy regime
    total3_enabled: bool = False
    total3_sma_period: int = 96
    total3_mode: str = 'block_longs'  # block_longs | block_all | conf_mod


# T100_S38_PBT base config (R8 user pick)
BASE_R7_CFG = R7Config(
    name='', desc='',
    target_retrace=1.0,
    stop_mult=0.382,
    exit_mode='partial_be',
    breakeven_after_tp1=True,
    trail_activation=0.5,
    trail_factor=0.3,
    partial_fraction=0.5,
    trail_atr_mult=1.5,
    max_slots=2,
    min_conf_gate=3,
    dir_diversity=True,
    max_hold=20,
    stop_decay=0.0,
    steep_cap_curve=True,
    cap_mult_conf2=0.3,
    cap_mult_conf3=0.5,
    cap_mult_conf4=1.5,
    cap_mult_conf5=2.0,
    vol_adjusted_size=True,
    soft_equity_scale=True,
    soft_eq_dd1=0.02,
    soft_eq_scale1=0.80,
    soft_eq_dd2=0.04,
    soft_eq_scale2=0.60,
    soft_eq_dd3=0.06,
    soft_eq_scale3=0.40,
)


def make_variants() -> list[R9Variant]:
    """Generate all R9 test variants."""
    v = []

    # ── Reference ──
    v.append(R9Variant(
        name='★REF_T100S38',
        desc='T100_S38_PBT baseline — no MA enhancements',
    ))

    # ── Group A: Asset 4h MA Trend Filter ──
    v.append(R9Variant(
        name='A1_MA25',
        desc='4h SMA(25) trend gate: long above, short below',
        ma_trend_periods=[25],
    ))
    v.append(R9Variant(
        name='A2_MA99',
        desc='4h SMA(99) trend gate: Binance default long-term',
        ma_trend_periods=[99],
    ))
    v.append(R9Variant(
        name='A3_MA200',
        desc='4h SMA(200) trend gate: universal bull/bear',
        ma_trend_periods=[200],
    ))
    v.append(R9Variant(
        name='A4_MA25x99',
        desc='Price must be on right side of BOTH MA(25) AND MA(99)',
        ma_trend_periods=[25, 99],
    ))
    v.append(R9Variant(
        name='A5_GoldX',
        desc='Golden cross: long only if MA(25)>MA(99), short only if <',
        ma_trend_periods=[25, 99],
        ma_golden_cross=True,
    ))

    # ── Group B: MA Proximity Confidence Boost ──
    v.append(R9Variant(
        name='B1_P25',
        desc='+1 conf within 2% of 4h SMA(25)',
        proximity_periods=[25],
    ))
    v.append(R9Variant(
        name='B2_P99',
        desc='+1 conf within 2% of 4h SMA(99)',
        proximity_periods=[99],
    ))
    v.append(R9Variant(
        name='B3_P200',
        desc='+1 conf within 2% of 4h SMA(200)',
        proximity_periods=[200],
    ))
    v.append(R9Variant(
        name='B4_PANY2',
        desc='+1 conf within 2% of ANY of MA(25/99/200)',
        proximity_periods=[25, 99, 200],
    ))
    v.append(R9Variant(
        name='B5_PANY1',
        desc='+1 conf within 1% of ANY of MA(25/99/200) — tighter',
        proximity_periods=[25, 99, 200],
        proximity_pct=0.01,
    ))

    # ── Group C: TOTAL3 Proxy Regime ──
    v.append(R9Variant(
        name='C1_T3_96',
        desc='TOTAL3 > SMA(96) → allow longs; else block longs',
        total3_enabled=True,
        total3_sma_period=96,
        total3_mode='block_longs',
    ))
    v.append(R9Variant(
        name='C2_T3_192',
        desc='TOTAL3 > SMA(192) → allow longs; else block longs',
        total3_enabled=True,
        total3_sma_period=192,
        total3_mode='block_longs',
    ))
    v.append(R9Variant(
        name='C3_T3_BLK',
        desc='TOTAL3 risk-off → block ALL entries',
        total3_enabled=True,
        total3_sma_period=96,
        total3_mode='block_all',
    ))
    v.append(R9Variant(
        name='C4_T3_CONF',
        desc='TOTAL3 risk-on → +1 conf; risk-off → -1 conf',
        total3_enabled=True,
        total3_sma_period=96,
        total3_mode='conf_mod',
    ))

    # ── Group D: Best Combos ──
    v.append(R9Variant(
        name='D1_MA99+T3',
        desc='MA(99) trend + TOTAL3 SMA(96) block_longs',
        ma_trend_periods=[99],
        total3_enabled=True,
        total3_sma_period=96,
        total3_mode='block_longs',
    ))
    v.append(R9Variant(
        name='D2_MA99+PA',
        desc='MA(99) trend + proximity boost ANY 2%',
        ma_trend_periods=[99],
        proximity_periods=[25, 99, 200],
    ))
    v.append(R9Variant(
        name='D3_TRIPLE',
        desc='MA(99) trend + proximity ANY + TOTAL3 block_longs',
        ma_trend_periods=[99],
        proximity_periods=[25, 99, 200],
        total3_enabled=True,
        total3_sma_period=96,
        total3_mode='block_longs',
    ))
    v.append(R9Variant(
        name='D4_GX+T3',
        desc='Golden cross + TOTAL3 SMA(96) block_longs',
        ma_trend_periods=[25, 99],
        ma_golden_cross=True,
        total3_enabled=True,
        total3_sma_period=96,
        total3_mode='block_longs',
    ))

    return v


# ═══════════════════════════════════════════════════════════════════
# Filter Functions (numpy-vectorized)
# ═══════════════════════════════════════════════════════════════════

def apply_ma_trend_filter(
    dirs: np.ndarray,
    close_4h: np.ndarray,
    sma_4h_dict: dict[int, np.ndarray],
    periods: list[int],
    pad_offset: int,
    golden_cross: bool = False,
    ratio: int = 16,
) -> np.ndarray:
    """Zero out entries going against 4h MA trend. Vectorized.

    Standard mode: block longs if price <= SMA, shorts if price >= SMA.
    Golden cross mode: block longs if short_MA <= long_MA, shorts if >.
    """
    n = len(dirs)
    n_4h = len(close_4h)
    od = dirs.copy()

    bar_idx = np.arange(n)
    j_idx = np.clip((bar_idx - pad_offset) // ratio, 0, n_4h - 1)
    valid = (dirs != 0) & (bar_idx >= pad_offset)

    if golden_cross and len(periods) >= 2:
        short_p, long_p = min(periods), max(periods)
        short_sma = sma_4h_dict.get(short_p)
        long_sma = sma_4h_dict.get(long_p)
        if short_sma is None or long_sma is None:
            return od
        s_vals = short_sma[j_idx]
        l_vals = long_sma[j_idx]
        valid_p = valid & (~np.isnan(s_vals)) & (~np.isnan(l_vals))
        golden = s_vals > l_vals
        block = valid_p & (((dirs > 0) & (~golden)) | ((dirs < 0) & golden))
    else:
        prices_4h = close_4h[j_idx]
        block = np.zeros(n, dtype=bool)
        for p in periods:
            sma = sma_4h_dict.get(p)
            if sma is None:
                continue
            sma_vals = sma[j_idx]
            p_valid = valid & (~np.isnan(sma_vals))
            block |= p_valid & (dirs > 0) & (prices_4h <= sma_vals)
            block |= p_valid & (dirs < 0) & (prices_4h >= sma_vals)

    od[block] = 0.0
    return od


def apply_proximity_boost(
    dirs: np.ndarray,
    confs: np.ndarray,
    close_4h: np.ndarray,
    sma_4h_dict: dict[int, np.ndarray],
    periods: list[int],
    pad_offset: int,
    proximity_pct: float = 0.02,
    ratio: int = 16,
) -> np.ndarray:
    """Add +1 confidence when price is within proximity% of any specified MA.

    Returns modified confs array. Only boosts entries that haven't been
    zeroed out (dirs != 0).
    """
    n = len(dirs)
    n_4h = len(close_4h)
    oc = confs.copy()

    bar_idx = np.arange(n)
    j_idx = np.clip((bar_idx - pad_offset) // ratio, 0, n_4h - 1)
    valid = (dirs != 0) & (bar_idx >= pad_offset)
    prices_4h = close_4h[j_idx]

    near_any = np.zeros(n, dtype=bool)
    for p in periods:
        sma = sma_4h_dict.get(p)
        if sma is None:
            continue
        sma_vals = sma[j_idx]
        p_valid = valid & (~np.isnan(sma_vals)) & (sma_vals > 0)
        dist = np.abs(prices_4h - sma_vals) / np.maximum(sma_vals, 1e-10)
        near_any |= p_valid & (dist < proximity_pct)

    oc[near_any] += 1.0
    return oc


def apply_total3_filter(
    dirs: np.ndarray,
    confs: np.ndarray,
    total3_index: np.ndarray,
    total3_sma: np.ndarray,
    mode: str = 'block_longs',
) -> tuple[np.ndarray, np.ndarray]:
    """Apply TOTAL3 proxy regime filter. Vectorized.

    Modes:
        block_longs: risk-off → block long entries only
        block_all:   risk-off → block ALL entries
        conf_mod:    risk-on → +1 conf; risk-off → -1 conf
    """
    n = len(dirs)
    od = dirs.copy()
    oc = confs.copy()

    # Ensure arrays align
    t3_len = min(n, len(total3_index), len(total3_sma))
    risk_on = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)

    risk_on[:t3_len] = total3_index[:t3_len] > total3_sma[:t3_len]
    valid[:t3_len] = (dirs[:t3_len] != 0) & (~np.isnan(total3_sma[:t3_len]))

    if mode == 'block_longs':
        block = valid & (~risk_on) & (dirs > 0)
        od[block] = 0.0
    elif mode == 'block_all':
        block = valid & (~risk_on)
        od[block] = 0.0
    elif mode == 'conf_mod':
        boost = valid & risk_on
        reduce = valid & (~risk_on)
        oc[boost] += 1.0
        oc[reduce] -= 1.0
        kill = valid & (oc < 1.0)
        od[kill] = 0.0

    return od, oc


def compute_total3_proxy(
    sym_close: dict[str, np.ndarray],
    exclude: set[str] | None = None,
) -> np.ndarray:
    """Equal-weight normalized index of non-BTC/ETH symbols.

    Approximates TradingView TOTAL3 (Total Crypto Market Cap ex-BTC/ETH).
    Each symbol is rebased to 1.0 at first bar, then averaged across all alts.
    """
    if exclude is None:
        exclude = {'BTCUSDT', 'ETHUSDT'}
    arrays = []
    for sym, close in sym_close.items():
        if sym in exclude:
            continue
        if close[0] > 0:
            arrays.append(close / close[0])

    if not arrays:
        raise ValueError("No symbols for TOTAL3 proxy computation")

    return np.mean(np.array(arrays), axis=0)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    commit = _get_git_commit()
    variants = make_variants()

    n_a = len([v for v in variants if v.name.startswith('A')])
    n_b = len([v for v in variants if v.name.startswith('B')])
    n_c = len([v for v in variants if v.name.startswith('C')])
    n_d = len([v for v in variants if v.name.startswith('D')])

    print("=" * 130)
    print("  ALPHA SEARCH R9 — MA & TOTAL3 PROXY SWEEP (6-YEAR CRASH-ERA)")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Data: {DATA_DAYS}d (covers COVID, Luna, FTX crashes)")
    print(f"  Base: T100_S38_PBT (R8 pick: R:R=1.66, most regime-stable)")
    print(f"  Fixed: M-2SSTSX (2slt + SoftTight + SteepX)")
    print(f"  Groups: A=MA_Trend({n_a}) B=MA_Prox({n_b}) C=TOTAL3({n_c}) D=Combo({n_d})")
    print(f"  MAs: 4h SMA(25), SMA(99), SMA(200)")
    print(f"  TOTAL3: Equal-weight alt proxy (48 symbols ex-BTC/ETH)")
    print(f"  Total: {len(variants)} variants")
    print("=" * 130)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Load data (identical to R7)
    # ═══════════════════════════════════════════════════════════════

    print(f"\n📊 Loading 50-symbol data ({DATA_DAYS}d)...")
    btc_data = load_15m('BTCUSDT')
    if btc_data is None:
        print("ERROR: BTC 15m data not found")
        sys.exit(1)
    btc_close = btc_data['close']
    print(f"  BTC: {len(btc_close):,} bars ({len(btc_close)/96:.0f} days)")

    all_data: dict[str, dict] = {}
    all_4h: dict[str, np.ndarray] = {}
    for sym in SYMBOLS_50:
        d = load_15m(sym)
        if d is None:
            print(f"  SKIP {sym}: no {DATA_DAYS}d data")
            continue
        all_data[sym] = d
        h4 = load_4h(sym)
        if h4 is not None:
            all_4h[sym] = h4

    available = [s for s in SYMBOLS_50 if s in all_data]
    print(f"  Loaded: {len(available)} symbols ({len(all_4h)} with 4h data)")

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Scan entries + MTF filter (identical to R7)
    # ═══════════════════════════════════════════════════════════════

    print(f"\n🔍 Scanning entries & applying MTF filter...")
    t0 = time.time()

    sym_dirs: dict[str, np.ndarray] = {}
    sym_confs: dict[str, np.ndarray] = {}
    sym_moves: dict[str, np.ndarray] = {}
    sym_close: dict[str, np.ndarray] = {}
    sym_high: dict[str, np.ndarray] = {}
    sym_low: dict[str, np.ndarray] = {}
    sym_original_n: dict[str, int] = {}

    for sym in available:
        d = all_data[sym]
        n = min(len(d['close']), len(btc_close))
        c = d['close'][:n]
        h = d['high'][:n]
        l = d['low'][:n]
        v = d['volume'][:n]
        t = d['taker_buy'][:n]
        b = btc_close[:n]

        dirs, confs, moves = scan_entries(
            c, h, l, v, t, b,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
            ASSET_TREND_PERIOD, MIN_CONFIDENCE, RSI_PERIOD, VOL_PERIOD,
            STOCH_ENABLED, STOCH_K, STOCH_D,
            MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
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
        sym_original_n[sym] = n

    # Pre-compute ATR
    sym_atr: dict[str, np.ndarray] = {}
    for sym in available:
        d = all_data[sym]
        n = min(len(d['close']), len(btc_close))
        sym_atr[sym] = compute_atr_pct(d['high'][:n], d['low'][:n], d['close'][:n], ATR_PERIOD)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Left-pad all arrays to max_bars (identical to R7)
    # ═══════════════════════════════════════════════════════════════

    max_bars = max(len(sym_close[s]) for s in sym_close)

    sym_pad_offsets: dict[str, int] = {}
    for sym in list(sym_close.keys()):
        sym_pad_offsets[sym] = max_bars - len(sym_close[sym])
        for d_ref, fill in [
            (sym_close, 'first'), (sym_high, 'first'), (sym_low, 'first'),
            (sym_dirs, 'zero'), (sym_confs, 'zero'), (sym_moves, 'zero'),
            (sym_atr, 'nan'),
        ]:
            arr = d_ref[sym]
            if len(arr) < max_bars:
                pad_n = max_bars - len(arr)
                if fill == 'first':
                    fv = arr[0]
                elif fill == 'nan':
                    fv = np.nan
                else:
                    fv = 0.0
                d_ref[sym] = np.concatenate([np.full(pad_n, fv), arr])

    total_entries_base = sum(np.count_nonzero(sym_dirs[s]) for s in sym_dirs)
    scan_time = time.time() - t0
    print(f"  Total base entries (after MTF): {total_entries_base:,} ({scan_time:.1f}s)")
    print(f"  Simulation window: {max_bars:,} bars ({max_bars/96:.0f} days)")

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Pre-compute 4h MAs per symbol
    # ═══════════════════════════════════════════════════════════════

    print(f"\n📐 Computing 4h MAs per symbol: SMA(25), SMA(99), SMA(200)...")
    sym_4h_sma: dict[str, dict[int, np.ndarray]] = {}
    for sym in available:
        close_4h = all_4h.get(sym)
        if close_4h is None:
            continue
        sym_4h_sma[sym] = {}
        for period in MA_PERIODS_4H:
            sym_4h_sma[sym][period] = _compute_sma_inline(close_4h, period)
    print(f"  Computed for {len(sym_4h_sma)} symbols")

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Compute TOTAL3 proxy
    # ═══════════════════════════════════════════════════════════════

    print(f"\n🌍 Building TOTAL3 proxy (equal-weight alt index)...")
    total3_index = compute_total3_proxy(sym_close, exclude={'BTCUSDT', 'ETHUSDT'})
    n_proxy_syms = sum(1 for s in sym_close if s not in {'BTCUSDT', 'ETHUSDT'})
    print(f"  Proxy symbols: {n_proxy_syms}")
    print(f"  Proxy range: {total3_index[0]:.4f} → {total3_index[-1]:.4f}")

    # Pre-compute TOTAL3 SMAs
    total3_smas: dict[int, np.ndarray] = {}
    for period in [96, 192]:
        total3_smas[period] = _compute_sma_inline(total3_index, period)
        first_valid = period  # first non-NaN index
        t3_above_pct = np.nanmean(total3_index[first_valid:] > total3_smas[period][first_valid:]) * 100
        print(f"  TOTAL3 > SMA({period}): {t3_above_pct:.1f}% of time (risk-on ratio)")

    # ═══════════════════════════════════════════════════════════════
    # Phase 6: Run variants
    # ═══════════════════════════════════════════════════════════════

    print(f"\n🧪 Running {len(variants)} variants...")
    print(f"{'─' * 130}")

    results: dict[str, dict] = {}
    entry_counts: dict[str, int] = {}
    ref_result: dict | None = None

    sym_list = sorted(sym_close.keys())

    for var_idx, variant in enumerate(variants):
        t1 = time.time()

        # ── Copy base dirs/confs per symbol ──
        v_dirs = {sym: sym_dirs[sym].copy() for sym in sym_list}
        v_confs = {sym: sym_confs[sym].copy() for sym in sym_list}

        # ── Group A: MA trend filter ──
        if variant.ma_trend_periods:
            for sym in sym_list:
                if sym not in sym_4h_sma:
                    continue
                v_dirs[sym] = apply_ma_trend_filter(
                    v_dirs[sym], all_4h[sym], sym_4h_sma[sym],
                    variant.ma_trend_periods, sym_pad_offsets[sym],
                    golden_cross=variant.ma_golden_cross,
                )
                # Zero confs where dirs zeroed
                zeroed = (sym_dirs[sym] != 0) & (v_dirs[sym] == 0)
                v_confs[sym][zeroed] = 0.0

        # ── Group B: Proximity boost ──
        if variant.proximity_periods:
            for sym in sym_list:
                if sym not in sym_4h_sma:
                    continue
                v_confs[sym] = apply_proximity_boost(
                    v_dirs[sym], v_confs[sym], all_4h[sym], sym_4h_sma[sym],
                    variant.proximity_periods, sym_pad_offsets[sym],
                    variant.proximity_pct,
                )

        # ── Group C: TOTAL3 regime ──
        if variant.total3_enabled:
            t3_sma = total3_smas[variant.total3_sma_period]
            for sym in sym_list:
                v_dirs[sym], v_confs[sym] = apply_total3_filter(
                    v_dirs[sym], v_confs[sym],
                    total3_index, t3_sma, variant.total3_mode,
                )

        # ── Count surviving entries ──
        n_entries = sum(np.count_nonzero(v_dirs[s]) for s in v_dirs)
        entry_counts[variant.name] = n_entries
        pct_kept = n_entries / total_entries_base * 100 if total_entries_base > 0 else 0

        # ── Simulate ──
        cfg = replace(BASE_R7_CFG, name=variant.name, desc=variant.desc)
        result = simulate_r7(
            sym_close, sym_high, sym_low,
            v_dirs, v_confs, sym_moves,
            cfg=cfg, sym_atr=sym_atr,
        )
        results[variant.name] = result
        elapsed = time.time() - t1

        if variant.name.startswith('★'):
            ref_result = result

        # Progress line
        rr_flag = '🏆' if result['actual_rr'] >= 1.5 else ('✓' if result['actual_rr'] >= 1.0 else ' ')
        dd_flag = '✓' if result['max_dd_pct'] <= 10.0 else '!'
        print(f"  [{var_idx+1:2d}/{len(variants)}] {variant.name:<16} "
              f"WR={result['win_rate']:5.1f}%  R:R={result['actual_rr']:.2f}{rr_flag}  "
              f"DD={result['max_dd_pct']:5.1f}%{dd_flag}  Sharpe={result['sharpe']:5.2f}  "
              f"PF={result['profit_factor']:.2f}  Ret={result['total_return_pct']:>+8,.1f}%  "
              f"Trades={result['n_trades']:>5,}  "
              f"Entries={n_entries:>6,}({pct_kept:4.1f}%)  [{elapsed:.1f}s]")

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    print(f"\n\n{'=' * 130}")
    print(f"  R9 ANALYSIS — MA & TOTAL3 IMPACT ON T100_S38_PBT")
    print(f"{'=' * 130}")

    # ── Comparison Table: Delta vs REF ──
    if ref_result is not None:
        print(f"\n  📊 DELTA VS REFERENCE (★REF_T100S38)")
        print(f"  REF baseline: WR={ref_result['win_rate']:.1f}%  R:R={ref_result['actual_rr']:.2f}  "
              f"DD={ref_result['max_dd_pct']:.1f}%  Sharpe={ref_result['sharpe']:.2f}  "
              f"PF={ref_result['profit_factor']:.2f}  Ret={ref_result['total_return_pct']:+,.1f}%  "
              f"Trades={ref_result['n_trades']:,}")
        print()
        print(f"  {'Variant':<16} {'ΔWR':>7} {'ΔR:R':>7} {'ΔDD':>7} "
              f"{'ΔSharpe':>8} {'ΔPF':>7} {'ΔRet':>10} {'Trades':>7} {'Entry%':>7} {'Verdict':>10}")
        print(f"  {'─' * 100}")

        for variant in variants:
            if variant.name.startswith('★'):
                continue
            r = results[variant.name]
            d_wr = r['win_rate'] - ref_result['win_rate']
            d_rr = r['actual_rr'] - ref_result['actual_rr']
            d_dd = r['max_dd_pct'] - ref_result['max_dd_pct']
            d_sh = r['sharpe'] - ref_result['sharpe']
            d_pf = r['profit_factor'] - ref_result['profit_factor']
            d_ret = r['total_return_pct'] - ref_result['total_return_pct']
            entry_pct = entry_counts[variant.name] / total_entries_base * 100

            # Verdict: better/worse/mixed
            goods = sum([d_wr > 0.5, d_rr > 0.05, d_dd < -0.3, d_sh > 0.1, d_pf > 0.02])
            bads = sum([d_wr < -0.5, d_rr < -0.05, d_dd > 0.3, d_sh < -0.1, d_pf < -0.02])
            if goods >= 3 and bads <= 1:
                verdict = '✅ BETTER'
            elif bads >= 3 and goods <= 1:
                verdict = '❌ WORSE'
            else:
                verdict = '🔄 MIXED'

            print(f"  {variant.name:<16} {d_wr:>+6.1f}% {d_rr:>+6.2f} {d_dd:>+6.1f}% "
                  f"{d_sh:>+7.2f} {d_pf:>+6.2f} {d_ret:>+9,.1f}% "
                  f"{r['n_trades']:>6,} {entry_pct:>6.1f}% {verdict:>10}")

    # ── Group-level Summary ──
    print(f"\n\n  📈 GROUP-LEVEL SUMMARY")
    print(f"  {'─' * 90}")

    for group, prefix in [('A: MA_Trend', 'A'), ('B: MA_Prox', 'B'),
                          ('C: TOTAL3', 'C'), ('D: Combos', 'D')]:
        group_results = [(n, r) for n, r in results.items()
                         if n.startswith(prefix) and not n.startswith('★')]
        if not group_results:
            continue

        avg_wr = np.mean([r['win_rate'] for _, r in group_results])
        avg_rr = np.mean([r['actual_rr'] for _, r in group_results])
        avg_dd = np.mean([r['max_dd_pct'] for _, r in group_results])
        avg_sh = np.mean([r['sharpe'] for _, r in group_results])
        avg_pf = np.mean([r['profit_factor'] for _, r in group_results])
        avg_trades = np.mean([r['n_trades'] for _, r in group_results])

        best_name = max(group_results, key=lambda x: x[1]['sharpe'] * x[1]['profit_factor']
                        / max(x[1]['max_dd_pct'], 0.1))[0]
        best = results[best_name]

        if ref_result:
            d_wr = avg_wr - ref_result['win_rate']
            d_rr = avg_rr - ref_result['actual_rr']
            d_dd = avg_dd - ref_result['max_dd_pct']
            d_sh = avg_sh - ref_result['sharpe']

            print(f"\n  {group} ({len(group_results)} variants)")
            print(f"    Avg: WR={avg_wr:.1f}%({d_wr:+.1f})  R:R={avg_rr:.2f}({d_rr:+.2f})  "
                  f"DD={avg_dd:.1f}%({d_dd:+.1f})  Sharpe={avg_sh:.2f}({d_sh:+.2f})  "
                  f"PF={avg_pf:.2f}  Trades={avg_trades:.0f}")
            print(f"    Best: {best_name}  Sharpe={best['sharpe']:.2f}  "
                  f"R:R={best['actual_rr']:.2f}  DD={best['max_dd_pct']:.1f}%  "
                  f"PF={best['profit_factor']:.2f}  Ret={best['total_return_pct']:+,.1f}%")

    # ── TOTAL3 Proxy Regime Analysis ──
    print(f"\n\n  🌍 TOTAL3 PROXY REGIME ANALYSIS")
    print(f"  {'─' * 90}")

    t3_variants = [(n, r) for n, r in results.items()
                   if any(v.total3_enabled and v.name == n for v in variants)]
    for name, r in t3_variants:
        entry_pct = entry_counts[name] / total_entries_base * 100
        blocked_pct = 100 - entry_pct
        if ref_result:
            d_sh = r['sharpe'] - ref_result['sharpe']
            d_pf = r['profit_factor'] - ref_result['profit_factor']
            d_wr = r['win_rate'] - ref_result['win_rate']
            print(f"    {name:<16} Blocked={blocked_pct:.1f}% entries  "
                  f"WR={r['win_rate']:.1f}%({d_wr:+.1f})  "
                  f"Sharpe={r['sharpe']:.2f}({d_sh:+.2f})  "
                  f"PF={r['profit_factor']:.2f}({d_pf:+.2f})  "
                  f"Trades={r['n_trades']:,}")

    # ── Rankings ──
    print(f"\n\n  🏆 FULL RANKINGS BY COMPOSITE SCORE (Sharpe × PF / DD)")
    print(f"  {'─' * 120}")

    scored = []
    for variant in variants:
        r = results[variant.name]
        dd = max(r['max_dd_pct'], 0.1)
        score = r['sharpe'] * r['profit_factor'] / dd
        scored.append((variant.name, score, r))
    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"  {'#':>3} {'Name':<16} {'Score':>7} {'WR':>6} {'R:R':>6} {'DD':>6} "
          f"{'Sharpe':>7} {'PF':>6} {'Ret':>10} {'Trades':>7} {'Entry%':>7}")
    print(f"  {'─' * 100}")

    for rank, (name, score, r) in enumerate(scored, 1):
        entry_pct = entry_counts.get(name, total_entries_base) / total_entries_base * 100
        is_ref = '★' if name.startswith('★') else ' '
        print(f"  {rank:>3}{is_ref}{name:<16} {score:>6.2f} {r['win_rate']:>5.1f}% "
              f"{r['actual_rr']:>5.2f} {r['max_dd_pct']:>5.1f}% "
              f"{r['sharpe']:>6.2f} {r['profit_factor']:>5.2f} "
              f"{r['total_return_pct']:>+9,.1f}% {r['n_trades']:>6,} {entry_pct:>6.1f}%")

    # ── Dominators ──
    if ref_result:
        print(f"\n  ★ VARIANTS THAT DOMINATE REF (WR≥, R:R≥, DD≤, Sharpe≥):")
        dominators = []
        for variant in variants:
            if variant.name.startswith('★'):
                continue
            r = results[variant.name]
            if (r['win_rate'] >= ref_result['win_rate']
                    and r['actual_rr'] >= ref_result['actual_rr']
                    and r['max_dd_pct'] <= ref_result['max_dd_pct']
                    and r['sharpe'] >= ref_result['sharpe']):
                dominators.append((variant.name, r))

        if dominators:
            for name, r in sorted(dominators, key=lambda x: x[1]['sharpe'], reverse=True):
                d_sh = r['sharpe'] - ref_result['sharpe']
                print(f"    {name:<16} WR={r['win_rate']:.1f}%  R:R={r['actual_rr']:.2f}  "
                      f"DD={r['max_dd_pct']:.1f}%  Sharpe={r['sharpe']:.2f}(+{d_sh:.2f})  "
                      f"PF={r['profit_factor']:.2f}  Ret={r['total_return_pct']:+,.1f}%")
        else:
            print("    NONE — REF is Pareto-optimal (MA/TOTAL3 don't fully dominate)")

    # ── Key Insight ──
    winner_name, winner_score, winner = scored[0]
    ref_name = '★REF_T100S38'
    ref_rank = next(i for i, (n, _, _) in enumerate(scored, 1) if n == ref_name)

    print(f"\n\n  {'=' * 90}")
    print(f"  R9 KEY FINDINGS")
    print(f"  {'=' * 90}")
    print(f"  Champion: {winner_name} (composite score: {winner_score:.2f})")
    print(f"  REF rank: #{ref_rank}/{len(scored)}")

    if winner_name != ref_name:
        d = {
            'wr': winner['win_rate'] - ref_result['win_rate'],
            'rr': winner['actual_rr'] - ref_result['actual_rr'],
            'dd': winner['max_dd_pct'] - ref_result['max_dd_pct'],
            'sh': winner['sharpe'] - ref_result['sharpe'],
            'pf': winner['profit_factor'] - ref_result['profit_factor'],
        }
        print(f"  vs REF: WR{d['wr']:+.1f}%  R:R{d['rr']:+.2f}  DD{d['dd']:+.1f}%  "
              f"Sharpe{d['sh']:+.2f}  PF{d['pf']:+.02f}")
        print(f"  Trade count: {winner['n_trades']:,} "
              f"(REF: {ref_result['n_trades']:,})")
    else:
        print(f"  → REF is STILL the best. MA/TOTAL3 filters do NOT improve T100_S38_PBT.")

    # ── Save results ──
    output_path = Path(__file__).parent.parent / 'logs' / 'alpha_search_r9_results.json'
    save_data = {}
    for name, r in results.items():
        sr = {k: v for k, v in r.items() if k != 'equity_curve'}
        sr['entry_count'] = entry_counts.get(name, total_entries_base)
        sr['entry_pct'] = sr['entry_count'] / total_entries_base * 100 if total_entries_base > 0 else 0
        save_data[name] = sr
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'=' * 130}")
    print("  DONE — R9 MA & TOTAL3 PROXY SWEEP COMPLETE")
    print(f"{'=' * 130}")


if __name__ == '__main__':
    main()
