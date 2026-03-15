"""Trade-Event Backtesting Engine — Discrete SL/TP Simulation.

Production-grade backtesting engine for strategies with explicit
stop-loss, take-profit, and max-hold exits.  Unlike VectorizedBacktest
(which consumes pre-baked ±1 signal arrays and uses close-to-close PnL),
this engine:

  1. Takes binary trigger signals → simulates each trade with **intra-bar**
     SL/TP resolution using high/low prices.
  2. Models cross-symbol slot contention (``portfolio_sim``).
  3. Computes production-grade metrics (PF, EV, Calmar, etc.).

Pipeline:
    signals → sim_directional_trades / sim_straddle_trades
    → collect_trades (multi-TF, multi-symbol)
    → portfolio_sim (slot-filtered)
    → compute_metrics

Canonical baseline for VolStraddle (VR Squeeze) strategy.
Extracted from research/active/r31_r40/scaling_r38.py for reuse
and testability.  All research scripts should import from here.

WARNING: Do NOT reimplement these functions.  Subtle details
(momentum=0 handling, squeeze counter reset, dedup order) make
it very easy to produce slightly different numbers.
"""

from __future__ import annotations

import numpy as np
import numba
from concurrent.futures import ThreadPoolExecutor

from quantbox.strategy.vol_straddle.signals import (
    gen_vr_dur_signal,
    gen_vr_signal,
    make_direction_momentum,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Constants — derived from VolStraddleConfig (single source of truth)
# ══════════════════════════════════════════════════════════════════════════════


def _load_trade_sim_constants() -> tuple:
    """Read constants from VolStraddleConfig so trade_sim stays in sync with production."""
    try:
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        cost_dir = cfg.round_trip_cost  # fee+slippage × 2 (entry+exit)
        cost_str = cfg.round_trip_cost * 2  # 4 sides (long+short entry+exit)
        w15 = cfg.warmup_15m
        w1h = int(cfg.warmup_15m / 4)
        w4h = int(cfg.warmup_15m / 16) + 1
        w30 = cfg.warmup_30m
        mh15 = cfg.max_hold_15m
        mh30 = cfg.max_hold_30m
        mh1h = cfg.max_hold_1h
        mh4h = cfg.max_hold_4h
        return cost_dir, cost_str, w15, w30, w1h, w4h, mh15, mh30, mh1h, mh4h
    except Exception:
        # Fallback to hardcoded defaults if config cannot be loaded
        return 0.0008, 0.0016, 200, 100, 50, 13, 192, 96, 48, 12


(
    _COST_DIRECTIONAL,
    _COST_STRADDLE,
    _W15m,
    _W30m,
    _W1h,
    _W4h,
    _MH15m,
    _MH30m,
    _MH1h,
    _MH4h,
) = _load_trade_sim_constants()

COST_DIRECTIONAL: float = _COST_DIRECTIONAL  # 2 × round-trip (entry + exit)
COST_STRADDLE: float = _COST_STRADDLE  # 4 sides (long + short entry/exit)

# Warmup periods — bars to skip before first signal (long ATR needs data).
WARMUP_15m: int = _W15m
WARMUP_30m: int = _W30m
WARMUP_1h: int = _W1h
WARMUP_4h: int = _W4h

# Timeframe definitions: (label, ratio_vs_15m, warmup, max_hold_dir, max_hold_straddle)
TF_15m = ("15m", 1, WARMUP_15m, _MH15m, _MH15m * 2)
TF_30m = ("30m", 2, WARMUP_30m, _MH30m, _MH30m * 2)
TF_1h = ("1h", 4, WARMUP_1h, _MH1h, _MH1h * 2)
TF_4h = ("4h", 16, WARMUP_4h, _MH4h, _MH4h * 2)
G_3TF = [TF_15m, TF_1h, TF_4h]
G_4TF = [TF_15m, TF_30m, TF_1h, TF_4h]


# ══════════════════════════════════════════════════════════════════════════════
#  Tiered Slippage Model
#  Based on Binance USDT-M perpetual 24h volume tiers.
#  Calibrated for limit-chaser entries that enter on breakouts
#  (price moving fast → worse fill than flat 0.08% assumes).
# ══════════════════════════════════════════════════════════════════════════════

# High-liquidity symbols: consistently >$500M daily volume on USDT-M perps.
# These can be filled near mid-price with limit-chaser even during squeezes.
_HIGH_LIQ_SYMS: frozenset[str] = frozenset(
    {
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "DOTUSDT",
        "MATICUSDT",
        "LTCUSDT",
        "UNIUSDT",
        "ATOMUSDT",
        "FILUSDT",
        "NEARUSDT",
        "FTMUSDT",
        "TRXUSDT",
    }
)

# Thin/small-cap symbols: typically <$50M daily volume.
# Bid-ask spreads widen sharply during vol spikes; limit-chaser often fills
# near the far side of the spread. 0.16% round-trip is conservative.
_LOW_LIQ_SYMS: frozenset[str] = frozenset(
    {
        "BANDUSDT",
        "BELUSDT",
        "ICXUSDT",
        "IOSTUSDT",
        "IOTAUSDT",
        "KNCUSDT",
        "LRCUSDT",
        "RLCUSDT",
        "RSRUSDT",
        "SKLUSDT",
        "STORJUSDT",
        "TRBUSDT",
        "ZENUSDT",
        "ZRXUSDT",
    }
)

# Everything else (mid-tier alts) falls into medium liquidity: 0.12%.
_COST_HIGH_LIQ: float = 0.0008  # 0.08% round-trip (current flat default)
_COST_MED_LIQ: float = 0.0012  # 0.12% round-trip
_COST_LOW_LIQ: float = 0.0016  # 0.16% round-trip (thin markets)


def get_symbol_slippage(symbol: str, conservative: bool = False) -> float:
    """Return round-trip cost estimate for a symbol.

    Assigns cost by 24h volume tier on Binance USDT-M perpetuals:
      HIGH (≥$500M/day): 0.08%  — fills near mid even during squeezes
      MED  ($50-500M/day): 0.12% — some spread widening on breakouts
      LOW  (<$50M/day): 0.16%  — thin markets; spread widens sharply

    Args:
        symbol: Trading symbol e.g. 'BTCUSDT'.
        conservative: If True, add 50% buffer to the tier cost.

    Returns:
        Round-trip cost fraction (e.g. 0.0008 = 0.08%).
    """
    if symbol in _HIGH_LIQ_SYMS:
        cost = _COST_HIGH_LIQ
    elif symbol in _LOW_LIQ_SYMS:
        cost = _COST_LOW_LIQ
    else:
        cost = _COST_MED_LIQ
    if conservative:
        cost *= 1.5
    return cost


def simulate_with_tiered_slippage(
    cfg: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
    mode: str = "flat",
) -> list[tuple[str, int, int, float]]:
    """Run collect_trades then adjust PnL by per-symbol slippage tier.

    The base ``collect_trades()`` bakes in a flat cost (``COST_DIRECTIONAL``
    or ``COST_STRADDLE``).  This function replaces that flat cost with a
    per-symbol tiered cost by adjusting each trade's PnL:

        pnl_tiered = pnl_flat - (tiered_cost - flat_cost)

    For straddle trades (4 legs) the adjustment scales by 2×:
        tiered_str_cost = 2 × get_symbol_slippage(sym, ...)
        delta = tiered_str_cost - COST_STRADDLE

    Args:
        cfg: Strategy config dict (same as collect_trades).
        data_15m: 15m OHLCV data dict.
        data_1h: 1h OHLCV data dict.
        data_4h: 4h OHLCV data dict.
        strat: "D_mom8" or "S_d8".
        mode: "flat" (unchanged), "tiered" (per-symbol), or
              "conservative" (tiered + 50% buffer).

    Returns:
        List of (symbol, entry_bar_15m, exit_bar_15m, pnl_float).
    """
    if mode not in ("flat", "tiered", "conservative"):
        raise ValueError(f"mode must be 'flat', 'tiered', or 'conservative', got {mode!r}")

    trades = collect_trades(cfg, data_15m, data_1h, data_4h, strat)

    if mode == "flat":
        return trades

    is_straddle = strat == "S_d8"
    conservative = mode == "conservative"
    flat_cost = COST_STRADDLE if is_straddle else COST_DIRECTIONAL

    adjusted: list[tuple[str, int, int, float]] = []
    for sym, entry, exit_b, pnl in trades:
        sym_dir_cost = get_symbol_slippage(sym, conservative=conservative)
        tiered_cost = sym_dir_cost * 2 if is_straddle else sym_dir_cost
        delta = tiered_cost - flat_cost
        adjusted.append((sym, entry, exit_b, pnl - delta))
    return adjusted


# ══════════════════════════════════════════════════════════════════════════════
#  Numba-Compiled Trade Simulators
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_trades(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sig: np.ndarray,
    sl: float,
    tp: float,
    max_hold: int,
    cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate straddle (long+short simultaneous) trades with SL/TP.

    Opens both long and short at the trigger bar's close, then uses
    **intra-bar high/low** to check SL/TP on each subsequent bar.
    Both legs must close before the trade ends.

    Args:
        close: Close price array.
        high: High price array.
        low: Low price array.
        sig: Binary signal array (1 = enter, 0 = skip).
        sl: Stop-loss distance as fraction (e.g. 0.015 = 1.5%).
        tp: Take-profit distance as fraction (e.g. 0.05 = 5%).
        max_hold: Maximum bars to hold before forced exit.
        cost: Total round-trip cost (e.g. 0.0016 for straddle).

    Returns:
        (pnl, entry_bars, exit_bars) — each 1-D array of length N_trades.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue
        l_sl = entry * (1.0 - sl)
        l_tp = entry * (1.0 + tp)
        s_sl = entry * (1.0 + sl)
        s_tp = entry * (1.0 - tp)
        la = True  # long leg active
        sa = True  # short leg active
        le = 0.0  # long exit pnl
        se = 0.0  # short exit pnl
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                if low[j] <= l_sl:
                    le = -sl
                    la = False
                elif high[j] >= l_tp:
                    le = tp
                    la = False
            if sa:
                if high[j] >= s_sl:
                    se = -sl
                    sa = False
                elif low[j] <= s_tp:
                    se = tp
                    sa = False
            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1
        if la:
            le = (close[eb] - entry) / entry
        if sa:
            se = (entry - close[eb]) / entry
        net = le + se - cost
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_trades(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sig: np.ndarray,
    dire: np.ndarray,
    sl: float,
    tp: float,
    max_hold: int,
    cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate directional trades with intra-bar SL/TP.

    Enters at trigger bar's close in the direction given by ``dire``.
    Uses **intra-bar high/low** to detect SL/TP hits.

    Args:
        close: Close price array.
        high: High price array.
        low: Low price array.
        sig: Binary signal array (1 = potential entry).
        dire: Direction array (+1 = long, -1 = short, 0 = skip).
        sl: Stop-loss distance as fraction (e.g. 0.01 = 1%).
        tp: Take-profit distance as fraction (e.g. 0.05 = 5%).
        max_hold: Maximum bars to hold.
        cost: Round-trip cost (e.g. 0.0008).

    Returns:
        (pnl, entry_bars, exit_bars) — each 1-D array of length N_trades.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0 or dire[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue
        d = dire[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    break
        else:
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry
        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Trailing Stop — Directional
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_directional_trailing(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sig: np.ndarray,
    dire: np.ndarray,
    sl: float,
    tp: float,
    max_hold: int,
    cost: float,
    trail_activate: float,  # activate trailing after this % gain
    trail_distance: float,  # trailing stop distance from peak
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional sim with trailing stop.

    Once unrealized gain reaches `trail_activate`, stop-loss becomes
    a trailing stop at `trail_distance` below the peak gain.
    TP is kept as a hard ceiling.

    Args:
        trail_activate: e.g. 0.02 means trailing activates after +2% gain
        trail_distance: e.g. 0.01 means trail 1% below peak
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0 or dire[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue
        d = dire[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        trailing_active = False
        peak_gain = 0.0

        for j in range(i + 1, end):
            if d == 1:
                # SL checked FIRST (server SL fires intra-bar,
                # trail fires at bar close; SL always has priority)
                if low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break

                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain

                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True

                if trailing_active:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break

                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
            else:
                # SHORT — SL checked FIRST (same rationale)
                if high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break

                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain

                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True

                if trailing_active:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break

                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    break
        else:
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Trailing Stop — Straddle
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_trailing(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sig: np.ndarray,
    sl: float,
    tp: float,
    max_hold: int,
    cost: float,
    trail_activate: float,  # activate trailing on winning leg after this gain
    trail_distance: float,  # trail distance from peak of winning leg
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straddle sim with trailing on winning legs.

    Each leg gets its own trailing stop once it gains `trail_activate`.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        la = True
        sa = True
        le = 0.0
        se = 0.0
        l_peak = 0.0
        s_peak = 0.0
        l_trailing = False
        s_trailing = False
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            # Long leg — SL checked FIRST (server SL fires intra-bar,
            # trail fires at bar close; SL always has priority)
            if la:
                if low[j] <= entry * (1.0 - sl):
                    le = -sl
                    la = False
                else:
                    l_gain = (high[j] - entry) / entry
                    if l_gain > l_peak:
                        l_peak = l_gain
                    if not l_trailing and l_peak >= trail_activate:
                        l_trailing = True
                    if l_trailing:
                        trail_lvl = entry * (1.0 + l_peak - trail_distance)
                        if low[j] <= trail_lvl:
                            le = max(l_peak - trail_distance, -sl)
                            la = False
                    if la and high[j] >= entry * (1.0 + tp):
                        le = tp
                        la = False

            # Short leg — SL checked FIRST (same rationale)
            if sa:
                if high[j] >= entry * (1.0 + sl):
                    se = -sl
                    sa = False
                else:
                    s_gain = (entry - low[j]) / entry
                    if s_gain > s_peak:
                        s_peak = s_gain
                    if not s_trailing and s_peak >= trail_activate:
                        s_trailing = True
                    if s_trailing:
                        trail_lvl = entry * (1.0 - s_peak + trail_distance)
                        if high[j] >= trail_lvl:
                            se = max(s_peak - trail_distance, -sl)
                            sa = False
                    if sa and low[j] <= entry * (1.0 - tp):
                        se = tp
                        sa = False

            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1
        if la:
            le = (close[eb] - entry) / entry
        if sa:
            se = (entry - close[eb]) / entry
        net = le + se - cost
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Trade Collection — Multi-TF, Multi-Symbol
# ══════════════════════════════════════════════════════════════════════════════


def collect_trades(
    cfg: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
    *,
    tf_group: list | None = None,
    tf_priority: bool = False,
) -> list[tuple[str, int, int, float]]:
    """Collect all trades as (symbol, entry_15m, exit_15m, pnl).

    Scans multiple timeframes, converts bar indices to 15m base,
    and deduplicates overlapping trades per symbol.

    Args:
        cfg: Strategy config dict with keys:
            sp, lp, d_vr, s_vr, s_dur, d_sl, s_sl, tp.
            Build from VolStraddleConfig — do NOT hardcode values.
        data_15m: {symbol: {close, high, low, vr_X_Y, mom_8, ...}}.
        data_1h: Same structure, 1h resampled.
        data_4h: Same structure, 4h resampled.
        strat: "D_mom8" or "S_d8".
        tf_group: List of TF tuples to scan. None → G_3TF (backward-compat).
            Use G_4TF for production parity (includes 30m).
        tf_priority: If True, higher TFs take priority — trades from a higher
            TF block lower-TF trades on the same symbol during the same
            time window (matches live on_bar first-match-wins behavior).

    Returns:
        List of (symbol, entry_bar_15m, exit_bar_15m, pnl_float).
    """
    # ── Guard: verify required keys come from config, not hardcoded ──────────
    _REQUIRED_CFG_KEYS = {"sp", "lp", "d_sl", "s_sl", "tp", "d_vr", "s_vr", "s_dur"}
    _missing = _REQUIRED_CFG_KEYS - cfg.keys()
    if _missing:
        raise ValueError(
            f"collect_trades() cfg dict missing required keys: {sorted(_missing)}.\n"
            "Build from VolStraddleConfig():\n"
            "  from quantbox.strategy.vol_straddle.config import VolStraddleConfig\n"
            "  cfg_obj = VolStraddleConfig()\n"
            "  cfg = {'sp': cfg_obj.vr_short, 'lp': cfg_obj.vr_long,\n"
            "         'd_vr': cfg_obj.d_vr_threshold, 's_vr': cfg_obj.s_vr_threshold,\n"
            "         's_dur': cfg_obj.s_squeeze_dur, 'd_sl': cfg_obj.d_sl,\n"
            "         's_sl': cfg_obj.s_sl, 'tp': cfg_obj.d_tp}"
        )
    if not (0 < cfg["d_sl"] < 0.20):
        raise ValueError(f"cfg['d_sl']={cfg['d_sl']} out of valid range (0, 0.20). Check config source.")
    if not (0 < cfg["tp"] <= 1.0):
        raise ValueError(f"cfg['tp']={cfg['tp']} out of valid range (0, 1.0]. Check config source.")
    if strat not in ("D_mom8", "S_d8"):
        raise ValueError(f"strat must be 'D_mom8' or 'S_d8', got {strat!r}")
    # ─────────────────────────────────────────────────────────────────────────

    sp = cfg["sp"]
    lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    is_straddle = strat == "S_d8"

    d_sl = cfg["d_sl"]
    s_sl = cfg["s_sl"]
    tp = cfg["tp"]
    d_vr = cfg["d_vr"]
    s_vr = cfg["s_vr"]
    s_dur = cfg["s_dur"]

    # Resolve TF group — default G_3TF for backward compat
    active_tfs = tf_group if tf_group is not None else G_3TF

    # Build tf_data_map including 30m if needed
    tf_data_map: dict[int, dict] = {1: data_15m, 4: data_1h, 16: data_4h}
    if any(ratio == 2 for _, ratio, *_ in active_tfs):
        tf_data_map[2] = build_resampled(data_15m, 2)

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        if tf_priority:
            # TF priority mode: process higher TFs first (descending ratio),
            # each accepted trade blocks lower TFs for the same time window.
            # This matches live on_bar() first-match-wins behavior.
            sorted_tfs = sorted(active_tfs, key=lambda t: t[1], reverse=True)
            occupied: list[tuple[int, int]] = []  # (entry_15m, exit_15m) of accepted trades

            for tf_label, ratio, warmup, mh_d, mh_s in sorted_tfs:
                d_tf = tf_data_map.get(ratio)
                if d_tf is None or sym not in d_tf:
                    continue
                d = d_tf[sym]
                c = d["close"]
                h = d["high"]
                lo = d["low"]
                vr = d.get(vr_key)
                if vr is None:
                    continue

                sl = s_sl if is_straddle else d_sl
                mh = mh_s if is_straddle else mh_d

                if is_straddle:
                    sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
                    p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, COST_STRADDLE)
                else:
                    sig = gen_vr_signal(vr, d_vr, warmup)
                    dire = make_direction_momentum(sig, d["mom_8"])
                    p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL)

                for k in range(len(p)):
                    entry_15m = int(e[k]) * ratio
                    exit_15m = int(x[k]) * ratio + (ratio - 1)
                    # Check if this trade overlaps with any already-accepted higher-TF trade
                    blocked = False
                    for oe, ox in occupied:
                        if entry_15m <= ox and exit_15m >= oe:
                            blocked = True
                            break
                    if not blocked:
                        all_trades.append((sym, entry_15m, exit_15m, float(p[k])))
                        occupied.append((entry_15m, exit_15m))

            # Sort occupied for consistency (already mostly sorted due to TF processing order)
            occupied.sort()
        else:
            # Original behavior: collect from all TFs independently, then time-dedup
            sym_trades: list[tuple[int, int, float]] = []

            for tf_label, ratio, warmup, mh_d, mh_s in active_tfs:
                d_tf = tf_data_map.get(ratio)
                if d_tf is None or sym not in d_tf:
                    continue
                d = d_tf[sym]
                c = d["close"]
                h = d["high"]
                lo = d["low"]
                vr = d.get(vr_key)
                if vr is None:
                    continue

                sl = s_sl if is_straddle else d_sl
                mh = mh_s if is_straddle else mh_d

                if is_straddle:
                    sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
                    p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, COST_STRADDLE)
                else:
                    sig = gen_vr_signal(vr, d_vr, warmup)
                    dire = make_direction_momentum(sig, d["mom_8"])
                    p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL)

                for k in range(len(p)):
                    entry_15m = int(e[k]) * ratio
                    exit_15m = int(x[k]) * ratio + (ratio - 1)
                    sym_trades.append((entry_15m, exit_15m, float(p[k])))

            # Per-symbol time dedup: sort by entry, reject overlapping
            sym_trades.sort(key=lambda x: x[0])
            accepted: list[tuple[int, int]] = []
            for entry_15m, exit_15m, pnl_val in sym_trades:
                conflict = False
                for ae, ax in accepted:
                    if entry_15m <= ax and exit_15m >= ae:
                        conflict = True
                        break
                if not conflict:
                    all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                    accepted.append((entry_15m, exit_15m))

    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Portfolio Simulation — Slot Allocation
# ══════════════════════════════════════════════════════════════════════════════


def portfolio_sim(
    all_trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
    straddle_syms: frozenset[str] | None = None,
) -> np.ndarray:
    """Filter trades by max concurrent positions, no same-symbol overlap.

    Sorts trades chronologically, then greedily accepts trades that:
      1. Don't overlap with an existing position on the same symbol.
      2. Don't exceed ``max_concurrent`` total open slots.

    Args:
        all_trades: List of (symbol, entry_bar, exit_bar, pnl).
        max_concurrent: Max simultaneous slots (e.g. 10 for live config).
        straddle_syms: Optional set of symbols whose trades are S_d8 straddles.
            Each straddle trade consumes 2 slots (LONG+SHORT legs) matching
            live behaviour where each leg occupies one of the 10 slots.
            Usage for mixed VR+STRD backtest::

                strd_trades = collect_trades(cfg, ..., "S_d8")
                strd_syms = frozenset(sym for sym, *_ in strd_trades)
                pnl = portfolio_sim(vr_trades + strd_trades, 10, straddle_syms=strd_syms)

            For standalone STRD-only backtests use ``max_concurrent=5`` instead
            (r79 convention — avoids the need for this parameter).

    Returns:
        1-D array of accepted trade PnLs (in chronological order).
    """
    trades = sorted(all_trades, key=lambda x: x[1])
    accepted_pnl: list[float] = []
    active: list[tuple[str, int]] = []  # (symbol, exit_bar)

    for sym, entry, exit_b, pnl_val in trades:
        slots = 2 if (straddle_syms and sym in straddle_syms) else 1
        # Expire finished trades
        active = [(s, e) for s, e in active if e > entry]
        # Same-symbol block
        sym_active = any(s == sym for s, _ in active)
        if sym_active:
            continue
        # Slot limit: straddle needs 2 free slots
        if len(active) + slots > max_concurrent:
            continue
        # Consume `slots` entries so the active slot count is accurate
        active.extend([(sym, exit_b)] * slots)
        accepted_pnl.append(pnl_val)

    return np.array(accepted_pnl) if accepted_pnl else np.empty(0)


def portfolio_sim_with_details(
    all_trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
    straddle_syms: frozenset[str] | None = None,
) -> tuple[np.ndarray, list[tuple[str, int, int, float]]]:
    """Portfolio sim that also returns accepted trades with metadata.

    Same logic as ``portfolio_sim`` but additionally returns the list
    of accepted trade tuples for further analysis (symbol breakdown,
    timing analysis, correlation studies, etc.).

    Args:
        all_trades: List of (symbol, entry_bar, exit_bar, pnl).
        max_concurrent: Max simultaneous slots.
        straddle_syms: Optional set of straddle symbols (consume 2 slots each).
            See ``portfolio_sim`` docstring for usage.

    Returns:
        (pnl_array, accepted_trades_list).
    """
    trades = sorted(all_trades, key=lambda x: x[1])
    accepted: list[tuple[str, int, int, float]] = []
    active: list[tuple[str, int]] = []

    for sym, entry, exit_b, pnl_val in trades:
        slots = 2 if (straddle_syms and sym in straddle_syms) else 1
        active = [(s, e) for s, e in active if e > entry]
        sym_active = any(s == sym for s, _ in active)
        if sym_active:
            continue
        if len(active) + slots > max_concurrent:
            continue
        active.extend([(sym, exit_b)] * slots)
        accepted.append((sym, entry, exit_b, pnl_val))

    pnl = np.array([t[3] for t in accepted]) if accepted else np.empty(0)
    return pnl, accepted


def get_straddle_syms(strd_trades: list[tuple[str, int, int, float]]) -> frozenset[str]:
    """Return the set of symbols present in a straddle trade list.

    Convenience helper for building the ``straddle_syms`` argument to
    ``portfolio_sim`` / ``portfolio_sim_with_details`` in mixed backtests.

    Example::

        strd_trades = collect_trades(cfg, data_15m, data_1h, data_4h, "S_d8")
        vr_trades   = collect_trades(cfg, data_15m, data_1h, data_4h, "D_mom8")
        strd_syms   = get_straddle_syms(strd_trades)
        pnl = portfolio_sim(vr_trades + strd_trades, 10, straddle_syms=strd_syms)

    Args:
        strd_trades: Trades returned by ``collect_trades(..., strat="S_d8")``.

    Returns:
        frozenset of symbol strings.
    """
    return frozenset(t[0] for t in strd_trades)


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════


def pf_from_pnl(pnl: np.ndarray) -> float:
    """Compute Profit Factor = gross_profit / gross_loss.

    Args:
        pnl: Array of per-trade PnL values.

    Returns:
        Profit Factor (999.0 if no losing trades, 0.0 if empty).
    """
    if len(pnl) == 0:
        return 0.0
    pos = pnl[pnl > 0].sum()
    neg = abs(pnl[pnl < 0].sum())
    return pos / neg if neg > 0 else 999.0


def compute_metrics(pnl: np.ndarray, data_years: float) -> dict:
    """Compute comprehensive performance metrics from PnL array.

    Uses 1x compound model: ``equity *= (1 + pnl_per_trade)``.
    This is the canonical metrics function for all VR research.

    Args:
        pnl: Array of per-trade PnL values.
        data_years: Duration of data in years (for annualization).

    Returns:
        Dict with keys: N, PF, EV_bp, t, WR, RR, avg_win_pct,
        avg_loss_pct, ret_pct, max_dd_pct, tpd, ann_ret_pct, calmar.
    """
    n = len(pnl)
    if n < 2:
        return {
            "N": n,
            "PF": 0.0,
            "EV_bp": 0.0,
            "t": 0.0,
            "WR": 0.0,
            "RR": 0.0,
            "ret_pct": 0.0,
            "max_dd_pct": 0.0,
            "tpd": 0.0,
            "ann_ret_pct": 0.0,
            "calmar": 0.0,
        }

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    wr = len(wins) / n * 100
    avg_win = float(np.mean(wins)) * 100 if len(wins) > 0 else 0.0
    avg_loss = abs(float(np.mean(losses))) * 100 if len(losses) > 0 else 1e-9
    rr = avg_win / avg_loss if avg_loss > 0 else 999.0
    pf = pf_from_pnl(pnl)
    ev = float(np.mean(pnl)) * 10000  # basis points
    se = float(np.std(pnl) / np.sqrt(n)) * 10000
    t_stat = ev / se if se > 0 else 0.0

    # Equity curve (1x compound) — vectorized via numpy cumproduct
    equity_curve = np.cumprod(1.0 + pnl)
    equity = float(equity_curve[-1])
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / peaks
    max_dd = float(np.max(drawdowns))

    total_ret = (equity - 1.0) * 100
    tpd = n / (data_years * 365) if data_years > 0 else 0.0
    ann_ret = (equity ** (1.0 / data_years) - 1.0) * 100 if data_years > 0 else 0.0
    calmar = ann_ret / (max_dd * 100) if max_dd > 0 else 0.0

    return {
        "N": n,
        "PF": round(pf, 3),
        "EV_bp": round(ev, 1),
        "t": round(t_stat, 2),
        "WR": round(wr, 1),
        "RR": round(rr, 2),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "ret_pct": round(total_ret, 1),
        "max_dd_pct": round(max_dd * 100, 1),
        "tpd": round(tpd, 2),
        "ann_ret_pct": round(ann_ret, 1),
        "calmar": round(calmar, 1),
    }


def bootstrap_pf(pnl: np.ndarray, n_boot: int = 1000) -> dict:
    """Bootstrap confidence interval for Profit Factor.

    Args:
        pnl: Trade PnL array (min 10 trades for meaningful result).
        n_boot: Number of bootstrap samples.

    Returns:
        Dict with P_PF_ge_1 (% chance PF≥1), CI_lo, CI_hi, median.
    """
    if len(pnl) < 10:
        return {"P_PF_ge_1": 0.0, "CI_lo": 0.0, "CI_hi": 0.0, "median": 0.0}
    rng = np.random.default_rng(42)
    n = len(pnl)
    pfs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pfs[b] = pf_from_pnl(pnl[idx])
    return {
        "P_PF_ge_1": round(float(np.mean(pfs >= 1.0)) * 100, 1),
        "CI_lo": round(float(np.percentile(pfs, 2.5)), 3),
        "CI_hi": round(float(np.percentile(pfs, 97.5)), 3),
        "median": round(float(np.median(pfs)), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Data Helpers (used by research scripts)
# ══════════════════════════════════════════════════════════════════════════════


def build_resampled(data_15m: dict, ratio: int) -> dict:
    """Resample all symbols from 15m to higher TF.

    Args:
        data_15m: {symbol: {close, high, low, volume, bars, days}}.
        ratio: Resampling ratio (4 for 1h, 16 for 4h).

    Returns:
        Same structure with resampled OHLCV arrays.
    """
    from quantbox.strategy.vol_straddle.signals import resample_ohlcv

    data_htf: dict = {}
    for sym, d in data_15m.items():
        c, h, lo, v = resample_ohlcv(
            d["close"],
            d["high"],
            d["low"],
            d["volume"],
            ratio,
        )
        if len(c) < 100:
            continue
        data_htf[sym] = {
            "close": c,
            "high": h,
            "low": lo,
            "volume": v,
            "bars": len(c),
            "days": d.get("days", 0),
        }
    return data_htf


def _vol_percentile_vec(vol: np.ndarray, lookback: int) -> np.ndarray:
    """Rolling percentile-rank of volume — vectorized via sliding window.

    For each bar i, computes what fraction of the prior ``lookback`` bars
    (including bar i) had volume strictly below vol[i].  Non-positive
    volume bars are excluded from the denominator.

    This is a drop-in replacement for the triple-nested Python loop that
    previously made ``precompute`` the dominant bottleneck (~100× faster).

    Args:
        vol: Volume array.
        lookback: Rolling window size.

    Returns:
        Percentile-rank array in [0, 100]; 50.0 for unavailable bars.
    """
    n = len(vol)
    vol_pct = np.full(n, 50.0)
    if n < lookback:
        return vol_pct
    # sliding_window_view: shape (n-lookback+1, lookback)
    # windows[j] = vol[j : j+lookback], corresponds to position i = j+lookback-1
    windows = np.lib.stride_tricks.sliding_window_view(vol, lookback)
    cur = vol[lookback - 1 :, None]  # broadcast column
    valid = (windows > 0).sum(axis=1)  # count non-zero per window
    below = (windows < cur).sum(axis=1)  # count strictly-below-cur
    pct_vals = np.where(valid > 0, below / valid * 100.0, 50.0)
    vol_pct[lookback - 1 :] = pct_vals
    return vol_pct


def precompute(
    data: dict,
    vr_sp: int = 7,
    vr_lp: int = 50,
    mom_lb: int = 8,
    sq_thresh: float = 0.55,
    n_workers: int | None = None,
) -> None:
    """Precompute VR, momentum, vol-percentile, squeeze-duration in-place.

    Args:
        data: {symbol: {close, high, low, volume, ...}}.
        vr_sp: Short ATR period for VR.
        vr_lp: Long ATR period for VR.
        mom_lb: Momentum lookback.
        sq_thresh: Squeeze duration VR threshold.
        n_workers: Thread count for parallel symbol processing.  ``None``
            lets ThreadPoolExecutor choose (``min(32, cpu_count+4)``).
            Pass ``1`` to disable parallelism.
    """
    from quantbox.strategy.vol_straddle.signals import (
        compute_momentum,
        compute_squeeze_dur,
        compute_vr,
    )

    def _process_sym(item: tuple) -> None:
        sym, d = item
        c, h, lo, vol = d["close"], d["high"], d["low"], d["volume"]
        d["vr"] = compute_vr(c, h, lo, vr_sp, vr_lp)
        d["mom_8"] = compute_momentum(c, mom_lb)
        lookback = min(200, len(vol) // 2)
        d["vol_pct"] = _vol_percentile_vec(vol, lookback)
        d["sq_dur_055"] = compute_squeeze_dur(d["vr"], sq_thresh)

    items = list(data.items())
    if n_workers == 1 or len(items) <= 1:
        for item in items:
            _process_sym(item)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_process_sym, items))


def precompute_all_vr(
    datasets: list[dict],
    pairs: list[tuple[int, int]],
) -> None:
    """Precompute VR for all ATR short/long pairs across all datasets.

    Args:
        datasets: List of dataset dicts (data_15m, data_1h, data_4h).
        pairs: List of (short_period, long_period) tuples.
    """
    from quantbox.strategy.vol_straddle.signals import compute_vr

    for dataset in datasets:
        for sym, d in dataset.items():
            c, h, lo = d["close"], d["high"], d["low"]
            for sp, lp in pairs:
                key = f"vr_{sp}_{lp}"
                if key not in d:
                    d[key] = compute_vr(c, h, lo, sp, lp)
