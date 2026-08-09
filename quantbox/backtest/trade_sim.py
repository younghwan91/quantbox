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
    signals → sim_directional_trades
    → collect_trades (multi-TF, multi-symbol)
    → portfolio_sim (slot-filtered)
    → compute_metrics

Canonical baseline for VR Directional (D_mom8) strategy.
Extracted from research/active/r31_r40/scaling_r38.py for reuse
and testability.  All research scripts should import from here.

WARNING: Do NOT reimplement these functions.  Subtle details
(momentum=0 handling, squeeze counter reset, dedup order) make
it very easy to produce slightly different numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numba
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig

from quantbox.strategy.vol_straddle.signals import (
    gen_vr_signal,
    make_direction_momentum,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Constants — derived from VolStraddleConfig (single source of truth)
# ══════════════════════════════════════════════════════════════════════════════


def get_trade_sim_config() -> dict:
    """Read trade sim constants fresh from VolStraddleConfig.

    G4 parity fix: called per-invocation instead of cached at module load.
    Returns a dict with cost, warmup, max_hold, and TF tuples.
    """
    try:
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        cost_dir = cfg.round_trip_cost
        w15 = cfg.warmup_15m
        w30 = cfg.warmup_30m
        w1h = int(cfg.warmup_15m / 4)
        w4h = int(cfg.warmup_15m / 16) + 1
        mh15 = cfg.max_hold_15m
        mh30 = cfg.max_hold_30m
        mh1h = cfg.max_hold_1h
        mh4h = cfg.max_hold_4h
    except Exception:
        cost_dir = 0.0008
        w15, w30, w1h, w4h = 200, 100, 50, 13
        mh15, mh30, mh1h, mh4h = 192, 96, 48, 12

    return {
        "cost_directional": cost_dir,
        "tf_15m": ("15m", 1, w15, mh15),
        "tf_30m": ("30m", 2, w30, mh30),
        "tf_1h": ("1h", 4, w1h, mh1h),
        "tf_4h": ("4h", 16, w4h, mh4h),
    }


def _load_trade_sim_constants() -> tuple:
    """Legacy: read constants at module load. Use get_trade_sim_config() for fresh values."""
    c = get_trade_sim_config()
    tf15, tf30, tf1h, tf4h = c["tf_15m"], c["tf_30m"], c["tf_1h"], c["tf_4h"]
    return (
        c["cost_directional"],
        tf15[2],
        tf30[2],
        tf1h[2],
        tf4h[2],  # warmups
        tf15[3],
        tf30[3],
        tf1h[3],
        tf4h[3],  # max_holds
    )


(
    _COST_DIRECTIONAL,
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

# Warmup periods — bars to skip before first signal (long ATR needs data).
WARMUP_15m: int = _W15m
WARMUP_30m: int = _W30m
WARMUP_1h: int = _W1h
WARMUP_4h: int = _W4h

# Timeframe definitions: (label, ratio_vs_15m, warmup, max_hold)
TF_15m = ("15m", 1, WARMUP_15m, _MH15m)
TF_30m = ("30m", 2, WARMUP_30m, _MH30m)
TF_1h = ("1h", 4, WARMUP_1h, _MH1h)
TF_4h = ("4h", 16, WARMUP_4h, _MH4h)
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

    The base ``collect_trades()`` bakes in a flat cost (``COST_DIRECTIONAL``).
    This function replaces that flat cost with a per-symbol tiered cost by
    adjusting each trade's PnL:

        pnl_tiered = pnl_flat - (tiered_cost - flat_cost)

    Args:
        cfg: Strategy config dict (same as collect_trades).
        data_15m: 15m OHLCV data dict.
        data_1h: 1h OHLCV data dict.
        data_4h: 4h OHLCV data dict.
        strat: "D_mom8".
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

    conservative = mode == "conservative"
    flat_cost = COST_DIRECTIONAL

    adjusted: list[tuple[str, int, int, float]] = []
    for sym, entry, exit_b, pnl in trades:
        sym_dir_cost = get_symbol_slippage(sym, conservative=conservative)
        delta = sym_dir_cost - flat_cost
        adjusted.append((sym, entry, exit_b, pnl - delta))
    return adjusted


# ══════════════════════════════════════════════════════════════════════════════
#  Numba-Compiled Trade Simulators
# ══════════════════════════════════════════════════════════════════════════════


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
    pnl: np.ndarray = np.empty(cap, dtype=np.float64)
    ent: np.ndarray = np.empty(cap, dtype=np.int64)
    ext: np.ndarray = np.empty(cap, dtype=np.int64)
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
    # 위임: 공용 엔진 sim_core 가 유일 구현이다 (중복 sim 제거).
    # 기존 백테스트 숫자 불변은 tests/test_golden_trades.py 가 실데이터로 지킨다.
    from quantbox.backtest.sim_core import directional_conditions, sim_drive

    bar_kind, entry_ok, direction = directional_conditions(close, sig, dire)
    pnl, ent, ext, _dirs = sim_drive(
        close,
        high,
        low,
        bar_kind,
        entry_ok,
        direction,
        1,
        0,  # min_duration=1, max_duration=off (게이팅은 entry_ok가 전담)
        sl,
        tp,
        trail_activate,
        trail_distance,
        max_hold,
        cost,
        0,  # warmup=0 (이 함수는 warmup 인자가 없다)
        2,
        0,  # cap_divisor=2 (n//2+1), cooldown=off
    )
    return pnl, ent, ext


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
    tf_priority: bool = True,
) -> list[tuple[str, int, int, float]]:
    """Collect all trades as (symbol, entry_15m, exit_15m, pnl).

    Scans multiple timeframes, converts bar indices to 15m base,
    and deduplicates overlapping trades per symbol.

    Args:
        cfg: Strategy config dict with keys:
            sp, lp, d_vr, d_sl, tp,
            d_trail_activate, d_trail_distance.
            Legacy keys (s_vr, s_dur, s_sl, s_trail_*) are accepted but ignored.
            Build from VolStraddleConfig — do NOT hardcode values.
        data_15m: {symbol: {close, high, low, vr_X_Y, mom_8, ...}}.
        data_1h: Same structure, 1h resampled.
        data_4h: Same structure, 4h resampled.
        strat: "D_mom8".
        tf_group: List of TF tuples to scan. None → G_3TF (backward-compat).
            Use G_4TF for production parity (includes 30m).
        tf_priority: If True (default, G1 parity fix), higher TFs take priority —
            trades from a higher TF block lower-TF trades on the same symbol
            during the same time window (matches live on_bar first-match-wins).
            Set False for legacy behavior (time-dedup without TF priority).

    Returns:
        List of (symbol, entry_bar_15m, exit_bar_15m, pnl_float).
    """
    # ── Guard: verify required keys come from config, not hardcoded ──────────
    _REQUIRED_CFG_KEYS = {"sp", "lp", "d_sl", "tp", "d_vr"}
    _missing = _REQUIRED_CFG_KEYS - cfg.keys()
    if _missing:
        raise ValueError(
            f"collect_trades() cfg dict missing required keys: {sorted(_missing)}.\n"
            "Build from VolStraddleConfig():\n"
            "  from quantbox.strategy.vol_straddle.config import VolStraddleConfig\n"
            "  cfg_obj = VolStraddleConfig()\n"
            "  cfg = {'sp': cfg_obj.vr_short, 'lp': cfg_obj.vr_long,\n"
            "         'd_vr': cfg_obj.d_vr_threshold, 'd_sl': cfg_obj.d_sl,\n"
            "         'tp': cfg_obj.d_tp}"
        )
    if not (0 < cfg["d_sl"] < 0.20):
        raise ValueError(f"cfg['d_sl']={cfg['d_sl']} out of valid range (0, 0.20). Check config source.")
    if not (0 < cfg["tp"] <= 1.0):
        raise ValueError(f"cfg['tp']={cfg['tp']} out of valid range (0, 1.0]. Check config source.")
    if strat != "D_mom8":
        raise ValueError(f"strat must be 'D_mom8', got {strat!r}")
    # ─────────────────────────────────────────────────────────────────────────

    # G4 parity fix: read trade sim config fresh (not module-level cache)
    _tsc = get_trade_sim_config()
    _cost_dir = _tsc["cost_directional"]
    _G_3TF = [_tsc["tf_15m"], _tsc["tf_1h"], _tsc["tf_4h"]]
    _G_4TF = [_tsc["tf_15m"], _tsc["tf_30m"], _tsc["tf_1h"], _tsc["tf_4h"]]

    sp = cfg["sp"]
    lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"

    d_sl = cfg["d_sl"]
    tp = cfg["tp"]
    d_vr = cfg["d_vr"]

    # D_mom8 directional trailing params — parity fix: live uses trailing stop
    # for D_mom8 but backtest was using sim_directional_trades (no trail).
    if "d_trail_activate" in cfg:
        d_trail_activate = cfg["d_trail_activate"]
        d_trail_distance = cfg["d_trail_distance"]
    else:
        try:
            from quantbox.strategy.vol_straddle.config import VolStraddleConfig as _VSCfg

            _vs = _VSCfg()
            d_trail_activate = _vs.d_trail_activate
            d_trail_distance = _vs.d_trail_distance
        except Exception:
            d_trail_activate = 0.015  # config default
            d_trail_distance = 0.005  # config default (rule #8 floor)

    # Resolve TF group — default G_3TF (fresh from config, G4 fix)
    active_tfs = tf_group if tf_group is not None else _G_3TF

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

            for tf_label, ratio, warmup, mh_d in sorted_tfs:
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

                sig = gen_vr_signal(vr, d_vr, warmup)
                dire = make_direction_momentum(sig, d["mom_8"])
                p, e, x = sim_directional_trailing(
                    c,
                    h,
                    lo,
                    sig,
                    dire,
                    d_sl,
                    tp,
                    mh_d,
                    _cost_dir,
                    d_trail_activate,
                    d_trail_distance,
                )

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

            for tf_label, ratio, warmup, mh_d in active_tfs:
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

                sig = gen_vr_signal(vr, d_vr, warmup)
                dire = make_direction_momentum(sig, d["mom_8"])
                p, e, x = sim_directional_trailing(
                    c,
                    h,
                    lo,
                    sig,
                    dire,
                    d_sl,
                    tp,
                    mh_d,
                    _cost_dir,
                    d_trail_activate,
                    d_trail_distance,
                )

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

    # ── G3 parity fix: deduct funding costs ────────────────────────────────
    # Binance charges funding every 8h on open positions. At 30x leverage
    # this is a material drag (0.9-5.4% of notional over 48h holds).
    # Deduction is post-hoc on collected trades (no Numba signature change).
    # funding_rate_8h: cost per 8h as fraction of notional (e.g. 0.0001 = 0.01%)
    if "funding_rate_8h" in cfg:
        _funding_8h = cfg["funding_rate_8h"]
    else:
        try:
            from quantbox.strategy.vol_straddle.config import VolStraddleConfig as _VSCfg2

            _funding_8h = _VSCfg2().default_funding_rate_8h
        except Exception:
            _funding_8h = 0.0001  # conservative default

    if _funding_8h > 0:
        _BARS_PER_8H_15M = 32  # 8h × 4 bars/h = 32 bars at 15m
        adjusted: list[tuple[str, int, int, float]] = []
        for sym, entry_b, exit_b, pnl_val in all_trades:
            bars_held = exit_b - entry_b
            funding_drag = _funding_8h * (bars_held / _BARS_PER_8H_15M)
            adjusted.append((sym, entry_b, exit_b, pnl_val - funding_drag))
        all_trades = adjusted

    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Portfolio Simulation — Slot Allocation
# ══════════════════════════════════════════════════════════════════════════════

#: 15m 봉 기준 하루치 봉 수 (일손실 차단기의 하루 경계).
BARS_PER_DAY_15M = 96


def portfolio_sim(
    all_trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
    straddle_syms: frozenset[str] | None = None,
    *,
    directions: Sequence[int] | None = None,
    btc_close: np.ndarray | None = None,
    cfg: "VolStraddleConfig | None" = None,
    dir_diversity: bool = False,
    btc_gate: bool = False,
    daily_loss: bool = False,
) -> np.ndarray:
    """Filter trades by max concurrent positions, no same-symbol overlap.

    Sorts trades chronologically, then greedily accepts trades that:
      1. Don't overlap with an existing position on the same symbol.
      2. Don't exceed ``max_concurrent`` total open slots.

    Args:
        all_trades: List of (symbol, entry_bar, exit_bar, pnl).
        max_concurrent: Max simultaneous slots (e.g. 10 for live config).
        straddle_syms: Kept for backward compat. Ignored.

    Returns:
        1-D array of accepted trade PnLs (in chronological order).
    """
    pnl, _ = portfolio_sim_with_details(
        all_trades,
        max_concurrent,
        straddle_syms,
        directions=directions,
        btc_close=btc_close,
        cfg=cfg,
        dir_diversity=dir_diversity,
        btc_gate=btc_gate,
        daily_loss=daily_loss,
    )
    return pnl


def portfolio_sim_with_details(
    all_trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
    straddle_syms: frozenset[str] | None = None,
    *,
    directions: Sequence[int] | None = None,
    btc_close: np.ndarray | None = None,
    cfg: "VolStraddleConfig | None" = None,
    dir_diversity: bool = False,
    btc_gate: bool = False,
    daily_loss: bool = False,
) -> tuple[np.ndarray, list[tuple[str, int, int, float]]]:
    """Portfolio sim that also returns accepted trades with metadata.

    Same logic as ``portfolio_sim`` but additionally returns the list
    of accepted trade tuples for further analysis (symbol breakdown,
    timing analysis, correlation studies, etc.).

    Args:
        all_trades: List of (symbol, entry_bar, exit_bar, pnl).
        max_concurrent: Max simultaneous slots.
        straddle_syms: Kept for backward compat. Ignored.

    Returns:
        (pnl_array, accepted_trades_list).
    """
    if dir_diversity and directions is None:
        raise ValueError("dir_diversity=True requires `directions` (one +1/-1 per trade)")
    if btc_gate and (directions is None or btc_close is None):
        raise ValueError("btc_gate=True requires `directions` and `btc_close`")

    if cfg is not None:
        _cfg = cfg
    else:
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        _cfg = VolStraddleConfig()
    _dirs = list(directions) if directions is not None else [0] * len(all_trades)
    if len(_dirs) != len(all_trades):
        raise ValueError(f"directions length {len(_dirs)} != trades {len(all_trades)}")

    # 정렬은 방향과 함께 — 인덱스가 어긋나면 필터가 조용히 엉뚱한 방향을 본다.
    order = sorted(range(len(all_trades)), key=lambda i: all_trades[i][1])
    accepted: list[tuple[str, int, int, float]] = []
    active: list[tuple[str, int, int]] = []  # (symbol, exit_bar, direction)

    # 실현 이벤트 (청산봉, 자기자본 대비 손익) — 차단기는 **이미 청산된** 손익만 본다.
    realized: list[tuple[int, float]] = []
    ma_period = _cfg.btc_long_ma_period

    for i in order:
        sym, entry, exit_b, pnl_val = all_trades[i]
        d = _dirs[i]

        # 만료 정리 — 이 시점 이전에 끝난 트레이드는 슬롯을 비운다.
        active = [a for a in active if a[1] > entry]

        # 일손실 차단기(G7): 이 진입 시점까지 **청산 완료된** 당일 손익만 센다.
        # (진입 순서로 순회하므로, 아직 안 끝난 트레이드의 손실을 미리 반영하면
        #  라이브가 허용했을 진입을 부당하게 막게 된다.)
        if daily_loss:
            day = entry // BARS_PER_DAY_15M
            day_loss = sum(v for eb, v in realized if eb <= entry and eb // BARS_PER_DAY_15M == day)
            if day_loss < -_cfg.daily_loss_limit_pct:
                continue

        if any(s == sym for s, _, _ in active):
            continue
        if len(active) + 1 > max_concurrent:
            continue

        # 방향 분산: 같은 방향 동시 보유 상한.
        if dir_diversity:
            same_dir = sum(1 for _, _, ad in active if ad == d)
            if same_dir >= _cfg.max_same_direction:
                continue

        # G6 BTC 트렌드 게이트: BTC가 MA 아래면 LONG 차단.
        if btc_gate and d == 1 and btc_close is not None:
            if entry >= ma_period - 1 and entry < len(btc_close):
                window = btc_close[entry - ma_period + 1 : entry + 1]
                if float(btc_close[entry]) < float(np.mean(window)):
                    continue

        active.append((sym, exit_b, d))
        accepted.append((sym, entry, exit_b, pnl_val))

        if daily_loss:
            # 자기자본 대비 영향 = pnl / 슬롯수 (등분 사이징). 실현 시점은 청산 봉.
            realized.append((exit_b, pnl_val / max(max_concurrent, 1)))

    pnl = np.array([t[3] for t in accepted]) if accepted else np.empty(0)
    return pnl, accepted


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


def compute_metrics(pnl: np.ndarray, data_years: float, portfolio_size: int = 1) -> dict:
    """Compute comprehensive performance metrics from PnL array.

    Uses compound model with optional portfolio-size scaling:
    ``equity *= (1 + pnl_per_trade / portfolio_size)``.

    When ``portfolio_size=1`` (default, legacy behavior), each trade compounds
    at full equity (1x reinvestment). This matches a single-slot strategy where
    every trade uses all capital — historically correct for per-strategy research
    at moderate N but mathematically explodes when N × avg_per_trade is large
    (e.g. Donchian N=1876 × 2%/trade → ann_ret ≈ 1.7B%, Calmar ≈ 61M — real
    observed r116-prep baseline artifact).

    When ``portfolio_size>1``, each trade uses 1/portfolio_size of equity —
    the realistic model for a multi-slot portfolio (e.g. ``max_concurrent=10``).
    Matches live sizing semantics where a single trade cannot consume more than
    1/N of account equity.

    Args:
        pnl: Array of per-trade PnL values (fractional, e.g. 0.015 = 1.5%).
        data_years: Duration of data in years (for annualization).
        portfolio_size: Capital-allocation divisor. Default 1 preserves legacy
            1x-compound behavior (backward-compat for r41_r65 research). Pass
            ``VolStraddleConfig().max_concurrent`` (=10) for portfolio-aware
            metrics. Must be ≥ 1.

    Returns:
        Dict with keys: N, PF, EV_bp, t, WR, RR, avg_win_pct,
        avg_loss_pct, ret_pct, max_dd_pct, tpd, ann_ret_pct, calmar.

    Raises:
        ValueError: If ``portfolio_size < 1``.
    """
    if portfolio_size < 1:
        raise ValueError(f"portfolio_size must be >= 1, got {portfolio_size}")

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

    # Equity curve — scaled by portfolio_size to reflect per-slot capital allocation
    scaled_pnl = pnl / portfolio_size
    equity_curve = np.cumprod(1.0 + scaled_pnl)
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


# ══════════════════════════════════════════════════════════════════════════════
#  MA Cluster Trailing Stop — Canonical Sim (promoted from r75)
# ══════════════════════════════════════════════════════════════════════════════


def sim_ma_cluster_trailing(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sma20: np.ndarray,
    sma50: np.ndarray,
    sma100: np.ndarray,
    sma200: np.ndarray,
    momentum: np.ndarray,
    cluster_pct: float,
    min_duration: int,
    max_duration: int,
    min_momentum: float,
    sl: float,
    tp: float,
    trail_activate: float,
    trail_distance: float,
    max_hold: int,
    cost: float,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MA cluster squeeze with trailing stop exit (numba-accelerated).

    Signal detection (inlined):
      - All 4 SMAs must be within cluster_pct of each other (spread / close).
      - Cluster must persist for [min_duration, max_duration] consecutive bars.
      - Price must break out of the cluster zone on the entry bar.
      - |momentum| >= min_momentum required for directional confirmation.
      - direction = sign of momentum.

    Exit logic (priority order — matches exchange execution reality):
      1. Peak gain updated from intra-bar high/low.
      2. Trailing activates once peak_gain >= trail_activate.
      3. When trailing active: trail level is tighter than SL →
         trail checked first (exchange fires tighter stop first).
         SL is skipped since trail level is always above SL.
      4. When trailing NOT active: SL checked normally.
      5. Hard TP cap: exit at tp (safety ceiling above the trail).
      6. Max hold: exit at close[max_hold_bar].

    G5 parity fix (refactor/mega-v2): once trailing activates, trail level
    is always tighter than the original SL. On the exchange, both are
    independent server-side orders — the tighter one fires first. Previous
    code checked SL first unconditionally, causing backtest to exit at -sl
    when the exchange would exit at the trail level (much better PnL).

    Args:
        close: 4h close price array.
        high: 4h high price array.
        low: 4h low price array.
        sma20: Pre-computed SMA(20) on 4h bars.
        sma50: Pre-computed SMA(50) on 4h bars.
        sma100: Pre-computed SMA(100) on 4h bars.
        sma200: Pre-computed SMA(200) on 4h bars.
        momentum: Pre-computed momentum(8) on 4h bars.
        cluster_pct: MA spread threshold (e.g. 0.02 = 2%).
        min_duration: Minimum consecutive cluster bars before entry.
        max_duration: Maximum cluster bars — zombie filter (0 = disabled).
        min_momentum: Minimum |momentum| for direction confirmation.
        sl: Stop-loss fraction (e.g. 0.02 = 2%).
        tp: Hard TP cap fraction (e.g. 0.08 = 8%).
        trail_activate: Peak gain to activate trailing (e.g. 0.04 = 4%).
        trail_distance: Trail distance from peak (e.g. 0.005 = 0.5%).
        max_hold: Maximum bars to hold (in 4h bar units).
        cost: Round-trip transaction cost (e.g. 0.0008).
        warmup: Skip first N bars (SMA warmup).

    Returns:
        (pnl, entry_bars, exit_bars) — each 1-D array of length N_trades.
        Bar indices are in 4h bar units.
    """
    # 위임: 공용 엔진 sim_core 가 유일 구현이다 (중복 sim 제거).
    # 기존 백테스트 숫자 불변은 tests/test_golden_trades.py 가 실데이터로 지킨다.
    from quantbox.backtest.sim_core import ma_cluster_conditions, sim_drive

    bar_kind, entry_ok, direction = ma_cluster_conditions(
        close, sma20, sma50, sma100, sma200, momentum, cluster_pct, min_momentum
    )
    pnl, ent, ext, _dirs = sim_drive(
        close,
        high,
        low,
        bar_kind,
        entry_ok,
        direction,
        min_duration,
        max_duration,
        sl,
        tp,
        trail_activate,
        trail_distance,
        max_hold,
        cost,
        warmup,
        4,
        0,  # cap_divisor=4 (n//4+1), cooldown=off
    )
    return pnl, ent, ext


def collect_ma_cluster_trades(
    data_15m: dict,
    data_4h: dict | None = None,
    *,
    cluster_pct: float | None = None,
    min_duration: int | None = None,
    max_duration: int | None = None,
    min_momentum: float | None = None,
    sl: float | None = None,
    tp: float | None = None,
    trail_activate: float | None = None,
    trail_distance: float | None = None,
    max_hold: int | None = None,
    cost: float | None = None,
) -> list[tuple[str, int, int, float]]:
    """Collect MA cluster trailing trades for all whitelist symbols (4h only).

    Precomputes SMA(20/50/100/200) and momentum(8) on 4h bars, then runs
    ``sim_ma_cluster_trailing`` and converts bar indices to the 15m base
    timeline used by ``portfolio_sim``.

    ⚠️ **재구현 경고 — 라이브와 2.32x 차이 (2026-07-18 실측).** 이 함수의 진입
    스캔은 라이브 ``signals.check_ma_cluster`` 를 재구현한 것이고, 클러스터 에피소드당
    1회여야 할 진입이 재진입으로 **2.32배** 발생한다(3,836 vs 라이브 1,652, 유령 56.9%).
    즉 **라이브 실거래는 이 함수를 쓰지 않으며(strategy.py 가 check_ma_cluster 를 호출),
    이 백테스트만 과대**하다. 라이브 기준 성과가 필요하면 ``sim_core.run_ma_cluster_live``
    를 써라(진입 PF 1.9→2.0, 연 40.5%→17.8%). 이 함수는 기존 백테스트 숫자 재현용으로만
    유지된다. 상세: docs/refactor/research_audit.md, tests/test_sim_core_live_parity.py.

    Parameters default to ``VolStraddleConfig`` values (single source of truth).
    Pass explicit values only for parameter sweeps.

    Args:
        data_15m: {symbol: {close, high, low, volume, _offset, ...}} — 15m bars.
        data_4h: Pre-resampled 4h data dict.  If None, resampled from data_15m.
        cluster_pct: MA spread threshold (default: cfg.ma_cluster_pct).
        min_duration: Minimum cluster bars (default: cfg.ma_cluster_min_duration).
        max_duration: Maximum cluster bars / zombie filter
            (default: cfg.ma_cluster_max_duration).
        min_momentum: Minimum |momentum| for entry (default: cfg.ma_cluster_min_momentum).
        sl: Stop-loss fraction (default: cfg.ma_cluster_sl).
        tp: Hard TP cap fraction (default: cfg.ma_cluster_tp).
        trail_activate: Gain to activate trailing (default: cfg.ma_cluster_trail_activate).
        trail_distance: Trail distance from peak (default: cfg.ma_cluster_trail_distance).
        max_hold: Maximum bars to hold in 4h units (default: cfg.ma_cluster_max_hold_4h).
        cost: Round-trip transaction cost.  G4: ``None`` reads ``cfg.round_trip_cost``
            at call time — module-level constants are frozen at import and would
            silently ignore a config change made after import.

    Returns:
        List of (symbol, entry_15m_bar, exit_15m_bar, pnl) sorted by symbol.
        Compatible with ``portfolio_sim()``.
    """
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig
    from quantbox.strategy.vol_straddle.signals import compute_momentum

    cfg = VolStraddleConfig()

    # Resolve params from config defaults
    _cost = cost if cost is not None else cfg.round_trip_cost
    _warmup_4h = get_trade_sim_config()["tf_4h"][2]
    _cluster_pct = cluster_pct if cluster_pct is not None else cfg.ma_cluster_pct
    _min_duration = min_duration if min_duration is not None else cfg.ma_cluster_min_duration
    _max_duration = max_duration if max_duration is not None else cfg.ma_cluster_max_duration
    _min_momentum = min_momentum if min_momentum is not None else cfg.ma_cluster_min_momentum
    _sl = sl if sl is not None else cfg.ma_cluster_sl
    _tp = tp if tp is not None else cfg.ma_cluster_tp
    _trail_activate = trail_activate if trail_activate is not None else cfg.ma_cluster_trail_activate
    _trail_distance = trail_distance if trail_distance is not None else cfg.ma_cluster_trail_distance
    _max_hold = max_hold if max_hold is not None else cfg.ma_cluster_max_hold_4h

    ratio = 16  # 4h = 16 × 15m bars

    # Resample to 4h if not provided
    if data_4h is None:
        data_4h = build_resampled(data_15m, ratio)

    # Precompute SMAs on 4h bars in-place (idempotent: skip if already present)
    for sym, d4 in data_4h.items():
        c = d4["close"]
        n = len(c)
        for period, key in ((20, "sma_20"), (50, "sma_50"), (100, "sma_100"), (200, "sma_200")):
            if key not in d4:
                out = np.full(n, np.nan)
                if n >= period:
                    s = float(np.sum(c[:period]))
                    out[period - 1] = s / period
                    for idx in range(period, n):
                        s += c[idx] - c[idx - period]
                        out[idx] = s / period
                d4[key] = out
        if "mom_8" not in d4:
            d4["mom_8"] = compute_momentum(c, 8)

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        d4 = data_4h.get(sym)
        if d4 is None:
            continue
        if "sma_200" not in d4 or "mom_8" not in d4:
            continue

        offset = int(data_15m[sym].get("_offset", 0))

        p, e, x = sim_ma_cluster_trailing(
            d4["close"],
            d4["high"],
            d4["low"],
            d4["sma_20"],
            d4["sma_50"],
            d4["sma_100"],
            d4["sma_200"],
            d4["mom_8"],
            _cluster_pct,
            _min_duration,
            _max_duration,
            _min_momentum,
            _sl,
            _tp,
            _trail_activate,
            _trail_distance,
            _max_hold,
            _cost,
            _warmup_4h,
        )

        for k in range(len(p)):
            entry_bar = int(e[k]) * ratio + offset
            exit_bar = int(x[k]) * ratio + ratio - 1 + offset
            all_trades.append((sym, entry_bar, exit_bar, float(p[k])))

    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Donchian Channel Breakout + Volume Spike — Canonical Sim
# ══════════════════════════════════════════════════════════════════════════════


def sim_donchian_trailing(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    lookback: int,
    vol_sma_period: int,
    vol_mult: float,
    breakout_pct: float,
    cooldown_bars: int,
    sl: float,
    tp: float,
    trail_activate: float,
    trail_distance: float,
    max_hold: int,
    cost: float,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Donchian channel breakout + volume spike with trailing stop exit (numba-accelerated).

    Signal detection (inlined from check_donchian_breakout):
      - Donchian channel: chan_high = max(high[i-lookback:i]), chan_low = min(low[i-lookback:i])
        using completed bars only (no lookahead).
      - Volume spike: volume[i] > vol_mult * mean(volume[i-vol_sma_period:i]).
      - LONG: close[i] > chan_high AND (close[i] - chan_high) / chan_high > breakout_pct AND spike.
      - SHORT: close[i] < chan_low AND (chan_low - close[i]) / chan_low > breakout_pct AND spike.
      - Cooldown: skip cooldown_bars after a signal fires.

    Exit logic (matches sim_ma_cluster_trailing - G5 parity fix applies):
      1. Peak gain updated from intra-bar high/low.
      2. Trailing activates once peak_gain >= trail_activate.
      3. When trailing active: trail checked first (tighter stop).
         SL skipped since trail level is always above SL once activated.
      4. When trailing NOT active: SL checked normally.
      5. Hard TP cap: exit at tp.
      6. Max hold: exit at close[max_hold_bar].

    Args:
        close: 4h close price array.
        high: 4h high price array.
        low: 4h low price array.
        volume: 4h volume array.
        lookback: Donchian channel period in bars (e.g. 50).
        vol_sma_period: Volume SMA period for spike detection (e.g. 20).
        vol_mult: Volume spike multiplier threshold (e.g. 2.5).
        breakout_pct: Minimum breakout fraction beyond channel (e.g. 0.02 = 2%).
        cooldown_bars: Bars to skip after a signal fires (e.g. 3).
        sl: Stop-loss fraction (e.g. 0.015 = 1.5%).
        tp: Hard TP cap fraction (e.g. 0.20 = 20%).
        trail_activate: Peak gain to activate trailing (e.g. 0.02 = 2%).
        trail_distance: Trail distance from peak (e.g. 0.008 = 0.8%).
        max_hold: Maximum bars to hold (in 4h bar units).
        cost: Round-trip transaction cost (e.g. 0.0008).
        warmup: Skip first N bars.

    Returns:
        (pnl, entry_bars, exit_bars) - each 1-D array of length N_trades.
        Bar indices are in 4h bar units.
    """
    # 위임: 공용 엔진 sim_core 가 유일 구현이다 (중복 sim 제거).
    # 기존 백테스트 숫자 불변은 tests/test_golden_trades.py 가 실데이터로 지킨다.
    from quantbox.backtest.sim_core import donchian_conditions, sim_drive

    bar_kind, entry_ok, direction = donchian_conditions(
        close, high, low, volume, lookback, vol_sma_period, vol_mult, breakout_pct
    )
    pnl, ent, ext, _dirs = sim_drive(
        close,
        high,
        low,
        bar_kind,
        entry_ok,
        direction,
        1,
        0,  # min_duration=1, max_duration=off (게이팅은 entry_ok가 전담)
        sl,
        tp,
        trail_activate,
        trail_distance,
        max_hold,
        cost,
        warmup,
        2,
        cooldown_bars,  # cap_divisor=2 (n//2+1), donchian은 감소 쿨다운 사용
    )
    return pnl, ent, ext


def collect_donchian_trades(
    data_15m: dict,
    data_4h: dict | None = None,
    *,
    lookback: int | None = None,
    vol_sma_period: int | None = None,
    vol_mult: float | None = None,
    breakout_pct: float | None = None,
    cooldown_bars: int | None = None,
    sl: float | None = None,
    tp: float | None = None,
    trail_activate: float | None = None,
    trail_distance: float | None = None,
    max_hold: int | None = None,
    cost: float | None = None,
) -> list[tuple[str, int, int, float]]:
    """Collect Donchian channel breakout trades for all whitelist symbols (4h only).

    Runs ``sim_donchian_trailing`` per symbol and converts 4h bar indices to
    the 15m base timeline used by ``portfolio_sim``.

    Parameters default to ``VolStraddleConfig`` values (single source of truth).
    Pass explicit values only for parameter sweeps.

    Args:
        data_15m: {symbol: {close, high, low, volume, _offset, ...}} - 15m bars.
        data_4h: Pre-resampled 4h data dict.  If None, resampled from data_15m.
        lookback: Donchian channel period in 4h bars (default: cfg.donchian_lookback).
        vol_sma_period: Volume SMA period (default: cfg.donchian_vol_sma).
        vol_mult: Volume spike multiplier (default: cfg.donchian_vol_mult).
        breakout_pct: Min breakout fraction beyond channel (default: cfg.donchian_breakout_pct).
        cooldown_bars: Bars to skip after signal fires (default: cfg.donchian_cooldown_bars).
        sl: Stop-loss fraction (default: cfg.donchian_sl).
        tp: Hard TP cap fraction (default: cfg.donchian_tp).
        trail_activate: Gain to activate trailing (default: cfg.donchian_trail_activate).
        trail_distance: Trail distance from peak (default: cfg.donchian_trail_distance).
        max_hold: Maximum bars to hold in 4h units (default: cfg.donchian_max_hold_4h).
        cost: Round-trip transaction cost.  G4: ``None`` reads ``cfg.round_trip_cost``
            at call time (see ``collect_ma_cluster_trades``).

    Returns:
        List of (symbol, entry_15m_bar, exit_15m_bar, pnl) sorted by symbol.
        Compatible with ``portfolio_sim()``.
    """
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig

    cfg = VolStraddleConfig()

    # Resolve params from config defaults
    _cost = cost if cost is not None else cfg.round_trip_cost
    _warmup_4h = get_trade_sim_config()["tf_4h"][2]
    _lookback = lookback if lookback is not None else cfg.donchian_lookback
    _vol_sma_period = vol_sma_period if vol_sma_period is not None else cfg.donchian_vol_sma
    _vol_mult = vol_mult if vol_mult is not None else cfg.donchian_vol_mult
    _breakout_pct = breakout_pct if breakout_pct is not None else cfg.donchian_breakout_pct
    _cooldown_bars = cooldown_bars if cooldown_bars is not None else cfg.donchian_cooldown_bars
    _sl = sl if sl is not None else cfg.donchian_sl
    _tp = tp if tp is not None else cfg.donchian_tp
    _trail_activate = trail_activate if trail_activate is not None else cfg.donchian_trail_activate
    _trail_distance = trail_distance if trail_distance is not None else cfg.donchian_trail_distance
    _max_hold = max_hold if max_hold is not None else cfg.donchian_max_hold_4h

    ratio = 16  # 4h = 16 x 15m bars

    # Resample to 4h if not provided
    if data_4h is None:
        data_4h = build_resampled(data_15m, ratio)

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        d4 = data_4h.get(sym)
        if d4 is None:
            continue
        if "volume" not in d4:
            continue

        offset = int(data_15m[sym].get("_offset", 0))

        p, e, x = sim_donchian_trailing(
            d4["close"],
            d4["high"],
            d4["low"],
            d4["volume"],
            _lookback,
            _vol_sma_period,
            _vol_mult,
            _breakout_pct,
            _cooldown_bars,
            _sl,
            _tp,
            _trail_activate,
            _trail_distance,
            _max_hold,
            _cost,
            _warmup_4h,
        )

        for k in range(len(p)):
            entry_bar = int(e[k]) * ratio + offset
            exit_bar = int(x[k]) * ratio + ratio - 1 + offset
            all_trades.append((sym, entry_bar, exit_bar, float(p[k])))

    return all_trades
