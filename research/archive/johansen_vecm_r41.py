"""R41 — Johansen Multi-Cointegration Spread Research

Goal: Find 3~5 coin baskets whose VECM spread has:
  1. Faster mean-reversion (shorter half-life) than any 2-coin pair
  2. Higher spread volatility (bigger % moves per reversion cycle)
  3. Net profit > transaction costs (the fatal flaw of old pairs strategy)

The key insight: with N>2 assets, Johansen gives multiple cointegration vectors.
By choosing the vector with optimal half-life/volatility ratio and adjusting
portfolio weights via VECM loading coefficients, we can engineer spreads that
are both fast-reverting AND high-amplitude.

Cost reality check (from pairs repo autopsy):
  - 2-coin pair: round-trip = 4 × (0.05% fee + 0.10% slip) = 0.60%
  - 3-coin basket: round-trip = 6 × (0.05% + 0.05%) = 0.60% (maker-only via LimitChaser)
  - 4-coin basket: round-trip = 8 × 0.10% = 0.80%
  - 5-coin basket: round-trip = 10 × 0.10% = 1.00%
  - → Need spread moves >> 1% per cycle to be profitable.

Usage:
    uv run python research/active/johansen_vecm_r41.py

Author: QuantBox Research
Date: 2026-03-03
Baseline: v13.0.1
"""
from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numba as nb
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "data" / "cache_extended"
LOG_DIR = ROOT / "logs" / "research"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class R41Config:
    """Research configuration for multi-cointegration sweep."""

    # Data
    interval: str = "15m"
    data_days: int = 2190            # 6 years
    min_overlap_bars: int = 50_000   # ~1.4 years of 15m bars

    # Basket construction
    basket_sizes: list[int] = field(default_factory=lambda: [3, 4, 5])
    max_baskets_per_size: int = 400  # wider search
    prefilter_corr_min: float = 0.50 # skip uncorrelated combos (tighter)

    # Johansen test — walk-forward
    det_order: int = 0               # 0=constant in CI, -1=no deterministic
    k_ar_diff: int = 2               # VAR lag order for Johansen
    formation_bars: int = 2000       # formation window (~21 days @15m)
    oos_bars: int = 2000             # OOS trading window (~21 days @15m)
    max_wf_steps: int = 30           # cap walk-forward steps per basket

    # Spread quality thresholds
    min_half_life: float = 2.0       # bars (too fast = noise)
    max_half_life: float = 500.0     # bars (~5 days at 15m)
    min_spread_vol: float = 0.003    # 0.3% min std of normalized spread
    min_hurst_threshold: float = 0.5 # H < 0.5 = mean-reverting

    # Trading simulation
    entry_z: float = 1.5             # lower → more trades
    exit_z: float = 0.3
    stop_z: float = 3.0              # tighter stop to limit losses
    max_hold_bars: int = 192         # 48 hours at 15m

    # Cost model (maker-only via LimitChaser)
    fee_per_side: float = 0.0005     # 0.05% maker
    slippage_per_side: float = 0.0005  # 0.05% with LimitChaser
    funding_rate_8h: float = 0.0001  # 0.01% per 8h per leg

    # Timeframe experiments — 4h only (1h proven unprofitable after costs)
    resample_factors: list[int] = field(default_factory=lambda: [16])
    # 16=4h  (1h/15m add later only if 4h establishes clear edge)


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════

def load_symbol(symbol: str, cfg: R41Config) -> np.ndarray | None:
    """Load close prices from npz cache. Returns None if not found."""
    pattern = f"{symbol}_{cfg.interval}_{cfg.data_days}d_*.npz"
    files = sorted(CACHE.glob(pattern))
    if not files:
        return None
    data = np.load(files[-1])  # latest
    return data["close"].astype(np.float64)


def load_all_symbols(cfg: R41Config) -> dict[str, np.ndarray]:
    """Load all available symbols with sufficient data."""
    pattern = f"*_{cfg.interval}_{cfg.data_days}d_*.npz"
    symbols = sorted(set(
        f.name.split(f"_{cfg.interval}")[0]
        for f in CACHE.glob(pattern)
    ))
    result = {}
    for sym in symbols:
        close = load_symbol(sym, cfg)
        if close is not None and len(close) >= cfg.min_overlap_bars:
            result[sym] = close
    return result


def align_series(*arrays: np.ndarray) -> list[np.ndarray]:
    """Truncate all arrays to the length of the shortest."""
    min_len = min(len(a) for a in arrays)
    return [a[-min_len:] for a in arrays]


def resample_close(close: np.ndarray, factor: int) -> np.ndarray:
    """Downsample close prices by taking every `factor`-th bar."""
    if factor == 1:
        return close
    # Take the last value in each block (= close of the resampled bar)
    n = len(close) // factor * factor
    return close[:n].reshape(-1, factor)[:, -1]


# ════════════════════════════════════════════════════════════════════════════════
# NUMBA-ACCELERATED SPREAD ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

@nb.jit(nopython=True, cache=True)
def calc_half_life(spread: np.ndarray) -> float:
    """AR(1) half-life of the spread: spread_t = α + λ·spread_{t-1} + ε."""
    n = len(spread)
    if n < 20:
        return np.inf
    y = spread[1:]
    x = spread[:-1]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = 0.0
    den = 0.0
    for i in range(len(x)):
        dx = x[i] - x_mean
        num += dx * (y[i] - y_mean)
        den += dx * dx
    if den < 1e-15:
        return np.inf
    lam = num / den
    if lam >= 1.0 or lam <= 0.0:
        return np.inf
    return -np.log(2.0) / np.log(lam)


@nb.jit(nopython=True, cache=True)
def calc_hurst(spread: np.ndarray, max_lag: int = 50) -> float:
    """Variance-ratio Hurst exponent. H < 0.5 = mean-reverting."""
    n = len(spread)
    if n < max_lag * 2:
        return 0.5
    lags = np.arange(2, min(max_lag, n // 4))
    if len(lags) < 3:
        return 0.5
    log_lags = np.empty(len(lags))
    log_vars = np.empty(len(lags))
    for i in range(len(lags)):
        lag = lags[i]
        diffs = np.empty(n - lag)
        for j in range(n - lag):
            diffs[j] = spread[j + lag] - spread[j]
        v = np.var(diffs)
        if v <= 0:
            return 0.5
        log_lags[i] = np.log(float(lag))
        log_vars[i] = np.log(v)

    # OLS: log_var = α + 2H·log_lag
    x_mean = np.mean(log_lags)
    y_mean = np.mean(log_vars)
    num = 0.0
    den = 0.0
    for i in range(len(log_lags)):
        dx = log_lags[i] - x_mean
        num += dx * (log_vars[i] - y_mean)
        den += dx * dx
    if den < 1e-15:
        return 0.5
    slope = num / den
    return slope / 2.0


@nb.jit(nopython=True, cache=True)
def calc_rolling_zscore(spread: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score of the spread."""
    n = len(spread)
    z = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = spread[i - window + 1 : i + 1]
        mu = np.mean(seg)
        s = np.std(seg)
        if s > 1e-10:
            z[i] = (spread[i] - mu) / s
    return z


@nb.jit(nopython=True, cache=True)
def simulate_spread_trading(
    z: np.ndarray,
    spread: np.ndarray,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_hold: int,
    cost_per_trade: float,  # total round-trip cost as fraction of notional
) -> tuple[int, int, float, float, float]:
    """Simulate mean-reversion trading on the z-score.

    Returns:
        (n_trades, n_wins, total_pnl_pct, avg_bars_held, max_dd_pct)
    """
    n = len(z)
    in_trade = False
    signal = 0       # +1 = long spread, -1 = short spread
    entry_idx = 0
    entry_spread = 0.0
    spread_std = 1.0

    n_trades = 0
    n_wins = 0
    total_pnl = 0.0
    total_bars = 0
    equity = 1.0
    peak_eq = 1.0
    max_dd = 0.0

    for i in range(1, n):
        if np.isnan(z[i]):
            continue

        if not in_trade:
            # Entry
            if z[i] <= -entry_z:
                signal = 1  # long spread
                in_trade = True
                entry_idx = i
                entry_spread = spread[i]
                # Estimate std from recent window
                win = min(200, i)
                seg = spread[i - win:i + 1]
                spread_std = np.std(seg)
                if spread_std < 1e-10:
                    in_trade = False
            elif z[i] >= entry_z:
                signal = -1  # short spread
                in_trade = True
                entry_idx = i
                entry_spread = spread[i]
                win = min(200, i)
                seg = spread[i - win:i + 1]
                spread_std = np.std(seg)
                if spread_std < 1e-10:
                    in_trade = False
        else:
            bars_held = i - entry_idx
            spread_change = spread[i] - entry_spread

            # Normalize PnL by spread std
            pnl_z = signal * spread_change / spread_std

            exit = False
            # Mean reversion exit
            if signal == 1 and z[i] >= -exit_z:
                exit = True
            elif signal == -1 and z[i] <= exit_z:
                exit = True
            # Stop loss
            elif signal == 1 and z[i] <= -(stop_z):
                exit = True
            elif signal == -1 and z[i] >= stop_z:
                exit = True
            # Max hold
            elif bars_held >= max_hold:
                exit = True

            if exit:
                # PnL = z-score change × spread_std, normalized
                trade_pnl = pnl_z - cost_per_trade
                total_pnl += trade_pnl
                total_bars += bars_held
                n_trades += 1
                if trade_pnl > 0:
                    n_wins += 1

                equity *= (1.0 + trade_pnl * 0.01)  # scale
                if equity > peak_eq:
                    peak_eq = equity
                dd = (peak_eq - equity) / peak_eq
                if dd > max_dd:
                    max_dd = dd

                in_trade = False
                signal = 0

    avg_bars = total_bars / max(n_trades, 1)
    return n_trades, n_wins, total_pnl, avg_bars, max_dd


# ════════════════════════════════════════════════════════════════════════════════
# JOHANSEN / VECM ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class BasketResult:
    """Result of analyzing one cointegration basket."""
    symbols: tuple[str, ...]
    n_assets: int
    resample_factor: int
    timeframe: str

    # Johansen results
    n_coint_vectors: int       # number of CI relationships at 5%
    trace_stats: np.ndarray    # trace test statistics
    trace_crit: np.ndarray     # 5% critical values

    # Best vector properties
    weights: np.ndarray        # portfolio weights (normalized)
    half_life: float
    hurst: float
    spread_std: float          # std of the normalized spread
    spread_mean_abs: float     # mean |spread| 
    adf_tstat: float

    # VECM loading (alpha) — speed of adjustment
    loading_speeds: np.ndarray  # alpha coefficients

    # Trading simulation
    n_trades: int = 0
    n_wins: int = 0
    total_pnl: float = 0.0
    avg_bars_held: float = 0.0
    max_dd: float = 0.0
    cost_per_trade: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    pnl_per_trade: float = 0.0
    sharpe: float = 0.0
    coint_stability: float = 0.0   # fraction of WF steps with cointegration
    n_wf_steps: int = 0            # total walk-forward steps
    n_wf_coint: int = 0            # steps where cointegration found

    @property
    def score(self) -> float:
        """Composite score for ranking baskets.
        
        PnL is now in fractional price terms (e.g. 0.001 = 10 bps).
        Higher = better. Balances:
          - More trades (statistical significance)
          - Higher net PnL per trade (after costs)
          - Lower half-life (faster reversion)
          - Lower Hurst (stronger mean-reversion)
          - Higher cointegration stability (survives across WF windows)
        """
        if self.n_trades < 10 or self.total_pnl <= 0:
            return -999.0
        # Convert pnl_per_trade to bps for scoring so magnitudes are reasonable
        pnl_bps = self.pnl_per_trade * 10000
        return (
            pnl_bps * 2                     # reward net profit (in bps)
            + np.log(self.n_trades) * 3      # reward more trades
            - self.hurst * 5                 # penalize non-MR
            - (self.half_life / 100) * 2     # penalize slow reversion
            + self.coint_stability * 10      # bonus for stable cointegration
        )


def analyze_basket(
    symbols: tuple[str, ...],
    all_data: dict[str, np.ndarray],
    cfg: R41Config,
    resample_factor: int = 1,
) -> BasketResult | None:
    """Walk-forward Johansen test + trading simulation on a basket.
    
    Walk-forward approach:
      1. Formation window (cfg.formation_bars) → Johansen test → get weights
      2. OOS window (cfg.oos_bars) → trade with frozen weights/stats
      3. Slide forward by oos_bars and repeat
    
    This avoids lookahead bias and tests stability of cointegration.
    """

    # Load and align
    arrays = [all_data[s] for s in symbols]
    aligned = align_series(*arrays)

    # Resample if needed
    if resample_factor > 1:
        aligned = [resample_close(a, resample_factor) for a in aligned]

    n_bars = len(aligned[0])
    form_bars = cfg.formation_bars // resample_factor
    oos_bars = cfg.oos_bars // resample_factor
    min_required = form_bars + oos_bars * 3  # need at least 3 OOS steps

    if n_bars < min_required:
        return None

    # Log prices matrix
    log_prices = np.column_stack([np.log(a) for a in aligned])

    # Cost params — weight-based, not leg-count-based
    n_legs = len(symbols)

    # ── Walk-forward loop ──
    all_trades_pnl: list[float] = []
    all_trades_bars: list[int] = []
    best_weights = None
    best_half_life = np.inf
    best_hurst = 0.5
    best_spread_std = 0.0
    best_adf_t = 0.0
    best_alpha = np.zeros(len(symbols))
    best_cost_rt = 0.0
    n_coint_total = 0
    n_steps_tested = 0
    n_steps_coint = 0
    total_trace_stats = None
    total_trace_crit = None

    start = 0
    max_hold_oos = cfg.max_hold_bars // resample_factor
    bars_per_8h = max(1, 32 // resample_factor)
    wf_step = 0

    while start + form_bars + oos_bars <= n_bars and wf_step < cfg.max_wf_steps:
        formation = log_prices[start : start + form_bars]
        oos_start = start + form_bars
        oos_end = min(oos_start + oos_bars, n_bars)
        oos_data = log_prices[oos_start : oos_end]
        n_steps_tested += 1
        wf_step += 1

        # ── Johansen on formation window ──
        try:
            jres = coint_johansen(formation, det_order=cfg.det_order,
                                   k_ar_diff=cfg.k_ar_diff)
        except Exception:
            start += oos_bars
            continue

        trace_stats = jres.lr1
        trace_crit = jres.cvt[:, 1]
        n_coint = int(np.sum(trace_stats > trace_crit))

        if total_trace_stats is None:
            total_trace_stats = trace_stats
            total_trace_crit = trace_crit

        if n_coint == 0:
            start += oos_bars
            continue

        n_steps_coint += 1

        # ── Find best eigenvector on formation ──
        best_vec = None
        best_vec_score = -np.inf

        for vec_idx in range(n_coint):
            weights = jres.evec[:, vec_idx].copy()
            max_w = np.max(np.abs(weights))
            if max_w < 1e-10:
                continue
            weights = weights / max_w

            spread_form = formation @ weights
            hl = calc_half_life(spread_form)
            if hl < cfg.min_half_life or hl > cfg.max_half_life:
                continue

            h = calc_hurst(spread_form)
            if h >= cfg.min_hurst_threshold:
                continue

            s_std = float(np.std(spread_form))
            if s_std < cfg.min_spread_vol:
                continue

            vec_score = -hl * h
            if vec_score > best_vec_score:
                best_vec_score = vec_score
                best_vec = dict(weights=weights, hl=hl, hurst=h, std=s_std)

        if best_vec is None:
            start += oos_bars
            continue

        # ── Trade on OOS with frozen formation stats ──
        weights = best_vec["weights"]
        form_spread = formation @ weights
        form_mean = float(np.mean(form_spread))
        form_std = float(np.std(form_spread))

        if form_std < 1e-10:
            start += oos_bars
            continue

        # OOS spread + z-score (using frozen formation stats)
        oos_spread = oos_data @ weights
        oos_z = (oos_spread - form_mean) / form_std

        # Cost model — proportional to total weight deployed
        total_weight = float(np.sum(np.abs(weights)))
        cost_rt = total_weight * 2 * (cfg.fee_per_side + cfg.slippage_per_side)

        # Funding cost estimate (per trade)
        avg_hold_est = min(best_vec["hl"] * 1.5, max_hold_oos)
        n_funding_periods = avg_hold_est / bars_per_8h
        funding = cfg.funding_rate_8h * total_weight * 2 * n_funding_periods
        total_cost = cost_rt + funding

        # Simple trading loop on OOS
        in_trade = False
        signal = 0
        entry_idx = 0
        entry_z = 0.0

        for i in range(len(oos_z)):
            z_i = oos_z[i]
            if np.isnan(z_i):
                continue

            if not in_trade:
                if z_i <= -cfg.entry_z:
                    signal = 1; in_trade = True; entry_idx = i; entry_z = z_i
                elif z_i >= cfg.entry_z:
                    signal = -1; in_trade = True; entry_idx = i; entry_z = z_i
            else:
                bars_held = i - entry_idx
                exit_trade = False

                if signal == 1 and z_i >= -cfg.exit_z:
                    exit_trade = True
                elif signal == -1 and z_i <= cfg.exit_z:
                    exit_trade = True
                elif signal == 1 and z_i <= -cfg.stop_z:
                    exit_trade = True
                elif signal == -1 and z_i >= cfg.stop_z:
                    exit_trade = True
                elif bars_held >= max_hold_oos:
                    exit_trade = True

                if exit_trade:
                    # PnL in real price terms (fraction of notional)
                    # z_change is in std units → multiply by form_std to get
                    # log-price move of the spread portfolio
                    z_change = signal * (z_i - entry_z)
                    spread_pnl = z_change * form_std   # log-price PnL
                    trade_pnl = spread_pnl - total_cost  # net after costs
                    all_trades_pnl.append(trade_pnl)
                    all_trades_bars.append(bars_held)
                    in_trade = False

        # Track best formation stats (from the step with shortest half-life)
        if best_vec["hl"] < best_half_life:
            best_half_life = best_vec["hl"]
            best_hurst = best_vec["hurst"]
            best_spread_std = best_vec["std"]
            best_weights = best_vec["weights"]
            best_cost_rt = total_cost
            n_coint_total = n_coint
            total_trace_stats = trace_stats
            total_trace_crit = trace_crit

            # ADF on formation spread
            from statsmodels.tsa.stattools import adfuller
            try:
                adf_r = adfuller(form_spread, maxlag=int(np.sqrt(len(form_spread))),
                                  regression='c', autolag=None)
                best_adf_t = adf_r[0]
            except Exception:
                best_adf_t = 0.0

        start += oos_bars

    # ── Aggregate results ──
    if not all_trades_pnl or len(all_trades_pnl) < 5:
        return None

    n_trades = len(all_trades_pnl)
    wins = [p for p in all_trades_pnl if p > 0]
    losses = [p for p in all_trades_pnl if p <= 0]
    n_wins = len(wins)
    total_pnl = sum(all_trades_pnl)
    wr = n_wins / n_trades * 100
    pnl_per = total_pnl / n_trades
    avg_bars = sum(all_trades_bars) / n_trades

    # PF
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_win / max(gross_loss, 1e-10)

    # Max DD
    equity_curve = np.cumsum(all_trades_pnl)
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve)
    max_dd = float(np.max(dd)) / max(float(peak[-1]), 1.0) if len(peak) > 0 else 0

    # Stability: fraction of WF steps that had cointegration
    coint_stability = n_steps_coint / max(n_steps_tested, 1)

    tf_map = {1: "15m", 4: "1h", 16: "4h"}

    if best_weights is None:
        return None

    return BasketResult(
        symbols=symbols,
        n_assets=len(symbols),
        resample_factor=resample_factor,
        timeframe=tf_map.get(resample_factor, f"{resample_factor}x"),
        n_coint_vectors=n_coint_total,
        trace_stats=total_trace_stats,
        trace_crit=total_trace_crit,
        weights=best_weights,
        half_life=best_half_life,
        hurst=best_hurst,
        spread_std=best_spread_std,
        spread_mean_abs=best_spread_std * 0.8,  # approximate
        adf_tstat=best_adf_t,
        loading_speeds=best_alpha,
        n_trades=n_trades,
        n_wins=n_wins,
        total_pnl=total_pnl,
        avg_bars_held=avg_bars,
        max_dd=max_dd,
        cost_per_trade=best_cost_rt,
        profit_factor=pf,
        win_rate=wr,
        pnl_per_trade=pnl_per,
        coint_stability=coint_stability,
        n_wf_steps=n_steps_tested,
        n_wf_coint=n_steps_coint,
    )


# ════════════════════════════════════════════════════════════════════════════════
# PREFILTER — correlation clustering
# ════════════════════════════════════════════════════════════════════════════════

def prefilter_combinations(
    symbols: list[str],
    data: dict[str, np.ndarray],
    basket_size: int,
    cfg: R41Config,
) -> list[tuple[str, ...]]:
    """Pre-filter combinations by pairwise correlation.
    
    Only keep baskets where ALL pairs have |corr| >= threshold.
    This dramatically reduces the search space.
    """
    n = len(symbols)

    # Compute pairwise correlation matrix on log-returns
    min_len = min(len(data[s]) for s in symbols)
    returns = np.column_stack([
        np.diff(np.log(data[s][-min_len:]))
        for s in symbols
    ])
    corr = np.corrcoef(returns.T)

    # For each combination, check if all pairs meet threshold
    valid = []
    for combo in itertools.combinations(range(n), basket_size):
        ok = True
        for i, j in itertools.combinations(combo, 2):
            if abs(corr[i, j]) < cfg.prefilter_corr_min:
                ok = False
                break
        if ok:
            valid.append(tuple(symbols[k] for k in combo))

    return valid


# ════════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ════════════════════════════════════════════════════════════════════════════════

def run_sweep(cfg: R41Config | None = None):
    """Main research sweep: load data → prefilter → Johansen → simulate → rank."""
    if cfg is None:
        cfg = R41Config()

    print("=" * 72, flush=True)
    print("R41 — Johansen Multi-Cointegration Spread Research", flush=True)
    print("=" * 72, flush=True)

    # ── Load data ──
    print("\n[1/5] Loading data…", flush=True)
    t0 = time.time()
    all_data = load_all_symbols(cfg)
    symbols = sorted(all_data.keys())
    print(f"  {len(symbols)} symbols loaded ({time.time()-t0:.1f}s)", flush=True)
    print(f"  Bars range: {min(len(v) for v in all_data.values()):,} — "
          f"{max(len(v) for v in all_data.values()):,}", flush=True)

    all_results: list[BasketResult] = []

    for resample in cfg.resample_factors:
        tf_name = {1: "15m", 4: "1h", 16: "4h"}.get(resample, f"{resample}x")
        print(f"\n{'─'*72}", flush=True)
        print(f"TIMEFRAME: {tf_name} (resample factor={resample})", flush=True)
        print(f"{'─'*72}", flush=True)

        for bsize in cfg.basket_sizes:
            # ── Prefilter ──
            print(f"\n[2/5] Pre-filtering {bsize}-asset baskets…", flush=True)
            t1 = time.time()
            combos = prefilter_combinations(symbols, all_data, bsize, cfg)
            total_possible = 1
            for i in range(bsize):
                total_possible = total_possible * (len(symbols) - i) // (i + 1)
            print(f"  {len(combos):,} / {total_possible:,} combos pass "
                  f"corr>{cfg.prefilter_corr_min:.2f} filter ({time.time()-t1:.1f}s)",
                  flush=True)

            if not combos:
                print("  → No valid combinations. Skipping.", flush=True)
                continue

            # Limit to max_baskets_per_size (random sample if too many)
            if len(combos) > cfg.max_baskets_per_size:
                rng = np.random.default_rng(42)
                indices = rng.choice(len(combos), cfg.max_baskets_per_size, replace=False)
                combos = [combos[i] for i in sorted(indices)]
                print(f"  Sampling {len(combos)} combos for analysis", flush=True)

            # ── Johansen + simulate ──
            print(f"[3/5] Analyzing {len(combos)} baskets (Johansen WF + sim)…",
                  flush=True)
            print(f"  Config: form={cfg.formation_bars}→{cfg.formation_bars//resample} "
                  f"oos={cfg.oos_bars}→{cfg.oos_bars//resample} "
                  f"max_wf_steps={cfg.max_wf_steps}", flush=True)
            t2 = time.time()
            results: list[BasketResult] = []
            n_coint_found = 0
            n_profitable = 0

            for idx, combo in enumerate(combos):
                if (idx + 1) % 10 == 0 or idx == 0:
                    elapsed = time.time() - t2
                    eta = elapsed / (idx + 1) * (len(combos) - idx - 1)
                    print(f"  [{idx+1}/{len(combos)}] "
                          f"coint={n_coint_found} profitable={n_profitable} "
                          f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)",
                          flush=True)

                result = analyze_basket(combo, all_data, cfg, resample)
                if result is not None:
                    n_coint_found += 1
                    results.append(result)
                    if result.total_pnl > 0:
                        n_profitable += 1

            elapsed = time.time() - t2
            print(f"  Done: {n_coint_found} cointegrated / {len(combos)} tested "
                  f"({elapsed:.1f}s)", flush=True)
            print(f"  Profitable: {n_profitable} / {n_coint_found}", flush=True)

            all_results.extend(results)

    # ── Rank & Report ──
    print(f"\n{'='*72}")
    print(f"[4/5] RANKING — {len(all_results)} total baskets analyzed")
    print(f"{'='*72}")

    if not all_results:
        print("\n  ⚠ No cointegrated baskets found. Try relaxing filters.")
        return

    # Sort by score
    all_results.sort(key=lambda r: r.score, reverse=True)

    # Summary statistics
    profitable = [r for r in all_results if r.total_pnl > 0]
    print(f"\n  Profitable baskets: {len(profitable)} / {len(all_results)}")

    if profitable:
        avg_pnl = np.mean([r.pnl_per_trade for r in profitable])
        avg_hl = np.mean([r.half_life for r in profitable])
        avg_hurst = np.mean([r.hurst for r in profitable])
        avg_trades = np.mean([r.n_trades for r in profitable])
        avg_cost = np.mean([r.cost_per_trade for r in profitable])
        print(f"  Avg PnL/trade: {avg_pnl*10000:.1f} bps")
        print(f"  Avg cost/trade: {avg_cost*10000:.1f} bps")
        print(f"  Avg half-life: {avg_hl:.1f} bars")
        print(f"  Avg Hurst: {avg_hurst:.3f}")
        print(f"  Avg trades: {avg_trades:.0f}")

    # Top 20 baskets (PnL/cost in bps for readability)
    print(f"\n[5/5] TOP 20 BASKETS (PnL & Cost in basis points = 0.01%)")
    print(f"{'─'*130}")
    header = (f"{'Rank':>4} {'TF':>3} {'N':>1} {'Symbols':<40} "
              f"{'Trades':>6} {'WR%':>5} {'PF':>5} {'PnL/t':>8} "
              f"{'Cost':>6} {'Net':>8} {'HL':>5} {'Hurst':>5} "
              f"{'Stab%':>5} {'Score':>7}")
    print(header)
    print(f"{'─'*130}")

    for rank, r in enumerate(all_results[:20], 1):
        syms = "+".join(s.replace("USDT", "") for s in r.symbols)
        if len(syms) > 39:
            syms = syms[:36] + "..."
        net_bps = (r.pnl_per_trade) * 10000
        cost_bps = r.cost_per_trade * 10000
        pnl_gross_bps = net_bps + cost_bps  # gross before cost
        print(f"{rank:>4} {r.timeframe:>3} {r.n_assets:>1} {syms:<40} "
              f"{r.n_trades:>6} {r.win_rate:>5.1f} {r.profit_factor:>5.2f} "
              f"{pnl_gross_bps:>+8.1f} {cost_bps:>6.1f} {net_bps:>+8.1f} "
              f"{r.half_life:>5.1f} {r.hurst:>5.3f} "
              f"{r.coint_stability*100:>5.0f} {r.score:>7.2f}")

    # Detailed top 5
    print(f"\n{'='*72}")
    print("DETAILED TOP 5")
    print(f"{'='*72}")

    for rank, r in enumerate(all_results[:5], 1):
        print(f"\n#{rank} — {' + '.join(r.symbols)} ({r.timeframe})")
        print(f"  Weights: {np.array2string(r.weights, precision=4)}")
        print(f"  Total |weight|: {np.sum(np.abs(r.weights)):.3f}")
        print(f"  CI vectors: {r.n_coint_vectors}, "
              f"ADF t-stat: {r.adf_tstat:.2f}")
        print(f"  Half-life: {r.half_life:.1f} bars "
              f"(~{r.half_life * r.resample_factor * 15 / 60:.1f}h)")
        print(f"  Hurst: {r.hurst:.4f}")
        print(f"  Spread σ: {r.spread_std:.6f}")
        print(f"  ──── Walk-Forward ────")
        print(f"  WF steps: {r.n_wf_steps}, Cointegrated: {r.n_wf_coint} "
              f"({r.coint_stability:.0%} stability)")
        print(f"  ──── Trading (PnL in bps = 0.01%) ────")
        print(f"  Trades: {r.n_trades}, Win Rate: {r.win_rate:.1f}%")
        print(f"  PF: {r.profit_factor:.3f}")
        net_bps = r.pnl_per_trade * 10000
        cost_bps = r.cost_per_trade * 10000
        total_bps = r.total_pnl * 10000
        print(f"  PnL/trade (net): {net_bps:+.1f} bps")
        print(f"  Cost/trade:      {cost_bps:.1f} bps "
              f"(|w|={np.sum(np.abs(r.weights)):.2f} × 2 sides)")
        print(f"  Total PnL:       {total_bps:+.1f} bps over {r.n_trades} trades")
        print(f"  Avg hold: {r.avg_bars_held:.1f} bars "
              f"(~{r.avg_bars_held * r.resample_factor * 15 / 60:.1f}h)")
        print(f"  Max DD: {r.max_dd:.1%}")

    # ── Cost analysis ──
    print(f"\n{'='*72}")
    print("COST REALITY CHECK")
    print(f"{'='*72}")
    for bsize in cfg.basket_sizes:
        subset = [r for r in all_results
                  if r.n_assets == bsize and r.n_trades >= 10]
        if not subset:
            continue
        avg_cost = np.mean([r.cost_per_trade for r in subset])
        avg_pnl = np.mean([r.pnl_per_trade for r in subset])
        profit_after = [r for r in subset if r.total_pnl > 0]
        print(f"\n  {bsize}-asset baskets ({len(subset)} tested, "
              f"{len(profit_after)} profitable):")
        print(f"    Avg cost/trade: {avg_cost*10000:.1f} bps")
        print(f"    Avg net PnL/trade: {avg_pnl*10000:+.1f} bps")
        if profit_after:
            best = max(profit_after, key=lambda r: r.pnl_per_trade)
            print(f"    Best: {'+'.join(s.replace('USDT','') for s in best.symbols)} "
                  f"net={best.pnl_per_trade*10000:+.1f}bps "
                  f"({best.n_trades} trades, WR={best.win_rate:.1f}%)")

    # ── Summary verdict ──
    print(f"\n{'='*72}")
    print("VERDICT")
    print(f"{'='*72}")
    top = all_results[0] if all_results else None
    if top and top.total_pnl > 0 and top.n_trades >= 20:
        net_bps = top.pnl_per_trade * 10000
        cost_bps = top.cost_per_trade * 10000
        print(f"\n  ✓ Best basket: {'+'.join(top.symbols)} ({top.timeframe})")
        print(f"    {top.n_trades} trades, WR={top.win_rate:.1f}%, "
              f"PF={top.profit_factor:.2f}")
        print(f"    Net PnL/trade: {net_bps:+.1f} bps")
        print(f"    Cost/trade:    {cost_bps:.1f} bps")
        gross_bps = net_bps + cost_bps
        print(f"    Gross/trade:   {gross_bps:+.1f} bps")
        margin = gross_bps / max(cost_bps, 0.01)
        print(f"    → Gross/Cost ratio = {margin:.2f}x")
        if net_bps > 0 and margin > 2.0:
            print("    → PROMISING — profit clearly exceeds cost")
        elif net_bps > 0 and margin > 1.2:
            print("    → MARGINAL — tight margin, needs more validation")
        elif net_bps > 0:
            print("    → WEAK — profit barely exceeds cost, risky in live")
        else:
            print("    → UNPROFITABLE after costs")
    else:
        print("\n  ✗ No basket with sufficient trades and profitability found.")
        print("    Consider: wider z-score entry, longer timeframe, "
              "different basket sizes")

    return all_results


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_sweep()
