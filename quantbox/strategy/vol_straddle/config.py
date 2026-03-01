"""VolStraddle strategy configuration dataclass.

A1-D50: VR compression detector with multi-TF (15m/1h/4h) signal generation.
OOS validated: N=799 (D_mom8 only), EV=+29.3bp, PF=1.376, t=3.48.

Parameters are locked to R35 A1-D50 backtest values for parity.
Do NOT change without re-running backtest validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VolStraddleConfig:
    """VolStraddle (VR compression) strategy configuration.

    All values match R35 A1-D50 exactly for backtest-live parity.
    """

    # ── VR Computation ──
    vr_short: int = 7               # Short ATR period for VR
    vr_long: int = 50               # Long ATR period for VR

    # ── D_mom8: Directional Momentum ──
    d_vr_threshold: float = 0.50    # A1-D50: D VR entry threshold (was 0.45)
    d_sl: float = 0.01              # 1% stop loss
    d_tp: float = 0.05              # 5% take profit

    # ── S_d8: Straddle (requires Binance Hedge Mode) ──
    s_vr_threshold: float = 0.55    # S VR entry threshold
    s_squeeze_dur: int = 8          # Min consecutive bars below S VR threshold
    s_sl: float = 0.015             # 1.5% stop loss (per leg)
    s_tp: float = 0.05              # 5% take profit (per leg)
    straddle_enabled: bool = True    # S_d8 straddle mode enabled
    hedge_mode: bool = True          # Binance Hedge Mode (dualSidePosition=true)

    # ── Direction ──
    momentum_period: int = 8        # Momentum lookback for D direction

    # ── Max Hold (bars at each TF, all ≈ 48h real-time) ──
    max_hold_15m: int = 192         # 192 × 15m = 48h
    max_hold_1h: int = 48           # 48 × 1h = 48h
    max_hold_4h: int = 12           # 12 × 4h = 48h

    # ── Multi-TF Control ──
    enable_15m: bool = True
    enable_1h: bool = True
    enable_4h: bool = True

    # ── VR warmup (bars before first signal per TF) ──
    warmup_15m: int = 200           # ~50h of 15m bars (~vr_long×4)
    warmup_1h: int = 50             # ~50h of 1h bars
    warmup_4h: int = 13             # ~52h of 4h bars

    # ── Position Sizing & Leverage ──
    base_leverage: float = 1.5      # Flat leverage (same as rubber band baseline)
    max_leverage: float = 20.0      # Hard cap (exchange safety limit)
    risk_per_trade: float = 0.01    # 1% risk per trade

    # ── Risk Management ──
    max_concurrent: int = 2         # Max simultaneous positions
    max_drawdown_pct: float = 0.10  # 10% session drawdown → halt

    # ── Costs (conservative) ──
    fee_rate: float = 0.0002        # 0.02% maker fee per side
    slippage_rate: float = 0.0002   # 0.02% slippage with limit chaser

    # ── Direction Diversity ──
    dir_diversity_enabled: bool = True
    max_same_direction: int = 2

    # ── Dynamic Capital ──
    dynamic_capital_enabled: bool = True

    # ── SoftTight DD Scaling (safety mechanism) ──
    soft_equity_scale: bool = True
    soft_eq_dd1: float = 0.02       # 2% DD → 80% sizing
    soft_eq_scale1: float = 0.80
    soft_eq_dd2: float = 0.04       # 4% DD → 60% sizing
    soft_eq_scale2: float = 0.60
    soft_eq_dd3: float = 0.06       # 6% DD → 40% sizing
    soft_eq_scale3: float = 0.40

    # ── Symbol Selection ──
    symbol_whitelist: list[str] = field(default_factory=list)
    exclude_symbols: list[str] = field(default_factory=lambda: [
        'USDCUSDT', 'TUSDUSDT', 'BUSDUSDT',
    ])
    min_volume_rank: int = 30

    # ── Compatibility fields (used by scalper internals) ──
    # These map vol straddle concepts to the scalper's existing interface
    # so that shared code (risk checks, brackets, logging) works unchanged.
    lookback: int = 8               # Used for move calc in logging — not for signal
    threshold: float = 0.0          # Not used — signal is VR-based
    bar_interval_minutes: int = 15

    @property
    def cost_per_side(self) -> float:
        return self.fee_rate + self.slippage_rate

    @property
    def round_trip_cost(self) -> float:
        return 2 * self.cost_per_side

    def get_max_hold(self, tf: str) -> int:
        """Get max hold bars for a timeframe."""
        return {
            '15m': self.max_hold_15m,
            '1h': self.max_hold_1h,
            '4h': self.max_hold_4h,
        }.get(tf, self.max_hold_15m)

    def get_warmup(self, tf: str) -> int:
        """Get warmup bars for a timeframe."""
        return {
            '15m': self.warmup_15m,
            '1h': self.warmup_1h,
            '4h': self.warmup_4h,
        }.get(tf, self.warmup_15m)
