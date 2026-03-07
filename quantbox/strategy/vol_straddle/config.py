"""VolStraddle strategy configuration dataclass.

R43-Exit: VR compression detector with trailing stop exit (R43 validated).
Full 6yr validated: N=2,336 @10 slots, PF=1.351, Calmar=9.4, DD=30.0%.
6 consecutive profitable years (2020-2025). 0 losing years.

v13.1 changes (R43): Directional TP 5%→12% + trailing stop (activate 2%, distance 0.3%).
Straddle exit unchanged (SL=1.5%, TP=5%).

R44d V17: MA Cluster Squeeze (4h-only) as complementary strategy.
4 SMAs (20/50/100/200) cluster → breakout + momentum direction.
Do NOT change without re-running backtest validation.

Half-Kelly Leverage Derivation (20260304_ef55e2e strategy report):
  Kelly f* = WR - (1-WR)/RR,  Leverage = (f*/2) × N / SL
  VR:  WR=25.9%, PF=1.310, RR=3.75 → f*=6.1%, f*/2=3.07% → L=30.7→30x
  MA:  WR=38.6%, PF=1.285, RR=2.05 → f*=8.65%, f*/2=4.33% → L=21.6→21x
  Per-trade risk: VR=3.0%, MA=4.2% of total balance.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VolStraddleConfig:
    """VolStraddle (VR compression) strategy configuration.

    All values match R39-VR856 exactly for backtest-live parity.
    """

    # ── VR Computation ──
    vr_short: int = 8               # Short ATR period for VR (R39: was 7)
    vr_long: int = 56               # Long ATR period for VR (R39: was 50)

    # ── D_mom8: Directional Momentum ──
    d_vr_threshold: float = 0.50    # A1-D50: D VR entry threshold (was 0.45)
    d_sl: float = 0.01              # 1% stop loss
    d_tp: float = 0.12              # 12% hard TP ceiling (R43: was 5%)

    # ── D_mom8: Trailing Stop (R43 validated) ──
    # Once unrealized gain reaches trail_activate, a trailing stop
    # follows at trail_distance below peak. Hard TP at d_tp (12%).
    # Mechanism: lets VR squeeze explosive releases run beyond fixed TP.
    d_trail_activate: float = 0.02  # Activate trailing after +2% gain
    d_trail_distance: float = 0.003 # Trail 0.3% behind peak

    # ── S_d8: Straddle (requires Binance Hedge Mode) ──
    s_vr_threshold: float = 0.55    # S VR entry threshold
    s_squeeze_dur: int = 4          # Min consecutive bars below S VR threshold (R39: was 8)
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

    # ── Position Sizing & Leverage (Half-Kelly derived) ──
    base_leverage: float = 30.0      # Half-Kelly: f*/2=3.07%, L=0.0307×10/0.01≈30
    max_leverage: float = 30.0      # Hard cap (matches base_leverage)
    risk_per_trade: float = 0.01    # 1% risk per trade

    # ── Risk Management ──
    max_concurrent: int = 10         # 10 slots (VR L/N=30/10=3.0, MA L/N=21/10=2.1)
    max_drawdown_pct: float = 0.10  # 10% session drawdown → halt

    # ── Costs (conservative) ──
    fee_rate: float = 0.0002        # 0.02% maker fee per side
    slippage_rate: float = 0.0002   # 0.02% slippage with limit chaser

    # ── Direction Diversity ──
    dir_diversity_enabled: bool = True
    max_same_direction: int = 2

    # ── Dynamic Capital ──
    # Disabled: vol_straddle uses fixed balance/max_concurrent (backtest parity).
    # Dynamic formula (balance/n_active+1) causes -2019 margin errors on 10-slot config.
    dynamic_capital_enabled: bool = False

    # ── SoftTight DD Scaling (safety mechanism) ──
    soft_equity_scale: bool = True
    soft_eq_dd1: float = 0.02       # 2% DD → 80% sizing
    soft_eq_scale1: float = 0.80
    soft_eq_dd2: float = 0.04       # 4% DD → 60% sizing
    soft_eq_scale2: float = 0.60
    soft_eq_dd3: float = 0.06       # 6% DD → 40% sizing
    soft_eq_scale3: float = 0.40

    # ── CW-Tier Conviction Sizing (R34 validated) ──
    # Score = count of 4 conditions: E2, C2, B3, A3 (0-4)
    # CW-Tier multiplier applied to position size based on score.
    # T3-Triple: optional high-conviction filter (score ≥ min_conviction_score).
    cw_enabled: bool = True          # Enable conviction-weighted sizing
    min_conviction_score: int = 0    # Min score to enter (0=all, 3=T3-Triple)
    cw_multipliers: list[float] = field(default_factory=lambda: [
        0.5,   # score 0 → 0.5x
        1.0,   # score 1 → 1.0x (baseline)
        1.5,   # score 2 → 1.5x
        2.0,   # score 3 → 2.0x
        2.5,   # score 4 → 2.5x (max conviction)
    ])

    # ── E2: SMA Proximity ──
    # |close - SMA(20)| / SMA(20) < threshold → near mean → squeeze valid
    e2_enabled: bool = True
    e2_sma_period: int = 20          # SMA lookback for mean proximity
    e2_threshold: float = 0.005      # 0.5% max distance from SMA

    # ── C2: BTC Neutral ──
    # |BTC momentum(8)| < threshold → BTC not trending → altcoin squeeze valid
    c2_enabled: bool = True
    c2_btc_mom_threshold: float = 0.01  # 1% max BTC momentum magnitude

    # ── B3: Confluence ──
    # Both D_mom8 AND S_d8 would fire simultaneously (overlap)
    b3_enabled: bool = True

    # ── A3: Quality+ ──
    # D_mom8: VR < deep_vr_threshold (extra deep compression)
    # S_d8: squeeze duration ≥ long_squeeze_dur (extended squeeze)
    a3_enabled: bool = True
    a3_deep_vr_threshold: float = 0.35   # D_mom8 deep compression
    a3_long_squeeze_dur: int = 16        # S_d8 extended squeeze duration

    # ── F5: TBR z-score Gate (R46 validated, D_mom8 only) ──
    # Taker Buy Ratio z-score computed over lookback 15m bars.
    # Removes crowded-flow trades: skip LONG if z > thresh (retail FOMO),
    # skip SHORT if z < -thresh (retail panic selling).
    # Backtest result (6yr, 46 symbols): PF 1.306→1.348 (+0.042),
    # WR 42.6%→46.4% (+3.8%), N -14.8%, p=0.029.
    # IMPORTANT: Applied to D_mom8 only. S_d8 straddle is direction-neutral.
    f5_tbr_enabled: bool = True         # R46 validated, activated v14.3.0
    f5_tbr_z_threshold: float = 0.5    # R46b optimal (lb=192, p=0.029)
    f5_tbr_z_window: int = 192         # Lookback bars (~2 days of 15m)

    # ── Symbol Selection ──
    symbol_whitelist: list[str] = field(default_factory=list)
    exclude_symbols: list[str] = field(default_factory=lambda: [
        'USDCUSDT', 'TUSDUSDT', 'BUSDUSDT',
    ])
    min_volume_rank: int = 30

    # ── MA Cluster Squeeze (R44d V17, 4h-only) ──
    # Complementary strategy: 4 SMAs cluster → breakout.
    # Fires when VR squeeze is idle (93.7% non-overlapping).
    ma_cluster_enabled: bool = True          # Enable MA cluster signals
    ma_cluster_pct: float = 0.03             # Max spread among 4 SMAs (3%)
    ma_cluster_min_duration: int = 8         # Min 4h bars in cluster (≈32h)
    ma_cluster_max_duration: int = 24        # Max 4h bars in cluster (R45: zombie filter)
    ma_cluster_min_momentum: float = 0.01    # Min |momentum| for entry (R45: was 0.003)
    ma_cluster_sl: float = 0.02              # 2% stop loss
    ma_cluster_tp: float = 0.08              # 8% take profit
    ma_cluster_max_hold_4h: int = 12         # 12 × 4h = 48h
    ma_cluster_leverage: float = 21.0        # Half-Kelly: f*/2=4.33%, L=0.0433×10/0.02≈21
    ma_cluster_sma_periods: list[int] = field(
        default_factory=lambda: [20, 50, 100, 200],
    )

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
