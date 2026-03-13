"""VolStraddle strategy configuration dataclass.

r47-Exit (2026-03-07): Exit param overhaul validated vs v14.3.0 (N=3713, PF=1.324).
  Option D chosen: r47 best exits + TBR OFF (N=4097, PF=1.521, Sharpe=5.06, MaxDD=83.3%).
  D_mom8: trail_activate 2%→0.5%, trail_distance 0.3%→0.2%, F5 TBR disabled.
  MA Cluster: flat TP=8% → trailing stop (activate=4%, distance=0.5%).
  All 6 years profitable. Bootstrap CI [1.418, 1.634], P(PF≥1)=100%.

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

import os

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VolStraddleConfig(BaseSettings):
    """VolStraddle (VR compression) strategy configuration.

    All values match R39-VR856 exactly for backtest-live parity.
    Reads from .env / environment via env_prefix='VS_'.
    """

    model_config = SettingsConfigDict(
        env_prefix="VS_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_list_separator=",",
        extra="ignore",
    )

    # ── VR Computation ──
    vr_short: int = 8  # Short ATR period for VR (R39: was 7)
    vr_long: int = 56  # Long ATR period for VR (R39: was 50)

    # ── D_mom8: Directional Momentum ──
    d_vr_threshold: float = 0.50  # A1-D50: D VR entry threshold (was 0.45)
    d_sl: float = 0.01  # 1% stop loss
    d_tp: float = 0.20  # 20% safety cap — dead parameter (r67: trail exits first; was 0.12)

    # ── D_mom8: Trailing Stop (R43 validated) ──
    # Once unrealized gain reaches trail_activate, a trailing stop
    # follows at trail_distance below peak. Hard TP at d_tp (20% safety cap, rarely hit).
    # Mechanism: lets VR squeeze explosive releases run beyond fixed TP.
    d_trail_activate: float = 0.010  # Activate trailing after +1.0% gain (r69: 0.005→0.010, OOS +38%/+23%/+40%)
    d_trail_distance: float = 0.002  # Trail 0.2% behind peak (r47: was 0.003)

    # ── S_d8: Straddle (requires Binance Hedge Mode) ──
    s_vr_threshold: float = 0.57  # S VR entry threshold (r58/r59: 0.55→0.57 +28%N +11%PF)
    s_squeeze_dur: int = 3  # Min consecutive bars below S VR threshold (r58/r59: 4→3 +25%N +12%PF)
    s_sl: float = 0.015  # 1.5% stop loss (per leg)
    s_tp: float = 0.05  # 5% hard TP ceiling (per leg, safety)
    s_trail_activate: float = 0.015  # r53: activate trailing after +1.5% gain per leg
    s_trail_distance: float = 0.002  # r53: trail 0.2% behind peak
    straddle_enabled: bool = True  # S_d8 straddle mode enabled
    hedge_mode: bool = True  # Binance Hedge Mode (dualSidePosition=true)

    # ── Direction ──
    momentum_period: int = 8  # Momentum lookback for D direction

    # ── Max Hold (bars at each TF, all ≈ 48h real-time) ──
    max_hold_15m: int = 192  # 192 × 15m = 48h
    max_hold_30m: int = 96  # 96  × 30m = 48h
    max_hold_1h: int = 48  # 48 × 1h = 48h
    max_hold_4h: int = 12  # 12 × 4h = 48h

    # ── Multi-TF Control ──
    enable_15m: bool = True
    enable_30m: bool = True  # 30m TF (resampled from 15m, ratio=2) — r57 A0: N+8.9%, Calmar+43.7%
    enable_1h: bool = True
    enable_4h: bool = True

    # ── VR warmup (bars before first signal per TF) ──
    warmup_15m: int = 200  # ~50h of 15m bars (~vr_long×4)
    warmup_30m: int = 100  # ~50h of 30m bars (same wall-clock as warmup_15m)
    warmup_1h: int = 50  # ~50h of 1h bars
    warmup_4h: int = 13  # ~52h of 4h bars

    # ── Position Sizing & Leverage (Half-Kelly derived) ──
    base_leverage: float = 30.0  # Half-Kelly: f*/2=3.07%, L=0.0307×10/0.01≈30
    max_leverage: float = 30.0  # Hard cap (matches base_leverage)
    risk_per_trade: float = 0.01  # 1% risk per trade

    # ── Risk Management ──
    max_concurrent: int = 10  # 10 slots (VR L/N=30/10=3.0, MA L/N=21/10=2.1)
    max_drawdown_pct: float = 0.30  # 30% session drawdown → halt (r48 analysis: P99 session DD)

    # ── Costs (conservative) ──
    fee_rate: float = 0.0002  # 0.02% maker fee per side
    slippage_rate: float = 0.0002  # 0.02% slippage with limit chaser

    # ── Direction Diversity ──
    dir_diversity_enabled: bool = True
    max_same_direction: int = 2

    # ── Dynamic Capital ──
    # Disabled: vol_straddle uses fixed balance/max_concurrent (backtest parity).
    # Dynamic formula (balance/n_active+1) causes -2019 margin errors on 10-slot config.
    dynamic_capital_enabled: bool = False
    # Safety cap: when dynamic capital enabled, never allocate more than this fraction
    # of balance to a single slot (prevents first-slot over-concentration).
    dynamic_capital_max_fraction: float = 0.5

    # ── Live-Backtest Divergence Monitor ──
    # Sends Discord alert when live WR/PF diverges from v14.5.0 backtest baseline.
    divergence_min_trades: int = 30  # Minimum live trades before checking
    divergence_pf_threshold: float = 0.80  # Alert if live PF < threshold (baseline=1.563)
    divergence_wr_gap_pp: float = 10.0  # Alert if live WR is X pp below baseline (43.4%)
    divergence_cooldown_hours: float = 24.0  # Min hours between alerts

    # ── SoftTight DD Scaling (safety mechanism) ──
    soft_equity_scale: bool = True
    soft_eq_dd1: float = 0.02  # 2% DD → 80% sizing
    soft_eq_scale1: float = 0.80
    soft_eq_dd2: float = 0.04  # 4% DD → 60% sizing
    soft_eq_scale2: float = 0.60
    soft_eq_dd3: float = 0.06  # 6% DD → 40% sizing
    soft_eq_scale3: float = 0.40

    # ── CW-Tier Conviction Sizing (R34 validated) ──
    # Score = count of 4 conditions: E2, C2, B3, A3 (0-4)
    # CW-Tier multiplier applied to position size based on score.
    # T3-Triple: optional high-conviction filter (score ≥ min_conviction_score).
    cw_enabled: bool = True  # Enable conviction-weighted sizing
    min_conviction_score: int = 0  # Min score to enter (0=all, 3=T3-Triple)
    cw_multipliers: list[float] = Field(
        default_factory=lambda: [
            0.5,  # score 0 → 0.5x
            1.0,  # score 1 → 1.0x (baseline)
            1.5,  # score 2 → 1.5x
            2.0,  # score 3 → 2.0x
            2.5,  # score 4 → 2.5x (max conviction)
        ]
    )

    # ── E2: SMA Proximity ──
    # |close - SMA(20)| / SMA(20) < threshold → near mean → squeeze valid
    e2_enabled: bool = True
    e2_sma_period: int = 20  # SMA lookback for mean proximity
    e2_threshold: float = 0.005  # 0.5% max distance from SMA

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
    a3_deep_vr_threshold: float = 0.35  # D_mom8 deep compression
    a3_long_squeeze_dur: int = 16  # S_d8 extended squeeze duration

    # ── F5: TBR z-score Gate (R46 validated, D_mom8 only) ──
    # Taker Buy Ratio z-score computed over lookback 15m bars.
    # Removes crowded-flow trades: skip LONG if z > thresh (retail FOMO),
    # skip SHORT if z < -thresh (retail panic selling).
    # Backtest result (6yr, 46 symbols): PF 1.306→1.348 (+0.042),
    # WR 42.6%→46.4% (+3.8%), N -14.8%, p=0.029.
    # IMPORTANT: Applied to D_mom8 only. S_d8 straddle is direction-neutral.
    f5_tbr_enabled: bool = (
        False  # r47: disabled; r71: re-validated with trail_activate=1.0% → KEEP disabled (2024/2025 OOS degrade)
    )
    f5_tbr_z_threshold: float = 0.5  # R46b optimal (lb=192, p=0.029)
    f5_tbr_z_window: int = 192  # Lookback bars (~2 days of 15m)

    # ── G6: BTC Trend Gate (live regime filter) ──
    # Suppress LONG entries when BTC is below its moving average.
    # Prevents systematic losses during multi-day crypto downtrends.
    # SHORT entries are always allowed (downtrend = short opportunity).
    # Live only — backtest uses all historical periods including bear markets.
    btc_long_filter_enabled: bool = True
    btc_long_ma_period: int = 50  # 50 bars × 15m = 12.5h trend MA
    btc_long_ma_interval: str = "15m"  # timeframe for MA computation (must match cycle interval)

    # ── Symbol Selection ──
    # Source of truth for the production universe (60 symbols, r53 validated).
    # To override: set VS_SYMBOL_WHITELIST in .env — but update config.py default too.
    symbol_whitelist: list[str] = Field(
        default_factory=lambda: [
            "ADAUSDT",
            "ALGOUSDT",
            "ATOMUSDT",
            "AVAXUSDT",
            "AXSUSDT",
            "AAVEUSDT",
            "BANDUSDT",
            "BATUSDT",
            "BCHUSDT",
            "BELUSDT",
            "BNBUSDT",
            "BTCUSDT",
            "COMPUSDT",
            "CRVUSDT",
            "DASHUSDT",
            "DOGEUSDT",
            "DOTUSDT",
            "EGLDUSDT",
            "ENJUSDT",
            "ETCUSDT",
            "ETHUSDT",
            "FILUSDT",
            "GRTUSDT",
            "ICXUSDT",
            "IOSTUSDT",
            "IOTAUSDT",
            "KAVAUSDT",
            "KNCUSDT",
            "KSMUSDT",
            "LINKUSDT",
            "LRCUSDT",
            "LTCUSDT",
            "NEARUSDT",
            "NEOUSDT",
            "ONTUSDT",
            "QTUMUSDT",
            "RLCUSDT",
            "RSRUSDT",
            "RUNEUSDT",
            "SANDUSDT",
            "SKLUSDT",
            "SNXUSDT",
            "SOLUSDT",
            "STORJUSDT",
            "SUSHIUSDT",
            "THETAUSDT",
            "TRBUSDT",
            "TRXUSDT",
            "UNIUSDT",
            "VETUSDT",
            "XLMUSDT",
            "XMRUSDT",
            "XRPUSDT",
            "XTZUSDT",
            "YFIUSDT",
            "ZECUSDT",
            "ZENUSDT",
            "ZILUSDT",
            "ZRXUSDT",
            "1INCHUSDT",
        ]
    )
    exclude_symbols: list[str] = Field(
        default_factory=lambda: [
            "USDCUSDT",
            "TUSDUSDT",
            "BUSDUSDT",
        ]
    )
    min_volume_rank: int = 30

    # ── MA Cluster Squeeze (R44d V17, 4h-only) ──
    # Complementary strategy: 4 SMAs cluster → breakout.
    # Fires when VR squeeze is idle (93.7% non-overlapping).
    ma_cluster_enabled: bool = True  # Enable MA cluster signals
    ma_cluster_pct: float = 0.03  # Max spread among 4 SMAs (3%)
    ma_cluster_min_duration: int = 8  # Min 4h bars in cluster (≈32h)
    ma_cluster_max_duration: int = 24  # Max 4h bars in cluster (R45: zombie filter)
    ma_cluster_min_momentum: float = 0.01  # Min |momentum| for entry (R45: was 0.003)
    ma_cluster_sl: float = 0.02  # 2% stop loss
    ma_cluster_tp: float = 0.08  # 8% hard TP ceiling (safety; r47: trailing fires first)
    ma_cluster_trail_activate: float = 0.04  # r47: activate trailing at +4% gain
    ma_cluster_trail_distance: float = 0.005  # r47: trail 0.5% behind peak

    # r65: Server-side trail SL ratchet — closes bar-close execution gap (ΔPF=-0.950)
    # When True, places/updates Binance STOP order at trail_stop_gain price after
    # trail activation, ensuring intrabar execution matching backtest assumption.
    trail_server_ratchet_enabled: bool = True
    ma_cluster_max_hold_4h: int = 12  # 12 × 4h = 48h
    ma_cluster_leverage: float = 21.0  # Half-Kelly: f*/2=4.33%, L=0.0433×10/0.02≈21
    ma_cluster_sma_periods: list[int] = Field(
        default_factory=lambda: [20, 50, 100, 200],
    )

    # ── Compatibility fields (used by scalper internals) ──
    # These map vol straddle concepts to the scalper's existing interface
    # so that shared code (risk checks, brackets, logging) works unchanged.
    lookback: int = 8  # Used for move calc in logging — not for signal
    threshold: float = 0.0  # Not used — signal is VR-based
    bar_interval_minutes: int = 15

    # ── Daily Loss Circuit Breaker ──
    # Block new entries for the rest of the day after realized loss exceeds limit.
    # Prevents the "keep entering and losing" spiral during bad market sessions.
    # Resets at UTC midnight. Set to 0 to disable.
    daily_loss_limit_pct: float = 0.05  # 5% of account balance

    @model_validator(mode="after")
    def _populate_symbol_whitelist(self) -> "VolStraddleConfig":
        if not self.symbol_whitelist:
            rb_whitelist = os.environ.get("RB_SYMBOL_WHITELIST", "")
            if not rb_whitelist:
                # pydantic-settings reads .env for VS_* fields but doesn't set
                # os.environ — fall back to reading .env directly for RB_ fields.
                try:
                    from dotenv import dotenv_values

                    rb_whitelist = dotenv_values(".env").get("RB_SYMBOL_WHITELIST", "")
                except Exception:
                    pass
            if rb_whitelist:
                object.__setattr__(self, "symbol_whitelist", [s.strip() for s in rb_whitelist.split(",") if s.strip()])
        return self

    @property
    def cost_per_side(self) -> float:
        return self.fee_rate + self.slippage_rate

    @property
    def round_trip_cost(self) -> float:
        return 2 * self.cost_per_side

    def get_max_hold(self, tf: str) -> int:
        """Get max hold bars for a timeframe."""
        return {
            "15m": self.max_hold_15m,
            "30m": self.max_hold_30m,
            "1h": self.max_hold_1h,
            "4h": self.max_hold_4h,
        }.get(tf, self.max_hold_15m)

    def get_warmup(self, tf: str) -> int:
        """Get warmup bars for a timeframe."""
        return {
            "15m": self.warmup_15m,
            "30m": self.warmup_30m,
            "1h": self.warmup_1h,
            "4h": self.warmup_4h,
        }.get(tf, self.warmup_15m)
