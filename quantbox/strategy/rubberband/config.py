"""Rubber Band strategy configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


# ──────────────────────────── Configuration ────────────────────────────
@dataclass
class RubberBandConfig:
    """Rubber Band Scalper configuration.

    All parameters are walk-forward validated. Do NOT tighten stops
    or widen targets without re-running walk_forward_test.py.
    """
    # ── Signal parameters (v10.0, 365-day A/B validated) ──
    lookback: int = 8           # Bars to measure move (2h at 15m) — SACRED, never change
    threshold: float = 0.032   # v10.0: Lower threshold (+162% ret, +324% trades vs 0.045)
    target_retrace: float = 1.5    # v12.4 HybTight: 1.5× retrace TP1 (was 1.0, AB-test winner)
    stop_mult: float = 0.382   # v12.3 T100_S38_PBT: Fibonacci stop (regime-stable)
    max_hold: int = 25          # v12.4 HybTight: 25 bars patience (was 20)
    stop_decay: float = 0.0    # Stop tightening rate (0=off, disabled for robustness)

    # ── Entry Mode (v6.0, A/B validated) ──
    # 'confidence': v6.0 — regime-switch + multi-evidence confidence scoring.
    #   Same regime-switch entry + RSI, volume, taker ratio, SMA distance evidence.
    #   min_confidence=2 filters 34 bad trades → Calmar 8.76 (+9% vs v5.0).
    # 'regime_switch': v5.0 — BTC regime detection → dual algo dispatch.
    # 'btc_directional': v4.4 legacy. Long only when BTC falling.
    # 'raw': No filter (baseline).
    entry_mode: str = 'confidence'         # 'confidence' | 'regime_switch' | 'btc_directional' | 'raw'
    btc_regime_period: int = 96            # BTC SMA period for regime detection (96 = 24h)
    asset_trend_period: int = 192          # Asset SMA for classic pullback (192 = 48h)

    # ── Confidence Scoring (v10.0, MTF acts as quality gate) ──
    min_confidence: int = 1                # v10.0: Lowered to 1 (MTF is the real filter, +86.9% ret)
    confidence_leverage_step: float = 0.0  # Extra leverage per confidence point above 1 (0=flat)
    rsi_period: int = 14                   # RSI calculation period
    vol_period: int = 20                   # Volume SMA period for spike detection

    # ── Multi-timeframe confirmation (DISABLED — old single-TF SMA overlay) ──
    htf_period: int = 0          # Higher TF trend lookback (0=DISABLED)
    htf_block_threshold: float = 0.01  # Block entry if HTF trend > 1% against trade dir
    htf_agree_discount: float = 0.85   # Threshold multiplier when HTF agrees
    htf_agree_stop_mult: float = 0.95  # Stop multiplier when HTF agrees

    # ── v7.0 Multi-Timeframe Confirmation (A/B validated, 365-day) ──
    # Uses ACTUAL 4h candle data (not SMA overlay on 15m).
    # 4h SMA(12) trend confirms 15m entries: blocks counter-trend trades.
    # Result: MDD 16.5% → 4.3% (-74%), Calmar 8.65 → 32.42 (+275%)
    # Walk-forward: Q3 DD drops 16.5% → 3.0% (the critical quarter).
    mtf_confirmation_enabled: bool = True    # Enable 4h trend confirmation
    mtf_sma_period: int = 12                 # 4h SMA period (12 bars = 2 days)
    mtf_leverage_mult: float = 2.5           # v10.1: MTF leverage boost (2.0→2.5, +35% ret, +7.5% Calmar)
                                             # Effective: 1.5 × 2.5 = 3.75x per trade, 1.875x on total equity

    # ── v8.0 Dynamic Target (A/B validated, winning strategy insight) ──
    # "When certain, ride the wave" — scale target by confidence level.
    # Higher confidence → wider target → capture more of the snap-back.
    # DD unchanged (4.3%), Calmar +9.3% (27.71 → 30.29).
    # Inspired by competition winner: "ride 3rd/5th waves at high conviction."
    dynamic_target_enabled: bool = False     # Disabled in v9.0 — Fib discipline outperforms
    target_conf3: float = 0.45              # Target at confidence=3 (vs 0.4 default)
    target_conf4: float = 0.50              # Target at confidence=4
    target_conf5: float = 0.55              # Target at confidence≥5

    # ── v9.0 S+R+M Confluence (A/B validated, winning strategy deep integration) ──
    # MACD histogram reversal as confidence evidence (+7.6% Calmar).
    # Stochastic %K/%D cross as confidence evidence (+3.7% Calmar).
    # Combined with Fib exits: +33.6% Calmar, -8.5% DD.
    # v12.6: DISABLED — R21 (6yr×50sym backtest) shows Stoch/MACD = noise.
    # Removing both boosted WR 51%→62%, Calmar 728→1171.
    macd_enabled: bool = False              # v12.6: DISABLED (WR +11pp without)
    macd_fast: int = 12                     # MACD fast EMA period
    macd_slow: int = 26                     # MACD slow EMA period
    macd_signal_period: int = 14            # MACD signal EMA period
    stoch_enabled: bool = False             # v12.6: DISABLED (WR +11pp without)
    stoch_k_period: int = 3                 # v10.0: Faster stochastic (+5% Calmar vs 5)
    stoch_d_period: int = 3                 # Stochastic %D period

    # ── v12.6 Climax Volume Filter (R21 champion [E]: Calmar 728→1171) ──
    # Climax = extreme volume spike >N× 20-bar SMA.  Signals exhaustion at
    # the rubber-band extreme → high-conviction mean-reversion entry.
    # R21: Clmx×3.0 + ATR2×stop = WR 62.2%, DD 5.8%, Calmar 1171.
    climax_enabled: bool = True             # vol spike as confidence evidence
    climax_vol_mult: float = 3.0            # vol > 3.0×vol_SMA(20) = climax (R21 best)

    # ── v12.6 ATR-Based Dynamic Stop (R21 champion [E]) ──
    # Replaces fixed 0.382×move stop with dynamic 2×ATR%.
    # Wider on noisy small moves (fewer ghost SLs), tighter on big moves
    # (faster profit lock-in). R21: WR 51%→62%, Max DD 7.7%→5.8%.
    atr_stop_enabled: bool = True           # ATR-based SL replaces fixed ×move stop
    atr_stop_mult: float = 2.0             # SL = 2.0×ATR% (R21 best)
    atr_stop_floor: float = 0.005          # Min SL distance 0.5% (prevent noise SL)
    atr_stop_cap: float = 0.04             # Max SL distance 4.0% (risk control)

    # ── v11.4 Adaptive Threshold (Walk-Forward Validated) ──
    # Per-bar threshold = clip(atr_multiplier × ATR%(atr_period), floor, cap)
    # Chosen config: 2.5x_f1 — OOS Sharpe 8.23, OOS/IS 1.09, zero overfitting.
    # When atr_period=0, falls back to fixed `threshold` above.
    atr_period: int = 48                    # ATR EMA period (48 × 15m = 12h)
    atr_multiplier: float = 2.5             # ATR multiplier for threshold calc
    threshold_floor: float = 0.01           # Minimum threshold (1%)
    threshold_cap: float = 0.06             # Maximum threshold (6%)

    # ── v7.0 Independent 4H Rubber Band (A/B validated) ──
    # Trades rubber band signals on 4h timeframe independently.
    # 4h-only: Ret +52.2%, DD 4.8%, Calmar 10.87.
    # Combined with MTF-confirmed 15m: 70%/30% allocation.
    # v10.2: Disabled — on_bar_4h() never called in scalper.py,
    # and not used in backtest scripts. Dead code in production.
    htf_independent_enabled: bool = False    # Disabled: dead code in production
    htf_rb_lookback: int = 3                 # 4h lookback (12h)
    htf_rb_threshold: float = 0.10           # 10% threshold (higher for 4h noise)
    htf_rb_target_retrace: float = 0.5       # 50% retracement target
    htf_rb_stop_mult: float = 0.8            # 0.8x stop
    htf_rb_max_hold: int = 12                # 12 bars (2 days)
    htf_rb_btc_sma_period: int = 48          # BTC SMA for 4h regime (8 days)

    # ── Regime filter ──
    regime_filter_enabled: bool = False   # Disabled: blocks the very moves that generate alpha
    regime_lookback: int = 96    # 24h of 15m bars
    regime_trend_threshold: float = 0.08  # 8% BTC move in 24h → halt

    # ── Costs (conservative) ──
    fee_rate: float = 0.0002     # 0.02% maker fee per side
    slippage_rate: float = 0.0002  # 0.02% slippage with limit chaser
    funding_rate_8h: float = 0.0001  # 0.01% per 8h
    bar_interval_minutes: int = 15

    # ── Dynamic Leverage (v6.0: flat 1.5x default, confidence optional) ──
    # Conservative (DD ≈ 16.9%): base=1.5 step=0.0 max=1.5 → Calmar 8.76
    # Moderate (DD ≈ 22%): base=1.0 step=1.0 max=2.0 → Calmar 9.99
    # 1.5x flat is conservative sweet spot for live trading.
    dynamic_leverage_enabled: bool = True  # Enable leverage pipeline
    base_leverage: float = 1.5     # Flat leverage for all trades
    leverage_scale: float = 0.0    # Legacy: No confidence scaling (use confidence_leverage_step)
    max_leverage: float = 20.0    # Final hard cap after MTF mult

    # ── BTC Directional Alignment (v4.2, A/B validated) ──
    # Counter-trend entries only when BTC confirms: long when BTC falling,
    # short when BTC rising. Filters out idiosyncratic altcoin moves that
    # are less likely to revert.
    # v11.0: Re-enabled with period=48 (12h BTC momentum window).
    # A/B test: unlocks Risk-Off trades (sympathy pullback).
    # Result: Ret 1331% → 6556% (+393%), Calmar 5.14 → 19.65 (+282%).
    # Robust across all tested periods (16-96), not curve-fit.
    btc_directional_enabled: bool = True   # Align entry with BTC momentum
    btc_directional_period: int = 48       # v11.0: 12h BTC momentum window (A/B validated)

    # ── v12.3 Partial-BE-Trail Exit Mode (PBT, R7/R8 validated) ──
    # 50% partial at TP1 → breakeven floor → ATR trailing for runner.
    # R:R structural improvement: 0.49 → 1.66, regime-stable.
    partial_exit_enabled: bool = True      # v12.3: PBT mode (R7/R8 champion)
    partial_exit_fraction: float = 0.5     # Fraction to close at TP1 (0.5 = 50%)
    trail_atr_mult: float = 1.0            # v12.4 HybTight: 1.0×ATR (tighter trail, was 1.5)
    breakeven_after_partial: bool = True    # Move stop to breakeven after TP1
    runner_max_hold_mult: float = 3.0      # v12.4 HybTight: runner max_hold = N × max_hold (was hardcoded 2)

    # ── v12.3.3 Pre-TP1 Trailing Parameters (R5 A/B validated defaults) ──
    # Controls when the "give-back prevention" trailing kicks in BEFORE TP1.
    # Activation: trail starts when best_pnl > target × activation.
    # Factor: exit when gain < best_pnl × factor (gave back too much).
    # Higher activation → trail only when closer to target (more R:R headroom).
    # Higher factor → wider retracement tolerance (less trigger-happy).
    pre_tp1_trail_activation: float = 0.5  # R5 default: trail after 50% of target
    pre_tp1_trail_factor: float = 0.3      # R5 default: exit at 30% of best

    # ── v12.3 SteepX Confidence Sizing (R6 M-2SSTSX, A/B validated) ──
    # Scale position size by confidence level. Low conf → tiny, high conf → big.
    steep_cap_curve: bool = True           # Enable confidence-based sizing
    cap_mult_conf2: float = 0.3            # conf=2: 30% of base slot
    cap_mult_conf3: float = 0.5            # conf=3: 50% of base slot
    cap_mult_conf4: float = 1.5            # conf=4: 150% of base slot
    cap_mult_conf5: float = 2.0            # conf≥5: 200% of base slot

    # ── v12.3 SoftTight DD Scaling (R6 M-2SSTSX, A/B validated) ──
    # Scale down position size during drawdowns (double safety with circuit breaker).
    soft_equity_scale: bool = True         # Enable DD-based size reduction
    soft_eq_dd1: float = 0.02              # 2% DD → 80% sizing
    soft_eq_scale1: float = 0.80
    soft_eq_dd2: float = 0.04              # 4% DD → 60% sizing
    soft_eq_scale2: float = 0.60
    soft_eq_dd3: float = 0.06              # 6% DD → 40% sizing
    soft_eq_scale3: float = 0.40

    # ── v12.3 Vol-Adjusted Sizing (R6 M-2SSTSX, A/B validated) ──
    # Normalize position size by current volatility vs median.
    # High vol → smaller position, low vol → bigger position.
    vol_adjusted_size: bool = True         # Enable vol-adjusted sizing
    vol_adj_median_atr: float = 0.01       # Assumed median ATR% (1%)
    vol_adj_min: float = 0.4               # Min vol factor (0.4x)
    vol_adj_max: float = 2.0               # Max vol factor (2.0x)

    # ── v12.3 Direction Diversity (R6 M-2SSTSX, A/B validated) ──
    # Prevent concentrated same-direction exposure (13:00 cascade protection).
    dir_diversity_enabled: bool = True     # Enable direction diversity filter
    max_same_direction: int = 2            # Max positions in same direction

    # ── R:R-Gated Leverage (v4.4, safety gate) ──
    # With stop=0.9, target=0.4: theoretical R:R = 0.44 > 0.30, gate always
    # passes. Kept as safety: if params changed to worse R:R, it disables leverage.
    rr_leverage_enabled: bool = True       # Gate leverage by R:R ratio
    min_rr_for_leverage: float = 0.30      # Minimum R:R to allow leverage

    # ── Risk management ──
    risk_per_trade: float = 0.01   # Risk 1% of capital per trade
    max_concurrent: int = 5        # Max simultaneous positions
    max_drawdown_pct: float = 0.10  # 10% session drawdown → halt

    # ── Symbol selection (v10.0: whitelist filter) ──
    min_volume_rank: int = 30  # Top 30 by volume
    exclude_symbols: list[str] = field(default_factory=lambda: ['USDCUSDT', 'TUSDUSDT', 'BUSDUSDT'])
    symbol_whitelist: list[str] = field(default_factory=lambda: [
        'FILUSDT', '1000PEPEUSDT', 'XRPUSDT', 'XLMUSDT', 'ADAUSDT',
        'FETUSDT', 'ETHUSDT', 'ARBUSDT', '1000SHIBUSDT', 'LINKUSDT',
        'ATOMUSDT', 'OPUSDT', 'DOTUSDT', 'NEARUSDT', 'AAVEUSDT',
        'BCHUSDT', 'HBARUSDT', 'SOLUSDT', 'ICPUSDT', 'BNBUSDT',
    ])  # v11.1 Top-20: 3384% ret, Calmar 155.22, A/B validated vs Top-15 baseline

    # ── Dynamic capital allocation (v10.0) ──
    dynamic_capital_enabled: bool = True  # balance / max(1, n_active+1) instead of fixed slots

    @property
    def cost_per_side(self) -> float:
        return self.fee_rate + self.slippage_rate

    @property
    def round_trip_cost(self) -> float:
        return 2 * self.cost_per_side

    @property
    def bars_per_funding(self) -> int:
        return (8 * 60) // self.bar_interval_minutes


