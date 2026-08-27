"""VolStraddle strategy configuration — pydantic-settings class.

All strategy parameters live here. .env values (prefix VS_) override defaults.

Half-Kelly Leverage Derivation (20260304_ef55e2e strategy report):
  Kelly f* = WR - (1-WR)/RR,  Leverage = (f*/2) × N / SL
  VR:  WR=25.9%, PF=1.310, RR=3.75 → f*=6.1%, f*/2=3.07% → L=30.7→30x
  MA:  WR=38.6%, PF=1.285, RR=2.05 → f*=8.65%, f*/2=4.33% → L=21.6→21x
  Per-trade risk: VR=3.0%, MA=4.2% of total balance.
"""

from __future__ import annotations

import os

from typing import Annotated

from pydantic import Field, field_validator, model_validator
from quantbox.core.strategy_names import DONCHIAN, MA_CLUSTER, VR_DIRECTIONAL
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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
    d_trail_activate: float = (
        0.015  # r106→live: 4.0%→1.5% (bar-close MFE avg 1.04%, 4% unreachable; 10x lev → 1.5% raw = 15% pos)
    )
    d_trail_distance: float = 0.005  # r106: 0.2%→0.5% (Binance callbackRate floor, rule #8)

    # ── D_mom8: Symbol Blacklist (r116b quickwin) ──
    # Symbols excluded from D_mom8 entries (strategy.py:on_bar gate).
    # Derived from baseline per-symbol PF analysis; expected quarterly refresh.
    # Does NOT affect C2 BTC momentum signal (reads BTCUSDT data separately),
    # nor MA Cluster / Donchian strategies.
    # Env var format: VS_D_SYMBOL_BLACKLIST="SYM1,SYM2,SYM3"
    # Annotated with NoDecode so pydantic-settings passes the raw env value to
    # the validator below instead of JSON-decoding. Enables comma-separated
    # env form: VS_D_SYMBOL_BLACKLIST="FOO,BAR,BAZ"
    d_symbol_blacklist: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Symbols to skip for D_mom8 entries (PF<baseline-threshold). "
        "Quarterly refresh expected. Applied in strategy.py:on_bar post-signal gate.",
    )

    hedge_mode: bool = True  # Binance Hedge Mode (dualSidePosition=true)

    # ── Server-Side Trailing Stop (r90 migration) ──
    # When True, places Binance TRAILING_STOP_MARKET at entry alongside SL.
    # Binance tracks peak tick-by-tick and fires MARKET when price drops
    # callbackRate% from peak after activationPrice. Matches IDEAL sim model.
    # Server-side TRAILING_STOP_MARKET via regular API (POST /fapi/v1/order).
    trail_server_native: bool = True

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
    # 2026-08-10: 0.06 → 0.005. 기존 값은 아티팩트가 섞인 백테스트로 산출한 Kelly에
    # 기반했다. 최근 3년 실측(N=828, PF 1.195)을 블록 부트스트랩(4000회)한 결과
    # 0.06에서는 80% 초과 드로다운 확률이 99.1%, 3년 손실 확률이 45.2%였다 —
    # 켈리 정점을 한참 지나 수익 자체를 깎는 구간이다(15배가 30배보다 수익이 높다).
    # 0.005에서 P(DD>80%)=0.0%, P(DD>50%)=0.3%, 중앙 연수익 +15.31%.
    # (ref: docs/backtest_results/202608/20260810_f3191ec_trail_removal_long_hold.md §12)
    risk_per_trade: float = 0.005

    # ── Risk Management ──
    max_concurrent: int = 10  # 10 slots (VR L/N=30/10=3.0, MA L/N=21/10=2.1)
    max_drawdown_pct: float = 0.25  # 25% session drawdown → halt (2026-04-19: raised 15→25% for quarter-Kelly risk=6%/trade; allows ~4 SL before halt)
    # 2026-08-27: 원금 캡핑. 선물 계좌 잔고가 이 값을 넘으면 **초과분은 사이징에서
    # 제외한다** — 계좌에 남아 있어도 포지션을 그만큼 키우지 않는다. 초과분은 디스코드로
    # 알리고 사람이 현물/USDT로 뺀다(출금 권한을 API 키에 주는 게 더 큰 위험이라 봇이
    # 직접 출금하지 않는다).
    # 근거: 실전 트레이더들의 공통 자금관리 규칙 — 복리로 노출을 키우면 단일 청산 사건의
    # 절대 손실이 같이 커진다. 전략 성과와 무관하게 작동하는 유일한 안전장치다.
    # 0 = 비활성(기존 동작과 동일).
    equity_cap_usdt: float = Field(default=0.0, ge=0.0)

    max_exposure_multiplier: float = (
        35.0  # Max total open notional / balance. 35x = 17% above 10×30x/10=30x theoretical max
    )

    # ── Costs (conservative) ──
    # 2026-08-12 실측 정정. 이전 값 0.0002는 "양쪽 다 maker 체결"을 전제했으나 사실이 아니다.
    # Binance income history 실조회(2026-02-05~05-04, 7,624행) + ma_cluster 60건 대조:
    #   실제 요율   maker 2bp / taker 5bp, feeTier 0, BNB 할인 미적용
    #   왕복 실측   중앙 9.90bp → 편도 블렌디드 **4.3bp**
    #   내역        10bp×36건(양쪽 taker) / 7bp×18건(maker진입+taker청산) / 4~6bp×6건
    # 진입 maker 체결률이 30~40%뿐이고 청산은 전량 taker다. 청산이 taker인 것은 구조적이다 —
    # exit_reason의 97%가 서버 스톱 트리거(STOP_MARKET/TRAILING_STOP_MARKET)이고, 이를 지정가로
    # 바꾸는 것은 안전장치 제거다. 따라서 이 값은 완화 가능한 가정이 아니라 측정된 사실이다.
    fee_rate: float = 0.00043  # 4.3bp per side (measured blended maker/taker)
    # 슬리피지는 실측 불가 상태다 — 소켓 수집 구간(2026-08-09~)과 라이브 체결 구간(~2026-05-04)이
    # 겹치지 않는다. 스프레드 실측(중앙 2.42bp)에 기반한 모델값은 왕복 3.2bp이나, 모델을 근거로
    # 비용 가정을 낮추지 않는다(보수성 유지). 라이브 재개 후 실측되면 그때 정정한다.
    slippage_rate: float = 0.0002  # 0.02% slippage — modeled, deliberately not relaxed
    slippage_mode: str = "flat"  # flat | tiered | conservative

    # ── Funding Cost (G3 parity fix) ──
    # Binance charges funding every 8h on open positions.
    # At 30x leverage, 0.01% funding = 0.3% equity drag per 8h.
    # Default is conservative average from 2024-2025 BTC/altcoin funding data.
    # Set to 0.0 to disable funding in backtest (legacy behavior).
    default_funding_rate_8h: float = 0.0001  # 0.01% per 8h (on notional)
    # 2026-08-27: 펀딩 부호. 기본 모델은 |펀딩|의 평균을 양방향 모두에게 물린다 —
    # 실측(34심볼)으로는 20h 보유 기준 4.78bp를 양쪽에 물리지만, 실제로는 롱이 1.47bp를
    # 내고 숏이 1.47bp를 받는다(부호 있는 평균이 양수인 심볼 85.3%). 즉 한쪽 편들기가
    # 아니라 **양쪽 과대 청구**이고, 방향 간 상대 격차는 20h당 약 2.9bp다.
    # True로 켜면 부호 있는 실측을 쓴다 = 백테스트가 낙관적으로 바뀐다. 그래서 기본은
    # False다 — 비용 가정 완화는 라이브 실측 검증 후에만 한다.
    funding_direction_aware: bool = False

    # ── Direction Diversity ──
    dir_diversity_enabled: bool = True
    max_same_direction: int = 5  # r108: 2→5 (Calmar +67%, MaxDD -2%p, 10슬롯 완전 활용)

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
    # 2026-04-19: bands widened ~3× to match quarter-Kelly risk_per_trade=6%.
    # Old 1.5/3/5% triggered on single SL hit (6% > 1.5%), defeating the risk increase.
    soft_equity_scale: bool = True
    soft_eq_dd1: float = 0.05  # 5% DD → 80% sizing (~1 SL at 6%)
    soft_eq_scale1: float = 0.80
    soft_eq_dd2: float = 0.10  # 10% DD → 60% sizing (~2 SL)
    soft_eq_scale2: float = 0.60
    soft_eq_dd3: float = 0.15  # 15% DD → 40% sizing (~2.5 SL)
    soft_eq_scale3: float = 0.40

    # ── CW-Tier Conviction Sizing (R34 validated) ──
    # Score = count of 4 conditions: E2, C2, B3, A3 (0-4 range).
    # NOTE (2026-04-17, US-URGENT-6): B3_confluence is reserved-always-False in
    # live code (strategy.py:_compute_conviction_score). Max achievable score in
    # production is therefore 3 (E2 + C2 + A3). The score-index 4 in cw_multipliers
    # is dead code preserved for backwards compat; min_conviction_score=3 == gate to
    # max-achievable, =4 or 5 blocks all entries.
    # T3-Triple: optional high-conviction filter (score ≥ min_conviction_score).
    cw_enabled: bool = True  # Enable conviction-weighted sizing
    min_conviction_score: int = 0  # Min score to enter (0=all, 3=max-achievable post-b3-reserve)
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
    b3_enabled: bool = True

    # ── A3: Quality+ ──
    # D_mom8: VR < deep_vr_threshold (extra deep compression)
    a3_enabled: bool = True
    a3_deep_vr_threshold: float = 0.35  # D_mom8 deep compression
    a3_long_squeeze_dur: int = 16  # Legacy param (kept for config compat)

    # ── F5: TBR z-score Gate (R46 validated, D_mom8 only) ──
    # Taker Buy Ratio z-score computed over lookback 15m bars.
    # Removes crowded-flow trades: skip LONG if z > thresh (retail FOMO),
    # skip SHORT if z < -thresh (retail panic selling).
    # Backtest result (6yr, 46 symbols): PF 1.306→1.348 (+0.042),
    # WR 42.6%→46.4% (+3.8%), N -14.8%, p=0.029.
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
    ma_cluster_enabled: bool = True  # r92 gate PASS: PF=2.749, MaxDD=17.7%, all 6yr profitable
    ma_cluster_pct: float = 0.03  # autoresearch: 2%→3% (PFxN +145%, MS=0.891, filter=0.956, 6yr profitable)
    ma_cluster_min_duration: int = 6  # Min 4h bars in cluster (r75: 8→6, ≈24h)
    ma_cluster_max_duration: int = 24  # Max 4h bars in cluster (R45: zombie filter)
    ma_cluster_min_momentum: float = 0.01  # Min |momentum| for entry (R45: was 0.003)
    ma_cluster_sl: float = 0.02  # 2% stop loss
    ma_cluster_tp: float = 0.08  # 8% hard TP ceiling (safety; r47: trailing fires first)
    # 2026-08-10: 트레일 비활성. 봉 해상도를 높여 재측정한 결과 트레일이 만들던 수익의
    # 95.4%가 봉 집계 아티팩트였고("고점이 먼저 왔다" 가정), 실제로는 신호의 엣지를
    # 조기 청산으로 잘라내고 있었다. 트레일 제거 + 보유 연장이 OOS 5/5 구간 양(+).
    # (ref: docs/backtest_results/202608/20260810_f3191ec_trail_removal_long_hold.md)
    ma_cluster_trail_enabled: bool = False
    ma_cluster_trail_activate: float = 0.01  # autoresearch apr03: 1.5%→1.0% (earlier trail, PF +2%)
    ma_cluster_trail_distance: float = 0.005  # r106: 0.3%→0.5% (Binance callbackRate floor, rule #8; pending sweep)

    # r65: Server-side trail SL ratchet — closes bar-close execution gap (ΔPF=-0.950)
    # When True, places/updates Binance STOP order at trail_stop_gain price after
    # trail activation, ensuring intrabar execution matching backtest assumption.
    trail_server_ratchet_enabled: bool = True
    # 2026-08-10: 6 → 24 (4일). 신호는 "맞히는 빈도"가 아니라 "맞혔을 때 오래 간다"는
    # 성질이다 — 비용 제외 평균이 보유 1봉 +0.067%(t=1.47)에서 12봉 +0.601%(t=3.97)로
    # 커진다. 24h 청산은 그 엣지를 잘라내고 있었다. 워크포워드에서 학습 구간이 고른
    # 값도 매 구간 24~36이었고 h∈[18,48] 전체가 고원이다.
    ma_cluster_max_hold_4h: int = 24  # 24 × 4h = 4일
    ma_cluster_leverage: float = 21.0  # Half-Kelly: f*/2=4.33%, L=0.0433×10/0.02≈21
    ma_cluster_sma_periods: list[int] = Field(
        default_factory=lambda: [20, 50, 100, 200],
    )
    # r92: 4h BTC trend gate for MA cluster LONGs — prevents entries during multi-day BTC downtrends.
    # Root cause of March 2026 live failure: G6 used 15m MA50 (12.5h), too short for 4h strategy.
    # 4h MA50 = 50 bars × 4h = 200h ≈ 8.3 days — matches MA cluster's strategic timeframe.
    # r122 (2026-08-27): ma_cluster 롱 차단. 6.58년/760심볼 측정에서 롱은 **타이밍 정보가
    # 없다** — 같은 심볼·같은 보유기간(20h)·같은 비용의 '무조건 롱' 대비 초과 EV가 ~1bp다.
    # 연도별로도 2024 −23.7bp(t=−2.24) / 2025 −19.8bp(t=−1.74) / 2026 −4.2bp로 3년 연속 음수.
    # BTC 게이트(4h MA50 + G6)를 다 걸어도 부호가 안 바뀐다(게이트 후 −7.5bp).
    # 반면 숏은 무조건 숏 대비 초과 +38.6bp(Welch t=6.82) — 드리프트가 아니라 신호다.
    # 단일 숫자 최적화가 아니라 '정보 없는 다리'를 떼는 변경이다.
    # (ref: docs/backtest_results/202608/20260827_b3906aa_r122_long_short_split.md)
    # 라이브는 risk_guard.allow_strategy_direction, 백테스트는 run_ma_cluster*/collect_*가
    # **같은 이 필드**를 읽는다 — 한쪽만 막히면 파리티가 깨지므로 분기점을 하나로 둔다.
    ma_cluster_long_enabled: bool = False
    ma_cluster_btc_gate_enabled: bool = True  # Block MA cluster LONGs when BTC below 4h MA50
    ma_cluster_btc_ma_period_4h: int = 50  # 50 × 4h = 200h ≈ 8.3 days
    # r115: CW conviction sizing — reduces MaxDD 14.1%→10.1% with ~flat PF
    ma_cluster_cw_enabled: bool = True
    ma_cluster_cw_spread_threshold: float = 0.015  # "tight" proxy: spread < 1.5% (backtest median ≈1.0-1.5%)
    ma_cluster_cw_volume_mult: float = 1.5  # volume spike: entry bar > N× symbol avg
    ma_cluster_cw_mom_mult: float = 2.0  # strong momentum: |mom| > N× min_momentum
    ma_cluster_cw_multipliers: list[float] = Field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.5],
    )

    # ── Donchian Channel Breakout + Volume Spike (4h-only) ──
    # Complementary strategy: 50-bar Donchian channel breakout with volume spike confirmation.
    # Fires when VR and MA Cluster are idle (95.9% non-overlapping).
    # Research: PF=4.934, WR=58.8%, MaxDD=27.6%, N=4780, 6yr all profitable.
    # Robust to 3x cost + 0.2% slippage: PF=3.803. OOS no decay (test PF=5.20).
    #
    # Half-Kelly derivation: WR=58.8%, RR=3.46, f*=46.9%, L=f*/2*N/SL=156x.
    # This is unrealistically high — likely caused by selection bias (59/59 symbols
    # profitable is suspiciously clean) and backtest-optimistic PF. Using provisional
    # 15x until live data validates actual WR/RR. Will re-derive after 50+ live trades.
    donchian_enabled: bool = False  # Disabled until live validation
    donchian_lookback: int = 50  # Channel period (4h bars = 200h ≈ 8.3 days)
    donchian_vol_sma: int = 20  # Volume SMA period for spike detection
    donchian_vol_mult: float = 2.5  # Volume must be > 2.5× SMA(20)
    donchian_breakout_pct: float = 0.02  # 2% beyond channel for confirmation
    donchian_sl: float = 0.015  # 1.5% stop loss
    donchian_tp: float = 0.20  # 20% safety cap (trail exits first)
    donchian_trail_activate: float = 0.02  # 2% to activate trailing stop
    donchian_trail_distance: float = 0.008  # 0.8% trail distance (>= 0.5% rule #8)
    donchian_max_hold_4h: int = 24  # 24 × 4h = 96h = 4 days
    donchian_leverage: float = 15.0  # PROVISIONAL — see Half-Kelly note above
    donchian_cooldown_bars: int = 3  # N 4h bars to skip after signal fires
    donchian_btc_gate_enabled: bool = True  # Block LONG during BTC downtrend (same as MA Cluster)
    donchian_btc_ma_period_4h: int = 50  # 50 × 4h = 200h ≈ 8.3 days

    # ── Compatibility fields (used by scalper internals) ──
    # These map vol straddle concepts to the scalper's existing interface
    # so that shared code (risk checks, brackets, logging) works unchanged.
    # Phase 2.5: expanded for rubberband deprecation — shared code paths
    # access these without strategy_name guards.
    lookback: int = 8  # Used for move calc in logging — not for signal
    threshold: float = 0.0  # Not used — signal is VR-based
    max_hold: int = 192  # Alias for max_hold_15m — used by shared scalper code
    bar_interval_minutes: int = 15
    stop_mult: float = 1.0  # SL multiplier (shared sizing code)
    target_retrace: float = 1.5  # TP multiplier (shared code)
    partial_exit_enabled: bool = False  # VR uses trail, not partial exit
    breakeven_after_partial: bool = False
    entry_mode: str = VR_DIRECTIONAL
    min_confidence: int = 0
    mtf_confirmation_enabled: bool = False  # VR has its own multi-TF via on_bar
    atr_stop_enabled: bool = False  # VR uses fixed SL
    atr_stop_mult: float = 2.0
    atr_stop_floor: float = 0.005
    atr_stop_cap: float = 0.04
    atr_period: int = 48
    atr_multiplier: float = 2.0
    htf_independent_enabled: bool = False
    reconciliation_qty_tolerance_pct: float = 5.0  # % tolerance for qty mismatch
    partial_exit_fraction: float = 0.5  # 50% partial close (not used by VR)
    mtf_leverage_mult: float = 2.5
    mtf_sma_period: int = 12
    threshold_cap: float = 0.10
    threshold_floor: float = 0.01
    # LimitOrderChaser params (shared execution path)
    initial_offset_ticks: int = 1
    max_chase_ticks: int = 10
    chase_interval_seconds: float = 1.0
    fallback_to_market: bool = True
    timeout_seconds: float = 30.0

    # ── Daily Loss Circuit Breaker ──
    # Block new entries for the rest of the day after realized loss exceeds limit.
    # Prevents the "keep entering and losing" spiral during bad market sessions.
    # Resets at UTC midnight. Set to 0 to disable.
    daily_loss_limit_pct: float = (
        0.12  # 12% of account balance (2026-04-19: 5→12% for quarter-Kelly risk=6%/trade; 2 SL triggers)
    )

    # ── Entry Retry / SL Cancel (shared execution params) ──
    entry_retry_count: int = 2  # VS_ENTRY_RETRY_COUNT: max retries for entry
    entry_retry_delay_s: float = 2.0  # VS_ENTRY_RETRY_DELAY_S: delay between entry retries (s)
    rollback_retry_count: int = 2  # VS_ROLLBACK_RETRY_COUNT: max retries for rollback _close_position()
    rollback_retry_delay_s: float = 2.0  # VS_ROLLBACK_RETRY_DELAY_S: delay between rollback retries (s)
    sl_cancel_retry_count: int = 2  # VS_SL_CANCEL_RETRY_COUNT: max retries for own SL cancel on TP fill
    sl_cancel_retry_delay_s: float = 1.0  # VS_SL_CANCEL_RETRY_DELAY_S: delay between SL cancel retries (s)

    @model_validator(mode="after")
    def _validate_safety_constraints(self) -> "VolStraddleConfig":
        """Validate logical constraints between config parameters.

        Catches misconfigurations that would cause silent misbehavior
        or real money loss in live trading.
        """
        errors: list[str] = []

        if self.d_sl <= 0:
            errors.append(f"d_sl must be positive, got {self.d_sl}")
        if self.d_tp <= self.d_sl:
            errors.append(
                f"d_tp must be greater than d_sl (can't TP before SL), got d_tp={self.d_tp}, d_sl={self.d_sl}"
            )
        if self.base_leverage < 1:
            errors.append(f"base_leverage must be >= 1, got {self.base_leverage}")
        if self.max_leverage < self.base_leverage:
            errors.append(
                f"max_leverage must be >= base_leverage, "
                f"got max_leverage={self.max_leverage}, base_leverage={self.base_leverage}"
            )
        if self.max_concurrent < 1:
            errors.append(f"max_concurrent must be >= 1, got {self.max_concurrent}")
        if self.d_trail_distance <= 0:
            errors.append(f"d_trail_distance must be positive, got {self.d_trail_distance}")
        if self.d_trail_distance < 0.005:
            errors.append(
                f"d_trail_distance must be >= 0.005 (Binance callbackRate >= 0.5%, rule #8), "
                f"got {self.d_trail_distance}"
            )
        if self.d_trail_distance >= self.d_trail_activate:
            errors.append(
                f"d_trail_distance must be less than d_trail_activate, "
                f"got d_trail_distance={self.d_trail_distance}, d_trail_activate={self.d_trail_activate}"
            )
        if self.ma_cluster_trail_distance < 0.005:
            errors.append(
                f"ma_cluster_trail_distance must be >= 0.005 (Binance callbackRate >= 0.5%, rule #8), "
                f"got {self.ma_cluster_trail_distance}"
            )
        if self.ma_cluster_trail_distance >= self.ma_cluster_trail_activate:
            errors.append(
                f"ma_cluster_trail_distance must be less than ma_cluster_trail_activate, "
                f"got ma_cluster_trail_distance={self.ma_cluster_trail_distance}, "
                f"ma_cluster_trail_activate={self.ma_cluster_trail_activate}"
            )
        if self.max_drawdown_pct <= self.soft_eq_dd3:
            errors.append(
                f"max_drawdown_pct must be greater than soft_eq_dd3 "
                f"(DD halt must be above max scaling threshold), "
                f"got max_drawdown_pct={self.max_drawdown_pct}, soft_eq_dd3={self.soft_eq_dd3}"
            )
        if self.daily_loss_limit_pct <= 0:
            errors.append(f"daily_loss_limit_pct must be positive, got {self.daily_loss_limit_pct}")
        if self.ma_cluster_sl <= 0:
            errors.append(f"ma_cluster_sl must be positive, got {self.ma_cluster_sl}")
        # ── Donchian safety constraints (only when enabled) ──
        if self.donchian_enabled:
            if self.donchian_sl <= 0:
                errors.append(f"donchian_sl must be positive, got {self.donchian_sl}")
            if self.donchian_trail_distance < 0.005:
                errors.append(
                    f"donchian_trail_distance must be >= 0.005 (Binance callbackRate >= 0.5%, rule #8), "
                    f"got {self.donchian_trail_distance}"
                )
            if self.donchian_trail_distance >= self.donchian_trail_activate:
                errors.append(
                    f"donchian_trail_distance must be < donchian_trail_activate, "
                    f"got {self.donchian_trail_distance} >= {self.donchian_trail_activate}"
                )
            if self.donchian_leverage > 50:
                errors.append(f"donchian_leverage cap exceeded (max 50x), got {self.donchian_leverage}")
            if self.donchian_leverage < 1:
                errors.append(f"donchian_leverage must be >= 1, got {self.donchian_leverage}")
            if self.donchian_tp <= self.donchian_sl:
                errors.append(
                    f"donchian_tp must be > donchian_sl, "
                    f"got donchian_tp={self.donchian_tp}, donchian_sl={self.donchian_sl}"
                )
        if self.fee_rate < 0:
            errors.append(f"fee_rate must be non-negative, got {self.fee_rate}")
        if self.slippage_mode not in ("flat", "tiered", "conservative"):
            errors.append(f"slippage_mode must be 'flat', 'tiered', or 'conservative', got {self.slippage_mode!r}")
        if self.max_exposure_multiplier < self.base_leverage:
            errors.append(
                f"max_exposure_multiplier must be >= base_leverage, "
                f"got max_exposure_multiplier={self.max_exposure_multiplier}, "
                f"base_leverage={self.base_leverage}"
            )

        if errors:
            raise ValueError("VolStraddleConfig safety constraint violations:\n  - " + "\n  - ".join(errors))
        return self

    @field_validator("d_symbol_blacklist", mode="before")
    @classmethod
    def _parse_d_symbol_blacklist(cls, v):
        """Parse VS_D_SYMBOL_BLACKLIST as comma-separated string, JSON array,
        or pass-through for list input.

        NoDecode annotation on the field causes pydantic-settings to pass the
        raw env string here without JSON-decoding; we handle both formats.
        """
        if v is None:
            return []
        if isinstance(v, list):
            return [str(s).strip() for s in v if str(s).strip()]
        if isinstance(v, str):
            s = v.strip().strip('"').strip("'")
            if not s:
                return []
            if s.startswith("["):
                import json

                return json.loads(s)
            return [part.strip() for part in s.split(",") if part.strip()]
        return v

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

    def is_trail_enabled(self, strategy_type: str) -> bool:
        """해당 전략이 서버측 트레일링 스탑을 쓰는지.

        Args:
            strategy_type: 정규화된 전략 타입.

        Returns:
            트레일을 걸어야 하면 True. 명시적 플래그가 없는 전략은 기존 동작(True)을
            유지한다 — 이 스위치는 측정으로 근거가 확인된 전략에만 적용한다.
        """
        if strategy_type == MA_CLUSTER:
            return self.ma_cluster_trail_enabled
        return True

    def get_trail_params(self, strategy_type: str) -> tuple[float, float]:
        """Return (trail_activate, trail_distance) for a strategy type."""
        _map = {
            VR_DIRECTIONAL: (self.d_trail_activate, self.d_trail_distance),
            MA_CLUSTER: (self.ma_cluster_trail_activate, self.ma_cluster_trail_distance),
            DONCHIAN: (self.donchian_trail_activate, self.donchian_trail_distance),
        }
        if strategy_type not in _map:
            raise ValueError(f"Unknown strategy_type for trail params: {strategy_type!r}")
        return _map[strategy_type]

    def get_trail_distance(self, strategy_type: str) -> float:
        """Return trail_distance for a strategy type (ratchet SL use)."""
        return self.get_trail_params(strategy_type)[1]

    def get_warmup(self, tf: str) -> int:
        """Get warmup bars for a timeframe."""
        return {
            "15m": self.warmup_15m,
            "30m": self.warmup_30m,
            "1h": self.warmup_1h,
            "4h": self.warmup_4h,
        }.get(tf, self.warmup_15m)
