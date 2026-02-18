"""
QuantBox 환경변수 설정 — pydantic-settings 기반 검증.

앱 시작 시 .env 파일을 읽어 모든 환경변수를 검증한다.
필수 변수가 없거나 타입이 틀리면 즉시 ValidationError를 발생시켜
런타임에 이상한 값으로 동작하는 것을 원천 차단한다.

사용법:
    from quantbox.core.settings import get_settings
    settings = get_settings()
    print(settings.binance_api_key)

환경변수 이름 규칙:
    - 모든 필드는 대문자 환경변수와 매핑 (e.g. binance_api_key → BINANCE_API_KEY)
    - BINANCE_SECRET은 BINANCE_API_SECRET의 호환 별칭
"""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class QuantBoxSettings(BaseSettings):
    """QuantBox 환경변수 설정.

    .env 파일에서 자동으로 값을 읽는다.
    필수 필드에 값이 없으면 앱 시작 시 ValidationError 발생.

    Attributes:
        binance_api_key: 바이낸스 API 키 (필수).
        binance_api_secret: 바이낸스 API 시크릿 (필수).
        binance_secret: BINANCE_SECRET 호환 별칭 (deprecated).
        binance_testnet: 테스트넷 사용 여부.
        leverage: 레버리지 배수 (동적 레버리지 미사용 시 fallback).
        max_leverage: 동적 레버리지 안전 상한.
        target_r_pct: 트레이드당 목표 리스크 (밸런스의 비율, 0.01=1%).
        max_traders: 최대 동시 트레이더 수.
        position_size_pct: 슬롯당 자본 비율 (-1=자동).
        entry_zscore: 진입 z-score 임계값.
        exit_zscore: 청산 z-score 임계값.
        stop_loss_zscore: 손절 z-score.
        trading_fee: 거래 수수료.
        slippage: 슬리피지.
        funding_rate_8h: 8시간 펀딩비.
        n_coins: 유니버스 코인 수.
        paper_balance: 페이퍼 트레이딩 가상 잔고.
        discord_webhook_url: Discord 웹훅 URL.
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',  # .env에 코드에서 안 쓰는 변수가 있어도 무시
    )

    # ── 필수: 바이낸스 API ──
    binance_api_key: str
    binance_api_secret: str = ''    # model_validator에서 채움
    binance_secret: str = ''        # 호환 별칭
    binance_testnet: bool = False

    # ── 레버리지 및 포지션 사이징 ──
    leverage: int = 2  # max_leverage의 fallback (동적 레버리지 미사용 시)
    max_leverage: int = 125  # 동적 레버리지 안전 상한 (바이난스 절대 상한)
    target_r_pct: float = 0.02  # 트레이드당 목표 리스크 (밸런스의 2%)
    max_traders: int = 10
    position_size_pct: float = -1.0  # -1 = auto (1/max_traders)

    # ── 전략 파라미터 ──
    entry_zscore: float = 2.2
    exit_zscore: float = 0.5
    stop_loss_zscore: float = 1.5  # entry에서 +1.5 이동 시 손절

    # ── 홀딩 제한 ──
    max_holding_hours: int = 20
    half_life_max_hold_multiplier: float = 0.0  # 0=disabled, e.g. 6 → max_hold=6×half_life
    exit_zscore_decay: float = 0.0  # 0=disabled, e.g. 0.5 → exit_z halves by max_hold time
    stale_trade_bars: int = 0  # 0=disabled, e.g. 40 → check at 40 bars if trade stale
    stale_min_improvement: float = 0.3  # min z improvement to keep position
    # ── Live cointegration check ──
    live_coint_check_interval: int = 0  # 0=disabled, bars between checks (e.g. 30)
    live_coint_check_window: int = 120  # bars of recent data for re-test
    live_coint_check_max_pvalue: float = 0.10  # p-value threshold (looser than entry)
    # ── Formation window ──
    formation_hours: int = 60  # Formation period in hours (default: 60h = 240 bars @ 15m)
    # ── Dynamic exit ──
    use_dynamic_exit: bool = True
    dynamic_exit_base: float = 0.1  # 빠른 페어 exit z-score (half_life=min)
    dynamic_exit_range: float = 0.9  # max exit = base + range = 1.0

    # ── Rubber Band Signal Parameters (v9.0, Fibonacci discipline) ──
    rb_lookback: int = 8               # Bars to measure move (2h at 15m) — SACRED
    rb_threshold: float = 0.032        # v10.0: Lower threshold (3.2%, +162% ret vs 4.5%)
    rb_target_retrace: float = 0.382   # Fibonacci 0.382 target (v9.0, +29% Calmar vs 0.4)
    rb_stop_mult: float = 0.786        # Fibonacci 0.786 stop (v9.0, DD -9.5% vs 0.9)
    rb_max_hold: int = 20              # v10.0: Extended hold (20 bars, +24% ret vs 10)
    rb_stop_decay: float = 0.0         # Stop tightening disabled for robustness

    # ── Multi-timeframe Confirmation (DISABLED) ──
    rb_htf_period: int = 0             # HTF trend lookback (0=DISABLED)
    rb_htf_block_threshold: float = 0.01  # Block entry if HTF trend > 1% against dir
    rb_htf_agree_discount: float = 0.85   # Threshold discount when HTF agrees
    rb_htf_agree_stop_mult: float = 0.95  # Stop multiplier when HTF agrees

    # ── Dynamic Leverage (v4.4: flat 1.5x, 365-day A/B validated) ──
    rb_dynamic_leverage_enabled: bool = True  # Enable leverage pipeline
    rb_base_leverage: float = 1.5             # Flat 1.5x leverage (sweet spot)
    rb_leverage_scale: float = 0.0            # No confidence scaling (flat leverage)
    rb_max_leverage: float = 1.5              # Aligned with base (safety cap)

    # ── Regime-Switch Dual Entry (v5.0) + Confidence Scoring (v6.0) ──
    rb_entry_mode: str = 'confidence'         # Entry mode: 'confidence' | 'regime_switch' | 'btc_directional' | 'raw'
    rb_btc_regime_period: int = 96            # BTC SMA period for regime detection (96봉 = 24시간)
    rb_asset_trend_period: int = 192          # Asset SMA period for classic pullback (192봉 = 48시간)

    # ── Confidence Scoring (v10.0, MTF acts as quality gate) ──
    rb_min_confidence: int = 1                # v10.0: 최소 1 (MTF가 진짜 필터, +86.9% ret)
    rb_confidence_leverage_step: float = 0.0  # 컨피던스당 추가 레버리지 (0=고정)
    rb_rsi_period: int = 14                   # RSI 계산 기간
    rb_vol_period: int = 20                   # 거래량 SMA 기간 (스파이크 감지)

    # ── v7.0 Multi-Timeframe Confirmation (365-day A/B validated) ──
    # Uses 4h candle data to confirm 15m entries. Blocks counter-trend trades.
    # MDD: 16.5% → 4.3% (-74%), Calmar: 8.65 → 32.42 (+275%)
    rb_mtf_confirmation_enabled: bool = True  # 4h 트렌드 확인 활성화
    rb_mtf_sma_period: int = 12               # 4h SMA 기간 (12봉 = 2일)
    rb_mtf_leverage_mult: float = 2.5         # v10.1: MTF 확인 후 레버리지 배수 (2.0→2.5, +35% ret)

    # ── v8.0 Dynamic Target (DISABLED in v9.0 — Fib discipline outperforms) ──
    # Fib 0.382 fixed target > dynamic scaling. Tighter stop (0.786) requires faster TP.
    rb_dynamic_target_enabled: bool = False   # 비활성화: Fib 0.382 고정 > 동적 스케일링
    rb_target_conf3: float = 0.45             # confidence=3 목표 되돌림 (사용 안 함)
    rb_target_conf4: float = 0.50             # confidence=4 목표 되돌림 (사용 안 함)
    rb_target_conf5: float = 0.55             # confidence≥5 목표 되돌림 (사용 안 함)

    # ── v9.0 S+R+M Confluence Pipeline (A/B validated, winning strategy) ──
    # MACD 히스토그램 반전 + 스토캐스틱 %K/%D 크로스 = 추가 컨피던스 증거
    # Combined with Fib exits: Calmar +33.6%, DD -8.5%
    rb_macd_enabled: bool = True              # MACD 히스토그램 기울기 컨피던스 증거
    rb_macd_fast: int = 12                    # MACD fast EMA 기간
    rb_macd_slow: int = 26                    # MACD slow EMA 기간
    rb_macd_signal_period: int = 14           # MACD signal EMA 기간 (14=최적)
    rb_stoch_enabled: bool = True             # 스토캐스틱 크로스 컨피던스 증거
    rb_stoch_k_period: int = 3                # v10.0: 더 빠른 스토캐스틱 (+5% Calmar vs 5)
    rb_stoch_d_period: int = 3                # 스토캐스틱 %D 기간

    # ── v11.4 Adaptive Threshold (Walk-Forward Validated) ──
    # Per-bar threshold = clip(atr_multiplier × ATR%(atr_period), floor, cap)
    # Chosen config: 2.5x_f1 — OOS Sharpe 8.23, OOS/IS 1.09, zero overfitting.
    rb_atr_period: int = 48                   # ATR EMA 기간 (48 × 15분 = 12시간)
    rb_atr_multiplier: float = 2.5            # ATR 배수
    rb_threshold_floor: float = 0.01          # 최소 임계값 (1%)
    rb_threshold_cap: float = 0.06            # 최대 임계값 (6%)

    # ── v7.0 Independent 4H Rubber Band ──
    # Trades mean-reversion on 4h timeframe independently from 15m.
    # 4h-only: Ret +52.2%, DD 4.8%, Calmar 10.87
    rb_htf_independent_enabled: bool = False  # v10.2: Disabled (dead code in production)
    rb_htf_rb_lookback: int = 3               # 4h lookback (12시간)
    rb_htf_rb_threshold: float = 0.10         # 10% 임계값 (4h는 더 높은 노이즈)
    rb_htf_rb_target_retrace: float = 0.5     # 50% 되돌림 목표
    rb_htf_rb_stop_mult: float = 0.8          # 0.8x 손절
    rb_htf_rb_max_hold: int = 12              # 12봉 (2일)
    rb_htf_rb_btc_sma_period: int = 48        # BTC SMA (8일)

    # ── BTC Directional Alignment (v4.2, A/B validated) ──
    rb_btc_directional_enabled: bool = True   # BTC 방향 정렬 활성화
    rb_btc_directional_period: int = 48       # v11.0: 12h BTC momentum (A/B validated)

    # ── Capital Preservation (v4.4: DISABLED — hurts with tighter stop) ──
    rb_partial_exit_enabled: bool = False     # Disabled: Calmar 3.75 vs 4.96 without
    rb_partial_exit_fraction: float = 0.5     # 부분 청산 비율 (0.5 = 50%)

    # ── R:R-Gated Leverage (v4.3, A/B validated) ──
    rb_rr_leverage_enabled: bool = True       # R:R 기준 레버리지 게이트 활성화
    rb_min_rr_for_leverage: float = 0.30      # 레버리지 허용 최소 R:R 비율

    # ── v10.0 Symbol Whitelist & Dynamic Capital ──
    rb_symbol_whitelist: str = 'FILUSDT,1000PEPEUSDT,XRPUSDT,XLMUSDT,ADAUSDT,FETUSDT,ETHUSDT,ARBUSDT,1000SHIBUSDT,LINKUSDT,ATOMUSDT,OPUSDT,DOTUSDT,NEARUSDT,AAVEUSDT,BCHUSDT,HBARUSDT,SOLUSDT,ICPUSDT,BNBUSDT'
    rb_dynamic_capital_enabled: bool = True  # balance / max(1, n_active+1) 동적 자본 배분

    # ── Regime Protection ──
    z_score_cap: float = 4.0           # |z| > cap 이면 진입 거부
    vol_expansion_gate: float = 2.0    # recent_std/formation_std > gate 이면 진입 거부

    # ── Trailing Stop (ratchet stop) ──
    trailing_stop_enabled: bool = True      # 트레일링 스톱 활성화
    trailing_stop_activate_delta: float = 0.3  # z가 entry 대비 0.3 개선 시 활성화
    trailing_stop_trail_z: float = 1.2      # best_z에서 trail_z만큼 되돌림 시 발동

    # ── Direction Diversity (방향 다양화) ──
    max_same_direction_slots: int = 2  # 동일 방향(Long/Short) 최대 슬롯 수

    # ── Alpha Filters (알파 필터) ──
    zscore_velocity_enabled: bool = False      # Z-score 속도 확인: mean 방향 이동 시만 진입
    zscore_velocity_lookback: int = 4          # Z 속도 측정용 lookback bars (15분봉 4개 = 1시간)
    spread_autocorr_enabled: bool = False      # 스프레드 자기상관 확인: AC1 < 0 시만 진입
    spread_autocorr_lookback: int = 40         # 자기상관 측정 lookback bars
    spread_autocorr_max: float = 0.0           # 최대 허용 lag-1 자기상관 (< 0 = 평균회귀)

    # ── 드로다운 제한 ──
    max_drawdown_pct: float = 0.10         # 세션 최고 잔고 대비 10% 드로다운 시 전체 청산+일시정지

    # ── 비용 모델 ──
    trading_fee: float = 0.0005
    slippage: float = 0.001
    funding_rate_8h: float = 0.0001

    # ── 유니버스 & 페이퍼 ──
    n_coins: int = 50
    paper_balance: float = 1000.0

    # ── 알림 ──
    discord_webhook_url: str | None = None
    discord_bot_token: str | None = None
    discord_channel_id: int | None = None

    @model_validator(mode='after')
    def _resolve_api_secret(self) -> 'QuantBoxSettings':
        """BINANCE_API_SECRET 또는 BINANCE_SECRET 중 하나만 있으면 OK.

        둘 다 있으면 BINANCE_API_SECRET 우선.
        둘 다 없으면 ValidationError.
        """
        if not self.binance_api_secret and self.binance_secret:
            self.binance_api_secret = self.binance_secret
        if not self.binance_api_secret:
            raise ValueError(
                'BINANCE_API_SECRET (또는 BINANCE_SECRET) 환경변수가 필요합니다. '
                '.env 파일을 확인하세요.'
            )
        return self

    def to_trader_config(self) -> dict:
        """live_trader._default_config() 형태의 dict로 변환.

        기존 코드와의 호환성을 위해 dict를 반환한다.

        Returns:
            트레이더 설정 dict.
        """
        pct = self.position_size_pct
        if pct < 0:
            pct = 1.0 / self.max_traders

        return {
            # Data settings
            'quote_currency': 'USDT',
            'n_coins': self.n_coins,
            'interval': '15m',
            'formation_hours': self.formation_hours,
            # Strategy
            'entry_zscore': self.entry_zscore,
            'exit_zscore': self.exit_zscore,
            'stop_loss_zscore': self.stop_loss_zscore,
            'use_dynamic_exit': self.use_dynamic_exit,
            'dynamic_exit_base': self.dynamic_exit_base,
            'dynamic_exit_range': self.dynamic_exit_range,
            'max_holding_hours': self.max_holding_hours,
            'half_life_max_hold_multiplier': self.half_life_max_hold_multiplier,
            'exit_zscore_decay': self.exit_zscore_decay,
            'stale_trade_bars': self.stale_trade_bars,
            'stale_min_improvement': self.stale_min_improvement,
            'live_coint_check_interval': self.live_coint_check_interval,
            'live_coint_check_window': self.live_coint_check_window,
            'live_coint_check_max_pvalue': self.live_coint_check_max_pvalue,
            # Risk
            'position_size_pct': pct,
            'trading_fee': self.trading_fee,
            'slippage': self.slippage,
            'funding_rate_8h': self.funding_rate_8h,
            'leverage': self.leverage,
            'max_leverage': self.max_leverage,
            'target_r_pct': self.target_r_pct,
            # Multi-trader
            'max_traders': self.max_traders,
            # Filters
            'min_correlation': 0.5,
            'max_cointegration_pvalue': 0.05,
            'min_half_life': 3,
            'max_half_life': 60,
            'max_hurst': 0.5,
            # Execution
            'check_interval_seconds': 60,
            'retry_attempts': 3,
            'retry_delay': 2,
            # Regime Protection
            'z_score_cap': self.z_score_cap,
            'vol_expansion_gate': self.vol_expansion_gate,
            # Trailing Stop
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'trailing_stop_activate_delta': self.trailing_stop_activate_delta,
            'trailing_stop_trail_z': self.trailing_stop_trail_z,
            # Direction Diversity
            'max_same_direction_slots': self.max_same_direction_slots,
            # Regime-Switch Dual Entry (v5.0) + Confidence (v6.0)
            'entry_mode': self.rb_entry_mode,
            'btc_regime_period': self.rb_btc_regime_period,
            'asset_trend_period': self.rb_asset_trend_period,
            'min_confidence': self.rb_min_confidence,
            'confidence_leverage_step': self.rb_confidence_leverage_step,
            'rsi_period': self.rb_rsi_period,
            'vol_period': self.rb_vol_period,
            # v7.0 Multi-Timeframe Confirmation
            'mtf_confirmation_enabled': self.rb_mtf_confirmation_enabled,
            'mtf_sma_period': self.rb_mtf_sma_period,
            'mtf_leverage_mult': self.rb_mtf_leverage_mult,
            # v8.0 Dynamic Target
            'dynamic_target_enabled': self.rb_dynamic_target_enabled,
            'target_conf3': self.rb_target_conf3,
            'target_conf4': self.rb_target_conf4,
            'target_conf5': self.rb_target_conf5,
            # v9.0 S+R+M Confluence
            'macd_enabled': self.rb_macd_enabled,
            'macd_fast': self.rb_macd_fast,
            'macd_slow': self.rb_macd_slow,
            'macd_signal_period': self.rb_macd_signal_period,
            'stoch_enabled': self.rb_stoch_enabled,
            'stoch_k_period': self.rb_stoch_k_period,
            'stoch_d_period': self.rb_stoch_d_period,
            # v7.0 Independent 4H Rubber Band
            'htf_independent_enabled': self.rb_htf_independent_enabled,
            'htf_rb_lookback': self.rb_htf_rb_lookback,
            'htf_rb_threshold': self.rb_htf_rb_threshold,
            'htf_rb_target_retrace': self.rb_htf_rb_target_retrace,
            'htf_rb_stop_mult': self.rb_htf_rb_stop_mult,
            'htf_rb_max_hold': self.rb_htf_rb_max_hold,
            'htf_rb_btc_sma_period': self.rb_htf_rb_btc_sma_period,
            # BTC Directional Alignment
            'btc_directional_enabled': self.rb_btc_directional_enabled,
            'btc_directional_period': self.rb_btc_directional_period,
            # Alpha Filters
            'zscore_velocity_enabled': self.zscore_velocity_enabled,
            'zscore_velocity_lookback': self.zscore_velocity_lookback,
            'spread_autocorr_enabled': self.spread_autocorr_enabled,
            'spread_autocorr_lookback': self.spread_autocorr_lookback,
            'spread_autocorr_max': self.spread_autocorr_max,
            # Drawdown Halt
            'max_drawdown_pct': self.max_drawdown_pct,
        }


@lru_cache
def get_settings() -> QuantBoxSettings:
    """싱글턴 설정 인스턴스 반환.

    처음 호출 시 .env를 읽고 검증한다.
    검증 실패 시 ValidationError가 발생하여 앱이 즉시 종료된다.

    Returns:
        검증된 설정 객체.
    """
    return QuantBoxSettings()
