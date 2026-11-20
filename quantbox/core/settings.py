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

from pydantic import model_validator
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
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env에 코드에서 안 쓰는 변수가 있어도 무시
    )

    # ── 필수: 바이낸스 API ──
    binance_api_key: str
    binance_api_secret: str = ""  # model_validator에서 채움
    binance_secret: str = ""  # 호환 별칭
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

    # ── Regime Protection ──
    z_score_cap: float = 4.0  # |z| > cap 이면 진입 거부
    vol_expansion_gate: float = 2.0  # recent_std/formation_std > gate 이면 진입 거부

    # ── Trailing Stop (ratchet stop) ──
    trailing_stop_enabled: bool = True
    trailing_stop_activate_delta: float = 0.3
    trailing_stop_trail_z: float = 1.2

    # ── Direction Diversity ──
    max_same_direction_slots: int = 2

    # ── Alpha Filters ──
    zscore_velocity_enabled: bool = False
    zscore_velocity_lookback: int = 4
    spread_autocorr_enabled: bool = False
    spread_autocorr_lookback: int = 40
    spread_autocorr_max: float = 0.0

    # ── 드로다운 제한 ──
    max_drawdown_pct: float = 0.10

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

    # ── 상태 파일 백업 ──
    state_remote_backup_dir: str | None = None  # 원격/2차 백업 경로 (e.g. /mnt/nas/quantbox)
    state_max_log_age_days: int = 30  # 로그 로테이션: N일 이상 된 로그 파일 삭제

    @model_validator(mode="after")
    def _resolve_api_secret(self) -> "QuantBoxSettings":
        """BINANCE_API_SECRET 또는 BINANCE_SECRET 중 하나만 있으면 OK.

        둘 다 있으면 BINANCE_API_SECRET 우선.
        둘 다 없으면 ValidationError.
        """
        if not self.binance_api_secret and self.binance_secret:
            self.binance_api_secret = self.binance_secret
        if not self.binance_api_secret:
            raise ValueError("BINANCE_API_SECRET (또는 BINANCE_SECRET) 환경변수가 필요합니다. .env 파일을 확인하세요.")
        # fail-closed: .env.example 의 자리표시자를 그대로 둔 채 실거래 경로가 시작되지 않게 한다.
        for name, value in (("BINANCE_API_KEY", self.binance_api_key), ("BINANCE_API_SECRET", self.binance_api_secret)):
            if value.strip().lower().startswith("your_"):
                raise ValueError(f"{name} 가 자리표시자 값입니다. 실제 값을 환경변수/.env 에 설정하세요.")
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
            "quote_currency": "USDT",
            "n_coins": self.n_coins,
            "interval": "15m",
            "formation_hours": self.formation_hours,
            # Strategy
            "entry_zscore": self.entry_zscore,
            "exit_zscore": self.exit_zscore,
            "stop_loss_zscore": self.stop_loss_zscore,
            "use_dynamic_exit": self.use_dynamic_exit,
            "dynamic_exit_base": self.dynamic_exit_base,
            "dynamic_exit_range": self.dynamic_exit_range,
            "max_holding_hours": self.max_holding_hours,
            "half_life_max_hold_multiplier": self.half_life_max_hold_multiplier,
            "exit_zscore_decay": self.exit_zscore_decay,
            "stale_trade_bars": self.stale_trade_bars,
            "stale_min_improvement": self.stale_min_improvement,
            "live_coint_check_interval": self.live_coint_check_interval,
            "live_coint_check_window": self.live_coint_check_window,
            "live_coint_check_max_pvalue": self.live_coint_check_max_pvalue,
            # Risk
            "position_size_pct": pct,
            "trading_fee": self.trading_fee,
            "slippage": self.slippage,
            "funding_rate_8h": self.funding_rate_8h,
            "leverage": self.leverage,
            "max_leverage": self.max_leverage,
            "target_r_pct": self.target_r_pct,
            # Multi-trader
            "max_traders": self.max_traders,
            # Filters
            "min_correlation": 0.5,
            "max_cointegration_pvalue": 0.05,
            "min_half_life": 3,
            "max_half_life": 60,
            "max_hurst": 0.5,
            # Execution
            "check_interval_seconds": 60,
            "retry_attempts": 3,
            "retry_delay": 2,
            # Regime Protection
            "z_score_cap": self.z_score_cap,
            "vol_expansion_gate": self.vol_expansion_gate,
            # Trailing Stop
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "trailing_stop_activate_delta": self.trailing_stop_activate_delta,
            "trailing_stop_trail_z": self.trailing_stop_trail_z,
            # Direction Diversity
            "max_same_direction_slots": self.max_same_direction_slots,
            # Alpha Filters
            "zscore_velocity_enabled": self.zscore_velocity_enabled,
            "zscore_velocity_lookback": self.zscore_velocity_lookback,
            "spread_autocorr_enabled": self.spread_autocorr_enabled,
            "spread_autocorr_lookback": self.spread_autocorr_lookback,
            "spread_autocorr_max": self.spread_autocorr_max,
            # Drawdown Halt
            "max_drawdown_pct": self.max_drawdown_pct,
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
