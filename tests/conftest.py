"""
Test fixtures and shared utilities for tests.

IMPORTANT: All test data comes from REAL Binance Futures data cached locally.
NO synthetic/fake data generation is allowed.

## Shared Scalper Fixtures
Avoid reimplementing scalper setup in every test file — use these fixtures instead.

  vs_scalper   — VolStraddle scalper, paper=True, mocked client, balance=$1000
  rb_scalper   — RubberBand scalper, paper=True, mocked client, balance=$1000
  vs_position  — Standard open position dict (hedge mode LONG, TESTUSDT:LONG key)

## Key signatures (frequently misused)

  _open_position(symbol, signal, closes, balance) → bool
  _partial_close_position(symbol, current_price) → bool          # 2 args only!
  _close_position(symbol, reason, current_price) → bool          # 3 args only!

## Compound key convention (hedge_mode=True)

  position key  : "TESTUSDT:LONG"   (used in _positions dict)
  api_symbol    : "TESTUSDT"        (used in Binance API calls, pos["symbol"])

## Code path map (what strategy reaches which lines in positions.py)

  L56-129  : vol_straddle sizing + bracket  ← vs_scalper reaches this
  L138-253 : rubberband sizing              ← rb_scalper reaches this
  L394-454 : vol_straddle bracket           ← vs_scalper reaches this
  L455-633 : rubberband bracket             ← rb_scalper reaches this
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """매 테스트마다 get_settings() 캐시를 초기화.

    @lru_cache 싱글턴이 테스트 간 환경변수 오염을 일으키는 것을 방지한다.
    """
    from quantbox.core.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_vol_straddle_env_file(monkeypatch):
    """VolStraddleConfig 가 저장소 루트의 운영 .env 를 읽지 않게 한다.

    두 경로로 새어 들어왔다 — trader .env 의 VS_MIN_CONVICTION_SCORE=5 가 신호를 걸러
    vol_straddle 테스트 4개가 그 호스트에서만 실패했다(2026-09-16).
      1. VolStraddleConfig 의 env_file=".env"
      2. quantbox.scalper 가 import 시점에 load_dotenv() 로 .env 전체를 os.environ 에 넣는다
         (_isolate_state_file 이 scalper 를 import 하므로 세션 내내 남는다)
    테스트가 monkeypatch.setenv 로 넣는 VS_* 는 이 fixture 뒤에 적용되므로 그대로 동작한다.
    """
    import os

    import dotenv
    import quantbox.scalper  # noqa: F401 — load_dotenv() 를 먼저 일으킨 뒤 지운다
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig

    monkeypatch.setitem(VolStraddleConfig.model_config, "env_file", None)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: {})
    for key in [k for k in os.environ if k.startswith("VS_") or k == "RB_SYMBOL_WHITELIST"]:
        monkeypatch.delenv(key)


@pytest.fixture(autouse=True)
def _isolate_state_file(tmp_path):
    """Redirect scalper state file to tmp_path so tests never corrupt
    the production state (data/state/scalper_state.json).

    This is CRITICAL — without this, any test calling _sync_server_orders()
    or _save_state() would overwrite the live bot's position state.
    """
    import quantbox.scalper as _mod
    import quantbox.core.scalper_state as _state_mod

    orig_file = _mod.STATE_FILE
    orig_state_file = _state_mod.STATE_FILE
    orig_state_dir = _state_mod.STATE_DIR
    _state_mod.STATE_DIR = tmp_path
    _state_mod.STATE_FILE = tmp_path / "scalper_state.json"
    _mod.STATE_FILE = tmp_path / "scalper_state.json"
    yield
    _mod.STATE_FILE = orig_file
    _state_mod.STATE_FILE = orig_state_file
    _state_mod.STATE_DIR = orig_state_dir


@pytest.fixture(autouse=True)
def _isolate_journal_db(monkeypatch):
    """Redirect all live journal writes to :memory: during tests.

    Without this, any test using paper_mode=False writes fake trades to
    data/journal.sqlite (the live trading DB), corrupting real trade history.

    Must patch quantbox.scalper.TradeJournal (where it was already imported),
    not the source module (that reference is stale after import).
    """
    import quantbox.core.lifecycle as _lifecycle_mod
    import quantbox.scalper as _scalper_mod
    from quantbox.core.trade_journal import TradeJournal as _OrigJournal

    class _MemJournal(_OrigJournal):
        def __init__(self, db_path: str = ":memory:", **kwargs):
            super().__init__(db_path=":memory:", **kwargs)

    monkeypatch.setattr(_scalper_mod, "TradeJournal", _MemJournal)
    monkeypatch.setattr(_lifecycle_mod, "TradeJournal", _MemJournal)


def _get_real_data_loader():
    """Get the real data loader instance."""
    from quantbox.data.real_data_loader import RealDataLoader

    return RealDataLoader()


@pytest.fixture(scope="session")
def real_data_available():
    """Check if real data is available in cache."""
    loader = _get_real_data_loader()
    symbols = loader.get_available_symbols()
    return len(symbols) >= 2


@pytest.fixture
def sample_ohlcv(real_data_available):
    """Load REAL OHLCV data from Binance cache."""
    if not real_data_available:
        pytest.skip("Real data not available - run data downloader first")

    loader = _get_real_data_loader()
    df = loader.load_symbol("BTCUSDT")

    if df is None:
        # Try any available symbol
        symbols = loader.get_available_symbols()
        if symbols:
            df = loader.load_symbol(symbols[0])

    if df is None:
        pytest.skip("No real data available - run data downloader first")

    return df


@pytest.fixture
def sample_pair_data(real_data_available):
    """Load REAL correlated pair data from Binance cache."""
    if not real_data_available:
        pytest.skip("Real data not available - run data downloader first")

    loader = _get_real_data_loader()
    data = loader.load_data(symbols=["BTCUSDT", "ETHUSDT"], min_bars=100, align_data=True)

    if len(data) < 2:
        pytest.skip("Insufficient real data - need BTCUSDT and ETHUSDT")

    return data


@pytest.fixture
def multi_symbol_data(real_data_available):
    """Load REAL multi-symbol data from Binance cache."""
    if not real_data_available:
        pytest.skip("Real data not available - run data downloader first")

    loader = _get_real_data_loader()
    data = loader.load_data(
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"], min_bars=100, align_data=True
    )

    if len(data) < 3:
        pytest.skip("Insufficient real data - need at least 3 symbols")

    return data


@pytest.fixture
def config():
    """Sample configuration."""
    from quantbox.core.config import Config

    return Config()


# ─────────────────────────────────────────────────────────────────────────────
# Shared scalper fixtures
# ─────────────────────────────────────────────────────────────────────────────

_SYMBOL_INFO = {
    "step_size": 0.001,
    "tick_size": 0.01,
    "quantity_precision": 3,
    "price_precision": 2,
    "min_notional": 5.0,
}


def _build_vs_scalper(balance: float = 1000.0, vs_config=None):
    """Factory — not a fixture so it can be called with custom params."""
    from quantbox.scalper import RubberBandScalper
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig

    if vs_config is None:
        vs_config = VolStraddleConfig()

    with (
        patch("quantbox.scalper.get_settings") as mock_settings,
        patch("quantbox.scalper.Client"),
        patch("quantbox.scalper.DiscordNotifier"),
        patch("quantbox.scalper.DiscordBotRunner", None),
    ):
        mock_settings.return_value = MagicMock(
            binance_api_key="k",
            binance_api_secret="s",
            binance_testnet=False,
            discord_webhook_url="",
            discord_bot_token="",
            discord_channel_id="",
            paper_balance=balance,
        )
        scalper = RubberBandScalper(
            paper_mode=True,
            strategy_name="vol_straddle",
            vs_config=vs_config,
        )

    for sym in ("TESTUSDT", "SHORTUSDT", "ETHUSDT"):
        scalper._symbol_info_cache[sym] = _SYMBOL_INFO.copy()

    scalper.client = MagicMock()
    scalper.discord = MagicMock()
    scalper._paper_balance = balance
    scalper._high_water_mark = 0.0
    scalper.get_balance = MagicMock(return_value=balance)
    scalper._save_state = MagicMock()
    return scalper


def _build_rb_scalper(balance: float = 1000.0, rb_cfg=None):
    """Factory for RubberBand scalper — DEPRECATED but kept for test compat."""
    from quantbox.scalper import RubberBandScalper
    from quantbox.strategy.rubberband.config import RubberBandConfig

    if rb_cfg is None:
        rb_cfg = RubberBandConfig(
            lookback=10,
            threshold=0.01,
            partial_exit_enabled=False,
            dynamic_leverage_enabled=False,
            vol_adjusted_size=False,
            steep_cap_curve=False,
            mtf_confirmation_enabled=False,
        )

    with (
        patch("quantbox.scalper.get_settings") as mock_settings,
        patch("quantbox.scalper.Client"),
        patch("quantbox.scalper.DiscordNotifier"),
        patch("quantbox.scalper.DiscordBotRunner", None),
    ):
        mock_settings.return_value = MagicMock(
            binance_api_key="k",
            binance_api_secret="s",
            binance_testnet=False,
            discord_webhook_url="",
            discord_bot_token="",
            discord_channel_id="",
            paper_balance=balance,
        )
        scalper = RubberBandScalper(
            paper_mode=True,
            strategy_name="rubberband",
            config=rb_cfg,
        )

    for sym in ("TESTUSDT", "SHORTUSDT"):
        scalper._symbol_info_cache[sym] = _SYMBOL_INFO.copy()

    scalper.client = MagicMock()
    scalper.discord = MagicMock()
    scalper._paper_balance = balance
    scalper._high_water_mark = 0.0
    scalper.get_balance = MagicMock(return_value=balance)
    scalper._save_state = MagicMock()
    return scalper


@pytest.fixture
def vs_scalper():
    """VolStraddle scalper — paper mode, mocked client, balance=$1000.

    Positions stored under compound keys: "TESTUSDT:LONG", "TESTUSDT:SHORT"
    (hedge_mode=True by default).

    Function signatures for this scalper:
        _open_position(symbol, signal, closes, balance) → bool
        _partial_close_position(symbol, current_price) → bool
        _close_position(symbol, reason, current_price) → bool
    """
    return _build_vs_scalper()


@pytest.fixture
def rb_scalper():
    """RubberBand scalper — paper mode, mocked client, balance=$1000.

    Uses rubberband code paths (L138-253 sizing, L455-633 bracket).
    Positions stored under plain keys: "TESTUSDT" (hedge_mode=False by default).
    """
    return _build_rb_scalper()


@pytest.fixture
def vs_position() -> dict:
    """Standard open position dict for TESTUSDT:LONG (hedge mode).

    Use as:
        scalper.strategy._positions["TESTUSDT:LONG"] = vs_position

    The "symbol" field is the pure Binance API symbol (no colon).
    """
    return {
        "symbol": "TESTUSDT",  # pure Binance symbol — NO colon!
        "signal": 1,
        "entry_price": 100.0,
        "notional": 3000.0,
        "bars_held": 10,
        "peak_gain": 0.02,
        "best_pnl": 0.02,
        "worst_pnl": -0.005,
        "server_sl_order_id": "sl_111",
        "server_tp_order_id": "tp_222",
        "sl": 0.01,
        "tp": 0.12,
        "tf": "15m",
        "position_side": "LONG",
        "trail_active": False,
        "opened_at": "2026-01-01T00:00:00+00:00",
        "strategy_type": "vr_directional",
    }
