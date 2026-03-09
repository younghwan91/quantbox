"""
Tests for LimitOrderChaser — maker-only 체결 로직.

Mock을 사용하여 바이낸스 API 호출 없이 순수 로직만 검증한다.
"""

import time
from unittest.mock import MagicMock, patch, call

import pytest

from quantbox.core.limit_chaser import (
    ChaseConfig,
    ChaseFillStatus,
    ChaseResult,
    LimitOrderChaser,
)


# ─────────────────────────── Fixtures ──────────────────────────────


@pytest.fixture
def mock_client() -> MagicMock:
    """모의 바이낸스 futures 클라이언트."""
    client = MagicMock()
    # 기본 호가: bid=50000, ask=50001
    client.futures_orderbook_ticker.return_value = {
        "bidPrice": "50000.0",
        "askPrice": "50001.0",
    }
    return client


@pytest.fixture
def default_config() -> ChaseConfig:
    return ChaseConfig(
        max_chase_ticks=3,
        chase_interval_seconds=0.01,  # 테스트에서 빠르게
        timeout_seconds=5.0,
        fallback_to_market=True,
        initial_offset_ticks=0,
    )


@pytest.fixture
def chaser(mock_client: MagicMock, default_config: ChaseConfig) -> LimitOrderChaser:
    return LimitOrderChaser(
        client=mock_client,
        config=default_config,
        paper_mode=False,
    )


@pytest.fixture
def paper_chaser(mock_client: MagicMock, default_config: ChaseConfig) -> LimitOrderChaser:
    return LimitOrderChaser(
        client=mock_client,
        config=default_config,
        paper_mode=True,
    )


# ─────────────────────── Paper Mode Tests ──────────────────────────


class TestPaperMode:
    """Paper mode에서는 실제 주문 없이 즉시 체결."""

    def test_paper_buy_fills_at_bid(self, paper_chaser: LimitOrderChaser):
        """Paper BUY는 best bid 가격에 즉시 체결."""
        result = paper_chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)

        assert result.status == ChaseFillStatus.FILLED
        assert result.filled_quantity == 0.001
        assert result.avg_price == 50000.0
        assert result.maker_fill is True
        assert result.chase_count == 0
        assert result.order_id.startswith("paper_")

    def test_paper_sell_fills_at_ask(self, paper_chaser: LimitOrderChaser):
        """Paper SELL은 best ask 가격에 즉시 체결."""
        result = paper_chaser.chase_limit_order("ETHUSDT", "SELL", 0.5, tick_size=0.01, price_precision=2)

        assert result.status == ChaseFillStatus.FILLED
        assert result.filled_quantity == 0.5
        assert result.avg_price == 50001.0
        assert result.maker_fill is True

    def test_paper_mode_no_exchange_calls(self, paper_chaser: LimitOrderChaser, mock_client: MagicMock):
        """Paper mode에서는 주문/취소 API 호출이 없어야 함."""
        paper_chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)

        mock_client.futures_create_order.assert_not_called()
        mock_client.futures_cancel_order.assert_not_called()
        mock_client.futures_get_order.assert_not_called()
        # 호가 조회만 호출됨
        mock_client.futures_orderbook_ticker.assert_called_once()


# ─────────────────── Immediate Fill Tests ──────────────────────────


class TestImmediateFill:
    """첫 limit 주문이 바로 체결되는 케이스."""

    def test_buy_filled_on_first_try(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """BUY 주문이 첫 시도에 바로 FILLED."""
        mock_client.futures_create_order.return_value = {
            "orderId": 12345,
            "status": "NEW",
        }
        mock_client.futures_get_order.return_value = {
            "status": "FILLED",
            "executedQty": "0.001",
            "avgPrice": "50000.0",
        }

        result = chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)

        assert result.status == ChaseFillStatus.FILLED
        assert result.filled_quantity == 0.001
        assert result.avg_price == 50000.0
        assert result.maker_fill is True
        assert result.chase_count == 0

        # Limit order가 best bid(50000)에 발주됨
        create_call = mock_client.futures_create_order.call_args
        assert create_call.kwargs["type"] == "LIMIT"
        assert create_call.kwargs["price"] == "50000.0"
        assert create_call.kwargs["side"] == "BUY"

    def test_sell_filled_on_first_try(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """SELL 주문이 첫 시도에 바로 FILLED."""
        mock_client.futures_create_order.return_value = {
            "orderId": 12346,
            "status": "NEW",
        }
        mock_client.futures_get_order.return_value = {
            "status": "FILLED",
            "executedQty": "0.5",
            "avgPrice": "50001.0",
        }

        result = chaser.chase_limit_order("ETHUSDT", "SELL", 0.5, tick_size=0.01, price_precision=2)

        assert result.status == ChaseFillStatus.FILLED
        assert result.filled_quantity == 0.5
        assert result.avg_price == 50001.0
        assert result.maker_fill is True

        # Limit order가 best ask(50001)에 발주됨
        create_call = mock_client.futures_create_order.call_args
        assert create_call.kwargs["price"] == "50001.0"


# ─────────────────── Chase / Retry Tests ──────────────────────────


class TestChaseRetry:
    """미체결 시 가격 추격 로직."""

    def test_chase_fills_on_second_try(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """첫 번째 미체결 → 추격 → 두 번째에 체결."""
        mock_client.futures_create_order.return_value = {
            "orderId": 100,
            "status": "NEW",
        }

        # 1차: 미체결(NEW) → 2차: 체결(FILLED)
        mock_client.futures_get_order.side_effect = [
            {  # 1차 조회: 미체결
                "status": "NEW",
                "executedQty": "0",
                "avgPrice": "0",
            },
            {  # 2차 조회: 체결됨
                "status": "FILLED",
                "executedQty": "0.001",
                "avgPrice": "50000.5",
            },
        ]

        # 가격이 올라가는 시뮬레이션 (BUY 추격)
        mock_client.futures_orderbook_ticker.side_effect = [
            {"bidPrice": "50000.0", "askPrice": "50001.0"},  # 초기
            {"bidPrice": "50000.5", "askPrice": "50001.5"},  # 추격 시
        ]

        result = chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)

        assert result.status == ChaseFillStatus.FILLED
        assert result.chase_count == 1
        assert result.maker_fill is True
        # 주문 2회, 취소 1회
        assert mock_client.futures_create_order.call_count == 2
        assert mock_client.futures_cancel_order.call_count == 1

    def test_chase_respects_max_ticks(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """max_chase_ticks 초과 시 추격 중단 → 시장가 fallback."""
        # limit 주문 4회 (chase 0~3) + market fallback 1회
        mock_client.futures_create_order.side_effect = [
            {"orderId": 200, "status": "NEW"},  # limit chase 0
            {"orderId": 201, "status": "NEW"},  # limit chase 1
            {"orderId": 202, "status": "NEW"},  # limit chase 2
            {"orderId": 203, "status": "NEW"},  # limit chase 3
            # market fallback: 정상 체결
            {
                "orderId": 204,
                "executedQty": "0.001",
                "avgPrice": "50005.0",
                "status": "FILLED",
            },
        ]
        # limit 주문은 항상 미체결
        mock_client.futures_get_order.return_value = {
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }

        result = chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)

        # max_chase_ticks=3이면: 초기 1회 + 추격 3회 = 4 iterations
        assert result.status == ChaseFillStatus.FALLBACK_MARKET
        assert result.chase_count >= chaser.config.max_chase_ticks


# ─────────────────── Fallback to Market Tests ──────────────────────


class TestFallbackToMarket:
    """추격 실패 시 시장가 전환."""

    def test_fallback_market_on_chase_exhaustion(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """추격 횟수 초과 시 시장가로 fallback."""
        # Limit orders all stay NEW
        mock_client.futures_create_order.side_effect = [
            {"orderId": 300, "status": "NEW"},  # limit 1
            {"orderId": 301, "status": "NEW"},  # limit 2
            {"orderId": 302, "status": "NEW"},  # limit 3
            {"orderId": 303, "status": "NEW"},  # limit 4
            # market fallback
            {
                "orderId": 304,
                "executedQty": "0.001",
                "avgPrice": "50005.0",
                "status": "FILLED",
            },
        ]
        mock_client.futures_get_order.return_value = {
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }

        result = chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)

        assert result.status == ChaseFillStatus.FALLBACK_MARKET
        assert result.filled_quantity == 0.001
        assert result.avg_price == 50005.0
        assert result.maker_fill is False

    def test_no_fallback_when_disabled(
        self,
        mock_client: MagicMock,
    ):
        """fallback_to_market=False면 FAILED 반환."""
        config = ChaseConfig(
            max_chase_ticks=1,
            chase_interval_seconds=0.01,
            timeout_seconds=5.0,
            fallback_to_market=False,
        )
        chaser = LimitOrderChaser(mock_client, config)

        mock_client.futures_create_order.return_value = {
            "orderId": 400,
            "status": "NEW",
        }
        mock_client.futures_get_order.return_value = {
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }

        result = chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)

        assert result.status == ChaseFillStatus.FAILED
        assert result.filled_quantity == 0.0
        assert result.maker_fill is False


# ───────────────── Partial Fill Tests ────────────────────────────


class TestPartialFill:
    """부분 체결 + 나머지 추격."""

    def test_partial_fill_then_complete(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """부분 체결 → 추격 → 나머지 체결."""
        mock_client.futures_create_order.side_effect = [
            {"orderId": 500, "status": "NEW"},
            {"orderId": 501, "status": "NEW"},
        ]
        mock_client.futures_get_order.side_effect = [
            {  # 1차: 부분 체결
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.0005",
                "avgPrice": "50000.0",
            },
            {  # 2차: 나머지 체결
                "status": "FILLED",
                "executedQty": "0.0005",
                "avgPrice": "50000.5",
            },
        ]
        mock_client.futures_orderbook_ticker.side_effect = [
            {"bidPrice": "50000.0", "askPrice": "50001.0"},
            {"bidPrice": "50000.5", "askPrice": "50001.5"},
        ]

        result = chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)

        assert result.status == ChaseFillStatus.FILLED
        assert result.filled_quantity == pytest.approx(0.001, abs=1e-8)
        assert result.maker_fill is True


# ─────────────────── Price Rounding Tests ──────────────────────────


class TestPriceRounding:
    """가격 반올림이 tick_size에 맞는지 검증."""

    def test_round_price_standard(self):
        """표준 tick_size=0.1 반올림."""
        chaser = LimitOrderChaser(MagicMock(), paper_mode=True)

        assert chaser.round_price(50000.34, 0.1, 1) == 50000.3
        assert chaser.round_price(50000.36, 0.1, 1) == 50000.4
        assert chaser.round_price(50000.0, 0.1, 1) == 50000.0

    def test_round_price_small_tick(self):
        """작은 tick_size=0.001 반올림."""
        chaser = LimitOrderChaser(MagicMock(), paper_mode=True)

        assert chaser.round_price(1.23456, 0.001, 3) == 1.235
        assert chaser.round_price(1.23444, 0.001, 3) == 1.234

    def test_round_price_large_tick(self):
        """큰 tick_size=10 반올림."""
        chaser = LimitOrderChaser(MagicMock(), paper_mode=True)

        assert chaser.round_price(50004.0, 10.0, 0) == 50000.0
        assert chaser.round_price(50006.0, 10.0, 0) == 50010.0


# ─────────────────── Best Price Tests ──────────────────────────────


class TestGetBestPrice:
    """호가 조회 로직."""

    def test_buy_gets_bid(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """BUY 시 best bid 반환."""
        price = chaser.get_best_price("BTCUSDT", "BUY")
        assert price == 50000.0

    def test_sell_gets_ask(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """SELL 시 best ask 반환."""
        price = chaser.get_best_price("BTCUSDT", "SELL")
        assert price == 50001.0

    def test_raises_on_api_error(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """API 에러 시 예외 전파."""
        mock_client.futures_orderbook_ticker.side_effect = RuntimeError("API down")
        with pytest.raises(RuntimeError):
            chaser.get_best_price("BTCUSDT", "BUY")


# ─────────────────── Config Tests ──────────────────────────────────


class TestChaseConfig:
    """ChaseConfig 기본값 및 생성."""

    def test_default_values(self):
        config = ChaseConfig()
        assert config.max_chase_ticks == 10
        assert config.chase_interval_seconds == 1.0
        assert config.timeout_seconds == 30.0
        assert config.fallback_to_market is True
        assert config.initial_offset_ticks == 0

    def test_custom_values(self):
        config = ChaseConfig(
            max_chase_ticks=5,
            chase_interval_seconds=0.5,
            timeout_seconds=15.0,
            fallback_to_market=False,
            initial_offset_ticks=2,
        )
        assert config.max_chase_ticks == 5
        assert config.fallback_to_market is False
        assert config.initial_offset_ticks == 2


# ─────────────────── Initial Offset Tests ──────────────────────────


class TestInitialOffset:
    """initial_offset_ticks로 초기 가격 오프셋."""

    def test_buy_offset_lowers_price(self, mock_client: MagicMock):
        """BUY에 offset 적용 시 더 낮은 가격에 limit."""
        config = ChaseConfig(
            max_chase_ticks=0,
            chase_interval_seconds=0.01,
            timeout_seconds=5.0,
            fallback_to_market=False,
            initial_offset_ticks=2,
        )
        chaser = LimitOrderChaser(mock_client, config)

        # bid=50000, tick=0.1, offset=2 → 50000 - 0.2 = 49999.8
        mock_client.futures_create_order.return_value = {
            "orderId": 600,
            "status": "NEW",
        }
        mock_client.futures_get_order.return_value = {
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }

        chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)

        create_call = mock_client.futures_create_order.call_args
        assert create_call.kwargs["price"] == "49999.8"

    def test_sell_offset_raises_price(self, mock_client: MagicMock):
        """SELL에 offset 적용 시 더 높은 가격에 limit."""
        config = ChaseConfig(
            max_chase_ticks=0,
            chase_interval_seconds=0.01,
            timeout_seconds=5.0,
            fallback_to_market=False,
            initial_offset_ticks=3,
        )
        chaser = LimitOrderChaser(mock_client, config)

        # ask=50001, tick=0.1, offset=3 → 50001 + 0.3 = 50001.3
        mock_client.futures_create_order.return_value = {
            "orderId": 700,
            "status": "NEW",
        }
        mock_client.futures_get_order.return_value = {
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }

        chaser.chase_limit_order("ETHUSDT", "SELL", 0.5, tick_size=0.1, price_precision=1)

        create_call = mock_client.futures_create_order.call_args
        assert create_call.kwargs["price"] == "50001.3"


# ─────────────────── Error Handling Tests ──────────────────────────


class TestErrorHandling:
    """API 에러 발생 시 안전한 동작 검증."""

    def test_orderbook_failure_returns_failed(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """호가 조회 실패 시 FAILED 반환."""
        mock_client.futures_orderbook_ticker.side_effect = RuntimeError("timeout")

        result = chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)

        # fallback_to_market=True지만 주문 자체가 안 들어가서 remaining=full
        # → market fallback 시도
        assert result.status in (ChaseFillStatus.FALLBACK_MARKET, ChaseFillStatus.FAILED)

    def test_create_order_failure_returns_failed(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """주문 생성 실패 시 fallback 또는 FAILED."""
        mock_client.futures_create_order.side_effect = RuntimeError("insufficient margin")

        result = chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)

        # 주문 실패 → market fallback (그것도 실패) → FAILED
        assert result.status == ChaseFillStatus.FAILED
        assert result.filled_quantity == 0.0

    def test_order_status_failure_cancels_and_fails(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """주문 상태 조회 실패 시 취소 시도 후 fallback."""
        mock_client.futures_create_order.side_effect = [
            {"orderId": 800, "status": "NEW"},
            # market fallback
            {
                "orderId": 801,
                "executedQty": "0.001",
                "avgPrice": "50005.0",
                "status": "FILLED",
            },
        ]
        mock_client.futures_get_order.side_effect = RuntimeError("network error")

        result = chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)

        # 취소 시도됨
        mock_client.futures_cancel_order.assert_called_once()
        # Market fallback
        assert result.status == ChaseFillStatus.FALLBACK_MARKET
        assert result.filled_quantity == 0.001


# ─────────────────── ChaseResult Tests ──────────────────────────────


class TestChaseResult:
    """ChaseResult 데이터클래스 기본 검증."""

    def test_dataclass_creation(self):
        result = ChaseResult(
            status=ChaseFillStatus.FILLED,
            filled_quantity=0.5,
            avg_price=50000.0,
            order_id="123",
            chase_count=2,
            elapsed_seconds=3.5,
            maker_fill=True,
        )
        assert result.status == ChaseFillStatus.FILLED
        assert result.filled_quantity == 0.5
        assert result.avg_price == 50000.0
        assert result.chase_count == 2
        assert result.maker_fill is True

    def test_enum_values(self):
        assert ChaseFillStatus.FILLED.value == "filled"
        assert ChaseFillStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert ChaseFillStatus.FALLBACK_MARKET.value == "fallback_market"
        assert ChaseFillStatus.FAILED.value == "failed"
        assert ChaseFillStatus.TIMEOUT.value == "timeout"


# ──────── Market Fallback Parsing Tests (fills 배열) ─────────────


class TestMarketFallbackParsing:
    """바이낸스 MARKET 주문 응답 파싱 — fills 배열 우선.

    max_chase_ticks=0 → limit 1회 시도 후 즉시 fallback.
    side_effect: [limit_order, market_fallback]
    """

    @pytest.fixture
    def fallback_chaser(self, mock_client: MagicMock) -> LimitOrderChaser:
        """max_chase_ticks=0, fallback=True인 chaser (limit 1회→즉시 fallback)."""
        config = ChaseConfig(
            max_chase_ticks=0,
            chase_interval_seconds=0.01,
            timeout_seconds=5.0,
            fallback_to_market=True,
        )
        return LimitOrderChaser(client=mock_client, config=config)

    def test_fills_array_single_fill(self, fallback_chaser: LimitOrderChaser, mock_client: MagicMock):
        """fills 배열에 1건 — 정상 파싱."""
        mock_client.futures_create_order.side_effect = [
            {"orderId": 100, "status": "NEW"},  # limit (chase 0)
            # market fallback
            {
                "orderId": 200,
                "executedQty": "0",
                "avgPrice": "0",
                "fills": [
                    {"qty": "185.0", "price": "3.456"},
                ],
            },
        ]
        mock_client.futures_get_order.return_value = {
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }

        result = fallback_chaser.chase_limit_order("NEARUSDT", "BUY", 185.0, tick_size=0.001, price_precision=3)

        assert result.status == ChaseFillStatus.FALLBACK_MARKET
        assert result.filled_quantity == 185.0
        assert abs(result.avg_price - 3.456) < 0.0001
        assert result.maker_fill is False

    def test_fills_array_multiple_fills(self, fallback_chaser: LimitOrderChaser, mock_client: MagicMock):
        """fills 배열에 여러 건 — 가중평균 계산."""
        mock_client.futures_create_order.side_effect = [
            {"orderId": 100, "status": "NEW"},  # limit
            # market fallback: 2개 fill에 분할 체결
            {
                "orderId": 201,
                "executedQty": "0",
                "avgPrice": "0",
                "fills": [
                    {"qty": "100.0", "price": "3.450"},
                    {"qty": "85.0", "price": "3.460"},
                ],
            },
        ]
        mock_client.futures_get_order.return_value = {
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }

        result = fallback_chaser.chase_limit_order("NEARUSDT", "BUY", 185.0, tick_size=0.001, price_precision=3)

        assert result.status == ChaseFillStatus.FALLBACK_MARKET
        assert result.filled_quantity == 185.0
        expected_avg = (100.0 * 3.450 + 85.0 * 3.460) / 185.0
        assert abs(result.avg_price - expected_avg) < 0.0001

    def test_fallback_to_executedQty_when_no_fills(self, fallback_chaser: LimitOrderChaser, mock_client: MagicMock):
        """fills 배열 없지만 executedQty/avgPrice가 유효한 경우."""
        mock_client.futures_create_order.side_effect = [
            {"orderId": 100, "status": "NEW"},
            {
                "orderId": 202,
                "executedQty": "185.0",
                "avgPrice": "3.456",
                "status": "FILLED",
            },
        ]
        mock_client.futures_get_order.return_value = {
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }

        result = fallback_chaser.chase_limit_order("NEARUSDT", "BUY", 185.0, tick_size=0.001, price_precision=3)

        assert result.status == ChaseFillStatus.FALLBACK_MARKET
        assert result.filled_quantity == 185.0
        assert abs(result.avg_price - 3.456) < 0.0001

    def test_fallback_to_cumQuote(self, fallback_chaser: LimitOrderChaser, mock_client: MagicMock):
        """fills 없고 avgPrice=0이지만 cumQuote가 있는 경우."""
        mock_client.futures_create_order.side_effect = [
            {"orderId": 100, "status": "NEW"},
            {
                "orderId": 203,
                "executedQty": "185.0",
                "avgPrice": "0",
                "cumQuote": "639.36",  # 185 * 3.456
                "status": "FILLED",
            },
        ]
        mock_client.futures_get_order.return_value = {
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }

        result = fallback_chaser.chase_limit_order("NEARUSDT", "BUY", 185.0, tick_size=0.001, price_precision=3)

        assert result.status == ChaseFillStatus.FALLBACK_MARKET
        assert result.filled_quantity == 185.0
        assert abs(result.avg_price - 639.36 / 185.0) < 0.0001

    def test_all_fields_zero_triggers_verify_and_fails(self, fallback_chaser: LimitOrderChaser, mock_client: MagicMock):
        """모든 필드가 0인 극단 케이스 — _verify_market_order 호출 후 FAILED.

        응답에 체결 데이터가 없으면 거래소에 직접 조회하여 확인하고,
        조회에서도 체결이 확인되지 않으면 안전을 위해 FAILED 처리한다.
        """
        mock_client.futures_create_order.side_effect = [
            {"orderId": 100, "status": "NEW"},
            {
                "orderId": 204,
                "executedQty": "0",
                "avgPrice": "0",
                "status": "NEW",
            },
        ]
        # limit 체결 확인(1회) + _verify_market_order 재시도(5회) = 6회
        mock_client.futures_get_order.return_value = {
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }

        result = fallback_chaser.chase_limit_order("NEARUSDT", "BUY", 185.0, tick_size=0.001, price_precision=3)

        # _verify_market_order에서 끝까지 NEW → None 반환 → FAILED
        assert result.status == ChaseFillStatus.FAILED
        assert result.filled_quantity == 0.0


# ──────── _verify_market_order Tests ─────────────────────────────


class TestVerifyMarketOrder:
    """_verify_market_order: 응답 파싱 실패 시 거래소 직접 조회 검증."""

    @pytest.fixture
    def chaser(self, mock_client: MagicMock) -> LimitOrderChaser:
        return LimitOrderChaser(client=mock_client, config=ChaseConfig(), paper_mode=False)

    def test_verify_filled_on_first_query(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """1회 조회에서 FILLED + 유효한 가격 발견."""
        mock_client.futures_get_order.return_value = {
            "status": "FILLED",
            "executedQty": "185.0",
            "avgPrice": "3.456",
        }

        result = chaser._verify_market_order("NEARUSDT", "BUY", 185.0, "12345")

        assert result is not None
        filled_qty, avg_price, order_id = result
        assert filled_qty == 185.0
        assert abs(avg_price - 3.456) < 0.0001
        assert order_id == "12345"
        assert mock_client.futures_get_order.call_count == 1

    def test_verify_filled_via_cumquote(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """avgPrice=0이지만 cumQuote로 역산 가능."""
        mock_client.futures_get_order.return_value = {
            "status": "FILLED",
            "executedQty": "100.0",
            "avgPrice": "0",
            "cumQuote": "345.6",
        }

        result = chaser._verify_market_order("BTCUSDT", "SELL", 100.0, "99999")

        assert result is not None
        filled_qty, avg_price, _ = result
        assert filled_qty == 100.0
        assert abs(avg_price - 3.456) < 0.0001

    def test_verify_filled_but_no_price_returns_none(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """FILLED이지만 가격 정보 없음 → 안전을 위해 None 반환."""
        mock_client.futures_get_order.return_value = {
            "status": "FILLED",
            "executedQty": "185.0",
            "avgPrice": "0",
            "cumQuote": "0",
        }

        result = chaser._verify_market_order("NEARUSDT", "BUY", 185.0, "12345")

        assert result is None  # 가격 미확인 → 안전 실패

    def test_verify_canceled_returns_none(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """주문 CANCELED 상태 → 즉시 실패."""
        mock_client.futures_get_order.return_value = {
            "status": "CANCELED",
            "executedQty": "0",
            "avgPrice": "0",
        }

        result = chaser._verify_market_order("NEARUSDT", "BUY", 185.0, "12345")

        assert result is None
        assert mock_client.futures_get_order.call_count == 1  # 재시도 없이 즉시 실패

    def test_verify_new_then_filled(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """처음 NEW → 재시도 후 FILLED 확인."""
        mock_client.futures_get_order.side_effect = [
            {"status": "NEW", "executedQty": "0", "avgPrice": "0"},
            {"status": "NEW", "executedQty": "0", "avgPrice": "0"},
            {"status": "FILLED", "executedQty": "185.0", "avgPrice": "3.456"},
        ]

        result = chaser._verify_market_order(
            "NEARUSDT",
            "BUY",
            185.0,
            "12345",
            max_retries=5,
            retry_interval=0.01,
        )

        assert result is not None
        filled_qty, avg_price, _ = result
        assert filled_qty == 185.0
        assert mock_client.futures_get_order.call_count == 3

    def test_verify_max_retries_exhausted(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """max_retries 소진 → 최종 확인 후 취소 시도 → None 반환."""
        mock_client.futures_get_order.return_value = {
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }

        result = chaser._verify_market_order(
            "NEARUSDT",
            "BUY",
            185.0,
            "12345",
            max_retries=3,
            retry_interval=0.01,
        )

        assert result is None
        # 3 retries + 1 final check before cancel = 4 calls
        assert mock_client.futures_get_order.call_count == 4
        # 최종 확인 후 NEW 상태이므로 cancel 호출됨
        mock_client.futures_cancel_order.assert_called_once_with(symbol="NEARUSDT", orderId=12345)

    def test_verify_query_exception_retries(self, chaser: LimitOrderChaser, mock_client: MagicMock):
        """조회 중 예외 → 재시도 후 성공."""
        mock_client.futures_get_order.side_effect = [
            Exception("Network error"),
            {"status": "FILLED", "executedQty": "50.0", "avgPrice": "100.5"},
        ]

        result = chaser._verify_market_order(
            "BTCUSDT",
            "BUY",
            50.0,
            "77777",
            max_retries=3,
            retry_interval=0.01,
        )

        assert result is not None
        assert result[0] == 50.0
        assert abs(result[1] - 100.5) < 0.0001


# ──────── Market Order newOrderRespType=RESULT Tests ─────────────


class TestMarketOrderRespType:
    """MARKET 주문에 newOrderRespType='RESULT' 포함 확인."""

    @pytest.fixture
    def fallback_chaser(self, mock_client: MagicMock) -> LimitOrderChaser:
        config = ChaseConfig(
            max_chase_ticks=0,
            chase_interval_seconds=0.01,
            timeout_seconds=5.0,
            fallback_to_market=True,
        )
        return LimitOrderChaser(client=mock_client, config=config)

    def test_market_fallback_includes_result_resp_type(self, fallback_chaser: LimitOrderChaser, mock_client: MagicMock):
        """_market_fallback이 newOrderRespType='RESULT' 파라미터를 포함."""
        mock_client.futures_create_order.side_effect = [
            {"orderId": 100, "status": "NEW"},  # limit
            {
                "orderId": 200,
                "executedQty": "10.0",
                "avgPrice": "50.0",
                "status": "FILLED",
            },
        ]
        mock_client.futures_get_order.return_value = {
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        }

        fallback_chaser.chase_limit_order("BTCUSDT", "BUY", 10.0, tick_size=0.1, price_precision=1)

        # 두 번째 futures_create_order 호출 (market fallback)
        market_call = mock_client.futures_create_order.call_args_list[1]
        assert market_call.kwargs.get("newOrderRespType") == "RESULT"
