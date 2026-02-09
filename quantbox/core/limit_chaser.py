"""
Limit Order Chaser - Maker-only 체결을 통한 슬리피지/수수료 최소화.

시장가(MARKET) 대신 지정가(LIMIT)를 사용하여 maker 수수료(0.02%)를 적용받고
슬리피지를 최소화한다. 미체결 시 가격을 추격하여 체결을 보장한다.

비용 절감 효과 (트레이드당):
  - 수수료: 0.05% → 0.02% (레그당 0.03% 절약 × 4 legs = 0.12%)
  - 슬리피지: 0.10% → ~0.03% (레그당 0.07% 절약 × 4 legs = 0.28%)
  - 합계: 트레이드당 ~0.40% 절약

Safety:
  - max_chase_ticks: 최대 추격 횟수 (무한 추격 방지)
  - timeout_seconds: 전체 주문 타임아웃 (시간 제한)
  - fallback_to_market: 추격 실패 시 시장가 전환 (체결 보장)
"""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("LimitChaser")


class ChaseFillStatus(Enum):
    """추격 주문 체결 결과."""

    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    FALLBACK_MARKET = "fallback_market"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ChaseConfig:
    """Limit order chaser 설정.

    Args:
        max_chase_ticks: 가격 추격 최대 횟수. 초과하면 fallback.
        chase_interval_seconds: 미체결 확인 주기 (초).
        timeout_seconds: 전체 주문 타임아웃. 0이면 무제한.
        fallback_to_market: True면 추격 실패 시 시장가로 전환.
            False면 ChaseFillStatus.FAILED 반환.
        initial_offset_ticks: 초기 지정가를 best bid/ask에서
            이 값만큼 유리한 방향으로 offset. 0이면 best에 걸기.
    """

    max_chase_ticks: int = 10
    chase_interval_seconds: float = 1.0
    timeout_seconds: float = 30.0
    fallback_to_market: bool = True
    initial_offset_ticks: int = 0


@dataclass
class ChaseResult:
    """추격 주문 결과.

    Args:
        status: 체결 상태.
        filled_quantity: 체결된 수량.
        avg_price: 평균 체결 가격.
        order_id: 거래소 주문 ID (또는 paper 주문 ID).
        chase_count: 실제 추격한 횟수.
        elapsed_seconds: 소요 시간.
        maker_fill: maker로 체결되었는지 여부 (True면 maker, False면 taker fallback).
    """

    status: ChaseFillStatus
    filled_quantity: float
    avg_price: float
    order_id: str
    chase_count: int
    elapsed_seconds: float
    maker_fill: bool


class LimitOrderChaser:
    """Limit 주문을 best bid/ask에 걸고, 미체결 시 가격을 추격하는 실행기.

    Binance futures client를 받아서 실제 주문을 실행한다.
    Paper mode에서는 즉시 체결을 시뮬레이션한다.

    사용법:
        chaser = LimitOrderChaser(client, chase_config)
        result = chaser.chase_limit_order("BTCUSDT", "BUY", 0.001, tick_size=0.1, price_precision=1)
    """

    def __init__(
        self,
        client,
        config: ChaseConfig | None = None,
        paper_mode: bool = False,
    ):
        """
        Args:
            client: binance.client.Client 인스턴스.
            config: ChaseConfig. None이면 기본값 사용.
            paper_mode: True면 실제 주문 없이 시뮬레이션.
        """
        self.client = client
        self.config = config or ChaseConfig()
        self.paper_mode = paper_mode

    def get_best_price(self, symbol: str, side: str) -> float:
        """현재 best bid(매수 시) 또는 best ask(매도 시) 조회.

        Args:
            symbol: 심볼 (e.g., "BTCUSDT").
            side: "BUY" 또는 "SELL".

        Returns:
            Best price. BUY면 best bid (maker로 걸기 위해),
            SELL이면 best ask.

        Raises:
            RuntimeError: 호가 조회 실패.
        """
        book = self.client.futures_orderbook_ticker(symbol=symbol)
        if side == "BUY":
            return float(book["bidPrice"])
        else:
            return float(book["askPrice"])

    def round_price(
        self, price: float, tick_size: float, price_precision: int
    ) -> float:
        """가격을 거래소 tick_size에 맞게 반올림.

        Args:
            price: 원래 가격.
            tick_size: 최소 가격 단위 (e.g., 0.1).
            price_precision: 가격 소수점 자릿수.

        Returns:
            tick_size 배수로 반올림된 가격.
        """
        rounded = round(price / tick_size) * tick_size
        return round(rounded, price_precision)

    def chase_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        tick_size: float = 0.01,
        price_precision: int = 2,
    ) -> ChaseResult:
        """Limit 지정가 추격 주문 실행.

        1. best bid/ask에 limit 주문 배치
        2. chase_interval마다 체결 여부 확인
        3. 미체결이면 기존 주문 취소 → 새 best에 재배치
        4. max_chase_ticks 초과 또는 timeout이면 fallback

        Args:
            symbol: 거래소 심볼.
            side: "BUY" 또는 "SELL".
            quantity: 주문 수량.
            tick_size: 최소 가격 변동 단위.
            price_precision: 가격 소수점 자릿수.

        Returns:
            ChaseResult with 체결 정보.
        """
        if self.paper_mode:
            return self._paper_fill(symbol, side, quantity)

        start_time = time.monotonic()
        chase_count = 0
        remaining_qty = quantity
        total_filled = 0.0
        weighted_price_sum = 0.0
        last_order_id = ""

        while chase_count <= self.config.max_chase_ticks:
            elapsed = time.monotonic() - start_time
            if self.config.timeout_seconds > 0 and elapsed >= self.config.timeout_seconds:
                logger.warning(
                    f"[{symbol}] Chase timeout after {elapsed:.1f}s, "
                    f"filled {total_filled}/{quantity}"
                )
                break

            # 1. Get best price
            try:
                best_price = self.get_best_price(symbol, side)
            except Exception as e:
                logger.error(f"[{symbol}] Failed to get best price: {e}")
                break

            # 2. Apply initial offset (첫 주문만)
            if chase_count == 0 and self.config.initial_offset_ticks > 0:
                offset = self.config.initial_offset_ticks * tick_size
                if side == "BUY":
                    best_price -= offset  # 더 싸게 사려고 시도
                else:
                    best_price += offset  # 더 비싸게 팔려고 시도

            limit_price = self.round_price(best_price, tick_size, price_precision)

            # 3. Place limit order
            try:
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="LIMIT",
                    timeInForce="GTC",
                    quantity=remaining_qty,
                    price=str(limit_price),
                )
                last_order_id = str(order["orderId"])
                logger.debug(
                    f"[{symbol}] Limit {side} {remaining_qty} @ {limit_price} "
                    f"(chase #{chase_count})"
                )
            except Exception as e:
                logger.error(f"[{symbol}] Limit order failed: {e}")
                break

            # 4. Wait and check fill
            time.sleep(self.config.chase_interval_seconds)

            try:
                order_status = self.client.futures_get_order(
                    symbol=symbol, orderId=int(last_order_id)
                )
                status = order_status["status"]
                filled_qty = float(order_status["executedQty"])
                avg_price = float(order_status.get("avgPrice", limit_price))

                if filled_qty > 0:
                    weighted_price_sum += filled_qty * avg_price
                    total_filled += filled_qty
                    remaining_qty = quantity - total_filled

                if status == "FILLED":
                    logger.info(
                        f"[{symbol}] Limit FILLED: {side} {total_filled} "
                        f"@ {weighted_price_sum / total_filled:.4f} "
                        f"(chase #{chase_count})"
                    )
                    return ChaseResult(
                        status=ChaseFillStatus.FILLED,
                        filled_quantity=total_filled,
                        avg_price=weighted_price_sum / total_filled,
                        order_id=last_order_id,
                        chase_count=chase_count,
                        elapsed_seconds=time.monotonic() - start_time,
                        maker_fill=True,
                    )

                if remaining_qty <= 0:
                    return ChaseResult(
                        status=ChaseFillStatus.FILLED,
                        filled_quantity=total_filled,
                        avg_price=weighted_price_sum / total_filled,
                        order_id=last_order_id,
                        chase_count=chase_count,
                        elapsed_seconds=time.monotonic() - start_time,
                        maker_fill=True,
                    )

                # 5. Cancel unfilled order
                if status in ("NEW", "PARTIALLY_FILLED"):
                    try:
                        self.client.futures_cancel_order(
                            symbol=symbol, orderId=int(last_order_id)
                        )
                    except Exception as cancel_err:
                        logger.warning(
                            f"[{symbol}] Cancel failed (may already be filled): {cancel_err}"
                        )
                        # Re-check after cancel failure
                        try:
                            recheck = self.client.futures_get_order(
                                symbol=symbol, orderId=int(last_order_id)
                            )
                            if recheck["status"] == "FILLED":
                                filled_qty = float(recheck["executedQty"])
                                avg_p = float(recheck.get("avgPrice", limit_price))
                                extra = filled_qty - (total_filled - remaining_qty + quantity - remaining_qty)
                                weighted_price_sum = filled_qty * avg_p
                                total_filled = filled_qty
                                return ChaseResult(
                                    status=ChaseFillStatus.FILLED,
                                    filled_quantity=total_filled,
                                    avg_price=weighted_price_sum / total_filled if total_filled > 0 else 0,
                                    order_id=last_order_id,
                                    chase_count=chase_count,
                                    elapsed_seconds=time.monotonic() - start_time,
                                    maker_fill=True,
                                )
                        except Exception:
                            pass

            except Exception as e:
                logger.error(f"[{symbol}] Order status check failed: {e}")
                # Try to cancel just in case
                try:
                    self.client.futures_cancel_order(
                        symbol=symbol, orderId=int(last_order_id)
                    )
                except Exception:
                    pass
                break

            chase_count += 1

        # Fallback
        elapsed = time.monotonic() - start_time

        if total_filled > 0 and remaining_qty <= 0:
            return ChaseResult(
                status=ChaseFillStatus.FILLED,
                filled_quantity=total_filled,
                avg_price=weighted_price_sum / total_filled,
                order_id=last_order_id,
                chase_count=chase_count,
                elapsed_seconds=elapsed,
                maker_fill=True,
            )

        if self.config.fallback_to_market and remaining_qty > 0:
            logger.warning(
                f"[{symbol}] Falling back to MARKET for remaining "
                f"{remaining_qty:.6f} (filled {total_filled:.6f} via limit)"
            )
            market_result = self._market_fallback(
                symbol, side, remaining_qty
            )
            if market_result is not None:
                mkt_qty, mkt_price, mkt_oid = market_result
                weighted_price_sum += mkt_qty * mkt_price
                total_filled += mkt_qty

                return ChaseResult(
                    status=ChaseFillStatus.FALLBACK_MARKET,
                    filled_quantity=total_filled,
                    avg_price=(
                        weighted_price_sum / total_filled
                        if total_filled > 0
                        else 0
                    ),
                    order_id=mkt_oid,
                    chase_count=chase_count,
                    elapsed_seconds=time.monotonic() - start_time,
                    maker_fill=False,
                )

        # Total failure
        return ChaseResult(
            status=ChaseFillStatus.FAILED,
            filled_quantity=total_filled,
            avg_price=(
                weighted_price_sum / total_filled if total_filled > 0 else 0
            ),
            order_id=last_order_id,
            chase_count=chase_count,
            elapsed_seconds=elapsed,
            maker_fill=False,
        )

    def _market_fallback(
        self, symbol: str, side: str, quantity: float
    ) -> tuple[float, float, str] | None:
        """시장가 fallback 주문.

        Returns:
            (filled_qty, avg_price, order_id) 또는 None.
        """
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
            )
            filled_qty = float(order.get("executedQty", quantity))
            avg_price = float(order.get("avgPrice", order.get("price", 0)))
            order_id = str(order["orderId"])
            logger.info(
                f"[{symbol}] Market fallback FILLED: {side} {filled_qty} @ {avg_price}"
            )
            return filled_qty, avg_price, order_id
        except Exception as e:
            logger.error(f"[{symbol}] Market fallback FAILED: {e}")
            return None

    def _paper_fill(
        self, symbol: str, side: str, quantity: float
    ) -> ChaseResult:
        """Paper mode: 즉시 체결 시뮬레이션 (best bid/ask 기준).

        Args:
            symbol: 심볼.
            side: "BUY" 또는 "SELL".
            quantity: 수량.

        Returns:
            ChaseResult with FILLED status.
        """
        try:
            price = self.get_best_price(symbol, side)
        except Exception:
            # Paper mode에서 호가 조회 실패 시 0으로 채움
            price = 0.0
            logger.warning(f"[PAPER] Could not get price for {symbol}, using 0")

        logger.info(
            f"[PAPER] Limit chase FILLED: {side} {quantity:.6f} {symbol} @ {price}"
        )
        return ChaseResult(
            status=ChaseFillStatus.FILLED,
            filled_quantity=quantity,
            avg_price=price,
            order_id=f"paper_{int(time.time() * 1000)}",
            chase_count=0,
            elapsed_seconds=0.0,
            maker_fill=True,
        )
