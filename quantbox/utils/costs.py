"""
Realistic slippage and transaction cost models for backtesting accuracy.

Accurate cost modeling is CRITICAL for profitability estimation.
Most failed strategies in production are due to underestimated costs.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class SlippageModel(Enum):
    """Available slippage models."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLUME_IMPACT = "volume_impact"
    SPREAD_BASED = "spread_based"


@dataclass
class TransactionCosts:
    """Transaction cost parameters for realistic backtesting."""
    
    # Commission structure
    maker_fee: float = 0.0002  # 0.02% maker fee (limit orders)
    taker_fee: float = 0.0004  # 0.04% taker fee (market orders)
    
    # Slippage parameters
    slippage_model: SlippageModel = SlippageModel.VOLUME_IMPACT
    fixed_slippage: float = 0.0005  # 0.05% fixed slippage
    
    # Volume impact parameters (square-root model)
    # Impact = sigma * sqrt(Q/ADV) where Q=order size, ADV=avg daily volume
    impact_coefficient: float = 0.1
    
    # Spread parameters
    min_spread_bps: float = 2.0  # Minimum spread in basis points
    
    # Funding rate (for perpetual futures)
    funding_rate_8h: float = 0.0001  # 0.01% per 8 hours
    
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        is_buy: bool,
        volume: Optional[float] = None,
        spread: Optional[float] = None
    ) -> float:
        """
        Calculate slippage for an order.
        
        Args:
            price: Current market price
            quantity: Order quantity
            is_buy: True if buying, False if selling
            volume: Average daily volume (for volume impact)
            spread: Current bid-ask spread (for spread-based)
            
        Returns:
            Slippage amount (always positive)
        """
        if self.slippage_model == SlippageModel.FIXED:
            return price * self.fixed_slippage
        
        elif self.slippage_model == SlippageModel.PERCENTAGE:
            return price * self.fixed_slippage
        
        elif self.slippage_model == SlippageModel.VOLUME_IMPACT:
            if volume is None or volume < 1e-10:
                return price * self.fixed_slippage
            
            # Square-root market impact model
            # Widely used in institutional trading
            order_value = abs(quantity) * price
            participation_rate = order_value / volume
            
            # Impact is proportional to sqrt of participation
            impact = self.impact_coefficient * np.sqrt(participation_rate)
            
            return price * min(impact, 0.05)  # Cap at 5%
        
        elif self.slippage_model == SlippageModel.SPREAD_BASED:
            if spread is None:
                spread = price * self.min_spread_bps / 10000
            
            # Cross half the spread
            return spread / 2
        
        return 0.0
    
    def calculate_total_cost(
        self,
        price: float,
        quantity: float,
        is_buy: bool,
        is_maker: bool = False,
        volume: Optional[float] = None,
        spread: Optional[float] = None,
        holding_periods: int = 0
    ) -> dict:
        """
        Calculate total transaction costs including all components.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            is_buy: Buy or sell
            is_maker: Maker (limit) or taker (market) order
            volume: Average daily volume
            spread: Current spread
            holding_periods: Number of 8h periods held (for funding)
            
        Returns:
            Dict with cost breakdown
        """
        notional = abs(quantity) * price
        
        # Commission
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        commission = notional * fee_rate
        
        # Slippage
        slippage = self.calculate_slippage(price, quantity, is_buy, volume, spread)
        slippage_cost = abs(quantity) * slippage
        
        # Funding (for perpetuals)
        funding_cost = 0.0
        if holding_periods > 0:
            # Funding can be positive (long pays) or negative (short receives)
            # For conservative estimation, assume longs pay
            if is_buy:
                funding_cost = notional * self.funding_rate_8h * holding_periods
        
        total = commission + slippage_cost + funding_cost
        
        return {
            'commission': commission,
            'slippage': slippage_cost,
            'funding': funding_cost,
            'total': total,
            'total_bps': (total / notional) * 10000
        }
    
    def estimate_roundtrip_cost(
        self,
        entry_price: float,
        quantity: float,
        is_maker: bool = False,
        volume: Optional[float] = None,
        expected_holding_8h: int = 3
    ) -> float:
        """
        Estimate round-trip cost for position sizing decisions.
        
        Args:
            entry_price: Entry price
            quantity: Position size
            is_maker: Using limit orders
            volume: Average daily volume
            expected_holding_8h: Expected holding period in 8h blocks
            
        Returns:
            Estimated total round-trip cost
        """
        # Entry costs
        entry = self.calculate_total_cost(
            entry_price, quantity, True, is_maker, volume
        )
        
        # Exit costs (assume similar conditions)
        exit_cost = self.calculate_total_cost(
            entry_price, quantity, False, is_maker, volume,
            holding_periods=expected_holding_8h
        )
        
        return entry['total'] + exit_cost['total']
    
    def minimum_profit_target(
        self,
        entry_price: float,
        quantity: float,
        is_maker: bool = False,
        volume: Optional[float] = None,
        expected_holding_8h: int = 3,
        risk_multiple: float = 1.5
    ) -> float:
        """
        Calculate minimum profit target to be profitable after costs.
        
        This is CRITICAL for strategy development.
        
        Args:
            entry_price: Entry price
            quantity: Position size
            is_maker: Using limit orders
            volume: Average daily volume
            expected_holding_8h: Expected holding period
            risk_multiple: Multiple of costs for safety margin
            
        Returns:
            Minimum price change required (as percentage)
        """
        roundtrip = self.estimate_roundtrip_cost(
            entry_price, quantity, is_maker, volume, expected_holding_8h
        )
        
        notional = abs(quantity) * entry_price
        
        return (roundtrip / notional) * risk_multiple


def create_binance_futures_costs() -> TransactionCosts:
    """
    Create cost model for Binance Futures.
    
    Default fees (without VIP discount):
    - Maker: 0.02%
    - Taker: 0.04%
    """
    return TransactionCosts(
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_model=SlippageModel.VOLUME_IMPACT,
        impact_coefficient=0.1,
        funding_rate_8h=0.0001
    )


def create_conservative_costs() -> TransactionCosts:
    """
    Create conservative cost model for realistic backtesting.
    
    Uses higher estimates to avoid overfitting.
    """
    return TransactionCosts(
        maker_fee=0.0003,  # Assume some taker fills
        taker_fee=0.0005,
        slippage_model=SlippageModel.VOLUME_IMPACT,
        impact_coefficient=0.15,  # Higher impact
        funding_rate_8h=0.00015  # Higher funding
    )
