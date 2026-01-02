"""
Slippage and market impact simulator for realistic paper trading.

Prevents overfitting to paper trading assumptions by modeling:
- Spread crossing costs
- Orderbook walking (market impact)
- Fill probability (adverse selection)
- Latency-induced price movement

Use this to train on realistic scenarios before going live.
"""
import random
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class FillResult:
    """Result of attempting to fill an order."""
    filled: bool
    fill_price: float
    fill_size: float
    slippage: float  # Actual - expected price
    latency_ms: float
    partial_fill: bool = False


class SlippageSimulator:
    """
    Simulate realistic order execution for paper trading.

    Models the gap between paper trading (instant fills at mid)
    and live trading (latency, slippage, partial fills).

    Calibrate parameters from live trading data once available.
    """

    def __init__(
        self,
        # Spread parameters
        avg_spread_bps: float = 15.0,  # Average spread in basis points (0.15%)
        spread_volatility: float = 0.3,  # Spread widens during volatility

        # Latency parameters
        avg_latency_ms: float = 150.0,  # Average order → fill latency
        latency_std_ms: float = 50.0,   # Latency variability

        # Market impact parameters
        impact_coefficient: float = 0.02,  # Price impact per $1000 traded
        orderbook_depth: float = 5000.0,   # Average liquidity at best prices

        # Fill probability parameters
        adverse_selection_factor: float = 0.15,  # Worse fills during fast moves
        partial_fill_threshold: float = 0.7,     # Size threshold for partial fills

        # Price movement during latency
        momentum_continuation: float = 0.6,  # Price continues moving (60% of time)
        reversion_strength: float = 0.3,     # Price mean-reverts (30% of time)
    ):
        # Spread modeling
        self.avg_spread_bps = avg_spread_bps
        self.spread_volatility = spread_volatility

        # Latency
        self.avg_latency_ms = avg_latency_ms
        self.latency_std_ms = latency_std_ms

        # Market impact
        self.impact_coefficient = impact_coefficient
        self.orderbook_depth = orderbook_depth

        # Fill probability
        self.adverse_selection_factor = adverse_selection_factor
        self.partial_fill_threshold = partial_fill_threshold

        # Price dynamics
        self.momentum_continuation = momentum_continuation
        self.reversion_strength = reversion_strength

        # Statistics tracking
        self.total_orders = 0
        self.total_slippage = 0.0
        self.fill_count = 0
        self.partial_fills = 0

    def simulate_fill(
        self,
        side: str,  # "BUY" or "SELL"
        expected_price: float,  # Mid-price when decision made
        size: float,  # Order size in dollars
        spread: float,  # Current bid-ask spread
        volatility: float,  # Recent volatility (for impact scaling)
        momentum: float,  # Recent price momentum (for adverse selection)
        time_to_expiry: float,  # Normalized [0, 1], closer to expiry = worse liquidity
    ) -> FillResult:
        """
        Simulate realistic order fill.

        Args:
            side: "BUY" or "SELL"
            expected_price: Price when decision was made (mid-price)
            size: Order size in dollars
            spread: Current bid-ask spread
            volatility: Recent volatility measure
            momentum: Recent price momentum
            time_to_expiry: How close to market expiry [0=expired, 1=just opened]

        Returns:
            FillResult with actual fill details
        """
        self.total_orders += 1

        # 1. Simulate latency
        latency_ms = max(10, np.random.normal(self.avg_latency_ms, self.latency_std_ms))

        # 2. Calculate spread cost
        # Spread widens during volatility and near expiry
        spread_multiplier = 1.0 + (volatility * self.spread_volatility)
        expiry_multiplier = 1.0 + (1.0 - time_to_expiry) * 0.5  # Spreads widen near expiry
        effective_spread = spread * spread_multiplier * expiry_multiplier

        # Base price: cross the spread
        if side == "BUY":
            base_price = expected_price + effective_spread / 2  # Pay the ask
        else:
            base_price = expected_price - effective_spread / 2  # Hit the bid

        # 3. Market impact (larger orders move price)
        # Impact = (size / depth) * impact_coefficient * volatility_scaling
        depth_ratio = size / self.orderbook_depth
        impact = depth_ratio * self.impact_coefficient * (1 + volatility)

        if side == "BUY":
            impact_price = base_price + impact  # Buying pushes price up
        else:
            impact_price = base_price - impact  # Selling pushes price down

        # 4. Price movement during latency
        # Market moves while order is in flight
        latency_seconds = latency_ms / 1000

        # Momentum continuation: price keeps moving in same direction
        if random.random() < self.momentum_continuation:
            momentum_move = momentum * latency_seconds * 0.01  # Scale by latency
            latency_price = impact_price + momentum_move
        # Mean reversion: price reverts toward mid
        elif random.random() < self.reversion_strength:
            reversion_move = (expected_price - impact_price) * 0.3
            latency_price = impact_price + reversion_move
        else:
            # Random walk
            random_move = np.random.normal(0, volatility * latency_seconds * 0.005)
            latency_price = impact_price + random_move

        # 5. Adverse selection
        # Fast-moving markets: orders fill at worse prices or don't fill
        adverse_selection_prob = abs(momentum) * self.adverse_selection_factor

        if random.random() < adverse_selection_prob:
            # Adverse fill: price moved against us
            if side == "BUY":
                adverse_move = abs(momentum) * 0.01  # Worse price
                latency_price += adverse_move
            else:
                adverse_move = abs(momentum) * 0.01
                latency_price -= adverse_move

        # 6. Partial fills for large orders
        filled_size = size
        partial_fill = False

        if depth_ratio > self.partial_fill_threshold:
            # Large order relative to liquidity → partial fill
            fill_probability = max(0.3, 1.0 - depth_ratio)
            if random.random() > fill_probability:
                # Only partial fill
                filled_size = size * random.uniform(0.3, 0.7)
                partial_fill = True
                self.partial_fills += 1

        # 7. Fill rejection for extreme adverse selection
        # Very fast moves → order doesn't fill at all
        if abs(momentum) > 0.02 and random.random() < 0.1:  # 10% rejection on fast moves
            return FillResult(
                filled=False,
                fill_price=expected_price,
                fill_size=0.0,
                slippage=0.0,
                latency_ms=latency_ms,
                partial_fill=False
            )

        # 8. Final fill price (clip to [0.01, 0.99] for binary markets)
        fill_price = np.clip(latency_price, 0.01, 0.99)

        # 9. Calculate slippage
        slippage = fill_price - expected_price

        # Track statistics
        self.fill_count += 1
        self.total_slippage += abs(slippage)

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_size=filled_size,
            slippage=slippage,
            latency_ms=latency_ms,
            partial_fill=partial_fill
        )

    def get_stats(self) -> dict:
        """Get execution statistics."""
        avg_slippage = self.total_slippage / max(1, self.fill_count)
        fill_rate = self.fill_count / max(1, self.total_orders)
        partial_fill_rate = self.partial_fills / max(1, self.fill_count)

        return {
            "total_orders": self.total_orders,
            "fill_count": self.fill_count,
            "fill_rate": fill_rate,
            "avg_slippage": avg_slippage,
            "avg_slippage_bps": avg_slippage * 10000,
            "partial_fills": self.partial_fills,
            "partial_fill_rate": partial_fill_rate,
        }

    def calibrate_from_live_data(self, live_fills: list):
        """
        Calibrate simulator from actual live trading fills.

        Args:
            live_fills: List of dicts with keys:
                - expected_price: Price when order placed
                - fill_price: Actual fill price
                - latency_ms: Time to fill
                - size: Order size
                - filled: Whether order filled
        """
        if not live_fills:
            return

        # Extract statistics from live data
        slippages = []
        latencies = []

        for fill in live_fills:
            if fill.get("filled"):
                slippage = fill["fill_price"] - fill["expected_price"]
                slippages.append(abs(slippage))
                latencies.append(fill["latency_ms"])

        # Update parameters
        if slippages:
            self.avg_spread_bps = np.mean(slippages) * 10000 * 2  # 2x for bid-ask spread

        if latencies:
            self.avg_latency_ms = np.mean(latencies)
            self.latency_std_ms = np.std(latencies)

        print(f"✓ Calibrated from {len(live_fills)} fills")
        print(f"  Avg slippage: {np.mean(slippages)*10000:.1f} bps")
        print(f"  Avg latency: {np.mean(latencies):.0f}ms")


class RealisticPaperTrading:
    """
    Wrapper for paper trading with realistic execution simulation.

    Use this instead of instant fills at mid-price to train on
    realistic scenarios.
    """

    def __init__(self, simulator: Optional[SlippageSimulator] = None):
        self.simulator = simulator or SlippageSimulator()

        # Track performance degradation vs ideal paper trading
        self.ideal_pnl = 0.0  # What we'd make with instant fills
        self.realistic_pnl = 0.0  # What we'd make with slippage

    def execute_with_slippage(
        self,
        side: str,
        expected_price: float,
        size: float,
        market_state: dict,
    ) -> Tuple[bool, float, float]:
        """
        Execute order with realistic slippage simulation.

        Args:
            side: "BUY" or "SELL"
            expected_price: Mid-price when decision made
            size: Order size in dollars
            market_state: Dict with:
                - spread: Current bid-ask spread
                - volatility: Recent volatility
                - momentum: Recent price momentum
                - time_to_expiry: Normalized time remaining

        Returns:
            (filled, actual_price, actual_size)
        """
        result = self.simulator.simulate_fill(
            side=side,
            expected_price=expected_price,
            size=size,
            spread=market_state.get("spread", 0.01),
            volatility=market_state.get("volatility", 0.01),
            momentum=market_state.get("momentum", 0.0),
            time_to_expiry=market_state.get("time_to_expiry", 0.5),
        )

        if not result.filled:
            return False, expected_price, 0.0

        # Track ideal vs realistic PnL for comparison
        # (This would be updated on position close)

        return True, result.fill_price, result.fill_size

    def get_performance_degradation(self) -> float:
        """
        Calculate performance degradation due to realistic execution.

        Returns:
            Percentage degradation (e.g., 0.25 = 25% worse than ideal)
        """
        if self.ideal_pnl == 0:
            return 0.0

        degradation = (self.ideal_pnl - self.realistic_pnl) / abs(self.ideal_pnl)
        return degradation


if __name__ == "__main__":
    print("=" * 60)
    print("Slippage Simulator - Example Usage")
    print("=" * 60)

    sim = SlippageSimulator()

    # Simulate 100 fills under different conditions
    scenarios = [
        # (side, price, size, spread, volatility, momentum, time_to_expiry)
        ("BUY", 0.45, 10, 0.01, 0.01, 0.005, 0.8),    # Normal conditions
        ("BUY", 0.45, 100, 0.01, 0.01, 0.005, 0.8),   # Large size
        ("BUY", 0.45, 10, 0.02, 0.05, 0.02, 0.8),     # High volatility
        ("BUY", 0.45, 10, 0.01, 0.01, 0.005, 0.1),    # Near expiry
        ("SELL", 0.55, 10, 0.01, 0.01, -0.01, 0.5),   # Selling into momentum
    ]

    print("\nSimulating fills under different conditions:\n")

    for side, price, size, spread, vol, mom, tte in scenarios:
        result = sim.simulate_fill(side, price, size, spread, vol, mom, tte)

        if result.filled:
            slippage_bps = result.slippage * 10000
            print(f"{side} ${size:.0f} @ {price:.3f}")
            print(f"  Filled: {result.fill_price:.4f} (slippage: {slippage_bps:+.1f} bps)")
            print(f"  Latency: {result.latency_ms:.0f}ms")
            if result.partial_fill:
                print(f"  Partial fill: ${result.fill_size:.0f} / ${size:.0f}")
            print()
        else:
            print(f"{side} ${size:.0f} @ {price:.3f}")
            print(f"  NOT FILLED (adverse selection)")
            print()

    # Print statistics
    stats = sim.get_stats()
    print("\n" + "=" * 60)
    print("EXECUTION STATISTICS")
    print("=" * 60)
    print(f"Total orders: {stats['total_orders']}")
    print(f"Fills: {stats['fill_count']} ({stats['fill_rate']*100:.1f}%)")
    print(f"Avg slippage: {stats['avg_slippage_bps']:.1f} bps")
    print(f"Partial fills: {stats['partial_fills']} ({stats['partial_fill_rate']*100:.1f}%)")
